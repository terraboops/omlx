# SPDX-License-Identifier: Apache-2.0
"""Hypercar MLX server — OpenAI-compatible API with TQ3 + streaming.

Wraps mlx_lm.server with all hypercar patches pre-applied:
  - Streaming TQ3 KV cache (8x compression + fused Givens kernel)
  - Vertical graph eval (prevents cross-layer memory hoarding)
  - Adaptive memory budget (chunk sizing based on live headroom)
  - fp16 layer 0 anchor (quality insurance)
  - min_quant_tokens threshold (TQ3 warmup phase)
  - prefill last-logit patch (saves 40GB logits tensor at long context)

Serves an OpenAI-compatible API that tools like OpenCode can consume.

Usage:
    # Basic: use defaults (Qwen3-Coder-30B-A3B-4bit, TQ3, port 8080)
    python -m omlx.hypercar_server

    # Custom config
    python -m omlx.hypercar_server \\
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \\
        --port 8080 --host 0.0.0.0 \\
        --fp16-layers 1 \\
        --prefill-step-size 2048

    # Point OpenCode at it
    export OPENAI_API_BASE=http://localhost:8080/v1
    export OPENAI_API_KEY=hypercar
    opencode ...

Endpoints (OpenAI-compatible):
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import mlx.core as mx


def apply_progress_logging(log_every: int = 8) -> None:
    """Monkey-patch mlx_lm.generate.stream_generate to log per-token progress.

    Every `log_every` generated tokens, emits:
      - Token count, decode tok/s, running wall time
      - Prefill tok/s (once, at start)
      - Total memory (Metal)
    """
    import time
    import mlx_lm.generate as gen_mod

    original = gen_mod.stream_generate
    _logger = logging.getLogger("hypercar.generate")

    def logged_stream_generate(model, tokenizer, prompt, **kwargs):
        # Log prompt size upfront
        prompt_len = len(prompt) if hasattr(prompt, '__len__') else 0
        t_start = time.perf_counter()
        t_first_token = None
        n_tokens = 0

        _logger.info(f"🔵 Generation starting: {prompt_len} prompt tokens")

        for result in original(model, tokenizer, prompt, **kwargs):
            if t_first_token is None:
                t_first_token = time.perf_counter()
                ttft = t_first_token - t_start
                prefill_toks = prompt_len / ttft if ttft > 0 else 0
                _logger.info(
                    f"🟢 First token @ {ttft:.2f}s (prefill: {prefill_toks:.0f} tok/s)"
                )

            n_tokens += 1
            if n_tokens % log_every == 0:
                elapsed_decode = time.perf_counter() - t_first_token
                decode_toks = n_tokens / elapsed_decode if elapsed_decode > 0 else 0
                total_elapsed = time.perf_counter() - t_start
                _logger.info(
                    f"⚡ gen={n_tokens:>4d} | "
                    f"decode={decode_toks:>5.1f} tok/s | "
                    f"total={total_elapsed:>5.1f}s | "
                    f"mem={mx.get_active_memory()/1e9:.1f}GB"
                )

            yield result

        # Final stats
        total_time = time.perf_counter() - t_start
        if t_first_token and n_tokens > 0:
            decode_time = time.perf_counter() - t_first_token
            final_toks = n_tokens / decode_time if decode_time > 0 else 0
            _logger.info(
                f"🏁 DONE: {n_tokens} tokens in {total_time:.1f}s "
                f"({final_toks:.1f} tok/s decode)"
            )

    gen_mod.stream_generate = logged_stream_generate
    # Also patch the imported reference in server
    try:
        import mlx_lm.server as server_mod
        server_mod.stream_generate = logged_stream_generate
    except ImportError:
        pass


def apply_hypercar_patches(fp16_layers: int = 1, bits: int = 3,
                           dequant_chunk_size: int = 2048,
                           min_quant_tokens: int = 512) -> None:
    """Apply all hypercar optimizations to mlx_lm runtime.

    Must be called BEFORE mlx_lm.server is imported/invoked.
    """
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    from omlx.patches.vertical_eval import apply_vertical_eval_patch
    from omlx.turboquant_kv import TurboQuantKVCache

    # 1. Patch SDPA to handle TQ caches
    apply_turboquant_attention_patch()

    # 2. Monkey-patch make_prompt_cache to use TQ3 caches
    import mlx_lm.models.cache as cache_mod
    from mlx_lm.models.cache import KVCache

    original_make = cache_mod.make_prompt_cache

    def hypercar_make_prompt_cache(model, max_kv_size=None):
        if hasattr(model, "make_cache"):
            # Model has its own — wrap it: fp16 for first N layers, TQ3 for rest
            caches = model.make_cache()
            # Replace non-fp16-anchor layers with TQ3
            result = []
            for i, c in enumerate(caches):
                if i < fp16_layers:
                    result.append(c)  # Keep standard cache
                else:
                    # Only replace if it's a standard KVCache (not Mamba state etc)
                    if isinstance(c, KVCache):
                        result.append(TurboQuantKVCache(
                            bits=bits,
                            dequant_chunk_size=dequant_chunk_size,
                            min_quant_tokens=min_quant_tokens,
                        ))
                    else:
                        result.append(c)  # Preserve SSM/Mamba caches
            return result

        # No model.make_cache — build hybrid manually
        num_layers = len(model.layers)
        result = []
        for i in range(num_layers):
            if i < fp16_layers:
                result.append(KVCache())
            else:
                result.append(TurboQuantKVCache(
                    bits=bits,
                    dequant_chunk_size=dequant_chunk_size,
                    min_quant_tokens=min_quant_tokens,
                ))
        return result

    cache_mod.make_prompt_cache = hypercar_make_prompt_cache

    # Also patch where mlx_lm.server imports it from
    import mlx_lm.utils as utils_mod
    if hasattr(utils_mod, "make_prompt_cache"):
        utils_mod.make_prompt_cache = hypercar_make_prompt_cache

    # 3. Hook model loading to apply per-model patches
    import mlx_lm.utils as mlx_utils
    original_load = mlx_utils.load

    def hypercar_load(*args, **kwargs):
        model, tokenizer = original_load(*args, **kwargs)
        try:
            apply_prefill_last_logit_patch(model)
            apply_vertical_eval_patch(model)
        except Exception as e:
            logging.warning(f"Some hypercar patches failed: {e}")
        return model, tokenizer

    mlx_utils.load = hypercar_load
    # Also update mlx_lm.server's load reference if already imported
    try:
        import mlx_lm.server as server_mod
        if hasattr(server_mod, "load"):
            server_mod.load = hypercar_load
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Hypercar MLX server (OpenAI-compatible, TQ3 KV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",
                        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
                        help="Model to serve (HF ID or local path)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fp16-layers", type=int, default=1,
                        help="Number of fp16 layers (rest use TQ3). Default 1.")
    parser.add_argument("--bits", type=int, default=3,
                        help="TQ KV quantization bits (3 or 4)")
    parser.add_argument("--dequant-chunk", type=int, default=2048,
                        help="Dequant chunk size for streaming")
    parser.add_argument("--min-quant-tokens", type=int, default=512,
                        help="Stay fp16 below this threshold per layer")
    parser.add_argument("--prefill-step-size", type=int, default=2048,
                        help="Tokens per prefill chunk")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-cache-size", type=int, default=8,
                        help="LRU prompt cache entries")
    parser.add_argument("--prompt-cache-bytes", type=int, default=20_000_000_000,
                        help="Max bytes across all cached prompts")
    parser.add_argument("--allowed-origins", default="*")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("hypercar.server")

    logger.info("=" * 70)
    logger.info("HYPERCAR MLX SERVER")
    logger.info("=" * 70)
    logger.info(f"Model:            {args.model}")
    logger.info(f"fp16 layers:      {args.fp16_layers}")
    logger.info(f"TQ bits:          {args.bits}")
    logger.info(f"Dequant chunk:    {args.dequant_chunk}")
    logger.info(f"Prefill step:     {args.prefill_step_size}")
    logger.info(f"Min quant tokens: {args.min_quant_tokens}")
    logger.info(f"Host:Port:        {args.host}:{args.port}")
    logger.info(f"Prompt cache:     {args.prompt_cache_size} entries, "
                f"{args.prompt_cache_bytes // 1_000_000_000}GB")
    logger.info("=" * 70)

    # Apply patches BEFORE mlx_lm.server runs
    logger.info("Applying hypercar patches...")
    apply_hypercar_patches(
        fp16_layers=args.fp16_layers,
        bits=args.bits,
        dequant_chunk_size=args.dequant_chunk,
        min_quant_tokens=args.min_quant_tokens,
    )
    apply_progress_logging(log_every=8)
    logger.info("Progress logging enabled (every 8 generated tokens)")

    # Build sys.argv for mlx_lm.server's argparse
    server_argv = [
        "mlx_lm.server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-tokens", str(args.max_tokens),
        "--temp", str(args.temp),
        "--top-p", str(args.top_p),
        "--prefill-step-size", str(args.prefill_step_size),
        "--prompt-cache-size", str(args.prompt_cache_size),
        "--prompt-cache-bytes", str(args.prompt_cache_bytes),
        "--allowed-origins", args.allowed_origins,
        "--log-level", args.log_level,
    ]
    sys.argv = server_argv

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info("OpenCode config:")
    logger.info(f"  export OPENAI_API_BASE=http://{args.host}:{args.port}/v1")
    logger.info(f"  export OPENAI_API_KEY=hypercar")

    # Delegate to mlx_lm.server
    from mlx_lm.server import main as mlx_server_main
    mlx_server_main()


if __name__ == "__main__":
    main()
