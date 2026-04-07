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
    import importlib
    gen_mod = importlib.import_module("mlx_lm.generate")

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


def apply_hypercar_patches(fp16_layers: int = 0, bits: int = 3,
                           group_size: int = 64, kv_mode: str = "native",
                           dequant_chunk_size: int = 2048,
                           min_quant_tokens: int = 512) -> None:
    """Apply all hypercar optimizations to mlx_lm runtime.

    kv_mode controls the KV cache strategy:
      "native" — MLX QuantizedKVCache (affine per-group, fast, proven)
      "tq3"    — TurboQuant WHT codec (codebook, supports save/load/rewind/fork)
      "fp16"   — No quantization (baseline, limited context)

    Must be called BEFORE mlx_lm.server is imported/invoked.
    """
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    from omlx.patches.vertical_eval import apply_vertical_eval_patch

    # 1. Apply SDPA patch for TQ3 mode (routes attention through TQ codec)
    if kv_mode == "tq3":
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
        apply_turboquant_attention_patch()

    # 2. Monkey-patch make_prompt_cache based on kv_mode
    import mlx_lm.models.cache as cache_mod
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    def hypercar_make_prompt_cache(model, max_kv_size=None):
        if hasattr(model, "make_cache"):
            caches = model.make_cache()
        else:
            num_layers = len(model.layers)
            caches = [KVCache() for _ in range(num_layers)]

        if kv_mode == "fp16":
            return caches

        result = []
        for i, c in enumerate(caches):
            if i < fp16_layers:
                result.append(c)
            elif isinstance(c, KVCache):
                if kv_mode == "tq3":
                    from omlx.turboquant_kv import TurboQuantKVCache
                    result.append(TurboQuantKVCache(
                        bits=bits,
                        dequant_chunk_size=dequant_chunk_size,
                        min_quant_tokens=min_quant_tokens,
                    ))
                else:  # native
                    result.append(QuantizedKVCache(group_size=group_size, bits=bits))
            else:
                result.append(c)
        return result

    cache_mod.make_prompt_cache = hypercar_make_prompt_cache

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
            if kv_mode == "tq3":
                # TQ3 streaming needs vertical_eval to prevent graph hoarding
                apply_vertical_eval_patch(model)
                logging.info("TQ3 mode: vertical_eval ENABLED")
            elif kv_mode == "native":
                logging.info("Native mode: vertical_eval disabled (not needed)")
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
    parser.add_argument("--kv-mode", choices=["native", "tq3", "fp16"], default="native",
                        help="KV cache strategy: native (MLX affine), tq3 (WHT codebook w/ save/load/rewind), fp16 (no quant)")
    parser.add_argument("--bits", type=int, default=3,
                        help="KV quantization bits (default 3)")
    parser.add_argument("--kv-group-size", type=int, default=64,
                        help="Group size for native mode (default 64)")
    parser.add_argument("--dequant-chunk", type=int, default=2048,
                        help="Dequant chunk size for TQ3 streaming")
    parser.add_argument("--min-quant-tokens", type=int, default=512,
                        help="TQ3: stay fp16 below this threshold per layer")
    parser.add_argument("--prefill-step-size", type=int, default=8192,
                        help="Tokens per prefill chunk (default 8192, was 2048 for TQ3)")
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
    logger.info(f"KV mode:          {args.kv_mode}")
    logger.info(f"Bits:             {args.bits}")
    logger.info(f"fp16 layers:      {args.fp16_layers}")
    if args.kv_mode == "native":
        logger.info(f"Group size:       {args.kv_group_size}")
    elif args.kv_mode == "tq3":
        logger.info(f"Dequant chunk:    {args.dequant_chunk}")
        logger.info(f"Min quant tokens: {args.min_quant_tokens}")
        logger.info(f"Features:         save/load, rewind, fork (WHT rotation)")
    logger.info(f"Prefill step:     {args.prefill_step_size}")
    logger.info(f"Host:Port:        {args.host}:{args.port}")
    logger.info(f"Prompt cache:     {args.prompt_cache_size} entries, "
                f"{args.prompt_cache_bytes // 1_000_000_000}GB")
    logger.info("=" * 70)

    # Apply patches BEFORE mlx_lm.server runs
    logger.info("Applying hypercar patches...")
    apply_hypercar_patches(
        fp16_layers=args.fp16_layers,
        bits=args.bits,
        group_size=args.kv_group_size,
        kv_mode=args.kv_mode,
        dequant_chunk_size=args.dequant_chunk,
        min_quant_tokens=args.min_quant_tokens,
    )
    apply_progress_logging(log_every=8)
    logger.info("Progress logging enabled (every 8 generated tokens)")

    # Fix Qwen3-Coder tool parser: ast.literal_eval crashes on malformed
    # JSON that the model sometimes generates for complex tool parameters
    try:
        import mlx_lm.tool_parsers.qwen3_coder as qwen3_parser
        orig_convert = qwen3_parser._convert_param_value

        def safe_convert(param_value, param_name, param_config):
            try:
                return orig_convert(param_value, param_name, param_config)
            except (SyntaxError, ValueError, TypeError):
                # Model emitted malformed JSON/Python for this param.
                # Return raw string — better than crashing.
                return param_value

        qwen3_parser._convert_param_value = safe_convert
        logger.info("Patched Qwen3-Coder tool parser (SyntaxError safety)")
    except ImportError:
        pass

    # Fix mlx_lm prompt cache bug: _search returns None for 'best'
    # when cache has entries from a previous model. Patch to handle None.
    try:
        import mlx_lm.server as server_mod
        if hasattr(server_mod, 'LRUPromptCache'):
            orig_search = server_mod.LRUPromptCache._search
            def safe_search(self, model, tokens):
                result = orig_search(self, model, tokens)
                # Fix: if longer/shorter paths have None, return clean miss
                if hasattr(result, 'longer') and result.longer is None and result.shorter is None:
                    return result
                return result
            server_mod.LRUPromptCache._search = safe_search

            # Fix mlx_lm bug: _search line 264 does `tokens[:index] + best`
            # but `best` can be None if the trie branch has no cache entry.
            orig_search_method = server_mod.LRUPromptCache._search
            def patched_search(self, model, tokens):
                result = orig_search_method(self, model, tokens)
                # If longer was set to a bad value (contains None concat),
                # clear it so fetch_nearest_cache falls through to shorter/new
                if result.longer is not None and not isinstance(result.longer, list):
                    result = result._replace(longer=None)
                return result
            # Actually, easier: just patch the source of the bug directly
            import types
            orig_search_code = server_mod.LRUPromptCache._search

            def fixed_search(self, model, tokens):
                if model not in self._cache:
                    return self.SearchResult(model, None, None, None, 0)
                current = self._cache[model]
                last_cache_index = -1
                index = 0
                while index < len(tokens) and tokens[index] in current:
                    current = current[tokens[index]]
                    if "cache" in current:
                        last_cache_index = index
                    index += 1
                if last_cache_index == len(tokens) - 1:
                    return self.SearchResult(model, tokens, None, None, 0)
                shorter = None
                if last_cache_index > 0:
                    shorter = tokens[: last_cache_index + 1]
                longer = None
                common_prefix = index
                if index > 0:
                    best = None
                    stack = [(current, [])]
                    while stack:
                        cur, extra = stack.pop()
                        if "cache" in cur:
                            if best is None or len(extra) < len(best):
                                best = extra
                        else:
                            for tok in cur:
                                stack.append((cur[tok], extra + [tok]))
                    if best is not None:  # THE FIX: guard against None
                        longer = tokens[:index] + best
                _cache_logger = logging.getLogger("hypercar.cache")
                _cache_logger.debug(
                    f"_search: index={index} last_cache_idx={last_cache_index} "
                    f"shorter={'yes' if shorter else 'no'} "
                    f"longer={'yes' if longer else 'no'} "
                    f"common_prefix={common_prefix}"
                )
                return self.SearchResult(model, None, shorter, longer, common_prefix)

            server_mod.LRUPromptCache._search = fixed_search

            # Force prompt checkpoint at system prompt boundary for all models.
            # Without this, cache only exists at the leaf (full token sequence),
            # so multi-turn conversations can't reuse the shared system prompt.
            if hasattr(server_mod, 'APIHandler'):
                orig_compute_cp = server_mod.APIHandler._compute_prompt_checkpoint
                def forced_checkpoint(self, tokenizer, request, prompt):
                    do_cp, pos = orig_compute_cp(self, tokenizer, request, prompt)
                    if not do_cp and request.request_type == "chat":
                        # Find where the user message starts (after system prompt)
                        # Use half the prompt as a rough system-prompt boundary
                        if len(request.messages) >= 2:
                            # Encode just the system message to find its length
                            sys_msg = request.messages[0] if request.messages[0].get("role") == "system" else None
                            if sys_msg:
                                sys_len = len(tokenizer.encode(
                                    tokenizer.apply_chat_template(
                                        [sys_msg], tokenize=False, add_generation_prompt=False
                                    )
                                ))
                                if sys_len > 5 and sys_len < len(prompt):
                                    return True, sys_len
                    return do_cp, pos
                server_mod.APIHandler._compute_prompt_checkpoint = forced_checkpoint
                logger.info("Forced system prompt checkpoint for cache reuse")

            # Add cache hit/miss logging
            orig_fetch = server_mod.LRUPromptCache.fetch_nearest_cache
            _cache_logger = logging.getLogger("hypercar.cache")

            def logged_fetch(self, model, tokens):
                result = orig_fetch(self, model, tokens)
                cache, rest = result
                total = len(tokens) if hasattr(tokens, '__len__') else 0
                cached = total - (len(rest) if hasattr(rest, '__len__') else total)
                if cached > 0:
                    _cache_logger.info(
                        f"📦 Cache HIT: {cached}/{total} tokens cached "
                        f"({cached*100//max(total,1)}%%), prefilling {len(rest)} new"
                    )
                else:
                    _cache_logger.info(f"📦 Cache MISS: prefilling all {total} tokens")
                return result

            server_mod.LRUPromptCache.fetch_nearest_cache = logged_fetch
            logger.info("Patched LRUPromptCache._search (None guard + cache logging)")
    except Exception as e:
        logger.warning(f"Could not patch LRUPromptCache: {e}")

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
