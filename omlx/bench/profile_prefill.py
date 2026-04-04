# SPDX-License-Identifier: Apache-2.0
"""Profile chunked prefill memory to find the 52GB spike.

Traces Metal memory at every stage:
  - Before/after each prefill chunk
  - Inside update_and_fetch (which path taken, dequant size)
  - Inside the attention patch (streaming vs standard SDPA)

Usage:
    # Quick trace (no profiler, just memory logging):
    python -m omlx.bench.profile_prefill --tokens 32768

    # With scalene CPU+memory profiler:
    python -m scalene --cpu --memory --profile-all \
        omlx/bench/profile_prefill.py --tokens 32768

    # With pprof-compatible output:
    python -m scalene --cpu --memory --pprof profile.pprof \
        omlx/bench/profile_prefill.py --tokens 32768
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time

import mlx.core as mx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("profile")

# ---------------------------------------------------------------------------
# Memory snapshot helper
# ---------------------------------------------------------------------------

def mem_snapshot(label: str) -> float:
    """Log and return current Metal active memory in GB."""
    gb = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    logger.info(f"  MEM [{label:>30s}]: active={gb:.2f}GB  peak={peak:.2f}GB")
    return gb


# ---------------------------------------------------------------------------
# Tracing hooks for update_and_fetch
# ---------------------------------------------------------------------------

_uaf_call_count = 0


def install_tracing(cache_list):
    """Monkey-patch update_and_fetch on each cache to log memory."""
    from omlx.turboquant_kv import TurboQuantKVCache

    for i, cache in enumerate(cache_list):
        if not isinstance(cache, TurboQuantKVCache):
            continue

        original_uaf = cache.update_and_fetch.__func__

        def make_traced(layer_idx, orig):
            def traced_uaf(self, keys, values):
                global _uaf_call_count
                _uaf_call_count += 1

                B, H, T_new, D = keys.shape

                # Only log first layer + every 8th layer to reduce noise
                if layer_idx > 1 and layer_idx % 8 != 0:
                    return orig(self, keys, values)

                before = mx.get_active_memory() / 1e9
                result = orig(self, keys, values)
                after = mx.get_active_memory() / 1e9

                path = "fp16_warmup" if not self._quantized else (
                    "streaming" if self._streaming_active else "short_dequant"
                )

                # Only log if there's a significant memory change
                delta = after - before
                if abs(delta) > 0.01 or layer_idx <= 1:
                    logger.info(
                        f"  UAF layer={layer_idx:>2d} offset={self.offset:>6d} "
                        f"T_new={T_new} path={path:>13s} "
                        f"mem={before:.2f}→{after:.2f}GB (Δ{delta:+.2f})"
                    )

                return result
            return traced_uaf

        import types
        cache.update_and_fetch = types.MethodType(make_traced(i, original_uaf), cache)


# ---------------------------------------------------------------------------
# Tracing for the attention patch
# ---------------------------------------------------------------------------

_attn_call_count = 0


def install_attention_tracing():
    """Add memory logging to the turboquant attention patch."""
    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return

    current_sdpa = mlx_base.scaled_dot_product_attention

    def traced_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        global _attn_call_count
        _attn_call_count += 1

        from omlx.turboquant_kv import TurboQuantKVCache

        real_cache = cache
        if hasattr(cache, "_cache"):
            real_cache = cache._cache

        # Only trace every 8th call to reduce noise
        if _attn_call_count % 8 != 1:
            return current_sdpa(queries, keys, values, cache, scale, mask, sinks)

        L = queries.shape[-2]
        is_tq = isinstance(real_cache, TurboQuantKVCache)
        streaming = getattr(real_cache, '_streaming_active', False)

        before = mx.get_active_memory() / 1e9
        result = current_sdpa(queries, keys, values, cache, scale, mask, sinks)
        # Force eval to see actual memory impact
        mx.eval(result)
        after = mx.get_active_memory() / 1e9

        logger.info(
            f"  SDPA call={_attn_call_count:>4d} L={L} "
            f"tq={is_tq} streaming={streaming} "
            f"mem={before:.2f}→{after:.2f}GB (Δ{after-before:+.2f})"
        )

        return result

    mlx_base.scaled_dot_product_attention = traced_sdpa

    # Re-patch model modules
    import sys as _sys
    for mod_name, mod in list(_sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models."):
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", traced_sdpa)


# ---------------------------------------------------------------------------
# Main profiling run
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile chunked prefill memory")
    parser.add_argument("--tokens", type=int, default=32768, help="Total tokens to prefill")
    parser.add_argument("--chunk", type=int, default=8192, help="Prefill chunk size")
    parser.add_argument("--max-gb", type=float, default=38.0, help="Abort if metal exceeds this")
    parser.add_argument("--no-trace-attn", action="store_true", help="Skip attention tracing (less noise)")
    args = parser.parse_args()

    logger.info(f"Profiling {args.tokens:,} tokens in {args.chunk}-token chunks")

    # Load model
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch

    from omlx.patches.vertical_eval import apply_vertical_eval_patch

    apply_turboquant_attention_patch()
    model, tokenizer = load('mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit')
    apply_prefill_last_logit_patch(model)
    apply_vertical_eval_patch(model)
    n_layers = model.args.num_hidden_layers

    mem_snapshot("model loaded")

    # Create cache with fp16 layer 0
    cache = [
        KVCache() if i == 0 else TurboQuantKVCache(bits=3, dequant_chunk_size=2048, min_quant_tokens=512)
        for i in range(n_layers)
    ]

    # Install tracing
    install_tracing(cache)
    if not args.no_trace_attn:
        install_attention_tracing()

    # Generate tokens
    base = "x = 1\n"
    base_tokens = tokenizer.encode(base)
    tokens = (base_tokens * ((args.tokens // len(base_tokens)) + 1))[:args.tokens]

    mem_snapshot("before prefill")

    # Chunked prefill with per-chunk memory tracking
    for chunk_idx, chunk_start in enumerate(range(0, len(tokens), args.chunk)):
        chunk_end = min(chunk_start + args.chunk, len(tokens))
        chunk_tokens = tokens[chunk_start:chunk_end]
        x = mx.array([chunk_tokens])

        logger.info(f"\n{'='*60}")
        logger.info(f"CHUNK {chunk_idx}: tokens {chunk_start:,}-{chunk_end:,} ({len(chunk_tokens)} tokens)")
        mem_before = mem_snapshot(f"chunk {chunk_idx} before forward")

        t0 = time.perf_counter()
        logits = model(x, cache=cache)
        mem_snapshot(f"chunk {chunk_idx} after forward (lazy)")

        mx.eval(logits)
        elapsed = time.perf_counter() - t0
        mem_after = mem_snapshot(f"chunk {chunk_idx} after eval")

        toks = len(chunk_tokens) / elapsed
        logger.info(
            f"CHUNK {chunk_idx} DONE: {toks:,.0f} tok/s | "
            f"mem {mem_before:.1f}→{mem_after:.1f}GB (Δ{mem_after-mem_before:+.1f}GB)"
        )

        # Safety abort
        if mem_after > args.max_gb:
            logger.error(f"ABORT: {mem_after:.1f}GB > {args.max_gb}GB limit")
            break

        # Force GC + clear Metal cache between chunks
        del logits
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mem_snapshot(f"chunk {chunk_idx} after gc+clear")

    mem_snapshot("after all chunks")

    # Report cache state
    logger.info("\n--- Cache state ---")
    for i, c in enumerate(cache):
        if isinstance(c, TurboQuantKVCache):
            if i <= 1 or i == n_layers - 1:
                logger.info(
                    f"  layer {i:>2d}: quantized={c._quantized} "
                    f"offset={c.offset} streaming={c._streaming_active}"
                )
        elif isinstance(c, KVCache):
            if hasattr(c, 'keys') and c.keys is not None:
                logger.info(f"  layer {i:>2d}: KVCache (fp16) keys.shape={c.keys.shape}")

    # Cleanup
    del cache, model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mem_snapshot("cleanup done")


if __name__ == "__main__":
    main()
