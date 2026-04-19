# SPDX-License-Identifier: Apache-2.0
"""Efficiency profiler: CPU + memory hot-path analysis for analyst findings.

Runs targeted micro-benchmarks on the hot paths identified in the code audit:
  1. SnapKV freshness scoring (Python loop overhead)
  2. TQ3 WHT quantize vs fused quantize (kernel dispatch overhead)
  3. DuoKVCache trimming (per-head Python loop overhead)
  4. SnapKV keep_mask construction (per-element .at[].add loop)
  5. KV buffer growth pattern (concatenate frequency)
  6. Decode token generation throughput breakdown

Outputs: /tmp/hypercar_efficiency_profile.json
"""

from __future__ import annotations

import cProfile
import gc
import io
import json
import logging
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger("omlx.bench.efficiency")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

RESULTS_PATH = Path("/tmp/hypercar_efficiency_profile.json")


def profile_snapkv_freshness(T: int = 4096) -> dict:
    """Profile compute_freshness_scores at context length T."""
    from omlx.patches.snapkv import compute_freshness_scores
    from mlx_lm.models.cache import KVCache

    logger.info(f"=== SnapKV freshness scoring @ {T} tokens ===")

    # Simulate a filled KV cache
    B, H_kv, D = 1, 4, 128
    cache = []
    for _ in range(4):  # 4 layers
        c = KVCache()
        k = mx.random.normal((B, H_kv, T, D))
        v = mx.random.normal((B, H_kv, T, D))
        c.state = (k, v)
        c.offset = T
        cache.append(c)
    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])

    # Profile
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    freshness = compute_freshness_scores(cache, target_layers=[0, 1, 2, 3])
    pr.disable()
    mx.eval(freshness)
    elapsed = time.perf_counter() - t0

    # Extract top functions
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_text = s.getvalue()
    logger.info(f"  Freshness @ {T} tokens: {elapsed:.3f}s")
    logger.info(f"  Top functions:\n{profile_text[:1500]}")

    del cache
    gc.collect()
    mx.clear_cache()

    return {
        "test": "snapkv_freshness",
        "context_tokens": T,
        "elapsed_s": round(elapsed, 4),
        "profile_top20": profile_text[:2000],
    }


def profile_snapkv_select(T: int = 16384) -> dict:
    """Profile snapkv_select keep_mask construction at context length T."""
    from omlx.patches.snapkv import snapkv_select

    logger.info(f"=== SnapKV select (keep_mask) @ {T} tokens ===")

    B, H_kv = 1, 4
    importance = mx.random.normal((B, H_kv, T))
    mx.eval(importance)

    keep_count = T // 4  # 25% keep ratio

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    mask = snapkv_select(importance, keep_count, segment_size=512)
    pr.disable()
    mx.eval(mask)
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_text = s.getvalue()
    logger.info(f"  select @ {T}: {elapsed:.3f}s")
    logger.info(f"  Top functions:\n{profile_text[:1500]}")

    return {
        "test": "snapkv_select",
        "context_tokens": T,
        "keep_count": keep_count,
        "elapsed_s": round(elapsed, 4),
        "profile_top20": profile_text[:2000],
    }


def profile_tq3_quantize(T: int = 4096) -> dict:
    """Profile TQ3 WHT quantize vs what a fused version would cost."""
    logger.info(f"=== TQ3 WHT quantize @ {T} tokens ===")

    from omlx.turboquant_kv import TurboQuantMSECodec

    B, H_kv, D = 1, 4, 128
    codec = TurboQuantMSECodec(D, bits=3, use_wht=True)

    vectors = mx.random.normal((B, H_kv, T, D))
    mx.eval(vectors)

    # Profile WHT quantize
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    norms, packed = codec.quantize(vectors)
    pr.disable()
    mx.eval(norms, packed)
    elapsed_wht = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_text = s.getvalue()
    logger.info(f"  WHT quantize @ {T}: {elapsed_wht:.3f}s")

    # Profile dequantize (has a fused path)
    t0 = time.perf_counter()
    restored = codec.dequantize_fused(norms, packed)
    mx.eval(restored)
    elapsed_dequant_fused = time.perf_counter() - t0
    logger.info(f"  Fused dequantize @ {T}: {elapsed_dequant_fused:.3f}s")

    t0 = time.perf_counter()
    restored2 = codec.dequantize(norms, packed)
    mx.eval(restored2)
    elapsed_dequant_unfused = time.perf_counter() - t0
    logger.info(f"  Unfused dequantize @ {T}: {elapsed_dequant_unfused:.3f}s")
    logger.info(f"  Dequant speedup from fusion: {elapsed_dequant_unfused/max(elapsed_dequant_fused,1e-6):.1f}x")

    del vectors, norms, packed, restored, restored2
    gc.collect()
    mx.clear_cache()

    return {
        "test": "tq3_quantize",
        "context_tokens": T,
        "wht_quantize_s": round(elapsed_wht, 4),
        "fused_dequant_s": round(elapsed_dequant_fused, 4),
        "unfused_dequant_s": round(elapsed_dequant_unfused, 4),
        "dequant_fusion_speedup": round(elapsed_dequant_unfused / max(elapsed_dequant_fused, 1e-6), 2),
        "profile_top20": profile_text[:2000],
    }


def profile_duo_kv_trim(T: int = 4096) -> dict:
    """Profile DuoKVCache update_and_fetch trimming overhead."""
    logger.info(f"=== DuoKVCache trim @ {T} tokens ===")

    from omlx.duo_kv_cache import DuoKVCache, load_duo_policy

    policy = load_duo_policy()
    cache = DuoKVCache(policy, layer_idx=24)  # mid-layer

    B, H_kv, D = 1, 4, 128

    # Simulate prefill: large chunk that will trigger trimming
    keys = mx.random.normal((B, H_kv, T, D))
    values = mx.random.normal((B, H_kv, T, D))
    mx.eval(keys, values)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    k_out, v_out = cache.update_and_fetch(keys, values)
    pr.disable()
    mx.eval(k_out, v_out)
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_text = s.getvalue()
    logger.info(f"  DuoKV trim @ {T}: {elapsed:.3f}s")
    logger.info(f"  Head types: {cache.head_types}")

    del cache, keys, values
    gc.collect()
    mx.clear_cache()

    return {
        "test": "duo_kv_trim",
        "context_tokens": T,
        "elapsed_s": round(elapsed, 4),
        "profile_top20": profile_text[:2000],
    }


def profile_kv_buffer_growth(max_tokens: int = 16384, chunk_size: int = 2048) -> dict:
    """Profile mx.concatenate growth pattern in TQ3 KV buffer."""
    logger.info(f"=== KV buffer concatenate growth @ {max_tokens} tokens ===")

    B, H_kv, D = 1, 4, 128
    pw = 12  # packed width for 3-bit dim=128

    tracemalloc.start()
    t0 = time.perf_counter()

    k_norms = None
    k_packed = None
    concat_times = []

    for offset in range(0, max_tokens, chunk_size):
        chunk_norms = mx.random.normal((B, H_kv, chunk_size))
        chunk_packed = mx.random.normal((B, H_kv, chunk_size, pw)).astype(mx.uint32)
        mx.eval(chunk_norms, chunk_packed)

        tc0 = time.perf_counter()
        if k_norms is None:
            k_norms = chunk_norms
            k_packed = chunk_packed
        else:
            k_norms = mx.concatenate([k_norms, chunk_norms], axis=2)
            k_packed = mx.concatenate([k_packed, chunk_packed], axis=2)
        mx.eval(k_norms, k_packed)
        tc1 = time.perf_counter()
        concat_times.append(round(tc1 - tc0, 6))

    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"  Buffer growth @ {max_tokens}: {elapsed:.3f}s total")
    logger.info(f"  Concat times: {concat_times}")
    logger.info(f"  Peak Python memory: {peak_mem / 1e6:.1f} MB")

    del k_norms, k_packed
    gc.collect()
    mx.clear_cache()

    return {
        "test": "kv_buffer_growth",
        "max_tokens": max_tokens,
        "chunk_size": chunk_size,
        "elapsed_s": round(elapsed, 4),
        "concat_times": concat_times,
        "peak_python_mem_mb": round(peak_mem / 1e6, 2),
    }


def profile_compact_cache(T: int = 8192) -> dict:
    """Profile compact_cache re-RoPE + quantize overhead."""
    from omlx.patches.snapkv import compact_cache, snapkv_select
    from mlx_lm.models.cache import KVCache

    logger.info(f"=== compact_cache @ {T} tokens ===")

    B, H_kv, D = 1, 4, 128
    cache = []
    for _ in range(4):
        c = KVCache()
        k = mx.random.normal((B, H_kv, T, D))
        v = mx.random.normal((B, H_kv, T, D))
        c.state = (k, v)
        c.offset = T
        cache.append(c)
    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])

    # Generate keep indices (25% keep)
    keep_count = T // 4
    importance = mx.random.normal((B, H_kv, T))
    mx.eval(importance)
    mask = snapkv_select(importance, keep_count)
    mx.eval(mask)
    from omlx.patches.snapkv import get_keep_indices
    indices = get_keep_indices(mask)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    compact_cache(cache, indices)
    pr.disable()
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_text = s.getvalue()
    logger.info(f"  compact_cache @ {T}: {elapsed:.3f}s")

    del cache
    gc.collect()
    mx.clear_cache()

    return {
        "test": "compact_cache",
        "context_tokens": T,
        "keep_count": keep_count,
        "elapsed_s": round(elapsed, 4),
        "profile_top20": profile_text[:2000],
    }


def profile_importance_scoring(T: int = 8192) -> dict:
    """Profile CAOTE importance scoring to find hottest path."""
    logger.info(f"=== CAOTE importance scoring @ {T} tokens ===")

    B, H_kv, H_q, D = 1, 4, 32, 128

    # Simulate captured queries
    captured = {}
    from mlx_lm.models.cache import KVCache
    cache = []
    for i in range(4):
        c = KVCache()
        k = mx.random.normal((B, H_kv, T, D))
        v = mx.random.normal((B, H_kv, T, D))
        c.state = (k, v)
        c.offset = T
        cache.append(c)
        q = mx.random.normal((B, H_q, T, D))
        captured[i] = (q, k, v)
    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])

    from omlx.patches.snapkv import compute_caote_importance
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    importance = compute_caote_importance(captured, cache, obs_window=64)
    pr.disable()
    mx.eval(importance)
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_text = s.getvalue()
    logger.info(f"  CAOTE scoring @ {T}: {elapsed:.3f}s")

    del cache, captured
    gc.collect()
    mx.clear_cache()

    return {
        "test": "caote_importance",
        "context_tokens": T,
        "elapsed_s": round(elapsed, 4),
        "profile_top20": profile_text[:2000],
    }


def main():
    logger.info("=" * 60)
    logger.info("HYPERCAR EFFICIENCY PROFILER")
    logger.info("=" * 60)

    results = []

    # 1. SnapKV freshness (the Python loop bottleneck)
    results.append(profile_snapkv_freshness(T=4096))
    gc.collect()
    mx.clear_cache()

    # 2. SnapKV select (keep_mask construction)
    results.append(profile_snapkv_select(T=16384))
    gc.collect()
    mx.clear_cache()

    # 3. TQ3 quantize
    results.append(profile_tq3_quantize(T=4096))
    gc.collect()
    mx.clear_cache()

    # 4. DuoKV trim
    results.append(profile_duo_kv_trim(T=4096))
    gc.collect()
    mx.clear_cache()

    # 5. KV buffer growth
    results.append(profile_kv_buffer_growth(max_tokens=16384, chunk_size=2048))
    gc.collect()
    mx.clear_cache()

    # 6. compact_cache
    results.append(profile_compact_cache(T=8192))
    gc.collect()
    mx.clear_cache()

    # 7. CAOTE importance scoring
    results.append(profile_importance_scoring(T=8192))
    gc.collect()
    mx.clear_cache()

    # Write results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults written to {RESULTS_PATH}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("EFFICIENCY PROFILE SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(f"  {r['test']:30s}  {r['elapsed_s']:8.4f}s  @ {r.get('context_tokens', r.get('max_tokens', '?'))} tok")


if __name__ == "__main__":
    main()
