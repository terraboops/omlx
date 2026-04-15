#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile KV allocator fragmentation at various context lengths (Task 31).

Measures the gap between Metal active memory and peak memory during KV cache
growth to quantify allocator fragmentation. Profiles at 4K, 16K, and 64K
context lengths (fitting within 48GB), then extrapolates to 1M.

The key metric is:
  fragmentation_pct = (peak - active) / active * 100

If fragmentation is > 10%, paging/compaction would meaningfully reduce the
memory footprint. If < 5%, the allocator is efficient and paging work is
not justified.

Usage:
    python scripts/profile_kv_fragmentation.py
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"


def get_memory_stats():
    """Get Metal memory stats in GB."""
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    return {"active_gb": active, "peak_gb": peak, "cache_gb": cache}


def profile_context_length(model, tokenizer, target_tokens: int,
                           sample_interval: int = 1000) -> dict:
    """Profile memory during prefill to target_tokens context length."""
    from mlx_lm.models.cache import KVCache

    n_layers = len(model.layers)

    # Reset peak tracking
    mx.reset_peak_memory()

    # Build a synthetic prompt
    prompt = "Explain the theory of " + "computation " * (target_tokens // 2)
    tokens = tokenizer.encode(prompt)[:target_tokens]

    logger.info(f"  Profiling {len(tokens)} tokens...")

    cache = [KVCache() for _ in range(n_layers)]
    samples = []
    chunk_size = 2048

    # Record baseline
    mx.synchronize()
    base = get_memory_stats()
    samples.append({"tokens": 0, **base})

    for start in range(0, len(tokens), chunk_size):
        end = min(start + chunk_size, len(tokens))
        chunk = mx.array([tokens[start:end]])
        logits = model(chunk, cache=cache)
        mx.eval(logits)
        mx.synchronize()

        current_tokens = end
        if current_tokens % sample_interval < chunk_size or end == len(tokens):
            stats = get_memory_stats()
            samples.append({"tokens": current_tokens, **stats})

    # Final measurement
    mx.synchronize()
    final = get_memory_stats()

    # Compute fragmentation
    active = final["active_gb"]
    peak = final["peak_gb"]
    cache_mem = final["cache_gb"]

    # Fragmentation = peak - active (memory allocated but not currently live)
    frag_gb = peak - active
    frag_pct = (frag_gb / active * 100) if active > 0 else 0

    # KV cache size estimate: active - model_base
    model_base = base["active_gb"]
    kv_estimate = active - model_base

    result = {
        "target_tokens": target_tokens,
        "actual_tokens": len(tokens),
        "model_base_gb": round(model_base, 2),
        "active_gb": round(active, 2),
        "peak_gb": round(peak, 2),
        "cache_gb": round(cache_mem, 2),
        "fragmentation_gb": round(frag_gb, 2),
        "fragmentation_pct": round(frag_pct, 1),
        "kv_estimate_gb": round(kv_estimate, 2),
        "samples": [
            {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}
            for s in samples
        ],
    }

    # Cleanup
    del cache, logits
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    return result


def main():
    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    # Apply prefill patch
    try:
        from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
        apply_prefill_last_logit_patch(model)
    except ImportError:
        logger.warning("prefill_last_logit patch not available")

    # Warmup
    logger.info("Warming up Metal kernels...")
    from mlx_lm.models.cache import KVCache
    warmup_cache = [KVCache() for _ in range(len(model.layers))]
    warmup_tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(warmup_tokens, cache=warmup_cache)
    mx.eval(logits)
    del warmup_cache, logits
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    # Profile at multiple context lengths
    context_lengths = [4096, 16384, 65536]
    results = []

    for ctx in context_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Context: {ctx//1024}K tokens")
        logger.info(f"{'='*60}")

        # Reset peak between runs
        mx.reset_peak_memory()

        r = profile_context_length(model, tokenizer, ctx)
        results.append(r)

        logger.info(f"  Active: {r['active_gb']:.2f} GB")
        logger.info(f"  Peak:   {r['peak_gb']:.2f} GB")
        logger.info(f"  Cache:  {r['cache_gb']:.2f} GB")
        logger.info(f"  Fragmentation: {r['fragmentation_gb']:.2f} GB ({r['fragmentation_pct']:.1f}%)")
        logger.info(f"  KV estimate:   {r['kv_estimate_gb']:.2f} GB")

    elapsed = time.perf_counter() - t0

    # Extrapolate to 1M
    if len(results) >= 2:
        # Linear regression on fragmentation_pct vs tokens
        frags = [(r["actual_tokens"], r["fragmentation_pct"]) for r in results]
        avg_frag_pct = sum(f[1] for f in frags) / len(frags)

        # At 1M context, KV is ~22.5 GB (from CLAUDE.md)
        kv_1m_gb = 22.5
        frag_1m_gb = kv_1m_gb * avg_frag_pct / 100
    else:
        avg_frag_pct = results[0]["fragmentation_pct"] if results else 0
        frag_1m_gb = 0

    # Write results
    output = {
        "model": MODEL_ID,
        "elapsed_s": round(elapsed, 1),
        "profiles": results,
        "extrapolation": {
            "avg_fragmentation_pct": round(avg_frag_pct, 1),
            "estimated_1m_kv_gb": 22.5,
            "estimated_1m_fragmentation_gb": round(frag_1m_gb, 1),
        },
    }

    output_path = Path("research/kv_fragmentation_profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    # Write research note
    note_path = Path("research/KV_FRAGMENTATION.md")
    with open(note_path, "w") as f:
        f.write("# KV Allocator Fragmentation Profile (Task 31)\n\n")
        f.write(f"**Date**: 2026-04-15\n")
        f.write(f"**Model**: {MODEL_ID}\n")
        f.write(f"**MLX version**: {mx.__version__}\n\n")

        f.write("## Results\n\n")
        f.write("| Context | Active (GB) | Peak (GB) | Frag (GB) | Frag % | KV est (GB) |\n")
        f.write("|--------:|------------:|----------:|----------:|-------:|------------:|\n")
        for r in results:
            f.write(f"| {r['actual_tokens']//1024}K | {r['active_gb']:.2f} | "
                    f"{r['peak_gb']:.2f} | {r['fragmentation_gb']:.2f} | "
                    f"{r['fragmentation_pct']:.1f}% | {r['kv_estimate_gb']:.2f} |\n")

        f.write(f"\n## Extrapolation to 1M Context\n\n")
        f.write(f"- Average fragmentation: **{avg_frag_pct:.1f}%**\n")
        f.write(f"- At 1M context (22.5 GB KV): ~{frag_1m_gb:.1f} GB fragmentation\n\n")

        if avg_frag_pct < 5:
            verdict = ("**VERDICT: Fragmentation is negligible (<5%).** "
                       "The MLX Metal allocator is efficient at these sizes. "
                       "Paging/compaction work (Tasks 43, 64) is NOT justified "
                       "for memory savings alone — pursue only if the tiering "
                       "architecture provides other benefits (e.g., CPU offload).")
        elif avg_frag_pct < 15:
            verdict = ("**VERDICT: Fragmentation is moderate (5-15%).** "
                       "Paging could save ~{:.1f} GB at 1M context. "
                       "Consider if other benefits (CPU offload, partial "
                       "eviction) justify the complexity.".format(frag_1m_gb))
        else:
            verdict = ("**VERDICT: Fragmentation is significant (>15%).** "
                       "Paging would save ~{:.1f} GB at 1M context — "
                       "this justifies the complexity of Tasks 43/64.".format(frag_1m_gb))

        f.write(f"## Verdict\n\n{verdict}\n")

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Average fragmentation: {avg_frag_pct:.1f}%")
    logger.info(f"Extrapolated 1M fragmentation: ~{frag_1m_gb:.1f} GB")
    if avg_frag_pct < 5:
        logger.info("VERDICT: Fragmentation negligible — skip paging work")
    elif avg_frag_pct < 15:
        logger.info("VERDICT: Fragmentation moderate — paging is optional")
    else:
        logger.info("VERDICT: Fragmentation significant — paging justified")

    logger.info(f"\nResults: {output_path}")
    logger.info(f"Note: {note_path}")
    logger.info(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
