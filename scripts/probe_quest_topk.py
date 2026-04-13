#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe MLX argpartition/argsort speed for Quest top-K page selection.

Measures whether MLX's top-K primitives are fast enough for Quest's
per-decode-step page selection at various context lengths.

Budget: < 200 microseconds per head at 64K pages (1M context).

Usage (must run under bench sandbox exclusion):
    .venv/bin/python -m omlx.bench.hypercar_bench --quick  # verify GPU works first
    .venv/bin/python scripts/probe_quest_topk.py           # then run this
"""

import time
import mlx.core as mx


def benchmark_topk(n_pages: int, n_heads: int, K: int, n_trials: int = 100):
    """Benchmark top-K selection on synthetic page scores."""
    scores = mx.random.normal(shape=(1, n_heads, n_pages))
    mx.eval(scores)

    # Method 1: argsort (what we currently use in quest_attention.py)
    # Warm up
    idx = mx.argsort(-scores, axis=2)[:, :, :K]
    mx.eval(idx)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        idx = mx.argsort(-scores, axis=2)[:, :, :K]
        mx.eval(idx)
    argsort_us = (time.perf_counter() - t0) / n_trials * 1e6

    # Method 2: argpartition (O(n) average vs O(n log n) for sort)
    # Check if MLX has argpartition
    has_argpartition = hasattr(mx, 'argpartition')

    if has_argpartition:
        # Warm up
        idx = mx.argpartition(-scores, kth=K - 1, axis=2)[:, :, :K]
        mx.eval(idx)

        t0 = time.perf_counter()
        for _ in range(n_trials):
            idx = mx.argpartition(-scores, kth=K - 1, axis=2)[:, :, :K]
            mx.eval(idx)
        argpart_us = (time.perf_counter() - t0) / n_trials * 1e6
    else:
        argpart_us = None

    return argsort_us, argpart_us


def main():
    print("Quest Top-K Page Selection Probe")
    print("=" * 70)
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"argpartition available: {hasattr(mx, 'argpartition')}")
    print()

    configs = [
        # (n_pages, context_desc)
        (32, "4K ctx"),
        (128, "16K ctx"),
        (512, "64K ctx"),
        (2048, "256K ctx"),
        (8192, "1M ctx"),
        (64000, "8M ctx (stress)"),
    ]

    K_values = [16, 32, 64, 256]
    n_heads = 4  # Qwen3-Coder KV heads

    print(f"{'Config':<20} {'K':>5} {'argsort µs':>12} {'argpart µs':>12} {'µs/head':>10} {'Verdict':>10}")
    print("-" * 75)

    all_pass = True
    for n_pages, desc in configs:
        for K in K_values:
            if K > n_pages:
                continue
            n_trials = 200 if n_pages < 1000 else 50

            argsort_us, argpart_us = benchmark_topk(n_pages, n_heads, K, n_trials)
            best_us = argpart_us if argpart_us is not None else argsort_us
            us_per_head = best_us / n_heads

            # Budget: 200µs per head at the target context length
            verdict = "PASS" if us_per_head < 200 else "FAIL"
            if verdict == "FAIL":
                all_pass = False

            ap_str = f"{argpart_us:.0f}" if argpart_us is not None else "N/A"
            print(f"  {desc:<18} {K:>5} {argsort_us:>11.0f} {ap_str:>12} {us_per_head:>9.1f} {verdict:>10}")

    print()
    print("=" * 70)
    if all_pass:
        print("VERDICT: PASS — Quest top-K selection is viable at all tested sizes")
    else:
        print("VERDICT: FAIL — some configurations exceed 200µs/head budget")
    print("=" * 70)


if __name__ == "__main__":
    main()
