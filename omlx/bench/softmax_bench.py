# SPDX-License-Identifier: Apache-2.0
"""MLX softmax microbench — audit whether mx.softmax is fused or unfused.

Derived from arXiv:2510.18921 which measured mx.softmax at 27.91 ms on M1
vs 1.06 ms CUDA baseline (26x gap), worse than matmul (6.6x). If confirmed
on M4 Pro, this motivates a fused softmax-matmul shader.

Usage:
    python -m omlx.bench.softmax_bench [--iterations N] [--warmup N]

Reports:
    - mx.softmax standalone at production shapes (B=1, H=32, D=64)
    - mx.fast.scaled_dot_product_attention (MLX's fused path)
    - softmax + matmul (unfused attention pattern)
    - Comparison with paper's M1 baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

# Qwen3-Coder-30B-A3B attention shapes
B = 1       # batch size (single-request inference)
H = 32      # query heads
H_KV = 4    # KV heads (GQA 8:1)
D = 64      # head_dim = hidden_size(2048) / n_heads(32)
LAYERS = 48


def bench_softmax(seq_lens: list[int], iterations: int, warmup: int) -> dict:
    """Benchmark mx.softmax at various sequence lengths."""
    results = {}
    for S in seq_lens:
        # Shape: attention scores before softmax = (B, H, S_q, S_kv)
        # During prefill with chunking, S_q = chunk_size, S_kv = accumulated
        # For this audit we test the square case: S_q = S_kv = S
        scores = mx.random.normal((B, H, min(S, 4096), S))
        mx.eval(scores)

        # Warmup
        for _ in range(warmup):
            out = mx.softmax(scores, axis=-1)
            mx.eval(out)

        # Timed
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = mx.softmax(scores, axis=-1)
            mx.eval(out)
            times.append(time.perf_counter() - t0)

        median = sorted(times)[len(times) // 2]
        results[S] = {
            "seq_len": S,
            "shape": f"({B}, {H}, {min(S, 4096)}, {S})",
            "median_ms": round(median * 1000, 3),
            "min_ms": round(min(times) * 1000, 3),
            "max_ms": round(max(times) * 1000, 3),
            "elements": B * H * min(S, 4096) * S,
        }
        del scores
        mx.clear_cache()

    return results


def bench_sdpa(seq_lens: list[int], iterations: int, warmup: int) -> dict:
    """Benchmark mx.fast.scaled_dot_product_attention (fused path)."""
    results = {}
    scale = D ** -0.5

    for S in seq_lens:
        S_q = min(S, 4096)  # chunk-like query length
        S_kv = S

        q = mx.random.normal((B, H, S_q, D))
        # GQA: K,V have H_KV heads, repeated internally by SDPA
        k = mx.random.normal((B, H_KV, S_kv, D))
        v = mx.random.normal((B, H_KV, S_kv, D))
        mx.eval(q, k, v)

        # Warmup
        for _ in range(warmup):
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            mx.eval(out)

        # Timed
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            mx.eval(out)
            times.append(time.perf_counter() - t0)

        median = sorted(times)[len(times) // 2]
        results[S] = {
            "seq_len": S,
            "q_shape": f"({B}, {H}, {S_q}, {D})",
            "kv_shape": f"({B}, {H_KV}, {S_kv}, {D})",
            "median_ms": round(median * 1000, 3),
            "min_ms": round(min(times) * 1000, 3),
            "max_ms": round(max(times) * 1000, 3),
        }
        del q, k, v
        mx.clear_cache()

    return results


def bench_unfused_attention(seq_lens: list[int], iterations: int, warmup: int) -> dict:
    """Benchmark unfused softmax + matmul (the naive attention pattern)."""
    results = {}
    scale = D ** -0.5

    for S in seq_lens:
        S_q = min(S, 4096)
        S_kv = S

        q = mx.random.normal((B, H, S_q, D))
        k = mx.random.normal((B, H, S_kv, D))  # Full heads for unfused path
        v = mx.random.normal((B, H, S_kv, D))
        mx.eval(q, k, v)

        # Warmup
        for _ in range(warmup):
            scores = (q @ k.swapaxes(-1, -2)) * scale
            weights = mx.softmax(scores, axis=-1)
            out = weights @ v
            mx.eval(out)

        # Timed
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            scores = (q @ k.swapaxes(-1, -2)) * scale
            weights = mx.softmax(scores, axis=-1)
            out = weights @ v
            mx.eval(out)
            times.append(time.perf_counter() - t0)

        median = sorted(times)[len(times) // 2]
        results[S] = {
            "seq_len": S,
            "median_ms": round(median * 1000, 3),
            "min_ms": round(min(times) * 1000, 3),
            "max_ms": round(max(times) * 1000, 3),
        }
        del q, k, v
        mx.clear_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MLX softmax microbench (Task 87)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Timed iterations per config (default: 50)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to file")
    args = parser.parse_args()

    # Report MLX version
    print(f"MLX version: {mx.__version__}")
    print(f"Metal device: {mx.default_device()}")
    print(f"Shapes: B={B}, H={H}, H_KV={H_KV}, D={D}, Layers={LAYERS}")
    print(f"Iterations: {args.warmup} warmup + {args.iterations} timed")
    print()

    # Sequence lengths matching Hypercar production shapes
    # Keep memory safe: skip very large sizes if they'd blow up
    seq_lens = [2048, 4096, 8192, 16384]

    # 1. Standalone softmax
    print("=" * 60)
    print("1. mx.softmax standalone")
    print("=" * 60)
    softmax_results = bench_softmax(seq_lens, args.iterations, args.warmup)
    for S, r in softmax_results.items():
        print(f"  S={S:>6d}  shape={r['shape']:>30s}  "
              f"median={r['median_ms']:>8.3f} ms  "
              f"min={r['min_ms']:>8.3f}  max={r['max_ms']:>8.3f}")
    print()

    # 2. SDPA (fused path)
    print("=" * 60)
    print("2. mx.fast.scaled_dot_product_attention (fused)")
    print("=" * 60)
    sdpa_results = bench_sdpa(seq_lens, args.iterations, args.warmup)
    for S, r in sdpa_results.items():
        print(f"  S={S:>6d}  Q={r['q_shape']:>25s}  "
              f"median={r['median_ms']:>8.3f} ms  "
              f"min={r['min_ms']:>8.3f}  max={r['max_ms']:>8.3f}")
    print()

    # 3. Unfused attention (softmax + matmul)
    print("=" * 60)
    print("3. Unfused: QK^T * scale -> softmax -> @V")
    print("=" * 60)
    unfused_results = bench_unfused_attention(seq_lens, args.iterations, args.warmup)
    for S, r in unfused_results.items():
        print(f"  S={S:>6d}  "
              f"median={r['median_ms']:>8.3f} ms  "
              f"min={r['min_ms']:>8.3f}  max={r['max_ms']:>8.3f}")
    print()

    # 4. Analysis
    print("=" * 60)
    print("4. Analysis")
    print("=" * 60)
    for S in seq_lens:
        sm = softmax_results[S]["median_ms"]
        sdpa = sdpa_results[S]["median_ms"]
        uf = unfused_results[S]["median_ms"]
        # Softmax fraction of unfused attention
        sm_frac = sm / uf * 100 if uf > 0 else 0
        # SDPA speedup over unfused
        speedup = uf / sdpa if sdpa > 0 else 0
        print(f"  S={S:>6d}: softmax={sm:.3f}ms ({sm_frac:.0f}% of unfused), "
              f"SDPA={sdpa:.3f}ms ({speedup:.1f}x faster than unfused)")

    # Paper comparison (M1, older MLX)
    print()
    print("  Paper baseline (arXiv:2510.18921, M1 Max, older MLX):")
    print("    mx.softmax: 27.91 ms (shape unknown)")
    print("    CUDA ref:    1.06 ms")
    if 2048 in softmax_results:
        our_2k = softmax_results[2048]["median_ms"]
        print(f"    Our M4 Pro @2K: {our_2k:.3f} ms")
        if our_2k < 5.0:
            print("    VERDICT: Softmax appears fixed/fast on current MLX — "
                  "no fused shader needed.")
        else:
            print("    VERDICT: Softmax still slow — fused shader work justified.")

    # Write results
    all_results = {
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "shapes": {"B": B, "H": H, "H_KV": H_KV, "D": D, "layers": LAYERS},
        "iterations": args.iterations,
        "warmup": args.warmup,
        "softmax": {str(k): v for k, v in softmax_results.items()},
        "sdpa": {str(k): v for k, v in sdpa_results.items()},
        "unfused": {str(k): v for k, v in unfused_results.items()},
    }

    output_path = args.output or "research/mlx_softmax_audit.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
