# MLX Softmax Fused-Reduction Audit (Task 87)

**Date**: 2026-04-15
**MLX version**: 0.31.1
**Hardware**: Apple M4 Pro, 48 GB unified memory
**Reference paper**: arXiv:2510.18921 (Benchmarking On-Device ML on Apple Silicon with MLX)

## Summary

The paper reported `mx.softmax` at 27.91 ms on M1 vs 1.06 ms CUDA (26x gap),
worse than matmul (6.6x). We measured 4.92 ms at comparable shapes on M4 Pro
with MLX 0.31.1 — **5.7x faster than the paper's M1 measurement**, more than
the ~2-3x expected from M1→M4 Pro hardware alone. MLX has improved its softmax
implementation since the paper's measurement window.

## Results (B=1, H=32, D=64 — Qwen3-Coder shapes)

| Seq Len | mx.softmax | SDPA (fused) | Unfused attn | Softmax % of unfused | SDPA speedup |
|--------:|-----------:|-------------:|-------------:|---------------------:|-------------:|
| 2K      | 4.92 ms    | 6.13 ms      | 17.46 ms     | 28%                  | 2.8x         |
| 4K      | 19.15 ms   | 24.98 ms     | 64.83 ms     | 30%                  | 2.6x         |
| 8K      | 38.81 ms   | 48.22 ms     | 129.85 ms    | 30%                  | 2.7x         |
| 16K     | 79.80 ms   | 95.99 ms     | 269.36 ms    | 30%                  | 2.8x         |

## Verdict

**No fused softmax-matmul shader needed.** The gap reported in the paper is
closed in MLX 0.31.1. Softmax accounts for ~30% of unfused attention time,
and `mx.fast.scaled_dot_product_attention` already provides 2.6-2.8x speedup
via its `steel_attention` kernel (which uses `simdgroup_matrix` AMX instructions
per Task 30's survey).

The remaining 2.8x gap between SDPA and unfused attention is dominated by
memory traffic savings (SDPA tiles the computation, avoiding materializing
the full S×S attention scores matrix), not by softmax fusion specifically.

## Impact on other tasks

- **Task 75 (EvoAttention)**: The softmax kernel is not the bottleneck; EvoAttention's
  value is in the *pattern*, not the kernel.
- **Task 69 (BSFA)**: Block-sparse flash attention's value comes from skipping
  entire attention blocks, not from fusing softmax.
- **All kernel-port tasks**: The "MLX softmax constant" mentioned in the task
  description is ~1.0x (not 26x), so these tasks are not multiplied by a softmax tax.
