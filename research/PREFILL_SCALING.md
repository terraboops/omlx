# Prefill Speed Scaling Analysis

Measured on M4 Pro 48GB, Qwen3-Coder-30B-A3B-Instruct-8bit, native 3-bit KV.

## Per-Context Prefill Speed (chunk=4096)

| Context | tok/s | vs 4K | Metal GB | Time |
|---------|-------|-------|----------|------|
| 2K | 376 | — | 32.5 | 5.4s |
| 4K | 483 | 100% | 32.5 | 8.5s |
| 8K | 461 | 95% | 32.6 | 17.8s |
| 16K | 332 | 69% | 32.8 | 49.4s |

## Chunk Size Sensitivity (at 2K context)

| Chunk Size | tok/s | Relative |
|-----------|-------|----------|
| 1024 | 311 | 0.45x |
| 2048 | 697 | 1.00x |
| 4096 | 699 | 1.00x |
| 8192 | 699 | 1.00x |
| 16384 | 697 | 1.00x |

Chunk=1024 is 2.25x slower — kernel launch overhead dominates.
Chunks ≥2048 are equivalent — the default 4096 is optimal.

## Root Cause

The prefill speed drop is NOT from:
- Chunk size (2048-16384 all same speed) ✗
- Metal memory growth (32.5→32.8 GB, negligible) ✗
- KV allocation (quantized KV adds <0.3 GB at 16K) ✗

It IS from:
- **O(n²) attention cost within each chunk**: The steel_attention kernel
  computes Q×K^T where K has accumulated_context tokens. At chunk 4 of
  a 16K prefill, the attention matrix is 4096×16384 — 4x larger than
  chunk 1's 4096×4096. This grows linearly per chunk.
- **MoE expert dispatch**: Each token routes through 8/128 experts.
  At 16K, the router must dispatch more total tokens across the expert
  set, increasing memory traffic.

## Implications

- Goal 4 target (500 tok/s constant) requires attention cost to be
  sublinear in accumulated context — exactly what MInference sparse
  prefill (Task 5) addresses.
- MInference currently only works with fp16 KV (guard added for
  quantized KV composability in commit 450cf9f).
- At 4K context, prefill already exceeds 500 tok/s target (483-566
  depending on measurement point).
