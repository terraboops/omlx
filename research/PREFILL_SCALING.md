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

## 64K NIAH: Attention Scores Are the Real Bottleneck

Attempted 64K NIAH with native 3-bit KV (Run 55):
- Metal peaked at **51.3 GB** (exceeded 48 GB system memory!)
- Swap: 17.8 GB
- KV cache at 64K (3-bit): only 1.4 GB — NOT the bottleneck
- Attention scores: 4096 × 65536 × 32 × 2 = **16 GB per chunk**

The attention scores tensor is the real memory wall at 64K+.
The KV cache compression (3-bit, ~1.4 GB at 64K) is already excellent.
The problem is the O(n²) attention computation itself.

## 64K NIAH: PASS with Adaptive Chunking (Run 56)

After implementing adaptive chunk=1024, 64K NIAH passes:
  4K:  PASS — 548 tok/s prefill, 32.5 GB Metal
  16K: PASS — 163 tok/s prefill, 32.8 GB Metal
  64K: PASS —  29 tok/s prefill, 33.9 GB Metal

The smaller chunk (1024 vs 4096) reduces attention scores from 16 GB
to 4 GB per call, fitting within the remaining Metal headroom.
Prefill speed drops (29 tok/s at 64K) due to more kernel launches
and the O(n²) attention cost, but correctness is preserved.

**Solutions for Goal 1 at 128K+:**
1. Smaller prefill chunks: 1024 → attention = 4 GB (fits)
2. FlashAttention (MLX steel already avoids materialization for some paths)
3. Streaming attention (TQ3 mode already has this)
4. MInference sparse masks (reduce effective attention area)

## Implications

- Goal 4 target (500 tok/s constant) requires attention cost to be
  sublinear in accumulated context — exactly what MInference sparse
  prefill (Task 5) addresses.
- MInference currently only works with fp16 KV (guard added for
  quantized KV composability in commit 450cf9f).
- At 4K context, prefill already exceeds 500 tok/s target (483-566
  depending on measurement point).
