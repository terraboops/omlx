# MLX Attention Dispatch Analysis (v0.31.1)

Survey of `mx.fast.scaled_dot_product_attention` dispatch for Qwen3-Coder-30B-A3B
on M4 Pro (Apple Silicon). Source: `ml-explore/mlx` tag `v0.31.1`.

## Dispatch Decision Tree

```
scaled_dot_product_attention(Q, K, V, scale, mask)
  │
  ├─ Q_seq_len > 8 ──────────► steel_attention (FlashAttention-2, AMX)
  │                             File: steel/attn/kernels/steel_attention.metal
  │
  └─ Q_seq_len ≤ 8 ──────────► sdpa_vector variants
      │                         File: sdpa_vector.h
      │
      ├─ Device is M2+ AND K_seq_len ≥ 1024 ──► sdpa_vector_2pass
      ├─ GQA active AND K_seq_len ≥ 4096 ─────► sdpa_vector_2pass
      └─ Otherwise ───────────────────────────► sdpa_vector (single pass)
```

**Key threshold**: `Q_seq_len > 8` is the prefill/decode split.
- Prefill (chunked, L=4096 per chunk) → always hits `steel_attention` (AMX path)
- Decode (L=1) → always hits `sdpa_vector` (scalar path, no AMX)

## Per-Shape Dispatch Table (Qwen3-Coder-30B-A3B)

Model config: B=1, H_q=32, H_kv=4, D=64, GQA=8

### Prefill (L > 8)

| Shape | Kernel | Backend | Tile Sizes | AMX? |
|-------|--------|---------|------------|------|
| B=1, H=32, L=4096, D=64 | `steel_attention_*_bq32_bk32_bd64_wm4_wn1` | Metal (steel) | BQ=32, BK=32, BD=64 | **YES** (`simdgroup_matrix`) |
| B=1, H=32, L=8192, D=64 | same | Metal (steel) | BQ=32, BK=32, BD=64 | **YES** |

Grid: `(ceil(L/BQ), H_q, B)` = `(128, 32, 1)` at L=4096
Threadgroup: `(32, WM=4, WN=1)` = `(32, 4, 1)` = 128 threads

### Decode (L=1)

| Shape | Kernel | Backend | Block Sizes | AMX? |
|-------|--------|---------|-------------|------|
| B=1, H_q=32, L_q=1, K=2K, D=64 | `sdpa_vector` | Metal (vector) | BN=32, BD=32 | **NO** (scalar dot product) |
| B=1, H_q=32, L_q=1, K=16K, D=64 | `sdpa_vector_2pass` | Metal (vector) | BN=32, BD=32 | **NO** |
| B=1, H_q=32, L_q=1, K=64K, D=64 | `sdpa_vector_2pass` | Metal (vector) | BN=32, BD=32 | **NO** |

Two-pass triggers: GQA=8 AND K ≥ 4096 → two-pass for all contexts ≥ 4K.
Single-pass only at very short context (< 4096 / GQA).

## AMX Acceleration Details

### Prefill (steel_attention) — YES, AMX via `simdgroup_matrix`
- `#include <metal_simdgroup_matrix>` in `steel/attn/mma.h`
- `simdgroup_multiply_accumulate()` for Q×K^T and attn×V
- Fragment size: 8×8 (`simdgroup_matrix<T, 8, 8>`)
- Online softmax (FlashAttention-2 style): maintains running max + sum-exp
- Tiling: outer loop over Q blocks (BQ=32), inner loop over K blocks (BK=32 or 16)

For D=64: BK=32 (D fits in 2 SIMD fragments of 32)
For D=128: BK=16 (D needs 4 SIMD fragments, so smaller K block to fit registers)

### Decode (sdpa_vector) — NO AMX
- Manual dot products: `score += q[j] * k[j]` accumulated per thread
- `simd_sum()` for horizontal reduction (SIMD shuffle, not AMX matrix)
- `fast::exp()` for softmax (scalar)
- No `simdgroup_matrix` usage

## GQA Handling

### Prefill
- Grid launches `H_q` heads (32), but K/V indexing uses `H_kv` (4)
- Each Q head maps to its KV head: `kv_head = q_head / gqa_factor`
- All 8 Q heads sharing a KV head are independent threadgroups

### Decode (vector)
- `gqa_factor` passed as kernel parameter
- Two-pass: `threadgroup.y` iterates over GQA heads within one KV head
- Constraint: `L_q * gqa_factor ≤ 32` → with GQA=8, max L_q=4 (but decode is L_q=1, fine)

## Quantized KV Cache

**Not supported inside any SDPA kernel.** All three kernel variants (steel, vector, 2pass)
accept only float/float16/bfloat16 tensors. Quantized KV must be dequantized upstream
before calling `mx.fast.scaled_dot_product_attention`.

This means:
- Our TQ3 fused decode kernel (`_fused_tq_sdpa`) bypasses MLX's SDPA entirely — correct
- Native `QuantizedKVCache` dequantizes before calling MLX SDPA — confirmed
- Any Quest page selection that gathers then calls MLX SDPA is fine (gathered data is fp16)

## Implications for Hypercar Optimization

### Prefill is already on AMX ✓
The `steel_attention` kernel uses `simdgroup_matrix` (Apple's AMX unit) for both
Q×K^T and attn×V matrix multiplies. At our prefill chunk size of 4096, the grid
launches 128 Q-blocks × 32 heads = 4096 threadgroups, each doing 8×8 AMX matmuls.
This is already the fast path — MInference sparse attention (Task 5) should be
careful not to fall off this path by using per-head masking that prevents the
kernel from launching.

### Decode is NOT on AMX ✗
The `sdpa_vector` kernel uses scalar dot products, not `simdgroup_matrix`. This is
the expected design — decode has L_q=1, so there's no matrix structure to exploit.
The bottleneck at decode is memory bandwidth (loading K/V from cache), not compute.
Our TQ3 fused kernel bypasses this entirely by operating on packed data.

### D=64 vs D=128 matters
Qwen3-Coder-30B-A3B has D=64 (hidden_size=2048, n_heads=32). The steel kernel
uses BK=32 at D=64 vs BK=16 at D=128. This means our model gets 2x more K tokens
per tile than a D=128 model, which is favorable for prefill throughput.

### The MInference interaction risk
MInference (Task 5) applies per-head sparse masks as additive masks. Since the
steel kernel accepts array masks and broadcasts them, this should work. But if
per-head masking prevents the kernel from batching heads efficiently, it could
regress prefill speed even while reducing compute. Worth measuring.
