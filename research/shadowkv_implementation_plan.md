# ShadowKV Implementation Plan

**Date**: 2026-04-15
**Status**: Design note — scopes the M-effort implementation
**Goal**: Reduce K cache memory by ~65% to enable 128K+ context under co-tenancy

## Background

Task 44 (ShadowKV SVD probe) validated that K cache is low-rank:
- Median rank@99%: 178/512 (35% of max dimension)
- Outlier layers: 3, 5, 8, 9, 45 have rank < 100 (extremely low-rank)
- No layers exceed rank 223 (44% of max)

Task 53 (MLA quality validation) showed that post-hoc joint KV SVD **fails**
during autoregressive decode (7% token agreement). However, K-only per-head
compression is more robust because:
1. K vectors are only used for scoring (QK^T) — errors affect attention
   weights but not directly the output value computation
2. V cache stays at full precision, so the weighted sum produces exact values
   for however the (approximate) attention weights distribute
3. Per-head SVD preserves the head-specific subspace better than joint SVD

## Architecture

### Core Idea (ShadowKV paper: arXiv:2410.21465)

For each attention layer, maintain TWO representations of the K cache:
1. **Low-rank shadow**: U_k × S_k (top-r singular vectors × values) — small, covers most energy
2. **Sparse residual**: original K vectors for the top-p% highest-norm tokens — exact, sparse

At decode time:
```
K_approx = U_k @ S_k                  # (r, D) dense, r ≈ 178
K_exact  = K_original[top_p_indices]   # (p*T, D) sparse, p ≈ 0.05
K_full   = merge(K_approx, K_exact)    # Used for attention
```

### Qwen3-Coder Specific Design

**Model dimensions**: 48 layers, H_kv=4 GQA heads, D=128, max_rank=512

**Per-layer configuration** (based on Task 44 probe data):

| Layer Group | Layers | Median Rank@99% | Strategy |
|-------------|--------|-----------------|----------|
| Ultra-low-rank | 3, 5, 8, 9, 45 | 10-70 | SVD only, r=64 |
| Standard | 0-2, 4, 6-7, 10-44, 46-47 | 150-223 | SVD (r=192) + sparse residual (p=5%) |

**Memory at 128K context** (the first milestone):

| Component | Current (3-bit GQA) | ShadowKV |
|-----------|-------------------|----------|
| K cache | ~2.7 GB | ~1.0 GB (SVD) + ~0.15 GB (sparse) = 1.15 GB |
| V cache | ~2.7 GB | ~2.7 GB (unchanged) |
| **Total KV** | **~5.4 GB** | **~3.85 GB (29% savings)** |

**Memory at 1M context** (the Goal 1 target):

| Component | Current (3-bit GQA) | ShadowKV |
|-----------|-------------------|----------|
| K cache | ~11.25 GB | ~4.0 GB (SVD) + ~0.6 GB (sparse) = 4.6 GB |
| V cache | ~11.25 GB | ~11.25 GB (unchanged) |
| **Total KV** | **~22.5 GB** | **~15.85 GB (30% savings)** |
| Model + KV | **~39.7 GB** | **~33.1 GB** (fits with 15 GB headroom) |

## Implementation Steps

### Step 1: Per-head SVD projections (S effort, GPU needed)
- Extend `scripts/compute_mla_projections.py` to compute per-head K-only SVD
- Output: per-layer, per-head (U_k, S_k) matrices saved to `omlx/patches/shadowkv_projections/`
- This replaces the joint KV projections that failed quality validation

### Step 2: ShadowKVCache class (S effort, no GPU)
- New class in `omlx/shadowkv_cache.py` that wraps MLX's KVCache
- Implements compress_k()/expand_k() using per-head SVD projections
- V cache passthrough (no compression)
- Token-norm tracking for sparse residual selection

### Step 3: Wire into model attention (M effort, GPU needed)
- Add `--kv-mode shadow` to hypercar_server and hypercar_bench
- In the model's attention call, replace `cache.state[0]` (K) with
  the ShadowKV approximate K when context exceeds a threshold
- Benchmark: NIAH 4K/16K/64K, Code Intel, RULER, MMLU-Pro quality gates

### Step 4: Quality validation at 128K (S effort, GPU needed)
- Run `--niah-only --niah-context 128K --kv-mode shadow`
- Compare Metal peak, swap, and NIAH accuracy vs native mode
- If PASS: update CLAUDE.md Goal 1 status to "validated to 128K"

## Risks

1. **Per-head SVD may also fail during decode** (same post-hoc problem as MLA,
   but less severe because V is exact). Mitigation: the sparse residual covers
   the highest-norm tokens where errors matter most.

2. **Overhead of merge operation** at decode time adds latency. Mitigation:
   the merge is a simple index_put at p=5% of tokens — ~0.5ms at 128K.

3. **Per-layer rank varies** (10 to 223). Using a fixed r wastes memory on
   low-rank layers. Mitigation: per-layer adaptive r from the probe data.

## Decision Gate

After Step 3, run the full benchmark (`--full --kv-mode shadow`).
The feature ships only if ALL existing gates pass AND Metal peak at 64K
drops by ≥ 20% vs the native 3-bit baseline.
