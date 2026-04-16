# Attention-Weighted Codec Selection for Two-Tier KV Cache

**Date**: 2026-04-15
**Status**: Design note — architectural principle derived from empirical findings

## The Core Insight

KV cache compression should use **different codecs for retrieval heads vs streaming
heads**, not different storage tiers for the same codec. The DuoAttention split
(Task 12/13: 41% retrieval, 59% streaming) determines not just where KV lives,
but *how it's compressed*.

## Why: The Argmax Instability Problem

SVD-based compression (MLA, ShadowKV) fails on retrieval tasks because:

1. **SVD minimizes uniform Frobenius error** — it preserves directions that explain
   the most *variance* across all tokens.

2. **Retrieval attention is sparse** — one needle token must win an argmax against
   64K competitors by a small margin.

3. **Uniform 0.85% error across all tokens** gives softmax enough perturbation to
   flip its argmax in at least 1 of 48 layers. Each flip is catastrophic (wrong
   token's embedding enters the residual stream).

4. **Code generation attention is distributed** — no single argmax matters. The same
   0.85% uniform error just slightly reshuffles weights without changing output.

### Empirical Evidence

| Approach | Code Agreement | NIAH Agreement | Mechanism |
|----------|---------------|----------------|-----------|
| MLA joint SVD (d_c=241) | N/A | 7% | Errors compound across K+V×48 layers |
| ShadowKV per-head SVD (r=85) | 100% | 16% | K errors flip sparse attention argmax |
| ShadowKV (r=124, 3% compress) | 100% | 47% | Less perturbation, fewer flips |
| SnapKV (50% keep, exact values) | TBD | 100% | Kept tokens have zero error |

The attention-weighted SVD error probe (commit 6b8d832) confirmed the error is
**uniformly distributed** (weighted/uniform ratio ≈ 0.9x), not concentrated on
needle tokens. The failure is discrete (argmax flip), not continuous (MSE).

## Architecture: Per-Head-Type Codec Selection

```
                    DuoAttention Head Classification
                    (Task 12: 41% retrieval, 59% streaming)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Retrieval Heads         Streaming Heads
            (sparse α, argmax-     (distributed α,
             sensitive)              MSE-tolerant)
                    │                   │
                    ▼                   ▼
            Attention-Weighted     Energy-Weighted
            Codec:                 Codec:
            • SnapKV eviction      • SVD low-rank K
            • Quest page select    • Aggressive 2-bit quant
            • Full precision on    • Ring-buffer trimming
              kept tokens          • Large compression OK
                    │                   │
                    ▼                   ▼
            Zero approx error      34%+ compression
            on kept tokens         tolerated (code=100%)
```

## Memory Projection at 128K Context

| Component | Current (DuoKV fp16) | Per-Head-Type Codec |
|-----------|---------------------|---------------------|
| Retrieval heads K (41%) | 1.1 GB | 0.55 GB (SnapKV 50% keep, exact) |
| Retrieval heads V (41%) | 1.1 GB | 0.55 GB (SnapKV 50% keep, exact) |
| Streaming heads (59%) | 0.1 GB (ring buf) | 0.1 GB (ring buf, unchanged) |
| **Total KV** | **2.3 GB** | **1.2 GB** (48% savings) |
| Model + KV | 19.5 GB | 18.4 GB |

At 1M context the savings are larger because SnapKV's 50% eviction saves
proportionally more as context grows (eviction ratio can increase with context).

## Connection to Existing Tasks

| Task | Role in This Architecture |
|------|--------------------------|
| **DuoAttention** (#12/13) | Head classification → codec selector |
| **SnapKV** (#46) | Retrieval-head codec (attention-weighted eviction) |
| **Quest** (#3) | Retrieval-head page selector (attention-weighted) |
| **ShadowKV** (#44) | Streaming-head codec only (SVD safe for distributed α) |
| **KIVI 2-bit** (#2) | Streaming-head aggressive quant (per-channel preserves structure) |

## Implementation Plan

1. **Extend DuoKVCache** to use different compression per head type
2. For retrieval heads: wire SnapKV selection at prefill end (Task 46)
3. For streaming heads: apply ShadowKV SVD at rank 85 (34% compression,
   100% code agreement — streaming heads don't do retrieval)
4. Gate: full benchmark `--kv-mode duo-compressed` must pass all existing gates

## Theoretical Connection

This is the **compressed-sensing insight** applied to KV caches:
- SVD doesn't satisfy RIP (Restricted Isometry Property) for sparse signals
- Random projections (MagicPIG LSH) do have RIP but are expensive
- Attention-weighted selection (SnapKV) is the optimal middle ground:
  importance-aware, no projection error on kept tokens
- The energy/attention metric boundary maps exactly to DuoAttention's
  streaming/retrieval classification
