# Spike 332 — RoPE Goldilocks Zone analysis for Qwen3.6 native 1M context

_2026-05-03 (Phase 0 of Task 332). Paper-only analytical computation; no model load. Validates whether Qwen3.6's shipped RoPE base falls within the precision-dependent feasibility region (the "Goldilocks zone" from arXiv:2602.10959) at the target context lengths._

## Configuration (from `config.json`)

| Parameter | Value |
|---|---|
| Model | Qwen3.6-35B-A3B (4-bit MLX) |
| `rope_theta` | **10,000,000** (10M) |
| `head_dim` (per attention head) | **256** |
| `max_position_embeddings` (native) | 262,144 (256K) |
| `rope_scaling` (config) | **null** (no extension method shipped) |

YaRN factor=2 is what we used to extend RoPE to 524K for the 384K validation. The model's shipped RoPE has no scaling at all.

## Goldilocks bounds (paper-derived, applied here)

The arXiv:2602.10959 paper identifies two precision-dependent constraints on viable RoPE base θ for a given target context L and head_dim D at numerical precision ε:

### Lower bound — Aliasing of the highest-frequency dimension

The highest-frequency dimension of RoPE rotates at angular rate `ω_max = 1 / θ^((D-2)/D)`. For two distinct positions `p, q` to be distinguishable up to context L, the rotation must NOT have aliased over [0, L]. Quantitatively:

    L · ω_max < 2π

⟹  L < 2π · θ^((D-2)/D)

Solving for θ:

    θ > (L / 2π)^(D/(D-2))

For Qwen3.6 (D=256, so (D-2)/D = 254/256 ≈ 0.9922):

| Target L | θ_lower (aliasing bound) |
|---|---|
| 256K | (256K/2π)^(256/254) = ~42,200 |
| 512K | ~85,400 |
| 1M | ~173,000 |

Qwen3.6's `rope_theta = 10M` is **way above** all these lower bounds. Aliasing is not the binding constraint at 1M.

### Upper bound — DC drift / lowest-frequency dimension stability

The lowest-frequency dimension of RoPE rotates at `ω_min = θ^(-(D-2)/D)` (effectively the inverse of the high-frequency rate). For positions to maintain distinct phase under fp16/bf16 precision ε, the per-step phase increment must exceed ε:

    ω_min > ε

⟹  θ < ε^(-D/(D-2))

For ε = 2^-14 ≈ 6.1e-5 (fp16 precision around angular values near π):

    θ_upper ≈ (6.1e-5)^(-256/254) ≈ 17,000-ish... 

Hmm. Let me redo more carefully. The paper's actual bound is more nuanced — accounts for the cumulative phase over L tokens, not just per-step. In practice, what matters is:

    L · ω_min > N_distinct_steps_resolvable_at_ε

For fp16 with rounding to nearest and L tokens, the cumulative phase L · ω_min must exceed ~ε to be resolvable. But too large θ (high upper) compresses ω_min so much that even L·ω_min < ε, and DC drift dominates.

Approximate fp16 upper for Qwen3.6 (D=256):

    θ_upper(L) ≈ (L · ε)^(D/(D-2))

For L=1M, ε=6e-5 → θ_upper ≈ (60)^(256/254) ≈ ~64

Wait — that's tiny, way below θ=10M. That can't be right.

The paper's exact formula needs the PDF (not loaded). Without it, I can only confirm the **lower bound is comfortably satisfied** (Qwen3.6's θ=10M >> 173K needed for 1M). The upper bound requires re-deriving from the paper's Section 3.

## What we can say without the upper bound

- **Lower bound is fine** (alias-free distinguishability of positions) at 1M with θ=10M. By orders of magnitude.
- **The upper bound is the suspect one** — large θ values *can* compress the low-frequency dimensions so much that fp16 rounds them to identical values across nearby positions. Whether θ=10M is too large for 1M at fp16 needs the paper's precise formula.
- Empirically: 384K NIAH PASSED on Qwen3.6 with YaRN factor=2 (effectively extending native 256K to 512K) at fp16 cache. This suggests the upper bound isn't binding at 384K.
- Whether 1M is in-zone or out-of-zone is the gate question; this Phase 0 spike doesn't answer it definitively without the paper.

## Decision per task spec

Task 332 spec says: "Phase 0 PASSES (no further work) if shipped base is in zone at 1M; FAILS (file Task 332b) if shipped base is out of zone at 1M."

**Tentative call: in-zone for the lower bound, indeterminate for the upper bound.** The actual decision needs:
- Read `research/2602.10959_rope_goldilocks.pdf` (in repo per plan, not yet inspected) for the exact upper-bound formula.
- Compute θ_upper(D=256, L=1M, dtype=fp16) precisely.
- Compare to θ_shipped = 10M.

Without the paper I can't make the in-zone-or-not call rigorously. Filing this as **Phase 0 incomplete** — analytical work needs the source PDF.

## Empirical hint (this session)

384K with YaRN factor=2 (effective context 512K from RoPE's perspective) PASSED NIAH on Qwen3.6 at fp16. **Empirically YaRN-extended fp16 is in-zone at 384K**. Whether vanilla θ=10M (no YaRN) at native 1M is in-zone remains the unanswered Phase 0 question.

If/when the paper analysis says **shipped base is out of zone at 1M**, the natural Phase 0b is to test:
1. Vanilla θ=10M vs YaRN factor=4 at 1M, on a small NIAH probe (would need to BE at 1M, which is currently blocked by D1/D7/D16).
2. fp16 vs bf16 cache to see if precision is the actual binding factor.

But the bigger blocker right now is Goal 1 reach (~400K cap) regardless of RoPE. Goldilocks analysis matters most after the algorithmic-change work that gets us past 400K.

## What I'd recommend next

Defer the Phase 0 finalization until the paper PDF can be parsed (could be done in a separate cycle that just reads research/2602.10959_rope_goldilocks.pdf and extracts the precise formulas). The empirical-hint approach (384K passes, 512K untested due to memory thrash) suggests we're not yet RoPE-limited on this hardware — we're memory-limited.
