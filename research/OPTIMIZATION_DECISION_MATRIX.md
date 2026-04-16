# Optimization Decision Matrix

Consolidated probe results from 44 completed tasks.
All measurements on Qwen3-Coder-30B-A3B-Instruct-8bit, M4 Pro 48GB, MLX 0.31.1.
Last updated: 2026-04-16.

## Summary Table

| Technique | Task | Viable? | Savings | Quality Impact | Priority |
|-----------|------|---------|---------|---------------|----------|
| **DuoKVCache** (fp16 + streaming ring buffer) | 13 | **SHIPPED** | Zero swap, +17% quality | MMLU-Pro 48→62%, HumanEval 90→95% | **DEFAULT** |
| **SnapKV + CAOTE** (value-aware eviction + re-RoPE compaction) | 46, 100 | **SHIPPED** | 75% KV eviction, physical compaction | 100% at 25% keep (CAOTE fixes 16K NIAH fail) | **CRITICAL for Goal 1** |
| **MLA joint KV** (post-hoc SVD) | 53 | **QUALITY FAIL** | 76% rank compression | 7% token agreement — errors compound | SKIP |
| **ShadowKV** (K-only post-hoc SVD) | 44 | **POOR TRADEOFF** | 3% K at 82% agree | NIAH 16% at 34% compression | SKIP |
| **DuoAttention** (streaming head calibration) | 12 | **SHIPPED** | 59% streaming heads | Feeds DuoKVCache | **DONE** |
| **Metal warmup** | 25 | **SHIPPED** | 2.25x decode, no cold-start | None | **DONE** |
| **EAGLE-2** (tree spec-decode) | 28 | **FEASIBLE** | 2.31x predicted speedup | Tree masks work in SDPA (<1e-6 error) | HIGH |
| **Quest** (page selection) | 3, 24b | **YES** | argpartition <200µs budget | Fails NIAH (norm-based scoring) | MEDIUM |
| **MInference** (sparse prefill) | 4, 5 | YES (code) | TBD | Needs GPU validation run | MEDIUM |
| **Spec-decode gate** (MagicDec cost model) | 56 | **SHIPPED** | Predicts 2.31x at 70% accept | Framework for all spec-decode | **DONE** |
| **TQ3 2-bit KV** (aggressive quantization) | 22 | PARTIAL | Goal 5 fixed | NIAH 4K regression | CONDITIONAL |
| **ProMoE** (expert lazy-load) | 16, 32 | MARGINAL | 3.6 GB at 87.5% | 0.1% experts fully cold, concentrated top-10 | LOW |
| **KV fragmentation** (allocator overhead) | 31 | **NEGLIGIBLE** | Only 2.1% (~0.5 GB at 1M) | N/A | **SKIP** |
| **MLX softmax** (fused kernel gap) | 87 | **CLOSED** | Gap is 1.0x (was 26x on M1) | N/A | **SKIP** |
| **LayerSkip** (self-speculative decode) | 38 | **NO** | N/A | MoE routing blocks early exit | SKIP |

## Recommended Priority Order

### Tier 1: Shipped (proven viable, high impact)
1. **DuoKVCache** — DEFAULT mode. Zero swap, 52.7 tok/s, MMLU-Pro 62%.
2. **DuoAttention** — 59% streaming heads. Feeds DuoKVCache. Calibration table shipped.
3. **Metal warmup** — Decode 20→45 tok/s. Default since commit 7a62b53.
4. **Spec-decode gate** — Cost model calibrated. Predicts when EAGLE-2/TriForce helps.
5. **SnapKV + CAOTE + BUZZ + submodular** — Full eviction stack with 6 composable layers.
   fp16 mode: 100% at 25% keep through 128K (4.5-9 GB saved). NIAH PASS at all lengths.
   Native 3-bit mode: PASS at 4K/16K, **FAIL at 64K** — dequant noise after re-RoPE
   causes output corruption. Fix needed: capture fp16 K for compaction (not just scoring).
   `--snapkv-keep K --caote --segmented-evict 512` on server.

### Tier 2: Next Up (validated, ready to build)
6. **EAGLE-2** — Tree masks work in SDPA (1.03-1.12x overhead). Draft-head training
   needed (Task 29, cloud GPU). Predicted 2.31x decode speedup at 70% accept rate.
7. **Quest** — argpartition <200µs at 64K pages. Wire into decode path for sublinear
   attention at long contexts.

### Tier 3: Conditional (viable with caveats)
8. **MInference** — Sparse prefill patterns calibrated. Runtime built but needs GPU
   validation for actual prefill speedup measurement.
9. **TQ3 2-bit** — Fixes Goal 5 but regresses NIAH 4K. Use for contexts >16K only.
10. **ProMoE** — 0.1% experts fully cold, but top-10 hold 14-27% of dispatches.
    Marginal savings (3.6 GB) with quality risk. Profile data available for retry.

### Tier 4: Skip (disproven or negligible)
11. **KV fragmentation paging** — Only 2.1% fragmentation, ~0.5 GB at 1M. Not worth it.
12. **MLX softmax fusion** — Gap closed in MLX 0.31.1 (4.9ms vs paper's 27.9ms on M1).
13. **LayerSkip** — MoE routing blocks early exit. 0% agreement at depth ≤40.

## Combined Goal 1 + 5 Story

Current: 17.2 GB model + 22.5 GB KV at 1M = 39.7 GB (barely fits, swap-thrashes at 128K under co-tenancy).

### ~~Path A: MLA joint compression~~ (BLOCKED — quality fail)
- Joint KV rank 241/1024 (24%) — good compression ratio
- BUT post-hoc SVD produces 7% token agreement during decode
- Error compounds: 0.85% per-layer × 48 layers × decode steps = drift
- Would need end-to-end MLA training (not viable for post-trained models)

### Path A (revised): ShadowKV + DuoAttention (best viable path)
- Streaming heads (59%): ring buffer ~0.1 GB
- Retrieval heads K (41%): SVD-compressed to 45% → 41% × 11.25 GB × 0.45 = 2.1 GB
- Retrieval heads V: full 3-bit → 41% × 11.25 GB = 4.6 GB
- Total KV: 0.1 + 2.1 + 4.6 = **6.8 GB** (70% savings)
- Total: 17.2 + 6.8 = **24.0 GB** (fits with 24 GB headroom)

## Data Sources

| File | What it measures | Task |
|------|-----------------|------|
| `research/mla_rank_20260415.json` | Per-layer K+V+KV joint SVD rank at 99/99.9% energy | 53 |
| `research/mla_quality_validation.json` | MLA post-hoc SVD decode quality: 7% agreement (FAIL) | 53 |
| `research/shadowkv_implementation_plan.md` | ShadowKV 4-step implementation plan for Goal 1 | — |
| `research/shadowkv_quality_validation.json` | ShadowKV K-only decode: 52% agreement, all answers correct | — |
| `research/shadowkv_rank_sweep.json` | Rank 64-128 sweep: code 100% all ranks, NIAH 16% at rank 85 | — |
| `research/attn_weighted_svd_error.json` | Weighted/uniform error ratio ≈ 0.9x — argmax flip, not error concentration | — |
| `research/attention_weighted_codec_selection.md` | Per-head-type codec: SnapKV for retrieval, SVD for streaming | 90b |
| `research/snapkv_selection_validation.json` | SnapKV: needle preserved at 25% keep ratio | 46 |
| `research/shadowkv_rank_20260413.json` | Per-layer K-only SVD rank at 99/99.5/99.9% energy | 44 |
| `research/MLX_ATTN_DISPATCH.md` | MLX SDPA kernel dispatch (AMX vs vector) | 30 |
| `research/mlx_softmax_audit.json` | mx.softmax vs SDPA vs unfused at production shapes | 87 |
| `research/KV_FRAGMENTATION.md` | Metal active vs peak memory at 4K/16K/64K | 31 |
| `research/kv_fragmentation_profile.json` | Raw fragmentation samples per context length | 31 |
| `omlx/specdec_constants.json` | Calibrated decode latency model (compute + KV-load) | 56 |
| `omlx/patches/duoattention_policies/*.json` | Per-head streaming/retrieval classification | 12 |
| `omlx/patches/layerskip_thresholds/*.json` | Per-layer early-exit agreement rates | 38 |
| `omlx/patches/minference_patterns/*.json` | Per-head attention pattern classification | 4 |
| `omlx/patches/promoe_profiles/*.json` | Per-layer×expert MoE dispatch frequencies | 32 |
| `docs/bimodal_timing_root_cause.md` | Metal JIT cold-start bimodality | 20 |
| `scripts/probe_eagle_tree_attn.py` | EAGLE-2 tree mask correctness + overhead | 28 |
| `scripts/probe_quest_topk.py` | argpartition vs argsort at various page counts | 24b |
