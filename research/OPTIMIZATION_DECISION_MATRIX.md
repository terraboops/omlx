# Optimization Decision Matrix

Consolidated probe results from Tasks 12, 16, 30, 38, 44.
All measurements on Qwen3-Coder-30B-A3B-Instruct-8bit, M4 Pro 48GB.

## Summary Table

| Technique | Task | Viable? | Savings | Quality Impact | Priority |
|-----------|------|---------|---------|---------------|----------|
| **DuoKVCache** (fp16 + streaming ring buffer) | 13 | **SHIPPED** | Zero swap, +17% quality | MMLU-Pro 48→62%, HumanEval 90→95% | **DEFAULT** |
| **ShadowKV** (K low-rank compression) | 44 | **YES** | ~65% K cache | <1% info loss | HIGH (for 64K+) |
| **DuoAttention** (streaming head calibration) | 12 | **SHIPPED** | 59% streaming heads | Feeds DuoKVCache | **DONE** |
| **Metal warmup** | 25 | **SHIPPED** | 2.25x decode, no cold-start | None | **DONE** |
| **Quest** (page selection) | 3, 24b | **YES** | 1.55x decode speedup | Fails NIAH (norm-based scoring) | MEDIUM |
| **MInference** (sparse prefill) | 4, 5 | YES (code) | TBD | Needs rectangular masks for chunked prefill | MEDIUM |
| **TQ3 2-bit KV** (aggressive quantization) | 22 | PARTIAL | Goal 5 fixed | NIAH 4K regression | CONDITIONAL |
| **ProMoE** (expert lazy-load) | 16 | MARGINAL | 3.6 GB at 87.5% | Repetition artifacts | LOW |
| **LayerSkip** (self-speculative decode) | 38 | **NO** | N/A | MoE routing blocks early exit | SKIP |

## Recommended Priority Order

### Tier 1: Ship Now (proven viable, high impact)
1. **Metal warmup** — DONE. Decode 20→45 tok/s. Default since commit 7a62b53.
2. **ShadowKV** — K cache is low-rank (median 177/512 at 99% energy). Implement
   SVD-compressed K storage for retrieval heads. Saves ~65% of K cache memory.
3. **DuoAttention** — 59% of heads are streaming. Implement ring-buffer KV for
   streaming heads. Saves ~6.5 GB at 1M context. Calibration table ready.

### Tier 2: Worth Pursuing (viable, moderate impact)
4. **Quest** — Page selection viable (argpartition <200µs at 64K pages). Wire
   into decode path for sublinear attention at long contexts.
5. **MInference** — Sparse prefill patterns calibrated. Runtime dispatch built.
   Needs GPU validation run to measure actual prefill speedup.
6. **TQ3 2-bit** — Fixes Goal 5 swap pressure but regresses NIAH 4K. Use
   conditionally for contexts >16K where swap is the bottleneck.

### Tier 3: Skip or Defer
7. **ProMoE** — Only 3.6 GB savings at 87.5% residency with quality artifacts.
   Fused QuantizedSwitchLinear makes per-expert lazy-load hard. Skip unless
   DuoAttention + ShadowKV combined is insufficient.
8. **LayerSkip** — 0% agreement at depths ≤40, 55% at depth 44 (4 layers
   skipped). MoE routing makes every layer critical. Not viable for this model.

## Combined Goal 5 Story

Current: 32.4 GB model + 22.5 GB KV at 1M = 54.9 GB (exceeds 48 GB).

After ShadowKV + DuoAttention:
- Streaming heads (59%): ring buffer ~0.1 GB (256 tokens × 4 heads × 64 dim × fp16)
- Retrieval heads K (41%): SVD-compressed to 35% → 41% × 11.25 GB × 0.35 = 1.6 GB
- Retrieval heads V: full 3-bit → 41% × 11.25 GB = 4.6 GB
- Total KV: 0.1 + 1.6 + 4.6 = **6.3 GB** (was 22.5 GB, **72% savings**)
- Total: 32.4 + 6.3 = **38.7 GB** (fits in 48 GB with 9.3 GB headroom)

## Data Sources

| File | What it measures |
|------|-----------------|
| `research/shadowkv_rank_20260413.json` | Per-layer SVD rank at 99/99.5/99.9% energy |
| `research/MLX_ATTN_DISPATCH.md` | MLX SDPA kernel dispatch (AMX vs vector) |
| `docs/bimodal_timing_root_cause.md` | Metal JIT cold-start bimodality |
| `omlx/patches/duoattention_policies/*.json` | Per-head streaming/retrieval classification |
| `omlx/patches/layerskip_thresholds/*.json` | Per-layer early-exit agreement rates |
| `omlx/patches/minference_patterns/*.json` | Per-head attention pattern classification |
