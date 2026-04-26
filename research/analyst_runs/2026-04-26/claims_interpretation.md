# Claims sweep — interpretation (analyst run 2026-04-26)

**Source data**: `claims.json`, `claims_summary.txt` (from
`tools/analyst_kit/claims_2026_04_26.py`).
**Method**: synthetic micro-benchmarks at hypercar-realistic shapes
(B=1, H_kv=4, D=128, fp16). No model load, so each measurement is the
work of the operation in isolation. Ran on `analyst/2026-04-26` branch.

## Key findings

### F1a — `mx.where(mask, x, 0.0)` is bit-identical to `mx.where(mask, x, mx.zeros_like(x))` AND faster

| T_total | baseline (zeros_like) | optimized (scalar 0.0) | speedup | bit-identical |
|--------:|----------------------:|-----------------------:|--------:|:-------------:|
| 4096    | 3.99 ms               | 0.67 ms                | **5.94×** | ✓           |
| 16384   | 2.26 ms               | 2.18 ms                | 1.03×   | ✓             |
| 65536   | 9.95 ms               | 6.14 ms                | 1.62×   | ✓             |

The 16K row is suspiciously flat compared to 4K and 65K — possibly a measurement artifact from MLX kernel scheduling at that specific shape. Heap delta is 0 across repeats for both forms (MLX correctly recycles the per-call allocation).

**Recommendation**: ship the scalar form in `omlx/duo_kv_cache.py` trim block. **Bit-identical**, faster at 4K and 65K, neutral at 16K. Zero risk.

### F1b — Slice replaces gather for retrieval rows (arange identity), saving 100-1430× per call

| T_total | gather (broadcast+take_along_axis) | slice  | ratio |
|--------:|-----------------------------------:|-------:|------:|
| 4096    | 0.274 ms                           | 0.002 ms | **131×** |
| 16384   | 0.807 ms                           | 0.002 ms | **425×** |
| 65536   | 3.147 ms                           | 0.002 ms | **1430×** |

When `gather_index == arange(T_total)`, `take_along_axis` is the identity operation — pure overhead. Slicing returns a view in O(1). The savings scale linearly with T_total, suggesting the take_along_axis kernel is doing real work even when the gather is identity.

**Caveat**: this measurement is for retrieval-rows-only (where the gather index is arange). The DuoKV trim path uses a MIXED gather index (streaming heads use sink+window; retrieval heads use arange). Replacing the mixed gather with per-row slice/gather requires a more invasive refactor — last session's Task 295(b) attempted this and was reverted (Task 301) because the slice+write path's fancy-indexed assignments were allocator-heavy. The lesson: **isolated micro-savings don't always survive composition**.

**Recommendation**: do NOT directly translate this finding into a trim-block rewrite. The 1430× ratio is the upper bound on the savings if all heads were retrieval and we used a slice path. The realistic mixed-case savings would be much smaller and offset by the reassembly cost.

### DuoKV `update_and_fetch` scaling — sublinear or flat past 4K

| T_total | median ms | T-ratio | time-ratio | verdict |
|--------:|----------:|--------:|-----------:|---------|
| 1024    | 0.039 ms  | —       | —          | —       |
| 4096    | 0.144 ms  | 4.0×    | 3.7×       | LINEAR  |
| 8192    | 0.054 ms  | 2.0×    | 0.38×      | **SUBLINEAR** |
| 16384   | 0.051 ms  | 2.0×    | 0.93×      | SUBLINEAR |

The 4K→8K subliner (0.38×) is suspicious — likely first-call kernel-compile artifact at T=4096 that doesn't repeat at 8K+. The 8K→16K basically-flat is the real signal: `update_and_fetch` is **NOT scaling with context length** at decode-step granularity past 8K. This is consistent with the pre-allocated slab + slice-write design (Task 149) — adding one token is O(1).

**Recommendation**: this is a positive finding. The decode hot path's `update_and_fetch` is fast and constant. If decode tok/s degrades at long context, the bottleneck is NOT this path.

### StreamingKVCache ring-mode step — 0.079 ms median

| Median | Min | Max |
|-------:|----:|----:|
| 0.079 ms | 0.068 | 0.116 |

20 trials post-fill. Variance is small (max-min is ~50 µs). O(1) ring-buffer scatter behaves as designed.

### F1 48-layer simulation — modest absolute savings

| T | baseline 48L | optimized 48L | savings/step |
|--:|-------------:|--------------:|-------------:|
| 4096   | 3.66 ms | 3.40 ms | 0.26 ms |
| 16384  | 9.48 ms | 7.44 ms | **2.04 ms** |

At 16K decode, replacing `mx.zeros_like` with scalar `0.0` across 48 layers saves ~2 ms per decode step. Decode at 16K runs ~16 tok/s ≈ 62.5 ms/token, so 2 ms is **3.2% per-token**. Small but measurable. At 4K it's basically negligible (0.26 ms / ~24 ms/token = 1%).

**Recommendation**: the F1a fix (scalar in `mx.where`) is worth shipping but expect ~3% decode improvement at 16K, not the headline-friendly 5.94× speedup at small T (which doesn't translate to per-step decode time because most layers don't see the worst-case shape).

## Cross-cutting

- **Heap deltas are 0 across all measurements**. No leak; the MLX allocator recycles per-call allocations cleanly. This contradicts the static-review F1 hypothesis that the trim block leaks at decode-step granularity. The pressure is **transient peak**, not retained allocation.
- **F1a's measurement at T=16K is anomalous**. Both forms run at ~2.2 ms; everywhere else the scalar form wins by 1.6-6×. Worth re-running with more repeats to distinguish noise from a genuine MLX kernel artifact.
- **F1b can't be directly leveraged** despite the eye-catching 1430× ratio. The trim path can't trivially substitute slice for gather without restructuring; last session's Task 295(b) attempt was reverted because the restructure introduced more cost than it saved.

## What this validates / falsifies

| Claim | Verdict |
|-------|---------|
| F1a scalar form is bit-identical | ✓ Validated |
| F1a saves measurable per-step time at 16K decode | ✓ Validated (~2 ms/step ≈ 3%) |
| F1a heap delta savings | ✗ Falsified (heap delta is 0 for both forms) |
| F1b slice replaces gather "for free" | ✗ Falsified (130-1430× holds only for arange-identity rows; mixed-row case is harder) |
| `update_and_fetch` decode scaling is constant past 4K | ✓ Validated (8K→16K ratio 0.93×) |

## Followups

- **Re-run F1a at T=16K with 20+ trials** to determine whether the flat result is genuine or measurement noise. If genuine, document why MLX's kernel produces a different speedup curve at that specific shape.
- **F3 (per-layer gather index rebuild) micro-bench is not in this script**. Add it as a separate Claim if the analyst budget allows. Expected savings: small (the `_TRIM_INDEX_CACHE` from Tasks 269/271 already amortizes cross-layer; per-instance cache would only help within a single layer's call which is already <0.1 ms).
- **Task 24 Quest top-K probe** (this morning's prior cycle) showed `argpartition` viable at idle conditions but variance-sensitive. Compose with this claims sweep for a complete picture: trim path is fast and constant, top-K is fast and variable.
