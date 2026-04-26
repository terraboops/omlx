# Analyst session log — 2026-04-26

Cron-fired at 00:00 PDT. Branch: `analyst/2026-04-26` based off
`analyst/observability-scaffold` (commit 2c1920e). User's in-flight
work on `hypercar` stashed to preserve their state.

## Working code state

This branch carries the **older committed** versions of the hot-path
files, NOT the user's in-flight modifications:

- `omlx/duo_kv_cache.py` is 467 lines (the latest commit `e0ed34c`).
  The user's in-flight version was 605 lines with `_TRIM_INDEX_CACHE`
  module-level cache (Tasks 269/271) and a `compute_attention` /
  `head_indices` / `get_streaming_kv` API for split-attention.
  Those are NOT on this branch. Implication: I cannot verify the
  +16% claim from Tasks 269/271 (the optimization being measured
  isn't even on the branch I'm running).

- `omlx/turboquant_kv.py` 2171 lines, includes the fused quantize
  kernel and the WHT codec. Tasks 149 (pre-alloc slab), 142 (gather
  trim), 152 (fused quantize), 168 (skip_rerope) are all present.

- `omlx/patches/snapkv.py` 1496 lines, full eviction stack present.

## Timeline

| Time | Event |
|------|-------|
| 00:31 | Kicked off `--full` baseline bench in background |
| 00:31-00:40 | Static review of streaming_attention.py, snapkv.py select fns, qwen3_moe.py, switch_layers.py, mlx-lm base.py |
| 00:40 | RULER 14/15 → MMLU-Pro started |
| ~01:00 | RULER complete (estimated) |
| ~01:15 | MMLU-Pro complete (estimated) |
| ~01:40 | LCB complete (estimated) |
| ~01:55 | HumanEval complete + bench done (estimated) |
| ~02:00 | Begin post-bench claim sweep |
| ~02:30 | Begin decode-template tracing at 4K/8K/16K |
| ~03:30 | Begin leak loop |
| ~04:30 | Compile findings, write tickets |
| ~05:30 | Final commit, summary |

## Static-analysis carryover findings (from yesterday)

12 findings F1-F12 logged at `research/analyst_runs/2026-04-25/static_review.md`.
10 tickets (290-299) appended to TASKS.md by yesterday's prep session.

## New static-analysis findings this session (NOT yet ticketed)

- **NF1 (compute_freshness_scores):** `omlx/patches/snapkv.py:319-405`
  recomputes `i_idx` and `j_idx` mx.arrays per chunk iteration despite
  shape being constant across iterations. Hoist the band-mask
  construction outside the loop. Per-eviction event so impact is bounded.
  Priority P3.

- **NF2 (`_select_segmented`):** `omlx/patches/snapkv.py:713-755` does
  per-segment Python loop with mx.eval + .tolist() on each iteration.
  At 64K with segment=512, that's 128 GPU→CPU sync points per eviction.
  Could be vectorized: reshape pooled into (n_segments, segment_size),
  argpartition along axis=-1, single .tolist() at end. Priority P2.

- **NF3 (`_select_submodular`):** `omlx/patches/snapkv.py:758-837`
  inner-loop runs `.item()` per selection step (lines 821-822) which
  forces GPU→CPU sync per iteration. For seg_k=256, that's 256 sync
  points per segment. The score-update via mx.where + matmul could
  pipeline if indices were materialized in batches. Submodular path is
  opt-in (default False), so impact is conditional. Priority P2.

- **NF4 (TQ3 dense quantize kernel):** `omlx/turboquant_kv.py:218-279`
  has a per-row inner loop that recomputes `inv_norm * vec[j]` for
  every output coordinate `idx`. Hoisting `vec[j] * inv_norm` to a
  shared/local cache saves D = 128 multiplies per output coordinate
  (1/3 of inner-loop FLOPs). The kernel is shipped as the WHT path
  even though WHT structure could be exploited for O(D log D) instead
  of O(D²) — doc'd as a future kernel direction, not a near-term ticket.
  Priority P3 (hoist) / future kernel direction.

- **NF5 (mlx-lm `quantized_scaled_dot_product_attention`):**
  `.venv/lib/python3.13/site-packages/mlx_lm/models/base.py:64-105`
  is the dispatcher path for quantized caches. Allocates intermediate
  `scores` tensor (B, H_q, L, T) then softmax then 2nd quantized_matmul.
  Three big tensor allocations per call. At 16K decode with H_q=32
  that's ~1 MB scores per call × 48 layers = 48 MB transient per
  decode step. The implementer's `_quant_bits` workaround in
  `DuoKVCache.__init__` (line 189-191) avoids this dispatcher because
  the dispatcher routes on `hasattr(cache, "bits")`. Documented; not
  a ticket on its own.

- **NF6 (Qwen3 MoE routing):** `mlx_lm/models/qwen3_moe.py:110-139`.
  `Qwen3MoeSparseMoeBlock` uses `mx.argpartition` (top-k) → `mx.take_along_axis` →
  `SwitchGLU` (which uses `mx.gather_qmm` from
  `mlx_lm/models/switch_layers.py:75-85`). 8 experts dispatch via a
  SINGLE gather_qmm call — not 8 separate dispatches. So MoE routing
  contributes ~3 dispatches per layer (gate softmax + argpartition + gather_qmm),
  not 8. The "MoE expert dispatch dominates decode" claim from
  `omlx/patches/duo_split_attention.py:24-25` likely refers to compute
  cost (8 experts × MLP work per token) rather than dispatch count.
  This reframing is worth confirming with timer measurements.
  Priority P2 (after measurement).

## Plan for post-bench

1. **Run claims sweep** (`tools/analyst_kit/claims_2026_04_26.py`):
   tests F1a (zeros_like vs scalar), F1b (broadcast_to+take_along_axis
   vs slice), update_and_fetch curve, StreamingKV ring overhead,
   F1 48-layer absolute cost.

2. **Run decode_template** at 4K/8K/16K with full tracer
   instrumentation. Compare timer breakdowns. Validate or refute
   "DuoKV gather O(N)" claim.

3. **Run leak_loop** with `tools/analyst_kit/leak_body_duokv:decode_body`
   for 600 iterations (~10-15 min). Quick variant: same body with
   `--gc-every 60 --clear-cache-every 60` to attribute any climbing.

4. **Cross-reference** measurements against my static findings F1-F12
   and the 10 tickets already filed. Where measurements support the
   ticket: keep. Where they refute: file a follow-up ticket noting
   the negative result.

5. **Write findings.md** + append measured-evidence tickets to
   TASKS.md.
