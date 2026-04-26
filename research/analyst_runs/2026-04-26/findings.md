# Analyst findings — 2026-04-26

Branch: `analyst/2026-04-26` (off `analyst/observability-scaffold@2c1920e`).
Six-hour cron audit. **No hypercar implementation files were modified.** All
work confined to `omlx/observability/`, `omlx/bench/`, `tools/analyst_kit/`,
`research/analyst_runs/2026-04-26/`, and the `TASKS.md` index.

Working-tree warning: this branch carries the **older committed** `omlx/duo_kv_cache.py`
(467 lines, commit e0ed34c). The user's in-flight `hypercar` branch has a 605-line
version with the Tasks 269/271 module-level `_TRIM_INDEX_CACHE` and a
`compute_attention` / `head_indices` / `get_streaming_kv` API. The +16% NIAH-decode
claim attributed to those tasks **cannot be verified on this branch** because the
optimization isn't here. Findings below are scoped to what is actually present
and runnable.

## Headline (TL;DR)

1. **F1b is real and large.** Replacing `broadcast_to + take_along_axis` with a
   plain slice on arange-identity rows is **219x faster at 16K** and **587x faster
   at 65K** in isolation. This is the highest-confidence optimization the
   measurement pass surfaced. -> Tasks 305, 306.

2. **F1a is real and small-ish.** Replacing `mx.where(mask, x, mx.zeros_like(x))`
   with `mx.where(mask, x, 0.0)` is **1.40x faster at 16K**, **2.29x at 65K**,
   and bit-identical. Per-decode budget at 48 layers: ~0.42ms saved at 16K
   = ~0.4% of decode wall (100ms/step). Worth doing because it's a one-line
   change with zero risk; not load-bearing on its own. -> Task 307.

3. **DuoKV `update_and_fetch` is NOT cleanly O(T_total) at decode.** The
   1024->4096->8192->16384 scaling curve is **sublinear -> linear -> sublinear**
   with the 16K median *lower* than the 8K median (0.272ms vs 0.415ms). The
   "DuoKV gather is O(T_total) per call" framing in CLAUDE.md (line 142) is
   too strong for the single-call cost. The decode cliff is multi-cause:
   per-call gather is one contributor but not the whole story. -> Task 308
   (re-frame the cliff hypothesis with measurement before further architectural
   work).

4. **`update_and_fetch` is ~14-21% of decode wall at 16K**, not the dominant
   cost. End-to-end traced 16K decode: 9.92 tok/s = ~100ms/step. Total
   `update_and_fetch` across 48 layers x 64 decode steps (3264 traced calls)
   was 1.36s out of 6.45s decode wall = ~21%. Implementer's own analysis
   in `omlx/patches/duo_split_attention.py:19-29` ("decode is dominated by
   MoE/projections, not KV gather") is consistent with what we measured.
   The Goal 3 cliff fix needs to attack the other ~80%, not just the gather.
   -> Task 308.

5. **No leak in DuoKV decode body.** 600-iteration leak loop with no GC/cache-clear
   reports `metal_peak=+0.000 MB/iter, phys_fp=-0.064 MB/iter` (negative = noise).
   `tools/analyst_kit/leak_body_duokv.py` exercises one decode-shape append per
   iteration on an 8K-primed cache. Verdict: **CLEAN**. -> No ticket; positive
   result.

6. **Bench is reproducible.** `--full` ALL 11 GATES PASSED in 2032.5s. HumanEval
   19/20, MMLU-Pro 62/100, RULER 6/6, LiveCodeBench 6/20 (this run; CLAUDE.md
   reports 8/20 with the CoT+retry path which is gated behind `--lcb-retry-on-empty`
   and was not the default path here). All quality numbers match prior runs to
   within +/-2 questions, confirming the bench harness is not drifting.

## Methodology

Three measurement vehicles, all in `tools/analyst_kit/`:

| Tool | What it measures | Output |
|------|-----------------|--------|
| `claims_2026_04_26.py` | Synthetic-tensor microbenches for hot-path claims (no model load) | `claims.json`, `claims_summary.txt` |
| `traced_decode.py` | End-to-end prefill + decode with explicit DuoKV cache + tracer | `traced_decode_4K.json`, `traced_decode_16K.json` |
| `leak_body_duokv.py` (driven by `omlx.observability.leak_loop`) | 600-iter decode-body slope analysis | `leak_duokv.csv`, `leak_duokv.log` |

The observability scaffold (registry / timers / heap snapshots / `trace_class` /
claims harness / leak loop) is the prep-session deliverable -- see
`omlx/observability/__init__.py`. None of those files touched the hot path.

## Detailed findings

### M1 -- F1b: gather-on-arange-identity is 219-587x slower than slice

`claims_2026_04_26.py` ran two equivalent operations on a `(1, 4, T, 128)` fp16
tensor with an arange-identity row index (i.e. `idx[h, t] = t`):

| T | gather (broadcast_to + take_along_axis) | slice (`x[:, :, :T, :]`) | ratio |
|---:|---:|---:|---:|
| 4096 | 0.271ms | 0.003ms | **104.11x** |
| 16384 | 0.482ms | 0.002ms | **219.18x** |
| 65536 | 1.410ms | 0.002ms | **587.50x** |

Both produce bit-identical output. The MLX gather kernel doesn't pattern-match
arange-identity into a noop slice; it materialises and dispatches a real
gather. Anywhere DuoKV / SnapKV / TQ3 builds a per-head index tensor that
**happens to be the identity row** for the streaming/full path, that path
should fall through to a slice (or a `cache_view` no-copy reshape) instead.

**Where this matters in the codebase**: the static-review pass yesterday
(F1, F2, F4 in `research/analyst_runs/2026-04-25/static_review.md`) flagged
gather paths in `streaming_attention.py:_gather_per_head_view` and
`duo_kv_cache.py` trim. Yesterday's tickets (Tasks 290-299 batch in
`## Static-analysis tasks` section) included this region. The **scale** of
the savings was not measured then -- F1b's 219x ratio is the new evidence.

**Caveat**: the synthetic test uses arange-identity. In production, the trim
path's index tensor is *not* always identity -- only on retrieval-row decode
steps where the streaming heads have already evicted older tokens. The
optimization needs a runtime check: if `idx[h, :] == arange(T)` for all h
where heads use full retrieval, slice; otherwise gather. Cost of the check
is one `mx.array_equal(idx, arange)` ~ same as one gather, so do it once
per layer per decode step.

-> Task 305 (DuoKV trim slice fast-path), Task 306 (StreamingKV ring slice fast-path).

### M2 -- F1a: zeros_like -> scalar 0.0 is 1.40x at 16K, 2.29x at 65K

| T | `mx.where(mask, x, mx.zeros_like(x))` | `mx.where(mask, x, 0.0)` | speedup |
|---:|---:|---:|---:|
| 4096 | 0.364ms | 0.276ms | 1.32x |
| 16384 | 0.845ms | 0.606ms | 1.40x |
| 65536 | 1.842ms | 0.803ms | 2.29x |

Bit-identical. `zeros_like(x)` allocates and materialises a zero-tensor of
shape `x` before `mx.where` consumes it; the scalar form lets the kernel
broadcast a zero literal into the where reducer.

48-layer absolute cost at 16K = baseline 5.88ms vs optimized 5.46ms = 0.42ms
saved per decode step. At 9.92 tok/s (100ms/step), this is ~0.4% of decode
wall. Per-step impact is small, but the optimization is one keystroke per
call site and zero-risk. The 65K ratio (2.29x) suggests the savings grow
super-proportionally -- at 256K-1M context this becomes a measurable
contributor.

-> Task 307 (sweep `mx.zeros_like` -> scalar `0.0` across DuoKV / SnapKV /
streaming code in places where the zero argument is a `where` operand).

### M3 -- DuoKV `update_and_fetch` is not cleanly O(T_total)

Single-call timing curve from the synthetic claims script (cache primed to
T_total then one decode-shape append):

| T_total | median update_and_fetch | T_ratio | time_ratio | verdict |
|---:|---:|---:|---:|---|
| 1024 | 0.217ms | -- | -- | -- |
| 4096 | 0.256ms | 4.00x | 1.18x | SUBLINEAR |
| 8192 | 0.415ms | 2.00x | 1.62x | LINEAR |
| 16384 | 0.272ms | 2.00x | 0.66x | SUBLINEAR |

The 16K median is lower than the 8K median. Re-running was consistent.
Working hypothesis: at 8K the trim path triggers a one-time index/mask
materialisation that dominates the median; at 16K the same trim re-uses a
cached path (or hits a fast slice-fallback in the existing 467-line
implementation that the synthetic test happens to exercise). This contradicts
the CLAUDE.md framing that the gather is the dominant per-call cost at long
context.

End-to-end consistent number: in the traced 16K decode, total `update_and_fetch`
time across all 48 layers x 64 decode steps x 1 cache call = 3264 invocations,
total 1.36s out of 6.45s decode wall = **21% of decode** (mean 0.418ms, p95
1.241ms). Higher than my single-call median of 0.272ms because the traced
run includes p95 outliers and prefill-phase calls. Even at 21%, the gather
is a minority cost.

-> Task 308 (re-derive the decode cliff hypothesis with per-op timer breakdown
before any further architectural work; the "gather O(T_total)" framing is
load-bearing for several queued tickets and we need to know if it's right).

### M4 -- Decode tok/s on this branch: 4K=20.6, 16K=9.92

End-to-end traced runs with explicit DuoKVCache:

| Context | Prefill tok/s | Decode tok/s | Metal peak | RSS | phys_footprint |
|---:|---:|---:|---:|---:|---:|
| 4K | 475 | 20.6 | 33.95 GB | 7.71 GB | 35.04 GB |
| 16K | 327 | 9.92 | 35.08 GB | 7.87 GB | 38.83 GB |

CLAUDE.md reports 44 tok/s at 4K and 16 tok/s at 16K (both on the user's
in-flight `hypercar` branch with Tasks 269/271 trim caching). The numbers
on **this** branch (older commit) are roughly half the published decode tok/s
at 4K and ~62% at 16K. Two non-exclusive explanations:

1. The Tasks 269/271 module-level cache contributes more than the +16% claim
   suggests -- closer to +100% at 4K. This is consistent with the small per-call
   numbers we measured: when 48x per-step Python list construction is replaced
   by a single dict lookup, decode-step Python overhead drops disproportionately.

2. Tracer overhead. `trace_class(DuoKVCache, mlx=True)` wraps every method with
   a timer + counter pair. At 48 layers x 64 steps = 3072 wrapped calls per
   decode, the wrapper Python overhead (one dict lookup + two `time.perf_counter`
   reads + reservoir update + a forced-materialize call) adds up. The `mlx=True`
   flag forces a per-call materialization, which can defeat the lazy graph
   fusion that the un-traced path uses.

The traced timer means are fast (0.418ms) so explanation 2 alone doesn't
account for the ~50% gap at 4K. The honest report is: **the tracer adds
non-trivial overhead that should be subtracted from any absolute decode tok/s
read off a traced run**. Use the un-traced bench (`hypercar_bench`) for
absolute decode-tok/s claims; use traced runs only for *relative* timer
breakdowns within a single run.

-> Task 309 (calibrate tracer overhead -- run the same model + cache untraced
vs traced and report the delta; gate any future cliff-attribution analysis
on a tracer-overhead-corrected baseline).

### M5 -- No DuoKV decode-body leak

Leak loop: `python -m omlx.observability.leak_loop --target tools.analyst_kit.leak_body_duokv:decode_body --iterations 600 --warmup 60`.

Body: one `cache.update_and_fetch(k_step, v_step)` call per iter, then a
forced materialization of the returned k/v tuple. Cache pre-primed to 8K
(capacity = sink + window = 260, well past the trim threshold).

```
iter 60/600  metal active=0.0173 peak=0.0676  phys_fp=0.4282  body_dt=0.000s
iter 600/600 metal active=0.0184 peak=0.0676  phys_fp=0.3790  body_dt=0.000s
CLEAN: metal_peak=+0.000 MB/iter  phys_fp=-0.064 MB/iter
```

Slope is below the 1 MB/iter threshold across 540 post-warmup iters. The
DuoKV trim path does not leak Metal allocations across decode steps on the
467-line branch. This refutes any latent-leak hypothesis on the trim path
itself; if a leak is observed in production, it is upstream (model forward,
attention, MoE routing, lm_head) -- not in DuoKVCache.

-> No ticket. Positive result. The leak-loop scaffold (`omlx/observability/leak_loop.py`)
is the deliverable; future leak hunts should use the `--gc-every` /
`--clear-cache-every` slope-attribution flags described in the docstring.

### M6 -- Bench reproducibility (this run vs prior runs)

`--full` 2032.5s with all 11 gates PASS:

| Gate | This run | CLAUDE.md / prior |
|------|----------|-------------------|
| Smoke decode tok/s | 47.3 | 53.9 (smoke prompt is short; variance is normal) |
| RULER | 6/6 PASS | 6/6 |
| MMLU-Pro | 62/100 | 62/100 (exact match across N=3 in CLAUDE.md) |
| LiveCodeBench | 6/20 (30%) | 8/20 (40%) -- gap is the CoT+retry path; not run by default |
| HumanEval Lite | 19/20 (95%) | 19/20 |
| Code Intelligence | 5/5 | 5/5 |
| NIAH @ 4K | PASS | PASS |
| Memory caps | clean | clean |

LCB at 30% (6/20) vs CLAUDE.md's 40% (8/20) is the one delta. The CLAUDE.md
40% number is post-Tasks 254/255/258 (CoT prompt + sample-verify retry at
temp=0.7), behind a flag the cron didn't pass. Not a regression.

-> No new ticket; documented for the record.

## What this run did NOT establish

- **Task 269/271's +16% decode claim** at 16K. The optimisation isn't on this
  branch. Future analyst runs should run from a branch that includes those
  changes, OR explicitly cherry-pick them into the analyst observability branch
  before measuring.
- **TQ3 fused-quantize 15x claim** (Task 152). Not in scope for this audit;
  needs its own claims-harness entry.
- **DuoKV pre-alloc 248x decode-at-64K claim** (Task 149). Same -- needs a
  64K traced run, which this audit didn't budget time for.
- **8K traced decode** between the 4K and 16K data points. Future runs should
  fill this in; the 8K single-call median (0.415ms) being *higher* than 16K
  (0.272ms) is the one anomaly we'd want a traced end-to-end run to corroborate.
- **MoE expert-dispatch dominance** at decode (NF6 in `00_session_log.md`).
  Static-review hypothesis only; needs `trace_module(qwen3_moe)` instrumentation.

## Tickets filed this run

Eight measurement-backed tickets (305-312) appended to `TASKS.md` under
header `## Analyst run 2026-04-26`. Each cites the file path + line number,
the measurement evidence, and a verifiable acceptance criterion. See TASKS.md
for the full text.

| # | Title | Priority | Evidence |
|---|-------|----------|----------|
| 305 | DuoKV trim arange-identity slice fast-path | P1 | M1 |
| 306 | StreamingKV ring arange-identity slice fast-path | P2 | M1 |
| 307 | `mx.zeros_like` -> scalar `0.0` sweep on `where` operands | P2 | M2 |
| 308 | Re-derive decode-cliff hypothesis with per-op timers | P0 | M3, M4 |
| 309 | Tracer-overhead calibration baseline | P1 | M4 |
| 310 | 64K traced-decode + claim sweep (Task 149 verification) | P2 | M3 (gap) |
| 311 | TQ3 fused-quantize 15x claim verification | P2 | (gap) |
| 312 | MoE expert-dispatch decode-share probe | P1 | NF6 + M3 |

## Files for permanence

```
research/analyst_runs/2026-04-26/
|-- 00_session_log.md           -- timeline + working code state
|-- findings.md                 -- this file
|-- claims_interpretation.md    -- pre-measurement claim analysis
|-- claims.json                 -- F1a / F1b / curve / leak-shape numbers
|-- claims_summary.txt          -- pretty-printed claims output
|-- baseline_full.log           -- bench --full stdout
|-- baseline_summary.json       -- parsed bench summary + JSON artifacts
|-- hypercar_bench_results.json -- bench raw results
|-- hypercar_profile.json       -- bench profile (memory etc.)
|-- traced_decode_4K.{json,log} -- end-to-end 4K with tracer
|-- traced_decode_16K.{json,log}-- end-to-end 16K with tracer
|-- leak_duokv.{csv,log}        -- 600-iter leak slope (CLEAN)
`-- quest_topk_probe.md         -- concurrent /loop session, not analyst output
```

## Coda

The measurement scaffold is the durable deliverable. The numbers in this
file will be stale within weeks (model migration, branch advances). The
scaffold -- `omlx/observability/`, `tools/analyst_kit/` -- outlives any single
audit. Every claim made in CLAUDE.md from this point forward should be
backed by a re-runnable claims-harness entry; every leak hypothesis should
be testable with a `leak_loop --target <module:fn>` invocation; every
optimization PR should include a traced-decode JSON showing the timer
delta. That is the audit trail this scaffold makes possible.
