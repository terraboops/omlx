# Proposed TASKS.md entries from Run 27

These are drafted by the analyst loop but NOT inserted into TASKS.md
to avoid sweeping in the user's uncommitted Tasks 12-15 WIP.
Engineers: review and merge in-band along with your pending TASKS.md edits.

Highest existing TASKS.md task number at time of drafting: 15
(Tasks 12-15 are the user's research-pass-2 entries; Tasks 7-11 are
the analyst's previous Run-23 entries, all still blocking.)

---

### 16. Identify root cause of per-task bimodal timing in Phase 3 NIAH and RULER 16K keys=5
- **Goal**: 3 (decode speed reliability) and 4 (prefill speed reliability)
- **Derived from**: Hypercar benchmark runs 23-27 (2026-04-13),
  bench/snapshots/run23..run27_*/ — five consecutive runs on
  SHA b49d2fc..49114c7 reveal a clean bimodal distribution for two
  specific tasks:
  - Phase 3 NIAH (4K + 16K combined): values 68, 69, 208, 232, 243s
    in 5 runs → fast cluster mean 68.5s, slow cluster mean 227.7s,
    inter-cluster gap 139s, slow/fast ratio 3.32x.
  - RULER 16K keys=5 task: values 61, 85, 221, 236, 259s in 5 runs →
    fast cluster mean 73s, slow cluster mean 238.7s, inter-cluster
    gap 136s, slow/fast ratio 3.27x.
  Both tasks show clean clusters with within-cluster std (12-15s) much
  smaller than inter-cluster gap (~140s). The two clusters are stable
  across runs — this is a real distribution, not noise. The fast/slow
  draw is INDEPENDENT per task (Run 23 was slow-NIAH + fast-k5;
  Run 24 was fast-NIAH + slow-k5).
- **Change**: This is fundamentally an INVESTIGATION task, not a fix.
  Concrete deliverable: a writeup at `docs/bimodal_timing_root_cause.md`
  that identifies which Metal/MLX subsystem is responsible. To get there:
  - Add `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1` to the bench
    subprocess env and capture Metal command buffer dispatch counts and
    kernel compile timestamps from the OS log.
  - Use `xcrun xctrace record --template 'Metal System Trace'` to capture
    one fast-mode and one slow-mode run side by side; diff the trace.
  - Hypothesis to test FIRST: is the bimodality from Metal kernel cache
    cold/warm state? If yes, prewarming the kernel cache before the
    timing measurement should eliminate the slow mode. Add a `warmup_pass`
    flag to phase3_niah and phase3b_ruler that runs the same task code
    once for warmup before the timed pass.
  - Hypothesis to test SECOND: is it page-fault cost on the first
    allocation of a particular tensor shape under unified memory pressure?
    If yes, the bimodality should disappear when running with reduced
    background memory pressure (e.g., after a fresh reboot with no other
    apps).
- **Verify**: After identifying the root cause, the writeup MUST include:
  (a) a reproduction recipe that can force fast-mode 3 times in 3 runs
  on the same SHA, (b) a reproduction recipe that can force slow-mode
  3 times in 3 runs, (c) one-line root cause statement that references
  a specific Metal/MLX subsystem.
- **Effort**: M-L (investigation depth uncertain; may decompose into
  follow-ups based on what the trace reveals)

---

### 17. Add multi-run statistical aggregation to bench/scoring.py
- **Goal**: 3, 4 (single-run hypercar_bench timings are not fit for
  Goal 3/4 regression detection at observed variance levels)
- **Derived from**: Hypercar benchmark runs 23-27 (2026-04-13). With
  N=5 the runtime CV is 21% (range 732-1303s, 78% of the mean span).
  To detect a real 10% performance regression with this variance,
  N >= 16 is needed; for 5%, N >= 64. Current BENCHMARKS.md records
  per-run prose entries which do not surface medians, percentiles, or
  per-phase distributions. The /tmp/hypercar_bench_results.json file
  is overwritten each run, so historical trend reconstruction requires
  walking bench/snapshots/runNN_*/results.json by hand.
- **Change**:
  - Add `omlx/bench/aggregate.py` that walks `bench/snapshots/run*/results.json`,
    groups by SHA + cache_mode + model_id, and emits a JSON file at
    `bench/snapshots/aggregate/<sha>_<mode>_<model>.json` containing for
    each phase: `[N, mean, std, median, p10, p50, p90, p99, min, max]`.
    Re-runnable; idempotent; updates incrementally as new snapshots land.
  - Add `omlx/bench/aggregate.py --report <sha>` that prints a
    pretty-printed table comparing the aggregate stats for two SHAs
    (current vs baseline) and flags any phase where the median delta
    exceeds 2 sigma of the combined std. This is the actual regression
    detector.
  - Hook the aggregate run into the analyst cron prompt's Step 9 final
    report so each fire shows N, median delta, and significance flags
    instead of single-sample numbers.
- **Verify**:
  `.venv/bin/python -m omlx.bench.aggregate` reads all existing
  bench/snapshots/run*/ directories, writes aggregate JSON files, and
  exits 0. Then `.venv/bin/python -m omlx.bench.aggregate --report HEAD`
  prints a per-phase median table and flags any 2-sigma deviations as
  REGRESSION/IMPROVEMENT. The script itself should fit in <300 lines
  and have no dependencies beyond stdlib + json.
- **Effort**: M
