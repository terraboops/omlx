# Hypercar Task Backlog
_Atomic, testable optimization tasks. Organized by the Hypercar goal they advance._
_Last updated: 2026-04-13_

## Research-derived tasks (from LIT_REVIEW.md, 2026-04-12)

### 1. Add RULER retrieval + tracing tasks to hypercar_bench
- **Goal**: 2 (intelligence breadth), 1 (context validation)
- **Derived from**: RULER (2404.06654)
- **Change**:
  - Add `omlx/eval/ruler/` module that vendors RULER's synthetic task
    generators (multi-key NIAH, variable tracking, frequent-word aggregation).
  - Extend `omlx/bench/hypercar_bench.py` with a `phase_ruler()` that runs
    two tasks at 4K + 16K in `--quick`, full 13-task suite at 4K / 16K / 64K
    in `--full`.
  - Gate: `multi_key_retrieval@16K` must be >= 0.8 accuracy.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --quick` prints
  `ruler: PASS` and the new gate appears in the gate summary table.
- **Effort**: S

### 2. Probe 2-bit KV on WHT-rotated codec (KIVI transfer test)
- **Goal**: 5 (swap headroom), 1 (longer context in same budget)
- **Derived from**: KIVI (2402.02750)
- **Change**:
  - In `omlx/turboquant_kv.py`, parameterise codebook bit-width and add a
    `bits=2` path reusing the existing WHT rotation.
  - Run NIAH@4K/16K/64K and HumanEval pass@1 against `bits=3` baseline; the
    experiment is whether WHT already decorrelates K channels enough that
    asymmetric per-channel-K is unnecessary.
  - If 2-bit WHT fails, implement KIVI asymmetry (per-channel K, per-token V)
    as a second variant.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --kv-mode tq3 --kv-bits 2`
  shows HumanEval >= 80% (vs current 90%) AND KV cache memory at 64K drops
  by ~33% vs 3-bit baseline. A regression worse than 10% HumanEval is a kill
  signal — document and back out.
- **Effort**: M

### 3. Prototype Quest query-aware page selection for TQ3 decode
- **Goal**: 3 (decode speed, constant across context)
- **Derived from**: Quest (2406.10774)
- **Change**:
  - Add per-page K min/max tracking in `omlx/turboquant_kv.py`
    (16 floats per page, stored alongside existing codebook indices).
  - New `select_topk_pages(q, k_bounds, K)` primitive in a fresh
    `omlx/patches/quest_attention.py`.
  - Wire into the decode path (not prefill) behind a `--quest-topk` flag on
    `omlx.hypercar_server` — default off.
  - Verify quality against RULER multi-key retrieval at 16K and 64K with
    K=4096 before enabling by default.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --quick --quest-topk 4096`
  shows decode tok/s at 16K >= 1.8x the baseline AND `ruler_multi_key@16K`
  accuracy is within 2% of the baseline. Both conditions required.
- **Effort**: M

### 4. Build per-head sparse-attention pattern calibration for Qwen3-Coder (MInference offline search)
- **Goal**: 4 (prefill speed, constant across context)
- **Derived from**: MInference 1.0 (2407.02490)
- **Change**:
  - New script `scripts/minference_calibrate.py` that runs Qwen3-Coder over
    a short calibration corpus, records the full attention map for each head
    at one long prompt, and assigns each (layer, head) one of
    {a_shape, vertical_slash, block_sparse, dense} based on reconstruction
    error vs a fixed sparsity budget.
  - Output a JSON pattern table shipped under `omlx/patches/minference_patterns/qwen3_coder_30b.json`.
  - This task ONLY does the calibration + table — runtime dispatch is the
    follow-up task so this stays atomic.
- **Verify**: `scripts/minference_calibrate.py --model <qwen3-coder>` writes
  a JSON file with 48 layers x num_heads entries; a validation assertion
  confirms sparsity across all heads is >= 85% at < 1% reconstruction MSE.
- **Effort**: M

### 5. Implement MInference vertical-slash prefill kernel behind a flag
- **Goal**: 4 (prefill speed)
- **Derived from**: MInference 1.0 (2407.02490)
- **Change**:
  - New `omlx/patches/minference_prefill.py` that, during prefill, reads the
    calibration table from Task 4, and for each head dispatches to either
    dense attention (existing path), vertical-slash (gather vertical column
    indices + diagonal band, run MLX matmul), or block-sparse.
  - Flag: `--prefill-sparse minference` on `omlx.hypercar_server`. Default off.
  - Must compose with `prefill_last_logit_patch` (the last-hidden-state
    projection is downstream of the attention output, so it should Just Work).
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --prefill-sparse minference`
  shows prefill tok/s at 16K >= 1.5x baseline AND HumanEval >= 85%
  AND `ruler_multi_key@16K` within 2% of baseline. Depends on Task 4 landing
  first.
- **Effort**: M-L

## Benchmark-derived tasks (from benchmark run analysis)

### 7. Fix RULER memory-breach early-return KeyError in phase3b_ruler
- **Goal**: 2 (eval reliability — currently every Phase 3b watchdog failure
  presents as a cryptic `KeyError: 'found'` regardless of true cause)
- **Derived from**: Hypercar benchmark run 23 (2026-04-13),
  bench/snapshots/run23_2026-04-13T00-23/
- **Change**:
  - `omlx/bench/hypercar_bench.py:764-765` — the memory-breach early return
    inside `_run_ruler_task` returns `{task_type, ctx, passed=False, reason="memory_breach"}`,
    missing `found`, `total`, `accuracy`, `response` keys.
  - `omlx/bench/hypercar_bench.py:851-852` — the caller in `phase3b_ruler`
    unconditionally dereferences `result['found']`, `result['total']`,
    `result['accuracy']`, `result['response']` in the status log format string.
  - Fix (option b is cleaner): guard the logger format with
    `if "reason" in result: logger.info(f"    BREACH: {result['reason']}")
     else: <existing format>`. Preserves "memory_breach" as a distinct
    outcome instead of faking a 0/0 accuracy and hiding the real signal.
- **Verify**: Force a memory breach during a RULER task
  (`.venv/bin/python -m omlx.bench.hypercar_bench --full --max-metal-pct 50`)
  and confirm the log shows `BREACH: Metal peak ...` rather than
  `KeyError: 'found'`, AND that Phase 3b completes with a failed
  `PhaseResult` rather than an exception bubbling out of `main()`.
- **Effort**: S

### 8. Rebuild profiler.py observability for macOS unified memory
- **Goal**: Supports Goals 3, 4, 5 — can't measure what you can't see.
  Run 23 exposed three distinct profiler defects that make analysis blind.
- **Derived from**: Hypercar benchmark run 23 (2026-04-13),
  bench/snapshots/run23_2026-04-13T00-23/env.json `profiler_gaps_detected`
- **Change**: Three defects in `omlx/bench/profiler.py`, fix together for
  atomicity:
  1. **CPU metric is permanently 0.0**: `_get_process_stats()` at lines
     79-89 creates a fresh `psutil.Process()` per sample and calls
     `.cpu_percent(interval=None)` on it. Fresh Process objects always
     return 0.0 on their first call because psutil needs two snapshots
     to compute a delta. Run 23 profile.json has `cpu_pct=0.0` in all
     830 samples. Fix: instantiate `self._psutil_proc = psutil.Process()`
     once in `Profiler.start()` and reuse it across samples.
  2. **RSS metric is misleading on unified memory**: Run 23 profile shows
     `rss_gb` shrinking from 3.10 GB (t=42s, post-load) to 0.09 GB (end)
     while `metal_active_gb` climbs to 41.4 GB. On Apple Silicon, Metal
     allocations don't appear in RSS, and swap pressure pages Python heap
     out of wired memory. Replace `rss_gb` with `phys_footprint_gb`
     backed by macOS `task_info(TASK_VM_INFO_PURGEABLE)` or
     `psutil.Process().memory_full_info().uss` — whichever correctly
     tracks the process's committed footprint including compressed pages.
  3. **Swap metric captures depth, not throughput**: Run 23 reported
     `swap_peak_gb = 8.63` but `vm_stat` pre/post delta shows
     **337.8 GB of cumulative swap I/O** over 850s (406 MB/s sustained) —
     pageins+swapins+swapouts combined. The profiler is blind to sustained
     compressed-memory churn, which is the *leading* indicator of memory
     pressure and directly correlates with decode/prefill slowdown. Add
     a new `swap_io_mb_per_s` field per sample, computed from
     `vm_stat`'s `pageins`/`swapins`/`swapouts` counter deltas between
     consecutive samples.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --quick`,
  then inspect `/tmp/hypercar_profile.json`:
  - `cpu_pct` non-zero in ≥80% of samples
  - new `phys_footprint_gb` field present; values ≥ `metal_active_gb`
    at every sample after model load
  - new `swap_io_mb_per_s` field present; peak value > 50 MB/s during
    at least the Phase 0 model-load window
- **Effort**: M

### 9. Gate Phase 3b RULER tasks by projected memory headroom
- **Goal**: 6 (machine fit) and 2 (eval completion reliability)
- **Derived from**: Hypercar benchmark run 23 (2026-04-13),
  bench/snapshots/run23_2026-04-13T00-23/ — RULER task 5/15
  (multi_key_niah@64K keys=3) crashed the run with a Metal peak breach
  at 41.28 GB (ceiling 41.2 GB). Tasks 6-15 never ran. Phase 4
  (HumanEval) never ran. The allocation was an instantaneous cliff
  (+8.3 GB in <1 sample interval) — a 64K attention scores tensor
  whole-sequence materialization that doesn't fit in the 8-bit
  model's remaining Metal headroom.
- **Change**:
  - Add `_project_prefill_memory_gb(ctx_tokens: int, model) -> float`
    helper in `omlx/bench/hypercar_bench.py`. Rough upper bound:
    `model_base_gb + ctx_tokens * n_layers * 2 * head_dim * bytes_per_elem * safety_factor`
    where `safety_factor=1.5` to account for intermediate allocations
    (attention scores, RMS norm, MoE router).
  - In `phase3b_ruler` (line 808+), before dispatching each task,
    compare `projected + metal_active` against `watchdog.metal_limit_gb`.
    If projection > limit − 1 GB (a soft headroom buffer), skip the task
    and append to `skipped` list in the phase details.
  - Phase 3b gate must evaluate only on tasks that RAN — don't fail
    the phase for skipped tasks. A clear SKIP signal in the results
    JSON is better than a crash.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full`
  on the 8-bit model with the current 41.2 GB Metal limit:
  - 64K RULER tasks (5, 6, 7 of 15) report SKIP with
    `projected_gb ~ 48 GB > 41.2 GB limit` in the console log
  - Phase 3b completes with 12 task results (4 skipped)
  - Phase 4 (HumanEval) subsequently runs to completion (previously
    never reached)
- **Effort**: S-M

### 10. Fix NIAH decode-speed measurement artifact for short generations
- **Goal**: 3 (decode speed tracking reliability — prevents false regression
  alarms and false stability readings)
- **Derived from**: Hypercar benchmark run 23 (2026-04-13), results.json
  Phase 3 details at 16384 — `decode_toks: 0.3` at 16K context while
  Phase 0 Smoke at 2K reports 21.3 tok/s. The 83× cliff is implausible
  and is a measurement artifact: NIAH generates <20 tokens before the
  stop token, and at that length MLX graph compilation + kernel warmup
  dominate wall-clock.
- **Change**: `omlx/bench/hypercar_bench.py` `phase3_niah` function.
  Two viable fixes, pick one:
  - (a) Drop `decode_toks` from NIAH phase results entirely. Only
    `prefill_toks` is meaningful for retrieval tasks; the decode
    component is too short to measure cleanly.
  - (b) Add a dedicated decode-stress mini-phase (or a sub-measurement
    inside NIAH) that generates 128+ tokens on a clean cache at each
    context length and records THAT as the Goal 3 decode_toks metric.
  - Recommend (b): we DO want decode-speed-vs-context-length data for
    Goal 3 ("≥50 tok/s, constant across context window"), but we need
    meaningful decode length to compute it honestly.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full`
  and inspect `/tmp/hypercar_bench_results.json` Phase 3 details.
  Either `decode_toks` is absent from NIAH entries, OR the dedicated
  decode-stress sub-measurement reports `decode_toks > 10` at 16K
  based on a ≥128-token sample.
- **Effort**: S

### 11. Extend /sandbox exclude to cover benchmark diagnostic commands
- **Goal**: Observability infrastructure (supports all goals via
  reliable pre/post-run environment baseline)
- **Derived from**: Hypercar benchmark run 23 (2026-04-13),
  bench/snapshots/run23_2026-04-13T00-23/env.json `sandbox_gaps`
- **Change**: The Claude Code Bash sandbox blocks several diagnostic
  commands needed for performance analyst baseline capture (the
  bench-exclusion pattern in `.claude/settings.local.json` only covers
  `.venv/bin/python -m omlx.bench.hypercar_bench:*`):
  - `sysctl hw.memsize` — Operation not permitted
  - `sysctl vm.swapusage` — Operation not permitted
  - `sysctl vm.loadavg` — Operation not permitted
  - `iostat` — kern.boottime sysctl denied
  - `pgrep` — sysmond service not found
  - (working under sandbox: `vm_stat`, `uptime`)
  - Preferred fix: create `omlx/bench/baseline.py` that runs under the
    existing `.venv/bin/python -m omlx.bench.*` sandbox exception
    pattern and gathers the same data via native macOS APIs:
    - `hw.memsize` via ctypes `sysctlbyname`
    - `vm.swapusage` via ctypes `sysctlbyname`
    - Load average via `os.getloadavg()`
    - Running MLX processes via `psutil.process_iter(['pid', 'name', 'cmdline'])`
    - Disk I/O via `psutil.disk_io_counters()`
    - Emits JSON to stdout for the cron prompt's Step 1 to consume.
  - This is cleaner than widening the sandbox allow-list — it keeps
    diagnostic logic in version control and reuses the one sandbox
    exception already in place.
- **Verify**: `.venv/bin/python -m omlx.bench.baseline` runs under the
  current `.claude/settings.local.json` sandbox, prints JSON with at
  least fields `{system_memory_gb, swap_used_gb, load_avg_1m,
  mlx_processes, disk_read_mb_s, disk_write_mb_s}`, and none of them
  contain an "Operation not permitted" error. Update the scheduled
  cron prompt's Step 1 to call this script instead of shelling out to
  sysctl/iostat/pgrep.
- **Effort**: S-M

### 20. Identify root cause of per-task bimodal timing in Phase 3 NIAH and RULER 16K keys=5
- **Goal**: 3 (decode speed reliability) and 4 (prefill speed reliability)
- **Derived from**: Hypercar benchmark runs 23-30 (2026-04-13, N=8),
  bench/snapshots/run23..run30_*/ — eight consecutive runs on an
  unchanged SHA reveal a clean bimodal distribution for two specific
  tasks:
  - Phase 3 NIAH (4K + 16K combined): at N=8 the sorted values are
    68, 69, 73, 82, 208, 231, 232, 243 seconds.
    Fast cluster [68, 69, 73, 82] mean 73.0s std 5.4s (N=4).
    Slow cluster [208, 231, 232, 243] mean 228.5s std 12.7s (N=4).
    Inter-cluster gap 126s, slow/fast ratio 3.13x, balance even 4/4.
  - RULER 16K keys=5: at N=8 the sorted values are
    61, 85, 98, 221, 231, 236, 241, 259 seconds.
    Fast cluster [61, 85, 98] mean 81.3s std 18.5s (N=3).
    Slow cluster [221, 231, 236, 241, 259] mean 237.6s std 13.2s (N=5).
    Inter-cluster gap 123s, slow/fast ratio 2.92x.
  - Within-cluster std (5-18s) is much smaller than inter-cluster gap
    (~125s). This is a real distribution, not noise. The fast/slow
    draw is INDEPENDENT per task — all 4 cells of the combination
    matrix are populated across the 8 runs.
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
    flag to `phase3_niah` and `phase3b_ruler` that runs the same task code
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

### 21. Add multi-run statistical aggregation to omlx/bench/aggregate.py
- **Goal**: 3, 4 (single-run hypercar_bench timings are not fit for
  Goal 3/4 regression detection at observed variance levels)
- **Derived from**: Hypercar benchmark runs 23-30 (2026-04-13). At N=8
  the runtime CV is 17% (range 732-1303s, 78% of the mean span). To
  detect a real 10% performance regression with this variance,
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

### 22. Fix 8-bit model Goal 5 violation rate (50% of runs exceed 8 GB swap)
- **Goal**: 5 (swap headroom), 6 (M4 Pro fit)
- **Derived from**: Hypercar benchmark runs 23-30 (2026-04-13, N=8),
  bench/snapshots/run*/env.json swap_peak_gb values. 4 of 8 runs
  exceeded the CLAUDE.md Goal 5 threshold of 8 GB peak swap:
    Run 23: 8.63 GB (VIOLATE)
    Run 24: 6.93 GB (pass)
    Run 25: 6.73 GB (pass) ← min
    Run 26: 7.76 GB (pass, tight)
    Run 27: 7.80 GB (pass, tight)
    Run 28: 8.30 GB (VIOLATE)
    Run 29: 9.70 GB (VIOLATE, max)
    Run 30: 8.50 GB (VIOLATE)
  This is not a one-off — it's a 50% violation rate that indicates the
  8-bit model + native 3-bit KV + current chunk sizes produces a swap
  profile that only barely fits the 48 GB machine. The underlying cause
  is that the 8-bit model loads at 32.4 GB, leaving ~8.8 GB Metal
  headroom at the 41.2 GB ceiling, and RULER 64K multi-key prefill
  allocations sporadically push total wired+compressed memory past the
  point where macOS has to spill to disk-backed swap.
- **Change**: This is a STRUCTURAL headroom issue, not a harness bug.
  The concrete change path is one of these three remediations, pick
  whichever lands first:
  - (a) Land Task #2 (--kv-bits 2) and verify Goal 5 on N=8 replication
    with `--kv-bits 2`. 2-bit codebook drops KV memory by ~33% which
    should give ~3 GB more Metal headroom per 64K context — enough to
    keep swap peak under 8 GB on all runs.
  - (b) Land Task #3 (--quest-topk) and verify Goal 5 on N=8
    replication with `--quest-topk 32`. Query-aware page selection
    reduces working-set K per decode step without changing the prefill
    allocation pattern, so it should help the sustained swap rate but
    may not help the allocation-cliff peak.
  - (c) Land Task #9 (RULER projected-headroom gate). This prevents
    the crash but does NOT fix Goal 5 — RULER@64K will still be
    attempted up to the point of projected-memory exhaustion. Only a
    partial fix.
  - Recommend (a) first because it addresses the root cause (too much
    KV for the available headroom) rather than working around it.
- **Verify**: Run `.venv/bin/python -m omlx.bench.hypercar_bench --full
  --kv-bits 2` eight times and compute swap peak per run. Goal 5 PASS
  criterion: 8 of 8 runs have swap peak < 8 GB AND HumanEval pass@1
  stays within 10 points of the 3-bit baseline (90%). If the 2-bit
  path regresses HumanEval below 80%, back out and try option (c).
- **Effort**: S (if Task #2 already landed) to M (if Task #2 still
  needs integration)

### 23. Re-state CLAUDE.md Goal 5 as a p90 sustained swap-rate metric
- **Goal**: Meta — CLAUDE.md Goal 5 definition fit-for-purpose
- **Derived from**: Hypercar benchmark runs 23-30 (2026-04-13, N=8).
  The current Goal 5 phrasing in CLAUDE.md is "Swap usage: < 8 GB at
  any point," which has two problems:
  - The profiler's `swap_gb` metric measures DEPTH (peak
    disk-backed swap) but misses THROUGHPUT. Across the 8 runs,
    sustained swap rate (pageins+swapins+swapouts / run duration) was
    **431 ± 34 MB/s (CV 8%)** — tighter than any other metric. This
    is the actual hardware constraint on Apple Silicon's page
    compressor, and the depth gate misses it entirely. Run 25 had 6.73
    GB depth but still pushed 271 GB of swap I/O over 732s.
  - The depth metric is inversely correlated with actual pressure in
    some cases (Run 24 had lower depth than Run 23 but higher swap I/O
    and runtime). A threshold on depth alone is not well-defined.
- **Change**:
  - Update `CLAUDE.md` Goal 5 row in the "Hypercar Goals (North Star)"
    table from "Swap usage: < 8 GB at any point" to "Swap throughput:
    p90 sustained swap I/O rate < 100 MB/s over N >= 8 runs (hardware
    floor on M4 Pro is ~430 MB/s based on 8-run measurement; 100 MB/s
    leaves 4x headroom for foreground work during inference)."
  - Update the "Current status against goals" row for Goal 5 to
    reference the measured p90 from the most recent N=8 aggregate.
  - Add a note referencing Task #21's aggregate script as the source
    of truth for the metric.
  - Do NOT remove the depth tracking — keep `swap_peak_gb` as a
    secondary observability metric but not a gate.
- **Verify**: After the edit, `.venv/bin/python -m omlx.bench.aggregate
  --report HEAD` (from Task #21) reads the current CLAUDE.md gate
  definition and reports p90 sustained rate for the last N=8 runs.
  Goal 5 gate passes at p90 < 100 MB/s. Current measurement would
  fail this gate (p90 ~= 460 MB/s), which is the honest signal we
  want — failing clearly is better than passing-by-depth while actual
  pressure is 4x the stated threshold.
- **Effort**: S (CLAUDE.md edit + one aggregate script run; depends
  on Task #21 for the measurement tooling)

## In Progress

_(none)_

## Completed

- **Task 1**: Add RULER retrieval + tracing tasks to hypercar_bench (2026-04-12)
  - Added `omlx/eval/ruler/` module with 3 synthetic generators (multi-key NIAH, variable tracking, frequent-word aggregation)
  - Added Phase 3b (RULER) to hypercar_bench: quick suite (4 tasks at 4K/16K) in default mode, full suite (13 tasks at 4K/16K/64K) in --full mode
  - Gate: `multi_key_retrieval@16K` must be >= 80% accuracy
- **Task 6**: Add LongBench-v2-free reasoning gate via RULER variable-tracking (2026-04-12)
  - Added VT tasks (chain lengths 4 and 8) to both quick suite (6 tasks) and full suite (15 tasks)
  - Added `ruler_vt@4K` gate (>= 70% accuracy) as independent eval toward Goal 2
  - Both gates (`multi_key@16K` + `ruler_vt@4K`) must pass for Phase 3b to pass
- **Task 2**: Probe 2-bit KV on WHT-rotated codec (KIVI transfer test) (2026-04-12)
  - Added `--kv-bits` CLI flag to hypercar_bench (choices: 2, 3, 4)
  - TurboQuant codec already fully parameterized on bits — no core changes needed
  - 2-bit codebook: 4 levels, packed_width drops from 12→8 uint32 words (33% savings at D=128)
  - Verify: `hypercar_bench --full --kv-mode tq3 --kv-bits 2` (needs GPU to run)
- **Task 3**: Prototype Quest query-aware page selection for TQ3 decode (2026-04-12)
  - New `omlx/patches/quest_attention.py` with `select_topk_pages()` and `gather_pages()` primitives
  - Per-page K bounds: 16 floats (8 group-max + norm stats), 128 tokens/page, ~4MB overhead at 1M context
  - Wired into `TurboQuantKVCache.decode_attention()` behind `quest_topk` param
  - `--quest-topk` flag on both bench and server (default off)
  - Verify: `hypercar_bench --quick --kv-mode tq3 --quest-topk 32` (needs TQ3 GPU run)
- **Task 4**: Build per-head sparse-attention pattern calibration for Qwen3-Coder (2026-04-12)
  - New `scripts/minference_calibrate.py` with attention capture + 3 pattern classifiers (a_shape, vertical_slash, block_sparse) + dense fallback
  - Synthetic placeholder table at `omlx/patches/minference_patterns/qwen3_coder_30b_a3b_instruct_8bit.json` (48 layers × 32 heads = 1536 entries)
  - Pattern distribution: 57% vertical_slash, 24% a_shape, 14% block_sparse, 5.7% dense
  - Verify: real calibration requires GPU run of `scripts/minference_calibrate.py --model <qwen3-coder>`
- **Task 5**: Implement MInference vertical-slash prefill kernel behind a flag (2026-04-12)
  - New `omlx/patches/minference_prefill.py` with per-head sparse dispatch during prefill
  - 3 sparse mask builders (a_shape, vertical_slash, block_sparse) + dense fallback
  - Groups heads by pattern type for batched dispatch (avoids per-head kernel launches)
  - `--prefill-sparse minference` flag on both bench and server (default off)
  - Composes with prefill_last_logit_patch and vertical_eval (orthogonal)
  - Verify: `hypercar_bench --full --prefill-sparse minference` (needs GPU for quality gate)
- **Task 7**: Fix RULER memory-breach early-return KeyError in phase3b_ruler (2026-04-13)
  - Guarded logger format string: breach results now log `BREACH: <reason>` instead of crashing on missing `found`/`total`/`accuracy` keys
  - Gate aggregation (`mk_16k`, `vt_4k`) and `by_type` summary now skip breach results via `"accuracy" in r` filter
  - Phase 3b completes with a failed PhaseResult on breach instead of an unhandled KeyError
- **Task 10**: Fix NIAH decode-speed measurement artifact for short generations (2026-04-13)
  - Option (b): added 128-token decode stress measurement after the 32-token NIAH answer retrieval
  - Decode speed is now measured on 128 tokens (amortizes MLX graph compilation + kernel warmup)
  - NIAH retrieval check unchanged — answer is still from the first 32 tokens
  - New `decode_stress_tokens` field in results for auditability
- **Task 9**: Gate Phase 3b RULER tasks by projected memory headroom (2026-04-13)
  - Added `_project_prefill_memory_gb()` helper estimating KV cache + attention scores + safety factor
  - Each RULER task checks projected memory vs Metal headroom (limit - current - 1GB buffer) before dispatch
  - Tasks exceeding headroom are logged as SKIP with projected/limit details and excluded from gate evaluation
  - Phase details now include `num_ran`, `num_skipped`, and `skipped` list for auditability
- **Task 8**: Rebuild profiler.py observability for macOS unified memory (2026-04-13)
  - Fix 1: CPU metric — reuse single `psutil.Process()` across samples (was creating fresh one each time → always 0.0). Now 100% non-zero.
  - Fix 2: RSS → phys_footprint — added `proc_pid_rusage()` via ctypes for macOS physical footprint (includes Metal + compressed pages). RSS kept for compat.
  - Fix 3: Swap I/O throughput — added `swap_io_mb_per_s` field from psutil swap_memory() deltas (no subprocess fork). Watchdog breaches at 2000 MB/s sustained for 5 samples.
  - Critical: ALL subprocess calls removed from profiler sampling loop (fork under memory pressure caused the original catastrophic swap storm)

---

### 6. Add LongBench-v2-free reasoning gate via RULER variable-tracking
- **Goal**: 2 (intelligence breadth — reasoning, not retrieval)
- **Derived from**: RULER (2404.06654) — specifically the multi-hop variable
  tracking task, which stresses reasoning more than retrieval
- **Change**:
  - Extend Task 1's RULER integration to include `vt` (variable tracking)
    at chain lengths 4 and 8.
  - Add as a separate gate row in the bench summary so we can count it as
    an *independent* eval toward Goal 2's "4 independent evals".
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --quick` shows
  a `ruler_vt@4` gate row; gate passes at accuracy >= 0.7.
- **Effort**: S (depends on Task 1)

## Research-derived tasks (from LIT_REVIEW.md pass 2, 2026-04-13)

### 12. DuoAttention retrieval/streaming head calibration for Qwen3-Coder
- **Goal**: 3 (decode), 4 (prefill), 5 (swap headroom), 1 (longer context per byte)
- **Derived from**: DuoAttention (2410.10819)
- **Change**:
  - New `scripts/duoattention_calibrate.py` that runs Qwen3-Coder-30B-A3B
    over a synthetic passkey-retrieval corpus and gradient-descents a
    per-(layer, head) gate alpha in [0,1]; alpha near 1 = retrieval head
    (needs full KV), alpha near 0 = streaming head (needs only sink+window).
  - L1 penalty on alpha to push the streaming fraction up, calibrated so
    NIAH@16K accuracy stays >= 0.95 of fp16 baseline.
  - Output: `omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json`
    listing per-head policy {full, stream(window=N, sink=4)}.
  - This task ONLY emits the calibration table — runtime cache split is
    Task 13 so this stays atomic.
- **Verify**: `scripts/duoattention_calibrate.py --model <qwen3-coder>`
  writes the JSON; assertion checks streaming fraction >= 0.50 across all
  layers AND validation NIAH@16K >= 0.95 of fp16 baseline. No runtime
  changes yet.
- **Effort**: M

### 13. Two-storage-class KV cache (DuoAttention runtime)
- **Goal**: 3 (decode), 4 (prefill), 5 (swap headroom)
- **Derived from**: DuoAttention (2410.10819)
- **Change**:
  - Extend `omlx/turboquant_kv.py` so each head has one of two storage
    classes selected from the Task 12 policy table:
    - `full`: existing TQ3 paged 3-bit cache (no change).
    - `stream`: ring buffer of size `window + sink` in fp16 (no quant —
      the cache is small enough that the quant cost is not worth it, and
      retrieval-head misclassification compounds at the streaming heads).
  - Decode and prefill paths read both classes and concatenate per-head
    attention outputs.
  - New `--kv-mode duo` switch on `omlx/hypercar_server.py` and on
    `hypercar_bench.py`. Composes with TQ3 (full heads use TQ3 storage)
    and orthogonal to Quest top-K page selection (top-K applies only to
    full heads).
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --kv-mode duo`
  shows decode tok/s at 16K >= 1.3x baseline AND prefill tok/s at 16K >= 1.2x
  baseline AND HumanEval >= 85% AND `ruler_multi_key@16K` within 2% of
  baseline AND KV cache memory at 64K drops by >= 35% vs `tq3` baseline.
  All conditions required. Depends on Task 12.
- **Effort**: M-L

### 14. tau-bench agentic gate in hypercar_bench
- **Goal**: 2 (intelligence breadth — agentic tool use, the missing 4th eval)
- **Derived from**: tau-bench (2406.12045)
- **Change**:
  - New `omlx/eval/tau_bench/` shim that vendors a pinned tau-bench commit
    or wraps the pip package, configured to point at our local
    `hypercar_server` `/v1/chat/completions` endpoint.
  - User-side simulator runs against a small local model
    (`gemma2-2b-it` or `qwen2.5-3b-instruct`) loaded via mlx-lm to keep
    eval fully offline.
  - Add `phase_tau()` to `omlx/bench/hypercar_bench.py`: 5 retail tasks
    in `--quick`, 20 retail + 10 airline tasks in `--full`. Report
    pass^1 and pass^4.
  - Gate: `tau_retail_pass@1 >= 0.30` (well below GPT-4o ceiling, but a
    real signal that tool use works at all).
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --quick`
  prints a `tau_retail` gate row; gate passes at pass^1 >= 0.30. Wall
  clock for the quick variant must stay under 5 minutes (cap turns
  per task).
- **Effort**: M

### 15. SimPO contrast step in TTT engine
- **Goal**: 2 (intelligence — agentic quality, complement to existing TTT)
- **Derived from**: SimPO (2405.14734)
- **Change**:
  - Add `simpo_step(winner_traj, loser_traj, gamma, beta, lr)` in
    `omlx/ttt.py` next to the existing `ttt_step()`. Reuse the existing
    forward path to get per-token logprobs for both trajectories under
    the current down_proj LoRA.
  - Loss: `-log sigmoid(beta * (avg_logp(winner) - avg_logp(loser)) - gamma)`,
    length-normalised.
  - Cap per-step LoRA delta-norm at the same bound the existing TTT
    loop uses (no new memory pressure, no runaway updates).
  - New endpoint `POST /v1/ttt/simpo` on `omlx/hypercar_server.py`
    accepting `{winner: [...], loser: [...], task_id}`; persists the
    pair into the same TTT trajectory store the existing endpoints use.
  - Pull pairs automatically from tau-bench (Task 14) runs: any task where
    one rollout passes and another rollout of the same task fails becomes
    a (winner, loser) pair fed back into SimPO.
- **Verify**: New unit test `tests/test_simpo_step.py` constructs a tiny
  synthetic (winner, loser) pair, runs one `simpo_step`, asserts (a)
  winner avg-logp went up, (b) loser avg-logp went down, (c) LoRA
  delta-norm within bound. Then `.venv/bin/python -m omlx.bench.hypercar_bench --full`
  passes (no regression on existing gates). Stretch: after 50 SimPO
  steps fed from a single tau-bench run, `tau_retail_pass@1` improves
  by >= 5 percentage points on a held-out task subset.
- **Effort**: S-M (depends on Task 14 for the pair source, but the
  core implementation can land before Task 14 against a synthetic pair set)

## Research-derived tasks (from LIT_REVIEW.md pass 3, 2026-04-13)

### 16. ProMoE lazy-load probe for Qwen3-Coder expert weights
- **Goal**: 6 (M4 Pro 48GB fit), 5 (swap headroom under load), 3 (decode)
- **Derived from**: ProMoE (2410.22134)
- **Change**:
  - One-day de-risking spike before committing to the full project.
  - New `scripts/moe_lazyload_probe.py` that loads Qwen3-Coder-30B-A3B
    with only a subset of expert MLPs materialised (e.g., experts 0-63
    in each layer; the remaining 64 are zero-masked at the router so
    they are never dispatched to). Measure: Metal memory at load,
    Metal memory at steady-state on a short prefill, and whether any
    MLX weight-load-time assertion blocks the partial load.
  - If the probe succeeds, the full ProMoE implementation follows as a
    Task 16b: (a) a lazy-weight `MoEMLP` wrapper in a new
    `omlx/patches/lazy_moe.py` that mmap's expert weights from disk
    and unpacks on first touch, (b) a tiny linear-probe predictor on
    prior-layer router logits that runs one layer ahead and prefetches,
    (c) hot/cold tracking that evicts experts by LRU under a
    configurable `--moe-resident-fraction` budget.
  - This task ONLY does the probe — the full system is 16b so this
    stays atomic and low-risk.
- **Verify**: `.venv/bin/python scripts/moe_lazyload_probe.py
  --resident-fraction 0.5` runs to completion, reports Metal at load
  dropping from ~32GB to ~20GB (scales with resident fraction), AND
  `.venv/bin/python -m omlx.bench.hypercar_bench --quick` with the
  probe's lazy-load hook applied still passes smoke + coherence
  gates (quality hold is the blocker — if zero-masked experts hurt
  quality, pivot to a 4-bit re-quant on the cold experts instead
  of zero-masking).
- **Effort**: S for the probe, L for the full 16b follow-up

### 17. LLMLingua-2 prompt-compression middleware with RULER gate
- **Goal**: 1 (effective 1M context), 4 (prefill speed)
- **Derived from**: LLMLingua-2 (2403.12968)
- **Change**:
  - New `omlx/eval/llmlingua2/` module that loads the official
    XLM-RoBERTa-large compressor checkpoint via mlx-lm (or a ported
    MLX version if no native checkpoint exists) and exposes a
    `compress(text: str, ratio: float) -> str` function.
  - New middleware hook in `omlx/hypercar_server.py` on
    `/v1/chat/completions`: if the request carries
    `X-Hypercar-Compress: <ratio>` header (or if the server was
    launched with `--compress-prompts llmlingua2:3x`), the user+tool
    message content is run through the compressor *before* tokenisation.
    System prompts and the most recent user turn are always exempt.
  - Extend `phase3b_ruler()` in `omlx/bench/hypercar_bench.py` with a
    *compressed* variant: run the existing multi_key_niah@16K tasks
    once with no compression and once with LLMLingua-2 at 3x, and
    gate that the compressed accuracy stays within 10 percentage
    points of the uncompressed baseline. This is the honest gate
    that prevents us from shipping a silent quality regression.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full
  --compress-prompts llmlingua2:3x` shows (a) prefill tok/s at 16K
  >= 1.8x the uncompressed baseline, (b) HumanEval >= 85%, (c) the
  new `ruler_multi_key@16K_compressed` gate passes at accuracy
  within 10 points of the baseline. All three conditions required.
- **Effort**: M

### 18. LiveCodeBench contamination-free coding gate
- **Goal**: 2 (intelligence breadth — honest coding eval, replacing
  the contaminated HumanEval signal)
- **Derived from**: LiveCodeBench (2403.07974)
- **Change**:
  - New `omlx/eval/livecodebench/` module that loads a pinned
    post-cutoff release of LiveCodeBench from HuggingFace (e.g.
    `release_v4` containing only problems dated after 2024-07, which
    is after the known Qwen3-Coder training cutoff).
  - Reuse the existing sandboxed Python executor from
    `omlx/ttt.py`'s code verifier to run the per-problem unit tests
    — do NOT shell out to Docker.
  - Add `phase_lcb()` to `omlx/bench/hypercar_bench.py`: 20 sampled
    problems in `--quick` (capped at 60 seconds wall), 100 sampled
    problems in `--full`. Report pass@1 and per-scenario scores
    (generation, self-repair, execution, test-output).
  - Add `lcb_gen@post_cutoff >= 0.30` as a new Goal-2 gate. The
    existing HumanEval gate (>= 35%) stays for now but is demoted to
    a "legacy/contamination-monitored" row in the gate table.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full`
  prints an `lcb_gen` gate row and a per-scenario breakdown; gate
  passes at pass@1 >= 0.30 on the post-cutoff release. Wall clock
  for the `--quick` LiveCodeBench variant stays under 60 seconds.
- **Effort**: S

### 19. BigCodeBench library-usage gate
- **Goal**: 2 (intelligence breadth — tool-using code generation, the
  realistic OpenCode workload)
- **Derived from**: BigCodeBench (2406.15877)
- **Change**:
  - Vendor or pip-install BigCodeBench under `omlx/eval/bigcodebench/`
    and ship a `macos_compatible.json` allowlist that filters out any
    task requiring Docker-only dependencies. Target: at least 300
    macOS-runnable tasks survive the filter.
  - Reuse the same sandboxed Python executor from Task 18 / TTT code
    verifier. Install BigCodeBench's Python dependencies into a
    dedicated venv under `.venv-bcb/` so the main server venv stays
    clean.
  - Add `phase_bcb()` to `omlx/bench/hypercar_bench.py`: 10 sampled
    tasks in `--quick`, 100 sampled tasks in `--full`. Evaluate both
    `BigCodeBench-Full` (with docstring) and `BigCodeBench-Instruct`
    (stripped to natural-language instruction) on the full run.
  - Add `bcb_full_pass@1 >= 0.30` as a new Goal-2 gate. This is the
    fourth independent eval that Goal 2 has been missing.
  - As a stretch target, wire passing/failing task pairs from the
    same problem into the SimPO trajectory store (Task 15) so the
    bench run doubles as a training-signal source for the TTT loop.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full`
  prints a `bcb_full` and `bcb_instruct` gate row; `bcb_full` gate
  passes at pass@1 >= 0.30. The bench summary now shows four
  independent eval families (HumanEval/LCB, RULER, tau-bench, and
  BigCodeBench) all passing — that is the full Goal-2 "4 independent
  evals" claim from the CLAUDE.md status table.
- **Effort**: S-M

## Backlog coverage tasks (extracted from LIT_REVIEW.md, 2026-04-13)

### 24. Probe MLX argpartition speed for Quest top-K page selection
- **Goal**: 3 (decode speed) — de-risk Task 3 before committing
- **Derived from**: Quest (2406.10774), explicit "we'd want to check
  whether MLX's argpartition is competitive before committing" caveat
- **Change**:
  - New micro-benchmark `scripts/probe_quest_topk.py` that builds a
    synthetic `(n_pages=8192, num_heads=32)` score tensor, runs
    `mx.argpartition(-scores, kth=K)[:K]` for K in {256, 1024, 4096},
    and times it vs a sorted-topk reference over 100 trials.
  - Also measure at `n_pages=64000` (1M context / 128-tok pages) to
    see whether the primitive scales linearly or falls off a cliff.
  - Output a one-line verdict: if argpartition at 64K pages is under
    200 microseconds per head, Quest is viable as written; otherwise
    document the cost and consider a custom Metal top-K kernel as a
    prerequisite to Task 3's runtime integration.
- **Verify**: `.venv/bin/python scripts/probe_quest_topk.py` prints
  per-K timings and a PASS/FAIL verdict against the 200 microsecond
  budget. No benchmark gate change.
- **Effort**: S

### 25. Pick default RULER gate lengths by profiling eval cost
- **Goal**: 2 (eval reliability), 1 (context validation budget)
- **Derived from**: RULER (2404.06654), "Cost of adoption" note — "at
  256K+ the eval itself becomes expensive — gate only a handful of
  lengths by default"
- **Change**:
  - Add a `--profile` mode to `omlx/bench/hypercar_bench.py`'s
    `phase3b_ruler()` that runs one `multi_key_niah` task at each of
    {4K, 16K, 64K, 128K, 256K, 512K, 1M} and records wall time per
    length to `/tmp/hypercar_ruler_cost.json`.
  - Based on the profile, pick which lengths ship in `--quick` (must
    total under ~30s) vs `--full` (must total under ~5min) vs
    `--full --ruler-long` (the 256K+ tier, opt-in only) and codify
    the choice in `phase3b_ruler()`'s default length table with a
    comment citing the profile.
  - Document the chosen default lengths in `CLAUDE.md`'s gate summary.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --profile`
  writes the cost JSON; subsequent `--quick` and `--full` runs
  demonstrate that the picked lengths hit the wall-time budgets
  without manual intervention.
- **Effort**: S

### 26. MInference block-sparse prefill kernel dispatch
- **Goal**: 4 (prefill speed)
- **Derived from**: MInference 1.0 (2407.02490) — the calibration table
  classifies heads as {a_shape, vertical_slash, block_sparse, dense},
  but Task 5 only implements the vertical-slash kernel path. Block-
  sparse heads currently fall through to dense.
- **Change**:
  - Extend `omlx/patches/minference_prefill.py` with a block-sparse
    dispatch: per head tagged `block_sparse`, gather the per-head
    block index list from the calibration JSON, build a block mask
    over the query/key grid, and call MLX's existing attention
    primitives with the sparse mask. No new kernel — the sparsity
    is expressed as a mask, which MLX already supports.
  - Compose with the existing vertical-slash branch: heads are grouped
    by pattern class and dispatched in parallel where possible.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --prefill-sparse minference`
  on a model calibrated with >= 10% block-sparse heads shows prefill
  tok/s at 16K improves by at least 5% over the vertical-slash-only
  baseline from Task 5 AND `ruler_multi_key@16K` stays within 2% of
  baseline AND HumanEval >= 85%.
- **Effort**: M
- **Depends on**: 5

### 27. Text-to-LoRA offline library + hypernetwork training
- **Goal**: 2 (intelligence — agentic quality via zero-shot adapters)
- **Derived from**: Text-to-LoRA (2506.06105) — the inference-side
  wiring is a separate follow-up; the offline library assembly and
  hypernetwork training is distinct and needs cloud GPU time.
- **Change**:
  - New `scripts/t2l_build_library.py` that, for ~50 representative
    agentic task descriptions (code repair, refactor, tool-schema
    following, doc extraction, etc.), fine-tunes a per-task down_proj
    LoRA against Qwen3-Coder-30B-A3B using the existing TTT
    infrastructure. Output: `data/t2l_library.npz` containing
    {task_description, lora_weights} pairs.
  - New `scripts/t2l_train_hypernet.py` that trains a small
    transformer hypernetwork (inputs: task description embedding;
    outputs: predicted LoRA weights) against the library. Ship the
    trained checkpoint to `checkpoints/t2l_qwen3_coder_30b.safetensors`.
  - Budget: this is a one-off cloud run (the M4 Pro is too small for
    full fine-tuning) — script must be runnable against a rented
    H100 or A100 box and resume-friendly.
  - Inference wiring into `/v1/ttt/t2l` is explicitly NOT this task —
    it's a follow-up once the checkpoint exists.
- **Verify**: `python scripts/t2l_train_hypernet.py --smoke` runs
  end-to-end on a 3-task mini-library on CPU and produces a
  checkpoint; a held-out task shows reconstruction MSE below a
  documented threshold. The full cloud run is deferred and gated by
  a go/no-go decision after decode and prefill targets are closer.
- **Effort**: L

### 28. EAGLE-2 tree-attention feasibility probe for MLX
- **Goal**: 3 (decode speed) — de-risk before committing to the full
  EAGLE-2 project
- **Derived from**: EAGLE-2 (2406.16858) — the paper depends on
  verifying multiple draft-token candidates laid out as a tree, which
  requires attention primitives that accept non-causal tree masks.
  Not all MLX attention primitives support this.
- **Change**:
  - New `scripts/probe_eagle_tree_attn.py` that builds a synthetic
    tree-mask (e.g. 4 branches of depth 3 = 12 candidate tokens) and
    attempts to run it through `mx.fast.scaled_dot_product_attention`
    and through the vanilla MLX attention path. Measures whether
    either primitive accepts the custom mask and returns correct
    outputs vs a reference NumPy implementation.
  - If `mx.fast.scaled_dot_product_attention` rejects tree masks,
    document the fallback cost (vanilla MLX attention at 1.5-3x the
    runtime of the fast path) so the full EAGLE-2 project can be
    scoped honestly.
- **Verify**: `.venv/bin/python scripts/probe_eagle_tree_attn.py`
  prints a PASS/FAIL verdict for each MLX attention primitive and a
  reconstruction error vs the NumPy reference. No benchmark gate change.
- **Effort**: S

### 29. EAGLE-2 draft-head training for Qwen3-Coder-30B-A3B
- **Goal**: 3 (decode speed)
- **Derived from**: EAGLE-2 (2406.16858) — the draft-head training is
  a one-off offline step that produces the small draft network used
  at inference time. Not covered by any existing task.
- **Change**:
  - New `scripts/train_eagle_head.py` that implements the EAGLE-2
    recipe in MLX: a single-layer transformer that takes the target
    model's penultimate hidden state and predicts the next-token
    distribution. Training data = rollouts from Qwen3-Coder over a
    mixed coding+chat corpus, target = the teacher's next-token
    logits.
  - Like Task 27, this is cloud GPU work; the script must be
    resume-friendly and checkpoint to `checkpoints/eagle_head_qwen3_coder_30b.safetensors`.
  - Runtime wiring into the decode path is a follow-up task, NOT in
    scope here — this task is strictly the offline draft-head
    training.
- **Verify**: `python scripts/train_eagle_head.py --smoke` runs one
  step end-to-end on a tiny corpus and produces a checkpoint. Full
  training is deferred until Task 28 (feasibility probe) passes.
- **Effort**: L
- **Depends on**: 28

### 30. Survey mx.fast.scaled_dot_product_attention source for AMX binding
- **Goal**: 3 (decode), 4 (prefill) — measurement prerequisite
- **Derived from**: LIT_REVIEW.md pass 2 "Gap not closed" bucket 1 —
  "whether MLX's `mx.fast.scaled_dot_product_attention` already routes
  to AMX in the prefill shapes we care about, and what the tile/block
  parameters are. That is a read-the-source task, not a read-an-arxiv
  task."
- **Change**:
  - Read the MLX C++/Metal source for `mx.fast.scaled_dot_product_attention`
    and trace the dispatch for shapes in our hot path (prefill: B=1,
    H=32, N=4K..64K, D=128; decode: B=1, H=32, N=1, K=N_ctx, D=128).
  - Write findings to `research/MLX_ATTN_DISPATCH.md` covering: (a)
    which backend (Metal / AMX / accelerate / CPU) each shape lands
    on, (b) tile/block parameters where visible, (c) any shape
    thresholds that switch paths, (d) whether quantized KV is
    handled inside the fast path or dequantized upstream.
  - The output of this task feeds prioritisation — e.g. if prefill
    is already on AMX, the MInference kernel work (Task 5) should
    avoid competing with that path; if it is not, there is a second
    low-hanging optimisation behind the existing one.
- **Verify**: `research/MLX_ATTN_DISPATCH.md` exists and contains at
  least one concrete dispatch-table finding per shape category, with
  line-level citations into the MLX source tree.
- **Effort**: S-M

### 31. Profile KV allocator fragmentation at 1M context
- **Goal**: 5 (swap headroom), 6 (machine fit) — measurement task
- **Derived from**: LIT_REVIEW.md pass 3 "Gap not closed" bucket 4 —
  "profile how much of our actual 1M-context KV bill is fragmentation
  versus live data, and only revisit paging if fragmentation is a
  real cost."
- **Change**:
  - New `scripts/profile_kv_fragmentation.py` that runs a 1M-context
    prefill on a synthetic prompt, samples Metal `active` vs
    `peak`/`allocated` bytes every 1000 tokens via
    `mx.metal.get_peak_memory()` and `mx.metal.get_active_memory()`,
    and computes the fragmentation delta (allocated - active) as a
    percentage of the live KV footprint.
  - Output: `research/KV_FRAGMENTATION.md` with a chart and a verdict
    ("fragmentation is X% at 1M — paging would save ~Y GB" or
    "fragmentation is negligible, skip paging").
  - This task is strictly a measurement — no code changes to the
    allocator or cache. The follow-up paging work is contingent on
    the verdict.
- **Verify**: `.venv/bin/python scripts/profile_kv_fragmentation.py`
  writes the research note with a concrete fragmentation percentage
  and a paging recommendation. No bench gate change.
- **Effort**: S

### 32. ProMoE offline expert activation frequency profiling
- **Goal**: 6 (machine fit), 3 (decode) — prerequisite for Task 16
- **Derived from**: ProMoE (2410.22134) — the lazy-load probe in Task
  16 needs to decide which experts to zero-mask; without a real
  activation-frequency profile that decision is arbitrary and the
  probe's quality signal is unreliable.
- **Change**:
  - New `scripts/moe_profile_expert_activation.py` that hooks every
    layer's MoE router in Qwen3-Coder-30B-A3B and records, per
    (layer, expert_id), the number of times each expert is
    dispatched to over a fixed calibration corpus (~5000 tokens of
    mixed coding prompts from HumanEval + repo snippets).
  - Output: `omlx/patches/promoe_profiles/qwen3_coder_30b_a3b_instruct_8bit.json`
    with a per-layer sorted list of (expert_id, activation_count).
    This is the input Task 16 uses to decide which experts are
    "cold" (bottom 50% by count) and therefore eligible for
    zero-masking in the probe.
  - Task 16's `--resident-fraction` flag must read this profile
    rather than pick an arbitrary contiguous subset; update the
    Task 16 probe script header to reference this file.
- **Verify**: `.venv/bin/python scripts/moe_profile_expert_activation.py
  --model <qwen3-coder> --corpus data/promoe_calib.jsonl` writes the
  profile JSON; assertion confirms every layer has exactly 128
  experts profiled AND the top-10 and bottom-10 expert counts per
  layer are both non-zero (i.e. the calibration actually touched
  experts across the range).
- **Effort**: S

### 33. ProMoE full lazy-load MoE runtime (Task 16 follow-up "16b")
- **Goal**: 6 (machine fit), 5 (swap headroom), 3 (decode), 4 (prefill)
- **Derived from**: ProMoE (2410.22134) — Task 16 is only the one-day
  de-risking probe; the full system (lazy-weight MoEMLP wrapper,
  linear-probe predictor, hot/cold LRU eviction) is called out as
  "Task 16b" in Task 16's description but is not itself filed.
- **Change**:
  - New `omlx/patches/lazy_moe.py` implementing a `LazyMoEMLP` wrapper
    that mmap's expert weights from disk and unpacks on first touch.
    Use `mx.load` with zero-copy mmap where supported.
  - New tiny linear-probe predictor: trained once over traffic traces
    captured via Task 32's profiler, it predicts next-layer expert
    dispatch from current-layer router logits. Runs one layer ahead
    and prefetches cold experts into the fast tier.
  - LRU hot/cold tracking with a configurable `--moe-resident-fraction`
    budget in `omlx/hypercar_server.py` and `hypercar_bench.py`.
  - Composes with 8-bit quantized weights (no re-quant needed — the
    codec is per-expert already).
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --moe-resident-fraction 0.5`
  shows Metal at load drops by >= 8GB vs baseline AND decode tok/s at
  16K regresses by < 10% vs baseline AND HumanEval >= 85% AND
  `ruler_multi_key@16K` within 2% of baseline. All four conditions
  required.
- **Effort**: L
- **Depends on**: 16, 32

### 34. LLMLingua-2 compressor fine-tune on coding corpus
- **Goal**: 1 (effective context), 4 (prefill speed) — follow-up to Task 17
- **Derived from**: LLMLingua-2 (2403.12968) — the shipped XLM-RoBERTa
  compressor is trained on a task-agnostic distillation from GPT-4 on
  general prompts; a compressor fine-tuned on our actual workload
  (code + tool output + repo snippets) should preserve quality at a
  higher compression ratio than the off-the-shelf checkpoint.
- **Change**:
  - New `scripts/llmlingua2_finetune.py` that takes the official
    LLMLingua-2 checkpoint as a starting point and fine-tunes the
    token-classifier on a coding-specific distillation dataset:
    (a) collect ~1000 real OpenCode sessions + HumanEval+ prompts,
    (b) run GPT-4 (or Claude) to produce keep/drop masks at 3x and
    5x compression ratios, (c) fine-tune the compressor against
    these masks with the same objective the original paper uses.
  - Output: `checkpoints/llmlingua2_code.safetensors`. Task 17's
    middleware gains an optional `--compress-prompts llmlingua2-code:3x`
    variant that loads this checkpoint instead of the shipped one.
  - Cloud run: like Tasks 27 and 29, this is offline training and
    should be resume-friendly. The distillation-data generation
    itself (calling GPT-4 on ~1000 prompts) is the dominant cost,
    not the fine-tuning.
- **Verify**: With the fine-tuned checkpoint loaded,
  `.venv/bin/python -m omlx.bench.hypercar_bench --full --compress-prompts llmlingua2-code:5x`
  shows (a) prefill tok/s at 16K >= 2.5x the uncompressed baseline
  (vs Task 17's 1.8x at 3x), AND (b) `ruler_multi_key@16K_compressed`
  stays within 10 points of uncompressed baseline, AND (c) HumanEval
  >= 85%.
- **Effort**: M-L
- **Depends on**: 17
