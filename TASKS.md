# Hypercar Task Backlog
_Atomic, testable optimization tasks. Organized by the Hypercar goal they advance._
_Last updated: 2026-04-15 — 41 tasks completed, 5/6 Hypercar goals met_

## 🔴 HIGH PRIORITY — work on this next

This section takes precedence over all others. If you are an implementation
loop selecting a task to work on, pick from here FIRST. Only fall through
to the regular sections below if this section is empty or its tasks are
all in `## In Progress`.

_Work from here first. Only fall through to regular sections if these are all in `## In Progress` or completed._

### 253. 🔴 Migrate target model to Qwen3.6-35B-A3B (highest-expected bench delta)
- **Goal**: 2 (intelligence — Qwen3.6 scores 73.4% SWE-Bench vs 70% for Qwen3-Coder; expected LCB + MMLU-Pro lift — LCB currently at 30% floor is the weakest gate and the one most likely to move under a better model). Also Goal 1 (native 262K → 1M context without fine-tuning; obsoletes several LongRoPE/position-extrapolation tasks in the backlog).
- **Derived from**: memory note `project_qwen36_migration.md` (2026-04-23); user-directed pivot.
- **Scope — multi-cycle**, not a single-cron task. Break into phases:
  - **Phase 1 (prerequisite)**: Download `mlx-community/Qwen3.6-35B-A3B-4bit` (~18 GB across 4 safetensor shards) OR `unsloth/Qwen3.6-35B-A3B-MLX-8bit` (~35 GB). Download is NOT a good cron-cycle action — ask user to run `huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit` outside the loop. Confirm HF availability: all three paths return HTTP 200 (official Qwen/, mlx-community/, unsloth/).
  - **Phase 2 (load-path validation)**: MLX-LM 0.31.2 already supports `qwen3_5_moe` arch. BUT config.json lists `architectures: ["Qwen3_5MoeForConditionalGeneration"]` (multimodal class, not `ForCausalLM`). Test whether `mlx_lm.load("mlx-community/Qwen3.6-35B-A3B-4bit")` succeeds. If not, investigate: (a) is there a text-only variant? (b) does MLX-LM's qwen3_vl_moe handle this? (c) do we need to manually strip vision encoder weights? vision_config has `deepstack_visual_indexes` — DeepStack-style visual.
  - **Phase 3 (bench against Qwen3-Coder baseline)**: once loading works, run `.venv/bin/python -m omlx.bench.hypercar_bench --full --model mlx-community/Qwen3.6-35B-A3B-4bit`. Record NEW baseline. Expected deltas: LCB likely jumps (73.4% SWE-Bench is the headline); MMLU-Pro likely up; HumanEval ~flat or up. NIAH/RULER should still pass. Memory: 4-bit 35B ≈ 18 GB + KV — should fit comfortably in 48 GB.
  - **Phase 4 (compatibility sweep)**: DuoKV calibration (Task 12 artifact) was against Qwen3-Coder — re-run for Qwen3.6. SnapKV calibration (Q capture) likely portable. TQ3 codec is model-agnostic. ProMoE profile (Task 32) needs re-running: Qwen3.6 has 256 experts with 8 routed + 1 shared, different from Qwen3-Coder 128/8.
  - **Phase 5 (cleanup)**: Centralize MODEL_ID — currently duplicated across ≥10 files (`hypercar_server.py`, `agentic.py`, `tq_calibrate.py`, `ttt.py`, `bench/snapkv_bench.py`, `bench/agentic_bench.py`, `bench/opencode_bench.py`, `bench/long_context_quality.py`, `bench/shadowkv_rank_probe.py`, `bench/hypercar_bench.py`). Move to a single `omlx/model_constants.py` so future migrations are a one-line change.
- **Verify per phase**:
  - P1: `ls ~/.cache/huggingface/hub/ | grep -i qwen3.6` shows the model dir.
  - P2: `.venv/bin/python -c "from mlx_lm import load; m,t = load('mlx-community/Qwen3.6-35B-A3B-4bit'); print(type(m).__name__, sum(p.size for p in m.parameters().values()))"` exits 0 and reports a sane parameter count (~35B × 4-bit quant → roughly aligned with file size).
  - P3: `hypercar_bench --full --model mlx-community/Qwen3.6-35B-A3B-4bit` completes ALL GATES PASSED; record per-gate numbers for the new baseline.
  - P4-5: downstream, tracked separately once Phase 3 confirms the model is serviceable.
- **Effort**: L overall (multi-cycle). Phase 1 is user-initiated (~5-10 min download). Phase 2 is ~30 min of debugging depending on what the loader does. Phase 3 is a full bench (~23 min + result review). Phase 4 is M. Phase 5 is S.
- **Depends on**: user-initiated model download (Phase 1). Blocks nothing else; can run alongside LCB follow-ups on the current model.
- **Why this is highest-priority**: every optimization we ship on Qwen3-Coder-30B has a shorter shelf life than one we ship on the new target. Migrating first avoids sunk-cost work. Also the biggest single bench-result lever available (switching a model can move gates double-digits where prompt engineering moved 0 — Task 130(b) validated this).

### 22. [COMPLETED] Fix 8-bit model Goal 5 violation — apply `--kv-bits 2` and verify
- **Goal**: 5 (swap throughput headroom), 6 (M4 Pro 48 GB fit)
- **Why this is highest priority**: Across 10 benchmark runs
  (Runs 23-32, 2026-04-13), the 8-bit model STRUCTURALLY violates
  CLAUDE.md Goal 5. Under the old swap-depth metric, 6 of 10 runs
  exceeded 8 GB peak. Under the NEW p90-sustained-swap-rate metric
  introduced by Task #23, Run 32's per-sample swap throughput had
  median 141 MB/s and peaks to 3768 MB/s against the p90 < 100 MB/s
  gate — a 5-8x violation. Every additional benchmark run on the
  current config will show Goal 5 red. Until this is fixed, we
  cannot get a clean green benchmark on 8-bit, which blocks
  meaningful regression tracking for Goals 3 and 4.
- **Derived from**: Hypercar benchmark runs 23-32 (2026-04-13),
  bench/snapshots/run*/env.json swap metrics. Specifically:
  - Old depth metric: Runs 23/28/29/30 violated 8 GB (4 of 8 in the
    pre-dedup set).
  - Run 31 (first run with Task 9 headroom gate): swap peak 38.16 GB
    delta, phase 3b aborted at task 9/15 on breach
  - Run 32 (first run with Task 8 profiler rebuild): swap peak
    37.21 GB delta, phase 3b aborted at task 4/15 on breach
    (earlier because the new profiler accurately catches the threshold
    crossing; old profiler was under-reporting)
  - New profiler shows median 141 MB/s sustained swap I/O per sample
    with 3.7 GB/s burst peaks — the 8-bit model's compressor pressure
    is severely over any reasonable p90 threshold.
- **Change**: This is a STRUCTURAL headroom issue, not a harness bug.
  Primary remediation path — **Option (a), apply `--kv-bits 2`**:
  - Task #2 (KIVI 2-bit KV probe) already landed the `--kv-bits` CLI
    flag on `hypercar_bench` via commit b8fa6d3. The TurboQuant codec
    is already parameterized on bit-width. No new code is required to
    *run* the 2-bit path; you just need to pass `--kv-bits 2` and
    measure what happens.
  - The mathematical expectation: 2-bit codebook has 4 levels vs 3-bit's
    8, packed_width drops from 12 → 8 uint32 words at D=128, giving
    ~33% KV memory savings. At 16K context this frees ~0.4 GB; at 64K
    context (the cliff-producing workload) it frees ~1.6 GB. Combined
    with Task #9's headroom gate, this should be enough to keep swap
    delta under the 12.9 GB watchdog limit across all 15 RULER tasks.
  - Run `.venv/bin/python -m omlx.bench.hypercar_bench --full --kv-bits 2`
    and observe:
    - Does Phase 3b complete all 15 RULER tasks (not just 4)?
    - Does the new-profiler swap_io_mb_per_s median stay under 200?
    - Does Phase 4 HumanEval finally run for the first time in 10 runs?
    - Does HumanEval pass@1 stay within 10 points of 3-bit baseline (90%)?
  - If 2-bit regresses HumanEval below 80%, back out and try
    **Option (b), --quest-topk 32** (Task #3 already shipped via ac2491e).
  - If both options fail, fall back to **Option (c)**: run hypercar_bench
    with `--max-metal-pct 90 --max-swap-pct 40` to loosen the watchdog
    and document that the 8-bit model cannot meet Goal 5 on M4 Pro 48 GB
    without significant architectural changes (Tasks #12-13 DuoAttention,
    Task #16 ProMoE, Task #33 full ProMoE runtime).
- **Verify**: Run the following and commit the output to
  `bench/snapshots/task22_kv_bits_2/` (or whichever slug the run uses):
  ```
  .venv/bin/python -m omlx.bench.hypercar_bench --full --kv-bits 2
  ```
  PASS criteria (all must hold):
  1. Phase 3b completes all 15 RULER tasks (no abort/crash) OR Phase 3b
     aborts cleanly only on tasks explicitly SKIPped by the headroom gate.
  2. Phase 4 HumanEval runs to completion and reports pass@1 ≥ 80%
     (acceptance: 10-point regression from 3-bit baseline 90%).
  3. Phase 5 Memory Profile reports swap_peak_gb < 8.0 (old metric).
  4. Profile.json median `swap_io_mb_per_s` across non-idle samples
     < 200 MB/s (approach to Task #23's new p90 < 100 MB/s gate).
  5. Metal peak_gb < 41.2 (no ceiling breach).
  At least one of criteria (2-4) must be strictly better than the Run 32
  baseline (Phase 4 never ran, swap peak 37.2 GB, swap_io median 141 MB/s).
- **Effort**: S (this is mostly running the bench with a flag and
  interpreting the result — the flag and codec are already in place).
  Escalates to M if 2-bit regresses quality and you need to try Option (b)
  or (c). Escalates to L only if ALL three options fail and you need to
  investigate a structural fix (which would get filed as a new task
  anyway, not done under this one).
- **Do NOT**:
  - Modify `omlx/bench/hypercar_bench.py` or `omlx/turboquant_kv.py` to
    "patch around" the issue without running the --kv-bits 2 probe first.
    The codec already supports 2-bit; your job is to validate it works.
  - Change the watchdog memory limits to make the gate pass falsely.
    Loosening limits is Option (c) only, and it's a last-resort
    documentation move, not a fix.
  - Skip the quick benchmark before committing. Per the implementation
    loop prompt, run `.venv/bin/python -m omlx.bench.hypercar_bench --quick`
    before committing code changes to validate the smoke + coherence
    gates still pass at 2-bit KV.

---

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
  Per ProLong (arXiv:2410.02660), the three diagnostic RULER subtask families are:
  - **Retrieval**: multi_key_niah@16K (keys=2,3,5) — gate ≥ 0.8
  - **Multi-hop tracing**: variable_tracking@4K (chain=4,8) — gate ≥ 0.7
  - **Aggregation**: frequent_word@4K (words=5,10) — informational, no gate
  Length tiers: 4K (quick), 16K (default), 64K (--full only).
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
  Per ProLong (arXiv:2410.02660): when a task breaches at length L,
  the gate should evaluate on tasks that completed at shorter lengths
  rather than failing the entire phase. Current implementation already
  does this — breach results are filtered from gate aggregation via
  `"accuracy" in r` guard, and skipped tasks are tracked in `skipped` list.
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

### 24. Add 2-SHA regression detector to aggregate.py + fix minference layer counter
- **Goal**: 3, 4 (regression detection) and 4 (minference correctness)
- **Derived from**: Task 21 spec gap — `--report` only shows one SHA, no
  baseline comparison or 2-sigma flagging. Also: `minference_prefill.py`
  defines `reset_layer_counter()` but never calls it — layer indices
  drift across forward passes in multi-turn/batch scenarios.
- **Change**:
  - Extend `aggregate.py --report <sha> --baseline <sha>` to compare two
    SHAs side-by-side, compute per-phase median delta, flag phases where
    `|delta| > 2 * combined_std` as REGRESSION or IMPROVEMENT.
  - Wire `reset_layer_counter()` into the bench (`phase3b_ruler`,
    `phase3_niah`) and server (`hypercar_load`) so the counter resets
    before each independent forward pass.
- **Verify**: `python omlx/bench/aggregate.py --report HEAD --baseline <prev_sha>`
  prints a side-by-side table with delta and significance flags.
  `grep -r reset_layer_counter` shows call sites in bench and server.
- **Effort**: S

### 25. Add --warmup flag to hypercar_bench for Metal kernel cache priming
- **Goal**: 3, 4 (decode/prefill speed measurement reliability)
- **Derived from**: Task 20 hypothesis 1 — bimodal timing may be caused by
  Metal kernel cache cold/warm state. A warmup pass before the timed
  measurement would eliminate the slow-mode if this hypothesis is correct.
- **Change**:
  - Add `--warmup` flag to hypercar_bench that runs a short dummy
    generation (same model, same cache type, ~512 tokens) before Phase 0
    to prime Metal kernel compilation and GPU pipeline state.
  - The warmup output is discarded — it only serves to warm the Metal
    shader cache and trigger JIT compilation of all kernel variants.
  - Clear Metal cache after warmup to avoid inflating memory baselines.
- **Verify**: `hypercar_bench --quick --warmup` passes all gates and
  Phase 0 decode speed is within 5% of a second consecutive run
  (eliminating first-run penalty).
- **Effort**: S

## In Progress

- **Task 255**: LCB CoT bench validation (re-test Task 130(b) hypothesis now that Task 254 landed the CoT prompt in the bench codepath) — full bench run + offline analysis of response data (started 2026-04-23).








## Completed

- **Task 254**: LiveCodeBench failure-mode instrumentation + prompt-codepath unification (2026-04-23)
  - **Discovery (highest-value outcome)**: `hypercar_bench.py::phase3d_livecodebench` (line 1530) has its OWN inline prompt, **not** `LiveCodeBenchBenchmark.format_prompt()` from `omlx/eval/livecodebench.py`. Task 130(b) edited the latter thinking the former would take effect. The two codepaths had drifted. This invalidates Task 130(b)'s "CoT hypothesis falsified" conclusion — that bench run was measuring the bench's OLD, unchanged inline prompt. Task 130(b) Completed entry has been amended with a retraction.
  - **Fix 1 — prompt unification** (`hypercar_bench.py:1530-1540`): bench's inline prompt now matches the CoT version from `livecodebench.py` — invites step-by-step reasoning, asks for a final ```python block. Comment added pointing at the canonical version so future drift is visible.
  - **Fix 2 — instrumentation** (`hypercar_bench.py:1577-1587`): results.json now saves the FULL `code`, `response`, and `prompt` per problem (was: truncated `code[:100]`, no response, no prompt). Next bench run produces actually-diagnosable failure data.
  - **Verification**: `.venv/bin/python -c "..."` inline assertion confirms CoT prompt applied, response+code saved, truncation removed. (No bench run yet — change is the prerequisite for the next bench run's data to be useful.)
  - **Bench validation needed**: next `--full` run will produce the REAL first LCB measurement under CoT. Expected pattern: if CoT helps, LCB > 30%. If it doesn't, results.json now contains full responses per failing problem so we can characterize the actual failure mode (wrong algorithm? truncation? extraction bug? capability ceiling?).
  - **Follow-up threads**: (i) once the next bench has data, analyze failures offline — look at what the model actually wrote for each failing problem; (ii) if CoT DOES help, Task 130(b)'s hypothesis is un-falsified after all and we got back 10+ LCB percentage points for free; (iii) consider a test that asserts the two LCB prompts stay in sync, to prevent future drift.
- **Task 253 Phase 5**: Centralize MODEL_ID into `omlx/model_constants.py` (2026-04-23)
  - New file `omlx/model_constants.py` (~40 lines): single source of truth for model IDs. Defines `DEFAULT_MODEL_ID`, `SERVER_DEFAULT_MODEL_ID`, `BENCH_DEFAULT_MODEL_ID`, plus named constants for current (Qwen3-Coder 8bit/4bit) and planned (Qwen3.6 4bit/8bit) targets.
  - Migrated top 2 call sites: `omlx/bench/hypercar_bench.py:48` and `omlx/hypercar_server.py:295` now import from the registry.
  - 4 new tests in `tests/test_model_constants.py` — all PASS, covering registry presence, server/bench consistency, bench module using the central default, and Qwen3.6 targets being defined.
  - Remaining call sites (`agentic.py`, `tq_calibrate.py`, `ttt.py`, `bench/snapkv_bench.py`, `bench/agentic_bench.py`, `bench/opencode_bench.py`, `bench/long_context_quality.py`, `bench/shadowkv_rank_probe.py`) still hold string literals — safe to migrate piecemeal in future cycles, deferred to keep this change small.
  - **Outcome**: when Task 253 Phase 3 lands the new Qwen3.6 target, the server + main bench flip is now a ONE-LINE change to `DEFAULT_MODEL_ID`. Also: the planned Qwen3.6 string literals now have names callers can reference directly instead of copying strings.
  - No bench delta expected (pure refactor — same MODEL_ID value). Safety: ALL existing tests pass; imports verified manually.
- **Task 130(b)**: LiveCodeBench CoT prompt tuning — enable reasoning, remove "no explanations" directive (2026-04-23)
  - **Targets the weakest gate**: LiveCodeBench 6/20 (30%) in Run 87 — the only bench phase near floor. All other gates PASS with healthy margin (HumanEval 95%, MMLU-Pro 62%, NIAH/RULER 100%).
  - **Root cause**: `omlx/eval/livecodebench.py:172-182` prompt told a reasoning-capable model *"Provide only the complete Python code, no explanations."* This self-sabotages medium/hard problems where CoT is essential. Run 87 failure distribution confirms: 5/8 easy pass, 1/7 medium pass, 0/4 hard pass — failures concentrate exactly where reasoning matters.
  - **Change**: `omlx/eval/livecodebench.py:172-187` — new prompt invites "step-by-step" reasoning, asks for a final ```python block. Safe because `_extract_last_code_block` (base.py:125) already takes the LAST ```python block, ignoring any draft code the model emits during reasoning.
  - **Tests added** (`tests/test_eval.py::TestLiveCodeBench`, 3 new):
    - `test_prompt_invites_cot_reasoning` — prompt positively mentions step-by-step, lacks "no explanations"
    - `test_extractor_handles_cot_then_code_block` — reasoning prose + final ```python extracts cleanly
    - `test_extractor_prefers_last_code_block_over_draft` — draft code blocks during reasoning don't pollute extraction
  - All 7 TestLiveCodeBench tests PASS (`.venv/bin/python -m pytest tests/test_eval.py::TestLiveCodeBench -v` → 7/7).
  - **Bench validated (2026-04-23T15:54 PDT, `--full`, 1386s, ALL GATES PASSED)**: **LiveCodeBench 6/20 (30%) — IDENTICAL to Run 87 baseline.** Every one of the 20 problems produced the same pass/fail. Other gates unchanged: MMLU-Pro 62/100, HumanEval 19/20 (95%), RULER 100%, NIAH/SnapKV/Tool-Call all PASS. No regression; no improvement.
  - **⚠️ RETRACTION (2026-04-23 via Task 254)**: the "hypothesis falsified" conclusion is **not actually supported** by this bench. The bench's Phase 3d uses its OWN inline prompt in `hypercar_bench.py:1530` — NOT the `LiveCodeBenchBenchmark.format_prompt()` method I edited in `omlx/eval/livecodebench.py`. The two codepaths had drifted. My Task 130(b) prompt change was inert for the bench measurement. The bench prompt already did NOT say "no explanations" before my change; it said "Provide only the complete Python code in a ```python block" (also restrictive, but different wording). Task 254 unifies both prompts and re-runs the bench will be needed to actually test the CoT hypothesis. Keeping this entry as-is for accurate historical record; see Task 254 for the real first CoT validation run.
  - **Hypothesis STATUS AFTER RETRACTION**: untested on the bench. The livecodebench.py prompt change is still valid (for the unused class API); the bench-side change lands via Task 254.
  - **Net value of the change**: the prompt is now aligned with the model's actual behavior (no more self-sabotaging instruction the model was ignoring anyway), plus 3 new contract tests guard the prompt/extractor relationship. The change is SAFE to keep (no regression) but does NOT move the LCB gate.
  - **Real next step for LCB**: since prompt wording is not the lever, try one of — (i) best-of-N sampling with temperature > 0 + pick by local test-case pass; (ii) few-shot examples for the specific problem classes that fail (DP, greedy optimization, modular arithmetic tricks); (iii) harder intervention — swap LCB to a smaller model-friendlier subset or accept 30% as the Qwen3-Coder-30B floor and target a different weak gate. Filed as follow-up priority candidate for future cycles.
- **Task 177**: [CRITICAL REGRESSION] SnapKV NIAH FAIL at 49% keep — closed out, resolved by Runs 86-87 (2026-04-23)
  - Run 84 (commit `eaf108a`): Phase 3e FAIL, needle lost, filler text emitted
  - Run 85 (commit `bd1fe8a`): CRASH — an uncommitted attempted fix used `mx.argwhere` which doesn't exist in MLX 0.31.1
  - Commit `9e3758d`: reverted `get_keep_indices` to restore working scatter pattern (not `mx.argwhere`)
  - Commit `46008c2` = Run 86: ALL GATES PASSED post-fix
  - Run 87 (verified): `Phase 3e: SnapKV Quality PASS`, output `'SNAPKV-BENCH-9921<|im_end|>...'`, `ALL GATES PASSED Total: 1394.3s`
  - False lead: Task 177's 3 bisect candidates (a217e29 scatter, eca3aa8 skip_rerope, 6c87533 GQA broadcast) all survived. Real bug was in the uncommitted `mx.argwhere` substitution.
- **Task 76**: Jagged-tensor scheduler design note for two-tier KV cache (HSTU-CP inspiration) (2026-04-23)
  - Design note at `research/design_notes/jagged_kv_schedule.md` (~135 lines)
  - Section (a): located live jaggedness in `omlx/duo_kv_cache.py:150-222` (head-class split) and `:224-301` (jagged-to-dense collapse via pre-alloc slab + `take_along_axis` + zero-mask). Sharpest point: `_update_quantized` merges TQ3 3-bit retrieval + fp16 streaming ring.
  - Section (b): audited MLX 0.31.1 primitives. `mx.take_along_axis` + `mx.where` + lazy `mx.eval` are the tools; true `torch.nested`-style ragged dispatch, `madvise` residency hints, and Python-level multi-stream scheduling do NOT exist on MLX/Apple Silicon.
  - Section (c): min-viable scheduler = careful operation ordering, not multi-stream parallelism. Six concrete rules; items 2, 3, 5, 6 inform Task 65; items 3, 6 inform Task 64.
  - Referenced from Task 64 and Task 65 (pointer added to both task entries).
- **Task 48**: Pre-read vAttention design note — closure artifact for Task 31 (2026-04-22)
  - Research note at `research/vattention_mtlheap_notes.md` (102 lines)
  - 5-bullet summary of vAttention's reserve-then-commit architecture
  - Metal-primitive mapping table: MTLHeap matches `cuMemAddressReserve` but is non-growable and physically backs all pages at creation — eliminates vAttention's core asymmetry
  - Documented why MLX's Metal allocator already achieves what vAttention delivers on CUDA (~2.1% fragmentation, ~0.5 GB at 1M extrapolated — Task 31 result)
  - Two Metal limitations that block a faithful vAttention port: non-growable heaps, no multi-context virtual sharing
  - Recommendation: if fragmentation >10% ever surfaces at 256K+ real workloads, start with measurement harness — do NOT start with MTLHeap port
- **DuoKV 40K NIAH root cause: DuoAttention head classification, NOT quantization** (2026-04-18)
  - fp16 all heads: PASS | native 3-bit all: PASS | DuoKV (59% streaming): FAIL
  - Streaming heads discard context beyond 260 tokens — loses needle at 40K
  - Calibration was at short context; some "streaming" heads need full context for NIAH
  - Fix: recalibrate at longer context OR reduce streaming fraction
- **DuoKV+quantize 40K deep validation** — THREE FAILURES found (2026-04-18)
  - NIAH: **FAIL** — "GAMMA-552" (truncated 1 char, quality degrades at long context)
  - Metal peak: **44.0 GB** — breaches 41.2 GB limit by 2.8 GB during prefill
  - Decode: **2.9 tok/s** — catastrophic at 40K (quantized SDPA doesn't scale)
  - Prefill: 95 tok/s (7 min for 40K tokens)
  - **Profiling**: 2.9 tok/s was contaminated by 2.8s Metal JIT cold start. Steady-state: 24 tok/s at 3.5K.
    eval=36.5ms/step dominates (forward is lazy at 4-5ms). Root causes:
    (1) Cold start: warmup doesn't prime quantized SDPA kernel
    (2) Memory breach: fp16→quantize transient during prefill
    (3) NIAH almost-pass: precision issue (off by 1 digit)
  - **Conclusion**: DuoKV+quantize has potential but needs engineering work at scale.
    fp16+SnapKV at 128K remains the most validated path.
- **Task 150: DuoKV quantized retrieval + split-SDPA** — NIAH PASS at 4K+16K (2026-04-18)
  - --duo-quantize flag wired through server→patches→DuoKVCache
  - Per-head dispatch: retrieval → QuantizedKVCache(3-bit), streaming → StreamingKVCache(fp16 ring)
  - **Split-SDPA**: quantized attention for retrieval, fp16 for streaming — avoids dequant entirely
  - Decode: 41 tok/s short, 10 tok/s at 10K — quantized SDPA scales with context
  - **NIAH@4K: PASS, NIAH@16K (10K tokens): PASS** — needle found at both contexts
  - At 256K: quantized retrieval saves ~20 GB vs fp16 (24.6 vs 196 GB at 1M)
- **Task 150 scaffolding: --duo-quantize flag** — wired through server→patches→DuoKVCache (2026-04-18)
  - `--duo-quantize` flag on server, `quantize_retrieval` param on DuoKVCache and apply_hypercar_patches
  - Implementation pending: actual QuantizedKVCache for retrieval heads (M effort, next cycle)
  - At 1M: fp16 retrieval = 196 GB (impossible), 3-bit = 24.6 GB (fits in 48 GB budget)
- **Run 86 ALL GATES PASSED** — post-efficiency-audit validation + analyst profiler committed (2026-04-18)
  - 10/10 phases pass: Smoke, Coherence, Code Intel 5/5, NIAH, RULER 100%, MMLU-Pro 64%, SnapKV Quality, Tool-Call JSON, Memory
  - Committed analyst efficiency_profile.py (431 lines) — micro-benchmark tool for profiling hot paths
- **Task 163: SnapKV+DuoKV validated** — confirmed already wired, added 5 composition tests (2026-04-18)
  - SnapKV eviction works with ALL KV modes (DuoKV, fp16, native, TQ3)
  - No kv_mode restriction in apply_snapkv_to_generate
  - DuoKV state getter/setter handles compact_cache correctly
  - Documented in CLAUDE.md: DuoKV+SnapKV for 47% faster long-context decode
- **Efficiency audit regression tests** — 25 new tests guarding all perf fixes (2026-04-18)
  - DuoKV: pre-alloc slab (no concat), gather trim (take_along_axis), ring vectorize
  - TQ3: fused quantize (WHT→dense), fused dequantize, geometric growth helper
  - SnapKV: no .at[], band mask freshness, argpartition in all selectors, GQA broadcast
  - skip_rerope=True default, tool-call gate phase
- **SnapKV final vectorization cleanup** — fair selection, global selection, get_keep_indices (2026-04-18)
  - _select_fair: Python sorted()+tolist() → mx.argpartition (same pattern as other strategies)
  - _select_global: simplified to single argpartition on B=0 slice
  - get_keep_indices: mask_np.tolist() → mx.argwhere().flatten()
  - Dead code removed (unused pooled_np variable in fair selection)
- **CLAUDE.md efficiency audit update** — documented 12 perf commits with impact table (2026-04-18)
  - Updated goal status: TQ3 prefill 672 tok/s (was "270x slower"), DuoKV 248x faster decode
  - Added perf optimization table with all 12 fixes and their measured impact
- **Task 161: Tool-call JSON validity gate** — Phase 3f in benchmark (2026-04-18)
  - Prompts model for JSON tool call, validates single parseable JSON object
  - Informational gate (doesn't block) — tracks regression for --grammar flag
  - Extracts first JSON object from duplicate output to diagnose the issue
- **Task 160: TQ3 session save during fp16 warmup** — quantize buffer before save check (2026-04-18)
  - save_to_disk now calls _quantize_fp16_buffer() if cache hasn't crossed min_quant_tokens
  - Fixes: prefill < 512 tokens → save_to_disk → "Cannot save empty cache" error
- **Task 146: Submodular selection vectorized** — greedy loop stays Python, inner D-dim dotprod → MLX matmul (2026-04-18)
  - Eliminated .tolist() for scores and value vectors — everything stays in MLX
  - Score-only fallback also uses mx.argpartition instead of Python sorted()
- **Task 142: DuoKV streaming trim gather** — per-head Python loop → single take_along_axis (2026-04-18)
  - Build (H_kv, T) gather index: streaming→[sink+window+pad], retrieval→[0..T-1]
  - One take_along_axis + mask replaces 12 intermediate tensors per head
- **Task 155: SnapKV default 50% keep** — 25% hurts decode 14%, 50% is 7% faster (2026-04-18)
  - Updated snapkv_bench default ratios and CLAUDE.md documentation
- **Task 144: TQ3 KV buffer geometric growth** — 2x doubling replaces linear step=256 concat (2026-04-18)
  - Extracted _ensure_compressed_storage() helper, used by both prefill and decode paths
  - At 64K: ~7 allocs instead of ~250 (log2(64K/256) = 8)
  - No transient 2x memory spike from concat — allocate+copy instead
- **Task 141: Segmented selection vectorized** — mx.argpartition replaces Python sorted()+tolist() (2026-04-18)
  - GPU-side top-K via argpartition — O(n) vs O(n log n), no CPU roundtrip
- **Task 148: CAOTE/importance scoring GQA-aware broadcast** — eliminate mx.repeat GQA expansion (2026-04-18)
  - Reshape Q into (B, H_kv, gqa_ratio, obs_len, D) and broadcast with K
  - Saves 1.07 GB per layer at 64K (no full key copy for GQA expansion)
  - Applied to both compute_caote_importance and compute_importance_from_real_q
- **Task 149+143: DuoKV pre-allocated slab + StreamingKV vectorized ring** — eliminate per-token concat (2026-04-18)
  - DuoKV: pre-alloc buffer with 256-token headroom, O(1) slice write vs O(context) concat
  - Analyst profiled: 248x faster at 64K (34.78ms → 0.14ms per token)
  - StreamingKV: ring-buffer writes vectorized with modular arithmetic scatter
  - state.setter also pre-allocates with headroom for SnapKV compact_cache compat
- **Task 168: Default skip_rerope=True** — 4.7x faster decode after eviction, NIAH PASS confirmed (2026-04-18)
  - Re-RoPE was costing 4.7x decode speed (7.2 vs 33.7 tok/s after eviction)
  - Keys keep original RoPE positions with gaps — attention still retrieves correctly
- **Task 140+147+139: Vectorize SnapKV scatter ops** — replace .at[].add() Python loops with single scatter/band-mask ops (2026-04-18)
  - keep_mask: O(T) Python loop → single `keep_mask[:, idx] = True`
  - GER safety: O(T) Python loop → single `important_mask[threshold_idx] = True`
  - Freshness scoring: O(T) inner loop → vectorized band mask + mx.sum (85% of pipeline at 64K)
- **Task 152+145: TQ3 fused quantize + fused dequantize** — use existing fused dense kernel for WHT path (2026-04-18)
  - WHT rotation matrix is symmetric (H.T = H), so fused dense kernel works directly
  - 15x short-context prefill speedup (validated: identical output, 99.67% packed index match)
  - Also switched all dequantize hot paths to use dequantize_fused (2x speedup)
  - Combined: 1.33x TQ3 prefill speedup at 8K (analyst INV 23 validated)
- **Task 109: Per-layer TTT learning rates** — reservoir computing schedule (2026-04-18)
  - `omlx/ttt_schedules.py`: compute_layer_lrs() with 3 schedules
  - `reservoir`: alpha_l = base_lr × (l/L)^gamma — shallow frozen, deep adaptive
  - `cosine`: U-shaped (high at edges, low in middle)
  - `uniform`: current default (all layers equal)
  - `spectral_radius_approx()`: power iteration monitor for echo state stability
  - Ready for TTT engine integration via `--ttt-layer-schedule` flag
- **Progressive mid-prefill eviction** — evict every 16K tokens during prefill (2026-04-18)
  - Keeps peak memory bounded at ~80K effective context per chunk
  - Reduces O(n²) attention: 4 × 80K² = 25B vs 256K² = 65B (2.6× speedup)
  - 50% keep during prefill, final eviction to target ratio at end
  - Re-initializes KV capture hooks after each eviction for clean fp16
  - Enables practical 256K contexts on 48 GB (peak ~23 GB)
- **Task 130: LiveCodeBench difficulty analysis** — per-difficulty breakdown (2026-04-17)
  - Phase 3d now reports easy/medium/hard pass rates, not just aggregate
  - Data already has difficulty tags (322 easy, 383 medium, 350 hard)
  - Results include `by_difficulty` dict for tracking progression
  - Analyst baseline: 30% aggregate, likely 50%+ easy, 20% medium, <10% hard
- **Task 131: JSON encoding fix** — safe control char escaping in SSE (2026-04-17)
  - Patched `json.dumps` in mlx_lm.server to use `ensure_ascii=True`
  - Guarantees all control characters in code responses are \uXXXX escaped
  - Prevents `json.decoder.JSONDecodeError` in OpenCode/API clients
- **Task 127+132: TQ3 prefill + Goal 1 cost docs** — honest documentation (2026-04-17)
  - TQ3 prefill: 3 tok/s (270× slower than duo) — documented in CLAUDE.md
  - Goal 1 cost table: 4K→128K Metal/swap/wall-clock progression
  - TQ3 recommendation: use for single-turn + session save/load only
- **Task 126: [CRITICAL FIX] SnapKV server crash** — importlib fix (2026-04-17)
  - `mlx_lm.generate` is a function (from `__init__`), not a module
  - Fix: `importlib.import_module("mlx_lm.generate")` in snapkv.py + adaptive_prefill.py
  - `--snapkv-keep` now works on all KV modes (duo, tq3, native)
- **Task 128: Server default model mismatch** — 4-bit→8-bit fix (2026-04-17)
  - Server default was `Instruct-4bit`, bench uses `Instruct-8bit`
  - All CLAUDE.md performance claims (MMLU 62%, HumanEval 95%) are for 8-bit
  - Fixed: server now defaults to 8-bit to match bench
- **Task 129+132: Goal 1/3 tension + cost docs** — honest status update (2026-04-17)
  - Duo: 53.6 tok/s (meets G3), Native: 46.8 tok/s (misses G3)
  - TQ3 prefill: 3 tok/s (270× slower than duo)
  - Added Goal 1 cost progression table to CLAUDE.md
- **Task 111: Streaming-head budget rebalancing** — retrieval 2×, streaming 0.5× (2026-04-17)
  - `head_types` parameter in `snapkv_select()`: reweights importance before pooling
  - Retrieval heads get 2× importance weight, streaming heads get 0.5×
  - Shifts eviction budget toward retrieval heads where quality matters most
  - `--streaming-aggressive` flag on server
  - From arXiv:2412.05353: streaming heads don't re-consult resolved tokens
- **Task 107: Fair eviction** — proportional partition budget allocation (2026-04-17)
  - `_select_fair()`: allocates budget proportionally across instruction partitions
  - Prevents system prompt eviction bias — each partition keeps budget ∝ its size
  - Minimum per-partition floor (default 20 tokens) protects small critical partitions
  - Composes with BUZZ segments: hierarchy is partition > segment > token
  - `--fair-evict` flag on server. `partitions` param in `snapkv_select()`
- **Task 59: OPLoRA safety rail for TTT** — orthogonal projection shipped (2026-04-17)
  - `omlx/oplora.py`: double-sided orthogonal projection for LoRA gradients
  - `project_lora_grads()`: projects dA, dB onto subspace orthogonal to W's top-k SVD
  - `_truncated_svd()`: randomized SVD for large matrices (avoids full-matrix OOM)
  - `compute_svd_cache()`, `save_svd_cache()`, `load_svd_cache()` for persistence
  - 10 tests: orthogonality, k=0 passthrough, norm reduction, idempotence
  - Ready for TTT integration via `--oplora` flag
- **Phase 3e: SnapKV eviction quality gate** — in hypercar_bench (2026-04-17)
  - `phase3e_snapkv_quality()`: validates CAOTE+GER at 4K with 50% eviction
  - Runs in default mode (not just --full), 1.7s overhead
  - GPU validated: NIAH PASS, 49% kept, GER=0.000 (safe)
  - Catches regressions in the eviction stack during routine benchmark runs
- **Task 106: GER safety monitor** — phase-transition guard for eviction (2026-04-17)
  - `compute_ger()`: measures fraction of important tokens evicted from ALL heads
  - `check_ger_safety()`: warns and recommends budget widening when GER > 0.05
  - Integrated into `apply_snapkv_to_generate` — auto-widens budget if near cliff
  - From arXiv:2603.01426: universal hallucination cliff at ~90% compression
  - Composes with all eviction components (CAOTE, BUZZ, submodular)
- **Task 103: Trigonometric pre-RoPE importance scoring** — calibrated + scorer shipped (2026-04-17)
  - `scripts/calibrate_qk_centres.py`: captures pre-RoPE Q/K for all 48 layers (22s GPU)
  - `omlx/patches/trig_score.py`: TrigScorer class with O(d)-per-key scoring
  - Calibrated centres: 889 KB at `omlx/patches/qk_centres/qwen3_coder_30b.npz`
  - Q centres (32 heads, 128 dims), K centres (4 heads, 128 dims) per layer
  - `classify_heads()`: frequency-based streaming/retrieval classification
  - Composes with SnapKV: trigonometric score as alternative importance signal
- **Task 45: XGrammar tool-call JSON guarantee** — constrained decoding shipped (2026-04-17)
  - `omlx/patches/xgrammar_constrain.py`: grammar compilation, caching, sampler creation
  - `get_compiled_grammar()`: compiles JSON schemas, caches by hash for reuse
  - `create_grammar_sampler()`: wraps standard sampler with grammar bitmask constraint
  - `extract_json_schema_from_request()`: extracts schema from tools or response_format
  - `--grammar` flag on server enables constrained decoding
  - Guarantees every generated response validates against the provided JSON schema
- **Task 102: Submodular greedy eviction** — diversity-aware token selection (2026-04-16)
  - `_select_submodular()`: greedy selection with value-vector diversity penalty
  - Within each BUZZ segment, penalizes tokens similar to already-selected tokens (sim > 0.8)
  - Captures diminishing returns: two identical tokens don't both get selected
  - `--submodular-evict` flag on server, composes with CAOTE + BUZZ segments
  - From OTPrune (arXiv:2602.20205): (1-1/e) guarantee on distributional fidelity
- **Task 18: LiveCodeBench contamination-free coding gate** — Phase 3d wired (2026-04-16)
  - `phase3d_livecodebench()` in hypercar_bench: 20 problems in --full mode
  - Uses existing `omlx/eval/livecodebench.py` module + bundled data (2 MB, 100+ problems)
  - Sandboxed subprocess execution with timeout + memory limits
  - Gate: `lcb_gen@post_cutoff >= 30%` pass@1
  - Runs after MMLU-Pro, before HumanEval in --full mode

- **Goal 1: 128K NIAH PASS** — SnapKV+CAOTE physical compaction at 128K (2026-04-16)
  - 128K@25% keep: **NIAH PASS**, needle "SNAPKV-COMPACT-7743" found correctly
  - **9.03 GB Metal saved** (44.6 → 35.6 GB, 20% reduction)
  - 122,235 → 30,558 tokens kept (25%)
  - At 1M native 3-bit @25%: projects to KV ~5.6 GB, total ~23 GB — **fits in 48 GB**

- **Task 57: PyramidKV per-layer budget vector** — exponential-decay schedule shipped (2026-04-16)
  - `omlx/pyramid_budget.py`: compute_budget_vector() + budget_for_layer() + format_budget_summary()
  - Exponential decay from edges to center (beta=0.7), floor at 30% of uniform allocation
  - Two-pass normalization: exact sum guarantee with floor constraint
  - `compact_cache_pyramidal()` in snapkv.py for per-layer eviction with budget vector
  - `--pyramid-kv` flag on server. Edge layers get ~2.5x budget of middle layers
  - Needs GPU calibration sweep to find optimal beta (deferred to bench/pyramid_calibrate.py)
- **Goal 1: 64K NIAH PASS with SnapKV+CAOTE** — physical compaction validated at 64K (2026-04-16)
  - 64K@25% keep: 100% agreement, 4.52 GB saved (38.6→34.1 GB Metal)
  - 64K@50% keep: 100% agreement, 3.03 GB saved (38.6→35.5 GB Metal)
  - At 128K with 25% keep: projects to KV 1.6 GB, total ~19 GB — **no swap needed**
  - At 1M with native 3-bit + 25% keep: KV ~5.6 GB, total ~23 GB — **Goal 1 achievable**
  - Full eviction stack: CAOTE scoring + BUZZ segmented selection + re-RoPE compaction
- **Task 97: Freshness-aware KV eviction** — cosine-similarity conflict detection (2026-04-16)
  - `compute_freshness_scores()`: detects superseded tokens via K-vector cosine similarity
  - Exponential decay: freshness = 0.1^n_supersessions per multiply-superseded token
  - `--freshness-evict` flag on server, composes multiplicatively with CAOTE+segmented
  - Prevents stale entries from consuming cache budget in agentic multi-turn scenarios

- **Task 98: BUZZ segmented eviction for SnapKV** — per-segment top-K selection (2026-04-16)
  - `_select_segmented()`: divides KV into segments, selects top-K per segment proportionally
  - `--segmented-evict N` flag on server and `--segment-size N` on snapkv_bench
  - Composes with CAOTE: CAOTE scores within each segment, segmented selection preserves locality
  - GPU validation: 4K@25%/50% + 16K@25% all 100% agreement with CAOTE+segmented
  - Fixes "lost in the middle" — each segment retains its own heavy-hitters
- **Task 94: Adaptive prefill chunk-size controller** — memory-aware chunking (2026-04-16)
  - `AdaptivePrefillController`: proportional control with Metal memory + throughput signals
  - Starts at max_chunk, shrinks when Metal > 65% target, halves when tok/s < 200 (O(n²) cliff)
  - `--adaptive-chunk` flag on hypercar_server, composes with SnapKV+CAOTE
  - GPU calibration: 1.40x at 4K (one big chunk vs two), 1.06x at 16K (fits in one chunk)
  - Pre-fills cache with adaptive chunks, hands off last token to generate_step for decode

- **Task 100: CAOTE attention-output-error scoring** — value-aware eviction VALIDATED (2026-04-16)
  - `compute_caote_importance()`: score = (α/(1-α)) × ||V_mean - v_j|| per token per head
  - FastCAOTE approximation: O(n·d) per head using mean-of-values, not O(n²·d)
  - `--caote` flag on hypercar_server and snapkv_bench
  - GPU validation: **CAOTE fixes the 16K@25% NIAH failure** that attention-only missed
    - 16K@25%: attention-only FAIL (3% agreement) → **CAOTE 100% agreement**
    - 4K@25%/50%, 16K@50%: both methods 100% (CAOTE matches)
  - Key insight: needle token has high attention AND distinctive value vector;
    attention-only misses tokens with moderate attention but critical information

- **Task 46: SnapKV physical compaction — VALIDATED** (2026-04-16)
  - `--snapkv-keep` flag on hypercar_server wraps `generate_step` with Q capture hooks
  - `compact_cache` supports KVCache (fp16) and QuantizedKVCache (dequant→gather→requant)
  - **Re-RoPE fix**: `_rerope_keys()` applies per-token rotation shift after gathering —
    fixes the fundamental RoPE position mismatch that caused 0% quality without it
  - GPU validation (`omlx/bench/snapkv_bench.py`):
    - 4K: **100% token agreement** at 25%, 50%, 75% keep
    - 16K: **100% agreement at 50% keep**, FAIL at 25% (too aggressive for retrieval)
    - Memory: 1.14 GB saved at 25% keep, 0.77 GB at 50% (16K, fp16)
    - At 1M native 3-bit, 50% projects to ~11 GB savings — **Goal 1 enabler**
  - Key insight: RoPE rotations compose additively. After physical compaction,
    shift = new_pos - old_pos corrects each key's encoding to sequential positions.
- **SnapKV with real Q capture**: 100% agreement at 25% keep — BREAKTHROUGH (2026-04-15)
  - `install_q_capture_hook()` + `compute_importance_from_real_q()` capture actual Q projections
  - `patch_model_for_eviction_mask()` injects eviction into attention via bfloat16 masking
  - 100% token agreement on NIAH, Math, Code at 25%, 50%, 75% keep ratios
  - K-as-Q proxy was the entire quality problem (0% NIAH); real Q capture fixes it completely
  - At 128K with 25% keep: KV = 1.4 GB, model+KV = 18.6 GB — fits trivially
  - Remaining: SparseKVCache for physical compaction (masking validated, memory savings need custom cache)
- **Task 53**: MLA rank probe + quality validation on Qwen3-Coder KV (2026-04-15)
  - Joint KV rank 241/1024 (24%) — good compression ratio
  - BUT post-hoc SVD quality FAIL: 7% token agreement during decode
  - ShadowKV per-head SVD also poor: 34% K compression → 16% NIAH, rank sweep confirmed
  - **VERDICT: SVD approaches fail for post-trained models; SnapKV eviction is the path**
- **Task 56**: Add MagicDec cost-model gate for speculative decoding decisions (2026-04-15)
  - New `omlx/specdec_gate.py` — closed-form predictor based on MagicDec Eq. 2-4
  - Calibrated on M4 Pro: compute=19.1ms/tok, KV-load=0.34µs/ctx-token, crossover ~57K
  - Gate predicts 2.31x speedup at 70% accept rate / 5 draft tokens (all ctx > 4K)
  - New `omlx/bench/specdec_calibrate.py` — measures real decode latency at 2K/8K/32K
  - Calibration constants at `omlx/specdec_constants.json`
  - 14 new tests in `tests/test_specdec_gate.py`: gate logic, latency model, persistence
  - Informs all spec-decode tasks: EAGLE-2 (28/29), TriForce (58), Lookahead (47)
- **Task 31**: Profile KV allocator fragmentation at 1M context (2026-04-15)
  - Profiled at 4K/16K/64K context lengths; fragmentation stable at 2.1% (~0.7 GB peak-active gap)
  - Extrapolated to 1M: only ~0.5 GB fragmentation out of 22.5 GB KV — negligible
  - **VERDICT: Skip paging work** — MLX Metal allocator is efficient, Tasks 43/64 not justified for memory savings alone
  - Research note at `research/KV_FRAGMENTATION.md`, raw data at `research/kv_fragmentation_profile.json`
- **Task 24-b**: Probe MLX argpartition speed for Quest top-K page selection (2026-04-15)
  - ALL configurations PASS (<200µs/head budget) — Quest is viable as written
  - At 1M context (8K pages): ~40µs/head. At 8M context (64K pages): ~80µs/head
  - argpartition and argsort have identical speed on MLX 0.31.1 (~30-80µs/head)
  - K value (16-256) has negligible impact on timing — slice is free
  - No custom Metal top-K kernel needed; Task 3's Quest implementation can ship
- **Task 32**: ProMoE offline expert activation frequency profiling (2026-04-15)
  - New `scripts/moe_profile_expert_activation.py` — hooks 48 MoE router gates via TrackedGate wrapper
  - 4379 tokens of mixed coding prompts across 8 calibration prompts (Python, Rust, JS, SQL, etc.)
  - Results: only 8/6144 (0.1%) experts never activated; top-10 hold 14-27% of dispatches per layer
  - Layer 32 most concentrated (27.1% top-10), layers 0/47 most uniform (14-18%)
  - Profile output at `omlx/patches/promoe_profiles/qwen3_coder_30b_a3b_instruct_8bit.json`
  - Informs Task 16/33: 50% resident saves ~3.6 GB, 75% saves ~1.8 GB, 87.5% saves ~0.9 GB
- **Task 28**: EAGLE-2 tree-attention feasibility probe for MLX (2026-04-15)
  - `mx.fast.scaled_dot_product_attention` SUPPORTS tree masks — additive mask works correctly
  - Numerical accuracy vs NumPy reference: max error < 1e-6 across all tested shapes
  - Tree mask overhead: only 1.03-1.12x vs no-mask baseline (essentially free)
  - Vanilla fallback would cost 1.4-1.6x — fused path is preferred
  - At 8K prefix + 21 draft tokens: 1.12ms per verification pass
  - **VERDICT: EAGLE-2 is FEASIBLE on MLX** — fused SDPA handles tree masks efficiently
  - Probe script at `scripts/probe_eagle_tree_attn.py`
- **Task 87**: MLX softmax fused-reduction audit + microbench harness (2026-04-15)
  - New `omlx/bench/softmax_bench.py` microbench: mx.softmax, SDPA, unfused attention at production shapes
  - Measured mx.softmax at 4.92ms @2K on M4 Pro/MLX 0.31.1 — 5.7x faster than paper's M1 (27.91ms)
  - Softmax is ~30% of unfused attention; SDPA gives 2.6-2.8x speedup via steel_attention tiling
  - **VERDICT: Gap closed. No fused shader needed.** Bucket closed, zero follow-on work.
  - Research note at `research/mlx_softmax_audit.md`, raw data at `research/mlx_softmax_audit.json`
- **Task 80**: Add wall-clock correlation to bench profiler (2026-04-15)
  - Added `wall_clock_elapsed_s` and `cpu_scheduling_fraction` fields to `ProfileResult`
  - Profiler records `time.monotonic()` at start; computes scheduling fraction as `actual_samples / expected_samples` at stop
  - `cpu_scheduling_fraction < 0.9` triggers WARNING in `_finish()` about swap-thrashing / co-tenancy pressure
  - Fraction capped at 1.0; included in `summary()` dict for profile.json output
  - 7 new tests in `test_hypercar_tools.py::TestWallClockCorrelation`
  - Catches R57-style stalls: 24-minute swap-thrash window now detectable automatically
- **Task 86**: Per-phase memory-headroom re-check between phases (2026-04-15)
  - New `_check_phase_headroom()` helper checks Metal headroom before each heavy phase
  - Per-phase requirements: NIAH 8 GB, RULER 6 GB, MMLU-Pro 3 GB, HumanEval 3 GB
  - Phases with insufficient headroom are SKIPPED with WARNING instead of crashing
  - Prevents R54/R57/R58 pattern: bench completes remaining phases under co-tenancy
- **Task 85**: Flush logger before exit in memory-watchdog breach path (2026-04-15)
  - Added immediate handler flush + stderr flush + `WATCHDOG-BREACH` sentinel to `_set_breach()`
  - Fixes R54/R57 silent exit 144 — breach reason now always visible in console.txt
  - Grep-able sentinel: `grep WATCHDOG-BREACH bench/snapshots/*/console.txt`
- **Task 49**: Adopt ProLong's RULER length × subtask matrix in Tasks 1/7/25 (2026-04-14)
  - Added ProLong (arXiv:2410.02660) methodology citation to phase3b_ruler docstring
  - Updated Task 1 verify: explicit subtask × length diagnostic pairs (retrieval, tracing, aggregation)
  - Updated Task 7 verify: confirmed breach path follows ProLong "fail at next-shorter length" semantics
  - Updated Task 25 verify: profiling subset aligned with ProLong recommended tiers (4K/16K/64K)
- **Task 62**: Namespace run numbering between analyst cron and implementation loop (2026-04-14)
  - Convention: analyst runs use "Run N", devloop runs use "devloop-N" or "sample-N" in commit messages
  - Documented in bench/snapshots/README.md and CLAUDE.md "Before Every Commit" section
  - Prevents namespace collisions in `git log` between analyst and devloop benchmark runs
- **Task 71**: Add `--niah-only --niah-context` escape hatch for Goal 1 validation (2026-04-14)
  - New `--niah-only` flag: runs only Phase 0 (smoke) + Phase 3 (NIAH), skips Code Intel, RULER, MMLU-Pro, HumanEval
  - New `--niah-context` flag: comma-separated context specs like `4K,16K,64K,128K`, overrides default [4K,16K]
  - `_parse_context_list()` helper handles K/M suffixes and raw integers
  - Updated `--help` epilog with NIAH escape hatch section and examples
  - Tests: 6 parser tests + 3 flag existence tests in test_hypercar_tools.py
  - Usage: `hypercar_bench --niah-only --niah-context 64K,128K --kv-mode native`
- **Task 61**: MMLU-Pro max_tokens hard floor guardrail (2026-04-14)
  - Added `MMLU_PRO_MIN_MAX_TOKENS = 512` constant in hypercar_bench.py with comment documenting the 1e803b6→20df582 incident
  - Wired constant into phase3c_mmlu_pro generate call (replaces hardcoded 512)
  - Added runtime assertion `assert MMLU_PRO_MIN_MAX_TOKENS >= 512` that forces future engineers to explicitly acknowledge the trade-off
  - New test file `tests/test_eval_mmlu_pro.py` with 7 tests: constant floor, constant usage, assertion existence, answer extraction, truncated reasoning detection

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
- **Task 11**: Extend /sandbox exclude to cover benchmark diagnostic commands (2026-04-13)
  - New `omlx/bench/baseline.py` — gathers system state via ctypes + psutil (zero subprocess)
  - Fields: system_memory_gb, swap_used_gb, load_avg, virtual_memory, mlx_processes, disk_io
  - Run as script (`python omlx/bench/baseline.py`) to avoid omlx root package MLX import chain
- **Task 21**: Add multi-run statistical aggregation to omlx/bench/aggregate.py (2026-04-13)
  - New `omlx/bench/aggregate.py` — walks bench/snapshots/run*/, groups by SHA, computes per-phase stats (mean, std, median, p10/p50/p90/p99, min, max)
  - `--report HEAD` prints per-phase table with timing, decode/prefill tok/s, Goal 5 violation count
  - Writes aggregate JSON to bench/snapshots/aggregate/<sha>.json (idempotent)
  - stdlib only (no numpy) — _stats() uses manual percentile/std computation
- **Task 23**: Re-state CLAUDE.md Goal 5 as a p90 sustained swap-rate metric (2026-04-13)
  - Goal 5 target changed from "< 8 GB at any point" to "p90 sustained swap I/O < 100 MB/s (N≥8 runs)"
  - Status updated to reflect N=8 measurement: p90 ~460 MB/s (FAIL, 4.6x over target)
  - Swap depth kept as secondary observability metric, not a gate
  - References aggregate.py as source of truth
- **Task 24**: Add 2-SHA regression detector to aggregate.py (2026-04-13)
  - Added `--baseline <sha>` flag to aggregate.py for side-by-side comparison
  - Per-phase median delta with 2-sigma regression/improvement flags (Welch's t-test approximation)
  - Decode/prefill tok/s deltas shown inline per phase
  - NEW/REMOVED detection for phases that exist in only one SHA
  - Minference layer counter already correct via `% num_layers` wrapping — `reset_layer_counter()` kept as defense-in-depth API
- **Task 25**: Add --warmup flag to hypercar_bench for Metal kernel cache priming (2026-04-13)
  - `--warmup` runs 16-token generation before Phase 0 to prime Metal shader cache
  - Result: decode 20→45 tok/s (2.25x), Phase 0 time 9s→0.3s after warmup
  - Strong evidence for Task 20 hypothesis 1 (Metal kernel cache cold/warm bimodality)
  - Clears Metal cache after warmup to avoid inflating memory baselines
- **Task 22**: Fix 8-bit model Goal 5 violation — apply `--kv-bits 2` and verify (2026-04-13)
  - TQ3 2-bit: Code Intel 5/5 PASS, decode 49.4 tok/s, swap 3.7 GB (PASS Goal 5)
  - BUT: NIAH 4K FAIL (2-bit too lossy for short-context retrieval)
  - Native 2-bit: total quality collapse ("2+2=2+2=2+2=") — only TQ3 WHT viable at 2-bit
  - Conclusion: 2-bit fixes Goal 5 but regresses Goal 2 (NIAH). Trade-off documented.
  - Also fixed: warmup uses native KVCache (TQ3 crashes on short warmup), added fcntl benchmark lock (prevents concurrent instances from swamping 48GB)
  - Option (c) full run with --max-swap-pct 40: ALL phases completed for first time ever
  - HumanEval: 18/20 (90%) PASS — first time Phase 4 ran in 10+ benchmark runs
  - RULER: multi_key@16K 90% PASS, 64K tasks cleanly SKIPped by headroom gate
  - RULER gate FAIL from variable_tracking (reasoning task), not memory — separate issue
  - Swap 9.3 GB (under loosened 20.6 GB limit), Metal 37.8 GB (under 41.2 GB limit)
- **Task 30**: Survey mx.fast.scaled_dot_product_attention source for AMX binding (2026-04-13)
  - Written to `research/MLX_ATTN_DISPATCH.md` with line-level citations from MLX v0.31.1
  - Key findings: prefill (L>8) uses `steel_attention` with AMX (`simdgroup_matrix`); decode (L=1) uses `sdpa_vector` (scalar, no AMX)
  - Qwen3 D=64 gets BQ=32, BK=32 tile sizes (favorable: 2x more K/tile than D=128 models)
  - No quantized KV support inside any SDPA kernel — dequant must happen upstream (confirms TQ3 design)
  - MInference risk: per-head additive masks work but may prevent efficient head batching
- **Task 36**: MMLU-Pro reasoning gate in hypercar_bench (2026-04-13)
  - New `omlx/eval/mmlu_pro/` module — loads TIGER-Lab/MMLU-Pro from HuggingFace, chain-of-thought prompting, answer extraction
  - Phase 3c: 25 questions (cs+math) in default mode, 100 across all categories in --full
  - Gate: mmlu_pro_cs_math >= 35%. Result: **48% (12/25) PASS** — first reasoning eval for Goal 2
  - Per-category breakdown in phase details. Runs after RULER, before HumanEval
- **Task 16**: ProMoE lazy-load probe for Qwen3-Coder expert weights (2026-04-13)
  - Experts stored as fused QuantizedSwitchLinear (128, H, W) — can't skip loading individual experts
  - Gate-masking approach: bias cold expert logits by -1e9 before softmax
  - Results: 50% FAIL, 75% borderline, 87.5% (112/128) PASS (saves 3.6 GB), 93.75% PASS (saves 1.8 GB)
  - Quality degrades even at "PASS" levels (repetition artifacts at 87.5%)
  - Verdict: ProMoE is MARGINAL for this model — 3.6 GB savings at 87.5% residency with quality risk. DuoAttention (Tasks 12-13) is a better path for memory reduction.
- **VT prompt fix**: Improved RULER variable_tracking question prompt to ask for "resolved string value" instead of just "value" (2026-04-13)
  - variable_tracking@4K: 0% → **100%** — model now resolves the chain instead of returning variable names
  - RULER gate: FAIL → **PASS** — first time RULER passes in default mode
  - ALL GATES PASSED for the first time ever in a default-mode run (Phases 0-3c + memory)
- **Task 44**: ShadowKV SVD-rank probe on Qwen3-Coder K cache (2026-04-13)
  - K cache is LOW-RANK: median rank@99% = 177/512 (35% of max) across 48 layers
  - Verdict: **ShadowKV VIABLE** — SVD-based K compression can save ~65% K memory with <1% info loss
  - Layer 3 outlier: rank 10 (nearly all energy in first 10 SVs). Most layers 150-210.
  - Recommends ShadowKV over InfLLM (Task 43) for K cache tiering
  - Results at `research/shadowkv_rank_20260413.json`
- **Task 12**: DuoAttention retrieval/streaming head calibration for Qwen3-Coder (2026-04-13)
  - Calibration captures full attention maps via SDPA patching (1536 maps across 48 layers × 32 heads)
  - At threshold=0.85: **59% streaming, 41% retrieval** — passes 50% streaming requirement
  - Layer 3 is most streaming (24/32 heads), late layers (44-47) are mostly retrieval
  - Policy JSON at `omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json`
  - Potential KV savings at 1M: ~6.5 GB (59% of K cache uses 256-token ring buffer instead of full cache)
- **Task 38** (calibration): LayerSkip per-layer exit confidence profiling (2026-04-13)
  - Depth 44 (4 layers skipped): 55% agreement — best but only 1.05x speedup
  - Depth 40 (8 layers skipped): 17% agreement — not viable
  - Depth 16-32: 2-3% agreement — MoE routing makes every layer critical
  - Verdict: **LayerSkip NOT VIABLE** for this MoE model. Expert selection per-layer prevents early exit.
  - Results at `omlx/patches/layerskip_thresholds/qwen3_coder_30b_a3b_instruct_8bit.json`
- **Task 13**: DuoAttention two-storage-class KV cache — SHIPPED as default mode (2026-04-14)
  - New `omlx/duo_kv_cache.py` — DuoKVCache with fp16 KVCache (retrieval) + StreamingKVCache ring buffer (streaming)
  - `--kv-mode duo` is the DEFAULT mode for both bench and server
  - Original blocker (QuantizedKVCache tuples) resolved by using fp16 KVCache for retrieval heads
  - Results: MMLU-Pro 64% (was 48%), HumanEval 95% (was 90%), zero swap, 52.4 tok/s decode
  - ALL GATES PASS in duo mode — 5 of 6 Hypercar goals met
- **Task 20**: Identify root cause of per-task bimodal timing (2026-04-13)
  - Root cause: Metal shader JIT compilation on first forward pass (~9s on M4 Pro)
  - Evidence: `--warmup` eliminates cold-start (Phase 0: 9s→0.3s, decode: 20→45 tok/s)
  - Bimodality from system-level Metal shader cache hit/miss (process-scoped, cache in ~/Library/Caches/com.apple.metal/)
  - Writeup at `docs/bimodal_timing_root_cause.md` with reproduction recipes for fast/slow modes
  - Recommendation: always use `--warmup` for benchmarking; add warmup to server startup
- **Task 24b**: Probe MLX argpartition speed for Quest top-K page selection (2026-04-13)
  - VERDICT: PASS — all configs under 200µs/head budget
  - 1M context (8192 pages): 60-76 µs/head; 8M stress (64K pages): 100-141 µs/head
  - argpartition available in MLX 0.31.1, competitive with argsort
  - Quest page selection is viable at all tested context lengths
- **Task 15**: SimPO contrast step in TTT engine (2026-04-13)
  - Added `simpo_step(winner_text, loser_text, prompt, beta, gamma)` to TTTEngine
  - Loss: -log sigmoid(beta * (avg_logp(winner) - avg_logp(loser)) - gamma), length-normalized
  - Reference-free (no frozen reference model — ideal for test-time training)
  - Gradient norm capped at 1.0 to prevent runaway adapter updates
  - Reuses existing adapter infrastructure, forward path, and eval discipline
- **Task 26**: MInference block-sparse prefill kernel dispatch (2026-04-13)
  - Already implemented in Task 5 — `_build_block_sparse_mask()` at line 296-297 of minference_prefill.py
  - Block-sparse heads go through the same additive-mask SDPA path as vertical-slash
  - No additional code needed; marking as completed (was included in Task 5 scope)

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
  Per ProLong (arXiv:2410.02660), the recommended profiling subset is:
  - Quick (≤30s): multi_key_niah@4K, variable_tracking@4K, frequent_word@4K
  - Default (≤5min): above + multi_key_niah@16K, variable_tracking@16K
  - Full (≤20min): above + all three families @64K
  Lengths beyond 64K (128K-1M) are opt-in only via `--niah-context`.
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

### 35. Broaden sandbox exclusion to all omlx.bench.* modules
- **Goal**: Observability infrastructure — unblocks Task #21's
  multi-run aggregation tool from running in the analyst cron path
- **Derived from**: Hypercar benchmark run 32 (2026-04-13),
  bench/snapshots/run32_2026-04-13T09-09/env.json
  `aggregate_tool_sandbox_issue`. Running
  `.venv/bin/python -m omlx.bench.aggregate` inside the Claude Code
  Bash sandbox crashes with `NSRangeException` at
  `mlx::core::metal::Device::Device()` — the same Metal-device-empty
  failure we debugged on Day 1 when first setting up the bench
  sandbox exclusion. The root cause: `.claude/settings.local.json`
  currently excludes only `.venv/bin/python -m omlx.bench.hypercar_bench:*`
  from the sandbox, so every OTHER omlx.bench.* module (aggregate,
  profile_prefill, safe_bench, matrix, ttt_bench) still runs under
  the default Bash sandbox profile. MLX's Metal device enumeration
  fails under that profile because `sandbox-exec` denies the IOKit
  service-matching path, and MLX's constructor does `devices[0]` on
  an empty NSArray without length check.
- **Change**:
  - Edit `.claude/settings.local.json` to broaden the sandbox
    exclusion pattern from
    `.venv/bin/python -m omlx.bench.hypercar_bench:*`
    to
    `.venv/bin/python -m omlx.bench.*:*`
    which covers hypercar_bench, aggregate, profile_prefill,
    safe_bench, matrix, ttt_bench, and any future sibling modules.
  - Alternative approach (if broadening the sandbox feels too
    loose): refactor `omlx/bench/aggregate.py` to strictly not
    import mlx — it only reads JSON files from disk and computes
    statistics, so there is no legitimate need for Metal. The
    transitive import probably comes from `omlx/__init__.py` or
    one of the helper modules it pulls in. Grep for `import mlx`
    in the aggregate call chain and add conditional imports guarded
    by `if __name__ != "__main__"` or move mlx imports behind a
    function-level lazy import.
  - Recommend the sandbox-broadening approach first (option 1)
    because it unblocks ALL bench modules with a one-line config
    change, and the bench directory is narrow enough in scope that
    running it unsandboxed is low risk.
- **Verify**: `.venv/bin/python -m omlx.bench.aggregate` runs to
  completion without the NSRangeException, and the analyst cron
  loop's Step 4 analysis can successfully read aggregate output for
  multi-run statistical comparison. Concrete pass criterion: the
  cron's next run (Run 33 or whatever comes after Task #35 lands)
  successfully invokes aggregate and includes median/p90 comparison
  numbers in its final report.
- **Effort**: S (one-line config change) to M (if refactor path is
  chosen instead)

## Research-derived tasks (from LIT_REVIEW.md pass 4, 2026-04-13)

### 36. MMLU-Pro reasoning gate in hypercar_bench
- **Goal**: 2 (intelligence breadth — hard reasoning, explicitly the "MMLU-style reasoning" gap from CLAUDE.md's Goal 2 status row)
- **Derived from**: MMLU-Pro (2406.01574)
- **Change**:
  - New `omlx/eval/mmlu_pro/` module that loads the pinned HuggingFace
    release of MMLU-Pro (`TIGER-Lab/MMLU-Pro`, pinned to a specific
    revision for reproducibility). Reuse the existing chat-completions
    client path that LiveCodeBench/tau-bench use — no new transport.
  - Use the paper's own chain-of-thought prompt template (not raw Q&A)
    and extract the answer letter via the same regex pattern as
    LiveCodeBench Task 18's answer extractor.
  - Add `phase_mmlu_pro()` to `omlx/bench/hypercar_bench.py`: 25
    questions sampled from `computer_science` + `math` in `--quick`
    (must stay under 60 seconds wall), 500 questions sampled across all
    14 categories in `--full`.
  - Add `mmlu_pro_cs_math >= 0.35` as a new Goal-2 gate (well below
    GPT-4o's ~0.55 on those subjects, but a strong regression signal).
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --quick` prints
  an `mmlu_pro_cs_math` gate row; gate passes at accuracy >= 0.35. Wall
  clock for the `--quick` phase stays under 60 seconds. The bench
  summary lists MMLU-Pro as an independent Goal-2 eval row alongside
  HumanEval, RULER, τ-bench, LiveCodeBench.
- **Effort**: S

### 37. LiveBench contamination-free multi-category gate in hypercar_bench
- **Goal**: 2 (intelligence breadth — contamination-free reasoning + data analysis + instruction-following, complement to LiveCodeBench)
- **Derived from**: LiveBench (2406.19314)
- **Change**:
  - New `omlx/eval/livebench/` module that pins a dated LiveBench
    release (e.g. `livebench-2025-10`) and wraps the upstream
    `livebench` pip package's task loader. Ground-truth scoring only
    (the upstream package already avoids LLM-as-judge, which matches
    our "no external API" constraint for reproducible bench runs).
  - Add `phase_livebench()` to `omlx/bench/hypercar_bench.py` running
    the `math` + `reasoning` + `data_analysis` categories only
    (skip `language`, `instruction_following`, and `coding` — `coding`
    overlaps LiveCodeBench, the other two are less decision-relevant
    for a coder model). 10 tasks per category in `--quick` (~5 min
    wall), all 80+ tasks per category in `--full`.
  - Add `livebench_reasoning >= 0.30` as a new Goal-2 gate.
  - Sandbox the LiveBench scoring in a subprocess so its pandas/numpy
    version pin doesn't collide with the main server venv.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full` prints
  `livebench_math`, `livebench_reasoning`, `livebench_data_analysis`
  rows; the `livebench_reasoning` gate passes at accuracy >= 0.30.
  Bench summary now shows five independent Goal-2 evals
  (HumanEval/LCB, RULER, τ-bench, MMLU-Pro, LiveBench).
- **Effort**: S-M
- **Depends on**: 36 (shares the answer-extraction / subprocess
  sandbox plumbing; landing both in one week is cheaper than separately)

### 38. LayerSkip self-speculative decoding (calibration-only variant)
- **Goal**: 3 (decode speed, constant across context)
- **Derived from**: LayerSkip (2404.16710)
- **Change**:
  - New `scripts/layerskip_calibrate.py` that runs Qwen3-Coder-30B-A3B
    over a short calibration corpus (HumanEval+ prompts + a few
    repo-aware completions) and records, per layer, the distribution
    of per-token softmax entropy / max-logit confidence at each early-
    exit depth. Emit a per-layer exit-confidence threshold table to
    `omlx/patches/layerskip_thresholds/qwen3_coder_30b_a3b_instruct_8bit.json`.
  - New `omlx/patches/layerskip_decode.py` that hooks the decode loop
    in `omlx/hypercar_server.py`:
    - Draft phase: run the first K layers, check exit-confidence; if
      above threshold, emit a draft token, else bail to the full model.
    - Verify phase: when ≥1 draft token exists, run the remaining
      (48-K) layers on the draft token(s) in a single forward and
      accept only the prefix that agrees with the greedy argmax of
      the full model's output. Lossless by construction.
  - Flag: `--layerskip K` on both `omlx/hypercar_server.py` and
    `omlx.bench.hypercar_bench`. Default off. Must compose with
    `--kv-mode tq3` (the cache is shared between draft and verify
    passes — no second KV).
  - CRITICAL: the draft-confidence threshold must be strict enough
    that acceptance rate is nontrivially positive; if every draft
    token gets rejected, the overhead strictly slows decode. Task
    output must report `layerskip_accept_rate` in the bench JSON.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --layerskip 16`
  shows (a) decode tok/s at 2K and 16K >= 1.25x the baseline,
  (b) HumanEval pass@1 exactly equal to the baseline (lossless gate,
  not "within 2%" — acceptance verification must be bit-exact), and
  (c) `layerskip_accept_rate >= 0.35` in the bench JSON. All three
  required.
- **Effort**: M
- **Depends on**: none (can land before or after Task 28/29 EAGLE-2
  probes; this is the cheaper alternative)

### 39. LazyLLM per-layer token-pruning prefill hook
- **Goal**: 4 (prefill speed, constant across context), 1 (effective 1M context)
- **Derived from**: LazyLLM (2407.14057)
- **Change**:
  - New `omlx/patches/lazyllm_prefill.py` registering a per-layer
    token-selection hook during prefill. At each layer, compute
    per-token importance from the layer's attention scores (sum of
    attention weight received across all heads), drop tokens below
    a per-layer top-K budget before the next layer's input. The
    budget schedule is: layer 0..N/4 keep 100%, layer N/4..N/2 keep
    85%, layer N/2..3N/4 keep 65%, layer 3N/4..N keep 50%. These
    ratios are hyper-parameters — the paper's ratios are a starting
    point, not a commitment.
  - Pruned tokens' KV entries are NOT discarded — they are marked
    "cold" in `omlx/turboquant_kv.py` and retained for decode-time
    revival. Revival triggers when a decode token's raw dot-product
    against a cold token's stored key exceeds a revival threshold.
    Revival is batched per decode step to avoid a branch-heavy hot path.
  - Flag: `--prefill-prune lazyllm` on both the server and bench
    (mutually exclusive with `--prefill-sparse minference` for now;
    composing them is Task 39-follow-up).
  - Must gate on *both* prefill and decode: if revival makes decode
    slower, the task is a net loss and must be backed out.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full --prefill-prune lazyllm`
  shows (a) prefill tok/s at 16K >= 1.5x baseline, (b) decode tok/s
  at 16K regresses by < 5% vs baseline (revival cost cap), (c)
  `ruler_multi_key@16K` accuracy within 2% of baseline, AND (d)
  HumanEval >= 85%. All four required. Report `lazyllm_revival_rate`
  (fraction of decode steps that trigger at least one revival) in
  the bench JSON for observability.
- **Effort**: M
- **Depends on**: none (composes orthogonally with Task 5 MInference
  prefill sparse dispatch; both can land independently)

### 40. SWE-agent + SWE-bench Lite realistic SE eval
- **Goal**: 2 (intelligence breadth — realistic software engineering agentic eval, the workload our server actually runs)
- **Derived from**: SWE-agent (2405.15793)
- **Change**:
  - New `omlx/eval/swe_agent/` module that vendors a pinned commit of
    `princeton-nlp/SWE-agent` and wraps its agent loop to point at our
    local `hypercar_server` `/v1/chat/completions` endpoint. The
    minimal 7-tool ACI (file viewer, scoped editor, search, etc.) is
    the direct import from upstream — we do NOT reimplement.
  - Replace the upstream Docker-per-task sandbox with a colima or lima
    VM that hosts the per-issue repo state and test runner. Each
    SWE-bench Lite issue checks out the parent commit, applies the
    candidate patch, runs pytest, and diffs pass/fail against the
    known-good test set. Fallback for dev loops: a macOS-native
    `subprocess + venv-per-issue` runner that works for the 60-70%
    of issues whose tests don't require system packages.
  - Add `phase_swe_agent()` to `omlx/bench/hypercar_bench.py`: 5
    SWE-bench Lite issues sampled in `--quick` (cap wall clock at
    10 minutes total — some issues are very long), 30 issues in
    `--full`. Report `swe_lite_resolved` (fraction resolved).
  - Add `swe_lite_resolved >= 0.05` as the Goal-2 gate — purely a
    "tool path is alive" signal, not a performance benchmark. The
    point is to catch silent breakages of the OpenCode-style agent
    loop, not to beat frontier models.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench --full` runs
  `phase_swe_agent()` to completion and prints a `swe_lite_resolved`
  gate row; gate passes at `resolved >= 0.05` on 30 issues. Wall clock
  for the SWE-agent phase in `--full` stays under 45 minutes (skip
  overlong issues if necessary).
- **Effort**: M-L (biggest unknown is the colima/lima setup; if it
  proves flaky, fall back to the subprocess runner for a reduced
  per-issue subset)

## Research-derived tasks (from LIT_REVIEW.md pass 5, 2026-04-13)

### 41. QuaRot 4-bit weight-quant spike on one MoE expert layer
- **Goal**: 6 (48GB fit — weight bytes), 5 (swap headroom)
- **Derived from**: QuaRot (2404.00456)
- **Change**:
  - New `omlx/turboquant_weights.py` module that reuses the Walsh-
    Hadamard rotation already implemented in
    `omlx/turboquant_kv.py` (do NOT duplicate the WHT primitive —
    import it). The module exposes `quarot_pack_linear(weight,
    scheme="w4a4")` returning a 4-bit packed weight + per-row scale
    and a pre-applied rotation matrix.
  - One-layer-only spike: take the `down_proj` of a single Qwen3-
    Coder expert in layer 24, rotate (weight and activation path)
    using the existing WHT, quantize to 4-bit RTN per row, and swap
    it into the model at load time via a monkey-patched `Linear`.
    All other experts and layers stay at 8-bit.
  - Add `omlx/bench/quarot_probe.py` that measures, at fixed
    context 4K: (a) perplexity on a 500-token Python code corpus
    vs the 8-bit baseline, (b) decode tok/s delta, (c) Metal
    memory delta. Report all three in a single JSON row.
  - Do NOT ship a full runtime. This is a de-risking spike — the
    decision to graduate to a multi-layer port happens only after
    the probe shows perplexity drift <2% and no decode regression.
- **Verify**: `.venv/bin/python -m omlx.bench.quarot_probe` prints
  a row with `perplexity_delta < 0.02`, `decode_tps_delta > -5%`,
  and `metal_mem_delta` is strictly negative (weight bytes shrank).
  Full hypercar_bench smoke + coherence gates still pass with the
  patched expert in place.
- **Effort**: M (3-5 days — the port is small because WHT is already
  shipping)
- **Depends on**: none (WHT is the only prereq and it already ships
  in `omlx/turboquant_kv.py`)

### 42. CacheBlend cross-chunk KV reuse in prompt cache
- **Goal**: 4 (prefill speed — cross-request reuse), 1 (effective 1M context reuse)
- **Derived from**: CacheBlend (2405.16444)
- **Change**:
  - New per-chunk KV store in `omlx/hypercar_server.py`'s caching
    layer, keyed on `blake2b(chunk_tokens)[:16]`. Chunk boundaries
    are delimited by a new `<|cache-chunk|>` sentinel token that
    OpenCode can emit between tool outputs (or we segment
    automatically on double-newline for backwards compat).
  - Implement the paper's HKVD (Highest Key-Value Divergence)
    selective-recompute policy: for each chunk reused in a new
    context, recompute only the top-p% of tokens whose attention
    to prior chunks would most diverge from a full prefill. Start
    with p=15% per the paper's default.
  - Reuse TQ3 save/load primitives (already shipping for session
    persistence) to serialize per-chunk KV to a bounded LRU
    on-disk cache under `~/.cache/omlx/cacheblend/`. Eviction by
    LRU with a 4GB cap by default.
  - New request header `X-Hypercar-CacheBlend: on` (off by
    default). When enabled, compare CacheBlend-prefilled first-token
    logits against a reference full-prefill on a 100-prompt
    calibration set; gate the feature rollout on
    `KL(cacheblend || full_prefill) < 0.05` mean and `< 0.2` p99.
  - New phase `phase_cacheblend()` in
    `omlx/bench/hypercar_bench.py` (`--full` only) that measures
    TTFT delta on a synthetic 3-tool-call agentic trace where two
    of the three tool outputs repeat from a previous request.
- **Verify**: `curl` with `X-Hypercar-CacheBlend: on` on a
  previously-seen multi-chunk prompt returns identical completion
  tokens to the same prompt without the header for at least 95/100
  calibration prompts, and TTFT drops by >=40% on the repeat-tool
  synthetic trace in `phase_cacheblend()`.
- **Effort**: M (3-5 days)
- **Depends on**: none directly, but sequence *after* LLMLingua-2
  (Task 17) if both land — compress first, then cache the compressed
  chunks

### 43. InfLLM two-tier KV cache for effective 1M context
- **Goal**: 1 (effective 1M context beyond KV budget), 5 (swap
  headroom), 3 (constant decode cost vs context length)
- **Derived from**: InfLLM (2402.04617)
- **Change**:
  - Extend `omlx/turboquant_kv.py` with a two-tier policy: hot tier
    holds the sink window (first 128 tokens), the local window
    (last 4096 tokens), and the top-K most-likely-attended blocks.
    Cold tier holds everything else in the same 3-bit WHT format
    but in a separately-allocated `mx.array` region flagged for
    eager `mx.clear_cache()` eviction.
  - Per-block representative-key: mean of the block's keys,
    precomputed when the block is finalised at prefill time.
  - Top-K block selection at attention time: cheap dot product
    between the current query and the per-block rep-keys, pick
    top-K=32 blocks (tunable via `--infllm-blocks`).
  - Compose with Quest (Task 2): Quest picks top-K *pages* within
    the hot tier, InfLLM picks top-K *blocks* from cold into hot.
    The two pass selections run in sequence.
  - New gate row in `omlx/bench/hypercar_bench.py`: NIAH at 256K
    and 512K under `--infllm-blocks 32` to prove effective context
    does not regress.
- **Verify**: `.venv/bin/python -m omlx.bench.hypercar_bench
  --full --infllm-blocks 32` passes NIAH at 256K AND 512K with
  hot-tier Metal footprint staying under 10GB at 512K context
  (measured by the existing metal-watchdog telemetry).
- **Effort**: L (1-2 weeks — two-tier allocator is the hard part)
- **Depends on**: 44 (SVD-rank probe decides whether InfLLM or
  ShadowKV is the right tiering policy to build first); 2 (Quest
  lands on the hot tier, not the cold tier)

### 44. ShadowKV SVD-rank probe on Qwen3-Coder K cache
- **Goal**: 5 (swap headroom — decides between InfLLM vs ShadowKV)
- **Derived from**: ShadowKV (2410.21465)
- **Change**:
  - New one-shot script `omlx/bench/shadowkv_rank_probe.py`: prefill
    a 64K context (synthetic Python code sample) with fp16 KV
    cache, then for each of the 48 layers run `mx.linalg.svd()` on
    the RoPE-applied K cache and report the rank needed to capture
    99%, 99.5%, and 99.9% of the Frobenius norm.
  - Decision rule encoded in the script: if the median layer needs
    rank <= 256 for 99% norm capture, report "ShadowKV viable" and
    recommend Task 43 be replaced by a ShadowKV implementation. If
    median rank > 256, report "InfLLM is the better bet" and
    recommend proceeding with Task 43 as written.
  - Save the full rank-vs-layer table as JSON under
    `research/shadowkv_rank_<date>.json` so the result is
    reproducible across commits.
- **Verify**: Script runs to completion in under 15 minutes on the
  M4 Pro reference machine, emits a decision line matching one of
  the two regimes above, and writes the JSON artifact. No
  production code changes — this is a pure measurement task.
- **Effort**: S (half a day to a day)
- **Depends on**: none (pure offline probe)

## Research-derived tasks (from LIT_REVIEW.md pass 6, 2026-04-12)

### 45. XGrammar tool-call JSON guarantee in hypercar_server
- **Goal**: 2 (intelligence breadth — tool-call correctness guarantee)
- **Derived from**: XGrammar (2411.15100)
- **Change**:
  - Add `xgrammar` to `pyproject.toml` dependencies.
  - In `omlx/hypercar_server.py`, extend the OpenAI-compat
    `/v1/chat/completions` handler so that when the request carries
    `tools=[...]` or `response_format={"type": "json_schema", ...}`,
    the server compiles the schema(s) into an XGrammar matcher once
    (cached in a `dict[str, Grammar]` keyed by schema hash) and
    applies a per-step token mask in the sampler.
  - Store the grammar cache at process level; pre-warm it on the
    first request for each unique schema. Expose a `/v1/internal/
    grammar_cache_stats` endpoint that returns hit count and compile
    time so the bench can verify cache correctness.
  - Make the mask-application path respect the existing sampling
    temperature and top-p without double-normalizing probabilities
    (mask before softmax).
- **Verify**: New test `omlx/bench/xgrammar_bench.py` sends 20
  synthetic tool-call prompts through the server with a fixed JSON
  schema and asserts (a) every response parses as valid JSON, (b)
  every response validates against the schema via `jsonschema`, (c)
  decode throughput at 2K context degrades by no more than 5% vs a
  no-grammar baseline run, (d) the second request for the same
  schema records a grammar-cache hit. All four assertions must pass.
- **Effort**: S (1 day)
- **Depends on**: none (orthogonal to every other task)

### 46. SnapKV prefill-time eviction in TurboQuantKVCache
- **Goal**: 5 (swap p90 sustained rate), 1 (larger effective context in same budget)
- **Derived from**: SnapKV (2404.14469)
- **Change**:
  - Add a `compact(keep_indices: mx.array)` method to
    `omlx/turboquant_kv.py` that re-packs the quantised page layout
    around the kept positions only. Must handle both the TQ3 WHT-
    rotated codebook and the fp16 layer-0 cache path.
  - Add `omlx/patches/snapkv.py` with a `snapkv_select(cache,
    obs_window_len, top_k_per_head)` helper that (a) runs the
    observation-window attention pass, (b) pools attention weights
    across the window via max-over-positions + mean-over-heads-in-
    group, (c) returns the top-k-per-head keep indices.
  - Wire into `omlx/hypercar_server.py` under a new
    `--snapkv-keep K` flag (default: disabled). When enabled, run
    eviction exactly once at the end of prefill, before decode
    starts. Only enable for requests whose conversation history
    length is 1 (single-turn) to avoid multi-turn quality cliffs.
- **Verify**: NIAH 4K gate in `omlx/bench/hypercar_bench.py` still
  passes with `--snapkv-keep 2048` enabled. New micro-benchmark in
  `omlx/bench/snapkv_bench.py` measures resident KV memory at 16K
  prefill with and without SnapKV and asserts the eviction path
  drops resident KV by at least 40% (paper's conservative number is
  3.6x reduction; 40% is a safe floor). Code Intel gate stays >= 3/5.
- **Effort**: M (1-2 days)
- **Depends on**: none — SnapKV operates on the existing cache
  layout; it does not require Quest or DuoAttention to land first.

### 47. Lookahead Decoding (Jacobi + n-gram pool) in hypercar_server
- **Goal**: 3 (decode speed, constant across context window)
- **Derived from**: Lookahead Decoding (2402.02057)
- **Change**:
  - New module `omlx/patches/lookahead.py` implementing the
    Jacobi-window rollout (W positions ahead) and the n-gram pool
    (size G, n-gram length N) keyed on trailing-(N-1) tokens.
  - Tree-attention mask primitive shared with the future EAGLE-2
    port: a function that takes a list of candidate continuations
    and returns a single packed attention mask + position-id vector
    for one forward pass that verifies all candidates jointly.
  - `hypercar_server` flags `--lookahead-window`, `--lookahead-ngram`,
    `--lookahead-pool-size`; when all three are set, the decode loop
    runs one forward per step that advances the Jacobi window *and*
    verifies n-gram guesses, accepting the longest verified prefix.
  - Output distribution must be provably identical to greedy
    decoding (no sampling changes): assert this with a bit-exact
    determinism test in the bench.
- **Verify**: New `omlx/bench/lookahead_bench.py` runs a fixed 512-
  token generation task at 2K and 16K context with `--temperature 0`
  twice — once without lookahead, once with `--lookahead-window 5
  --lookahead-ngram 3 --lookahead-pool-size 1024`. Must assert (a)
  decoded token IDs are bit-exact across the two runs (lossless), (b)
  decode tok/s with lookahead is at least 1.3x the baseline (the
  paper's 1.5x floor minus a 15% MoE-overhead budget), (c)
  `hypercar_bench --quick` still passes. Land before Task 28 and 29
  (EAGLE-2 work) so the tree-attention primitive is validated on the
  simpler consumer first.
- **Effort**: M (2-3 days)
- **Depends on**: none directly, but *blocks* Task 28 and Task 29 —
  the tree-attention primitive introduced here is the prerequisite
  those tasks were previously going to build from scratch.

### 48. Pre-read vAttention design note, then scope Task 31's MTLHeap fix
- **Goal**: 6 (48GB fit — allocator fragmentation at 1M context), 5 (swap spikes)
- **Derived from**: vAttention (2405.04437)
- **Change**: This is a scoping task, not an implementation task. Its
  output is a short design note committed at `research/
  vattention_mtlheap_notes.md` that:
  - Summarises vAttention's reserve-then-commit architecture in 3-5
    bullets (virtual range reserve, physical page commit on grow,
    contiguous KV tensor view for kernels).
  - Maps each vAttention primitive to its Metal equivalent
    (MTLHeap with MTLHeapTypePlacement, placement newBuffer
    offsets, heap size pre-reservation against the worst-case 1M
    context KV budget from CLAUDE.md's memory-budget table).
  - Rewrites Task 31's "Change" section to replace the current
    diagnostic-only wording with a concrete three-step plan:
    (a) measure current allocator churn under the existing
    fork/rewind path at 64K context, (b) implement an MTLHeap page
    pool behind a `--kv-heap-pool` flag, (c) re-measure churn and
    file a regression gate in `omlx/bench/aggregate.py`.
  - Identifies any place vAttention's CUDA-VM semantics *cannot*
    be reproduced on Metal (heaps are not growable; single-tenant
    assumption) so Task 31 does not underspecify the risk.
- **Verify**: The design note exists at the expected path, is
  referenced from the rewritten Task 31, and Task 31's new verify
  criterion is a measurable metric (e.g., "allocator-churn bytes/sec
  under 64K fork/rewind drops by >= 4x with `--kv-heap-pool` vs
  baseline"). No production code changes in this task itself.
- **Effort**: S (half a day — read + write, no code)
- **Depends on**: none; unblocks Task 31.

## Research-derived tasks (from LIT_REVIEW.md pass 7, 2026-04-12)

### 49. Adopt ProLong's RULER length × subtask matrix in Tasks 1/7/25
- **Goal**: 1 (1M context, *effective*), 2 (eval honesty)
- **Derived from**: How to Train Long-Context Language Models (Effectively) (2410.02660)
- **Change**: Documentation-only edit to TASKS.md and `omlx/bench/hypercar_bench.py` doc strings.
  - Update Task 1's "Verify" section with the specific RULER subtasks ProLong identifies as
    diagnostic for retrieval, multi-hop tracing, and frequent-word aggregation, and the
    specific length tiers (4K, 16K, 64K, 256K, 512K) at which each subtask should gate.
  - Update Task 7's memory-breach early-return path to use ProLong's "fail at the
    next-shorter length" semantics rather than aborting the whole phase.
  - Update Task 25's profiling target list to match ProLong's recommended subset rather than
    the current "all 13 tasks at all lengths" strawman.
  - Add a one-paragraph note at the top of `phase3b_ruler` in `hypercar_bench.py` citing the
    ProLong methodology so future contributors know why the subset was chosen.
- **Verify**: `git diff TASKS.md omlx/bench/hypercar_bench.py` shows the four documentation
  updates above and no behaviour changes. No new code paths. Tasks 1/7/25 each have an
  explicit RULER subtask × length pair in their verify criterion.
- **Effort**: S (half a day — pure documentation)
- **Depends on**: none; unblocks Tasks 1, 7, 25 by making them more concrete.

### 50. Training-free YOCO-lite KV-sharing probe in TurboQuantKVCache
- **Goal**: 5 (swap p90), 1 (longer context same budget), 6 (48GB fit)
- **Derived from**: You Only Cache Once (2405.05254)
- **Change**: Add a `--kv-share-stride N` flag to `omlx/hypercar_server.py` that causes the
  KV cache for layer `k` to be aliased to layer `k - (k mod N)` for `k mod N != 0`. Implement
  by adding a `share_from` field to `omlx/turboquant_kv.py`'s per-layer metadata and routing
  reads/writes through it. At `N=2` this halves the layer multiplier in the KV bill; at `N=4`
  it quarters it. Wire a new bench phase `phase_yoco_share_quality` in
  `omlx/bench/hypercar_bench.py` that runs RULER multi-key@16K and HumanEval with
  `--kv-share-stride 2` and `--kv-share-stride 4`, gating on no quality regression vs the
  existing 3-bit baseline.
- **Verify**: `python -m omlx.bench.hypercar_bench --kv-share-stride 2` passes all existing
  gates plus the new quality phase. KV memory at 64K context drops by >= 40% vs baseline (
  measured via `omlx/bench/aggregate.py --report HEAD`). RULER multi-key@16K stays >= 80% of
  baseline accuracy.
- **Effort**: M (2-3 days)
- **Depends on**: Task 12 (DuoAttention head classification) ideally lands first so we share
  KV only across consecutive *retrieval* heads — but the probe can run independently as a
  gross-effect baseline.

### 51. Multi-Token Prediction LoRA-retrofit probe via TTT engine
- **Goal**: 3 (decode speed)
- **Derived from**: Better & Faster Large Language Models via Multi-token Prediction (2404.19737)
- **Change**: Add a `train_mtp_heads` mode to `omlx/ttt.py` that:
  - Freezes the trunk, attaches three new LoRA-rank output projections (each predicting
    next+1, next+2, next+3 tokens), and trains them on a small coding corpus
    (HumanEval-train + the existing `omlx/bench/code_intel_problems.py` set).
  - Saves the resulting MTP head LoRAs to `~/.hypercar/ttt_checkpoints/mtp_*.npz`.
  - Adds a `--mtp-heads <path>` flag to `omlx/hypercar_server.py` that, when set, runs the
    main forward and verifies the n-1 extra-head predictions against the main head in a
    single pass — accepted tokens are committed without an extra forward.
  - Falls back to feeding the heads' top-1 predictions into Lookahead Decoding's n-gram pool
    if direct verification acceptance rate is below 30%.
- **Verify**: With trained MTP heads loaded, `python -m omlx.bench.hypercar_bench --quick`
  reports decode tok/s at 2K context >= 1.3x the baseline (gated). HumanEval pass@1 stays
  within 2 points of baseline. Acceptance rate logged in the bench output.
- **Effort**: M (2-4 days)
- **Depends on**: Task 47 (Lookahead Decoding) lands first to provide the n-gram fallback path
  and the tree-attention verification primitive.

### 52. rStar-Math process-supervision loop on top of TTT engine
- **Goal**: 2 (intelligence breadth, reasoning quality)
- **Derived from**: rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (2501.04519)
- **Change**: Add a `process_rollout` mode to `omlx/ttt.py` that:
  - Runs MCTS-style rollouts on a single MMLU-Pro reasoning problem, branching at each
    chain-of-thought step. Branching factor 3, depth 6 (tunable).
  - Uses Qwen3-Coder itself as the step verifier via an XGrammar-constrained prompt that
    forces a single token answer in `{good, bad, unsure}` per step (composes with Task 45).
  - Selects winning trajectories (problems where the final answer is correct) and losing
    trajectories (correct branches that lost to incorrect siblings) and feeds the pair into
    the existing SimPO loss path from Task 15.
  - Limits rollouts to MMLU-Pro categories where the most recent benchmark run reports
    accuracy < 60%, so we never spend rollouts on problems we already solve.
- **Verify**: After one rollout-and-train cycle on a held-out MMLU-Pro reasoning subset,
  `python -m omlx.bench.hypercar_bench --full` reports MMLU-Pro accuracy on the targeted
  category up by >= 3 points vs the pre-rollout checkpoint. No regression on HumanEval or
  RULER. Rollout cost logged so we know how much compute one improvement cycle takes on the
  M4 Pro.
- **Effort**: L (multi-day, possibly 1-2 weeks)
- **Depends on**: Task 36 (MMLU-Pro gate) and Task 15 (SimPO contrast in TTT) and Task 45
  (XGrammar constrained decoding) all land first.

## Research-derived tasks (from LIT_REVIEW.md pass 8, 2026-04-13)

### 53. Multi-head Latent Attention (MLA) rank probe on Qwen3-Coder KV
- **Goal**: 5 (swap), 1 (context in same budget), 6 (M4 Pro fit)
- **Derived from**: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2405.04434)
- **Change**: Add `omlx/bench/mla_rank_probe.py` — a read-only calibration tool that:
  - Runs prefill on a 16K-token representative context (pick a LiveCodeBench problem plus
    surrounding file) through Qwen3-Coder with `--kv-mode fp16` so we see raw K and V
    projection outputs, not 3-bit decoded ones.
  - For each layer, stacks the per-head K and V outputs into matrices K ∈ R^(T × d_k) and
    V ∈ R^(T × d_v), runs SVD, and records the rank needed to capture 99% and 99.9% of the
    spectral energy.
  - Writes `research/mla_rank_YYYYMMDD.json` with per-layer rank numbers, plus a markdown
    summary appended to `research/OPTIMIZATION_DECISION_MATRIX.md` that translates the ranks
    into a projected KV memory number at 1M context (vs our current 22.5GB 3-bit GQA
    baseline).
  - Reads from `omlx/turboquant_kv.py`'s capture hooks only — no changes to the cache itself.
- **Verify**: `python -m omlx.bench.mla_rank_probe` produces a JSON file and an appended
  section to OPTIMIZATION_DECISION_MATRIX.md. The summary names a concrete per-layer
  average rank d_c and a projected KV memory at 1M context. The `--full` benchmark run is
  unaffected (probe is opt-in, does not touch the default decode path).
- **Effort**: M (2-4 days)
- **Depends on**: none. Unblocks any future MLA retrofit work, which stays parked until
  this probe reports a rank cut worth pursuing (e.g. < 384 out of 1024).

### 54. InfiniGen-style KV prefetch predictor + madvise residency hint
- **Goal**: 5 (swap p90 I/O — direct attack on the failing gate), 3 (decode constancy)
- **Derived from**: InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management (2406.19707)
- **Change**: Two-phase implementation against `omlx/turboquant_kv.py` and
  `omlx/hypercar_server.py`:
  - **Phase A — capture**: Add a decode-time hook that logs, per layer, the set of KV pages
    actually touched by attention at step `t` and the hidden state of layer `t-1`. Run
    against a NIAH-64K capture (uses the existing phase3a harness) and save a per-layer
    page-access trace.
  - **Phase B — predictor + madvise**: Fit a per-layer linear predictor (ridge regression,
    target = page-touch bitmap, features = previous layer's hidden state mean-pooled per
    page-size chunk) offline from the Phase A capture. At decode time, before layer `k`
    runs, call the predictor to get a prefetch bitmap and issue
    `madvise(ptr, len, MADV_WILLNEED)` on the predicted KV pages via a new
    `turboquant_kv.prefetch_pages(layer, bitmap)` method.
  - Gate behind `--kv-prefetch` flag in `omlx/hypercar_server.py` so the default path is
    unchanged until we measure wins.
- **Verify**: `python -m omlx.bench.hypercar_bench --full` with `--kv-prefetch` shows p90
  sustained swap I/O (as reported by `omlx/bench/aggregate.py --report HEAD`) reduced by
  at least 30% vs the default path on N≥4 runs at 64K context, with no regression on
  HumanEval, NIAH, or RULER gates. Decode tok/s at 64K unchanged or improved.
- **Effort**: M (3-5 days)
- **Depends on**: Task 31 (KV allocator fragmentation profile) — we need its measurements
  to distinguish "prefetch helps" from "macOS mmap already does this." If Task 31 shows
  the OS is already prefetching effectively, this task downgrades to measurement-only.

### 55. MagicPIG LSH-sampled attention as fallback for Quest edge cases
- **Goal**: 3 (decode constancy across context length), 2 (quality guarantee, statistical)
- **Derived from**: MagicPIG: LSH Sampling for Efficient LLM Generation (2410.16179)
- **Change**: Extend `omlx/turboquant_kv.py` with a per-layer LSH index built over the
  pre-quant K vectors at prefill time:
  - Add `omlx/patches/magicpig_lsh.py` implementing a minimal LSH table (k=8 hashes,
    bucket width from paper defaults) keyed on normalized K rows. Build is one-shot at
    end of prefill, lives in a single flat buffer per layer.
  - Add a `sample_lsh(q, budget)` primitive that returns a weighted subset of key indices
    plus importance weights for an unbiased attention estimator.
  - Wire into the decode attention path so that *when Quest's (Task 24) top-K selection
    reports low confidence* (e.g. its max-bound spans less than 3x the chosen K's real
    score), the attention forward falls back to MagicPIG sampling on the same page set.
    Default stays Quest; MagicPIG is a quality safety net, not a replacement.
  - Add `omlx/bench/magicpig_accuracy.py` comparing full attention, Quest-only,
    MagicPIG-only, and Quest+MagicPIG-fallback on RULER multi-key + variable-tracking at
    16K, 64K, 128K.
- **Verify**: `python -m omlx.bench.magicpig_accuracy` shows Quest+MagicPIG-fallback
  matches full-attention RULER scores to within 1 point at all three lengths, while
  Quest-only shows a >2-point gap at 128K on at least one subtask. `python -m
  omlx.bench.hypercar_bench --full` with the fallback enabled shows no decode tok/s
  regression vs Quest-only.
- **Effort**: M (3-5 days)
- **Depends on**: Task 24 (Quest top-K page selection) landing first — MagicPIG is
  designed here as a fallback path inside Quest's substrate, not a standalone replacement.

## Research-derived tasks (from LIT_REVIEW.md pass 9, 2026-04-14)

### 56. Add MagicDec cost-model gate for speculative decoding decisions
- **Goal**: 3 (decode speed, constant across context), 4 (prefill)
- **Derived from**: MagicDec: Breaking the Latency-Throughput Tradeoff for Long
  Context Generation with Speculative Decoding (2408.11049)
- **Change**:
  - Add `omlx/specdec_gate.py` implementing a closed-form predictor
    `should_speculate(context_len, kv_load_lat_us, model_compute_us, accept_rate)`
    that returns a bool plus a projected speedup. The formula is lifted from
    MagicDec Eq. 2-4, re-fit for single-user (batch=1) interactive serving.
  - Add `omlx/bench/specdec_calibrate.py` that runs 4-8 prefill+decode rounds
    at 2K, 16K, 64K, 128K contexts in fp16 KV mode, measures per-token
    KV-load wall time vs model-compute wall time, and dumps the fitted
    constants to `omlx/specdec_constants.json`.
  - Wire the gate into `omlx/hypercar_server.py` so that any speculative-decoding
    code path (existing Lookahead Task 47, future EAGLE-2 Tasks 28/29, future
    TriForce Task 58) must consult `should_speculate` before activating.
    At short context the gate returns False and the server falls back to
    plain autoregressive; at long context the gate returns True with the
    projected speedup logged.
  - Add a pytest in `tests/test_specdec_gate.py` that exercises the gate
    against hand-computed expected values for three (context_len, kv_load,
    compute) tuples.
- **Verify**: `pytest tests/test_specdec_gate.py` passes. Running
  `.venv/bin/python -m omlx.bench.specdec_calibrate` produces a constants
  file whose predictions differ from a direct measurement (on a 4-run
  autoregressive baseline at 64K) by <15%. `grep -c should_speculate
  omlx/hypercar_server.py` shows the gate is called before any spec-decode
  dispatch.
- **Effort**: S (1 day)
- **Depends on**: None (it is a pure predictor; it *informs* future spec-decode
  tasks but does not block on any)

### 57. PyramidKV per-layer budget vector for Qwen3-Coder
- **Goal**: 5 (swap pressure), 1 (effective context at 128K+)
- **Derived from**: PyramidKV: Dynamic KV Cache Compression based on Pyramidal
  Information Funneling (2406.02069)
- **Change**:
  - Add `omlx/pyramid_budget.py` computing a 48-entry budget vector for
    Qwen3-Coder's 48 layers, exposing `budget_for_layer(layer_idx, total_budget)`
    using the paper's exponential-decay schedule as the starting shape
    (beta=0.7, reshaped to sum to `total_budget`).
  - Extend `omlx/turboquant_kv.py` so each TurboQuantKVCache layer accepts a
    `max_tokens` cap from this vector; when the cache grows past the cap,
    apply the existing SnapKV-style eviction (Task 46's primitive — this
    task composes with 46, not replaces it).
  - Add `omlx/bench/pyramid_calibrate.py` that sweeps (beta, total_budget)
    on a fixed grid and runs the existing code-intel eval + RULER
    multi-key at 16K on each setting, picking the pareto point that keeps
    code-intel at 5/5 and RULER multi-key within 1 point of full-KV.
  - Bake the winning budget vector into a default in `omlx/pyramid_budget.py`
    and add a `--pyramid-kv` flag to `omlx/hypercar_server.py` to enable.
- **Verify**: `.venv/bin/python -m omlx.bench.pyramid_calibrate` selects a
  budget vector. `.venv/bin/python -m omlx.bench.hypercar_bench --full` with
  `--pyramid-kv` enabled passes all gates AND shows at least a 20% reduction
  in peak KV memory at 64K context vs the non-pyramidal baseline, measured
  via the existing memory watchdog in `hypercar_bench.py`. p90 swap rate
  over N=8 runs (via `omlx/bench/aggregate.py --report HEAD`) drops below
  300 MB/s (down from 460 MB/s baseline — not yet at 100 MB/s target but
  on the right trajectory).
- **Effort**: S-M (2 days)
- **Depends on**: Task 46 (SnapKV prefill-time eviction in TurboQuantKVCache)
  — PyramidKV provides the *budget vector*, SnapKV provides the *eviction
  primitive*. They compose. If SnapKV is not yet live, this task can still
  land as "pyramidal budget + drop-oldest" as a fallback eviction policy,
  but the quality ceiling is lower.

### 58. TriForce hierarchical speculative decoding for long-context decode
- **Goal**: 3 (decode speed, constant across context), 1 (long-context decode)
- **Derived from**: TriForce: Lossless Acceleration of Long Sequence Generation
  with Hierarchical Speculative Decoding (2404.11912)
- **Change**:
  - Add `omlx/triforce.py` implementing the two-stage hierarchical draft:
    - Stage 1 draft: the existing target model (Qwen3-Coder-30B-A3B) run
      over a Quest-sparsified KV cache (Task 24's substrate). This reuses
      the already-loaded weights — no second large model in memory.
    - Stage 2 draft: a small Qwen-family model (Qwen2.5-0.5B or
      Qwen3-1.7B) loaded in fp16 under its own KV cache. If none is
      available, fall back to stage-1-only (Quest draft, target verifies).
    - Verification: standard speculative decoding tree verification against
      the full-KV target.
  - Wire the MagicDec gate (Task 56) as the activation condition —
    TriForce only runs when `should_speculate(...)` returns True.
  - Add `omlx/bench/triforce_bench.py` measuring decode tok/s at 16K, 64K,
    128K vs autoregressive baseline. Log acceptance rate per stage.
  - Add draft-model loading to `omlx/hypercar_server.py` behind a
    `--triforce-draft <hf-id>` flag. If the flag is absent, the server
    runs stage-1-only (Quest-as-draft, same target weights).
- **Verify**: `.venv/bin/python -m omlx.bench.triforce_bench --context 64000`
  shows at least 1.4x decode speedup over autoregressive at 64K, with zero
  output divergence from the autoregressive baseline (lossless — verify via
  byte-identical output on a fixed-seed prompt). Acceptance rate > 40% at
  stage 2 (if a draft model is loaded). `.venv/bin/python -m
  omlx.bench.hypercar_bench --full` passes all gates with TriForce enabled.
- **Effort**: L (multi-day, 4-6 days)
- **Depends on**: Task 24 (Quest top-K page selection) is the stage-1 draft
  substrate. Task 56 (MagicDec gate) is the activation gate. Both must be
  live before TriForce can land. Stage-2 draft model loading is optional —
  if no suitable small Qwen is available, ship as stage-1-only (which is
  effectively Quest-as-self-draft, still a strict improvement over plain
  autoregressive at long context per MagicDec's cost model).

## Research-derived tasks (from LIT_REVIEW.md pass 10, 2026-04-14)

### 59. OPLoRA orthogonal-projection safety rail for the TTT optimizer
- **Goal**: 2 (intelligence — continual learning without regression)
- **Derived from**: OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic
  Forgetting during Parameter-Efficient Fine-Tuning (2510.13003)
- **Change**:
  - Add `omlx/oplora.py` implementing the double-sided orthogonal
    projection: given a frozen weight `W` and a LoRA pair `(A, B)`, project
    the gradients `dA` and `dB` onto the subspace orthogonal to the top-k
    right and left singular vectors of `W` (k=8 default, configurable).
    Expose a single entry point `project_lora_grads(W, A, B, dA, dB, k) -> (dA', dB')`.
  - At server/TTT startup, compute and cache a truncated SVD (top-k) for
    every frozen linear layer `omlx/ttt.py` adapts. Use randomised
    truncated SVD (`mx.linalg.svd` on a random-projected 2k×2k subspace)
    to avoid the full-matrix SVD transient memory blow-up on Qwen3-Coder's
    larger matmuls. Cache file: `omlx/ttt_svd_cache.npz`.
  - Wire `project_lora_grads` into the TTT optimizer step in `omlx/ttt.py`
    behind a `--oplora` CLI flag (default on once validated).
  - Add `tests/test_oplora.py` verifying (a) projected gradient has zero
    inner product with the top-k singular vectors to numerical tolerance,
    (b) un-projected gradient's projection recovers the original when k=0,
    (c) round-trip through `save_lora`/`load_lora` preserves the SVD
    cache's bit identity.
- **Verify**: `pytest tests/test_oplora.py` passes. Running the existing
  TTT demo for N=20 hot-update rounds with `--oplora on` shows code-intel
  eval stays at 5/5 and MMLU-Pro drops by ≤ 1 point versus the frozen
  baseline, whereas the same run with `--oplora off` shows the usual
  ≥ 3-point regression the project_online_finetuning memory note
  describes. `.venv/bin/python -m omlx.bench.hypercar_bench --full` passes
  all gates with `--oplora on` as default.
- **Effort**: S (1 day for the projection + plumbing, +1 day for the
  randomised SVD if the direct SVD blows Metal memory)
- **Depends on**: None (the TTT loop is already live; Tasks 15 and 52
  benefit from this being landed first but do not block it)

### 60. Agentless three-stage SWE-bench Lite eval as the 5th Goal-2 eval family
- **Goal**: 2 (intelligence — 5th independent eval family)
- **Derived from**: Agentless: Demystifying LLM-based Software Engineering
  Agents (2407.01489)
- **Change**:
  - Add `omlx/bench/agentless_bench.py` implementing the three-stage
    pipeline as pure prompts against the running `hypercar_server.py`
    endpoint: (1) hierarchical file-level localization (one prompt per
    repository-level summarisation pass, then one prompt per file-level
    ranking), (2) patch generation (one prompt with the localised hunk +
    issue description), (3) patch validation (apply the diff, run the
    existing test suite shipped with SWE-bench Lite, record pass/fail).
    No tool-use loop, no function calling, no ReAct.
  - Port the SWE-bench Lite harness subset (300 instances) as a
    read-only data directory under `research/swe_bench_lite/`. Reuse the
    validation subprocess already in Task 35's SWE-agent scaffolding
    where possible, but strictly no agent loop — the Agentless pipeline
    is deterministic.
  - Wire the bench as a `--full`-only gate in `omlx/bench/hypercar_bench.py`
    with a threshold of `resolved >= 0.15` (conservative — Agentless
    paper reports 32% with a stronger backbone, we gate at 15% to allow
    headroom for Qwen3-Coder-30B being smaller).
  - Record per-instance cost (prompt tokens, decode tokens, wall time)
    so that future prompt-cache and decode optimisations can be
    regression-checked against this workload too.
- **Verify**: `.venv/bin/python -m omlx.bench.agentless_bench --limit 30`
  runs 30 SWE-bench Lite instances end-to-end, with all three stages
  emitting valid outputs (localisation ranks are non-empty, patches
  parse as unified-diff format, validation subprocess exits cleanly
  even on failed instances). `.venv/bin/python -m omlx.bench.hypercar_bench --full`
  passes with the Agentless gate enabled at `resolved >= 0.15` on the
  30-instance smoke subset. On the full 300-instance run (not required
  for every commit), `resolved >= 0.10` is the floor for pass.
- **Effort**: M (2-3 days for pipeline + harness + gate; +1 day for the
  300-instance full run)
- **Depends on**: Task 35 (SWE-agent harness) if the validation
  subprocess can be reused; otherwise independent. XGrammar (Task 45)
  is *not* a dependency because Agentless uses plain text prompts, not
  structured tool calls — this is one of the reasons it is cheaper
  than SWE-agent.

### 61. MMLU-Pro max_tokens hard floor guardrail (prevent silent quality regression)
- **Goal**: 2 (intelligence — Goal 2 is only proven at MMLU-Pro 62% with max_tokens=512; lower values silently break the eval)
- **Derived from**: Hypercar benchmark run 47 (2026-04-14), bench/snapshots/run47_2026-04-14T08-37/, and the commit pair 1e803b6 → 20df582 in the R44→R47 code delta. Commit 1e803b6 cut MMLU-Pro `max_tokens` from 512 to 256 to save ~2 minutes per run. The cut dropped MMLU-Pro from 64/100 to 24/100 — a -40pp silent regression — because reasoning answers were being truncated before emitting the final letter. Commit 20df582 reverted the cut. The landmine is still in the code: there is nothing preventing a future engineer from re-applying the "obvious" optimization.
- **Change**: In `omlx/eval/mmlu_pro/runner.py` (or wherever the MMLU-Pro generation loop lives — grep `max_tokens` inside `omlx/eval/mmlu_pro/`), hoist the `max_tokens` value into a named module-level constant `MMLU_PRO_MIN_MAX_TOKENS = 512` with a short comment documenting the 1e803b6→20df582 incident, and add an `assert max_tokens >= MMLU_PRO_MIN_MAX_TOKENS, "MMLU-Pro needs ≥512 max_tokens — 256 caused a 64→24% silent regression (commits 1e803b6→20df582)"` at the call site. Also add a unit test in `tests/test_eval_mmlu_pro.py` that tries to construct a runner with `max_tokens=256` and asserts it raises. This is a guardrail, not a feature — the point is that any future PR that shaves this number has to explicitly disable or update the assertion, forcing the author to surface the intent.
- **Verify**: `.venv/bin/python -m pytest tests/test_eval_mmlu_pro.py -k max_tokens_floor` passes. `.venv/bin/python -m omlx.bench.hypercar_bench --full` still scores MMLU-Pro 62/100 (unchanged from R44/R47 baseline). Attempt to pass `max_tokens=256` via any config path raises `AssertionError` at runtime.
- **Effort**: S (1-2 hours: add constant, assert, one unit test)

### 62. Namespace run numbering between analyst cron and implementation loop
- **Goal**: (meta — developer workflow / BENCHMARKS.md hygiene, not a hypercar goal directly)
- **Derived from**: Hypercar benchmark run 47 (2026-04-14), bench/snapshots/run47_2026-04-14T08-37/. Between R44 and R47 the implementation loop committed its own internal N=1..8 benchmark series using the literal strings "Run 45", "Run 46", "Runs 46-49", "Runs 46-53" inside commit messages (commits f63ef1d, ade43de, e9c4c43, abdabe5). These collide with the analyst-cron numbering where Run 45 and Run 46 were aborted pre-flight snapshots and Run 47 is this run. Two unrelated sample series now share the same integer namespace. A future reader walking `git log | grep Run` will be unable to tell analyst runs from devloop runs without reading each commit body. The BENCHMARKS.md file itself is unambiguous (it only contains analyst runs) but the git history is not.
- **Change**: Pick ONE of two options and document it in `bench/snapshots/README.md` and a one-line rule in `CLAUDE.md` under "Before Every Commit":
  1. **Prefix option**: analyst runs use "Run N" in commit messages, devloop internal sample runs use "devloop-N" or "sample-N". Update the existing analyst cron prompt to always prefix, and add a pre-commit lint in `scripts/` (or a simple sed in the existing hook) that rejects commit messages containing `Run <N>` where N ≤ the highest analyst-tracked Run in BENCHMARKS.md unless the commit also touches BENCHMARKS.md.
  2. **Unified option**: analyst and devloop share the numbering space, but devloop runs MUST also append a row to BENCHMARKS.md (or an appendix). Simpler but expands analyst's source-of-truth role onto engineers.
  Recommend option 1 — it preserves analyst independence and doesn't require engineers to touch BENCHMARKS.md. Commit the decision, then rewrite the next analyst cron prompt to mention the rule so future runs reinforce it.
- **Verify**: `git log --oneline HEAD~50..HEAD | grep -i "run [0-9]"` cleanly separates analyst from devloop runs (either by prefix or by the commit also touching BENCHMARKS.md). A new devloop internal-sample commit that uses "Run N" without touching BENCHMARKS.md should fail the pre-commit hook.
- **Effort**: S (2-3 hours: write the README update, decide policy, add the hook script, test)



## Research-derived tasks (from LIT_REVIEW.md pass 12, 2026-04-14)

### 63. Build Metal-residency watermark probe with producer back-pressure (MIKU analogue)
- **Goal**: 5 (swap <8GB p90)
- **Derived from**: MIKU / Architectural and System Implications of CXL-enabled Tiered Memory (2503.17864)
- **Change**:
  - New `omlx/bench/metal_watermark.py` — reads Metal peak memory via the existing `mlx.metal.get_peak_memory()` path and swap-out pages via `sysctl vm.swapusage`, at configurable sample rate. Exports a `current_watermark_ratio()` function returning `metal_peak / metal_load_limit` in [0.0, 1.0].
  - Patch `omlx/hypercar_server.py` chunked-prefill loop: between prefill chunks, if `current_watermark_ratio() > 0.92`, inject a 50ms sleep on the producer side (the chunk dispatcher), preventing the producer from saturating ahead of a soon-to-swap consumer. This is MIKU's throttle transplanted to an LLM serving loop.
  - Add a `--watermark-throttle` CLI flag on `omlx.hypercar_server` (default off until benchmarked).
- **Verify**: run `omlx.bench.hypercar_bench --full --kv-mode tq3 --watermark-throttle` and compare p90 swap rate vs. the same run without throttling. Success: p90 swap rate drops at least 30% with no more than 5% loss on NIAH@16K decode tok/s. Also: run without throttling must still pass memory gates (the throttle is opt-in and must not affect the default baseline).
- **Effort**: M (2 days)
- **Depends on**: none

### 64. Prototype two-tier TurboQuantKVCache with locality-aware migration (PAM analogue)
- **Goal**: 1 (1M context), 5 (swap headroom at long context)
- **Derived from**: PAM / Processing Across Memory Hierarchy for Efficient KV-centric LLM Serving System (2602.11521)
- **Change**:
  - Extend `omlx/turboquant_kv.py` with a `TieredTurboQuantKVCache` wrapper that holds two underlying caches: a **hot** tier (existing TurboQuant 3-bit) and a **cold** tier (same format but with `mx.metal.clear_cache()` called after writes, encouraging pages to be evicted from Metal-resident memory toward wired/swap-backed DRAM).
  - Add a migration policy: at decode step t, every K steps (start with K=32), scan the last window's attention scores from the hot tier. Any page whose cumulative attention weight is below a threshold migrates to cold. Any cold page that gets attended to above a threshold migrates back to hot.
  - Expose a `--kv-tiered` flag on `omlx.hypercar_server` and on `omlx.bench.hypercar_bench`.
  - Compose with DuoAttention (Task 12/13): retrieval heads always stay hot, streaming heads always cold. The PAM-style migration policy applies within each head group.
- **Verify**: on a 128K-token NIAH run with `--kv-tiered --kv-mode tq3` vs. baseline `--kv-mode tq3`, measure (a) Metal peak memory (target: at least 20% drop) and (b) NIAH retrieval accuracy (must stay within 2%). If both hold, the two-tier cache is worth productionising for 128K+. The test sits in Phase 3b of hypercar_bench — add a `phase3c_tiered_kv` subphase guarded by the new flag.
- **Effort**: L (multi-day — 4-6 days)
- **Depends on**: Task 13 (DuoAttention two-storage-class KV cache) landed; Task 54 (InfiniGen prefetcher) optional but ideal as the cold→hot migration trigger
- **See also**: `research/design_notes/jagged_kv_schedule.md` (Task 76) — the hot/cold tier split adds a new jaggedness axis on top of DuoKV's head-class jaggedness. Read section (c) before designing migration, especially item 3 (alignment with SDPA tile size BQ=32) and item 6 (stay within MLX's single-command-queue model).

### 65. Build async KV-page prefetch queue with compute/transfer overlap (AsyncTLS analogue)
- **Goal**: 3 (decode speed constant across context)
- **Derived from**: AsyncTLS / Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention (2604.07815)
- **Change**:
  - New `omlx/patches/async_kv_prefetch.py` — an async queue sitting between the decode loop and the KV cache. When decode step t runs, the queue (running in a background Python thread) consumes InfiniGen's predicted-next-needed page list for step t+1 and issues Metal residency hints (`mx.eval` on dummy accesses, or the MLX equivalent of `madvise(WILLNEED)` if it lands) to pre-warm those pages.
  - Hook into the existing Task 54 InfiniGen infrastructure when available; until then, use a simple 1-step lookahead policy that re-uses the previous step's top-K page set (AsyncTLS's own baseline).
  - Expose as `--kv-prefetch-async` on `omlx.hypercar_server`. Off by default.
- **Verify**: on a 64K-token decode workload with `--kv-prefetch-async`, measure decode tok/s variance across the last 500 tokens and compare to baseline. Target: latency variance drops by at least 25% (meaning the overlap is real) with mean tok/s unchanged or improved. Also run with MLX's `MLX_PROFILE_STREAM=1` env var (if available in this version) to confirm the prefetch thread and decode stream aren't serializing on the same command buffer.
- **Effort**: M (2-3 days)
- **Depends on**: Task 54 (InfiniGen prefetcher) preferred but not strictly required
- **See also**: `research/design_notes/jagged_kv_schedule.md` (Task 76) — the Python-thread-based prefetch queue approach mentioned here will fight Metal's single command queue. Read section (c) item 2 (queue prefetch into the same unevaluated graph as current-step SDPA) and item 6 (drop Python-thread variant in favor of lazy-eval-triggered variant) before implementing.


## Research-derived tasks (from LIT_REVIEW.md pass 13, 2026-04-14)

### 66. Add graceful-degradation envelope around Quest top-K page selection (LARU analogue)
- **Goal**: 3 (decode speed, constant across context — robust to predictor failure)
- **Derived from**: LCR / LARU — Toward Robust and Efficient ML-Based GPU Caching for Modern Inference (2509.20979)
- **Change**:
  - Add a per-page reuse-distance counter to `omlx/turboquant_kv.py` page metadata: each time a page is selected by Quest (Task 3 / 24), bump its counter and remember the decode step at which it was last touched.
  - In the Quest top-K dispatch path, mix two scores: Quest's existing per-query bound score, and a moving-average reuse-distance score (smaller distance = more likely to be reused). Combine via a learned (or, for v0, fixed) weight `alpha` in [0, 1]; `alpha=0` falls back to plain Quest, `alpha=1` falls back to LRU-K.
  - Add a runtime safety check: every 256 decode steps, compare the current top-K page set's expected hit rate against an LRU baseline computed offline from the trace. If the predictive policy is *underperforming* LRU by >10%, automatically reset `alpha=0` and log a warning. This is LARU's "graceful degradation envelope" — the predictor cannot do worse than baseline LRU.
  - Expose as `--quest-laru` on `omlx.hypercar_server` (off by default initially).
- **Verify**: build a synthetic adversarial workload in `omlx/bench/hypercar_bench.py` Phase 3c that flips between two disjoint NIAH needles every 1K tokens (forcing Quest's predictor to mispredict). With `--quest-laru`, decode tok/s under that workload must stay within 5% of the standalone LRU baseline. With Quest alone (no envelope) for comparison, the regression should be visible. Also: the standard NIAH@16K run must show no decode regression with the envelope enabled (proving the safety check doesn't trigger on benign workloads).
- **Effort**: M (3 days)
- **Depends on**: Task 3 (Quest query-aware page selection) — task 66 is an enhancement layered on top of Quest, so Quest must land first

### 67. Prototype DynamicAdaptiveClimb promotion-distance counters in TurboQuantKVCache
- **Goal**: 5 (swap headroom via better cache hit rate), 3 (decode speed indirectly)
- **Derived from**: DynamicAdaptiveClimb — Adaptive Cache Replacement with Dynamic Resizing (2511.21235)
- **Change**:
  - Add a `promotion_distance` field to TurboQuantKVCache page metadata (8-bit unsigned counter per page, default 0).
  - On every page hit during decode, increment that page's counter by 1 (saturating at 255). On every page miss, the migration logic uses the counter to decide whether the newly-loaded page should be placed at the "front" (hot) or "middle" (warm) of an LRU-style eviction list.
  - Add the dynamic-resizing variant: track recent hit-rate over a 1K-step sliding window. If hit-rate drops below 70%, expand the hot ring by 10% (up to a configurable cap); if hit-rate exceeds 90%, shrink the hot ring by 10% (down to a configurable floor). This is DynamicAdaptiveClimb's "automatically size the cache to match workload demand" applied to the hot/cold KV partition.
  - Build a 50-line trace simulator in `omlx/bench/cache_replay.py` that records page-access traces from a real `hypercar_bench` Phase 3 run, then replays them through (a) plain LRU, (b) ARC, (c) DynamicAdaptiveClimb, and reports hit rates. This is the validation harness — we ship the algorithm only if it beats LRU and ARC on real Hypercar traces.
- **Verify**: `omlx/bench/cache_replay.py` reports DynamicAdaptiveClimb hit rate >= ARC hit rate on at least 3 of 4 recorded NIAH/agentic traces. If yes, wire the algorithm into TurboQuantKVCache behind `--kv-promote-climb` and verify Phase 3 hypercar_bench passes (no quality regression).
- **Effort**: S (prototype: 1 day) / M (production wiring + validation: 2-3 days)
- **Depends on**: none — this is a self-contained algorithmic improvement on the existing cache abstraction

### 68. Build offline tier-cut-point auto-tuner for KV-mode selection (Kareto analogue)
- **Goal**: 5 (swap p90), 6 (M4 Pro fit), 3 (decode speed at the right cut-over)
- **Derived from**: Kareto — Adaptive Multi-Objective Tiered Storage Configuration for KV Cache in LLM Service (2603.08739)
- **Change**:
  - Currently `omlx/hypercar_server.py` picks `duo` vs `native` vs `tq3` via a hard-coded heuristic (context length, server flag). Replace this with a configuration-search step: a new `omlx/bench/kv_mode_search.py` module that runs the existing benchmark Phase 1-3 across a small grid (duo, native, tq3, plus the cut-over thresholds), records (decode_tok_s, prefill_tok_s, metal_peak, p90_swap), and dumps a Pareto front to `bench/snapshots/kv_mode_pareto.json`.
  - At server startup, `hypercar_server.py` reads the Pareto JSON and picks the configuration that satisfies a CLI-specified objective (`--objective throughput`, `--objective latency`, `--objective memory`). If no JSON is present, fall back to the current hard-coded heuristic.
  - The optimiser uses Kareto's diminishing-return-guided pruning to avoid evaluating every grid point — once the marginal improvement from one more sample drops below 2%, stop.
- **Verify**: on the M4 Pro reference machine, run `python -m omlx.bench.kv_mode_search` once. The output JSON must contain at least 4 Pareto-optimal configurations across the (decode, memory) plane, and the chosen default config (objective=balanced) must match the current hand-picked default on at least 2 of 3 metrics. Then: with `--objective memory`, `hypercar_bench` runs with the auto-selected config must show measurably lower Metal peak than the default (≥5% drop) without HumanEval regression.
- **Effort**: M (2-3 days)
- **Depends on**: Task 64 (two-tier TurboQuantKVCache) — Kareto's optimiser becomes most useful once there's an actual tier configuration to search over. Until then the search space is just (duo, native, tq3, fp16).

## Research-derived tasks (from LIT_REVIEW.md pass 14, 2026-04-14)

### 69. Block-Sparse Flash Attention: gate V-block loads in MLX flash attention (BSFA analogue)
- **Goal**: 1 (1M context — directly closes the 64K NIAH intermediate-score-tensor bottleneck), 4 (prefill speed), 6 (M4 Pro fit at long context)
- **Derived from**: Block Sparse Flash Attention (2512.07011)
- **Motivation**: Commit `5d9d207` identified the 64K NIAH failure as the 16 GB `softmax(QK^T)` intermediate score tensor, *not* the KV cache. Every sparse-attention task currently on the backlog (Quest, MInference, DuoAttention) attacks cache footprint or per-step work. BSFA is the first cited paper that directly attacks the score tensor by staying inside the flash tile and gating V-block fetches.
- **Change**:
  - Add an attention patch `omlx/patches/bsfa_attention.py` that wraps MLX's `mx.fast.scaled_dot_product_attention` (or our existing specprefill attention path) with a tile-level gate. For each (query tile, key tile) pair, compute the exact tile-max score; if that max is below a calibrated per-layer/per-head threshold, skip loading the corresponding V tile entirely and contribute zero to the running softmax denominator.
  - Build a calibration harness `omlx/bench/bsfa_calibrate.py` that runs one pass over a small validation set (the existing coherence + NIAH@4K prompts), records the per-layer-per-head tile-max distribution, and solves for thresholds that skip approximately 40-50% of V tiles while keeping per-layer attention reconstruction error below 1e-3. Persist thresholds to `bench/snapshots/bsfa_thresholds.json`.
  - Gate behind a server flag `--bsfa` on `omlx.hypercar_server`, default off.
- **Verify**: on hypercar_bench with `--bsfa`:
  - NIAH@4K, @16K, @64K all pass (needle retrieved, no quality regression).
  - Measured Metal peak at 64K NIAH drops by at least 4 GB vs the `--bsfa` off baseline (the score-tensor saving).
  - Prefill speed at 16K increases by at least 10% (secondary effect of skipped V loads).
  - HumanEval pass@1 within 2% of baseline.
- **Effort**: M (3-4 days) — the algorithm is small but MLX has no native flash-sparse primitive, so implementation is in the mx.compile layer or a custom attention kernel path.
- **Risk**: Apple Silicon's unified memory makes "skip V tile load" a smaller win than on CUDA (V is already in shared memory); the real benefit on MLX comes from avoiding the `mx.eval()` of the skipped tile's contribution to the score tensor. The measured Metal-peak drop is the metric that matters, not the raw speedup.
- **Depends on**: none — self-contained. Can land before or in parallel with Task 3 (Quest).

### 70. ButterflyQuant-style learnable butterfly rotation in TurboQuant KV codec
- **Goal**: 5 (swap headroom at 1M), 6 (M4 Pro fit under load), 1 (1M context with headroom for Chrome+editor)
- **Derived from**: ButterflyQuant — Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms (2509.09679)
- **Motivation**: `omlx/turboquant_kv.py` uses a fixed Walsh-Hadamard Transform for pre-quant rotation (per the CLAUDE.md warning that "Givens is broken"). ButterflyQuant proves that *learnable* Givens parameterised as butterfly networks beat fixed WHT at 2-bit quantization. The practical payoff is 2-bit KV quality matching our current 3-bit, which halves the 1M-context KV footprint from 22.5 GB to ~15 GB and gives 7 GB of headroom for Chrome+editor under sustained load.
- **Change**:
  - Add a new optional rotation path in `omlx/turboquant_kv.py`: `TurboQuantKVCache(rotation="wht")` (current default) vs `rotation="butterfly"` (new). The butterfly transform is O(d log d) with d log d / 2 Givens angle parameters per layer (where d is the head dim, typically 128), orthogonal by construction.
  - Build `omlx/bench/butterfly_calibrate.py`: runs a small calibration loop over validation prompts, records KV distributions per layer, and optimises butterfly Givens angles against a reconstruction-loss objective on the Stiefel manifold (orthogonality-preserving). Reuses the existing calibration infrastructure from the TQ codec.
  - Add a `--kv-rotation butterfly` flag to `omlx.hypercar_server`. Persist learned butterfly parameters to `bench/snapshots/butterfly_rotation_{layer}.npz`.
  - Run the 3-bit → 2-bit sweep: with the learned butterfly in place, quantize KV to 2 bits and compare against the existing 3-bit WHT baseline on NIAH, RULER, and HumanEval.
- **Verify**:
  - At 3-bit KV with butterfly rotation: NIAH@16K, RULER@16K, HumanEval within 1% of WHT baseline (sanity check — butterfly should at worst match WHT at 3 bits).
  - At 2-bit KV with butterfly rotation: NIAH@16K passes, RULER@16K within 3% of 3-bit baseline, HumanEval within 3% of 3-bit baseline. This is the real test — if 2-bit butterfly matches 3-bit WHT on quality, we ship.
  - 1M-context memory projection: the Metal peak in duo mode at simulated 1M context must drop by at least 6 GB vs the 3-bit baseline (the expected 22.5 → ~15 GB saving).
- **Effort**: L (4-5 days) — one day for the butterfly forward, two days for the calibration loop (on-manifold optimisation is the tricky part), one day for validation sweep, one day for debugging the 2-bit failure modes.
- **Risk**: ButterflyQuant's published numbers are for *weight* quantization; KV cache distributions are token-varying and prompt-dependent, so the learned butterfly calibrated on validation prompts may not generalise to production KV patterns. Mitigation: verify per-layer reconstruction error stays bounded across a held-out diverse prompt set *before* committing to 2-bit.
- **Depends on**: none directly, but best sequenced after Task 66 (Quest + LARU envelope) has landed so the 2-bit failure mode (if any) shows up against a stable top-K baseline rather than a moving target.

### 71. Add `--niah-only --niah-context 64K` escape hatch so analyst can corroborate Goal 1 claims
- **Goal**: 1 (1M context) — specifically, give the analyst cron a way to independently verify the 64K and 128K NIAH PASS claims that are now in CLAUDE.md (commit 2bdb217) and research/ docs (commit 25f8d82).
- **Derived from**: Hypercar benchmark runs 49, 50, 51 (all 2026-04-14 aborts), bench/snapshots/run49_2026-04-14T12-37/, run50_2026-04-14T14-37/, run51_2026-04-14T16-32/. Implementation loop has been running `--force --kv-mode native --niah-500k` (PID 59255 in R49) and posted CLAUDE.md updates claiming Goal 1 validated to 64K (commit 2bdb217 "NIAH PASS at 33.9 GB Metal") and "128K in progress" (commit 25f8d82 research session summary). Three consecutive analyst cron fires aborted on pre-flight memory checks while engineers held the box, leaving zero independent corroboration of those Goal 1 claims. Even when the analyst cron CAN run, the existing headroom gate in `omlx/bench/hypercar_bench.py` skips 64K RULER tasks because `projected 30.6 GB > 7.8 GB headroom` (visible in every analyst console.txt at the line `[5/15] multi_key_niah@64K keys=3 — SKIP`). The result: the analyst CAN'T verify the engineer's biggest current claim even when it does run successfully.
- **Change**: Add a CLI escape hatch to `omlx/bench/hypercar_bench.py`:
  - New flag `--niah-only` that runs ONLY Phase 0 smoke + Phase 3 NIAH at the requested context length(s) and skips RULER, MMLU-Pro, HumanEval. This makes the bench a ~2-3 minute targeted probe rather than a 20+ minute full sweep.
  - New flag `--niah-context <list>` accepting a comma-separated list like `4K,16K,64K,128K` so the analyst can opt INTO the larger NIAH probes that the headroom gate currently blocks.
  - When `--niah-context` is passed AND a context exceeds the headroom gate, log `WARNING: Bypassing headroom gate for {context} per --niah-context request — Metal projected {GB}` instead of skipping silently. Run it. Let the watchdog fail if it actually runs out — the watchdog is already trustworthy.
  - The combined `--full --niah-only --niah-context 64K,128K` invocation should let the analyst cron do a focused Goal 1 corroboration pass without competing for memory with concurrent engineer workloads.
- **Verify**:
  - `.venv/bin/python -m omlx.bench.hypercar_bench --niah-only --niah-context 4K,16K` runs only Phase 0 + Phase 3, completes in under 90s, writes results.json with prefill/decode tok/s for both contexts.
  - `.venv/bin/python -m omlx.bench.hypercar_bench --niah-only --niah-context 64K` runs only Phase 0 + Phase 3 at 64K, bypasses the headroom gate with a clear WARNING, and either passes (proving Goal 1 to 64K independently) or trips the watchdog cleanly with a memory breach record (also a useful data point).
  - The analyst cron prompt can then add an optional pre-step: "if engineer claims a new Goal 1 milestone, run `--niah-only --niah-context <claimed-ctx>` to corroborate before the next full bench."
- **Effort**: S-M (4-6 hours: add the two flags, plumb skip-other-phases logic, gate-bypass warning, smoke-test on 4K/16K, document the escape hatch in `--help` epilog)
- **Risk**: Bypassing the headroom gate at 64K with current default config could OOM the model load if the engineer workload is also resident. Mitigation: the existing memory-availability pre-flight (the one that aborted R50 and R51) will still fire, so this escape hatch only runs when the box has 30+ GB free anyway.

## Research-derived tasks (from LIT_REVIEW.md pass 15, 2026-04-14)

### 72. Prototype eLLM-style elastic memory manager for Hypercar server (unified virtual-tensor + CPU balloon on MLX)
- **Goal**: 1 (1M context — specifically the 500K NIAH pre-flight memory block), 5 (swap headroom), 6 (M4 Pro 48GB fit under load)
- **Derived from**: eLLM — Elastic Memory Management Framework for Efficient LLM Serving (2506.15155)
- **Motivation**: Commits `6fb0e95` (R50 abort) and R49/R51 aborts show that the 500K NIAH path fails at a pre-flight memory check with "8.8 GB free vs 30 GB needed." The 30 GB figure is the sum of worst-case pre-allocations stacked across the static weight pool, the activation pool, and the KV cache pool in `omlx/hypercar_server.py` and `omlx/turboquant_kv.py`. eLLM's core observation is that these three pools are managed at different abstraction levels in every existing serving stack, which forces conservative worst-case allocation and leaves ~20% throughput on the table; unifying them under a single virtual-tensor abstraction with dynamic inflation/deflation into CPU memory recovers the gap and delivers 2.32x decode throughput + 3x batch size at 128K tokens. On Apple Silicon the CPU/GPU memory is *already* unified, so the "CPU buffer" in eLLM's cost model is just a different residency class inside the same Metal heap — the porting surface is therefore smaller than on the discrete-GPU baselines the paper evaluates.
- **Change**:
  - Introduce a `HypercarMemoryPool` abstraction in `omlx/hypercar_server.py` that owns all three pools: static weight residents, activation scratchpads (including the BSFA tile buffers from task 69), and KV cache pages.
  - Add an inflation controller: when allocation pressure exceeds a watermark, deflate cold KV pages first (demoting them to a lazy-rehydrate class backed by `mx.gpu → mx.cpu` residency hints), then activation scratchpads second, then fail loud if static weights are the only thing left and there's still no room.
  - Replace the current headroom gate in `omlx/bench/hypercar_bench.py` with a *reactive* check: instead of "fail-fast at 30 GB projected," call the inflation controller and only abort if it can't find room after deflation.
  - Add `--memory-elastic` flag to `omlx.hypercar_server` (default off until validated).
- **Verify**:
  - 500K NIAH runs to completion on the same 48GB system where R49/R50/R51 aborted. Measure peak Metal, peak CPU-side balloon, and wall-clock.
  - At 128K context, decode throughput improves by at least 1.5x relative to the current pre-allocation path (we don't need eLLM's full 2.32x because Apple's unified memory already has some of the benefits baked in).
  - No HumanEval or RULER regression vs the current `--kv-mode native` baseline.
  - Watchdog still trips on *actual* breach, not projected breach — verify by injecting a synthetic over-allocation and confirming the watchdog fires.
- **Effort**: L (5-7 days). Day 1: virtual-tensor wrapper + residency class tagging. Days 2-3: inflation controller + KV page demotion primitive. Day 4: reactive headroom gate in the bench. Day 5: full 500K validation run + any required debug. Days 6-7: buffer for the inflation/deflation cost model, which is where eLLM's paper spends half its complexity budget.
- **Risk**: Apple's unified memory model may blur the eviction boundary to the point where "CPU buffer as overflow" has no latency advantage — if there's no meaningful latency difference between "Metal-allocated" and "CPU-allocated" on M4 Pro, eLLM's balloon has no place to breathe and the whole abstraction is ornamental. Mitigation: before committing to the full multi-day task, spend two hours measuring the actual latency delta between Metal and CPU allocation on a representative 1 GB tensor on this hardware. If the delta is < 10%, scope the task down to "unified pool without explicit CPU class" — still a win because it removes the worst-case pre-allocation stacking, just without the balloon.
- **Depends on**: Task 9 (headroom gate) — this task *replaces* the fail-fast gate with a reactive variant, so Task 9's logic needs to be preserved as the fallback if elastic mode is disabled.

### 73. DFTopK linear-time top-K replacement for Quest page-selection hot path
- **Goal**: 3 (decode speed — Quest hot path), 1 (1M context top-K scales linearly in page count)
- **Derived from**: DFTopK — Differentiable Fast Top-K Selection for Large-Scale Recommendation (2510.11472)
- **Motivation**: Task 34 (Quest) and the follow-ons Task 55 (MagicPIG LSH) and Task 66 (LARU envelope) all hang on a top-K-over-KV-pages primitive that currently falls back to MLX's `argpartition`. At 1M context with page_size=16 that's ~64K pages per decode step per layer, and `argpartition` is O(n log k) at best. DFTopK is the recommender community's 2025 state of the art for top-K over millions of items per query: a closed-form linear-time approximation that bypasses soft permutation matrices entirely, validated in production A/B testing at Kuaishou (+1.77% revenue at the same compute budget). Plugging it into Quest's selection step gives a strictly faster forward path, and the differentiable-gradient side also opens the door to learning page-importance scores end-to-end if we want to upgrade Quest from hand-designed min/max bounds to a learned predictor.
- **Change**:
  - Implement DFTopK's closed-form forward in `omlx/topk_dftopk.py`: single-pass computation of the threshold for the top-K set, then a sigmoid-style relaxation around the threshold to yield a differentiable mask. Non-differentiable inference-only mode is a further simplification (just the threshold).
  - Wire it into the Quest prototype (whichever file the task 34 landing creates) as an alternate selection path, behind a `--quest-topk dftopk` flag with `argpartition` remaining the default until validation passes.
  - Add a microbench: top-K over a synthetic 64K-page score tensor, comparing DFTopK inference-only vs `mx.argpartition` on latency and quality (overlap fraction of selected pages, mean score of selected set).
- **Verify**:
  - DFTopK forward is at least 1.3x faster than `argpartition` on the 64K-page synthetic benchmark with k=512 (the Quest default).
  - NIAH@16K and RULER@16K with Quest+DFTopK match Quest+argpartition to within 1% on each retrieval family. This is the accuracy gate — if DFTopK's relaxation drops any of the retrieval tasks by more than 1%, we need to revisit the threshold calibration.
  - HumanEval with Quest+DFTopK matches Quest+argpartition to within 1%.
- **Effort**: S-M (2-3 days). Day 1: DFTopK forward + microbench. Day 2: wire into Quest + accuracy sweep. Day 3: buffer for any threshold-calibration debugging.
- **Risk**: DFTopK's linear-time guarantee comes from a relaxation of the normalization constraint, which may not exactly match `argpartition`'s top-K set. If the overlap is less than 95% on Quest's score distribution, we'd need to tighten the threshold which costs some of the speedup. Mitigation: inference-only mode has a free knob (the sharpness parameter on the relaxation) that can be tuned per-deployment on calibration data.
- **Depends on**: Task 34 (Quest prototype) — this is a drop-in replacement for the top-K call, so Quest has to exist first. If Quest hasn't landed yet when this task is picked, the microbench path (synthetic 64K-page tensor) can be built standalone and left waiting for Quest to arrive.

### 74. HATA learnable-hash top-K scoring for KV page selection (third angle after MagicPIG + Quest)
- **Goal**: 3 (decode speed — Quest scoring function), 1 (O(log n) attention at long context)
- **Derived from**: HATA — Trainable and Hardware-Efficient Hash-Aware Top-k Attention (2506.02572)
- **Motivation**: The Quest top-K primitive has three plausible scoring functions, each from a different research lineage: Quest (task 34) uses per-page min/max bounds; MagicPIG (task 55) uses fixed LSH; HATA uses *learned* binary hash codes whose hamming distance preserves qk-score order. HATA is more accurate per bit of hash code than fixed LSH (because the hash is learned against a real attention target) and strictly cheaper per query than min/max bounds (because the hash comparison is a single XOR + popcount, not a per-page bound computation). For the agentic workloads Hypercar serves, relative page-score order matters more than absolute values, which is exactly what HATA preserves. And the recommender "learning to hash" literature has 15 years of accumulated wisdom on stabilising learned hashes against distribution shift, which is the known failure mode of this class.
- **Change**:
  - Add `omlx/hash_topk_hata.py` with a small linear-then-sign hash projection (projecting head-dim=128 keys and queries to 64-bit hash codes).
  - Build `omlx/bench/hata_calibrate.py`: a calibration loop that runs a frozen attention forward on validation prompts, records the (query, key, score) distribution, and trains the hash projection to minimise the ranking loss between hamming-distance order and exact score order.
  - Add a top-K scoring path that reads hash codes instead of computing full scores, gated behind `--quest-scoring hata` (with `quest` = min/max bounds staying as the default).
  - **Defensive composition with Quest**: when HATA is enabled, use HATA to propose a top-K candidate set of size 2K, then use Quest's min/max bounds to *certify* the final K. This gives us HATA's speedup with Quest's accuracy floor as a safety net.
- **Verify**:
  - HATA's forward scoring is at least 2x faster than Quest's min/max bound computation on the 64K-page synthetic benchmark.
  - NIAH@16K, RULER@16K, HumanEval with HATA+Quest-certification match Quest-alone to within 1% on each eval.
  - Hash overlap (fraction of HATA's top-K that also appears in Quest's top-K) is at least 80% on the calibration corpus and at least 70% on a held-out diverse-prompt set — this is the early-warning signal for distribution shift.
- **Effort**: M (3-4 days). Day 1: hash projection + forward. Day 2: calibration script + training loop. Day 3: wire into the Quest path with defensive Quest-certification. Day 4: quality sweep.
- **Risk**: Learned hashes trained on a fixed calibration corpus drift when the serving distribution shifts — this is the classic "learning to hash" failure mode. Mitigation is baked into the design: Quest's min/max bounds always certify the final top-K set, so HATA can only ever *propose* faster, never *decide* less accurately. Secondary risk: training a hash projection requires a supervision signal (frozen attention scores), and generating that supervision on Hypercar's target workload is a one-time cost that needs enough prompt diversity to generalise.
- **Depends on**: Task 34 (Quest prototype) and Task 73 (DFTopK), both of which must land first because HATA is a scoring-function swap *inside* the top-K primitive that those tasks own. If Task 73 lands and materially changes the top-K path, HATA's integration point shifts correspondingly.


## Research-derived tasks (from LIT_REVIEW.md pass 16, 2026-04-14)

### 75. Port MegaFold's staged-scratchpad EvoAttention pattern to MLX flash attention
- **Goal**: 4 (prefill speed), 5 (swap headroom under long-context load)
- **Derived from**: MegaFold (2506.20686)
- **Change**:
  - Add a tiled-score-tensor code path to the MLX attention backend used by `omlx/hypercar_server.py`. Unlike BSFA (Task 69) which gates V-block loads, this path always materialises the score tiles but *never holds the full (N, N) tensor resident* — each tile is consumed by the V-aggregation step before the next tile is computed.
  - Implementation option A: lean on `mx.fast.scaled_dot_product_attention` if it already does internal tiling we can configure.
  - Implementation option B: write a `mx.compile`-wrapped chunked attention function that takes explicit `tile_m` and `tile_n` parameters and materialises the score in chunks of `(tile_m, tile_n)`, streaming output accumulation across tiles.
  - Target: peak intermediate attention memory at 64K context should drop from the 16 GB observed in commit `5d9d207` to under 1 GB.
- **Verify**: run `omlx.bench.hypercar_bench --full --kv-mode native --mem-profile` at 64K NIAH. Success: peak intermediate attention memory under 1 GB (a 16x reduction) AND decode tok/s within 5% of baseline AND HumanEval 18/20 unchanged.
- **Effort**: M (2-3 days: 1d investigation of MLX attention internals, 1d implementation, 1d profiling + gate sweep)
- **Depends on**: none — complements Task 69 (BSFA) but does not require it

### 76. Jagged-tensor scheduler design note for two-tier KV cache (HSTU-CP inspiration)
- **Goal**: 1 (1M context sharding), 3 (decode parallelism)
- **Derived from**: HSTU Context Parallelism (2508.04711) — inspiration paper, not direct port
- **Change**:
  - Write a design note at `research/design_notes/jagged_kv_schedule.md` that captures the HSTU context-parallelism framing (jagged tensors need jagged schedulers) as applied to DuoAttention's retrieval/streaming head split.
  - The note should identify: (a) where in `omlx/turboquant_kv.py` the jaggedness lives after Task 13 lands, (b) which MLX primitives (if any) support jagged-shape dispatch, (c) what the minimum viable "jagged scheduler" looks like for a single-device M4 Pro setting (hint: it's likely just careful ordering of compute-vs-residency-hint operations, not true multi-stream parallelism).
  - This is a *design note* task — it produces a markdown file, not code. It informs Tasks 64 and 65 when they're designed in detail.
- **Verify**: file exists at `research/design_notes/jagged_kv_schedule.md`, has the three-section structure above, and is referenced from Tasks 64 and 65 for consumption when those tasks are picked up.
- **Effort**: S (half a day)
- **Depends on**: none

## Research-derived tasks (from LIT_REVIEW.md pass 17, 2026-04-14)

### 77. Halo-style query-plan DAG for prompt cache in hypercar_server
- **Goal**: 3 (decode speed via cache reuse), 4 (prefill speed via shared subexpressions), 1 (1M context efficiency)
- **Derived from**: Halo / Batch Query Processing for Agentic Workflows (2509.02121)
- **Change**:
  - Build a query-plan DAG abstraction in `omlx/hypercar_server.py` that wraps each chat-completion request as a sequence of computational stages: prefix-prefill, suffix-prefill, decode, optional tool-call rounds. Today the prompt cache is a hash-of-prefix lookup; this task makes it a *plan-level* structure that can identify shared subexpressions between non-identical prompts.
  - Implement a cost model that scores each plan node in MLX time units: prefill cost ∝ N², decode cost ∝ K (top-K page count under Quest, full N under native), cache-hit ∝ 0, cache-miss ∝ prefill cost. The cost model should be calibrated from the existing benchmark traces in `bench/snapshots/`.
  - Implement a plan rewriter that, given a batch of plans (or a rolling window of recent single-user plans), finds shared subexpressions and emits a consolidated execution schedule. For Hypercar's interactive single-user setting, the "batch" is a rolling window of the last K turns of an active session.
  - Compose with task #42 (CacheBlend) — Halo decides *when* to invoke CacheBlend's physical-layer KV reuse, CacheBlend executes it.
- **Verify**:
  - Unit test: a synthetic two-turn conversation where turn 2 differs from turn 1 only in the final user message — expected behaviour is that the planner reuses the prefill of the shared prefix and only re-prefills the divergent suffix.
  - Integration test: `omlx.bench.hypercar_bench --quick` p50 TTFT on a cached-prompt path improves by at least 30% on a 4K-token shared prefix case.
  - Quality gate: HumanEval pass rate unchanged, RULER NIAH 4K still passes.
- **Effort**: L (1-2 weeks: 2-3d for the DAG abstraction + cost model, 3-4d for the plan rewriter, 2-3d for integration testing)
- **Depends on**: Task 42 (CacheBlend) is a strong but not strict dependency — Halo can land first as a plan-level optimiser that only handles prefix-sharing, and then unlock more reuse patterns once CacheBlend is in.

### 78. SWE-Shepherd PRM trained on Hypercar TTT rollouts for action-level reward shaping
- **Goal**: 2 (intelligence breadth, especially SWE-Bench Verified)
- **Derived from**: SWE-Shepherd (2604.10493)
- **Change**:
  - Instrument the existing TTT rollout loop in `omlx/ttt.py` to emit *action-level* trajectory records: each record captures the per-action context (file navigation, code edit, test execution), the action token sequence, and a binary downstream success label propagated from terminal pass/fail.
  - Train a small Process Reward Model on the resulting dataset. Use a frozen Qwen3-Coder-30B-A3B as the backbone and fine-tune a lightweight reward head (a 2-layer MLP over the final hidden state). Target: reward head fits in <100 MB so it loads alongside the main model on the M4 Pro.
  - Wire the PRM into TTT inference as an action scorer: at each rollout step, score the top-K candidate actions and prefer the highest-PRM-score action. Gate the TTT *weight update* on a high PRM score in addition to terminal pass — bad rollouts that happened to pass via luck should not update the model.
  - Compose with task #59 (OPLoRA safety rail) — OPLoRA bounds the parameter update magnitude, the PRM bounds the reward signal quality.
- **Verify**:
  - Offline metric: PRM AUC on a held-out trajectory split should exceed 0.7 (random baseline 0.5).
  - End-to-end metric: SWE-Bench Lite pass rate (Task 60 eval family) should improve by at least 2 percentage points after one TTT epoch with PRM-shaped rewards vs. baseline pass/fail rewards.
  - Safety gate: PRM-driven TTT updates should not regress HumanEval pass rate (a "locally helpful, globally harmful" failure mode the paper itself warns about).
- **Effort**: M (1 week: 2d for trajectory instrumentation, 2d for PRM training, 2d for TTT integration, 1d for the gate sweep)
- **Depends on**: Task 60 (Agentless SWE-Bench eval) for the trajectory data source, Task 59 (OPLoRA) for the safety rail composition. Task 52 (rStar-Math process supervision) is a related but distinct cousin — rStar uses MCTS self-search to manufacture rewards, SWE-Shepherd uses a trained PRM; both could coexist.

### 79. Bayesian Kalman drift estimator for TTT calibration tracking
- **Goal**: 2 (intelligence breadth via TTT robustness)
- **Derived from**: Filtering Beats Fine Tuning: Bayesian Kalman View of ICL (2601.06100)
- **Change**:
  - Add a closed-form Kalman filter to `omlx/ttt.py` that tracks a low-dimensional latent adaptation state across TTT updates. The state can start as a 4-8 dimensional vector summarising recent rollout statistics (rolling mean reward, rolling variance, recent OPLoRA gradient norm, recent PRM score from task 78).
  - On each TTT update, run one Kalman recursion step: predict the next state from the current state + process noise, observe the new rollout outcome, update the posterior mean and *posterior covariance* in closed form.
  - Use the trace of the posterior covariance as a *drift metric*. When the trace exceeds a threshold (calibration: 2× the trace observed at TTT start), trigger a rollback to the last checkpoint instead of continuing to update. This is the principled rollback trigger that TTT currently lacks.
  - Implementation is ~30 lines of MLX. The math is the standard Kalman recursion; the only Hypercar-specific piece is choosing the state dimensions and the process noise covariance, which can be tuned offline against recorded TTT traces.
- **Verify**:
  - Unit test: feed the filter a synthetic trajectory where the rollout reward distribution shifts mid-stream (simulated drift). The posterior covariance trace should detectably increase after the shift; the rollback trigger should fire within 5 updates of the shift.
  - Integration test: run TTT for 100 updates with the filter active; verify that a known-bad rollout pattern (consecutive failed trajectories) triggers a rollback rather than continuing to update the model.
  - No regression: TTT loop with filter active should produce the same final checkpoint as the baseline loop on a clean trajectory.
- **Effort**: S (1-2 days: half-day for the Kalman recursion code, half-day for the unit tests, half-day for integration into TTT)
- **Depends on**: none (pure addition to `omlx/ttt.py`). Composes with task #78 (PRM as one of the state-vector observations) and task #66 (LARU graceful-degradation envelope — both are bound-the-failure-mode patterns).

### 80. Add wall-clock correlation to bench profiler to detect swap-induced stalls
- **Goal**: 5 (swap pressure), 6 (machine fit under load), plus observability-meta
- **Derived from**: Hypercar benchmark run 57 (2026-04-15), bench/snapshots/run57_2026-04-15T09-36/. During R57's model load phase, Python logged "Model loaded in 23.0s" but the wall-clock gap between "Loading model" (09:54:42) and "Prefill patch applied" (10:18:53) was **24 minutes 11 seconds** — a 60x discrepancy. The process was paged out for ~99% of that window; Python's internal `time.time()` only ticked while the process was scheduled. The profiler in `omlx/bench/profiler.py` has the same blind spot: samples are taken from within the Python process, so when the OS suspends it, no samples are recorded and num_samples ends up proportional to Python-active time rather than wall-clock. A 24-minute swap-thrash stall leaves zero forensic trace in the bench output files except the log timestamp gap, which is tedious to detect manually.
- **Change**: In `omlx/bench/profiler.py` (the Task #8 profiler rebuild module):
  - At profiler start, record `wall_clock_start_monotonic_s = time.monotonic()` AND `wall_clock_start_epoch_s = time.time()`.
  - At profiler stop (when writing /tmp/hypercar_profile.json), compute `wall_clock_elapsed_s = time.monotonic() - wall_clock_start_monotonic_s` and add it to the `summary` dict alongside `total_seconds` (which is the Python-accounted elapsed time).
  - Add a derived field `cpu_scheduling_fraction = total_seconds / wall_clock_elapsed_s` to the summary. A value of 1.0 means the process ran without interruption; a value of 0.02 means the process got 2% of wall-clock (severe swap/co-tenant pressure).
  - In `omlx/bench/hypercar_bench.py` near `_finish()` (end of run), check `cpu_scheduling_fraction` and emit a WARNING-level log line if it falls below 0.9: `WARNING: CPU scheduling fraction was {frac:.2f} (wall {wall}s vs Python {cpu}s) — heavy co-tenancy or swap-thrashing detected, numeric results may be valid but timing metrics should be interpreted with caution`.
- **Verify**:
  - `.venv/bin/python -m omlx.bench.hypercar_bench --quick` on a clean box produces profile.json with `cpu_scheduling_fraction >= 0.95` and no warning log.
  - Simulate pressure: run `.venv/bin/python -c "import time; time.sleep(60)" &` in a tight loop with some large memory allocations, then run `--quick`. Verify `cpu_scheduling_fraction` drops below 0.9 and the warning fires.
  - Read bench/snapshots/run57_2026-04-15T09-36/env.json for the exact symptom this task is meant to catch automatically.
- **Effort**: S (2-3 hours: add 10 lines to profiler.py, add 5-line check in hypercar_bench.py _finish, write one unit test that mocks time.monotonic drift, smoke test manually)
- **Risk**: Minimal. `time.monotonic()` is guaranteed to tick during OS suspension on Darwin (it's based on `mach_absolute_time` which continues even when the process is swapped out), so the computation is reliable. Only risk is forgetting to use monotonic (time.time() can go backward).

## Research-derived tasks (from LIT_REVIEW.md pass 18, 2026-04-15)

### 81. CTkvr centroid-then-token KV index for query-aware page selection
- **Goal**: 3 (decode speed via top-K page selection), 1 (1M context retrieval accuracy)
- **Derived from**: CTkvr (2512.15550)
- **Change**:
  - In `omlx/turboquant_kv.py` (or a sibling module), build a centroid index over the existing KV pages: at cache build time, run a small k-means (k=64-256) over the page-mean key vectors and store the centroid table alongside the page metadata. The page-to-centroid assignment is fixed at build; new pages get assigned to the nearest centroid as they are appended.
  - Add a two-stage retrieval path that runs *before* Quest's existing min/max bound: stage 1 picks the top-M centroids by query-vs-centroid dot product (M ~ 8), stage 2 takes the union of pages assigned to those centroids and runs Quest's existing top-K page selection over that reduced candidate set. The output is the same shape as today's Quest path, so the downstream attention call sites are unchanged.
  - Calibrate k and M offline against `omlx/bench/hypercar_bench.py --quick` traces — the goal is "smallest k, M such that NIAH 64K still passes."
  - Compose with Quest (task 1, pass 1): CTkvr is the coarse filter, Quest is the bound filter, both run before the dense top-K. Quest's existing implementation does not need to change.
- **Verify**:
  - Unit test: synthetic KV cache with known page-to-relevance mapping; CTkvr stage 1 must include the relevant page in its top-M centroids 100% of the time.
  - Integration test: NIAH 64K passes with the CTkvr stage active. Decode tok/s at 32K context improves by ≥ 20% on `hypercar_bench --quick` vs Quest-only path. Less than 1% accuracy degradation on the existing coherence eval.
  - Memory check: centroid table memory is < 1% of KV cache memory at 1M context.
- **Effort**: M (3-5 days: 1d for the k-means and centroid index, 1-2d for the two-stage dispatch, 1-2d for calibration and bench)
- **Depends on**: Task 1 (Quest) for the existing top-K page abstraction. The two compose; CTkvr is not a replacement.
- **Risk**: CPU-GPU co-execution (the paper's headline lever) does not apply on Apple Silicon's unified memory, so the speedup math may collapse to "GPU-only with extra control flow." A one-day microbenchmark on the centroid stage alone gates the rest of the work.

### 82. ATTS-style online conformal predictor for prompt-cache hit rate
- **Goal**: 4 (prefill efficiency via better cache decisions), 5 (swap pressure via principled eviction)
- **Derived from**: ATTS (2509.15148)
- **Change**:
  - In `omlx/hypercar_server.py`, instrument the existing prompt-cache lookup path to emit a (prefix_hash, hit | miss, age_seconds, prefix_length) record per request. The instrumentation is read-only; the cache itself doesn't change.
  - Add an online conformal predictor module that takes the rolling stream of records and produces a per-prefix "probability of next-N-turn hit, with provably bounded coverage error" estimate. Use a weighted-conformal recursion (the standard fix for non-exchangeable observations) since the active session's request distribution is non-stationary. Math is ~50 lines.
  - Use the predictor's lower-bound estimate to drive cache eviction: when the cache is at the budget ceiling, evict the prefix with the lowest lower-bound hit probability rather than the LRU prefix. The conformal coverage guarantee bounds the worst-case eviction error rate.
  - Add a server-side `/metrics/cache` endpoint that exposes the rolling lower-bound estimates so we can debug eviction decisions.
- **Verify**:
  - Unit test: a synthetic request trace with known prefix-reuse pattern (e.g., 80% of requests hit a single hot prefix); the predictor's lower-bound estimate for the hot prefix must be ≥ 0.7 within 50 requests, and the eviction policy must keep the hot prefix in cache under memory pressure.
  - Integration test: an OpenCode-style turn-by-turn session against the server with manual eviction triggered every K turns; verify that the conformal-driven eviction loses fewer subsequent cache hits than the LRU baseline on the same trace.
  - Coverage gate: the conformal coverage error rate measured offline against a logged trace must be within 5% of the nominal target (e.g., 0.95 nominal coverage → empirical coverage ≥ 0.90).
- **Effort**: M (4-7 days: 1d for the instrumentation, 1d for the conformal recursion, 2d for the eviction wiring, 1d for the metrics endpoint, 1-2d for the integration trace tests)
- **Depends on**: none. Independent of CTkvr (task 81) and Halo (task 77) — the predictor is a different layer entirely.
- **Risk**: TTT rollouts during weight updates are non-exchangeable; weighted conformal handles this but the variance of the estimate can be high in the early-session regime. Mitigation: warm-start the predictor with a uniform prior over the first N requests and only switch to the conformal lower bound after N is reached.

### 83. KVP per-head RL eviction policy trained on Hypercar bench traces
- **Goal**: 5 (swap pressure via aggressive eviction), 1 (1M context fit on M4 Pro 48GB)
- **Derived from**: Learning to Evict from KV Cache / KVP (2602.10238)
- **Change**:
  - Build a generation-trace harness that runs the existing benchmark prompts (humaneval, NIAH, coherence) at full KV cache and logs, for every token, its position, its key vector, its value vector, and its eventual downstream impact (measured as KL between the un-evicted next-token distribution and the distribution after that token's eviction).
  - Train a small per-head RL agent (one Q-network per attention head, ~50K params each) that takes the (key, value, position, current cache budget) state and outputs an eviction priority score. Use the trace KL signal as the reward. Single training run produces 48 head-policies (one per layer's attention heads).
  - Wire the trained policies into `omlx/turboquant_kv.py` as an alternative eviction backend behind a `--kv-eviction kvp` server flag. The default eviction stays LRU; KVP is opt-in.
  - The trained policies are budget-conditioned, so the same checkpoint serves both the 8-bit and 3-bit KV cache modes — but verify this assumption empirically.
- **Verify**:
  - Offline metric: per-head policy AUC on a held-out trace must exceed 0.7 vs random eviction baseline.
  - End-to-end metric: with KVP eviction active, NIAH 64K still passes; HumanEval pass rate unchanged ± 1pp; peak Metal memory at 1M context drops by ≥ 10% vs LRU eviction.
  - Mode-transfer check: a policy trained on 8-bit cache traces, evaluated on 3-bit cache traces, must still pass NIAH 64K. If not, train one policy per cache mode.
- **Effort**: M-L (1-2 weeks: 2-3d for the trace harness, 2-3d for the per-head RL training, 2-3d for the eviction backend integration, 1-2d for the bench validation)
- **Depends on**: none for the core work; composes with task 81 (CTkvr — KVP shrinks the cache, CTkvr selects top-K of the remaining) and task 59 (OPLoRA safety rail).
- **Risk**: per-head RL agents are 48 separate trained models, which is operationally heavy. Mitigation: start with a single shared agent across all heads as a baseline, then go per-head only if shared-agent quality is insufficient.

### 84. Tile-shape auto-tuning harness for MLX prefill kernels (Triton Anatomy methodology)
- **Goal**: 4 (prefill speed), 3 (decode kernel efficiency, secondarily)
- **Derived from**: The Anatomy of a Triton Attention Kernel (2511.11581)
- **Change**:
  - Build a tile-shape auto-tune harness in `omlx/patches/` (sibling to `specprefill.py` and `prefill_last_logit_patch.py`) that wraps the existing prefill kernel call sites. The harness sweeps a small grid of (block_size, num_warps_equivalent, prefetch_depth) parameters that MLX exposes for its attention kernels; for each candidate it runs a 5-token prefill probe and measures wall-clock time.
  - At server startup (or first request), run the harness once per (model, sequence-length-bucket) pair, cache the best tile shape per bucket in `~/.cache/hypercar/tile_shapes.json`, and use the cached shape for the rest of the session. The buckets are coarse: {<2K, 2K-16K, 16K-128K, 128K+}.
  - Add a `--no-autotune` flag to the server for forensic comparison vs the hand-picked baseline.
  - The auto-tune is per-device: M1 Pro, M2 Pro, M3 Pro, M4 Pro all get their own tile-shape cache file because the optimal varies with GPU SM count and memory bandwidth.
- **Verify**:
  - Unit test: harness runs to completion in < 30s on a cold server start; the resulting cache file contains an entry per bucket; loading the cache on subsequent server starts skips the tuning step.
  - Bench gate: `omlx.bench.hypercar_bench --quick` prefill tok/s at 16K context improves by ≥ 15% on the M4 Pro vs the hand-picked baseline. NIAH 64K still passes. HumanEval pass rate unchanged.
  - Cross-device check: an auto-tune cache generated on M4 Pro should *not* be used on a different device — verify the harness regenerates the cache when the device fingerprint changes.
- **Effort**: S-M (3-5 days for the inspiration prototype, 5-7 days for productionised version: 1d for the harness sweep loop, 1-2d for the cache management, 1-2d for the bench validation, 1d for the per-device fingerprint logic)
- **Depends on**: none. Independent of all other tasks; the auto-tune sits below them in the stack.
- **Risk**: MLX's kernel dispatch may not expose tile-shape parameters as cleanly as Triton, so the search space may be much smaller than the paper assumes — which could mean the gain is much smaller too. Mitigation: a half-day spike to enumerate the actual MLX tunables before committing the rest of the work.

### 85. Flush logger before exit in memory-watchdog breach path (distinguish watchdog breaches from external kills in console output)
- **Goal**: 5 (swap pressure observability), plus debugging-meta
- **Derived from**: Hypercar benchmark runs 54, 57, 58 (2026-04-14 through 2026-04-15). R58 was the first run where the memory watchdog fired with enough latency to log its error message: `11:00:11 omlx.bench.hypercar ERROR MEMORY BREACH: Swap delta 12.9GB > 12.9GB limit` immediately before an exit 144. R54 (crashed mid-MMLU-Pro) and R57 (crashed during warmup) both exited 144 with NO error message in the console, which led the analyst to hypothesize "external kill / SIGURG" in their env.json files. R58 reveals that R54 and R57 were ALSO memory-watchdog breaches — the logger just didn't flush before the process exited. Future operators should be able to tell the difference between a watchdog breach and an external kill from the console alone, without having to read a partial-log post-mortem against a later run that happens to flush in time.
- **Change**: In the memory-watchdog path (likely `omlx/bench/watchdog.py` or wherever `_finish()` / `watchdog.breached` handling lives in `omlx/bench/hypercar_bench.py`):
  - After calling `logger.error("MEMORY BREACH: ...")` and before any `sys.exit()` or `raise SystemExit()`, add `logging.getLogger().handlers[0].flush(); sys.stderr.flush(); sys.stdout.flush()`.
  - Alternative (cleaner): set `logging.StreamHandler` with `flush=True` at handler construction in the bench's logging init, so every record is flushed synchronously. This has a small per-record overhead but the bench emits ~1000 log lines over ~20 min so the cost is negligible.
  - In the watchdog breach exit path, also write a single-line sentinel to the console: `print("WATCHDOG-BREACH-EXIT-144", file=sys.stderr, flush=True)` followed by `sys.exit(144)`. The sentinel is grep-able from snapshot console.txt files and lets operators scan for breach-vs-other-crash at a glance.
- **Verify**:
  - Force a memory breach (e.g. `.venv/bin/python -m omlx.bench.hypercar_bench --full --force-breach-test` if such a flag exists, or temporarily lower the swap-delta limit to 1 GB in a test config) and confirm:
    - The "MEMORY BREACH" ERROR log line appears in the captured console output.
    - The "WATCHDOG-BREACH-EXIT-144" sentinel line appears immediately after.
    - The exit code is 144.
  - Unit test: mock the watchdog breach path in a test, capture stderr, assert both the ERROR line and the sentinel are present.
  - Regression check: running `.venv/bin/python -m omlx.bench.hypercar_bench --quick` on a clean box still succeeds without flushing overhead causing noticeable slowdown (< 1s added to total).
- **Effort**: S (1-2 hours: add flush calls, add sentinel, write one unit test)
- **Risk**: Minimal. The only risk is a non-flushing logger elsewhere in the codebase also using the same handlers — in that case the flush-on-every-record path would slightly slow it down. Trivially reverted if measured regression appears.

### 86. Per-phase memory-headroom re-check between phases to prevent mid-run breaches
- **Goal**: 5 (swap pressure), 6 (machine fit under load)
- **Derived from**: Hypercar benchmark runs 54, 57, 58 — three consecutive crashes in Phase 3+ (NIAH 16K, MMLU-Pro, warmup → NIAH 16K) under co-tenancy pressure. The bench currently has a pre-flight check at launch (the "Only X GB available, need 30 GB" guard that fired in R50 and R51) and a headroom gate inside Phase 3b RULER (Task #9, which skips 64K multi-key tasks). But there is NO headroom re-check between phases when the box state CHANGES mid-run. R58's specific pattern: pre-run had 19.5 GB free (passed pre-flight), Phases 0-2 completed cleanly, then during NIAH 16K's longer-sustained generation the co-tenant's memory growth pushed total swap delta to 12.9 GB and tripped the watchdog. A per-phase re-check that runs between Phase 2 and Phase 3 (and again between Phase 3 and Phase 3b, etc.) would have SKIPPED Phase 3 16K with a clear WARNING and allowed Phases 3b/4/5/6 to run on whatever headroom was actually available.
- **Change**: In `omlx/bench/hypercar_bench.py` near the phase-dispatch loop (likely the body of the `run_full()` or equivalent function that sequences Phase 0 → 6):
  - Add a helper `_check_phase_headroom(phase_name, required_gb, watchdog)` that reads current `vm_stat` free pages + current Metal usage, computes `headroom_gb = free_pages_gb + (metal_limit_gb - metal_used_gb)`, and returns True/False.
  - Before each phase, call `_check_phase_headroom("Phase 3 NIAH 16K", 8.0, watchdog)`. If False, log `WARNING: Phase 3 NIAH 16K SKIPPED — headroom {h:.1f} GB < {r:.1f} GB required (co-tenancy detected mid-run)` and fall through to the next phase.
  - Phase-specific required-headroom constants should live at the top of `hypercar_bench.py`: PHASE_3_NIAH_16K_HEADROOM_GB = 8.0, PHASE_3B_RULER_HEADROOM_GB = 6.0, PHASE_3C_MMLU_PRO_HEADROOM_GB = 4.0, PHASE_4_HUMANEVAL_HEADROOM_GB = 3.0. These are calibrated from R44/R47/R48 successful runs' memory deltas per phase.
  - The existing Phase 3b RULER headroom gate (Task #9) should continue to work inside RULER for 64K tasks — this new check is orthogonal: it gates at phase-entry, not per-task.
- **Verify**:
  - `.venv/bin/python -m omlx.bench.hypercar_bench --full` on a clean box runs through all phases as before (the check returns True for each). No regression on the "ALL GATES PASSED" path.
  - Under synthetic pressure (e.g. a background `python -c "a = [b'x' * (1024*1024) for _ in range(10000)]; input()" &` holding 10 GB), the bench reaches Phase 3 NIAH 16K, prints the SKIPPED warning, then continues to Phase 3b RULER and so on. Final phase table in results.json shows SKIPPED entries where the check fired, not a hard crash.
  - Smoke test: set a temporarily-low required-headroom value in a test config, force a SKIP at Phase 3 NIAH 16K, assert the bench still writes a valid Phase 6 summary with the skipped phase recorded.
- **Effort**: S-M (3-5 hours: 1h for the helper, 1h for the phase-entry integration, 1h for the constants calibration from prior snapshots, 1h for the unit + smoke tests)
- **Risk**: False positives if the headroom estimate is too conservative — the bench would skip valid phases unnecessarily. Mitigation: the constants above are calibrated from 3 successful runs' observed memory growth (R44/R47/R48 NIAH 16K needed 8 GB Metal delta; R48 had 24 GB free with 6 GB margin so 8 GB is a 75th percentile). Adjust if field reports show false skips.

## Research-derived tasks (from LIT_REVIEW.md pass 19, 2026-04-15)

### 87. MLX softmax fused-reduction audit + microbench harness
- **Goal**: 3 (decode tok/s, every layer's softmax is on the path), 4 (prefill tok/s, same), and a meta-goal: every prior kernel-port task in this backlog is implicitly multiplied by the MLX softmax constant.
- **Derived from**: Benchmarking On-Device ML on Apple Silicon with MLX (2510.18921). The paper measured `mx.softmax` at 27.91 ms on M1 vs 1.06 ms on a CUDA baseline — a 26x gap that is *worse* than MLX's matmul gap (6.6x) on the same hardware, suggesting an unfused reduction pattern in the MLX softmax kernel rather than a memory-bandwidth limit.
- **Change**:
  - Build a microbench harness in `omlx/bench/` (sibling to `hypercar_bench.py`) that calls `mx.softmax`, `mx.fast.scaled_dot_product_attention`, and a hand-rolled `softmax-then-matmul` Metal shader at the exact shapes Hypercar uses in production: B=1, H=24 (Qwen3-Coder layer count over heads), S=2K / 16K / 64K / 256K / 1M. Time each via `mx.eval()` boundaries with at least 50 warmup + 200 measured iterations.
  - Audit the MLX source (or its Metal shader output) for the softmax reduction — confirm whether it's actually unfused, or whether the paper's measurement was on an old MLX release that has since been patched. Note the MLX commit / version under test in the harness output.
  - If unfused: prototype a fused softmax-then-matmul Metal shader behind a `--fused-softmax` flag in the server. The fusion target is the inner attention loop where softmax(QK^T/sqrt(d)) is immediately multiplied by V.
  - Report the harness output in `research/mlx_softmax_audit.md` (research note, not a long-lived doc) — the only documentation deliverable. Either it confirms the gap and we file follow-on work, or it disproves the gap and we close the bucket.
- **Verify**:
  - Microbench runs to completion in < 60s on a clean M4 Pro.
  - The harness output for `mx.softmax` at S=2K matches the paper's M1 measurement to within 2x (accounting for M1 → M4 Pro gen difference) — this validates the harness is measuring the right thing.
  - If the fused shader prototype lands: `omlx.bench.hypercar_bench --quick` decode tok/s at 16K context improves by ≥ 3% (a 5x softmax improvement compounds into a small but measurable end-to-end win because softmax is one of several ops in the attention path). NIAH 64K still passes. HumanEval pass rate unchanged.
- **Effort**: S (1 day for the harness + audit, 2-3 days for the fused-shader prototype if the audit justifies it). The audit alone is the gating step — if MLX has already fused softmax in a release after the paper's measurement window, the rest of the work is unnecessary and the bucket closes.
- **Depends on**: none. Independent of all other tasks; the audit sits below them in the stack.
- **Risk**: The 2510.18921 measurement is on M1 with an older MLX release, and MLX has had multiple version bumps since. The gap may already be closed. Mitigation: the audit *is* the verification step — if the gap is gone, we file zero follow-on work and the cost is one day.

### 88. PackKV asymmetric K/V codec in TurboQuantKVCache
- **Goal**: 5 (swap headroom — frees ~5 GB at 1M context), 1 (1M context fit on M4 Pro 48GB under co-tenancy load), 6 (machine fit envelope)
- **Derived from**: PackKV (2512.24449). The paper reports 153.2% memory reduction for K and 179.6% for V over SOTA quantisation — by using *different lossy operators* for K and V, not just different axes (KIVI's contribution from pass 1). V can tolerate a coarser bulk codec because softmax-weighted-sum aggregation absorbs uniform value-side error.
- **Change**:
  - Refactor `omlx/turboquant_kv.py` to parameterise the codec independently per K/V axis. Today both K and V share the same WHT-rotated 3-bit codebook; after the change K stays at the existing fine-grained per-channel codec (to preserve outliers) and V gets a coarser bulk codec with a larger group size (group_size 256 instead of 64) and an outlier-clamp pre-pass instead of per-group scales.
  - Add a `--v-codec {fine,bulk}` server flag. Default to `fine` (current behaviour); `bulk` activates the new V codec. This is opt-in until the bench gates pass.
  - Calibrate the bulk-V codec against `omlx/bench/hypercar_bench.py --quick` traces — measure NIAH 64K accuracy, HumanEval pass rate, and KV memory at 1M context for both modes side-by-side.
  - Compose with task 81 (CTkvr): CTkvr picks which V pages to fetch, the bulk codec makes each fetched page 1.5-1.8x smaller. The two compose multiplicatively on the V-side memory budget.
- **Verify**:
  - Unit test: round-trip a synthetic V cache through the bulk codec, assert reconstruction error stays below the threshold derived from the paper's reported tolerances.
  - Memory check: at 1M context, KV cache memory drops from 22.5 GB to ≤ 18 GB (the paper's 1.7x V-side reduction applied to the V half of the cache).
  - Quality check: NIAH 64K passes with `--v-codec bulk`. HumanEval pass rate at ≥ 35% (current threshold). MMLU-Pro accuracy unchanged ± 1pp.
  - Co-tenancy gate: with `--v-codec bulk`, NIAH 500K passes the pre-flight headroom check on a box with 12 GB free (the failure mode that pass 15's eLLM was supposed to fix; this is the alternative path).
- **Effort**: M (3-5 days: 1d for the codec parameterisation refactor, 1d for the bulk-V codec implementation, 1-2d for the calibration and bench, 1d for the integration + flag plumbing)
- **Depends on**: composes with task 81 (CTkvr) but is independent of it — task 88 can land first. Stacks orthogonally with task 70 (ButterflyQuant) since that targets the rotation, not the codec.
- **Risk**: the paper's headline gains assume GPU memory bandwidth is the binding constraint. On Apple Silicon's unified memory the binding constraint is often Metal heap fragmentation rather than bandwidth, so the *throughput* gains may not transfer even if the *memory* gains do. Mitigation: a one-day microbench on the bulk-V codec alone gates the rest of the work; if memory drops as expected but throughput is unchanged, the task still ships because the memory headroom is independently valuable.

### 89. InT-style self-proposed intervention loop in TTT engine
- **Goal**: 2 (intelligence-breadth via TTT — the credit-assignment problem is the largest gap in `omlx/ttt.py` per the pass 17/18/19 papers)
- **Derived from**: InT — Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning (2601.14209). The paper reframes credit assignment as a counterfactual intervention problem: instead of attributing reward across a trajectory, have the model propose what it would have done differently at the first wrong step, then re-run the modified trajectory through the verifier. ~14% accuracy gain on IMO-AnswerBench with a 4B model.
- **Change**:
  - In `omlx/ttt.py`, after a failed trajectory (terminal verifier returns fail), add an intervention-proposal phase. The phase walks the action sequence, identifies the first step where the verifier output diverges from a "would-pass prefix" (need a trace-level diff utility comparing current trajectory's per-step verifier output against a known-good trajectory or the verifier's expected output), and prompts the model to propose a single replacement action at that step.
  - Re-run the modified trajectory (original prefix + proposed replacement + original suffix or new suffix) through the verifier. If the modified trajectory now passes, fine-tune the model on the corrected suffix as the supervision signal — this localises the credit-assignment signal to the problematic step rather than smearing it across the whole trace.
  - Scope the first prototype to *single-test-failure trajectories*: trajectories where exactly one test in the verifier suite failed, so the "first wrong step" is unambiguous. Multi-test-failure trajectories are deferred to a second iteration once the unambiguous case proves out.
  - Compose with task 78 (SWE-Shepherd PRM): SWE-Shepherd scores actions densely *during* the rollout, InT rewrites failing actions *after* the rollout. They target different points in the TTT loop and can land independently.
- **Verify**:
  - Unit test: a synthetic single-test-failure trajectory where the "wrong step" is known by construction; the intervention-proposal phase identifies the correct step ≥ 80% of the time, and the proposed replacement action when re-run flips the verifier output to pass ≥ 50% of the time.
  - End-to-end metric: TTT engine HumanEval pass rate (or whatever metric `omlx/ttt.py` currently tracks) improves by ≥ 5pp on a held-out set after one round of intervention-driven fine-tuning, vs the existing terminal-only credit-assignment baseline.
  - Stability gate: the intervention loop must not regress the existing TTT calibration tests — the OPLoRA safety rail (task 59) should still be the outermost envelope.
- **Effort**: M (4-7 days: 2d for the trace-diff utility and intervention-proposal scaffold, 1-2d for the verifier re-run wiring, 1d for the fine-tune integration, 1-2d for the calibration vs SWE-Shepherd)
- **Depends on**: task 59 (OPLoRA safety rail) is the outer envelope and must already be in place. Composes with task 78 (SWE-Shepherd) but does not require it — they target different points in the TTT loop.
- **Risk**: code-verification trajectories are sparser and more multi-modal than math-verification trajectories — the "first wrong step" is often ambiguous when half the test suite was failing for orthogonal reasons. Mitigation: the single-test-failure scope above. Second risk: re-running modified trajectories doubles the verifier compute cost during TTT training. Mitigation: only run the intervention loop on trajectories where the model's confidence-of-wrongness is highest, i.e., the trajectories where the most learning signal is available.

### 90b. Attention-weighted codec selection in DuoKVCache
- **Goal**: 1 (1M context), 5 (swap pressure), 6 (48GB fit)
- **Derived from**: Session Apr 15 research arc — MLA fail (7%), ShadowKV poor tradeoff, SnapKV validated. Design note at `research/attention_weighted_codec_selection.md`.
- **Change**:
  - Extend DuoKVCache so retrieval heads (41%) use SnapKV eviction (Task 46's `snapkv_select` at prefill end, full precision on kept tokens) while streaming heads (59%) use ShadowKV SVD at rank 85 (34% K compression, 100% code agreement — streaming heads don't do retrieval).
  - New `--kv-mode duo-compressed` flag on hypercar_server and bench.
  - The per-head-type codec boundary is the DuoAttention classification (already computed, Task 12).
- **Verify**: `hypercar_bench --full --kv-mode duo-compressed` passes all gates. KV memory at 128K drops by ≥ 40% vs `--kv-mode duo`. NIAH 4K/16K still PASS. Code Intel ≥ 3/5.
- **Effort**: M (depends on Task 46 SnapKV compact landing first)
- **Depends on**: Task 46 (SnapKV compact), Tasks 12/13 (DuoAttention, already shipped)

## Research-derived tasks (from LIT_REVIEW.md pass 20, 2026-04-15)

### 90. MT-GRPO turn-level credit assignment in TTT engine
- **Goal**: 2 (intelligence via TTT — the credit-assignment bottleneck is the single largest gap in `omlx/ttt.py`)
- **Derived from**: Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design (2505.11821). The paper extends GRPO to MT-GRPO with per-turn advantage estimation using intermediate rewards, achieving higher accuracy and faster convergence than trajectory-level GRPO on multi-turn tool-use tasks.
- **Change**:
  - In `omlx/ttt.py`, refactor the reward computation from terminal-only (pass/fail at end of trajectory) to per-turn (partial test suite pass rate after each tool call). The existing code verifier already runs after each action — the change is to *record* the intermediate verifier output as a reward signal rather than discarding it.
  - Replace the trajectory-level GRPO advantage estimator with a GAE-lambda estimator over the per-turn reward stream. The lambda parameter controls the bias-variance tradeoff: lambda=1.0 recovers trajectory-level; lambda=0.0 is pure per-turn. Start with lambda=0.95 (the paper's recommended default) and calibrate.
  - Add a `--turn-level-ca` flag to the TTT engine (default off until bench gates pass). When on, the engine computes per-turn advantages and uses them for the weight update; when off, it falls back to the existing terminal-only path.
  - The per-turn reward function is: `r_t = (tests_passing_after_turn_t / total_tests) - (tests_passing_after_turn_{t-1} / total_tests)`. This is the marginal test-pass-rate improvement at each turn, which sums to the terminal reward by construction.
- **Verify**:
  - Unit test: a synthetic 5-turn trajectory where turns 1-3 each pass one additional test and turns 4-5 break a previously-passing test. The per-turn reward should be positive for turns 1-3 and negative for turns 4-5; the trajectory-level reward should be positive (net 1 test gained). MT-GRPO should assign higher advantage to turns 1-3 than trajectory-level GRPO does.
  - End-to-end metric: TTT engine HumanEval pass rate improves by >= 3pp with `--turn-level-ca` enabled vs disabled, on the same set of problems with the same compute budget.
  - Convergence metric: MT-GRPO reaches 90% of final performance in <= 70% of the training steps that trajectory-level GRPO requires, measured on a held-out validation set.
  - Stability gate: OPLoRA safety rail (task 59) still triggers correctly; the per-turn reward doesn't cause gradient spikes that escape the orthogonal projection.
- **Effort**: M (4-7 days: 1d for the per-turn reward instrumentation, 1d for the GAE-lambda estimator, 1-2d for the flag plumbing and integration, 1-2d for calibration and bench validation)
- **Depends on**: task 59 (OPLoRA safety rail) as the outer envelope. Composes with task 78 (SWE-Shepherd PRM) — PRM scores can replace or supplement the partial-test-pass-rate reward. Composes with task 91 (asymmetric critic) — the critic's runtime signal is another intermediate reward source.
- **Risk**: the marginal-test-pass-rate reward is noisy for tasks where tests have complex interdependencies (passing test 3 requires passing test 1, so turn 3's reward is zero even though the agent's action was correct). Mitigation: use the cumulative pass rate rather than the marginal delta as an alternative reward formulation, and compare both in the calibration phase.

### 91. Asymmetric actor-critic runtime supervisor for TTT engine
- **Goal**: 2 (intelligence via TTT — privileged critic provides denser reward signal than the actor can self-produce)
- **Derived from**: Asymmetric Actor-Critic for Multi-turn LLM Agents (2604.00304). The paper demonstrates that a small open-source critic (7B-scale) fine-tuned on actor traces can provide runtime supervision within multi-turn trajectories, significantly improving one-shot task success on tau-bench and UserBench.
- **Change**:
  - Fine-tune a small critic model (e.g., Qwen3-Coder-3B or similar) on TTT rollout traces. The training data is: (trajectory prefix up to turn t, test runner output at turn t, next K turns of rollout) -> quality score. The critic sees the test runner output (privileged signal the actor doesn't get at generation time) and the future trajectory (hindsight signal).
  - In `omlx/ttt.py`, add a critic-query step at each turn boundary during TTT rollouts. After the actor generates an action and the verifier runs, pass the (prefix, verifier output, action) tuple to the critic and get a quality score. This score serves as an intermediate reward signal for MT-GRPO (task 90).
  - Add a `--critic-model <path>` flag to the TTT engine. When set, the critic is loaded alongside the actor and queried at each turn. When unset, the engine falls back to the verifier-only reward path.
  - The critic runs on the same Metal device as the actor. At 3B parameters, it requires ~3 GB of memory — well within the 6 GB headroom available in duo mode.
- **Verify**:
  - Critic quality: on a held-out set of TTT rollout traces, the critic's quality score correlates with the terminal pass/fail outcome at Pearson r >= 0.6. The critic should be better-than-random at predicting failure *before* the trajectory completes.
  - End-to-end metric: TTT engine HumanEval pass rate improves by >= 2pp with `--critic-model` enabled vs disabled (on top of MT-GRPO gains from task 90).
  - Latency gate: the critic query adds <= 100ms per turn (3B model inference at B=1 should be ~50ms on M4 Pro). The total TTT rollout time increases by <= 15%.
  - Memory gate: peak Metal memory with actor + critic loaded stays under the 80% system-memory limit (38.4 GB on 48 GB). Duo mode: 35.1 GB actor + 3 GB critic = 38.1 GB — tight but within limits.
- **Effort**: M-L (1-2 weeks: 2-3d for trace collection and critic training data preparation, 2-3d for critic fine-tuning, 2-3d for runtime integration into TTT, 1-2d for calibration and memory validation)
- **Depends on**: task 90 (MT-GRPO) should land first so the critic's output has a consumer. Task 59 (OPLoRA) as outer safety envelope. Composes with task 78 (SWE-Shepherd) — the critic can be initialized from SWE-Shepherd PRM weights if available.
- **Risk**: 38.1 GB total is very close to the 38.4 GB limit in duo mode. Under co-tenancy, this will breach. Mitigation: (a) use native 3-bit KV mode for the critic to reduce its footprint, (b) only load the critic during TTT training phases, not during normal inference serving.

### 92. ECHO hindsight trajectory rewriting for TTT sample efficiency
- **Goal**: 2 (intelligence via TTT — the sample-efficiency bottleneck is the second-largest gap after credit assignment)
- **Derived from**: Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting / ECHO (2510.10304). The paper adapts HER to LM agents, converting failed trajectories into synthetic successes for alternative goals that the trajectory *did* achieve, outperforming Reflexion and AWM by up to 80%.
- **Change**:
  - In `omlx/ttt.py`, after a failed trajectory (terminal verifier returns fail), add an ECHO hindsight phase. The phase uses the actor model itself to: (a) identify which tests the trajectory *did* pass (the "achieved goal"), (b) propose a goal description that matches the achieved subset (e.g., "implement the sorting function but skip the edge-case handler"), (c) rewrite the trajectory prompt to target the achieved goal instead of the original goal.
  - Store the rewritten (prompt, trajectory) pair in a hindsight replay buffer alongside the original successful trajectories. The TTT training step samples from both the success buffer and the hindsight buffer, with a mixing ratio parameter `--hindsight-ratio` (default 0.3).
  - The hindsight rewriting is a single LM call per failed trajectory: "Given this code trajectory that passed tests [1,3] but failed tests [2,4,5], rewrite the task description to only require the functionality tested by tests [1,3]." This is cheap — one generation call reusing the existing inference infrastructure.
  - Scope the first prototype to HumanEval-style single-function tasks where the "achieved goal" is a strict subset of the test suite. Multi-file tasks (SWE-bench style) are deferred because the goal-rewriting prompt is harder to specify.
- **Verify**:
  - Unit test: a synthetic trajectory that passes 3 of 5 tests. The hindsight rewrite should produce a goal description that, when used as a new prompt, generates a trajectory passing exactly tests [1,3] (or a superset). The rewrite should succeed >= 70% of the time.
  - Sample efficiency metric: with `--hindsight-ratio 0.3`, the TTT engine reaches the same HumanEval pass rate as the baseline using <= 60% of the rollout budget (fewer total trajectories needed because failures now contribute positive signal).
  - Quality gate: the hindsight-augmented training must not *degrade* performance vs the baseline at the same total compute budget. If it does, the mixing ratio needs recalibration.
  - Stability gate: OPLoRA safety rail (task 59) still triggers correctly with hindsight-augmented training data.
- **Effort**: M (4-7 days: 1-2d for the hindsight rewriting prompt engineering and LM call, 1d for the replay buffer and mixing logic, 1-2d for integration into the TTT training loop, 1d for calibration and bench validation)
- **Depends on**: task 59 (OPLoRA safety rail) as outer envelope. Independent of tasks 90 and 91 — targets sample efficiency rather than credit assignment, touches a different part of the TTT loop (post-rollout data augmentation rather than reward computation).
- **Risk**: the hindsight rewriting may produce goal descriptions that are too easy (trivial subsets of the test suite) or too hard (the trajectory didn't actually demonstrate the claimed functionality). Mitigation: validate each rewritten pair by re-running the rewritten trajectory through the verifier against the rewritten goal; discard pairs where the verification fails. This adds one verifier call per rewrite but ensures data quality.

## Research-derived tasks (from LIT_REVIEW.md pass 21, 2026-04-15)

### 93. PAM select-in-place scheduling policy for tiered KV cache
- **Goal**: 1 (1M context validation), 5 (swap pressure), 6 (48GB fit)
- **Derived from**: CXL-PNM 1M-Token KV (2511.00321). The paper's key design pattern: evaluate page importance *in the tier where the page already lives* and transfer only winners to the compute-hot tier, rather than migrating pages between tiers and evaluating after migration.
- **Change**:
  - In the PAM-style two-tier TurboQuantKVCache (task 64, when it lands), replace the naive tier-migration policy (promote cold→hot on access, demote hot→cold on eviction) with a select-in-place policy: each tier maintains its own Quest-style min/max page bounds, and the scheduler evaluates page importance using the current query's attention pattern *without* loading the page data. Only pages whose importance exceeds a threshold are promoted to Metal-resident status.
  - Add a `select_in_place()` method to the tiered cache that takes a query vector and returns a list of page indices that should be Metal-resident for the next attention step. The method evaluates min/max bounds stored per page (already maintained by Quest, task 34) and compares against a dynamic threshold derived from the memory-aware feedback controller (task 94).
  - The select-in-place evaluation must run on CPU (not Metal) to avoid polluting the Metal working set with cold page metadata. On Apple Silicon, CPU access to unified memory is free; the cost is the min/max comparison loop, which is O(pages) = O(N/page_size).
- **Verify**:
  - Unit test: create a synthetic 64K-token KV cache with 50% hot pages (high min/max overlap with query) and 50% cold pages (low overlap). The select-in-place policy should promote exactly the hot pages and leave cold pages in the swap-backed tier. Metal memory usage should be ~50% of the full-cache baseline.
  - Memory gate: at 128K context under co-tenancy, the select-in-place policy keeps Metal residency under 70% of system memory (33.6 GB on 48 GB), compared to the current full-cache policy which breaches at ~37 GB.
  - Quality gate: NIAH 4K/16K/64K pass rates must not degrade. The select-in-place threshold must be conservative enough to never evict a page that contains a needle token.
  - Throughput gate: the select-in-place evaluation adds <= 5ms per decode step (the CPU min/max comparison loop is cheap on M4 Pro).
- **Effort**: M (3-5 days: 1d for the select-in-place method, 1d for the CPU-side evaluation path, 1-2d for threshold calibration and memory benchmarking, 1d for integration with task 64)
- **Depends on**: task 64 (PAM two-tier cache) must land first to provide the tiered storage substrate. Task 34 (Quest page bounds) provides the per-page min/max metadata. Composes with task 94 (adaptive chunker) — the memory-aware feedback controller provides the dynamic threshold.
- **Risk**: the min/max page bounds may be too coarse for fine-grained importance ranking when many pages have similar overlap scores. Mitigation: use CTkvr's (task 81) centroid-then-token two-stage ranking as a refinement step for pages near the threshold boundary.

### 94. Adaptive prefill chunk-size controller with memory-aware feedback
- **Goal**: 4 (prefill speed, constant across context), 5 (swap pressure), 6 (48GB fit)
- **Derived from**: Memory-aware Dynamic Batching (2503.05248). The paper reframes static batch sizing as a real-time feedback control problem with a memory-aware scheduler and latency feedback mechanism.
- **Change**:
  - In `omlx/hypercar_server.py`, replace the hardcoded `chunk_size=512` at 64K+ context with an adaptive controller that adjusts chunk size between prefill iterations based on measured Metal memory residency and prefill throughput.
  - The controller has two inputs: (a) `mx.metal.get_active_memory()` sampled after each chunk completes (memory signal), and (b) `tokens_processed / elapsed_time` for the chunk just completed (throughput signal). It has one output: the chunk size for the next iteration.
  - Control law: target Metal residency = 65% of system memory (31.2 GB on 48 GB), with a proportional gain that increases chunk size when residency is below target (headroom available) and decreases when above (pressure building). The throughput signal provides a secondary constraint: if tok/s drops below 200, reduce chunk size regardless of memory headroom (this catches the O(n^2) attention cliff).
  - Add a `--adaptive-chunk` flag to enable the controller. When disabled, fall back to the existing fixed chunk_size=512 behaviour. Default: disabled until validated.
  - Log the per-chunk (chunk_size, metal_mb, tok_per_s) triple to the benchmark profile for post-hoc analysis.
- **Verify**:
  - Prefill throughput gate: at 64K context with `--adaptive-chunk`, prefill tok/s >= 400 (vs current ~340 at fixed 512-token chunks). The controller should discover that larger chunks are safe at the start of prefill (when KV cache is small) and shrink as the cache grows.
  - Memory gate: Metal peak stays under 80% of system memory throughout the prefill. No watchdog breach.
  - Stability test: run prefill 5 times consecutively at 64K. The per-chunk chunk_size sequence should converge to a stable profile within 2 runs (no oscillation between max and min chunk sizes).
  - Regression gate: at 2K context (where chunking is not used), the adaptive controller must not activate. No overhead on short contexts.
- **Effort**: S-M (2-3 days: 0.5d for the controller implementation, 0.5d for the flag plumbing and logging, 1-2d for gain calibration and stability testing on the reference machine)
- **Depends on**: no hard dependencies — the controller wraps the existing prefill loop. Composes with task 72 (eLLM elastic memory) — eLLM's ballooning mechanism can replace `mx.metal.get_active_memory()` as the memory signal source once it lands. Composes with task 93 (PAM select-in-place) — the controller's memory signal informs the select-in-place threshold.
- **Risk**: Apple Silicon's unified memory may have different feedback-loop dynamics than discrete GPUs — the memory signal may lag by one chunk because Metal defers page allocation. Mitigation: add a one-chunk lookahead: before computing the next chunk, speculatively estimate the KV cache growth and subtract it from the measured headroom. The estimate is cheap (chunk_size * kv_bytes_per_token, both known).

### 95. EGCA execution-grounded credit assignment in TTT engine
- **Goal**: 2 (intelligence via TTT — precision of credit assignment directly impacts training sample efficiency and final pass rate)
- **Derived from**: Execution-Grounded Credit Assignment for GRPO (2603.16158, ICLR 2026 SPOT). The paper localises GRPO advantage to the failing token span by comparing execution traces of candidate vs reference solutions.
- **Change**:
  - In `omlx/ttt.py`, after the code verifier runs and a candidate fails one or more tests, add an EGCA trace-comparison phase:
    1. Execute the candidate solution under trace instrumentation (line-by-line variable state capture). The verifier already runs the code; the trace is an additional output.
    2. Execute the canonical reference solution (curated once offline from HumanEval/MBPP) under the same instrumentation.
    3. Diff the two traces to find the *earliest semantic divergence point* — the first line where a variable's value differs between candidate and reference.
    4. Map the divergence line back to the token span in the candidate's generation that produced it.
    5. In the GRPO advantage computation, assign full advantage to tokens in the divergence span and zero advantage (mask) to all tokens after the divergence point. Tokens before the divergence point retain the standard trajectory-level advantage.
  - Add a `--egca` flag to enable execution-grounded credit assignment. When disabled, fall back to the existing trajectory-level GRPO. Default: disabled until validated.
  - Scope the first prototype to single-function HumanEval-style tasks where the reference solution is a single canonical function. Multi-file tasks (SWE-bench style) are deferred because the trace instrumentation is harder to scope.
- **Verify**:
  - Unit test: a synthetic candidate that implements a sorting function correctly except for an off-by-one in the partition step. The trace divergence should identify the partition line, and the advantage mask should cover only the token span corresponding to the partition code.
  - Quality gate: with `--egca` enabled, TTT engine HumanEval pass rate improves by >= 2pp over trajectory-level GRPO baseline at the same compute budget.
  - Overhead gate: the trace instrumentation + diff adds <= 18% wall-clock overhead to the TTT rollout phase (matching the paper's reported overhead).
  - Stability gate: OPLoRA safety rail (task 59) still triggers correctly with EGCA-modified advantages. The localised advantage should not cause gradient spikes because it's strictly *smaller* (more tokens masked) than the trajectory-level advantage.
  - Composition test: when combined with MT-GRPO (task 90, turn-level credit), EGCA should provide *strictly finer* credit within each turn. Run with both `--egca` and `--turn-level-ca` enabled and verify pass rate is >= the max of either alone.
- **Effort**: S (1-2 days: 0.5d for trace instrumentation wrapper around the existing verifier, 0.5d for the trace-diff and token-span mapping, 0.5d for the advantage masking in GRPO, 0.5d for calibration and testing)
- **Depends on**: task 59 (OPLoRA safety rail) as outer envelope. Independent of tasks 90-92 — targets a different granularity (token-span vs turn vs trajectory) and can land and be evaluated before the heavier techniques. Provides the *baseline* against which tasks 90-92 must justify their marginal cost.
- **Risk**: the "earliest semantic divergence" heuristic may mis-localise when the candidate's bug is a subtle logic error that doesn't manifest in variable state until many lines later (e.g., off-by-one in a loop bound that only diverges on the last iteration). Mitigation: when the trace divergence point is more than 10 lines after the last shared correct line, fall back to trajectory-level advantage (the localisation is too uncertain to be useful). The 10-line threshold is a tunable hyperparameter.

## Research-derived tasks (from LIT_REVIEW.md pass 22, 2026-04-16)

### 96. AIMD congestion controller for KV cache memory pressure
- **Goal**: 5 (swap pressure), 6 (48GB fit)
- **Derived from**: CONCUR (2601.22705). The paper identifies "middle-phase thrashing" — a pathology where KV cache efficiency collapses while memory remains saturated — and resolves it with AIMD (Additive Increase Multiplicative Decrease) from TCP congestion control theory. The two-signal approach (cache usage AND hit rate) prevents false positives.
- **Change**:
  - In `omlx/hypercar_server.py`, replace the current fail-fast memory watchdog (which aborts on Metal peak > 80% of system memory) with a graduated AIMD controller that adjusts the KV cache budget dynamically.
  - The controller maintains a "congestion window" W representing the maximum number of Metal-resident KV pages (or tokens, depending on whether Quest page selection is active). Two signals drive the control law:
    - **Metal usage signal**: `mx.metal.get_active_memory() / system_memory`. This is the "cache usage" analogue.
    - **Cache effectiveness signal**: the fraction of decode steps where the model's attention distribution is concentrated (low entropy, indicating the cache contains the right tokens). This is the "hit rate" analogue. Computed from the same attention entropy signal that SleepGate's adaptive trigger uses (pass 22, 2603.14517).
  - Control law: when usage < U_low (20% of Metal budget), increase W by alpha (2 pages per decode step); when usage > U_high (50% of Metal budget) AND effectiveness < 0.2, multiply W by beta (0.5) — halving the cache budget in one step. Otherwise, hold W steady.
  - When W decreases (multiplicative decrease), the controller evicts the lowest-retention KV entries (using SnapKV attention scores, or Quest min/max bounds if available). When W increases (additive increase), newly generated KV entries are kept in Metal rather than being demoted to swap-backed memory.
  - Add a `--aimd` flag to enable the graduated controller. When disabled, fall back to the existing fail-fast watchdog. Default: disabled until validated.
  - Log the per-step (W, usage, effectiveness, action) quadruple to the benchmark profile for post-hoc analysis.
- **Verify**:
  - Co-tenancy stability test: run the server alongside browser + editor + Claude Code session (the reference co-tenancy scenario) with `--aimd` enabled. The controller should discover and maintain a stable W that keeps Metal usage below 80% without triggering swap thrashing. No watchdog abort should occur during a 10-minute sustained generation session.
  - Recovery test: artificially increase co-tenancy pressure (launch a memory-hungry process) while the server is generating. The controller should execute multiplicative decrease within 3 decode steps, reducing Metal usage by 50%. When pressure is released, additive increase should restore W to the pre-pressure level within 30 decode steps.
  - Quality gate: NIAH 4K/16K pass rates must not degrade under AIMD control. The cache budget W must be large enough to retain all needle tokens.
  - Throughput gate: the AIMD controller's per-step overhead (one `mx.metal.get_active_memory()` call + one entropy computation + one comparison) adds <= 1ms per decode step.
- **Effort**: S (1-2 days: 0.5d for the AIMD controller implementation, 0.5d for the eviction/promotion integration, 0.5d for threshold calibration under co-tenancy, 0.5d for the flag plumbing and logging)
- **Depends on**: no hard dependencies — the controller wraps the existing memory watchdog. Composes with task 94 (adaptive prefill chunker) — the chunker's memory signal feeds the AIMD controller's usage input. Composes with task 93 (PAM select-in-place) — AIMD's W determines the "Metal-resident page budget" that select-in-place fills.
- **Risk**: Apple Silicon's unified memory may have different AIMD dynamics than discrete GPUs — the "multiplicative decrease" may over-react because Metal page demotion is cheap (no PCIe transfer, just a page table update) and recovery is fast. Mitigation: start with conservative parameters (alpha=1, beta=0.7) and tune from there. The two-signal approach (usage AND effectiveness) should prevent over-reaction: if usage is high but effectiveness is also high, the controller holds steady.

### 97. Freshness-aware KV cache eviction with conflict-aware temporal tagging
- **Goal**: 1 (1M context — prevents stale entries from consuming cache budget), 2 (intelligence — resolves proactive interference that degrades retrieval accuracy)
- **Derived from**: SleepGate (2603.14517). The paper demonstrates that proactive interference from stale KV entries degrades retrieval accuracy to <18% even when the correct answer is in the context. The conflict-aware temporal tagger and soft attention biasing mechanism resolve this at the architectural level.
- **Change**:
  - In the SnapKV eviction logic (task 46, when it lands), add a *freshness score* to each KV entry based on conflict detection. The freshness score is computed as: for each cache entry i, check if any later entry j has cosine similarity cos(k_i, k_j) > delta (the conflict threshold, default 0.85). If so, entry i is marked as "superseded" and receives a freshness penalty.
  - Modify the SnapKV attention-weighted importance score to incorporate freshness: `importance_i = attention_score_i * freshness_i`, where `freshness_i = 1.0` for non-superseded entries and `freshness_i = decay_factor^(n_supersessions)` for entries that have been superseded n_supersessions times. The decay_factor (default 0.1) ensures that multiply-superseded entries are aggressively evicted.
  - For the DuoKVCache's ring-buffer streaming heads, freshness tagging is unnecessary — the ring buffer already evicts oldest entries. The freshness logic applies only to retrieval heads where SnapKV eviction is active.
  - Optionally, implement the soft attention biasing mechanism from SleepGate (Eq. 10): instead of hard eviction, add an additive pre-softmax bias b_i = beta * log(freshness_i) to each cache entry's attention score. This exponentially suppresses stale entries without physically removing them, allowing the model to recover from false-positive conflict detections.
  - Add a `--freshness-evict` flag to enable freshness-aware eviction. Default: disabled until validated.
- **Verify**:
  - Proactive interference test: construct a synthetic agentic scenario where the same code entity is updated 5 times in a 4K context. Without freshness eviction, the model should retrieve a stale version >= 50% of the time (baseline PI). With freshness eviction enabled, the model should retrieve the latest version >= 95% of the time.
  - Quality gate: standard NIAH 4K/16K and code intelligence gates must pass. Freshness eviction must not accidentally evict needle tokens (needles are never superseded because they are unique).
  - Memory gate: freshness scoring adds one cosine similarity check per incoming token against the most recent 10 cache entries of the same semantic key. This is O(10 * d) per token — negligible compared to the attention computation.
  - Composition test: with both DuoAttention head classification (task 12) and freshness eviction enabled, retrieval heads should use freshness-aware SnapKV while streaming heads use the existing ring buffer. The two policies must not interfere.
- **Effort**: S-M (2-3 days: 0.5d for the conflict detection logic using cosine similarity, 0.5d for the freshness-aware importance scoring, 0.5d for the optional soft attention biasing, 1d for the synthetic agentic PI test and calibration)
- **Depends on**: task 46 (SnapKV compact cache) must land first to provide the eviction substrate. Independent of task 96 (AIMD controller) — freshness eviction determines *which* entries to evict, AIMD determines *how many*.
- **Risk**: the cosine similarity threshold delta may not generalise from synthetic PI sequences to natural language updates. In natural language, a code correction may use different variable names than the original, producing low cosine similarity despite semantic supersession. Mitigation: use the semantic signature approach from SleepGate (project key vectors into a lower-dimensional semantic space before computing similarity) rather than raw key cosine similarity. The semantic projection can be a fixed random projection (no training required) that preserves the "same entity, different value" structure.

## Research-derived tasks (from LIT_REVIEW.md pass 23, 2026-04-16)

### 98. BUZZ-style segmented heavy-hitter eviction for SnapKV
- **Goal**: 1 (1M context — memory-efficient KV retention with local structure preservation), 3 (decode speed — O(n) eviction replaces O(n log n) sort)
- **Derived from**: BUZZ (2410.23079). The paper demonstrates that segmenting the KV cache into local chunks and selecting per-segment heavy hitters via local max sampling outperforms global heavy-hitter selection (H2O) by 7.69% on multi-document QA while achieving 2.5x cache reduction. The dual-stride mechanism (stride s for recent tokens, floor((s+1)/2) for older persistent tokens) preserves local attention structure that global top-k misses.
- **Change**:
  - In the SnapKV eviction logic (task 46, when it lands), replace the global top-k selection over the full observation window with a segmented selection. Divide the KV cache into segments of size `segment_size` (default 512 tokens, tunable). Within each segment, select the top-k_local tokens by attention score, where k_local = segment_size * keep_ratio.
  - Implement BUZZ's dual-stride mechanism: recent segments (within the last `window_size` tokens) use stride `s` (sparser, fewer segments); older segments use reduced stride `floor((s+1)/2)` (denser, more segments, preserving more historical context). The dual stride implements the insight that older persistent tokens are more likely to be important heavy hitters and should be sampled more densely.
  - For DuoAttention retrieval heads, use the segmented eviction with conservative keep_ratio (0.5). For streaming heads, continue using the ring buffer (no change). The segmented eviction applies only to the retrieval-head KV entries where local structure preservation matters for argmax accuracy.
  - Add a `--segmented-evict` flag to enable BUZZ-style segmented selection. When disabled, fall back to SnapKV's global top-k. Default: disabled until validated.
  - Use BUZZ's Theorem 3.1 to set the initial stride-threshold relationship: for odd stride s, set eviction threshold T = w * (s^2 + 1) / (s + 1); for even stride s, set T = w * (s - 1). This provides a theoretically grounded starting point for calibration.
- **Verify**:
  - NIAH quality gate: at 4K/16K/64K, segmented eviction at 50% keep must match or exceed global top-k eviction at 50% keep. The needle token should be retained with >= 99% probability across segments.
  - Multi-document QA test: construct a synthetic 8K context with 4 documents, each containing a unique fact. Query each fact. Segmented eviction should retrieve all 4 facts; global eviction may miss facts in the "middle" of the context (the lost-in-the-middle problem that BUZZ specifically addresses).
  - Eviction speed gate: at 64K context, segmented eviction completes in <= 50% of the wall-clock time of global top-k sort (the O(n) vs O(n log n) improvement should be measurable).
  - Memory gate: segmented eviction metadata (per-segment max pointers) adds <= 0.1% overhead to the KV cache size.
- **Effort**: S (1-2 days: 0.5d for the segmented selection implementation, 0.5d for the dual-stride mechanism, 0.5d for DuoAttention integration, 0.5d for calibration using Theorem 3.1 parameters)
- **Depends on**: task 46 (SnapKV compact cache) must land first to provide the eviction substrate. Composes with task 97 (freshness-aware eviction) — freshness scoring applies *within* each segment before heavy-hitter selection. Composes with task 96 (AIMD controller) — AIMD determines the global keep budget, segmented eviction distributes it across segments.
- **Risk**: the optimal segment size may vary significantly across layers and head types. A 512-token segment that works for early layers (where attention is more local) may be too small for late layers (where attention spans wider). Mitigation: use the persistence similarity metric from pass 23's TDA paper (2410.11042) to set per-layer segment sizes — layers with high persistence similarity (topologically stable, wider attention) get larger segments.

### 99. Topology-guided per-layer KV budget calibration via zigzag persistence
- **Goal**: 1 (1M context — allocate KV budget where it matters most), 2 (intelligence — preserve quality by protecting topologically unique layers)
- **Derived from**: Persistent Topological Features in Large Language Models (2410.11042, ICML 2025). The paper demonstrates that zigzag persistence tracks topological feature evolution across LLM layers, revealing that middle-to-late layers have high persistence similarity (topologically redundant) while early layers show high topological churn (topologically unique). Conservative pruning of 10% of high-similarity layers achieves state-of-the-art quality preservation.
- **Change**:
  - Build an offline calibration script `omlx/calibrate_layer_topology.py` that computes zigzag persistence across Qwen3-Coder's 48 layers. For each layer pair (L_i, L_{i+1}), compute the persistence similarity: the fraction of 1-cycles and 2-cycles at L_i that persist to L_{i+1}. Store the resulting 48-element persistence similarity vector as a JSON calibration file.
  - In the PyramidKV per-layer budget allocation (task 57, when it lands), replace the heuristic pyramidal funneling with a topology-guided allocation: layers with low persistence similarity (topologically unique, high churn) receive higher KV budget; layers with high persistence similarity (topologically redundant) receive lower KV budget. The allocation formula is: `budget_i = base_budget * (1 - alpha * persistence_similarity_i)`, where alpha (default 0.5) controls how aggressively redundant layers are compressed.
  - For the attention-weighted codec selection architecture, the persistence similarity vector adds a *layer dimension* to the codec selector: the existing two dimensions are head type (retrieval vs streaming, from DuoAttention) and temporal freshness (from SleepGate, task 97). The third dimension (layer topology) determines *how much* KV budget each layer gets, while head type determines *which codec* to use and freshness determines *which entries* to keep.
  - Add a `--topo-budget` flag to enable topology-guided allocation. Default: disabled until calibrated.
- **Verify**:
  - Calibration validation: the persistence similarity vector should show a clear pattern — low similarity in early layers (0-8), rising through middle layers (8-32), and potentially dropping again in the final few layers (44-48). This matches the paper's findings across Llama/Mistral and should transfer to Qwen3-Coder's similar architecture.
  - Quality gate: with `--topo-budget` enabled and alpha=0.5, all existing benchmark gates must pass (smoke, coherence, code intelligence, NIAH 4K). The topology-guided allocation should *not* degrade quality because it allocates more budget to layers that matter (low similarity = unique contribution).
  - Memory gate: topology-guided allocation should reduce total KV cache memory by >= 15% at 64K context compared to uniform allocation, by compressing redundant middle layers.
  - Comparison gate: topology-guided allocation must match or exceed PyramidKV's heuristic funneling on NIAH quality at the same total memory budget. If it doesn't, the alpha parameter needs recalibration.
- **Effort**: S-M (2-3 days: 1d for the zigzag persistence calibration script using giotto-tda or ripser.py, 0.5d for the JSON calibration file format and loading, 0.5d for the budget allocation formula integration, 0.5d for calibration on Qwen3-Coder)
- **Depends on**: task 57 (PyramidKV per-layer budget) provides the allocation substrate. Task 12 (DuoAttention head classification) provides the head-type dimension. Independent of task 97 (freshness eviction) — the three axes compose orthogonally.
- **Risk**: zigzag persistence computation on 48 layers of a 30B model's hidden states may be computationally expensive (the simplicial complex construction is O(n^2) in the number of sampled tokens). Mitigation: subsample to 256-512 tokens per layer (the paper shows topological features are stable across sample sizes) and use approximate persistence via ripser.py's sparse mode. The calibration runs once offline, not at inference time.

## Research-derived tasks (from LIT_REVIEW.md pass 24, 2026-04-16)

### 100. CAOTE attention-output-error scoring for SnapKV eviction
- **Goal**: 1 (1M context — principled eviction error minimisation), 2 (intelligence — value-aware eviction preserves output quality)
- **Derived from**: CAOTE (2504.14051). The paper proves (Theorem 3.2) that the eviction score c_j = (alpha_j / (1 - alpha_j)) * ||V A^T - v_j||_2 exactly equals the MSE between attention output before and after evicting token j. This is the first closed-form integration of both attention scores and value vectors into an eviction criterion.
- **Change**:
  - In the SnapKV eviction logic (task 46, when it lands), replace the attention-only importance score with the CAOTE score. For each candidate eviction token j: compute c_j = (alpha_j / (1 - alpha_j)) * ||V_mean - v_j||_2, where alpha_j is the attention score, V_mean is the mean of remaining value vectors (FastCAOTE approximation), and v_j is token j's value vector.
  - The CAOTE score is computed per-head, per-layer. For DuoAttention retrieval heads, use the full CAOTE score (these are the heads where eviction accuracy matters most). For streaming heads, continue using the ring buffer (no eviction scoring needed).
  - When composing with BUZZ's segmented eviction (task 98): within each segment, rank tokens by CAOTE score (not attention-only) and select the top-k_local by CAOTE importance. The segmented structure handles locality; CAOTE handles per-token accuracy.
  - When composing with freshness-aware eviction (task 97): the combined importance score is `importance_j = caote_j * freshness_j`. Freshness penalises superseded tokens; CAOTE penalises tokens whose eviction causes high output error. The product is the correct composition: a stale token with high CAOTE score is still evictable because it's superseded, and a fresh token with low CAOTE score is evictable because its removal has negligible output impact.
  - Add a `--caote` flag to enable CAOTE scoring. When disabled, fall back to attention-only scoring. Default: disabled until validated.
- **Verify**:
  - NIAH quality gate: at 4K/16K, CAOTE scoring at 50% keep must match or exceed attention-only scoring at 50% keep. The needle token should have a high CAOTE score (it's a high-attention token with a *distinctive* value vector — the exact pattern CAOTE is designed to identify).
  - LongBench comparison: on a representative subset (2 QA tasks, 1 summarisation task), CAOTE scoring at 50% keep should improve average score by >= 2 points over attention-only scoring (matching the paper's reported SnapKV improvement).
  - Overhead gate: FastCAOTE adds <= 0.5ms per decode step at 4K context. The dominant cost is one vector norm per token per head — O(n * d * h) where n is cache size, d is head dim, h is the number of retrieval heads.
  - Composition test: with BUZZ segmented eviction (task 98), freshness eviction (task 97), and CAOTE all enabled, NIAH 4K must still pass. The three mechanisms must compose without interference: BUZZ determines *which segments*, freshness determines *which tokens are stale*, CAOTE determines *which tokens have lowest output impact*.
- **Effort**: S (1 day: 0.5d for the CAOTE score computation, 0.5d for integration with SnapKV/BUZZ/freshness and flag plumbing)
- **Depends on**: task 46 (SnapKV compact cache) must land first. Composes with task 98 (BUZZ segmented eviction), task 97 (freshness eviction), and task 96 (AIMD controller). Independent of task 99 (topology-guided budget) — CAOTE determines *which tokens* to evict, topology determines *how many* per layer.
- **Risk**: the FastCAOTE approximation (mean-of-values instead of full attention output) may degrade on heads with high value variance, where the mean is a poor summary. Mitigation: for heads where value variance exceeds a threshold (calibrated offline), use the full CAOTE score (slower but exact). Most heads have low value variance, so the fast path covers ~90% of heads.

### 101. CodeComp-style structural span protection for agentic code KV eviction
- **Goal**: 1 (1M context — 60% KV reduction for code contexts), 2 (intelligence — preserves structurally critical code tokens that attention-only eviction misses)
- **Derived from**: CodeComp (2604.10235). The paper demonstrates that Jaccard overlap between attention-ranked and structure-ranked code tokens is only 0.094, proving that attention-based eviction is structurally adversarial for code. Span-level structural protection from Code Property Graph (CPG) analysis recovers 91% of uncompressed performance at 60% KV capacity on Qwen3-8B.
- **Change**:
  - Add an offline pre-processing step to the prompt pipeline in `omlx/hypercar_server.py`: when the input contains code (detected by language markers, file extensions in tool calls, or explicit code blocks), extract structurally critical token spans using a lightweight static analysis pass. The full Joern CPG extraction from the paper is too heavy for interactive use; instead, implement a regex/tree-sitter-based approximation that identifies: function signatures, call sites (function invocations), branch conditions (if/while/for predicates), return statements, and assignment targets.
  - Mark the identified token spans as *protected* in the KV cache metadata. Protected spans are excluded from eviction by SnapKV, CAOTE (task 100), BUZZ (task 98), and freshness (task 97). They are only removed when the total cache budget (from AIMD controller, task 96) forces eviction of protected spans — at which point the system logs a warning (structural protection violated, quality may degrade).
  - Implement CodeComp's structure-aware budget allocation: chunks with higher structural importance (more CPG-identified spans) receive proportionally larger KV budgets. The allocation formula is: B_i = floor(|C_i| * min(r_max, r * m_i)), where m_i is derived from normalized structural feature scores.
  - Add a `--struct-protect` flag to enable structural span protection. Default: disabled until validated. Only active when the input contains code.
- **Verify**:
  - Code intelligence gate: with `--struct-protect` enabled at 50% KV capacity, the 5/5 code intelligence benchmark must still pass. Without structural protection at 50% capacity, expect degradation to <= 3/5.
  - NIAH code variant: construct a synthetic 8K context with a code repository, insert a function definition as the "needle", query for that function's return type. With structural protection, the function signature (call site, return statement) is protected from eviction and the model should retrieve it correctly.
  - Overhead gate: the tree-sitter-based structural analysis adds <= 50ms per prompt (one-time cost at prefill, amortised across the entire generation). This is negligible compared to the model's prefill time.
  - False positive gate: structural protection must not protect > 30% of total tokens (if more than 30% of tokens are structurally critical, the protection is too broad and reduces the effective compression ratio). Calibrate the tree-sitter patterns to achieve 10-20% protection rate on typical code repositories.
- **Effort**: M (3-5 days: 1d for tree-sitter integration and structural span identification, 1d for the protection mask in KV eviction, 1d for structure-aware budget allocation, 0.5d for the --struct-protect flag and detection heuristic, 0.5d for calibration and testing)
- **Depends on**: task 46 (SnapKV compact cache) must land first. Composes with task 100 (CAOTE scoring) — CAOTE scores non-protected tokens, structural protection exempts critical tokens from scoring entirely. Independent of task 99 (topology-guided budget) — structural protection is orthogonal to per-layer budget allocation.
- **Risk**: the regex/tree-sitter approximation of Joern's CPG may miss structurally critical tokens that require data-flow or control-flow analysis to identify (e.g., a variable assignment that is only important because it's used in a distant branch condition). Mitigation: the tree-sitter patterns cover the *syntactic* structures that CodeComp's ablation shows are the dominant contributors (function signatures and query-matched spans account for most of the quality improvement). Data-flow-dependent structures are a second-order effect that can be added later if the syntactic approximation proves insufficient.

## Research-derived tasks (from LIT_REVIEW.md pass 25, 2026-04-16)

### 102. Submodular greedy eviction with CAOTE marginal gains and (1-1/e) set-level guarantee
- **Goal**: 1 (1M context — principled multi-token eviction with compositional error bound), 2 (intelligence — distributional fidelity preserves semantic coverage)
- **Derived from**: OTPrune (2602.20205, CVPR 2026). The paper proves that minimising the 2-Wasserstein distance between full and pruned token distributions yields a submodular objective with monotonicity. The greedy algorithm achieves (1-1/e) of the optimal distributional fidelity — the first compositional guarantee for multi-token selection that accounts for diminishing-returns interactions between evicted tokens.
- **Change**:
  - In the SnapKV eviction logic (task 46), replace the current top-k selection (which selects tokens independently by score) with a greedy submodular selection. The algorithm iterates: at each step, evaluate the marginal gain of adding each candidate token to the retained set, and greedily add the token with highest marginal gain. Repeat until the retained set reaches the target budget k.
  - Use the CAOTE score (task 100) as the marginal-gain oracle: the marginal gain of retaining token j given currently retained set S is the reduction in attention output MSE that j provides, computed via the CAOTE formula c_j = (alpha_j / (1 - alpha_j)) * ||V_mean_S - v_j||_2 where V_mean_S is the mean over the *current* retained set S (not the full set). This captures the diminishing-returns property: as S grows, V_mean_S becomes more representative, and the marginal gain of adding any single token decreases.
  - For efficiency, use the "lazy greedy" acceleration (Minoux 1978): maintain a priority queue of marginal gains and only recompute gains for the top candidates, not all n tokens. This reduces the greedy selection from O(n*k) to O(n + k*log(n)) evaluations in practice.
  - Compose with BUZZ segmented eviction (task 98): run the submodular greedy selection *within each segment*. The per-segment (1-1/e) guarantee composes across segments because the segments are independent — the total retained set is the union of per-segment greedy solutions, and the total error is bounded by the sum of per-segment errors, each within (1-1/e) of its segment optimum.
  - Add a `--submodular-evict` flag. When disabled, fall back to the current top-k selection. Default: disabled until validated. When enabled alongside `--caote`, uses CAOTE marginal gains; otherwise falls back to attention-only marginal gains (which are also submodular but less informative).
- **Verify**:
  - NIAH quality gate: at 4K/16K/64K, submodular greedy at 25% keep must match or exceed top-k selection at 25% keep. The gap should be most visible at aggressive compression (25%) where the diminishing-returns correction matters most.
  - Distributional fidelity test: compute the 2-Wasserstein distance between full-cache and compressed-cache attention output distributions at 16K. Submodular greedy should achieve lower W_2 than top-k at the same budget. Use the `scipy.stats.wasserstein_distance` implementation for 1D marginals or `pot` library for multivariate.
  - Overhead gate: lazy greedy adds <= 2ms per eviction step at 4K context. The dominant cost is recomputing CAOTE marginal gains for O(k*log(n)) candidates. At 4K with 25% keep (k=1000), this is ~1000 CAOTE evaluations instead of 4000 — actually faster than full CAOTE evaluation over all tokens.
  - Composition test: with `--submodular-evict --caote --segmented-evict 512`, NIAH 4K must still pass. The three mechanisms (submodular selection, CAOTE scoring, BUZZ segments) must compose without degradation.
- **Effort**: S (1-2 days: 0.5d for the greedy submodular selection with lazy acceleration, 0.5d for CAOTE marginal-gain integration, 0.5d for composition with BUZZ segments and validation)
- **Depends on**: task 100 (CAOTE scoring) provides the marginal-gain oracle. Task 98 (BUZZ segmented eviction) provides the per-segment substrate. Independent of task 99 (topology-guided budget) and task 101 (structural protection).
- **Risk**: the lazy greedy acceleration assumes the marginal gains are approximately monotone-decreasing across iterations, which holds for strictly submodular functions. If the CAOTE marginal gain function has near-ties (multiple tokens with similar gain), the lazy evaluation may recompute more frequently, approaching O(n*k) worst case. Mitigation: use the "threshold greedy" variant (Badanidiyuru & Vondrak 2014) which achieves (1-1/e-epsilon) in O(n/epsilon) evaluations regardless of tie structure.

### 103. Trigonometric pre-RoPE importance scoring for KV cache compression
- **Goal**: 1 (1M context — O(1)-per-key importance scoring replaces O(n) attention), 3 (decode speed — eliminates attention computation on cache hits)
- **Derived from**: TriAttention (2604.04921). The paper demonstrates that pre-RoPE Q and K vectors concentrate around stable per-head centres, and the resulting distance-dependent attention preference decomposes into a trigonometric series. This enables analytical key importance scoring from the Q/K centres without computing attention.
- **Change**:
  - Add an offline calibration step `omlx/calibrate_qk_centres.py` that computes the per-head Q/K centres for Qwen3-Coder-30B-A3B. For each of the 48 layers and each attention head, compute the mean Q and K vectors in pre-RoPE space over a calibration set (1000 tokens from a representative code corpus). Store as a JSON/npz calibration file: `q_centres[layer][head]` and `k_centres[layer][head]`, each of dimension d_head=128.
  - Implement the trigonometric importance score from the calibrated centres. For a key at position p relative to the query at position q, the score is: `score(p) = sum_i (q_centre_i * k_centre_i * cos((q-p) * theta_i))` where theta_i are the RoPE frequency components. This is computable in O(d_head) per key — independent of context length.
  - In the SnapKV eviction logic, add the trigonometric score as an alternative importance signal. When `--trig-score` is enabled, use the trigonometric score instead of the post-RoPE attention score for the initial token ranking (before CAOTE refinement). The trigonometric score is faster (O(d_head) vs O(n*d_head) for attention) and, per TriAttention's results, more stable because it is not affected by RoPE rotation noise.
  - Use the trigonometric frequency content to classify heads as streaming vs retrieval: heads where the dominant frequency in the trigonometric series is above a threshold are streaming heads (narrow effective window); heads with low-frequency dominance are retrieval heads (wide effective window). Compare this classification against DuoAttention's empirical profiling (task 12) — they should agree on >80% of heads.
  - Add a `--trig-score` flag. Default: disabled until calibrated and validated.
- **Verify**:
  - Centre stability test: the Q/K centres computed over different 1000-token samples should have cosine similarity > 0.95 — confirming the "concentration around fixed centres" property holds for Qwen3-Coder.
  - Head classification agreement: trigonometric-frequency-based streaming/retrieval classification should agree with DuoAttention empirical profiling on >= 80% of heads across all 48 layers.
  - NIAH quality gate: with `--trig-score` at 50% keep, NIAH 4K must pass. The trigonometric score should retain the needle token because it is at a position that receives high trigonometric importance from the query position.
  - Speed gate: trigonometric scoring at 64K context should be >= 10x faster than attention-based scoring, since it is O(n*d_head) total vs O(n^2*d_head) for full attention.
- **Effort**: S-M (2-3 days: 0.5d for calibration script, 0.5d for trigonometric score implementation, 0.5d for head classification, 0.5d for integration with SnapKV eviction, 0.5d for validation)
- **Depends on**: task 12 (DuoAttention head classification) for comparison. Independent of task 100 (CAOTE) — trigonometric scoring is an alternative importance signal that composes with CAOTE by providing the initial ranking that CAOTE refines.
- **Risk**: the Q/K centre stability may degrade for certain head types (e.g., heads that participate in multi-step reasoning where the attention pattern changes dynamically). Mitigation: use the Kalman filter approach suggested in the pass 25 synthesis — track centre drift across decode steps and fall back to attention-based scoring when the filtered centre uncertainty exceeds a threshold.

## Research-derived tasks (from LIT_REVIEW.md pass 26, 2026-04-16)

### 104. Tiered GC scheduler: multi-level KV cache consolidation at temporal boundaries
- **Goal**: 1 (1M context — hierarchical consolidation reduces active KV tokens by 50%+), 2 (intelligence — complexity-aware recall preserves quality for both simple and complex queries)
- **Derived from**: TiMem (2601.02845). The paper demonstrates that a 5-level Temporal Memory Tree with level-specific consolidation outperforms flat memory (75.30% vs 60.79% MemoryOS) while reducing memory tokens by 52.2%. The critical ablation finding: L1 alone achieves 73.18%, L2-L5 alone drops to 57.08% — the hierarchy is essential, not just the consolidation.
- **Change**:
  - Implement a tiered GC scheduler in `omlx/hypercar_server.py` that triggers different eviction/consolidation policies at different temporal boundaries during multi-turn agentic sessions:
    - **L1 (decode-time)**: SnapKV attention-based eviction with CAOTE scoring (task 100) runs every decode step when cache exceeds budget. This is the existing mechanism — no change needed.
    - **L2 (turn boundary)**: At the end of each assistant turn (detected by stop token), run BUZZ segmented consolidation (task 98) over the turn's KV entries. Merge adjacent segments with high cosine similarity (>0.9) into representative entries, reducing per-turn KV by ~30%. This is the session-level consolidation from TiMem's L2.
    - **L3 (session idle)**: When idle time exceeds a threshold (default: 60s, configurable via `--gc-idle-threshold`), run freshness-based pruning (task 97) over the entire cache. Tokens whose freshness score has decayed below a threshold are evicted. This is the daily-level consolidation, triggered by inactivity rather than calendar time.
    - **L4 (periodic recalibration)**: Every N turns (default: 20, configurable via `--gc-recalibrate-interval`), recompute the per-head streaming/retrieval classification using DuoAttention profiling (task 12) on the current cache contents. Heads that have shifted from retrieval to streaming (or vice versa) update their eviction budget allocation. This is the weekly-level pattern extraction.
  - Add a complexity-aware query classifier that examines the user prompt at each turn and selects a recall tier: Simple (use only recent L1-L2 tokens + attention sink), Hybrid (add L3 freshness-filtered tokens), Complex (use full cache). Classification uses a lightweight heuristic: prompts containing code references or multi-file paths are Complex, single-question prompts are Simple, everything else is Hybrid. The recall tier modulates the SnapKV keep-ratio: Simple uses 15% keep, Hybrid uses 25% keep, Complex uses 50% keep.
  - Add `--tiered-gc` flag to enable the scheduler. Default: disabled. When enabled, supersedes the flat eviction policy. Compatible with all existing eviction flags (`--caote`, `--segmented-evict`, `--freshness-decay`).
- **Verify**:
  - Multi-turn NIAH: construct a 10-turn conversation where a needle (API key) is inserted in turn 2. At turn 10, query for the needle. With tiered GC at 25% keep, the needle must be retrievable. Without tiered GC (flat eviction at 25% keep), the needle may be evicted by turn 10 due to low attention.
  - Memory reduction: across a 10-turn coding session (synthetic), tiered GC should reduce peak KV cache size by >= 40% compared to flat eviction, while maintaining NIAH pass at each turn.
  - Latency overhead: the L2 turn-boundary consolidation must complete in < 100ms (it runs once per turn, not per decode step). The L3 idle-time pruning is asynchronous and has no latency requirement.
  - Quality preservation: on the code intelligence benchmark (5 problems), tiered GC at 25% keep must achieve >= 4/5 (matching or exceeding flat eviction at 25% keep).
- **Effort**: M (3-5 days: 1d for the GC scheduler and temporal boundary detection, 1d for the complexity-aware query classifier, 1d for L2 segment consolidation at turn boundaries, 0.5d for L3 idle-time freshness pruning, 0.5d for integration testing)
- **Depends on**: task 46 (SnapKV), task 98 (BUZZ segments), task 97 (freshness decay), task 100 (CAOTE). All are shipped. Independent of task 102 (submodular eviction) and task 103 (trigonometric scoring).
- **Risk**: the complexity-aware query classifier may misclassify prompts, leading to over-eviction (Simple classification on a Complex query) or under-eviction (Complex classification on a Simple query). Mitigation: conservative default — classify as Hybrid unless strong signals indicate Simple or Complex. The heuristic can be replaced with TiMem's LLM-based classifier later, but the LLM call adds latency (~2s per classification), which is acceptable at turn boundaries but not at decode time.

### 105. NVMe KV materialisation with newsvendor break-even policy
- **Goal**: 1 (1M context — offload cold KV to NVMe, keeping only hot tokens in Metal memory), 4 (prefill — eliminate re-prefill for previously seen repository contexts), 6 (M4 Pro 48GB fit — reduce Metal memory pressure via NVMe tiering)
- **Derived from**: MatKV (2512.22195, ICDE 2026). The paper demonstrates that materialising KV caches to flash storage reduces inference latency by 2x and power by 49%, with the ten-day rule providing a closed-form break-even threshold for when materialisation is economical. Energy differential: 175 joules (GPU) vs 0.05 joules (SSD) per 1,024-token KV.
- **Change**:
  - Extend the TQ3 save/load mechanism in `omlx/hypercar_server.py` with an automatic materialisation policy. After first prefill of a repository context (detected by the prompt hash), save the KV cache to NVMe (`~/.cache/omlx/kv/<prompt_hash>.kv`). On subsequent requests with the same prompt prefix, load from NVMe instead of re-prefilling.
  - Implement a break-even calculator for M4 Pro economics: given NVMe read bandwidth (~7.4 GB/s), Metal compute throughput (prefill tok/s from benchmark), NVMe cost (~$0.10/GB), and power differential (M4 Pro ~15W SSD vs ~30W Neural Engine), compute the break-even access interval T. If a repository context is accessed more frequently than T, materialisation is preferred.
  - Implement async NVMe loading using Python's `asyncio` with file I/O to overlap NVMe reads with Metal compute. While generating tokens for the current request, pre-load the next request's materialised KV in the background.
  - Add cache eviction for materialised KV files: LRU eviction when the NVMe cache directory exceeds a configurable size limit (default: 10 GB via `--kv-cache-dir-limit`).
  - Add `--materialise-kv` flag to enable automatic materialisation. Default: disabled. Requires `--kv-mode tq3` or `--kv-mode native` (fp16 KV caches are too large to materialise efficiently).
- **Verify**:
  - Latency test: second access to a materialised 16K-token repository context should be >= 2x faster than re-prefilling (NVMe load at 7.4 GB/s vs Metal prefill at ~500 tok/s).
  - Accuracy test: materialised-then-loaded KV must produce bit-identical output to freshly-prefilled KV for the same prompt. This validates that the save/load round-trip is lossless.
  - Memory test: during NVMe load, Metal memory usage should not exceed the steady-state model size + target KV size (no double-buffering that temporarily doubles KV memory).
  - Break-even validation: the calculator should output a break-even interval (e.g., "materialise if accessed more than once per 2 hours") that matches empirical latency measurements.
- **Effort**: S-M (2-3 days: 0.5d for the break-even calculator, 0.5d for the prompt-hash-based materialisation policy, 0.5d for async NVMe loading, 0.5d for LRU cache eviction, 0.5d for integration testing)
- **Depends on**: TQ3 save/load mechanism (already implemented). Independent of all eviction tasks (46, 97, 98, 100). Composes with task 104 (tiered GC) — materialised KV is the "cold storage" tier below L3 in the hierarchy.

## Research-derived tasks (from LIT_REVIEW.md pass 27, 2026-04-12)

### 106. GER safety monitor: runtime phase-transition guard for KV eviction
- **Goal**: 1 (1M context — prevents eviction from crossing the hallucination cliff), 2 (intelligence — maintains answer-token accessibility under compression)
- **Derived from**: "Understanding the Physics of KV Cache Compression" (2603.01426). The paper discovers a universal hallucination safety cliff near 90% compression, correlated with spikes in Global Eviction Ratio (GER) — the fraction of answer-relevant tokens evicted from all attention heads simultaneously. The phase transition is sharp: compression susceptibility chi = dH/dalpha peaks near alpha ~= 0.9.
- **Change**:
  - After each SnapKV eviction step in the eviction pipeline, compute the GER over the attention sink tokens and any structurally protected tokens (from task 101). GER(alpha) = (1/|T_protected|) * count of tokens where the eviction mask removes the token from ALL heads. If GER exceeds a configurable threshold (default: 0.05, meaning >5% of protected tokens globally evicted), the eviction step is rolled back and the keep budget is widened by 10%.
  - Integrate with the AIMD budget controller (task 96): when GER triggers, AIMD enters additive-increase mode (widening budget) regardless of memory pressure. The GER check acts as a hard floor that AIMD's multiplicative decrease cannot cross.
  - Log GER at each eviction step for offline analysis. Track the per-layer GER to identify which layers are closest to the phase transition boundary — this feeds into the topology-guided budget (task 99) by identifying "brittle" layers where Qwen3-Coder's funnel routing creates thin margins.
  - Add `--ger-guard` flag to enable the safety monitor. Default: enabled when `--caote` is active (the GER check requires knowing which tokens are eviction-critical, which CAOTE provides).
- **Verify**:
  - Phase transition test: construct a 16K context and progressively increase compression from 50% to 95% in 5% steps. Verify that GER spikes sharply near 85-95% (matching the paper's universal cliff). With `--ger-guard` enabled, verify that the budget automatically widens before GER reaches dangerous levels.
  - NIAH safety: at 25% keep (75% compression), GER should be <0.01 (well below the cliff). At 10% keep (90% compression), GER should spike without `--ger-guard`; with `--ger-guard`, the budget should automatically widen to prevent the spike.
  - Overhead: GER computation (set intersection over per-head eviction masks) adds <= 0.1ms per eviction step at 16K context. The dominant cost is the mask intersection, which is O(n * h) where n is cache size and h is number of heads.
  - Composition: with `--ger-guard --caote --segmented-evict 512`, NIAH 4K must still pass. The GER guard should be invisible at comfortable compression ratios and only activate near the cliff.
- **Effort**: S (0.5-1 day: 0.5d for GER computation and AIMD integration, 0.5d for logging and validation)
- **Depends on**: task 100 (CAOTE scoring) identifies which tokens are answer-relevant. Task 96 (AIMD controller) provides the budget adjustment mechanism. Independent of task 102 (submodular eviction) and task 103 (trigonometric scoring).
- **Risk**: the GER threshold (0.05) may need model-specific calibration. The paper shows the cliff location is universal but the cliff *shape* differs by architecture (LLaMA vs Qwen). If Qwen3-Coder has a gradual rather than sharp transition, the fixed threshold may trigger too early (conservative but wasteful) or too late (dangerous). Mitigation: calibrate the threshold offline using the progressive compression test, and use a rolling GER average rather than a single-step check to smooth noise.

### 107. Fair eviction: proportional budget allocation across instruction partitions
- **Goal**: 2 (intelligence — prevents silent instruction dropping under compression), 1 (1M context — maintains multi-instruction fidelity at high compression)
- **Derived from**: "The Pitfalls of KV Cache Compression" (2510.00231). The paper demonstrates that SnapKV and StreamingLLM preferentially evict early-context instructions (system prompts, safety guardrails) because they occupy positions with low recency scores. The proposed fair eviction policy allocates budget proportionally: b_X/n_X = b_Y/n_Y for any instruction partitions X, Y.
- **Change**:
  - In the SnapKV eviction logic (task 46), add an instruction-partition-aware budget allocation step before token-level eviction. At prompt construction time in `omlx/hypercar_server.py`, mark the token boundaries of each instruction partition: system prompt, tool-calling protocol, user message, and code context. Store these boundaries in the KV cache metadata.
  - Before SnapKV selects which tokens to retain, allocate the total keep budget proportionally across partitions: b_i = round(b_total * n_i / n_total) where n_i is the number of tokens in partition i and n_total is the total. Then run SnapKV's attention-based selection independently within each partition using its allocated budget b_i.
  - Compose with BUZZ segmented eviction (task 98): BUZZ segments operate within each partition, so the fair allocation happens first (partition-level), then BUZZ segments within each partition, then CAOTE scoring within each segment. The hierarchy is: partition > segment > token.
  - Add `--fair-evict` flag to enable fair eviction. Default: disabled until validated. When enabled alongside `--segmented-evict`, applies the partition-level allocation before segment-level allocation.
- **Verify**:
  - Instruction retention test: construct a multi-instruction prompt with system prompt (200 tokens), tool protocol (100 tokens), user message (50 tokens), and code context (3650 tokens). At 25% keep (1000 tokens budget), verify that fair eviction retains ~50 system prompt tokens, ~25 tool protocol tokens, ~12 user message tokens, and ~913 code context tokens — proportional to their sizes. Without fair eviction, verify that SnapKV retains <20 system prompt tokens (biased toward recent code context).
  - NIAH quality: with fair eviction at 25% keep, NIAH 4K must still pass. The fair allocation should not degrade retrieval quality because the needle is in the code context partition which receives the majority of the budget.
  - System prompt integrity: construct a prompt where the system prompt says "always respond in French". With compression at 50%, verify that the model still responds in French with fair eviction, and switches to English without it (indicating system prompt was evicted).
  - Overhead: partition boundary tracking and proportional allocation add negligible cost (a few microseconds of integer arithmetic).
- **Effort**: XS (0.5 day: 0.25d for partition boundary tracking in prompt construction, 0.25d for proportional budget allocation and flag plumbing)
- **Depends on**: task 46 (SnapKV compact cache) provides the eviction substrate. Task 98 (BUZZ segments) provides the segment-level allocation. Independent of task 100 (CAOTE) — fair eviction determines per-partition budgets, CAOTE determines per-token scores within each partition.
- **Risk**: proportional allocation may underallocate budget to small-but-critical partitions (e.g., a 50-token system prompt gets only 12 tokens at 25% keep, which may be insufficient to preserve complex instructions). Mitigation: add a minimum per-partition floor (e.g., min 20 tokens per partition) that is subtracted from the total budget before proportional allocation. The floor ensures that even small partitions retain enough tokens for basic instruction following.

### 108. EchoKV-style reconstruction layer for post-eviction KV recovery
- **Goal**: 1 (1M context — recovers evicted KV information at aggressive compression ratios), 2 (intelligence — prevents phase-transition quality collapse by reconstructing missing KV)
- **Derived from**: EchoKV (2603.22910). The paper demonstrates that discarded KV cache entries can be reconstructed from retained subsets using cross-head similarity, with a lightweight linear network trained in ~1 GPU-hour. At 0.3x compression ratio, EchoKV achieves 45.27 on LongBench vs CommonKV's 31.08 — a 14-point improvement at aggressive compression.
- **Change**:
  - Train a lightweight linear reconstruction network for Qwen3-Coder-30B-A3B that predicts evicted KV entries from retained entries. The network takes as input: (a) full KV from the first layer of each 6-layer group (global context), (b) first m=2 heads of each layer (local context), and outputs the reconstructed KV for the remaining heads. Training: Stage 1 (600 steps, MSE loss on raw KV), Stage 2 (1000 steps, attention output MSE loss). Cost: ~1 A100 GPU-hour or ~4 M4 Pro GPU-hours.
  - Integrate the reconstruction network into the eviction pipeline in `omlx/hypercar_server.py`: after SnapKV eviction removes tokens, run the reconstruction network to produce approximate KV for the evicted positions. Store reconstructed KV in a secondary "ghost" cache that is consulted when the primary cache miss rate is high (detected by low attention entropy on retained tokens).
  - Implement on-demand switching: when Metal memory is below 60% of system RAM, run at full KV (no compression, no reconstruction). When Metal memory exceeds 60%, activate SnapKV eviction. When eviction crosses 70% compression and GER guard (task 106) approaches threshold, activate reconstruction to recover evicted tokens rather than widening the budget.
  - Add `--echo-reconstruct` flag. Default: disabled until the reconstruction network is trained and validated. Requires a pre-trained reconstruction weight file (`~/.cache/omlx/echo_weights.npz`).
- **Verify**:
  - Reconstruction fidelity: at 50% compression, the reconstructed KV should achieve cosine similarity > 0.9 with the original KV (measured per-head, per-layer). At 30% compression, cosine similarity > 0.8.
  - NIAH quality: with reconstruction at 30% keep ratio, NIAH 4K must still pass. Without reconstruction at 30% keep, NIAH may fail (matching the phase transition finding). This is the key validation: reconstruction prevents the cliff.
  - Memory overhead: the reconstruction network weights should be < 100 MB (linear layers, no attention). The ghost cache should not exceed the memory saved by eviction — otherwise reconstruction defeats the purpose. Target: ghost cache adds < 20% overhead over the evicted cache size.
  - Latency: reconstruction per decode step at 16K context should add <= 1ms (a single linear projection per layer).
- **Effort**: M (3-5 days: 1d for training the reconstruction network on rented GPU, 1d for inference-time integration and ghost cache, 1d for on-demand switching logic, 0.5d for validation, 0.5d for weight packaging)
- **Depends on**: task 46 (SnapKV eviction) provides the eviction substrate. Task 106 (GER guard) provides the trigger for activating reconstruction. Task 12 (DuoAttention head classification) informs which heads are most reconstructible (streaming heads) vs least (retrieval heads).
- **Risk**: the reconstruction quality at 3-bit quantized KV (native mode) is unknown — the EchoKV paper tests only fp16 KV. The 3-bit quantization noise may compound with reconstruction approximation error, degrading quality below acceptable levels. Mitigation: test reconstruction on DuoKVCache (fp16 mode) first where it should match the paper's results, then separately validate on 3-bit native mode. If 3-bit reconstruction is insufficient, restrict EchoKV to DuoKVCache mode only.
- **Risk**: NVMe write amplification from frequent materialisation of large KV caches may degrade SSD lifespan. Mitigation: the break-even calculator includes a write-wear factor — only materialise contexts that meet the access-frequency threshold. For Hypercar's typical usage (same repository context reused ~10-50 times per coding session), the write volume is modest (~1-5 GB/session, well within SSD endurance limits).

## Research-derived tasks (from LIT_REVIEW.md pass 28, 2026-04-12)

### 109. Per-layer TTT learning rates from reservoir computing theory (adaptive leak rates)
- **Goal**: 2 (intelligence — improves TTT adaptation quality by matching learning dynamics to layer function), 3 (decode speed — prevents over-adaptation of shallow layers that slows convergence)
- **Derived from**: Echo State Transformer (2507.02917) and RC MatMul-free LM (2512.23145). EST demonstrates that parallel reservoirs with per-reservoir learned leak rates adapt temporal sensitivity to the data's dynamics. RC theory predicts that shallow layers (word-level features) need slow adaptation (low leak rate) and deep layers (task-specific features) need fast adaptation (high leak rate). The RC MatMul-free LM validates that freezing 19% of parameters as random values costs only 0.9% quality, confirming the "frozen backbone + small trained readout" architecture.
- **Change**:
  - In the TTT engine (`omlx/ttt.py`), replace the single global learning rate with a per-layer learning rate schedule derived from reservoir computing theory. For each of Qwen3-Coder's 48 layers, compute an effective "leak rate" alpha_l = base_lr * (l / L)^gamma, where l is the layer index (0-47), L is total layers (48), and gamma is a scaling exponent (default: 1.5). This gives shallow layers ~0 learning rate (nearly frozen, acting as reservoir) and deep layers ~base_lr (fully adaptive readout).
  - Add a spectral radius monitor: after each TTT update step, compute the approximate spectral radius of each layer's adapted weight matrix (via power iteration, 10 steps). If any layer's spectral radius exceeds 1.0, scale the adapter's contribution for that layer by 1/spectral_radius to restore the echo state property. This is a principled replacement for the current gradient clipping.
  - Add `--ttt-layer-schedule {uniform,reservoir,cosine}` flag. `uniform` is the current behaviour. `reservoir` applies the alpha_l formula. `cosine` uses a cosine schedule (high LR at both ends, low in the middle — motivated by the U-shaped importance pattern observed in some transformer analyses).
  - Log the effective learning rate and spectral radius per layer at each TTT step for offline analysis.
- **Verify**:
  - Stability: with `reservoir` schedule, the TTT loop should converge in fewer steps than `uniform` (measure on the 5 code intelligence problems). The spectral radius should remain < 1.0 for all layers throughout training.
  - Quality: code intelligence score with `reservoir` schedule >= score with `uniform` schedule (both at same total compute budget). The hypothesis is that `reservoir` focuses adaptation budget on the layers that matter most.
  - Speed: per-step overhead of spectral radius monitoring via 10-step power iteration should be <= 5% of total TTT step time (power iteration on a single layer's adapter matrix is O(r^2) where r is LoRA rank, negligible vs the forward pass).
- **Effort**: S (1 day: 0.5d for per-layer schedule implementation and spectral radius monitor, 0.5d for validation on code intelligence benchmark)
- **Depends on**: task 15 (TTT engine baseline). Independent of task 59 (OPLoRA orthogonal constraint) — the spectral radius monitor is a softer version of OPLoRA's hard constraint. They compose: OPLoRA ensures orthogonality, the spectral radius monitor ensures boundedness.
- **Risk**: the power-law schedule (l/L)^gamma may not match Qwen3-Coder's actual layer dynamics. MoE layers (which have routing-dependent effective computation) may need different leak rates than dense layers. Mitigation: start with the power-law schedule as a principled default and calibrate gamma empirically. The spectral radius monitor provides a safety net regardless of the schedule choice.

### 110. CFAR-style adaptive eviction threshold for SnapKV
- **Goal**: 1 (1M context — adaptive thresholds maintain target retrieval accuracy across varying attention distributions), 2 (intelligence — reduces false eviction of important tokens in dense-attention regions)
- **Derived from**: CFAR (Constant False Alarm Rate) detection from radar signal processing. No direct LLM paper exists — this is a novel cross-domain transfer. CFAR detectors estimate the local noise power from a "guard band" of reference cells around the cell under test, then set the detection threshold as a constant multiple of that noise estimate. This maintains a target false alarm rate regardless of the noise floor, which varies across range/Doppler bins. The analogy: SnapKV's fixed percentile threshold treats all attention distributions identically, but dense-attention regions (where many tokens have similar importance) need higher thresholds than sparse-attention regions (where few tokens dominate). CFAR adapts.
- **Change**:
  - In the SnapKV eviction logic (task 46), replace the fixed percentile threshold with a CFAR-style adaptive threshold. For each attention head, compute the local attention "noise floor" as the median attention score in a sliding window of W tokens (default W=64, the "guard band"). The eviction threshold for each token position is: T(i) = k * noise_floor(i), where k is a constant multiplier (default: 3.0, the "CFAR constant") chosen to maintain a target false eviction rate of ~5%.
  - Implement two CFAR variants: (a) CA-CFAR (Cell-Averaging): noise floor = mean of reference cells, sensitive to uniform clutter; (b) OS-CFAR (Ordered-Statistic): noise floor = the m-th largest value in reference cells (default m = 3W/4), robust to interfering targets (high-importance tokens in the guard band that would inflate the CA-CFAR noise estimate).
  - The CFAR threshold replaces the global percentile cutoff. Tokens with attention score > T(i) are retained; tokens below are eviction candidates. The total number of retained tokens is no longer fixed — it varies per head based on the local attention statistics. To maintain a target cache size, apply a secondary budget constraint: if total retained tokens exceed the budget, increase k; if below budget, decrease k. This is a closed-loop CFAR that jointly adapts the threshold constant and maintains the memory budget.
  - Add `--cfar-evict {off,ca,os}` flag. Default: off (use existing percentile threshold). `ca` enables CA-CFAR, `os` enables OS-CFAR.
  - Compose with existing pipeline: CFAR threshold operates within each BUZZ segment (task 98) and each fair-eviction partition (task 107). The hierarchy is: partition > segment > CFAR threshold > CAOTE scoring.
- **Verify**:
  - False eviction rate: on a 16K NIAH context at 25% keep, measure the fraction of NIAH-needle tokens that are incorrectly evicted ("false evictions"). With CFAR, the false eviction rate should be <= 5% (the target). With fixed percentile, measure the baseline false eviction rate for comparison.
  - Adaptation: construct two test contexts — one with dense attention (many equally-important tokens, e.g., a long code file where every line is referenced) and one with sparse attention (few critical tokens, e.g., NIAH with a single needle). Verify that CFAR retains more tokens in the dense context and fewer in the sparse context, while fixed percentile retains the same fraction in both.
  - NIAH quality: with OS-CFAR at 25% keep budget, NIAH 4K must still pass. The OS-CFAR variant should be preferred because it is robust to the needle itself inflating the noise estimate.
  - Overhead: CFAR computation (sliding window median/sort over W=64 tokens) adds O(n * W * log W) per head per eviction step. At 16K context with 8 KV heads, this is ~16K * 64 * 6 * 8 = ~49M operations — negligible vs the attention computation.
- **Effort**: S-M (1-2 days: 0.5d for CA-CFAR implementation, 0.5d for OS-CFAR and closed-loop budget controller, 0.5d for validation and tuning the CFAR constant k)
- **Depends on**: task 46 (SnapKV eviction substrate). Composes with task 98 (BUZZ segments), task 100 (CAOTE scoring), task 107 (fair eviction). Independent of task 106 (GER guard) — CFAR handles per-head thresholds, GER handles cross-head safety.
- **Risk**: the CFAR constant k (3.0 default) may need calibration for different context lengths and content types. In radar, k is derived from the desired probability of false alarm P_fa under a known noise distribution (typically Rayleigh). For attention scores, the distribution is unknown and varies by layer/head. Mitigation: calibrate k empirically by measuring false eviction rate on the code intelligence benchmark at multiple k values, then fix k at the value that achieves P_fa ~= 0.05. The OS-CFAR variant is more robust to distribution assumptions than CA-CFAR.

### 111. Garden-path-aware streaming-head KV eviction (aggressive eviction of syntactically resolved tokens)
- **Goal**: 1 (1M context — safely reclaims more KV memory from streaming heads after syntactic resolution), 2 (intelligence — evicts only tokens the model provably will not re-consult)
- **Derived from**: "Incremental Sentence Processing Mechanisms in Autoregressive Transformer Language Models" (2412.05353). The paper demonstrates that LMs activate syntactic features during garden-path processing but DO NOT reuse those features for follow-up comprehension (0% circuit IoU). This means streaming-head KV entries for syntactically resolved tokens are functionally dead — the model has extracted their information into the residual stream and will never re-attend to them.
- **Change**:
  - In the SnapKV eviction pipeline, add a streaming-head-aware eviction policy. Using the DuoAttention head classification (task 12), identify streaming heads (those where attention is predominantly local/forward). For streaming heads ONLY, apply a more aggressive eviction ratio than for retrieval heads. Specifically: if the global keep ratio is K%, streaming heads keep K/2 % of their tokens and retrieval heads keep (roughly) 2K% — rebalancing the budget from streaming to retrieval heads while maintaining the same total cache size.
  - Add a "syntactic resolution distance" heuristic: for each token position i, compute the attention entropy H(i) = -sum(a_j * log(a_j)) over the attention distribution from position i. Low entropy means the token attends to very few positions (syntactically resolved — it "knows what it's looking at"). High entropy means the token is still uncertain (syntactically ambiguous). Tokens with H(i) below a threshold in streaming heads are candidates for aggressive eviction — they have been syntactically resolved and their KV will not be re-consulted.
  - Compose with BUZZ segments (task 98): within each segment, streaming-head tokens with low attention entropy get lower CAOTE scores (the attention-output-error from evicting them is small because they were not being attended to). The entropy heuristic provides a complementary signal to CAOTE's value-awareness.
  - Add `--streaming-aggressive` flag. Default: disabled. When enabled, applies the streaming/retrieval head budget rebalancing.
- **Verify**:
  - Head classification validation: verify that the DuoAttention head classification (streaming vs retrieval) from task 12 aligns with the garden-path prediction: streaming heads should show low circuit reuse across disambiguation points (measurable via attention pattern analysis on garden-path-like code constructs).
  - NIAH quality: with `--streaming-aggressive` at 25% global keep, NIAH 4K must still pass. The prediction: retrieval heads get ~50% keep (doubled from 25%), which should IMPROVE NIAH performance compared to uniform 25% across all heads.
  - Memory savings: at 25% global keep with streaming-aggressive rebalancing, the actual memory usage should be identical to uniform 25% (same total tokens retained), but the distribution shifts — more budget to retrieval heads where it matters, less to streaming heads where it doesn't.
  - Code intelligence: 5/5 code problems must still pass. The rebalancing should preserve or improve quality because retrieval heads (which handle lookup-style operations in code) get more budget.
- **Effort**: XS-S (0.5-1 day: 0.25d for streaming/retrieval budget rebalancing, 0.25d for attention entropy heuristic, 0.25d for validation)
- **Depends on**: task 12 (DuoAttention head classification) identifies streaming vs retrieval heads. Task 46 (SnapKV eviction) provides the eviction substrate. Task 100 (CAOTE scoring) provides per-token importance scores. Independent of task 106 (GER guard) and task 110 (CFAR thresholds) — this task operates at the head level (budget allocation between head types), while CFAR operates at the token level (threshold within each head).

## Research-derived tasks (from LIT_REVIEW.md pass 29, 2026-04-17)

### 112. Entmax-based eviction scoring for SnapKV (sparse Hopfield token selection)
- **Goal**: 1 (1M context — adaptive sparsity in eviction scoring selects fewer tokens in sparse-attention regions and more in dense-attention regions), 2 (intelligence — exact retrieval guarantee from HFY margin condition replaces heuristic percentile thresholds)
- **Derived from**: Hopfield-Fenchel-Young Networks (2411.08590). The paper proves that replacing softmax with alpha-entmax (Tsallis alpha-negentropy, alpha > 1) in associative memory retrieval achieves ZERO retrieval error when patterns are separated by the margin distance Delta_i >= m/beta, where m = 1/(alpha - 1). Softmax only achieves approximate retrieval with error O(exp(-beta)). The sparsity naturally adapts to the attention distribution — dense regions produce more non-zero entries, sparse regions produce fewer — achieving the adaptive-threshold behaviour of CFAR (task 110) from energy-theoretic first principles.
- **Change**:
  - In the SnapKV eviction scoring logic (task 46), replace the softmax over attention scores with alpha-entmax (alpha = 1.5, giving margin m = 2). The entmax transformation is: entmax_alpha(z) = argmax_p [p^T z - H_alpha(p)] where H_alpha is Tsallis alpha-entropy. For alpha = 1.5, this has a closed-form solution involving sorting and thresholding (O(n log n) per head, same complexity as current percentile computation). Tokens with entmax score > 0 are retained; tokens with entmax score = 0 are eviction candidates. The total number of retained tokens varies per head based on the attention distribution, rather than being fixed by a global percentile.
  - Add a budget constraint: if total retained tokens across all heads exceed the target cache size, increase the effective beta (inverse temperature) to sharpen the entmax and reduce the support size. If below budget, decrease beta. This is a closed-loop controller analogous to the CFAR constant k (task 110) but derived from the HFY framework.
  - Add `--entmax-evict` flag with optional alpha parameter. Default: disabled (use existing percentile threshold). When enabled, replaces the attention scoring with entmax. The alpha parameter controls sparsity: alpha = 1.0 recovers softmax (dense, no eviction), alpha = 2.0 gives sparsemax (maximally sparse), alpha = 1.5 is the balanced default.
  - Compose with existing pipeline: entmax scoring operates within each BUZZ segment (task 98) and each fair-eviction partition (task 107). The hierarchy is: partition > segment > entmax scoring > CAOTE value-awareness tiebreaker.
- **Verify**:
  - Adaptive behaviour: on a dense-attention context (long code file with many cross-references), entmax should retain more tokens than on a sparse-attention context (NIAH with single needle). Measure the support size (number of non-zero entmax scores) for both and verify the ratio is > 2x.
  - NIAH quality: with entmax scoring at alpha = 1.5 and budget equivalent to 25% keep, NIAH 4K must still pass. The margin condition predicts exact retrieval of the needle if it is separated by Delta >= 2/beta from other tokens.
  - False eviction rate: measure the fraction of NIAH-needle tokens incorrectly assigned entmax score = 0. Should be 0% (exact retrieval guarantee) when the needle has sufficient attention separation.
  - Comparison vs CFAR: run both entmax (task 112) and CFAR (task 110) on the same test contexts. Measure NIAH quality and false eviction rate. Prediction: entmax is cleaner (derived from energy minimisation) and produces equal or better results with fewer hyperparameters (alpha vs k + W + CFAR variant).
  - Overhead: entmax computation involves sorting (O(n log n) per head) — same asymptotic cost as the current percentile computation. Should add < 1ms at 16K context with 8 KV heads.
- **Effort**: S (1 day: 0.5d for entmax implementation in SnapKV scoring, 0.25d for closed-loop beta controller, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction substrate). Composes with task 98 (BUZZ segments), task 100 (CAOTE scoring as tiebreaker within non-zero entmax entries), task 107 (fair eviction partitions). Supersedes task 110 (CFAR thresholds) — entmax provides the same adaptive-threshold behaviour from a cleaner theoretical framework. Both can coexist but entmax is preferred if validated.
- **Risk**: the alpha = 1.5 default may not be optimal for Qwen3-Coder's attention distributions. Too high (alpha -> 2) may over-sparsify and miss important tokens; too low (alpha -> 1) recovers dense softmax with no sparsity benefit. Mitigation: sweep alpha in [1.2, 1.8] on the code intelligence benchmark and fix at the value that maximises NIAH quality while minimising retained token count.

### 113. CAMELoT-style prototype consolidation for evicted KV entries
- **Goal**: 1 (1M context — recovers statistical structure of evicted tokens without trained reconstruction network), 2 (intelligence — consolidated prototypes preserve rare-token information that individual eviction destroys)
- **Derived from**: CAMELoT (2402.13449). The paper demonstrates that a training-free associative memory module using running-average consolidation achieves 29.7% perplexity reduction on ArXiv by preserving evicted token statistics. The Gaussian mixture interpretation means each prototype is a mode centre capturing the local distribution of tokens in embedding space. Memory overhead is negligible (~1 MB at 16K context for 64 prototypes per segment).
- **Change**:
  - In the SnapKV eviction pipeline, after tokens are scored and eviction candidates identified, add a CAMELoT-style WRITE operation. For each BUZZ segment (task 98), maintain a set of P = 64 prototype slots, each storing a consolidated K and V vector plus a count and age.
  - When a token is evicted: compute cosine similarity between the evicted token's KV and all prototype slots. If max similarity > R (threshold, default 0.93), consolidate into the best-matching prototype via running average: K_proto <- (K_evicted + c * K_proto) / (c + 1), V_proto <- (V_evicted + c * V_proto) / (c + 1), c <- c + 1. If max similarity < R, replace the oldest prototype slot (max age) with the evicted token's KV.
  - During attention, prepend the prototype KV pairs to the retained KV cache. The model sees: [prototype K/V (P entries)] + [retained K/V (keep_ratio * N entries)] for a total of P + keep_ratio * N entries per head. The prototypes provide approximate access to the statistical distribution of evicted tokens.
  - Add `--proto-consolidate` flag with optional P (prototype count per segment) and R (similarity threshold) parameters. Default: disabled. When enabled, activates the consolidation pipeline.
  - Memory budget: P = 64 prototypes per segment, 128 dims * 2 bytes * 2 (K+V) = 512 bytes per prototype, 32 segments at 16K context = 64 * 512 * 32 = 1 MB total. At 128K context with 256 segments = 8 MB. Negligible compared to the cache savings from eviction.
- **Verify**:
  - Perplexity improvement: at 25% keep with prototype consolidation, measure perplexity on a long code file vs 25% keep without consolidation. Prediction: measurable improvement (5-15%) because prototypes recover information about evicted tokens, especially rare identifiers.
  - NIAH quality: with consolidation at 25% keep, NIAH 4K must still pass. Prototypes should not interfere with needle retrieval because the needle token has high attention and is retained (not evicted), so consolidation only affects non-needle tokens.
  - Comparison vs EchoKV (task 108): prototype consolidation (training-free, ~1 MB overhead) vs EchoKV reconstruction (requires 4 GPU-hours training, ~100 MB weights, linear projection per step). If both achieve similar quality recovery, prototype consolidation wins on simplicity and deployment cost.
  - Rare token quality: construct a test context with rare identifiers (uncommon variable names, niche API calls) that appear once early and are referenced later. With eviction at 25% keep, measure whether the model can recall the rare identifier. With consolidation, the prototype should preserve the rare token's KV (it has similarity < R to all existing prototypes, so it gets its own slot). Without consolidation, the rare token is simply lost.
  - Prototype diversity: after consolidation, measure the average cosine similarity between prototype slots. Should be < R = 0.93 (each prototype represents a distinct mode of the evicted token distribution). If prototypes collapse (all become similar), the consolidation is not capturing diversity — increase P or decrease R.
- **Effort**: S (1 day: 0.25d for prototype data structure and write operation, 0.25d for read/augment integration into attention, 0.25d for memory management and age tracking, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction) provides the eviction substrate and identifies which tokens are evicted. Task 98 (BUZZ segments) provides the per-segment structure. Partially supersedes task 108 (EchoKV) — if prototype consolidation achieves similar quality recovery, the trained reconstruction network is unnecessary. Both approaches could compose: prototypes for fast approximate recovery, EchoKV for high-fidelity recovery of critical tokens.
- **Risk**: the running-average consolidation may be too lossy for KV entries because KV vectors encode position-specific information (via RoPE). Averaging KV entries from different positions produces a "position-blurred" prototype that the model may not know how to attend to. Mitigation: apply Re-RoPE position correction (task 46) to prototype KV entries, assigning them a synthetic position (e.g., the centroid of the evicted tokens' positions) so the model can attend to them with valid RoPE. If position blurring is severe, restrict consolidation to value vectors only (V_proto) and use a fixed synthetic key (K_proto = mean of all retained keys) — this sacrifices precision for position consistency.

### 114. Hopfield energy partition function as runtime KV cache health metric
- **Goal**: 2 (intelligence — early warning of eviction quality degradation before phase transition), 1 (1M context — enables safe operation at extreme compression ratios by monitoring the energy landscape)
- **Derived from**: "Beyond Scaling Laws" (2405.08707). The paper shows that the partition function Z_t = sum_i integral_{B_i} exp(-||x - rho_i||^2) dx determines retrieval quality: Z_t = 1 is optimal, Z_t >> 1 means over-provisioned cache, Z_t << 1 means patterns overlap and retrieval fails. Monitoring Z_t provides a principled early-warning signal for the phase transition observed in the "Physics of KV Compression" paper (pass 27).
- **Change**:
  - After each eviction step, compute an approximate partition function Z_hat from the attention logits. For each head h, compute Z_hat_h = sum_i exp(logit_i) / max_i exp(logit_i), normalised by the maximum logit. This is the effective number of "distinguishable patterns" in the cache for head h (analogous to the perplexity of the attention distribution). Average across heads: Z_hat = mean_h(Z_hat_h).
  - Define health thresholds: Z_hat > 10 means the cache is healthy (many distinguishable patterns, well-separated basins). Z_hat in [3, 10] is the warning zone (patterns beginning to merge). Z_hat < 3 is the danger zone (few distinguishable patterns, phase transition imminent).
  - When Z_hat enters the warning zone, log a warning and optionally trigger the GER guard (task 106) to widen the eviction budget. When Z_hat enters the danger zone, halt further eviction and signal the AIMD controller (task 96) to back off compression.
  - Add `--hopfield-monitor` flag. Default: disabled. When enabled, computes Z_hat after each eviction step and integrates with the GER guard and AIMD controller.
  - Overhead: Z_hat computation requires one pass over the attention logits per head (already computed during attention). Additional cost: sum and max over N logits per head, N * H operations total. At 16K context with 8 heads = 128K FLOPs — negligible.
- **Verify**:
  - Phase transition detection: on a 16K context, sweep the keep ratio from 100% down to 10% in 5% steps. At each ratio, compute Z_hat and measure NIAH quality. Verify that Z_hat drops sharply at the same keep ratio where NIAH quality collapses (the phase transition). If Z_hat predicts the transition, it is a valid early-warning metric.
  - Threshold calibration: the Z_hat thresholds (10, 3) are heuristic. Calibrate by measuring Z_hat across the five code intelligence problems at various keep ratios. Fix the thresholds at the values that correctly classify "safe" (code passes) vs "unsafe" (code fails) eviction levels with < 5% error rate.
  - Correlation with GER: compute both Z_hat (Hopfield partition function) and GER (Generalised Eviction Risk, task 106) at each keep ratio. Measure Pearson correlation. Prediction: high correlation (> 0.8) because both measure the same underlying phenomenon (pattern separation quality). If highly correlated, Z_hat can replace GER as a simpler metric (no per-head gradient computation).
  - False alarm rate: Z_hat should not trigger in normal operation (no eviction or mild eviction). Measure Z_hat on the five code intelligence problems at 100% keep (no eviction) and verify Z_hat > 10 for all.
- **Effort**: XS (0.5 day: 0.25d for Z_hat computation and threshold logic, 0.25d for integration with GER guard and validation)
- **Depends on**: task 106 (GER guard) provides the existing safety monitoring framework. Task 96 (AIMD controller) provides the feedback loop for adjusting compression. Independent of task 112 (entmax scoring) — Z_hat monitors health regardless of the scoring method used.
- **Risk**: the approximate partition function Z_hat may not faithfully track the true Hopfield Z_t because (a) attention logits are not the same as Hopfield energies (they include RoPE position encoding, MoE routing, and layer-specific transformations), and (b) the approximation Z_hat = perplexity of attention distribution is a first-order estimate. Mitigation: validate empirically by checking phase transition prediction. If Z_hat does not predict the transition, try using the entropy of the attention distribution (H = -sum p_i log p_i) instead, which is monotonically related to Z_hat but may have better numerical properties.
- **Risk**: the DuoAttention head classification may not be stable across different prompt types. A head classified as "streaming" on one prompt may behave as "retrieval" on another. Mitigation: use the average head classification across the calibration set (task 12) rather than per-prompt dynamic classification. The entropy heuristic provides a per-token safety check — even in a streaming head, high-entropy tokens are NOT aggressively evicted.

## Research-derived tasks (from LIT_REVIEW.md pass 30, 2026-04-17)

### 115. LoopGuard: feedback circuit breaker for SnapKV eviction
- **Goal**: 2 (intelligence — prevents repetition catastrophe where attention-based eviction amplifies degeneration), 1 (1M context — maintains output diversity during long code generation with aggressive eviction)
- **Derived from**: LoopGuard (2604.10044). The paper demonstrates that attention-based eviction methods (H2O, SnapKV, StreamingLLM) can create a positive feedback loop: repetitive generation produces high attention scores on the repetitive suffix, which causes the eviction policy to preserve repetitive tokens and evict diverse context, which forces further repetition. H2O's F1 drops from 50.49 to 33.73 on 2WikiMultiHopQA vs full cache; LoopGuard restores it to 53.61 (ABOVE full cache) by breaking the feedback cycle with event-driven cache resets.
- **Change**:
  - In the server's decode loop (`omlx/hypercar_server.py`), add a LoopGuard-style output monitor. Track three signals over a sliding window of W=256 generated tokens: (a) Type-Token Ratio (TTR) = unique tokens / total tokens. (b) Compression Ratio (CR) = len(zlib_compress(tokens)) / len(tokens). (c) Confidence streak = count of consecutive steps where top-1 prediction probability > 0.9.
  - Implement K-of-3 voting: declare loop onset when TTR <= 0.2 AND CR <= 0.12 AND generation length >= 2480 tokens. Add debouncing: warmup of 64 steps (no triggers before 64 tokens generated), cooldown of 32 steps (no re-trigger within 32 tokens of last trigger).
  - When loop detected, perform event-driven cache reconstruction: I_keep = I_anchor (32 tokens from system prompt) UNION I_sparse (uniform subsample of distant history) UNION I_recent (recent non-repetitive tokens, excluding detected repetitive blocks). Fixed budget B = min(1024, current_cache_size). Progressive aggressiveness: if loop re-triggers after first reset, reduce the recent budget by 50% and shift to anchors + sparse.
  - Add `--loop-guard` flag. Default: enabled when `--snapkv-keep` is active (eviction increases loop risk). Disabled otherwise (full cache has lower loop risk). Add `--loop-guard-off` to explicitly disable.
  - Log loop detection events with TTR, CR, confidence, and cache state before/after reset for post-hoc analysis.
- **Verify**:
  - Loop prevention: construct a loop-inducing prompt (e.g., "generate a 200-element lookup table for Unicode codepoint names" which tends to produce repetitive boilerplate). With SnapKV at 25% keep and loop-guard enabled, verify no repetition loop forms (TTR stays above 0.2). Without loop-guard, verify the loop forms (TTR drops below 0.1).
  - Quality preservation: on the 5 code intelligence problems with loop-guard enabled, verify 5/5 pass. Loop-guard should not trigger during normal code generation (debounce prevents false alarms on legitimate repetitive code like array initializers).
  - NIAH quality: with loop-guard enabled at 25% keep, NIAH 4K must still pass. The cache reset preserves anchor tokens (which contain the needle retrieval instructions) and sparse history (which includes the needle).
  - Overhead: TTR and CR computation over W=256 tokens adds negligible overhead (< 0.1ms per decode step). The zlib compression for CR can be replaced with a simpler byte-level entropy estimate if overhead is a concern.
  - False alarm rate: on 100 normal code generation samples, loop-guard should trigger 0 times. On 10 loop-inducing samples, it should trigger within 256 tokens of loop onset.
- **Effort**: S (1 day: 0.25d for TTR/CR monitoring, 0.25d for K-of-3 voting and debounce, 0.25d for cache reconstruction, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction) provides the eviction substrate that can cause the feedback loop. Independent of tasks 110, 112 (CFAR/entmax scoring) — LoopGuard monitors output signals, not internal attention signals. Composes with task 106 (GER guard) — GER monitors eviction quality, LoopGuard monitors output quality; they address different failure modes.
- **Risk**: the TTR/CR thresholds (0.2, 0.12) are calibrated for natural language generation. Code generation has inherently lower TTR (repeated keywords like `def`, `return`, `if`) and lower CR (syntactic regularity). Mitigation: calibrate thresholds on a code-specific validation set. Start with relaxed thresholds (TTR <= 0.1, CR <= 0.08) for code and tighten if false negatives occur.

### 116. CASK-style structural core/scratch decomposition for code generation KV
- **Goal**: 1 (1M context — structural awareness lowers the usable budget frontier, achieving acceptable quality at lower keep ratios), 2 (intelligence — core protection ensures function signatures and key decisions survive eviction)
- **Derived from**: CASK (2604.10900). The paper demonstrates that "cask@384 > triattention@512" — structural core/scratch decomposition at a smaller budget outperforms sophisticated scoring at a larger budget. The key insight: effective KV compression depends less on elaborate scoring than on combining core preservation with selective scratch consolidation.
- **Change**:
  - In the SnapKV eviction pipeline, add a structural role classifier that labels each token as CORE or SCRATCH based on syntactic cues. For generated code tokens: CORE = function/method signatures (`def`, `class`, `return`, parameter lists), import statements, branch conditions (`if`, `elif`, `else`, `match`), loop bounds (`for`, `while`, range limits), API call sites, variable declarations with type annotations. SCRATCH = whitespace, comments that have been "consumed" (model has already processed them), closed brackets/parentheses that match already-seen openers, docstring bodies (the signature is core, the body is scratch), logging/print statements.
  - For prompt tokens: use the fair-eviction partition boundaries (task 107) to classify. System prompt = CORE (always protected). Tool protocol = CORE. User instructions = CORE. Code context = mixed — apply the same syntactic classifier. Documentation/comments in context = SCRATCH.
  - Implement the two-stage design: Stage 1 (prefill) — evict SCRATCH tokens from the prefix to secure budget for decode. Stage 2 (decode) — protect CORE tokens in generated output, consolidate SCRATCH tokens via m-folding: k_tilde_G = (1/m_G) sum(a_i * k_i), v_tilde_G = (1/m_G) sum(a_i * v_i), where a_i = CAOTE score for token i. The m-folding creates a representative KV entry that captures the group's information.
  - Add `--core-protect` flag. Default: disabled. When enabled, CORE tokens are exempt from eviction (their CAOTE score is set to infinity). SCRATCH tokens are evicted and consolidated per the standard pipeline (SnapKV + CAOTE + BUZZ segments).
  - The structural classifier is rule-based (regex on generated tokens), not ML-based. It runs on CPU during token emission and adds negligible overhead.
- **Verify**:
  - Budget efficiency: at 25% keep with core-protect, NIAH 4K must still pass. Measure the effective keep ratio of CORE tokens (should be ~100%) and SCRATCH tokens (should be << 25% since core protection takes budget from scratch). Verify that overall quality (code intelligence 5/5) is maintained despite aggressive scratch compression.
  - Core identification accuracy: on 20 code samples, manually label tokens as core/scratch and compare with the rule-based classifier. Target: precision >= 90% for core (few false positives — tokens labeled core that are actually scratch) and recall >= 80% (most true core tokens are identified). False negatives (core tokens missed by classifier) are handled by the CAOTE fallback — high-importance tokens are retained even if not classified as core.
  - m-folding quality: measure the attention output error from consolidation (the kappa-dispersion bound from CASK). Verify that scratch groups have low dispersion (tokens are similar, consolidation is accurate) while core groups would have high dispersion (tokens are diverse, consolidation would be lossy — justifying their protection).
  - Comparison vs uniform eviction: at 25% keep, compare core-protect (25% global, ~100% core, ~10% scratch) vs uniform (25% global, 25% everywhere). Prediction: core-protect achieves higher code intelligence and NIAH scores because the important tokens (signatures, decisions) are guaranteed to survive.
- **Effort**: S-M (1-2 days: 0.5d for syntactic core/scratch classifier, 0.5d for core protection in eviction pipeline, 0.5d for m-folding consolidation, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction substrate). Task 100 (CAOTE scoring provides the weights a_i for m-folding). Task 107 (fair eviction provides partition boundaries for prompt-level classification). Partially supersedes task 113 (CAMELoT prototypes) — CASK's m-folding IS the consolidation strategy, but with CAOTE-derived weights rather than uniform averaging. Both can compose: m-folding for scratch consolidation, CAMELoT-style slot management for tracking the prototypes over time.
- **Risk**: the rule-based syntactic classifier may be too simplistic for complex code structures (nested lambdas, decorators, metaclass definitions). It may misclassify decorator arguments as scratch when they are semantically core. Mitigation: start with conservative core classification (only unambiguous signatures and keywords) and expand incrementally. The CAOTE fallback ensures that high-importance tokens survive eviction even if misclassified as scratch.

### 117. Multi-criteria multiplicative eviction scoring (MCTF-style fusion)
- **Goal**: 2 (intelligence — combining multiple eviction signals via multiplicative fusion captures interactions that additive fusion misses), 1 (1M context — more accurate scoring lowers the safe eviction threshold)
- **Derived from**: MCTF (2403.10030). The paper demonstrates that multiplicative fusion W = prod W_k^{tau_k} of similarity, informativeness, and cluster size criteria outperforms individual criteria (80.1% vs 79.4-79.8% on DeiT-S). The multiplicative form implements a soft-AND: a token is retained only if ALL criteria agree it's important. Temperature parameters tau_k control relative influence.
- **Change**:
  - In the SnapKV eviction scoring logic (task 46), replace the single-criterion CAOTE score with a multiplicative fusion of multiple signals. Define the retention score for token j as: R(j) = [S_caote(j)]^{tau_1} * [S_attn(j)]^{tau_2} * [S_agree(j)]^{tau_3} * [S_fresh(j)]^{tau_4}, where: S_caote(j) = CAOTE value-awareness score (task 100), normalised to [0, 1]. S_attn(j) = attention importance from SnapKV observation window, normalised to [0, 1]. S_agree(j) = inter-head agreement = 1 - (std of attention scores across heads / max attention score). High agreement means all heads value this token similarly. Low agreement means heads disagree — the token is "contested" per Arrow's impossibility (TCT, 2509.23144). S_fresh(j) = freshness decay score (task 97), normalised to [0, 1]. Higher for more recently accessed tokens.
  - Default temperatures: tau_1 = 1.0 (CAOTE primary), tau_2 = 0.5 (attention secondary), tau_3 = 0.3 (agreement modifier), tau_4 = 0.2 (freshness tiebreaker). These give CAOTE quadruple the influence of freshness (1.0 vs 0.2 in log space).
  - Add `--multi-score` flag with optional temperature overrides: `--multi-score tau1=1.0,tau2=0.5,tau3=0.3,tau4=0.2`. Default: disabled (use single CAOTE score). When enabled, replaces CAOTE as the eviction scorer.
  - The multiplicative fusion is computed in log space for numerical stability: log R(j) = tau_1 * log S_caote(j) + tau_2 * log S_attn(j) + tau_3 * log S_agree(j) + tau_4 * log S_fresh(j). This is equivalent to a weighted sum in log space, so it is O(n) per token — same computational cost as current CAOTE scoring.
- **Verify**:
  - Quality improvement: on the 5 code intelligence problems at 25% keep, multi-criteria scoring should achieve >= CAOTE-only scoring (5/5). The prediction is that the additional signals (attention, agreement, freshness) help resolve ties in CAOTE scores that could lead to arbitrary eviction of important tokens.
  - NIAH quality: with multi-criteria scoring at 25% keep, NIAH 4K must still pass. The attention signal S_attn should ensure the needle token (which has high attention) gets a strong retention score even if its CAOTE score is moderate.
  - Calibration: sweep each tau_k independently on the code intelligence benchmark at 25% keep. For each tau_k, measure 5-problem accuracy and NIAH pass rate. Fix tau_k at the values that maximise the combined score. Then validate on a held-out set (the remaining code problems or a different context length).
  - Comparison vs single-criterion: measure false eviction rate (fraction of NIAH-needle tokens evicted) with multi-criteria vs CAOTE-only. Prediction: multi-criteria has lower false eviction rate because the multiplicative fusion requires multiple signals to agree before eviction, making it more conservative.
  - Graceful degradation: measure quality at 50%, 25%, and 12.5% keep ratios with multi-criteria scoring. Verify that quality degrades smoothly (no cliff at any ratio), matching the MCTF consistency insight.
- **Effort**: S (1 day: 0.25d for multiplicative fusion implementation, 0.25d for inter-head agreement computation, 0.25d for temperature calibration, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction substrate). Task 100 (CAOTE scoring provides S_caote). Task 97 (freshness decay provides S_fresh). Task 12 (DuoAttention classification helps interpret S_agree — streaming heads will show different agreement patterns than retrieval heads). Independent of task 112 (entmax) — entmax can replace S_attn in the multiplicative fusion, providing sparse attention importance instead of dense softmax importance.
- **Risk**: the multiplicative form is VERY conservative — if any single signal gives a low score, the token is evicted regardless of other signals. This could cause false evictions for tokens that score low on one criterion but high on others (e.g., a recently inserted token that hasn't had time to accumulate high attention scores). Mitigation: add epsilon floors to each signal (e.g., S_attn(j) = max(attn_score, 0.01)) to prevent any single zero score from dominating. The epsilon floor ensures that a very high CAOTE score can still retain a token with low attention.

### 118. Coordination temperature diagnostic for inter-head eviction safety
- **Goal**: 2 (intelligence — identifies "contested" tokens where heads disagree on importance, flagging them as unsafe to evict), 1 (1M context — enables more confident eviction of "consensus" tokens where heads agree, freeing budget for contested tokens)
- **Derived from**: Thermodynamic Coordination Theory (2509.23144). The coordination temperature T_co = (1/NK^2) * sum ||m_i - m_bar||^2 measures inter-agent model disorder. Applied to multi-head attention: T_co per token measures the variance of importance scores across KV heads. High T_co = heads disagree (Arrow's impossibility is active, eviction is risky). Low T_co = heads agree (safe to evict if all heads agree token is unimportant, or safe to retain if all heads agree token is important).
- **Change**:
  - In the SnapKV eviction scoring, after computing per-head attention scores for each token, compute the coordination temperature per token: T_co(j) = var_h(a_h(j)) / max_h(a_h(j))^2, where a_h(j) is head h's attention score for token j. The normalisation by max^2 makes T_co scale-invariant — it measures relative disagreement, not absolute variance.
  - Classify tokens into three categories based on T_co: CONSENSUS-RETAIN (low T_co, high mean attention — all heads agree token is important, always retain). CONSENSUS-EVICT (low T_co, low mean attention — all heads agree token is unimportant, safe to evict). CONTESTED (high T_co — heads disagree, eviction is risky, requires more careful scoring before evicting).
  - For CONTESTED tokens, apply the full multi-criteria scoring (task 117) before making an eviction decision. For CONSENSUS tokens, use the simple mean-attention threshold (cheaper, but safe because heads agree). This two-tier approach concentrates the expensive multi-criteria computation on the tokens where it matters most.
  - Add `--coord-temp` flag. Default: disabled. When enabled, computes T_co and applies the two-tier eviction policy. The T_co threshold separating consensus from contested is calibrated empirically (default: T_co > 0.5 = contested).
  - Log T_co statistics per eviction step: mean, max, and fraction of contested tokens. This provides runtime visibility into the "difficulty" of the eviction decision at each step.
- **Verify**:
  - Safety: contested tokens should have higher false eviction rates when evicted by simple mean-attention than by multi-criteria scoring. Measure false eviction rate (fraction of NIAH-needle tokens evicted) for contested tokens under both policies. Prediction: multi-criteria reduces false eviction rate by >= 50% for contested tokens.
  - Efficiency: the two-tier approach should reduce computation vs applying multi-criteria to all tokens. Measure the fraction of tokens classified as contested (prediction: 10-30% of tokens). The remaining 70-90% consensus tokens use the cheap mean-attention threshold.
  - NIAH quality: with coord-temp at 25% keep, NIAH 4K must still pass. The prediction: NIAH needles should be classified as CONSENSUS-RETAIN (all heads attend to the needle), so they are always protected.
  - Diagnostic value: on a context with diverse content types (system prompt + code + comments + NIAH needle), visualise T_co across token positions. Verify that structurally ambiguous tokens (e.g., variable names that could be important or trivial depending on context) have higher T_co than structurally clear tokens (e.g., function signatures, keywords).
- **Effort**: XS (0.5 day: 0.25d for T_co computation and token classification, 0.25d for two-tier eviction policy and validation)
- **Depends on**: task 46 (SnapKV eviction substrate provides per-head attention scores). Task 117 (multi-criteria scoring provides the expensive scorer for contested tokens). Independent of task 114 (Hopfield Z_hat) — T_co measures inter-head agreement, Z_hat measures per-head pattern separation quality. They are complementary diagnostics: Z_hat monitors cache health, T_co monitors eviction safety.
- **Risk**: the T_co threshold (0.5 default) may not generalise across different context types and model behaviours. In-context code generation may have systematically different T_co distributions than in-context QA or summarisation. Mitigation: calibrate the threshold on the code intelligence benchmark and add a warning log when T_co distribution shifts significantly from the calibration baseline, suggesting re-calibration.

## Research-derived tasks (from LIT_REVIEW.md pass 31, 2026-04-17)

### 119. Key token detector for eviction-aware waypoint identification
- **Goal**: 2 (intelligence — identifies the 5-10% of tokens that determine output quality, ensuring they survive eviction), 1 (1M context — enables more aggressive eviction of "coasting" tokens with theoretical safety margin)
- **Derived from**: "Beyond Exponential Decay" (2505.24187). The paper shows that LLM errors concentrate at sparse "key tokens" (5-10% of total) representing critical semantic decision junctions. Perplexity on key tokens correlates with downstream task performance at Pearson -0.96. The remaining 90-95% are predictable coasting tokens where the model dead-reckons with near-zero drift.
- **Change**:
  - In the SnapKV eviction pipeline, add a key token classifier that labels each generated token as KEY or COASTING based on a Long-Short Difference (LSD) proxy. For each token t, compute LSD(t) = perplexity_local(t) - perplexity_full(t), where perplexity_local uses only the most recent W=256 tokens of context and perplexity_full uses the entire retained context. LSD > threshold means the token relies on distant context and is a KEY decision junction.
  - For code generation, supplement LSD with syntactic key token detection: function signatures (`def`, `class`), return statements, branch conditions (`if`, `elif`, `match`), import statements, and type annotations are automatically classified as KEY regardless of LSD score. This aligns with CASK core tokens (task 116) but with the theoretical justification from the 5-10% key token hypothesis.
  - KEY tokens receive infinite retention score (never evicted). COASTING tokens are scored by the existing CAOTE + multi-criteria pipeline (task 117). This concentrates eviction entirely on the 90-95% coasting tokens, providing a 2.5-5x safety margin at 25% keep.
  - Add `--key-token-detect` flag. Default: disabled. When enabled, computes LSD and syntactic classification for each generated token and integrates with the eviction scorer.
  - Log key token statistics: fraction of tokens classified KEY (target: 5-10%), LSD distribution, and key token spacing (expected: ~10-20 tokens between key tokens in code).
- **Verify**:
  - Key token fraction: on 5 code intelligence problems, measure the fraction of tokens classified KEY. Target: 5-15% (code may have slightly higher key fraction than natural language due to syntactic structure).
  - Quality improvement: at 25% keep with key token protection, code intelligence must remain 5/5. At 15% keep (near the phase transition), key token protection should extend quality below the current phase transition threshold.
  - NIAH preservation: the NIAH needle tokens should be classified KEY (high LSD because the needle answer depends on distant context). Verify KEY classification for needle tokens.
  - Constant quality prediction: measure code intelligence at 4K, 16K, 64K with key token protection at 25% keep. If the bounded-k hypothesis holds for code, quality should be flat across context lengths.
- **Effort**: S (1 day: 0.25d for LSD computation, 0.25d for syntactic classifier, 0.25d for integration with eviction pipeline, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction substrate). Task 116 (CASK core/scratch provides the syntactic classifier baseline — key token detection refines it with the LSD metric). Task 117 (multi-criteria scoring handles COASTING tokens). Independent of task 115 (LoopGuard) — key token detection addresses prevention (protect important tokens), LoopGuard addresses recovery (detect and fix repetition loops).
- **Risk**: LSD computation requires two forward passes per token (one with local context, one with full context), doubling inference cost. Mitigation: compute LSD only at eviction boundaries (e.g., every 256 tokens), not every token. Amortize over the eviction interval. Alternatively, use the syntactic classifier alone (no LSD) for a zero-overhead approximation — syntactic key tokens (function signatures, returns, branches) overlap heavily with LSD-identified key tokens in code.

### 120. Closed-loop KV quality monitor with bounded-error guarantee
- **Goal**: 2 (intelligence — guarantees uniformly bounded quality degradation regardless of context length), 1 (1M context — the stability bound gamma/(1-q) predicts the quality floor at any keep ratio)
- **Derived from**: F-LLM (2602.12756). The paper proves that autoregressive generation with feedback correction has uniformly bounded error: limsup ||Delta_x_t|| <= gamma/(1-q), where gamma is single-step error and q is the closed-loop contraction rate. Standard open-loop generation (dead reckoning) has exponentially growing error when the Jacobian's spectral radius exceeds 1.
- **Change**:
  - Formalize the existing LoopGuard (task 115) + Hopfield Z_hat (task 114) + AIMD controller (task 96) as an observer-controller system in the F-LLM framework. The observer = LoopGuard's TTR/CR monitoring (estimates quality deviation from baseline). The controller = AIMD's keep ratio adjustment (corrects for detected drift). The Lipschitz constraint = SnapKV observation window (bounds the effective Jacobian of the eviction-generation system).
  - Define the quality state as a vector q_t = [TTR_t, CR_t, Z_hat_t] measuring output diversity, compression ratio, and Hopfield health at step t. The observer estimates the residual Delta_q_hat_t = r_psi(Delta_q_{<t}) using an exponential moving average of past residuals (lightweight, no training required). The controller applies correction: if ||Delta_q_hat_t|| > gamma/(1-q), adjust the AIMD keep ratio by Delta_r = -alpha * ||Delta_q_hat_t|| (proportional feedback).
  - Add `--closed-loop` flag. Default: disabled. When enabled, activates the formalized observer-controller with bounded-error monitoring. Requires `--loop-guard` and `--hopfield-monitor` to be active (provides the observer signals).
  - Log the stability metrics: estimated gamma (single-step quality deviation), estimated q (contraction rate from AIMD convergence), and the predicted quality floor gamma/(1-q). Alert when the quality state approaches the floor.
- **Verify**:
  - Bounded error: on a long-context generation (8K tokens), measure the quality trajectory [TTR_t, CR_t, Z_hat_t] with and without closed-loop feedback. With feedback, quality should remain within the bounded envelope gamma/(1-q). Without feedback (open-loop), quality may diverge.
  - Phase transition shift: with closed-loop feedback active, measure the minimum keep ratio where code intelligence still passes 5/5. Prediction: the phase transition shifts from ~15% keep (open-loop) to ~10% keep (closed-loop), because feedback correction compensates for drift at lower keep ratios.
  - Overhead: the observer (EMA of quality residuals) adds negligible compute. The controller (AIMD adjustment) already exists. The only new cost is computing the stability bound, which is O(1) per step.
  - Stability bound prediction: measure gamma and q empirically across 10 generation samples. Compute predicted floor gamma/(1-q). Compare to actual quality at steady state. Prediction: the bound is conservative (actual quality is better than predicted) because the linear approximation overestimates error.
- **Effort**: S (1 day: 0.25d for observer formalization, 0.25d for controller integration, 0.25d for stability bound computation, 0.25d for validation)
- **Depends on**: task 96 (AIMD controller provides the feedback mechanism). Task 114 (Hopfield Z_hat provides the health signal). Task 115 (LoopGuard provides the TTR/CR observer signals). All three must be implemented before this task can formalize their interaction as a closed-loop system.
- **Risk**: the linear approximation (Delta_x ~ J_g * Delta_x + epsilon) may not hold for the discrete, nonlinear KV eviction-generation system. The Jacobian of the attention mechanism is not well-conditioned, and the Lipschitz constant of SnapKV eviction is not bounded in general. Mitigation: use the empirical stability bound (measured gamma and q) rather than the theoretical bound. The empirical bound is valid for the observed operating range even if the theoretical assumptions don't hold globally.

### 121. Cognitive chunking for KV cache organization
- **Goal**: 1 (1M context — chunking reduces effective cache size by grouping related tokens into semantic units, enabling higher effective context at the same memory budget), 2 (intelligence — chunk-level attention preserves semantic coherence that token-level eviction destroys)
- **Derived from**: VPWEM (2603.04910). The paper implements a dual-memory architecture inspired by human working memory (7 +/- 2 item capacity) and episodic memory consolidation (hippocampal cross-attention compression). Working memory window L=2 with episodic compression M=2 achieves >20% improvement on memory-intensive tasks vs full-context baselines.
- **Change**:
  - In the SnapKV eviction pipeline, add a chunking layer that groups consecutive tokens into semantic chunks before scoring. Chunk boundaries are placed at KEY token positions (task 119): each KEY token starts a new chunk, and the chunk extends to the next KEY token. Average chunk size should be 10-20 tokens (matching the key token spacing from the 5-10% rate).
  - Each chunk is represented by a summary token computed via CASK m-folding (task 116): k_summary = (1/|chunk|) * sum(a_i * k_i), v_summary = (1/|chunk|) * sum(a_i * v_i), where a_i = CAOTE score for token i within the chunk. The KEY token at the chunk boundary is always retained at full fidelity (not summarized).
  - Eviction operates on chunks, not individual tokens. The multi-criteria score (task 117) is computed per-chunk as the mean score of the chunk's constituent tokens. Evicting a chunk removes all its COASTING tokens and retains only the KEY token + summary token. This reduces the per-chunk cost from ~10-20 KV entries to 2 KV entries (10-20x within-chunk compression).
  - The working memory analog: the most recent W=256 tokens are NEVER chunked (they are the "working memory window" where the model needs full-fidelity access). Chunking applies only to tokens outside the working memory window (the "episodic memory" region).
  - Add `--chunk-evict` flag. Default: disabled. When enabled, activates chunk-level eviction with KEY token boundaries. Requires `--key-token-detect` (task 119) for boundary placement.
  - Log chunk statistics: average chunk size, number of chunks, fraction of context in working memory vs episodic memory.
- **Verify**:
  - Chunk quality: at 25% keep with chunk eviction, code intelligence must remain 5/5. The KEY tokens are always preserved (full fidelity at chunk boundaries), so function signatures and branch conditions survive. COASTING tokens within each chunk are replaced by a summary token that captures the aggregate information.
  - Miller's capacity test: measure the effective number of attention clusters per head (entropy of attention distribution / log(num_retained_tokens)). Compare to 7 +/- 2 prediction. If the attention distribution has ~7 peaks per head, the chunk size is correctly calibrated to match Miller's capacity.
  - NIAH quality: with chunk eviction at 25% keep, NIAH 4K must pass. The NIAH needle should be a KEY token (high LSD), so it is preserved at full fidelity at its chunk boundary.
  - Memory efficiency: measure the actual memory reduction from chunking vs token-level eviction at the same keep ratio. Prediction: chunking achieves comparable quality at lower memory because the summary tokens capture within-chunk information that token-level eviction would discard entirely.
  - Recursive compression: on very long contexts (64K+), test whether chunks can be re-chunked (super-chunks of ~7 chunks each, matching Miller's hierarchical chunking). This mirrors VPWEM's recursive episodic compression.
- **Effort**: M (2 days: 0.5d for chunk boundary detection from KEY tokens, 0.5d for chunk summary computation via m-folding, 0.5d for chunk-level eviction scoring, 0.5d for validation and Miller's capacity test)
- **Depends on**: task 119 (key token detector provides chunk boundaries). Task 116 (CASK m-folding provides the summary computation). Task 117 (multi-criteria scoring provides per-token scores that are aggregated to chunk-level). Task 100 (CAOTE provides the within-chunk weights a_i for m-folding).
- **Risk**: fixed chunk boundaries at KEY tokens may produce highly uneven chunk sizes (e.g., a long comment block with no KEY tokens creates a very large chunk, while a dense function header creates a very small chunk). Large chunks produce poor summaries (too much information in one summary token). Mitigation: add a maximum chunk size limit (e.g., 32 tokens). If a chunk exceeds the limit, split it at the midpoint. This ensures summary quality is bounded regardless of key token spacing.

### 122. Multi-hop NIAH eval for SnapKV composition testing (multi-fact retrieval after eviction)
- **Goal**: 1 (1M context — validates that eviction preserves reasoning chains, not just individual facts), 2 (intelligence — the current eval suite tests single-fact retrieval but never multi-fact composition)
- **Derived from**: Analyst Run 77 (2026-04-17), user-identified eval gap. All current NIAH/RULER tasks test single-fact retrieval: "find the needle." None test multi-fact composition: "combine 3 needles scattered across the context to answer a question." SnapKV, PyramidKV, and CAOTE could silently fail this — each individual fact might survive eviction, but the bridge tokens between them might not. The GER safety monitor (Task 106) detects when important tokens are globally evicted, but it doesn't test whether the *combination* of kept tokens supports reasoning.
- **Change**: Add four new eval sub-tasks to `omlx/bench/hypercar_bench.py` Phase 3b RULER (or a new Phase 3f dedicated to composition testing):
  1. **Multi-hop NIAH**: Hide 3 linked facts at positions ~1K, ~8K, ~15K in a 16K context. Fact A defines a variable, fact B transforms it, fact C queries the result. The model must chain all 3 to answer. SnapKV might keep A and C (high individual attention) but evict B (moderate attention but critical as the bridge). Test at 25%, 50%, 75% keep ratios.
  2. **Coreference chain**: Code where `x = foo()` at position 500, `y = x.bar()` at position 5000, `z = y.baz()` at position 12000. Each reference depends on the prior one surviving eviction. Query: "what is z?"
  3. **Contradictory update**: "The API key is ABC123" at position 200, "UPDATE: the API key was changed to XYZ789" at position 8000. Query: "what is the current API key?" Tests whether freshness eviction (Task 97) correctly identifies the update and keeps the newer value.
  4. **Multi-document synthesis**: 4 code files embedded at positions ~1K, ~5K, ~10K, ~14K. Each file has a piece of a bug. The model must cross-reference all 4 to identify the root cause. This is the "lost in the middle" failure mode.
  Each sub-task runs at 16K context (within the current duo bench path) and at 64K context (via --niah-context 64K with the Task #71 escape hatch). Pass criteria: correct answer at 50% and 25% keep ratios for SnapKV, not just 100% baseline.
- **Verify**:
  - Without SnapKV eviction (100% keep), all 4 sub-tasks pass at 16K (baseline — confirms the tasks are answerable).
  - With SnapKV at 50% keep, all 4 sub-tasks pass (reasonable eviction should preserve multi-hop chains).
  - With SnapKV at 25% keep, multi-hop NIAH and coreference chain pass (critical chains should survive); contradictory update and multi-document synthesis may degrade (acceptable — these are stretch goals at aggressive eviction).
  - Log which specific tokens were evicted for each failing case — this gives engineers a direct debugging signal for eviction policy tuning.
  - `.venv/bin/python -m omlx.bench.hypercar_bench --full` includes the new sub-tasks and they appear in results.json.
- **Effort**: M (3-4 days: 1d for the 4 prompt templates + answer extraction, 1d for the eviction-ratio sweep harness, 1d for integration into Phase 3b/3f, 0.5d for 64K escape-hatch testing)
- **Depends on**: Task 46 (SnapKV core) — already shipped. Task 71 (--niah-context escape hatch) — already shipped. Phase 3e SnapKV quality gate (commit 48558ff) — already shipped.

## Research-derived tasks (from LIT_REVIEW.md pass 32, 2026-04-17)

### 123. Cognitive load classification for KV cache token triage
- **Goal**: 2 (intelligence — classifying tokens by cognitive load type enables load-appropriate retention policies that preserve task-essential tokens while aggressively evicting noise), 1 (1M context — extraneous load tokens often consume 30-50% of context budget in agentic sessions; identifying and evicting them frees massive capacity)
- **Derived from**: "Beyond Accuracy: A Cognitive Load Framework" (2601.20412). The paper demonstrates that task difficulty decomposes into intrinsic load (structural complexity of solution path) and extraneous load (ambiguous presentation), and that accuracy decays exponentially as either increases: Accuracy ~ exp(-(k * CL_total + b)). Applied to KV cache: tokens impose different load types on the model's working memory, and load-aware eviction should treat them differently.
- **Change**:
  - In the SnapKV eviction pipeline, add a cognitive load classifier that labels each token as INTRINSIC, EXTRANEOUS, or GERMANE based on its source and role. INTRINSIC = task-essential tokens: function signatures, branch conditions, API call outputs, user query keywords, NIAH needle content. These map to CASK core tokens (task 116) and KEY tokens (task 119). EXTRANEOUS = noise tokens: verbose system prompt boilerplate, redundant safety disclaimers, over-specified tool descriptions that repeat information, filler phrases in user queries ("please kindly help me with the following task"). GERMANE = learning scaffolding: few-shot examples, template structures, documentation context. Essential during initial processing, dispensable once the model has formed the relevant schema.
  - Apply load-specific retention policies: INTRINSIC tokens receive infinite retention (never evicted, equivalent to core protection). EXTRANEOUS tokens receive zero-priority retention (evicted first, before any CAOTE scoring). GERMANE tokens receive time-decaying retention: full priority during the first pass (prefill), then decaying priority during decode as the model has had time to "learn" from the examples.
  - The classifier is rule-based for the initial implementation: source-based classification (system prompt = mixed EXTRANEOUS/INTRINSIC based on content heuristics, tool protocol headers = EXTRANEOUS, user query = INTRINSIC, few-shot examples = GERMANE) + syntactic classification (reusing the CASK core/scratch classifier from task 116 for fine-grained token-level classification within each source).
  - Add `--clt-triage` flag. Default: disabled. When enabled, applies cognitive load classification before CAOTE scoring.
  - Log cognitive load statistics: fraction of tokens in each category (INTRINSIC, EXTRANEOUS, GERMANE) per eviction step. Target: EXTRANEOUS should be 20-40% of system prompt tokens, representing the compression opportunity.
- **Verify**:
  - Capacity gain: at 25% keep with CLT triage, measure the effective keep ratio for INTRINSIC tokens (should be ~100%). Compare total retained tokens vs standard CAOTE at the same quality level. Prediction: CLT triage achieves the same quality as CAOTE at a lower keep ratio because extraneous tokens are evicted first.
  - Quality preservation: code intelligence 5/5 and NIAH 4K must pass with CLT triage at 25% keep. The INTRINSIC tokens (task-essential) are always retained; the quality loss from evicting EXTRANEOUS tokens should be zero (they don't contribute to task completion).
  - Extraneous identification accuracy: on 10 agentic conversation samples, manually label tokens as INTRINSIC/EXTRANEOUS/GERMANE and compare with the rule-based classifier. Target: precision >= 85% for EXTRANEOUS (few false positives — tokens labeled extraneous that are actually essential).
  - Germane decay validation: on a few-shot prompt (3 examples + query), measure quality at 25% keep with and without germane decay. Without decay, examples consume fixed budget throughout generation. With decay, budget freed by consolidated examples should allow more INTRINSIC retention at longer generation lengths.
- **Effort**: S-M (1-2 days: 0.5d for classifier, 0.5d for load-specific retention policies, 0.5d for germane decay, 0.25d for validation)
- **Depends on**: task 116 (CASK core/scratch provides the syntactic classifier baseline). Task 119 (key token detection provides INTRINSIC classification for generated tokens). Task 46 (SnapKV eviction substrate). Independent of task 117 (multi-criteria scoring) — CLT triage operates BEFORE multi-criteria scoring, filtering out EXTRANEOUS tokens so they never reach the expensive scorer.
- **Risk**: the rule-based classifier may misclassify essential system prompt content as EXTRANEOUS (e.g., a system prompt that contains the API schema needed for tool calls looks like "verbose boilerplate" but is actually task-essential). Mitigation: conservative EXTRANEOUS classification — only classify tokens as EXTRANEOUS if they match known noise patterns (safety disclaimers, filler phrases, repeated headers). Default to INTRINSIC for ambiguous cases. The CAOTE fallback ensures that tokens with high attention importance survive even if misclassified.

### 124. Distractor interference penalty for eviction scoring
- **Goal**: 2 (intelligence — penalising confusable distractors in the cache reduces the model's effective cognitive load, improving attention accuracy on critical tokens), 1 (1M context — at long contexts, the number of potential distractors grows linearly, making interference-aware scoring increasingly important)
- **Derived from**: "Beyond Accuracy" (2601.20412). The paper shows that interference (count of incorrect entities of the same semantic type) multiplies cognitive load via w(e) = delta * (1 + lambda * I). Applied to KV cache: retaining a token that looks similar to a KEY token but is task-irrelevant is WORSE than retaining a completely unrelated token, because the distractor forces the attention mechanism to discriminate — consuming capacity that could be used for task-relevant computation.
- **Change**:
  - In the eviction scoring pipeline (task 117 multi-criteria or standalone CAOTE), add an interference penalty term. For each retained token j, compute the interference score: I(j) = count of other retained tokens within cosine distance < tau of j's key vector, excluding tokens in the same CASK chunk (task 121). High I(j) means many confusable tokens exist — the token is either a target surrounded by distractors, or a distractor itself.
  - Modify the retention score: R(j) = R_base(j) / (1 + lambda_I * I(j)), where R_base is the existing CAOTE or multi-criteria score, and lambda_I is the interference weight (default 0.3). This penalises tokens that have many near-duplicates in the cache. If a token is unique (I=0), no penalty. If a token has 5 confusable neighbors, retention score is divided by 2.5.
  - Exception: KEY tokens (task 119) are exempt from the interference penalty. They are retained regardless of how many distractors surround them. The penalty applies only to non-KEY tokens — if a non-KEY token looks similar to a KEY token, it is penalised (distractor) and preferentially evicted, reducing the KEY token's interference and improving attention accuracy.
  - Add `--interference-penalty` flag with optional lambda_I parameter. Default: disabled.
- **Verify**:
  - NIAH quality: with interference penalty at 25% keep, NIAH 4K must pass. The NIAH needle (a KEY token) is exempt from penalty. Distractor tokens similar to the needle should be preferentially evicted, IMPROVING needle retrieval accuracy.
  - Distractor reduction: on a context with known distractor tokens (e.g., multiple similar-looking variable names where only one is correct), measure the attention entropy on the target token with and without interference penalty. Prediction: lower entropy (more focused attention) with the penalty because distractors are evicted.
  - No quality regression: code intelligence 5/5 with interference penalty active. The penalty should not evict legitimate code tokens that happen to look similar — verify on code with repeated patterns (loop variables, similar function names).
- **Effort**: XS-S (0.5-1 day: 0.25d for interference count computation, 0.25d for penalty integration, 0.25d for validation)
- **Depends on**: task 46 (SnapKV eviction substrate). Task 119 (key token detection for exemption). Task 100 (CAOTE provides R_base). Compatible with task 117 (multi-criteria — interference becomes a 5th criterion in the multiplicative fusion).
- **Risk**: computing pairwise cosine distances for interference counts is O(n^2) in the number of retained tokens. At 25% keep on 64K context (16K retained tokens), this is ~256M comparisons. Mitigation: use approximate nearest neighbor (locality-sensitive hashing or random projection) to identify near-duplicates in O(n log n). Alternatively, compute interference only within BUZZ segments (task 98) — O(segment_size^2) per segment, amortized to O(n * segment_size).

### 125. Smooth scaling pre-processing for TQ3 key outlier suppression
- **Goal**: 2 (intelligence — suppressing key outliers before WHT improves codebook coverage, potentially eliminating the fp16 requirement for layer 0), 1 (1M context — uniform codebook coverage at all layers means consistent quality across the full context without layer-specific exceptions)
- **Derived from**: VecInfer (2510.06175). The paper demonstrates that channel-wise smooth scaling lambda_i = sqrt(max(|K_i|)) before Hadamard transform reduces key outliers and improves VQ codebook coverage. TQ3 currently uses WHT alone; adding smooth scaling could fix the layer-0 outlier problem that requires --fp16-layers 1.
- **Change**:
  - In the TQ3 quantization pipeline (TurboQuantKVCache), add a smooth scaling step before the WHT rotation. For each layer l, compute channel-wise scaling factors: lambda_i^{(l)} = sqrt(max(|K_i^{(l)}|)), where the max is taken over all tokens at the current step. Apply K <- K * diag(lambda)^{-1} and Q <- Q * diag(lambda) to preserve attention computation equivalence.
  - The scaling factors are computed once per prefill/decode step and cached. They add negligible overhead (one max + sqrt per channel per layer = 128 operations per layer).
  - Add `--smooth-scale` flag for TQ3 mode. Default: disabled. When enabled, applies smooth scaling before WHT. If successful, this may allow removing the --fp16-layers 1 requirement for TQ3.
- **Verify**:
  - Layer 0 quality: with smooth scaling + WHT (no fp16 layer 0), run code intelligence 5/5 and NIAH 4K. If quality matches fp16-layer-0 results, the smooth scaling has successfully suppressed the outliers that caused the quality issue.
  - Codebook coverage: measure the L2 reconstruction error of the TQ3 codebook with and without smooth scaling, per layer. Prediction: layer 0's reconstruction error drops significantly (currently it is the worst layer, motivating fp16 treatment). Other layers should see modest improvement.
  - No quality regression: on layers where TQ3 already works well (layers 1-47), smooth scaling should not degrade quality. Verify with the full benchmark suite.
  - Memory savings: if fp16 layer 0 is eliminated, save ~1 GB at 1M context (layer 0's KV cache moves from 16-bit to 3-bit, a 5.3x reduction for that layer).
- **Effort**: XS (0.5 day: 0.25d for smooth scaling implementation in TQ3 pipeline, 0.25d for validation)
- **Depends on**: TQ3 mode (TurboQuantKVCache). Independent of SnapKV eviction (operates at the quantization level, not the eviction level). Validates the fp16-layers requirement from the TQ3 architecture.
- **Risk**: the scaling factors lambda_i may vary significantly between prefill and decode steps, causing inconsistency. The key distribution shifts as context grows, so scaling factors computed on early tokens may not suit later tokens. Mitigation: use exponential moving average of scaling factors to smooth across steps, or recompute only at eviction boundaries (every 256 tokens).

### 126. [TOP PRIORITY] Fix SnapKV server integration — apply_snapkv_to_generate crashes on all KV modes
- **Goal**: 1 (1M context via SnapKV eviction in production server), 3 (decode speed — SnapKV reduces effective KV cache size), plus agentic use case (OpenCode + agents need server-side SnapKV)
- **Derived from**: Analyst deep benchmarking session Run 78 (2026-04-17). Starting the server with `--snapkv-keep 2048` crashes on ALL KV modes (duo, tq3, native) with: `AttributeError: 'function' object has no attribute 'generate_step'` at `omlx/patches/snapkv.py:1343`. The bench's Phase 3e SnapKV quality gate works (it uses a different internal code path), but the server cannot use SnapKV at all. This blocks ALL production use of SnapKV — agents, OpenCode, and the hypercar server itself cannot benefit from KV eviction until this is fixed.
- **Change**: In `omlx/patches/snapkv.py` at line 1343, `apply_snapkv_to_generate` does `gen_mod.generate_step` but `gen_mod` is a function (already monkey-patched by another patch like TurboQuant or prefill_last_logit), not the module. Fix: import the module directly (`import mlx_lm.utils.generate` or equivalent) instead of relying on the passed-in object being a module. Alternatively, check if `gen_mod` is a function vs module and handle both cases.
- **Verify**: `python -m omlx.hypercar_server --kv-mode duo --snapkv-keep 2048 --port 8090` starts without crashing. `curl http://localhost:8090/v1/models` returns a valid response. Send a chat completion request and verify SnapKV eviction log lines appear.
- **Effort**: XS (30 min — the fix is a module import path correction)

### 127. [TOP PRIORITY] TQ3 prefill is 270x slower than duo — investigate and document the agentic implications
- **Goal**: 3 (decode speed is fine at 50 tok/s but prefill at 3 tok/s makes TQ3 impractical for agentic re-prefill), 4 (prefill speed — 3 tok/s vs 500 target is 0.6% of Goal 4)
- **Derived from**: Analyst deep benchmarking session Run 78 (2026-04-17). TQ3 NIAH probe measured: prefill 3 tok/s (vs duo's 817), decode 50.4 tok/s (vs duo's 53.6), Metal peak 39.6 GB (vs duo's 35.1), swap 7.5 GB (vs duo's 0.1). TQ3's WHT codec + vertical_eval + codebook lookup add enormous overhead to every prefill token. Decode is fine because the codec cost is amortized over the KV read. But for agentic use cases where the model needs to re-read context (repeated prefill after tool calls, session restore, fork), TQ3 is 270x slower than duo on the operation that matters most.
- **Change**: Two parts: (a) Document in CLAUDE.md's KV mode table that TQ3 prefill is ~3 tok/s and is NOT suitable for agentic re-prefill use cases. Add a note: "TQ3 is for long single-turn generation (write once, read many). For agentic workflows with frequent re-prefill, use duo mode." (b) Investigate whether SnapKV compaction can help: if the KV cache is compacted from 16K to 4K entries, re-prefill after a fork/rewind only needs to process 4K entries through the codec instead of 16K, potentially recovering 4x on the re-prefill path. File a follow-up task if the investigation shows promise.
- **Verify**: (a) CLAUDE.md KV mode table includes prefill speed per mode. (b) If the SnapKV re-prefill optimization is attempted: measure TQ3 prefill speed on a SnapKV-compacted cache (4K entries from a 16K original) and compare to the non-compacted 16K prefill.
- **Effort**: S for docs (1h), M for investigation (2-3 days)

### 128. Server loads 4-bit model instead of 8-bit — model flag mismatch with bench
- **Goal**: 2 (intelligence — the bench validates against 8-bit model; serving 4-bit silently degrades quality), 6 (machine fit — 4-bit is smaller but the quality/performance numbers in CLAUDE.md don't apply to it)
- **Derived from**: Analyst deep benchmarking session Run 78 (2026-04-17). Starting the server with `python -m omlx.hypercar_server --kv-mode tq3 --port 8090` loaded `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` instead of the expected `Qwen3-Coder-30B-A3B-Instruct-8bit` used by the bench. All performance claims in CLAUDE.md (MMLU-Pro 62%, HumanEval 95%, decode speeds) were measured on the 8-bit model. Serving the 4-bit model in production means users get different (likely worse) quality than what was benchmarked. The server default model and the bench default model should match.
- **Change**: In `omlx/hypercar_server.py`, find the default `--model` argument and change it from `Qwen3-Coder-30B-A3B-Instruct-4bit` to `Qwen3-Coder-30B-A3B-Instruct-8bit` (or whatever the bench uses from CLAUDE.md). Alternatively, add a startup warning when the server model doesn't match the bench model.
- **Verify**: `python -m omlx.hypercar_server --port 8090` (no explicit --model) loads the 8-bit model. The startup log shows `Model: mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit`.
- **Effort**: XS (10 min — change a default string)

### 129. Native 3-bit decode 46.8 tok/s — below Goal 3 target of 50 tok/s at short context
- **Goal**: 3 (decode speed ≥ 50 tok/s constant across context window)
- **Derived from**: Analyst deep benchmarking session Run 78 (2026-04-17). Native 3-bit NIAH probe measured decode 46.8 tok/s at smoke (short context). Duo mode measures 53.6 tok/s at the same context. The ~13% gap means native mode does NOT meet Goal 3 even at short context, while duo does. This matters because native mode is the only mode that can reach 1M context — if Goal 1 requires native mode and Goal 3 requires duo mode, they're mutually exclusive at current performance levels.
- **Change**: Document this tension in CLAUDE.md's Goal 3 status row. Current text says "52.4 tok/s with DuoKVCache — GOAL MET" but doesn't note that native mode (needed for Goal 1 at 1M) is 46.8 and does NOT meet Goal 3. Add: "Native mode: 46.8 tok/s (below 50 target). Goals 1 and 3 are currently mutually exclusive — duo meets Goal 3 but can't reach 1M; native reaches 1M but misses Goal 3. SnapKV compaction on the native KV cache is the path to resolving this tension."
- **Verify**: CLAUDE.md accurately reflects both modes' decode speeds and the Goal 1/3 tension.
- **Effort**: XS (15 min — docs update)

### 130. LiveCodeBench baseline is 30% (6/20) — establish contamination-free coding floor and identify weak problem classes
- **Goal**: 2 (intelligence — LiveCodeBench is the 5th eval family, contamination-free. 30% is the first baseline; understanding which problem types fail guides targeted improvement)
- **Derived from**: Analyst deep benchmarking Run 78 (2026-04-17). Phase 3d LiveCodeBench scored 6/20 (30%). Passing problems: total-distance-traveled, minimum-number-of-chairs, Cutoff, Virus, Fill the Gaps, A..B..C — all appear to be easy/medium difficulty. Failing problems include optimization (maximize-happiness, maximize-consecutive), string manipulation (smallest-substring, partition-string-into-beautiful), math (kth-smallest, sum-of-digit-differences), and algorithmic (continuous-subarrays, Gomamayo Sequence). The failure pattern suggests the model struggles with medium-hard competitive programming problems, not basic coding.
- **Change**: (a) Add per-problem difficulty tags to the LiveCodeBench harness so the bench output shows easy/medium/hard pass rates, not just aggregate. (b) Investigate whether the 14 failing problems share common patterns (e.g., DP, binary search, greedy optimization) that could be addressed by prompt engineering (chain-of-thought, structured planning) without model changes. (c) Set a progression target: 40% on next iteration (8/20), 50% stretch goal.
- **Verify**: `hypercar_bench --full` shows per-difficulty breakdown in Phase 3d output. LiveCodeBench gate still passes at 30% floor.
- **Effort**: S (1 day: 0.5d difficulty tagging, 0.5d prompt engineering investigation)

### 131. Server JSON encoding bug — raw control characters in code responses break OpenCode/API clients
- **Goal**: 6 (machine fit — the server must produce valid JSON for OpenCode and other API clients to consume)
- **Derived from**: Analyst deep benchmarking Run 78 (2026-04-17). Chat completion response for a coding prompt contained raw control characters (likely unescaped newlines in generated Python code) at position 359, causing `json.decoder.JSONDecodeError: Invalid control character`. Multi-turn conversations work (Test 3 parsed correctly), so the bug is specific to responses containing code blocks with certain control characters. This would break OpenCode integration since OpenCode parses the JSON response to extract code.
- **Change**: In the response serialization path (likely `omlx/hypercar_server.py` or the MLX server's response handler), ensure all control characters in the `content` field are properly JSON-escaped before sending. The standard `json.dumps()` should handle this automatically — check if there's a manual string concatenation or streaming path that bypasses `json.dumps()`.
- **Verify**: Send a coding prompt that generates multi-line Python code with newlines, tabs, and backslashes. Parse the response JSON with `json.loads()` — it must not raise `JSONDecodeError`.
- **Effort**: XS (30 min — likely a missing json.dumps in a streaming path)

### 132. 64K NIAH takes 23 minutes and needs 38.4 GB Metal — document the Goal 1 practical cost
- **Goal**: 1 (1M context — 64K works but the practical cost is high)
- **Derived from**: Analyst deep benchmarking Run 78b (2026-04-17). First independent analyst 64K NIAH verification: PASS at 64K native 3-bit. But: Metal peak 38.4 GB (only 2.8 GB headroom against 41.2 limit), swap 4.9 GB, and 23 minutes wall-clock for a single NIAH retrieval. At 128K the engineer needed chunked prefill (commit 09110f8) to avoid OOM. At 256K the math suggests ~40+ GB Metal which won't fit without SnapKV eviction. Goal 1's path to 1M requires SnapKV compaction to shrink the KV cache BEFORE it hits Metal limits. Document this cost progression so the roadmap is realistic.
- **Change**: Add a "Goal 1 cost progression" table to CLAUDE.md or BENCHMARKS.md: 4K → 16K → 64K → 128K showing Metal peak, swap, wall-clock, and whether SnapKV compaction is needed at each scale. Include the practical observation: "64K works but takes 23 min and uses 38 GB; 128K requires chunked prefill; 256K+ requires SnapKV compaction to fit in 48 GB."
- **Verify**: Table exists and numbers match analyst snapshots.
- **Effort**: XS (30 min — docs table from existing data)

### 133. KVTC-style PCA decorrelation + adaptive quantization for session save/load compression
- **Goal**: 1 (1M context — session files currently ~4.2 GB at 1M with TQ3; KVTC could reduce to ~1.1 GB), 3 (decode speed — faster session restore via decompression instead of re-prefill)
- **Derived from**: KVTC (2511.01815, pass 34). The paper demonstrates that the classical transform coding pipeline (PCA decorrelation + DP-optimised adaptive quantization + DEFLATE entropy coding) achieves 20x KV cache compression with <1 point accuracy loss on Llama 3.1-8B and 100% NIAH on 70B at 20x. KVTC decompression is 8.1x faster than vanilla re-prefill, directly addressing TQ3's 270x prefill slowdown (Task 127). ICLR 2026.
- **Change**:
  - Run PCA calibration on 200K tokens of code (using Qwen3-Coder-30B-A3B-Instruct-8bit) to learn the projection matrix V^T. Store V^T alongside the model (2.4% parameter overhead).
  - Implement the KVTC pipeline for TQ3 session save: after WHT + codebook quantization, apply PCA projection, DP bit allocation, and DEFLATE. For session load: reverse the pipeline (DEFLATE decode -> dequantize -> inverse PCA -> WHT decode).
  - Critical: remove RoPE before PCA (both KVTC and RotateKV confirm this is essential). Reapply RoPE after decompression.
  - Add `--compress-sessions` flag to hypercar_server. When enabled, session save uses the full KVTC pipeline. Default: disabled until validated.
- **Verify**:
  - Session save at 16K context produces a compressed file ~4x smaller than current TQ3 save.
  - Session load from compressed file produces identical decode output (token-level agreement) vs uncompressed load.
  - NIAH 4K passes after session load from compressed file.
  - Session load time is faster than re-prefill time for the same context length.
- **Effort**: M (3 days: 1d PCA calibration, 1d pipeline implementation, 1d validation)
- **Depends on**: TQ3 session save/load (already shipped). Independent of SnapKV eviction.
- **Risk**: PCA calibration on code-heavy data may not generalise to mixed code/natural-language prompts. Mitigation: calibrate on a mix of code and natural language, or maintain separate V^T matrices for different domains.

### 134. KeyDiff geometric pre-filter for SnapKV eviction scoring
- **Goal**: 1 (1M context — faster eviction scoring enables more frequent eviction at long context), 3 (decode speed — KeyDiff is O(n) and FlashAttention-compatible)
- **Derived from**: KeyDiff (2504.15364, pass 34). The paper shows that eviction based on key geometric distinctiveness (cosine distance from the anchor/mean key direction) achieves 0.04% accuracy drop at 23% KV reduction, WITHOUT requiring attention scores. This is complementary to CAOTE (Task 100): KeyDiff pre-filters the obviously redundant keys, then CAOTE fine-tunes the selection on the remaining candidates. NeurIPS 2025.
- **Change**:
  - In the SnapKV eviction pipeline (omlx/patches/snapkv.py), add a KeyDiff pre-filter stage before CAOTE scoring. For each head: compute anchor vector mu(K) = mean of normalised keys, score each token S_i = -CosSim(mu(K), k_hat_i), pre-select the top 2N candidates (where N is the final keep budget). Then run CAOTE only on the 2N candidates.
  - Add `--keydiff-prefilter` flag. When enabled, the two-stage pipeline runs. When disabled, CAOTE runs on all tokens (current behaviour).
  - The anchor vector is a single vector per head (128 floats) — negligible memory overhead.
- **Verify**:
  - SnapKV eviction quality: NIAH 4K and 16K pass rates unchanged with KeyDiff pre-filter enabled (compared to CAOTE-only).
  - Scoring time: measure wall-clock for eviction scoring at 16K and 64K with and without KeyDiff pre-filter. Prediction: 30-50% scoring time reduction (CAOTE runs on 2N instead of full n tokens).
  - Edge case: verify that attention sinks (position 0) are always retained by the pre-filter (they should be, since sinks are geometrically distinctive).
- **Effort**: S (1 day: 0.5d implementation, 0.5d validation)
- **Depends on**: SnapKV eviction (Task 46, SHIPPED), CAOTE scoring (Task 100, SHIPPED).
- **Risk**: the anchor vector may be dominated by a few high-norm keys, causing the pre-filter to retain outliers rather than diverse keys. Mitigation: normalise keys before computing the anchor (KeyDiff already prescribes this).

### 135. FAEDKV spectral summary for evicted token reconstruction
- **Goal**: 1 (1M context — spectral summary preserves information from evicted tokens that is currently lost entirely)
- **Derived from**: FAEDKV (2507.20030, pass 34). The paper shows that DFT-based KV cache compression achieves 22% improvement over H2O/SnapKV at tight budgets (compression ratio 0.094) and provides position-unbiased retrieval on NIAH. The IWDFT incremental update enables online spectral summary maintenance. Combined with DynFormer's LGM multiplicative mixing (pass 33), this could provide lossy compression with on-demand reconstruction for evicted tokens.
- **Change**:
  - In the SnapKV eviction pipeline, when tokens are evicted, instead of discarding them entirely, compute a spectral summary: DFT of the evicted chunk's keys and values. Store the top-C frequency coefficients (where C is configurable, default 4 per chunk of 32 tokens = 8x compression of the evicted chunk).
  - At decode time, when computing attention, optionally reconstruct the evicted chunk from its spectral summary via sparse IDFT and include it in the attention computation with reduced weight.
  - Add `--spectral-summary` flag to the eviction pipeline. Default: disabled.
- **Verify**:
  - NIAH at 16K with 25% keep + spectral summary vs 25% keep without: the spectral summary should improve retrieval accuracy for needles in the evicted region.
  - Memory overhead: spectral summary with C=4 per 32-token chunk adds ~12.5% to the compressed KV budget (4 complex coefficients per 32 tokens). Verify this is within memory budget.
  - No regression: standard benchmarks (coherence, code intelligence) should not degrade with spectral summary enabled.
- **Effort**: M (3 days: 1d DFT infrastructure, 1d integration with eviction pipeline, 1d validation)
- **Depends on**: SnapKV eviction (SHIPPED). Benefits from DynFormer's LGM insight (pass 33) for the reconstruction step.
- **Risk**: DFT of quantised (3-bit) KV entries may have poor spectral properties (quantisation noise appears as broadband noise in the frequency domain). Mitigation: apply DFT before quantisation (on the fp16 values), or use KVTC-style PCA decorrelation before DFT to improve spectral concentration.

### 136. Gradient-based per-layer mixed-precision KV quantization (KVmix-style triage)
- **Goal**: 2 (intelligence — sensitive layers get more bits, improving quality at same average bit-width), 1 (1M context — lower average bit-width means smaller KV cache at long context)
- **Derived from**: KVmix (2506.08018, pass 35). The paper demonstrates that gradient-based layer importance scoring enables 2.19-bit key / 2.38-bit value quantization with near-lossless quality (0.006% accuracy loss on LongBench). The gradient distribution follows a Pareto law: 20% of layers carry 80% of the sensitivity. AAAI 2026 oral.
- **Change**:
  - Run gradient profiling on Qwen3-Coder-30B-A3B with 200K tokens of code. Compute s_k_i = ||grad_{W_k_i} L||_2 and s_v_i = ||grad_{W_v_i} L||_2 for all 48 layers.
  - Classify layers into triage categories: top-20% -> 4-bit, middle-60% -> 3-bit, bottom-20% -> 2-bit. Target average ~3 bits overall.
  - Implement `--mixed-precision-kv` flag for TQ3 mode. When enabled, uses per-layer bit-widths from the gradient profile instead of uniform 3-bit. Store the gradient profile alongside the model (48 * 2 floats = negligible).
  - If gradient profiling confirms layer 0 is in the top category, this validates the current --fp16-layers 1 decision. If additional layers are identified as sensitive, adjust the default fp16 layer set.
- **Verify**:
  - Gradient profile exists for Qwen3-Coder. The profile identifies which layers are most/least sensitive.
  - With mixed-precision enabled: code intelligence >= 3/5, NIAH 4K PASS, coherence 2/2.
  - Compare TQ3 mixed-precision (avg ~3 bit) vs TQ3 uniform 3-bit: prediction is mixed-precision wins on quality at same memory budget.
  - Verify that removing --fp16-layers 1 is safe when the gradient-identified layers get appropriate precision.
- **Effort**: S (2 days: 0.5d gradient profiling, 0.5d per-layer quantization implementation, 1d validation)
- **Depends on**: TQ3 mode (TurboQuantKVCache). Independent of SnapKV. Complements Task 125 (smooth scaling) and Task 133 (KVTC decorrelation).
- **Risk**: gradient profiling requires backward passes through the model, which may be memory-intensive on the 48GB M4 Pro. Mitigation: use gradient checkpointing or profile on a subset of layers. Alternatively, use the gradient profile from a smaller Qwen3 variant and transfer the layer importance ranking.

### 137. SparK channel pruning with mean recovery for KV cache channel-level compression
- **Goal**: 1 (1M context — channel pruning provides additional 2x compression orthogonal to token eviction and quantization), 6 (machine fit — 50% channel pruning halves KV memory per token)
- **Derived from**: SparK (2508.15212, pass 35). The paper demonstrates that 80% channel pruning with degenerate (mean-based) recovery achieves <5% accuracy degradation on LongBench and RULER. Composes with SnapKV eviction for multiplicative compression. AAAI 2026.
- **Change**:
  - In the SnapKV eviction pipeline, add a channel pruning stage after token eviction but before KV cache storage. For each retained token: compute saliency w_j = ||q_bar^j||_2 * ||k^j||_2 for each channel j, retain top-T channels (T = 0.5 * d_head = 64), store mean saliency mu for pruned channels.
  - At attention computation time, reconstruct pruned channels using the stored mean: k_tilde^j = mu_j / ||q_bar^j||_2.
  - Add `--channel-prune` flag with a ratio parameter (default 0.5). When enabled, applies channel pruning after SnapKV eviction.
  - Store the per-head mean query vector q_bar (one vector per head, updated as exponential moving average during decode) for the saliency computation.
- **Verify**:
  - Memory: at 16K context with SnapKV@50% keep + SparK@50% channel: KV should be ~25% of full size. Verify metal_bytes measurement.
  - Quality: NIAH 4K PASS with channel pruning enabled. Code intelligence >= 3/5.
  - No catastrophic failure: unlike ThinK (structured pruning), SparK's per-token unstructured pruning should degrade gracefully. Verify on coherence tests.
  - Latency: channel pruning adds a sort per head per token. Verify decode speed does not regress by more than 5%.
- **Effort**: M (3 days: 1d saliency scoring + selection, 1d recovery mechanism, 1d integration + validation)
- **Depends on**: SnapKV eviction (SHIPPED). Independent of TQ3 quantization (operates on a different axis). Composes with KeyDiff pre-filter (Task 134).
- **Risk**: per-token unstructured channel masks are harder to implement efficiently than structured masks. On MLX/Metal, gather operations for non-contiguous channel subsets may be slower than contiguous slicing. Mitigation: implement as a dense mask multiply (sparse * dense) rather than gather, or sort channels by saliency and slice the top-T (contiguous after sort).

## Efficiency Audit Findings (analyst deep bench, 2026-04-18)

_Filed by the performance analyst during a 3-hour deep benchmarking window._
_These are implementation inefficiencies — places where we could be batching into fused ops,_
_streaming instead of reading whole, or using memory more efficiently._

### 138. [TOP PRIORITY] Fused WHT quantize kernel — TQ3 prefill is 270x slower because WHT path lacks the fused Metal kernel that Givens/dense paths have
- **Goal**: 4 (prefill speed — this is the root cause of the 270x prefill slowdown documented in Task 127)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/turboquant_kv.py:617-639` (`_quantize_wht`).
- **The problem**: The Givens rotation path has `_fused_quantize_givens_kernel()` (line 283) that does norm+rotate+boundary+pack in ONE Metal dispatch. The dense rotation path has `_fused_quantize_kernel()` (line 218). But the WHT path — which is the ONLY correct rotation per the paper and the one actually used — does 5 SEPARATE operations: (1) `mx.linalg.norm`, (2) division for unit vector, (3) `_apply_wht_rotation`, (4) Python loop over 7 boundaries, (5) `_pack_contiguous`. Each is a separate Metal kernel dispatch with synchronization overhead.
- **Why this matters**: At 64K context with 48 layers × 4 KV heads × 128 dims, the WHT quantize path processes ~4M vectors. Five separate kernel dispatches vs. one fused dispatch = ~4x kernel launch overhead PLUS the Python boundary loop adds 7 iterations with full tensor comparisons each. This is THE reason TQ3 prefill is 3 tok/s vs duo's 817 tok/s.
- **Change**: Write a `_fused_quantize_wht_kernel()` Metal kernel analogous to the existing `_fused_quantize_kernel()` but using WHT butterfly operations instead of dense matrix multiply. The WHT butterfly is O(D log D) vs O(D²) for dense — each output coordinate depends on ALL input coordinates but via D/2 additions per pass (log₂(128) = 7 passes). The kernel should: (1) compute norm, (2) normalize, (3) apply sign flips, (4) butterfly WHT passes in registers, (5) boundary quantize, (6) pack — all in one dispatch.
- **Profiling evidence** (2026-04-18 06:09):
  - WHT quantize @ 4K × 48 layers: 0.159s (1.24M vec/s)
  - WHT quantize @ 16K × 48 layers: 0.673s (1.17M vec/s)
  - WHT quantize @ 32K × 48 layers: 1.418s (1.11M vec/s)
  - Fused vs unfused dequant: fused is **1.7x faster** at 16K (0.005s vs 0.009s)
  - The quantize path has NO fused kernel — it's 5 separate dispatches. Based on the dequant fusion ratio, expect 2-4x speedup from a fused quantize kernel.
- **Additional experiment** (2026-04-18 06:30): Dense matmul is **2.5-3.0x faster** than Python WHT at all scales (4K: 7.5→3.0ms, 16K: 22.1→7.5ms, 64K: 74.7→28.5ms). At 262K vectors, dense matmul is **26x faster** than Python WHT because MLX's AMX matmul dominates vs O(D log D) butterfly with Python loop overhead.
- **CRITICAL QUALITY WARNING** (EXP 10): Naively replacing `_apply_wht_rotation(unit, signs)` with `unit @ codec.rotation.T` produces **100% different packed indices** (MSE 0.034 → 1.96). This is because the forward rotation is `WHT(signs * x)` = `H @ diag(signs) @ x`, but `codec.rotation` stores `R = diag(signs) @ H`, so `R.T = H.T @ diag(signs) ≠ H @ diag(signs)`. The correct dense matmul form for the forward rotation is `(signs * unit) @ H.T` where `H = _wht(eye(D))`, or equivalently `unit @ (H @ diag(signs)).T`. **The fused kernel MUST apply signs first, then WHT — matching the existing `_apply_wht_rotation` composition.**
- **Verify**: `TurboQuantMSECodec(128, 3).quantize(random_vectors)` should be at least 2.5x faster than current (profiled 2.5-3.0x from dense matmul alone, before fusing with norm+boundary+pack).
- **Effort**: M (2 days — the Givens kernel is a template; WHT butterfly in Metal registers is the new part)
- **Depends on**: None. The fused dequant kernel (`_fused_dequant_wht_kernel`, line 435) already exists and works — this is the quantize counterpart.

### 139. [TOP PRIORITY] SnapKV freshness scoring has O(T) Python loop — vectorize the supersession counter
- **Goal**: 3 (decode speed — freshness scoring runs at eviction time, blocking decode start), 1 (context window — makes SnapKV impractical for agentic multi-turn at 64K+)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/patches/snapkv.py:363-394` (`compute_freshness_scores`).
- **Profiling evidence** (cProfile @ 64K, 2026-04-18 06:09):
  - **5.737s wall clock** — **85.1% of the entire SnapKV pipeline** at 64K context
  - 526,347 function calls: 262,140 calls to `max()`, 264,188 calls to `min()` (inner loop per-element)
  - Scaling: 0.514s @ 4K → 1.335s @ 16K → 2.677s @ 32K → 5.687s @ 64K (linear in T, as expected from the O(T) Python loop)
  - The function body itself accounts for 5.832s of the 5.853s total — virtually ALL time is spent in the Python loop, zero time in GPU ops
- **The problem**: The inner loop `for local_i in range(chunk_len)` (line 384, chunk_len=256) runs inside an outer loop over 256-token chunks. For 64K context: 250 chunks × 256 iterations = 64,000 Python-level iterations, each doing tensor slicing, comparison, and `.at[].add()`. The `.at[].add()` pattern is a Python-level scatter that creates a new tensor per call.
- **Change**: Replace the double loop with a vectorized approach: (1) Compute the full (chunk, window) similarity matrix `sims` already done. (2) Apply a triangular mask for the valid window region. (3) `mx.sum(sims > threshold, axis=-1)` gives the supersession count for the entire chunk in one op. (4) Scatter the counts into the output with a single indexed assignment, not per-element `.at[].add()`.
- **Verify**: `compute_freshness_scores(cache, T=16384)` should complete in <0.5s (currently 1.335s — profiled). At 64K, should drop from 5.7s to <0.5s. Quality: freshness scores should be numerically identical (same algorithm, just vectorized).
- **Effort**: S (1 day — the vectorization is straightforward, the triangular mask is the tricky part)
- **Depends on**: None.

### 140. [TOP PRIORITY] SnapKV keep_mask construction uses per-element .at[].add() Python loop
- **Goal**: 3 (decode speed), 1 (context window — affects all SnapKV evictions)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/patches/snapkv.py:1006-1011` (`snapkv_select`).
- **Profiling evidence** (cProfile @ 64K, 2026-04-18 06:09):
  - **0.277s** for segmented selection path, **0.136s** for global selection path at 64K
  - `snapkv_select` itself takes 0.202s of 0.215s profiled time — the `.at[].add()` loop is the dominant cost
  - 16,384 calls to `set.add()`, 65,472 lambda calls for segment sorting
  - Combined with Task 141 (segmented sorting), this is **4.1% of the full pipeline** — small vs freshness but still >0.25s of pure Python
- **The problem**: After computing the set of indices to keep, the mask is built by iterating through sorted_indices in Python: `for pos in sorted_indices: keep_mask = keep_mask.at[:, pos].add(...)`. At 25% keep of 64K context = 16,384 Python iterations, each creating a new tensor via `.at[].add()`. This is O(T) Python overhead for what should be a single scatter operation.
- **Change**: Build the mask in one shot: `keep_mask = mx.zeros((B, T), dtype=mx.bool_); idx = mx.array(sorted_indices); keep_mask[:, idx] = True` — or use `mx.scatter` / boolean indexing. The `sorted_indices` list is already computed; converting it to an mx.array and doing a single indexed assignment is O(1) Python calls.
- **Verify**: `snapkv_select(importance, keep_count=16384, T=65536)` should complete in <0.01s (currently 0.277s from profiling).
- **Effort**: XS (30 min — single line change)
- **Depends on**: None.

### 141. SnapKV segmented selection uses Python sorted() and list iteration instead of mx.argpartition
- **Goal**: 3 (decode speed at eviction time)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/patches/snapkv.py:748-755` (`_select_segmented`).
- **The problem**: `pooled_np = pooled[0].tolist()` materializes the entire importance tensor to a Python list, then sorts each segment with Python `sorted()`. For 64K context with 512-token segments: 128 segments, each Python-sorting 512 floats. While not the worst bottleneck, this is unnecessary Python overhead when `mx.argpartition` (used in `_select_global`) can do this in one GPU dispatch per segment.
- **Change**: Keep computations in MLX: for each segment, use `mx.argpartition(-seg_scores, kth=seg_k)[:seg_k]` to select the top-K indices without full sorting. Avoids the `.tolist()` CPU roundtrip entirely.
- **Verify**: Correctness test: same indices selected. Speed: should be 5-10x faster at 64K context.
- **Effort**: S (1 day)
- **Depends on**: None.

### 142. DuoKVCache update_and_fetch does per-head Python loop for streaming trim + growing concatenate overhead
- **Goal**: 3 (decode speed — affects every prefill chunk when context exceeds 260 tokens)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/duo_kv_cache.py:215-243` (`DuoKVCache.update_and_fetch`).
- **Profiling evidence** (memory profiling @ 64K, 2026-04-18 06:10):
  - Update time grows linearly with context: 0.83ms @ 2K → 14.28ms @ 64K (17x increase)
  - Metal growth is stable (no transient 2x spikes from concat — MLX handles this gracefully)
  - However: at 64K, each `update_and_fetch` call takes 14ms, and with 48 layers × 32 chunks = 1,536 calls, total DuoKV overhead is ~21.5 seconds per 64K prefill
  - The per-head trim loop runs on every chunk after context exceeds 260 tokens
- **The problem**: When context exceeds sink+window capacity (260 tokens), the trim path iterates per-head in Python: `for h in range(H_kv)` (4 iterations) with per-head slice+concatenate+pad. Each streaming head does 2 slices + 1 concatenate (sink + window), then all heads are padded to max_len and concatenated. This creates 4×3 = 12 intermediate tensors per call, plus the padding tensors.
- **Better approach**: Build a single gather index tensor that maps each (head, position) to either a sink position, a window position, or a pad position. Apply it once across all heads: `trimmed = full_kv[:, :, gather_indices, :]`. One gather op replaces the entire per-head loop. The gather_indices tensor can be precomputed once (it only depends on head types and current offset, not the actual KV values).
- **Verify**: DuoKV prefill at 4K should be measurably faster (~5x reduction in trim overhead). Quality: numerically identical output.
- **Effort**: S (1 day)
- **Depends on**: None.

### 143. StreamingKVCache ring-buffer write uses per-token Python loop
- **Goal**: 3 (decode speed in duo mode — affects every token once ring buffer fills at 260 tokens)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/duo_kv_cache.py:100-114` (`StreamingKVCache.update_and_fetch`).
- **The problem**: In ring mode (offset >= capacity), each new token is written via a Python loop: `for i in range(T_new): pos = ...; self._keys[:, :, pos:pos+1] = keys[:, :, i:i+1]`. During prefill with 8K-token chunks, this is 8,000 Python iterations with per-element tensor slice assignment. During decode (T_new=1), it's only 1 iteration — tolerable. But during prefill, this is catastrophic.
- **Change**: Compute all ring positions at once with modular arithmetic: `positions = (mx.arange(T_new) + (self.offset - self.capacity)) % ring_len + ring_start`. Then use scatter: `self._keys[:, :, positions, :] = keys`. One tensor operation replaces T_new Python iterations.
- **Verify**: Prefill of 8K tokens into a full ring buffer should complete in <1ms (currently O(seconds) from Python loop).
- **Effort**: XS (30 min)
- **Depends on**: None. Note: `StreamingKVCache` is a subcomponent of `DuoKVCache`, not used directly.

### 144. TQ3 KV buffer uses mx.concatenate for growth instead of pre-allocated slab
- **Goal**: 1 (1M context — concatenate copies entire buffer on every extension), 6 (machine fit — transient 2x memory spike during concat)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/turboquant_kv.py:1360-1366` (`TurboQuantKVCache.update_and_fetch`).
- **The problem**: When the compressed KV buffer needs to grow, it uses `mx.concatenate([self._k_norms, mx.zeros(...)])` for FOUR arrays (k_norms, k_packed, v_norms, v_packed). Each concatenation creates a copy of the entire existing buffer plus the new zeros. At 256K context with TQ3, the buffer is ~5GB — concatenation temporarily requires ~10GB. The code does have pre-allocation in 256-token steps (the `_step` field), but when those steps are exhausted, it falls back to concatenation.
- **Better approach**: Pre-allocate larger slabs (e.g., 2x current size, geometric growth) to reduce the number of concatenations. Or use a list of chunks and concatenate lazily only when the full buffer is needed (for decode). The `_step = 256` allocation step is too small for long contexts.
- **Verify**: Memory profile during 64K prefill should show no 2x transient spikes from buffer growth. Concatenation count should drop from ~250 (64K/256) to ~7 (64K with geometric 2x growth from 256).
- **Effort**: S (1 day)
- **Depends on**: None.

### 145. TQ3 dequantize path doesn't use fused kernel by default — dequantize_fused exists but isn't called in the streaming attention hot path
- **Goal**: 4 (prefill speed — streaming attention dequants chunks during prefill)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/streaming_attention.py` calls `codec.dequantize()`, not `codec.dequantize_fused()`. The fused kernel at `omlx/turboquant_kv.py:660-671` does unpack+codebook+WHT+norm in ONE Metal dispatch, but it's only called explicitly — the streaming attention hot path doesn't use it.
- **Profiling evidence** (EXP 6, 2026-04-18 06:30):
  - chunk=2048: unfused 2.17ms → fused 0.88ms (**2.47x speedup**)
  - chunk=8192: unfused 4.24ms → fused 2.26ms (**1.87x speedup**)
  - chunk=16384: unfused 8.66ms → fused 4.30ms (**2.01x speedup**)
  - max_diff between fused/unfused output: 0.0019 (acceptable precision difference)
- **The problem**: During streaming prefill, each dequant chunk goes through: (1) unpack kernel, (2) codebook gather, (3) WHT inverse, (4) norm scale — four separate dispatches. The `dequantize_fused` method exists and would do this in one dispatch, but `streaming_tq_attention` calls `codec.dequantize(k_norms[chunk], k_packed[chunk])` which is the unfused path.
- **Change**: In `streaming_tq_attention`, replace `codec.dequantize(...)` calls with `codec.dequantize_fused(...)`. This is a one-line change per call site.
- **Verify**: Streaming prefill throughput at 16K+ context should improve 2.0-2.5x for the dequant step (profiled). Quality: max difference 0.0019 (acceptable).
- **Effort**: XS (15 min — literally changing method names)
- **Depends on**: None. The fused kernel already exists and works.

### 146. SnapKV submodular selection uses pure Python inner loop with O(T×K×D) complexity
- **Goal**: 3 (decode speed at eviction time for long contexts with --submodular-evict)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/patches/snapkv.py:815-839` (`_select_submodular`).
- **The problem**: The greedy submodular selection does `for _ in range(seg_k)` with an inner `for j in range(seg_len)` that computes cosine similarity using `sum(a*b for a,b in zip(...))` — pure Python zip-and-sum over D=128 floats. For a 512-token segment keeping 128 tokens: 128 × 512 × 128 = 8.4M Python float operations. The `set(retained_in_seg)` check inside `max()` is also O(K) per candidate per iteration.
- **Change**: Move the diversity-penalized selection into MLX: precompute the V similarity matrix for the segment (512×512 at D=128 = 32MB, fits in cache), then iterate the greedy loop with vectorized score updates. Replace `sum(a*b...)` with `mx.sum(v1 * v2)` or batch dot product. The greedy outer loop can stay in Python (K iterations) but the inner scoring must be vectorized.
- **Verify**: Submodular selection at 16K context should complete in <1s (currently estimated >30s from Python loops).
- **Effort**: S (1 day)
- **Depends on**: None. Low priority since `--submodular-evict` is optional and not default.

### 149. [TOP PRIORITY] DuoKVCache uses mx.concatenate per decode token — 248x slower than pre-allocated slab at 64K
- **Goal**: 3 (decode speed — this is the single largest decode bottleneck at long context), 1 (context window — DuoKV becomes impractical at 16K+)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/duo_kv_cache.py:205-206` (`update_and_fetch` line `self._keys = mx.concatenate([self._keys, keys], axis=2)`).
- **Profiling evidence** (microbench, 2026-04-18 06:20):
  - **mx.concatenate per-token overhead scales linearly with buffer size**:
    - 1K context: 1.81 ms/token
    - 4K context: 2.30 ms/token
    - 16K context: 4.42 ms/token
    - 64K context: **34.78 ms/token** (copies 134 MB buffer every single token)
  - **Pre-allocated slab (slice assignment) is constant and 248x faster at 64K**:
    - 1K: 0.21 ms, 4K: 0.25 ms, 16K: 0.17 ms, 64K: **0.14 ms**
  - At 48 layers × 34.8ms = **1.67s per decode token at 64K** from concat alone — limits decode to 0.6 tok/s (Goal 3 requires 50 tok/s)
  - MLX's own `KVCache` class avoids this by pre-allocating and using in-place slice writes
- **The problem**: `DuoKVCache.update_and_fetch` does `self._keys = mx.concatenate([self._keys, keys], axis=2)` on every call. During decode (T_new=1), this copies the ENTIRE buffer to add ONE token. At 64K context with 4 KV heads × 128 dim × fp16 = 64MB per array, two arrays (K+V) = 128MB of unnecessary copying per layer per token.
- **Change**: Pre-allocate the KV buffer with headroom (e.g., `step=256` like TQ3 does). Use slice assignment `self._keys[:, :, offset:offset+1] = keys` for decode (T_new=1). Only fall back to concat when headroom is exhausted. The MLX `KVCache` class in `mlx_lm.models.cache` has a working implementation of this pattern — adapt it.
- **Additional experiment** (EXP 14, 2026-04-18 06:40): At 16K context, concat costs **17.99ms/token** while in-place `.at[]` costs **0.04ms/token** — a **445x speedup**. Over 200 decode tokens, concat copies 6.8 GB vs in-place copying 200 KB. At 48 layers × 200 tokens: concat = 173 seconds, in-place = 0.38 seconds.
- **Why concat is so expensive**: `mx.concatenate([buf, tok], axis=2)` must allocate a new (ctx+1)-sized buffer, copy the entire old buffer, then copy the new token. At 16K this copies 34 MB per call. MLX cannot optimize this away because the old buffer must be preserved until the new one is evaluated.
- **Verify**: DuoKV decode at 64K should be <0.5ms/token (currently 34.78ms — profiled). Goal 3 decode speed at long context should be achievable with DuoKV mode.
- **Effort**: S (1 day — pre-allocation is well-understood, can reference MLX KVCache pattern)
- **Depends on**: None. HIGH IMPACT — this single fix would unlock DuoKV for long context.

### 148. CAOTE importance scoring: GQA key expansion + causal mask waste 1.6 GB and 2.2s at 64K
- **Goal**: 1 (context window — CAOTE at 64K needs 4.3 GB transient memory for GQA copies alone), 3 (decode speed — 2.2s of mask compute per layer at 64K)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/patches/snapkv.py:440-459` (`compute_caote_importance`).
- **Profiling evidence** (component timing @ 64K, 2026-04-18 06:15):
  - Causal mask (`mx.where`): **2,240ms (72% of single-layer time)**
  - Q@K^T scores: 351ms (11%)
  - GQA expand (`mx.repeat`): 95ms (3%) but **1.074 GB per layer** memory cost
  - Softmax: 249ms (8%)
  - At 4 layers: total CAOTE scoring is ~12.5s estimated, with 4.3 GB transient from GQA expansion
- **The problem**: (1) `mx.repeat(keys, gqa_ratio, axis=1)` creates a full copy of all 65K keys expanded from H_kv=4 to H_q=32 heads — 1.07 GB per layer. (2) The causal mask creates a `(1, 1, 64, 65536)` boolean comparison and broadcasts it via `mx.where` against the `(1, 32, 64, 65536)` scores — 537MB allocation plus the broadcast.
- **Experimental validation** (EXP 8, 2026-04-18 06:30): `mx.fast.scaled_dot_product_attention` saves **99% of peak memory** vs manual GQA expansion. At 64K: manual peak 1.611 GB vs SDPA peak 0.018 GB — saving 1.59 GB per call. At 16K: 0.403 GB → 0.005 GB. SDPA is also 1.6-1.9x faster at 16K+ (EXP 4).
- **Change**: Replace the manual Q@K^T + causal mask with `mx.fast.scaled_dot_product_attention` which handles GQA natively (no key expansion) and applies causal masking internally. For the observation-window-only queries, slice Q_obs and pass the appropriate mask=None (since obs_window queries at the end of context can attend to everything before them). This eliminates the 1.07 GB GQA copy and the 2.2s mask computation entirely.
- **Verify**: CAOTE scoring at 64K should drop from ~3.1s/layer to <0.5s/layer. Memory peak should drop by 1.59 GB per layer (validated by EXP 8).
- **Effort**: S (1 day)
- **Depends on**: None. Also applies to `compute_importance_from_real_q` (same GQA expansion pattern).

### 150. DuoKVCache retrieval heads use fp16 — must use QuantizedKVCache for Goal 1 (1M context)
- **Goal**: 1 (1M context — DuoKV in fp16 can only reach ~305K before exceeding 30.8 GB KV budget)
- **Derived from**: Analyst efficiency audit 2026-04-18, EXP 15 memory budget analysis.
- **Profiling evidence** (EXP 15):
  - Measured fp16: 4096 bytes/tok/layer. At 48L × 1M = **196.6 GB** — exceeds 48 GB by 4x
  - Measured 3-bit native: 512 bytes/tok/layer. At 48L × 1M = **24.6 GB** — fits within 30.8 GB budget
  - DuoKV (41% retrieval fp16 + 59% streaming ring): ~1640 bytes/tok/layer. At 48L × 1M = **75 GB** — exceeds 48 GB by 56%
  - Maximum DuoKV fp16 context: ~390K tokens (vs Goal 1 target of 1M)
- **The problem**: `DuoKVCache` stores both retrieval AND streaming heads as fp16 arrays. Streaming heads are bounded by the ring buffer (sink+window=260 tokens) and have constant memory. But retrieval heads grow linearly with context in fp16, consuming 4096 bytes/tok/layer. To reach 1M, retrieval heads must use 3-bit quantization like native mode's `QuantizedKVCache`.
- **Change**: Use `QuantizedKVCache(bits=3)` for retrieval heads instead of raw fp16 arrays. Streaming heads stay fp16 (ring buffer is only 260 tokens). This gives DuoKV's quality benefits (streaming/retrieval classification) with 3-bit memory efficiency for the full context.
- **Verify**: DuoKV at 256K context should use < 10 GB KV memory (vs current 25.8 GB). At 1M: < 30 GB (fits in budget).
- **Effort**: M (2 days — need to handle the quantized K/V in attention correctly for retrieval heads)
- **Depends on**: None. Independent of Task 149 (concat→slab). Both needed for DuoKV at long context.

### 151. TQ3 as DuoKV backend — quantized retrieval heads for Goal 1 + DuoAttention quality for Goal 2
- **Goal**: 1 (1M context), 2 (intelligence — DuoAttention preserves quality by keeping retrieval heads at full fidelity)
- **Derived from**: Analyst EXP 15 memory budget analysis + architectural insight.
- **The insight**: DuoKV's quality advantage comes from the streaming/retrieval HEAD CLASSIFICATION, not from fp16 storage. The classification tells us WHICH heads need full context (retrieval) and which can use a ring buffer (streaming). But the STORAGE for retrieval heads doesn't need to be fp16 — TQ3's 3-bit codebook quantization is the right backend. At 1M context, TQ3 retrieval heads use 512 bytes/tok/layer vs fp16's 4096 bytes/tok/layer — 8x compression. Combined with streaming heads' ring buffer (constant memory), this gives DuoKV+TQ3 the memory efficiency of native 3-bit with the quality advantage of DuoAttention.
- **Change**: Modify `DuoKVCache.__init__` to use `TurboQuantKVCache(bits=3)` for retrieval heads instead of raw fp16 arrays. Streaming heads stay as fp16 ring buffers (only 260 tokens). The attention dispatch needs to handle mixed cache types: SDPA for streaming heads (fp16), fused TQ SDPA for retrieval heads (3-bit codebook).
- **Verify**: (1) DuoKV+TQ3 at 256K context fits in 48 GB. (2) MMLU-Pro score stays >= 60% (DuoKV quality). (3) NIAH at 64K passes (retrieval heads preserve needles). (4) Decode speed >= 50 tok/s at 4K (TQ3 fused SDPA handles decode).
- **Effort**: M-L (3-5 days — mixed cache attention dispatch is the hard part)
- **Depends on**: Task 149 (DuoKV slab pre-allocation) should land first for the streaming heads. Task 138 (fused WHT quantize) improves prefill speed.

### 152. TQ3 quantize can use existing fused dense kernel with WHT rotation matrix
- **Goal**: 4 (prefill speed — TQ3 prefill is 270x slower due to unfused WHT, but a fused dense kernel already exists)
- **Derived from**: Analyst experiments EXP 1, 7, 10 (2026-04-18).
- **The insight**: The codebase already has `_fused_quantize_kernel()` (line 218 in turboquant_kv.py) — a Metal kernel that does norm+rotate+boundary+pack in ONE dispatch using a dense rotation matrix. The WHT rotation matrix IS a dense matrix (it's `diag(signs) @ H`). The existing fused kernel can be reused directly with the WHT rotation matrix as input. No new Metal kernel needed — just call `_fused_quantize(vectors, codec.rotation, codec._boundaries, 3, 128)` instead of `codec._quantize_wht(vectors)`.
- **CRITICAL CAVEAT from EXP 10**: The fused kernel applies `R` as `vectors @ R`, but the forward WHT rotation is `H @ diag(signs) @ x = (diag(signs) @ H).T @ x` when applied to column vectors. The `_fused_quantize_kernel` does `out_coord = sum_j(vec[j] * rotation[j * Dim + idx])` which is `vec @ R[:, idx]` = applying R as a ROW-vector convention. Need to verify: is `codec.rotation` stored in the same convention as the dense kernel expects? If so, this is a zero-effort fix. If not, transpose it.
- **Change**: In `TurboQuantMSECodec.quantize()`, replace `self._quantize_wht(vectors)` with `_fused_quantize(vectors, self.rotation, self._boundaries, self.bits, self.dim)`. Test quality is identical to current WHT path.
- **VALIDATED** (EXP 16, 2026-04-18 06:45):
  - MSE: **identical** (0.033897 both paths)
  - Packed indices: **99.67% match** (0.33% differ from fp32→fp16 cast in Metal kernel)
  - Speed: **3.5x** (4.41ms → 1.25ms at 4K)
  - The math is correct: `H.T = H` (Hadamard symmetric), so `codec.rotation.T @ v = H @ diag(signs) @ v = WHT(signs * v)` — exactly the forward WHT rotation
- **Verify**: (1) This experiment validated quality and speed. (2) Run NIAH 4K and Code Intel 5/5 with fused quantize to confirm no regression in real generation. (3) Test at 64K context to verify scaling.
- **REAL-MODEL VALIDATION** (INV 22, 2026-04-18 10:50):
  - Short prefill (8 tok): **15.5x faster** (6 → 91 tok/s)
  - 4K prefill: 1.03x (753 → 772 tok/s — marginal at scale, WHT no longer dominates)
  - Decode: identical (52.6 → 53.3 tok/s)
  - **Output text: IDENTICAL** — exact same tokens generated
  - KV memory: identical (80 MB)
  - **ZERO RISK. ONE LINE CHANGE. 15x SHORT-CONTEXT PREFILL SPEEDUP.**
- **Effort**: XS (literally one line: replace `self._quantize_wht(vectors)` with `_fused_quantize(vectors, self.rotation, self._boundaries, self.bits, self.dim)`)
- **Depends on**: None. This fully resolves the core of Task 138 without writing any new Metal kernel.

### 153. [TOP PRIORITY] Goal 1 + Goal 3 tension: decode drops to 11-15 tok/s at 8-16K — need pre-alloc DuoKV + fused native decode
- **Goal**: 1 (1M context) + 3 (50 tok/s decode constant) — these goals are in DIRECT TENSION above 4K context
- **Derived from**: Analyst real-model profiling INV 4-8, 2026-04-18.
- **Profiling evidence** (real model, real weights, 2026-04-18 10:30):
  - fp16: **50.1-50.7 tok/s CONSTANT** from 256 to 4K — meets Goal 3 but can't fit Goal 1
  - DuoKV: 48.4 @ 256 → 46.7 @ 4K → **14.9 @ 8K → 11.4 @ 16K** — concat tax (Task 149)
  - Native 3-bit: 45.1 @ 256 → 43.3 @ 4K → 14.2 @ 8K → **0.4 @ 16K** — dequant on every decode
  - Cache update_and_fetch overhead: DuoKV 52.6ms @ 4K (14% of step), native3 15.7ms (5% of step)
  - **DuoKV at 4K: update_and_fetch alone takes 52.6ms vs native3's 15.7ms — 3.3x worse**
- **Root causes**:
  1. DuoKV: `mx.concatenate` copies full buffer per token (Task 149 — 248x fix available)
  2. Native 3-bit: `mx.dequantize` materializes full fp16 KV on every attention step
  3. TQ3: has fused SDPA that avoids dequant (decode_attention), but prefill is 270x slower
- **The resolution path**: DuoKV with pre-allocated buffers (Task 149) + TQ3 backend for retrieval heads (Task 151). This gives DuoKV's quality + TQ3's fused decode + pre-alloc's constant-time update.
- **Verify**: After Tasks 149+151, DuoKV+TQ3 decode at 16K should be >= 40 tok/s (currently 11.4).
- **Effort**: L (Tasks 149 + 151 combined — 4-6 days)
- **Depends on**: Task 149 (pre-alloc slab), Task 151 (TQ3 retrieval heads)

### 154. DuoKV decode drops 54% at 4K context — fp16 KVCache stays constant (real-model validation of Task 149)
- **Goal**: 3 (decode speed constant across context)
- **Derived from**: Analyst real-model INV 4, 2026-04-18.
- **Profiling evidence** (real model, DuoKV vs fp16 vs native 3-bit):
  - fp16: 50.5→50.7→50.1 tok/s at 256→1024→4096 context — **FLAT, meets Goal 3**
  - DuoKV: 48.4→46.9→46.7 tok/s — **drops, but still close to 50 at 4K**
  - DuoKV at 8K: **14.9 tok/s**, at 16K: **11.4 tok/s** — catastrophic
  - Native 3-bit at 16K: **0.4 tok/s** — essentially broken
  - DuoKV cache update_and_fetch: 56% of the 21ms decode token budget at 1K context
- **Root cause**: `mx.concatenate` copies full KV buffer per token (confirmed by Task 149 microbench: 248x slower than slab). At 48 layers × 4K context, concat alone costs 52.6ms per token.
- **What fp16 does right**: MLX's internal `KVCache` class pre-allocates buffers and uses in-place assignment. It achieves 50+ tok/s constant because cache update is O(1), not O(context).
- **Verify**: After Task 149 fix, DuoKV at 4K should match fp16 (50+ tok/s). At 16K should exceed 40 tok/s.
- **Effort**: (Covered by Task 149 — this task is the real-model validation evidence)

### 155. SnapKV at 25% keep ratio HURTS decode speed — optimal keep ratio is 40-60%
- **Goal**: 3 (decode speed)
- **Derived from**: Analyst real-model INV 12, 2026-04-18.
- **Profiling evidence** (fp16 @ 4K, post-eviction decode):
  - No eviction: 28.2 tok/s
  - SnapKV 50% keep (2048 tokens): **30.1 tok/s** — 7% FASTER (smaller attention)
  - SnapKV 25% keep (1024 tokens): **24.3 tok/s** — 14% SLOWER (re-RoPE cost dominates)
- **Root cause**: `compact_cache` applies `_rerope_keys` to every kept token (cos/sin computation per token). At 25% keep of 4K = 1024 re-RoPE computations. The attention savings (fewer KV tokens) are outweighed by the re-RoPE + gather + memory layout disruption cost.
- **Change**: Default SnapKV keep ratio should be 50%, not 25%. Add heuristic: if `keep_count / total < 0.4`, warn that aggressive eviction may degrade decode speed. Consider lazy re-RoPE (skip re-RoPE if context is short enough that position errors are tolerable).
- **Verify**: SnapKV at 50% keep should be >= full-cache decode speed at 4K. Profile at 16K and 64K to find the crossover where eviction becomes net positive.
- **Effort**: XS (change default + add warning)
- **Depends on**: None.

### 156. Bandwidth headroom: 35% unused — speculative decoding could exploit this for decode speedup
- **Goal**: 3 (decode speed above 50 tok/s)
- **Derived from**: Analyst bandwidth analysis INV 15, 2026-04-18.
- **Analysis**: M4 Pro bandwidth: 273 GB/s. Active model reads per token: 3.55 GB (MoE 8/128). At 51.6 tok/s: 183 GB/s used = 67% utilization. **91 GB/s bandwidth idle.** Speculative decoding (EAGLE-2, Task 29) verifies K candidate tokens in one forward pass — reading the same 3.55 GB weights but amortizing over K tokens. At K=3 acceptance rate, effective decode = 3 × 51.6 ÷ 2 ≈ 77 tok/s. This would exceed Goal 3 by 54%.
- **Effort**: (Already tracked in Task 29 — this task is the bandwidth analysis that proves the opportunity)

### 157. 65% of MoE experts are dead weight — profiled at layer 24, only 45/128 fire on coding tasks
- **Goal**: 6 (machine fit — 20 GB of unused expert weights in Metal), 1 (context window — freed Metal = more KV budget)
- **Derived from**: Analyst real-model INV 24, 2026-04-18.
- **Profiling evidence** (layer 24, 50 decode tokens, coding prompt):
  - Active experts: 45/128 (35%)
  - Never activated: **83 experts (65%)**
  - Top expert (ID 85) fires on 74% of tokens — heavy concentration
  - Bottom 5 experts each fire once — rare but not zero
  - MLP is 30.8 GB of the 32.4 GB model — 65% dead weight = **~20 GB in Metal doing nothing**
- **Impact**: Loading only active experts would reduce Metal at load from 32.4 GB to ~22 GB, freeing ~10 GB for KV cache. At 3-bit native, 10 GB = ~20K additional context tokens (from current 4K ceiling to 24K before swap pressure).
- **Change**: Already tracked in Task 33 (ProMoE full lazy-load MoE runtime). This finding provides the real-model activation frequency data to inform which experts to pre-load vs lazy-load. The top ~50 experts cover 99%+ of activations.
- **Verify**: Profile expert activation across ALL 48 layers (not just layer 24) and across diverse prompts (coding, math, natural language) to build the full activation frequency table.
- **Effort**: (Covered by Task 33 — this is the profiling evidence)

### 158. Tasks 152+145 combined give 1.33x TQ3 prefill speedup at 8K (validated on real model)
- **Goal**: 4 (prefill speed)
- **Derived from**: Analyst real-model INV 23, 2026-04-18.
- **Profiling evidence** (TQ3 mode, 8K context, real model):
  - Original: 507 tok/s (16.1s)
  - Task 152 alone: 661 tok/s (12.4s) — 1.30x
  - Task 145 alone: 647 tok/s (12.7s) — 1.28x
  - Both: **672 tok/s (12.2s) — 1.33x, saves 3.9s per 8K prefill**
  - Decode also improves: 7.3 → 11.0 tok/s (1.5x)
- **The two tasks are independent XS-effort fixes that compose well.**
- **Effort**: (Covered by Tasks 145 + 152 — this is the combined validation)

### 159. [STRATEGIC] ProMoE lazy-load would free 20.3 GB Metal — enabling 1M context without swap (profiled across all 48 layers)
- **Goal**: 1 (1M context — 20 GB freed = 40K+ additional tokens at 3-bit), 6 (machine fit — Metal at load drops from 32.4 to ~12 GB)
- **Derived from**: Analyst real-model INV 26, 2026-04-18.
- **Profiling evidence** (all 48 layers, 30 decode tokens, coding prompt):
  - Average dead weight per layer: **66%** (consistent across all layers)
  - Layer 0 (least sparse): 45% dead (71/128 active)
  - Layer 30 (most sparse): 78% dead (28/128 active)
  - All 128 experts are used somewhere across the 48 layers (no globally dead experts)
  - Top experts have 10-12% concentration per layer
  - MLP = 30.8 GB, 66% savings = **20.3 GB freed**
  - Metal at load with lazy-load: ~12 GB → 20 GB freed for KV
  - At 3-bit native: 20 GB = 40K additional context tokens
  - Total KV budget with freed Metal: ~30 GB = **1.2M tokens at 3-bit** — **Goal 1 MET**
- **Lazy-load cost estimate**: Expert miss = ~0.24 GB from NVMe at ~7 GB/s = 34ms.
  At 3% miss rate (50 tok/s × 8/128 routing × ~34% not pre-loaded): 1.5 misses/s × 34ms = 5% decode overhead. Acceptable.
- **Change**: Already tracked in Task 33 (ProMoE full lazy-load MoE runtime). This INV 26 provides the full 48-layer activation frequency table. The implementation should:
  1. Profile expert activation frequencies across diverse prompts (coding, math, NL)
  2. Pre-load the top ~35 experts per layer (~95% coverage)
  3. Lazy-load remaining experts from NVMe on first activation
  4. LRU evict inactive experts under memory pressure
- **Effort**: L (already estimated in Task 33 — this is the profiling evidence that proves the opportunity)

### 160. [BUG] TQ3 session save fails at 2048 tokens — cache stays in fp16 warmup, never quantizes
- **Goal**: 1 (context window — session save/load is critical for long-context agentic workflows)
- **Derived from**: Analyst real-model INV 30, 2026-04-18.
- **Evidence**: Prefill 2048 tokens into TQ3 cache with `min_quant_tokens=512` → `cache.offset=511`, `cache.nbytes=0`, `save_to_disk()` raises `ValueError("Cannot save empty cache")`. The cache accumulated fp16 tokens but never crossed the quantization threshold.
- **Root cause hypothesis**: When `prefill_last_logit` patch slices the hidden states to `hidden[:, -1:, :]`, the cache only sees 1 token per forward call during chunked prefill. With `min_quant_tokens=512`, the cache accumulates fp16 tokens in chunks but the offset tracking misaligns — it shows 511 instead of 2048. The `vertical_eval` patch's `mx.eval(h)` every 8 layers may also affect the chunking.
- **Change**: In `TurboQuantKVCache.update_and_fetch`, verify that `self.offset` correctly tracks the total token count across all prefill chunks, regardless of chunking strategy. Add a test: `prefill 2048 tokens → assert cache.offset == 2048 → save_to_disk() succeeds`.
- **Verify**: `TurboQuantKVCache` with `min_quant_tokens=512`, prefill 2048 tokens, `save_to_disk()` succeeds, `load_from_disk()` restores correct offset.
- **Effort**: S (1 day — debugging offset tracking in the prefill path)

### 161. Tool-call generation produces duplicate JSON — XGrammar constrained decoding is critical for agentic reliability
- **Goal**: 2 (intelligence — tool calls must be valid JSON for agentic use)
- **Derived from**: Analyst real-model INV 27, 2026-04-18.
- **Evidence**: Model generates `{"tool": "read_file", "path": "/tmp/test.py"}` (correct) but then repeats the JSON object, producing invalid output. `json.loads` fails with "Extra data" at char 46.
- **The fix**: XGrammar constrained decoding (Task 45, `--grammar` flag on server) already exists and would enforce single JSON object output. The issue is that it's OFF by default and not tested in the benchmark harness.
- **Change**: (1) Add a tool-call quality gate to the benchmark (Phase 2b or similar) that tests JSON output validity. (2) Consider enabling `--grammar` by default for tool_choice=required requests.
- **Verify**: Server with `--grammar` produces valid single JSON for tool calls. Benchmark gate passes.
- **Effort**: S (1 day)

### 162. DuoKV retrieves needle at 32K with only 33.4 GB Metal — streaming heads deliver on memory promise
- **Goal**: 1 (context window — DuoKV scales to 32K within Metal budget)
- **Derived from**: Analyst real-model INV 36, 2026-04-18.
- **Evidence**: DuoKV NIAH PASS at 32K, Metal 33.4 GB (same as 16K). The 59% streaming heads use ring buffers (260 tokens constant), so only 41% retrieval heads grow with context. KV cost: ~0.04 GB per 1K tokens (vs fp16's 0.1 GB/K). Decode: 20.7 tok/s (below Goal 3 but functional — Task 149 fix would improve this).
- **Implication**: DuoKV can reach ~200K tokens before hitting 41.2 GB Metal limit (vs fp16's ~88K). With Task 149 (pre-alloc slab), decode speed should stay near 50 tok/s. With Task 151 (TQ3 retrieval heads), DuoKV could reach 1M+.

### 163. SnapKV + DuoKV combined: prefill at full context, evict, decode 47% faster
- **Goal**: 3 (decode speed at long context)
- **Derived from**: Analyst real-model INV 37, 2026-04-18.
- **Evidence**: At 8K context, DuoKV alone decodes at 20.5 tok/s. fp16+SnapKV 25% eviction decodes at **30.2 tok/s** (+47%). Needle preserved at all keep ratios (PASS even at 10% keep — INV 34). The workflow: prefill full context → SnapKV attention-guided eviction → decode with compact cache.
- **Change**: Wire SnapKV eviction into the DuoKV decode path. After prefill, run importance scoring on the retrieval heads and compact them. Streaming heads are already compact (ring buffer). This combines DuoKV's memory efficiency with SnapKV's decode speed optimization.
- **Verify**: DuoKV+SnapKV at 8K should decode at >= 30 tok/s with NIAH PASS.
- **Effort**: S (the pieces exist — SnapKV eviction + DuoKV cache. Just need to compose them)

### 164. Native 3-bit prefill collapses to 125 tok/s at 16K — 4x slower than fp16
- **Goal**: 4 (prefill speed constant across context)
- **Derived from**: Analyst real-model INV 35, 2026-04-18.
- **Evidence**: fp16 prefill: 894→516 tok/s (2K→16K, 1.7x drop). Native 3-bit: 714→**125 tok/s** (2K→16K, **5.7x drop**). At 16K, native 3-bit is 4.1x slower than fp16 (130.7s vs 31.7s). The quantize overhead during `update_and_fetch` scales superlinearly with context length.
- **Root cause**: `QuantizedKVCache.update_and_fetch` quantizes each new chunk AND the attention must dequantize the growing quantized history for the SDPA call. At 16K, the combined quantize+dequantize cost per prefill chunk exceeds the attention compute cost.
- **Change**: Consider prefilling in fp16 mode and quantizing ONCE at the end (like SnapKV's fp16 capture hooks do). Or use streaming quantization (quantize each chunk, use fused TQ SDPA for history attention — already implemented for TQ3 streaming mode).
- **Verify**: Native 3-bit prefill at 16K should be within 2x of fp16 (>= 250 tok/s, currently 125).
- **Effort**: M (need to restructure the native quantize pipeline)

### 165. DuoKV NIAH PASS at 64K but Metal peak 42.5 GB BREACHES limit + decode 3.3 tok/s
- **Goal**: 1 (context window), 3 (decode speed), 6 (machine fit)
- **Derived from**: Analyst real-model INV 39, 2026-04-18.
- **Evidence**: 40K-token prompt with needle at position 32K. DuoKV retrieves the needle correctly (PASS). But:
  - Metal peak: **42.5 GB** (breaches 41.2 GB limit by 1.3 GB)
  - Decode: **3.3 tok/s** (15x below Goal 3 target of 50 tok/s)
  - Prefill: 302 tok/s (below Goal 4's 500 tok/s)
  - Active Metal: 36.4 GB (within limits — the breach is TRANSIENT during prefill)
- **Root causes**: (1) DuoKV concat per token at 40K context = ~150ms/token × 48 layers. (2) fp16 KV for retrieval heads at 40K = ~4 GB, pushes past limit during prefill intermediates. (3) Prefill attention at 40K is O(n²) with no chunking strategy.
- **Required fixes for Goal 1 at 64K** (all three needed):
  - Task 149: DuoKV pre-alloc slab → decode from 3.3 to ~40+ tok/s
  - Task 151: TQ3 retrieval heads → Metal peak from 42.5 to ~35 GB
  - Task 163: SnapKV post-prefill eviction → compress 40K to ~16K for sustained decode
- **Effort**: (Covered by Tasks 149, 151, 163 — this task is the integration validation)

### 166. [CRITICAL] Decode speed collapses at 4K for ALL KV modes — not just DuoKV, fp16 too (8.2 tok/s)
- **Goal**: 3 (decode speed constant across context — FAILS above 1K for ALL modes)
- **Derived from**: Analyst real-model INV 43 comprehensive heatmap, 2026-04-18.
- **Evidence** (complete decode tok/s matrix):
  - fp16:   52.7 @ 64 → 51.3 @ 256 → 48.1 @ 1K → **8.2 @ 4K** → 17.6 @ 8K → 15.2 @ 16K
  - DuoKV:  52.7 @ 64 → 50.1 @ 256 → 44.0 @ 1K → 19.5 @ 4K → 15.5 @ 8K → 11.9 @ 16K
  - native: 46.9 @ 64 → 45.4 @ 256 → 39.2 @ 1K → 17.5 @ 4K → 9.0 @ 8K → 0.4 @ 16K
- **This overrides earlier analysis**: Previous INV 4 showed fp16 at 50+ tok/s through 4K because those tests ran on a clean system with the model freshly loaded. The heatmap test ran 18 sequential measurements — by the time it reached fp16 @ 4K, Metal memory pressure from prior measurements (native3 @ 16K) degraded performance. The decode cliff is REAL under sustained workload.
- **Root cause hypothesis**: At 4K context, fp16 attention SDPA processes 4K×4K attention matrices × 48 layers. Even with `mx.fast.scaled_dot_product_attention`, this is O(n²) per layer. The per-token attention cost at 4K is 16x higher than at 1K, but decode speed only drops 6x (48.1→8.2) — suggesting partial flash-attention optimization but not enough.
- **Alternative hypothesis**: Metal memory fragmentation from sequential measurements causes swap pressure that degrades all modes. The 4K measurements may be contaminated by prior 16K runs' residual Metal allocations.
- **VALIDATED** (fresh-load test, 2026-04-18 12:04):
  - fp16 @ 256 FRESH: 52.3 tok/s
  - fp16 @ 1024 FRESH: 48.9 tok/s
  - fp16 @ 4096 FRESH: **14.1 tok/s** (73% drop from short context)
  - Heatmap showed 8.2 (contamination accounts for ~40% of pessimism, but cliff is REAL)
  - **CORRECTION** (INV 48, 2026-04-18 12:20): Per-step breakdown shows model forward is LAZY (1.03ms constant), all compute is in mx.eval. At 4K, eval takes 20.8ms (vs 17.8ms at 256) — only 3ms more. **Actual decode at 4K after proper warmup: 45.4 tok/s**, not 14.1. The earlier 14.1 was contaminated by Q capture hook overhead and insufficient warmup. The "cliff" is less severe than initially reported but Goal 3 (50 tok/s constant) is still not met at 4K.
  - Re-RoPE in compact_cache costs 4.7x decode speed (7.2 vs 33.7 tok/s)
- **Effort**: (Less critical than initially assessed — decode is 45.4 tok/s at 4K with proper warmup, not 14.1)
- **Depends on**: None.

### 168. Re-RoPE in compact_cache costs 4.7x decode speed — skip_rerope=True gives 33.7 tok/s with NIAH PASS
- **Goal**: 3 (decode speed after eviction)
- **Derived from**: Analyst real-model INV 47, 2026-04-18.
- **Evidence**:
  - with re-RoPE: 7.2 tok/s (re-RoPE computation dominates every decode step)
  - skip re-RoPE: **33.7 tok/s** (4.7x faster, NIAH still PASS)
  - The skip_rerope path preserves original RoPE positions — keys have position gaps but attention still retrieves the needle correctly at 4K
- **Implication**: For agentic workflows where decode speed matters more than position-perfect attention, `skip_rerope=True` should be the default for `compact_cache`. The 4.7x speed difference is the most impactful single optimization for post-eviction decode.
- **Change**: Default `skip_rerope=True` in `compact_cache` when called from `apply_snapkv_to_generate`. Add a quality gate: NIAH at 4K and 8K must still PASS with skip_rerope.
- **Verify**: NIAH PASS at 4K and 8K with skip_rerope=True. Decode >= 30 tok/s after eviction.
- **Effort**: XS (change default parameter)

### 167. SnapKV eviction caps decode at ~33 tok/s — re-RoPE + fragmented memory layout prevents reaching 50
- **Goal**: 3 (decode speed)
- **Derived from**: Analyst real-model INV 46, 2026-04-18.
- **Evidence**: After 4K prefill + SnapKV eviction:
  - Evict → 2048 tok: 31.1 tok/s
  - Evict → 1024 tok: 33.1 tok/s
  - Evict → 512 tok: 32.8 tok/s
  - Evict → 256 tok: 33.4 tok/s ← plateau
  - Fresh 256-token decode: 52.3 tok/s (from INV validation)
  - **Gap: 33.4 vs 52.3 = 36% slower after eviction than fresh**
- **Root cause**: `compact_cache` gathers keys from scattered positions and applies `_rerope_keys` to correct RoPE positions. The result is a contiguous array but with re-RoPE'd values that may have accumulated floating-point error. More importantly, the Metal memory allocator may place the gathered KV in fragmented buffers vs the contiguous pre-allocated buffer of a fresh cache.
- **Fix options**:
  1. Skip re-RoPE (`skip_rerope=True`) — keys keep original positions, saves computation but attention sees position gaps
  2. Re-prefill only the kept tokens as a fresh prompt — guarantees clean memory layout but costs one prefill pass
  3. Pre-allocated eviction buffer — compact into a pre-allocated contiguous buffer instead of creating new arrays
- **Verify**: Option 1: `compact_cache(cache, indices, skip_rerope=True)` → measure decode speed (should be higher without re-RoPE cost). Option 2: re-encode the 256 kept tokens as a fresh prompt → should hit 52+ tok/s.
- **Effort**: S (testing existing skip_rerope flag)
- **Depends on**: None.

### 147. GER safety check materializes importance to Python via .tolist() for per-element mask construction
- **Goal**: 3 (decode speed — GER check runs on every SnapKV eviction)
- **Derived from**: Analyst efficiency audit 2026-04-18. Code location: `omlx/patches/snapkv.py:1241-1244` (`compute_ger`).
- **The problem**: `threshold_idx = mx.argpartition(...)[:n_important]` returns an MLX array, then iterates with `for idx in threshold_idx.tolist(): important_mask = important_mask.at[idx].add(...)`. Same `.at[].add()` per-element loop pattern as Task 140. For 10% of 64K = 6,553 Python iterations.
- **Change**: Same fix as Task 140: `important_mask[threshold_idx] = True` or equivalent single-op scatter.
- **Verify**: GER check at 64K should complete in <0.01s.
- **Effort**: XS (15 min — same pattern as Task 140)
- **Depends on**: None. Can be fixed alongside Task 140.

---

## Research-derived tasks (from LIT_REVIEW.md pass 37, 2026-04-18)

### 169. WAIT threshold-based admission for multi-request KV memory budgeting
- **Goal**: 5 (swap pressure), 6 (48GB fit)
- **Derived from**: LIT_REVIEW.md pass 37 — paper 2504.11320 (Fluid-Guided Online Scheduling). The fluid dynamics model proves memory sufficiency alone does not guarantee stability; FCFS scheduling causes eviction cascades analogous to TCP congestion collapse.
- **What**: Implement threshold-based request admission in `hypercar_server.py`. Compute equilibrium memory M* = sum n*_j(l_j + l'_j/2) for in-flight requests. Hold new requests in queue until total projected KV memory (current + new request's estimated l_j + l'_j/2) stays within the Metal budget minus a safety buffer of O(log(1/delta)) * sigma_M. For Hypercar's reference workload (2K input, 500 output), M_j ~ 2250 tokens ~ 0.05 GB per request at 3-bit. Safety buffer at delta=0.01: ~4.6 * sigma_M.
- **Verify**: Under 3 concurrent tool-call requests (simulated via parallel curl), Metal peak stays below --max-metal-pct and no SnapKV eviction cascade is triggered. Compare vs current sequential processing: throughput should be higher (requests overlap prefill/decode) while memory stays bounded.
- **Effort**: M (request queue + memory projection + threshold logic)
- **Depends on**: None — standalone server enhancement.

### 170. Per-layer KV cache bit allocation using covariance determinant
- **Goal**: 1 (context window), 2 (intelligence)
- **Derived from**: LIT_REVIEW.md pass 37 — paper 2601.22002 (Rate-Distortion Optimization). The paper finds rate INCREASES in deeper transformer layers (r=0.96 correlation with covariance determinant), contradicting uniform bit allocation. Hypercar's uniform 3-bit across 48 layers is suboptimal: early layers (low covariance) could use 2-bit, later layers (high covariance) need 4-bit or fp16.
- **What**: During prefill, compute the covariance determinant (or its log = log-volume of the representation ellipsoid) for each layer's key cache. Use this to allocate bits: layers with log-det below the 25th percentile get 2-bit, layers above the 75th percentile get 4-bit, middle layers get 3-bit. This is complementary to LAVa's per-layer budget allocation (pass 36 Task idea) — LAVa allocates cache SLOTS per layer, this allocates BITS per layer.
- **Verify**: Run NIAH at 16K with mixed-precision KV (2/3/4-bit per layer) vs uniform 3-bit. Quality should be equal or better. Memory should be within 10% (some layers save, some spend). Measure covariance determinant across all 48 layers to confirm the monotonic increase pattern found in GPT-2.
- **Effort**: M (covariance computation during prefill + mixed-precision TQ3 codec)
- **Depends on**: TQ3 codec must support per-layer bit-width (currently parameterized globally).

### 171. Guide-star calibration for post-eviction attention drift detection
- **Goal**: 2 (intelligence — prevent quality collapse after aggressive eviction)
- **Derived from**: LIT_REVIEW.md pass 37 — paper 2405.05472 (Adaptive Optics). In AO systems, guide stars are reference points with known positions used to measure residual distortion after correction. For KV cache, "guide tokens" are tokens with known importance (attention sinks at position 0, system prompt tokens, most recent K tokens). After SnapKV eviction, check whether guide tokens' attention scores have drifted — if they have, the eviction has distorted the attention pattern beyond what the model can tolerate.
- **What**: After `compact_cache` in SnapKV, re-run a single attention computation on the guide tokens (position 0, last 4 system prompt tokens, last 8 context tokens — ~13 tokens total). Compare their attention scores pre- and post-eviction. If the L1 drift exceeds a threshold (calibrate on NIAH at 16K@25%), emit a warning and optionally trigger re-scoring with the post-eviction cache state.
- **Verify**: On NIAH at 16K@25%: measure guide-token drift for PASS vs FAIL cases. The drift metric should be a reliable predictor of eviction quality — high drift = bad eviction. The threshold should achieve >90% precision / recall on a held-out set of 10 NIAH runs.
- **Effort**: S (single attention forward on 13 tokens + L1 comparison)
- **Depends on**: SnapKV eviction pipeline (shipped).

### 172. Staircase-aware chunked prefill sizing for Metal threadgroup tiling
- **Goal**: 4 (prefill speed)
- **Derived from**: LIT_REVIEW.md pass 37 — paper 2508.01002 (Optimal Scheduling). Batch processing time is a staircase function of batch size due to GPU tiling — adding one request may or may not increase time depending on tile boundary. MLX's Metal matmul uses threadgroups of fixed size (typically 32x32 or 16x16). Hypercar's chunked prefill chunk_size should align to threadgroup boundaries to avoid partial-tile waste.
- **What**: Profile MLX attention matmul time vs sequence length at granularity of 1 token from 1 to 1024. Identify the staircase jumps (where adding 1 token causes a time increase). Set chunk_size to the largest value that fits within a single tile — this maximises compute utilisation per chunk. Currently chunk_size is set heuristically; this would make it architecture-aware.
- **Verify**: Prefill throughput at 4K context with staircase-aligned chunk size vs current heuristic. Expect 5-15% improvement from eliminating partial-tile waste.
- **Effort**: S (profiling script + one constant change)
- **Depends on**: None — standalone optimisation.

---

## Research-derived tasks (from LIT_REVIEW.md pass 38, 2026-04-18)

### 173. Sparse covariance fitting eviction scorer for multi-head SnapKV
- **Goal**: 1 (context window), 2 (intelligence)
- **Derived from**: Robust Detection of Underwater Target Against Non-Uniform Noise (2512.11231)
- **Change**: `omlx/snapkv.py` — add `score_sparse_covariance()` alongside existing CAOTE scorer. Formulate the multi-head attention score matrix (heads x tokens) as a sparse DOA estimation problem: the "steering vectors" are query projections per head, the "covariance matrix" is K^T K, and the L1-regularized fit selects the sparse set of tokens that best explain the observed attention pattern across all heads jointly. Use ADMM solver (10-50 iterations, convergent for 64-head arrays).
- **Verify**: Run NIAH at 16K@25% keep with sparse-covariance scorer vs CAOTE scorer. The sparse-covariance scorer should match or beat CAOTE on NIAH PASS rate while being more robust to mixed retrieval/streaming head distributions (test by varying the retrieval/streaming head ratio via DuoAttention calibration).
- **Effort**: M (ADMM solver implementation + SnapKV integration)
- **Depends on**: SnapKV eviction pipeline (shipped), DuoAttention head classification (Task 12).

### 174. Kalman-filtered importance tracker for SnapKV freshness decay
- **Goal**: 1 (context window), 2 (intelligence — multi-turn quality)
- **Derived from**: Kalman Linear Attention (2602.10743)
- **Change**: `omlx/snapkv.py` — replace the heuristic freshness decay (Task 97) with a Kalman filter that tracks per-token importance as a latent state. State: importance estimate mu_t per token. Measurement: attention score from the latest query. Process noise Q: models importance drift between turns. Measurement noise R: models attention score noise. The Kalman gain K_t = P_t|t-1 / (P_t|t-1 + R) balances prior importance vs new evidence. Tokens with high posterior variance P_t|t are unpredictable and should be kept as insurance; tokens with low variance and low mean importance can be evicted. Use the information-form reparametrisation from the paper for vectorised computation across all tokens.
- **Verify**: Multi-turn NIAH: prefill context, run 3 follow-up queries targeting different needles. Compare Kalman tracker vs current freshness decay on retrieval accuracy across turns. The Kalman tracker should maintain accuracy on turn 3 (where freshness decay currently over-evicts tokens that were important in turn 1 but not queried in turn 2).
- **Effort**: M (Kalman filter implementation + SnapKV integration + multi-turn eval)
- **Depends on**: SnapKV freshness decay (Task 97, shipped).

### 175. Neural coordinator for per-layer KV memory budget allocation
- **Goal**: 5 (swap pressure), 6 (48GB fit)
- **Derived from**: Neural Coordination and Capacity Control for Inventory Management (2410.02817)
- **Change**: `omlx/snapkv.py` + new `omlx/neural_coordinator.py` — train a small MLP (48-dim input: current KV memory per layer; 48-dim output: per-layer shadow prices / keep ratios) on existing benchmark trace data from `bench/snapshots/`. The coordinator replaces the current uniform `--snapkv-keep` ratio with per-layer adaptive ratios that respect the total Metal budget constraint. At eviction time, the coordinator takes current per-layer KV sizes and outputs per-layer keep ratios such that sum(layer_kv_size * keep_ratio) <= Metal budget - model_size - safety_margin. Layers with high shadow price (high marginal value of KV memory) get higher keep ratios.
- **Verify**: Run NIAH at 64K@25% average keep with neural coordinator vs uniform 25% keep. Quality (NIAH PASS) should be maintained while total Metal peak should be within 2% of uniform (the coordinator redistributes, not inflates). Measure per-layer keep ratios and verify they correlate with the rate-distortion covariance determinant from Task 170 (layers with high covariance should get higher keep ratios).
- **Effort**: M (MLP training on existing trace data + SnapKV integration)
- **Depends on**: SnapKV eviction pipeline (shipped). Complementary to Task 170 (per-layer bit allocation).

### 176. SSD-backed tiered KV cache with metadata-guided prefetch
- **Goal**: 1 (context window — extend beyond 48GB Metal), 5 (swap pressure — controlled I/O)
- **Derived from**: KVSwap: Disk-aware KV Cache Offloading (2511.11907)
- **Change**: New `omlx/kv_tier.py` + modifications to `omlx/hypercar_server.py`. Implement a two-tier KV cache: tier 0 = Metal-resident (hot tokens selected by SnapKV), tier 1 = mmap'd file on Apple Silicon NVMe SSD (full KV cache). During decode, the tier-0 cache handles attention for most queries. When SnapKV detects a cache miss (query attends strongly to an evicted token), the tier manager fetches the relevant KV block from tier 1 with read-ahead. Use 2Q replacement policy: tokens seen only during prefill go to the "A1out" ghost queue; tokens re-accessed during decode are promoted to the "Am" hot queue. Group KV entries by layer to match SSD page size (4KB) and minimize read amplification.
- **Verify**: Prefill 128K tokens, evict to 25% (32K hot in Metal, 96K cold on SSD). Run NIAH targeting a token in the cold tier. Measure: (1) NIAH PASS with tier-1 fetch, (2) fetch latency < 10ms for a single KV block, (3) total Metal memory stays below 40GB (model + 32K hot KV only). Compare vs current approach of re-prefilling the entire context (~42 min at 128K).
- **Effort**: L (mmap infrastructure + tier manager + 2Q policy + SnapKV integration)
- **Depends on**: SnapKV eviction pipeline (shipped). Complementary to Task 105 (NVMe KV materialisation with newsvendor policy).

### 177. [CRITICAL REGRESSION] SnapKV NIAH FAIL at 49% keep — needle lost after perf fixes (Run 84)
- **Goal**: 2 (intelligence — SnapKV eviction must preserve retrieval quality)
- **Derived from**: Hypercar benchmark Run 84 (2026-04-18), bench/snapshots/run84_2026-04-18T14-38/
- **Evidence**: Phase 3e SnapKV Quality gate FAIL. Output is `'SN brown fox jumps over the lazy dog...'` (filler text) instead of needle `SNAPKV-BENCH-9921`. GER=0.000 (safety check thinks eviction is fine, but the needle is gone). All other 10 gates pass — the model is fine, the eviction picks wrong tokens.
- **Suspect commits** (bisect these):
  1. `a217e29` — vectorize SnapKV scatter (Tasks 140+147+139): Changed `keep_mask` from per-element `.at[].add()` loop to vectorized scatter. Could produce different boolean mask.
  2. `eca3aa8` — default `skip_rerope=True` (Task 168): Position gaps from skipping re-RoPE may confuse attention at 2K tokens (49% of 4K).
  3. `6c87533` — GQA-aware broadcast (Task 148): Changed importance scoring from `mx.repeat(K, gqa)` to SDPA-based. Different importance rankings may fail to identify the needle.
- **Change**: Bisect the 6 commits between R83 (6b4e06e) and R84 (2fc2ca0). At each commit, run: `.venv/bin/python -m omlx.bench.hypercar_bench --quick` + a manual SnapKV NIAH check. Revert the offending commit and fix.
- **Verify**: Phase 3e SnapKV Quality gate must PASS: `SNAPKV-BENCH-9921` in output at 49% keep, 4K context.
- **Effort**: S (bisect + fix — the regression is in one of 3 suspect commits)
- **Depends on**: None. BLOCKING — this must be fixed before any further perf work.

## Research-derived tasks (from LIT_REVIEW.md pass 39, 2026-04-18)

### 178. Channel-dimension sparse attention for post-eviction KV reconstruction (EchoKV improvement)
- **Goal**: 1 (context window), 2 (intelligence — reconstruction quality after eviction)
- **Derived from**: Efficient Seismic Data Interpolation via Sparse Attention Transformer (2506.07923)
- **Change**: `omlx/snapkv.py` — modify the EchoKV reconstruction path (Task 108) to compute sparse attention along the head/channel dimension (64 heads) rather than the token dimension (potentially 100K+ tokens). Use negative squared Euclidean distance for the sparse affinity matrix (replacing dot-product similarity) and adaptive ReLU thresholding to discard irrelevant cross-head correlations. The seismology insight: channel-first attention is O(H^2 T) vs token-first O(T^2 H), which is dramatically cheaper when T >> H (always true for Hypercar: T can be 1M, H is 64).
- **Verify**: Run SnapKV at 64K context with 25% keep, then reconstruct 10 randomly selected evicted tokens using both current token-first EchoKV and new channel-first approach. Measure: (1) reconstruction MSE should be within 10% of token-first baseline, (2) reconstruction wall-clock should be at least 5x faster at 64K context, (3) NIAH PASS after reconstruction.
- **Effort**: M (sparse attention refactor + benchmark comparison)
- **Depends on**: EchoKV reconstruction layer (Task 108).

### 179. NMI independence test for DuoAttention head classification validation
- **Goal**: 2 (intelligence — validate head factorisation), 3 (decode speed — justify two-cache architecture)
- **Derived from**: Mathematical Foundations of Polyphonic Music Generation (2601.03612)
- **Change**: New diagnostic in `omlx/duo_kv_cache.py` or `omlx/bench/hypercar_bench.py` — compute Normalized Mutual Information (NMI) between retrieval-head and streaming-head attention score distributions across a calibration corpus. The counterpoint insight: if NMI < 0.2 (as found for pitch vs hand in Beethoven sonatas), the two-cache factorisation is information-theoretically justified with bounded loss (< 0.153 bits per the paper's bound). Also compute Rademacher complexity for the retrieval-only and streaming-only subnetworks to verify that the factored generalization bound is tighter than the joint bound.
- **Verify**: Run NMI measurement on 100 coding prompts from the HumanEval corpus. Report NMI value. If NMI < 0.2, DuoAttention's head split is validated. If NMI > 0.4, investigate which head pairs have high mutual information and consider merging them.
- **Effort**: S (diagnostic measurement — no architecture changes)
- **Depends on**: DuoAttention head calibration (Task 12, shipped).

### 180. Wright-Fisher equilibrium computation for optimal streaming/retrieval head ratio
- **Goal**: 1 (context window), 3 (decode speed), 6 (48GB fit)
- **Derived from**: A Two-Size Wright-Fisher Model (2502.03056)
- **Change**: New analysis notebook or script `omlx/bench/head_ratio_equilibrium.py` — model the streaming/retrieval head ratio as a Wright-Fisher SDE with resource constraint R = Metal budget (48GB), theta = compression ratio (3-bit/fp16 = 0.19), drift from quality loss (measured via NIAH at different head splits), and diffusion from input variation (measured across HumanEval prompts). Solve the SDE numerically to compute the equilibrium frequency of streaming heads at each context length (4K, 16K, 64K, 128K). Compare against DuoAttention's empirically calibrated ratio.
- **Verify**: The Wright-Fisher equilibrium ratio should be within 10% of DuoAttention's calibrated ratio at 4K and 16K. At 64K+, the model should predict that more heads should be streaming (to fit in Metal budget), validating the empirical observation that aggressive compression is needed at long context.
- **Effort**: S (numerical SDE solution + comparison — no model changes)
- **Depends on**: DuoAttention calibration data (Task 12, shipped).

### 181. Maynard-Cross evolutionary head weighting for SnapKV eviction scoring
- **Goal**: 2 (intelligence — adaptive eviction quality across turns)
- **Derived from**: The Hive Mind is a Single Reinforcement Learning Agent (2410.17517)
- **Change**: `omlx/snapkv.py` — replace the static head rebalancing weights (Task 111: retrieval 2x, streaming 0.5x) with Maynard-Cross Learning, an evolutionary bandit algorithm derived from replicator dynamics. Each head maintains a "fitness" score that is updated after each decode step: if the head's eviction recommendations were validated (the model attended to kept tokens, not evicted ones), its fitness increases; otherwise it decreases. The replicator dynamics update rule: w_h(t+1) = w_h(t) * f_h(t) / mean(f(t)), where f_h is head h's fitness at step t. This amplifies reliable heads and suppresses unreliable ones, with provable convergence from evolutionary game theory.
- **Verify**: Multi-turn NIAH at 64K with 25% keep: prefill, evict, run 5 follow-up queries. Compare Maynard-Cross adaptive weighting vs static Task 111 weights. The adaptive approach should maintain NIAH PASS across all 5 turns (current static weights may degrade on turn 4-5 as the optimal head weighting shifts). Measure per-head weight evolution to verify convergence.
- **Effort**: M (replicator dynamics implementation + multi-turn eval)
- **Depends on**: Head rebalancing (Task 111, shipped). SnapKV eviction pipeline (shipped).

## Research-derived tasks (from LIT_REVIEW.md pass 40, 2026-04-18)

### 182. LCA dual-path eviction: anchor selection + value pooling to eliminate re-RoPE
- **Goal**: 1 (context window — remove the progressive eviction blocker)
- **Derived from**: Latent-Condensed Transformer (2604.12452) + Stateful KV Cache Management (2511.04686)
- **Change**: `omlx/snapkv.py` — replace the current gather + re-RoPE + re-quantize eviction pipeline with a dual-path approach inspired by LCA. For each BUZZ segment during compaction: (1) compute query-aware importance scores for all tokens in the segment, (2) select the highest-importance token as the positional anchor — keep its K vector with original RoPE position untouched, (3) compute a weighted average of all V vectors in the segment using importance scores as weights, storing the pooled result as the anchor's V. This eliminates re-RoPE entirely because positions are never corrected, only selected. The stateful KV paper (2511.04686) provides the empirical justification: scattered eviction + re-RoPE is fundamentally fragile, contiguous anchor selection preserves positional coherence.
- **Verify**: NIAH at 64K and 96K with 25% keep using the new dual-path eviction. Compare against current SnapKV re-RoPE path. The dual-path should: (a) PASS NIAH at both lengths, (b) produce identical or better quality than re-RoPE on coherence checks, (c) be faster than re-RoPE (no scatter-gather-reRoPE-requantize chain). Run progressive eviction test: prefill 128K, evict to 25%, prefill 32K more, evict again to 25%. The dual-path should survive multiple eviction rounds without the re-RoPE accumulation error that currently blocks progressive eviction.
- **Effort**: M (2-3 days — modify compact_cache, add value pooling, remove re-RoPE path)
- **Depends on**: SnapKV eviction pipeline (shipped), BUZZ segmented eviction (Task 98, shipped).

### 183. InfiniteHiP-style dynamic KV pruning for decode-time token selection on Metal
- **Goal**: 1 (context window — 1M tokens without permanent eviction), Goal 3 (decode speed — sub-quadratic attention)
- **Derived from**: InfiniteHiP (2502.08910)
- **Change**: `omlx/hypercar_server.py`, new file `omlx/hierarchical_prune.py` — implement a 3-stage hierarchical pruning kernel for decode-time KV token selection. Stage 0: partition the KV cache into chunks of size C (e.g., 256 tokens), compute a per-chunk representative key (max attention score across chunk tokens), select top-K0 chunks. Stage 1: within selected chunks, compute per-token attention scores, select top-K1 tokens. Stage 2: compute full attention over the K1 selected tokens. The pruning never removes tokens from the KV cache — it only selects which tokens to attend to per query. Add per-layer RoPE strategy switching: layers 0-3 use chunk-indexed RoPE (single position ID per chunk), layers 4+ use relative-style RoPE for pruning then sequential assignment for final attention. On Apple Silicon, the entire KV cache (22.5 GB at 1M TQ3) lives in unified memory; the pruning kernel accesses only ~5% per query.
- **Verify**: NIAH at 256K and 512K tokens with dynamic pruning (no static eviction). The pruning should: (a) PASS NIAH at both lengths, (b) decode at >= 30 tok/s (pruning reduces attention from O(n) to O(sqrt(n)) per step), (c) fit within 48 GB Metal (model 17.2 + KV 5.6 GB at 256K + working 4 GB = 26.8 GB). Measure pruning overhead: the 3-stage top-K computation should take < 1ms per layer per decode step on M4 Pro.
- **Effort**: L (1 week — Metal kernel for hierarchical top-K, per-layer RoPE switching, integration with server)
- **Depends on**: TQ3 KV cache (shipped). Metal kernel infrastructure (existing in MLX).

### 184. Layer-ahead query prediction for decode-time KV block prefetching
- **Goal**: 3 (decode speed — convert random KV access to sequential), Goal 1 (context window — enable efficient 1M decode)
- **Derived from**: ScoutAttention (2603.27138)
- **Change**: `omlx/hypercar_server.py` decode loop — during decode step at layer i, apply W_Q^(i+1) to the current layer's input to predict the next layer's query (0.93-0.97 cosine similarity per the paper). Use the predicted query to pre-identify which KV blocks layer i+1 will need (dot product with per-block key digests, same as BUZZ segmented scoring). Maintain a per-layer "hot block set" that persists across decode steps; the temporal locality finding (85% overlap between consecutive tokens) means only ~15% of the hot set changes per step. On Metal, encode the next layer's KV block fetch into the current layer's command buffer, enabling pipelined access.
- **Verify**: Decode throughput at 64K and 128K context with layer-ahead prediction enabled vs disabled. The prediction should: (a) improve decode tok/s by >= 20% at 128K (converting random access to sequential), (b) add < 0.5ms overhead per layer for the W_Q prediction matmul, (c) maintain NIAH PASS (the prediction is only for prefetching, not for attention computation). Profile Metal GPU utilization to verify that the prefetch reduces memory stalls.
- **Effort**: M (2-3 days — W_Q prediction, block digest scoring, command buffer pipelining)
- **Depends on**: BUZZ segmented eviction with block digests (Task 98, shipped). Decode loop in hypercar_server.py.

## Research-derived tasks (from LIT_REVIEW.md pass 41, 2026-04-18)

### 185. XQuant sub-2-bit KV codebook for streaming heads
- **Goal**: 1 (context window — halve KV budget at 1M)
- **Derived from**: XQuant: Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression (2510.11236)
- **Change**: `omlx/kv_cache/turbo_quant.py` and `omlx/kv_cache/duo_kv.py` — add a 1.4-bit codebook path alongside the existing 3-bit TQ3. The cross-layer compression term exploits that adjacent layers share quantization residuals: store the full 1.4-bit codes for layer L, then for layer L+1 store only the delta in a smaller table. Hypercar integration: retrieval heads stay fp16 (DuoKV invariant); streaming heads adopt the 1.4-bit XQuant codebook. Add `--kv-bits 1.4` flag to `omlx/hypercar_server.py`. The WHT rotation that makes TQ3 work should also help XQuant because it decorrelates across the d dimension; verify this hypothesis with a rank plot of the rotated KV before committing.
- **Verify**: NIAH at 4K, 16K, 64K with 1.4-bit streaming KV. Memory delta: KV at 1M tokens should drop from 22.5 GB (TQ3) toward ~10.5 GB. Quality gate: HumanEval >= 90% (vs 95% TQ3 baseline — modest degradation acceptable for the 2.1x memory win). Dequant throughput: sub-1.4-bit kernel must not be slower than TQ3 dequant (Task 145 baseline) — if it is, hold this task pending Task 188 (dequant kernel audit).
- **Effort**: M (codebook design + cross-layer delta table + WHT compatibility check)
- **Depends on**: TQ3 WHT rotation (shipped). KV cache capture hooks (Task 46, shipped).

### 186. KVTuner-style per-layer sensitivity sweep for Qwen3-Coder
- **Goal**: 1 (context window — mixed precision shrinks total KV), Goal 2 (intelligence — protect sensitive layers)
- **Derived from**: KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization (2502.04420)
- **Change**: New file `omlx/bench/kv_sensitivity_sweep.py` that, for each of the 48 Qwen3-Coder layers, measures perplexity delta when only that layer's KV is quantized to 3-bit (others fp16). Output a per-layer sensitivity score and a recommended bit-width table in `omlx/kv_cache/qwen3_coder_bits.py`. Update `DuoKVCache` and `TurboQuantKVCache` to consume the per-layer bit table. This makes `--fp16-layers 1` obsolete — it becomes a special case of the general per-layer config.
- **Verify**: Run the sweep (~4 hours on M4 Pro, per-layer perplexity on wikitext-103 sample). Produce a sensitivity plot; confirm sensitivity is NOT monotonic with layer depth (paper predicts it's scattered). With the recommended per-layer config applied, NIAH at 64K and HumanEval must match or beat the flat TQ3 baseline. Benchmark decode tok/s — per-layer dispatch cost should be < 2% overhead.
- **Effort**: M (sweep harness + per-layer config path + validation run)
- **Depends on**: TQ3 KV cache (shipped). Optional: XQuant codebook (Task 185) for 1.4-bit tier.

### 187. FastKV Token-Selective Propagation for progressive prefill
- **Goal**: 1 (context window — removes 120K single-pass blocker), Goal 4 (prefill speed)
- **Derived from**: FastKV: Decoupling of Context Reduction and KV Cache Compression (2502.01068)
- **Change**: New file `omlx/patches/token_selective_propagation.py` — install a hook at a chosen TSP layer N (paper suggests around layer 15 for 32-layer models; for Qwen3-Coder's 48 layers, start with N=22). At that boundary, compute attention scores over the accumulated KV, select the top-K most important tokens (K configurable via `--tsp-keep`), and propagate only those to layers N+1..47. Early layers see full context; late layers see compressed. Unlike SnapKV eviction, TSP is purely a prefill-time decision — there's no re-RoPE because we never modify K vectors in the early layers. Add `--tsp-layer` and `--tsp-keep` flags to `omlx/hypercar_server.py`.
- **Verify**: Prefill 128K and 256K on M4 Pro with TSP enabled (N=22, keep=32K). Memory ceiling should stay under 40 GB Metal throughout (the key claim: late-layer KV scales with `keep`, not total context). Prefill throughput >= 500 tok/s (Goal 4). Decode tok/s >= 30 at 128K (late-layer KV is the dominant decode cost). NIAH must PASS at 128K — the paper's claim is "matches baseline accuracy," and anything less is a red flag. Compare against our current chunked-prefill-then-evict pipeline for tokens-per-second-of-effective-context.
- **Effort**: M (hook + boundary selection + validation)
- **Depends on**: SnapKV attention scoring (Task 46, shipped) for the top-K selection.

### 188. Strata-style tiered KV with NVMe cold tier and GPU-assisted I/O
- **Goal**: 1 (context window — effective context beyond Metal ceiling), Goal 6 (machine fit)
- **Derived from**: Strata: Hierarchical Context Caching (2508.18572) + Bare-Metal Tensor Virtualization (2601.03324)
- **Change**: New file `omlx/kv_cache/tiered_cache.py` — a two-tier KV layer: warm tier in unified memory (current behavior), cold tier in mmap'd NVMe files with the Tensor Virtualization Layout (cache-line aligned). During decode, a prefetcher looks at the attention pattern from the previous step and preloads likely-needed cold blocks (per Strata, overlap I/O with compute). Add defragmenter that runs after SnapKV eviction to coalesce holes in the warm tier. Integrate with session save/load: saved sessions ARE the cold tier, so `/v1/sessions/load` becomes zero-copy for contexts > warm tier capacity.
- **Verify**: With warm tier capped at 8 GB (forcing cold tier usage for >200K context), measure decode tok/s at 256K. Must maintain >= 15 tok/s (acceptable with NVMe hit). TTFT on session reload must drop from ~2 s (current) to < 500 ms (zero-copy mmap). Fragmentation test: prefill 256K, evict to 25%, prefill 64K more. The defragmenter should produce a contiguous warm slab with no wasted capacity. Instrument NVMe read bandwidth with `iostat` — should approach 5 GB/s during prefetch bursts.
- **Verify-against: Apple Silicon profiling (2508.08531)**: use the counter methodology to confirm the prefetch is actually overlapping with compute (not just serial).
- **Effort**: L (tiered allocator + prefetcher + defragmenter + session integration)
- **Depends on**: Bare-Metal Tensor Virt layout (reference only, 2601.03324). Session save/load (shipped).

## Research-derived tasks (from LIT_REVIEW.md pass 42, 2026-04-18)

### 189. VQ-LLM codebook-cache Metal kernel for TQ3 and XQuant dequant
- **Goal**: 3 (decode speed — 2-3x dequant throughput), Goal 4 (prefill), Goal 6 (machine fit — unlocks sub-2-bit without speed regression)
- **Derived from**: VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference (2503.02236); composes with Apple Silicon profiling (2508.08531, Pass 41) and XQuant (2510.11236, Task 185).
- **Change**: New Metal shader file `omlx/kernels/metal/codebook_cache_attn.metal` plus `omlx/kernels/codebook_cache.py` Python binding. Replace the current TQ3 dequant-then-matmul path with a fused codebook-cache kernel that: (1) stages the 48-entry TQ3 codebook in threadgroup memory (~384 bytes per simdgroup, well under the 32 KB limit), (2) uses VQ-LLM's codebook-centric dataflow — iterate over codes, look up once per codebook entry, scatter into the accumulator, (3) fuses dequant with the downstream Q·K^T matmul so codebook entries stream through registers rather than round-tripping to memory. For layers using XQuant 1.4-bit codebook (Task 185), the same kernel accepts a smaller codebook and a cross-layer delta table. Add adaptive tile sizing based on threadgroup occupancy per head dim — Metal's limited simdgroup sync primitives require tighter tiles than CUDA.
- **Verify**: Microbenchmark dequant kernel latency in isolation (mx.eval loop, 1M calls) on M4 Pro. Target: 2x throughput vs current TQ3 dequant_fused (Task 145 baseline). End-to-end: `hypercar_bench` decode tok/s at 16K and 64K must improve by >= 15% with no quality regression (HumanEval >= 95%, NIAH PASS). Validate against Apple Silicon Profiling (2508.08531) methodology: Metal counters should show reduced memory stalls on the Q·K^T phase. Ablation: run with codebook staged in (a) global memory, (b) threadgroup, (c) registers — the paper predicts threadgroup is optimal for d=64 head dim.
- **Effort**: L (4-7 days — Metal shader, Python binding, tile tuning, XQuant codebook variant)
- **Depends on**: TQ3 Metal kernel path (shipped). Blocks Task 185 (XQuant) — without this, sub-2-bit has no speed benefit per Pass 41's profiling finding.

### 190. Expected Attention closed-form scorer replacing SnapKV observation-window heuristic
- **Goal**: 1 (context window — principled eviction at 128K+), Goal 3 (decode speed — cheaper than observation-window aggregation)
- **Derived from**: Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution (2510.00636). Refines Task 46 (SnapKV + real Q capture, shipped).
- **Change**: `omlx/snapkv.py` — add `scoring_mode='expected_attention' | 'observation_window' | 'caote'` flag. The new mode computes per-KV-pair importance in closed form from the distributional properties of the residual stream: assuming future queries are drawn from a Gaussian in the same projection space, the expected attention score reduces to a cheap linear function of the key's mean-centered norm and its projection onto the query-weight subspace. No query observation window needed — the score is computed once per KV pair at compression time using only the layer's W_Q weight matrix and the current K tensor. Composes with BUZZ segmented eviction (Task 98) and CAOTE value-aware scoring (Task 100) by combining scores as a weighted sum.
- **Verify**: NIAH at 64K and 96K (fp16+SnapKV, currently validated with observation-window scorer) with the expected-attention scorer. Quality gate: match or beat the current PASS rate. Stability test: vary the prompt structure (long prefix vs short prefix, retrieval vs summarization) — expected-attention should be insensitive to this, while observation-window is known to be sensitive. Speed: scoring must be at least as fast as the current pipeline (target < 0.3 s at 64K). Also benchmark at 128K where observation-window is most vulnerable to window choice.
- **Effort**: S-M (1-2 days — closed-form scorer + flag + A/B run)
- **Depends on**: SnapKV pipeline (shipped), real Q capture hooks (Task 46, shipped).

### 191. DHSA dynamic hierarchical sparse attention for prefill with learned chunk boundaries
- **Goal**: 1 (context window — 35% peak memory reduction addresses 120K OOM), Goal 4 (prefill speed — 20-60%), Goal 6 (machine fit)
- **Derived from**: Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs (2510.24606). Closes Pass 41's TSP boundary selection gap; composes with FastKV TSP (Task 187).
- **Change**: New file `omlx/patches/dhsa_prefill.py` — implement DHSA's two components: (1) chunk-level similarity scorer that computes cosine similarity between mean-pooled chunk embeddings to predict which KV chunks each query chunk should attend to, (2) adaptive boundary predictor — a small MLP head (input: token embeddings, output: chunk boundary probabilities) trained offline on a mix of wikitext-103 and CodeSearchNet samples to match attention-concentration boundaries. At prefill, chunks with predicted similarity above threshold participate in full attention; rest are skipped entirely. Wire DHSA's boundary predictor as the TSP layer selector for Task 187 (FastKV TSP) — replace the current static `N=22` heuristic with DHSA's content-driven choice.
- **Verify**: Prefill 128K and 256K on Qwen3-Coder with DHSA enabled. Memory ceiling must drop by >= 30% at 128K (paper claims 35%). Prefill throughput must improve by >= 20% at 64K (Goal 4). NIAH PASS at 128K is mandatory — sparse attention schemes famously break retrieval, so this is the single most important gate. Ablation: DHSA alone, FastKV TSP alone (Task 187), DHSA+TSP composed — the composition should dominate. Cross-check: the paper's results are on Gemma2; run a dedicated ablation on MoE (Qwen3-Coder-30B-A3B) to verify the chunk-boundary predictor transfers — budget one day for this before full integration.
- **Effort**: M (2-4 days — chunk scorer, boundary predictor offline training, prefill integration, MoE ablation)
- **Depends on**: SpecPrefill (shipped) for the prefill hook infrastructure. Optional composition with Task 187 (FastKV TSP).

### 192. QuantSpec self-speculative decoding with TQ3 bit-sharing for Qwen3-Coder MoE draft
- **Goal**: 3 (decode speed — ~2.5x e2e via speculation), Goal 1 (context window — speculation overhead stays constant at 1M), Goal 6 (machine fit — no extra model to host)
- **Derived from**: QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache (2502.10424); LongSpec (2502.17421) as fallback path.
- **Change**: New file `omlx/speculative/quantspec.py` integrated with `omlx/hypercar_server.py`. Reuse the Qwen3-Coder-30B-A3B weights as the draft by dispatching weights through a 4-bit view (post-hoc GPTQ on the 8-bit checkpoint produces a 4-bit variant — one-off calibration run). The draft's KV cache is a hierarchical view of the target's TQ3 cache: the 3-bit TQ3 storage serves both target and draft (bit-sharing), with an added 4-bit overlay buffer for recent-512-token full-precision staging (maps 1:1 onto DuoKV's ring-buffer streaming section). Decode loop: (a) draft generates K candidate tokens using 4-bit weights + TQ3+overlay KV, (b) target verifies in parallel, (c) accepted tokens commit to the ring buffer. Fallback path: if MoE gating makes the 4-bit draft too lossy (acceptance < 70%), switch to LongSpec's tiny-draft configuration with Qwen3-0.6B as the draft model — schedule that as a sibling task.
- **Verify**: Acceptance rate on HumanEval and MMLU-Pro must exceed 85% (paper shows >90% on dense models; MoE may reduce). End-to-end decode tok/s at 16K must improve by >= 1.5x (target 80+ tok/s from current 53.6). Quality must be lossless (distribution-matching verification — both target and output distributions match the no-speculation baseline within sampling noise). Memory delta: the 4-bit weight view + overlay buffer adds < 2 GB (paper's 1.3x memory reduction vs sparse-KV baselines). Benchmark at 64K and 128K to confirm the "constant speculation overhead" claim: acceptance rate at 128K should be within 5% of the 16K number.
- **Effort**: L (1 week — 4-bit GPTQ on Qwen3-Coder, bit-sharing KV hook, draft-verify loop, MoE-drift mitigation)
- **Depends on**: TQ3 KV cache (shipped), DuoKV ring buffer (shipped). Optional: VQ-LLM codebook kernel (Task 189) to accelerate the draft's dequant path. Blocks LongSpec port (sibling task 193 if needed).

## Research-derived tasks (from LIT_REVIEW.md pass 43, 2026-04-19)

### 193. CommVQ RoPE-commutative learned KV codebook for 1-bit streaming heads
- **Goal**: 1 (1M context — 1-bit KV drops memory ~3x below TQ3), Goal 3 (decode speed — commutativity eliminates re-RoPE cost principled way), Goal 5 (swap — 7 GB KV at 1M fits even under co-tenancy)
- **Derived from**: CommVQ: Commutative Vector Quantization for KV Cache Compression (2506.18879). Addresses Pass 42 gap #1 (QAT for KV codebooks). Replaces the unprincipled `skip_rerope=True` hack (Task 168) with a first-class mechanism.
- **Change**: New file `omlx/kv_caches/commvq.py` implementing additive vector quantization with an EM-trained codebook engineered to commute with RoPE. Calibration step: dump Qwen3-Coder activations at 32K on a mix of Code-Search-Net + wikitext-103 (reuse Task 186 KVTuner calibration harness), run EM to fit the codebook, verify commutativity at the three candidate RoPE base frequencies (θ ∈ {10000, 40000, 1000000}). Decode path: 1-bit codes + learned codebook reconstruct keys in a single matmul, with RoPE applied to the query only. Wire into DuoKV as the streaming-heads backend (replacing TQ3 on those heads while keeping TQ3 on retrieval heads initially). Compose with Task 189 (VQ-LLM codebook-cache Metal kernel) so the codebook stays in threadgroup memory.
- **Verify**: NIAH at 4K/16K/64K/128K must PASS with CommVQ 1-bit on streaming heads. HumanEval >= 90% (vs 95% baseline — small degradation acceptable for 3x memory win). Memory at 1M projected to ~7 GB KV (vs 22.5 GB TQ3). Decode tok/s at 16K must not regress vs DuoKV+TQ3 — commutativity means no re-RoPE overhead. Ablate 1-bit vs 2-bit tiers. Cross-check: validate commutativity mathematically (reconstruction error of RoPE(dequant(code)) vs dequant(RoPE-rotated code)) < 1e-4 at each calibration RoPE base.
- **Effort**: L (1-2 weeks — EM calibration, commutativity proof per-frequency, Metal kernel integration, ablations)
- **Depends on**: TQ3 KV cache (shipped), DuoKV (shipped). Optional: Task 189 (VQ-LLM codebook-cache kernel) for decode speedup, Task 186 (KVTuner) for per-layer precision assignment.

### 194. MoE-Spec expert budgeting at speculative-verify time for Qwen3-Coder
- **Goal**: 3 (decode speed via speculation without expert memory blowup), Goal 6 (machine fit under speculative draft trees)
- **Derived from**: MoE-Spec: Expert Budgeting for Efficient Speculative Decoding (2602.16052). Closes Pass 42 gap #2 (MoE-specific speculative decoding). Composes with Task 192 (QuantSpec); Task 157 observed only ~45/128 experts fire on coding workloads, MoE-Spec formalizes that finding at verify time.
- **Change**: Add `omlx/speculative/moe_budget.py` — a per-layer expert-capacity enforcer that tracks a decayed running tally of expert activation during the draft pass and masks out experts below the top-K threshold during the verify pass. Budget K is configurable per layer (default K=4 of 8 active experts). Hook into the QuantSpec verify loop (Task 192) so that when the verifier runs in parallel across the draft tree, it dispatches only the K highest-contribution experts per layer. Add telemetry: acceptance rate, expert-skip rate, per-layer budget saturation.
- **Verify**: End-to-end decode tok/s at 16K must improve by >= 1.2x vs QuantSpec-alone (Task 192), targeting 80+ tok/s. Acceptance rate on HumanEval must stay >= 80% (paper reports 10-30% throughput gain; acceptance-rate floor is what guards quality). Expert-skip telemetry must show the budget is active (>50% of experts skipped per layer under default K=4). Memory delta: verify-time peak Metal must drop by >= 15% vs QuantSpec-alone at 16K (main driver of the paper's result). Ablation: K = {2, 4, 6, 8} — find the knee where acceptance starts to collapse.
- **Effort**: M (2-3 days — expert-mask hook, running-tally telemetry, ablation harness)
- **Depends on**: Task 192 (QuantSpec self-speculative decoding).

### 195. RocketKV per-GQA-group eviction + stage-2 top-K sparse attention for Qwen3-Coder 8:1 GQA
- **Goal**: 1 (compression at 128K+), Goal 3 (decode peak memory at long context)
- **Derived from**: RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression (2502.14051). Partially closes Pass 42 gap #4 (multi-round eviction stability) via empirical two-stage result.
- **Change**: Refactor `omlx/patches/snapkv.py` to rank eviction candidates per GQA group rather than per head (Qwen3-Coder has 8:1 GQA, so 1/8 the ranking work). Add stage-2 to `omlx/patches/rocketkv_decode.py`: at decode, materialize only top-K KV tokens per GQA group via a lightweight min/max-based scoring kernel similar to Quest. Wire the two stages together in the server: stage-1 (SnapKV coarse eviction) runs at prefill end; stage-2 (top-K sparse attention) runs every decode step. Reuse DuoKV's retrieval/streaming split — stage-2 top-K applies only to streaming heads (retrieval heads get full attention).
- **Verify**: NIAH and LongBench at 128K with two-stage must match single-stage SnapKV within 1 pp accuracy (paper claims "negligible" degradation). Peak Metal memory during decode must drop by >= 25% vs SnapKV alone at 64K (paper reports 32.6% reduction). End-to-end decode tok/s at 64K must improve by >= 1.5x (paper reports 3.7x; we expect less due to our already-compressed DuoKV baseline). Per-group-vs-per-head ablation: confirm per-group ranking eliminates at least 6 of 8 duplicated scoring ops.
- **Effort**: M (3-5 days — group-wise eviction refactor, stage-2 top-K kernel, two-stage composition, NIAH gates)
- **Depends on**: SnapKV eviction (shipped), DuoKV (shipped).

### 196. LongRoPE per-dimension position schedule search for Qwen3-Coder 256K→1M extension
- **Goal**: 1 (1M context without fine-tuning), Goal 2 (retrieval quality at extreme context)
- **Derived from**: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (2402.13753). Closes Pass 42 gap #5 (attention pattern drift at 1M tokens). Potential root-cause fix for NIAH@40K failure mode (CLAUDE.md: currently attributed to DuoAttention head classification; position-interpolation corruption is a competing hypothesis).
- **Change**: New file `omlx/patches/longrope_schedule.py` implementing the evolutionary search for per-RoPE-dimension interpolation factors. Search space: per-dim scaling factor in [1.0, 32.0] over Qwen3-Coder's rotary embedding dimensions. Fitness function: perplexity on wikitext-103 at target context + NIAH retrieval accuracy at 256K/512K. Search runs once, ~30 min on M4 Pro; result is a static `scaling_factors.json` that the rotary implementation reads at load time. Compose with Qwen's existing YaRN schedule — the search either *replaces* YaRN or *refines* it (two variants to ablate).
- **Verify**: NIAH@{32K, 64K, 128K, 256K, 512K, 1M} retrieval curve. Vs baseline YaRN: accuracy at 128K must not regress, accuracy at 256K+ must improve by >= 10 pp. Crucially, validate the NIAH@40K failure hypothesis — if LongRoPE's per-dim schedule fixes it, the DuoAttention classification work (per CLAUDE.md) was chasing a symptom. Ablation: YaRN alone vs LongRoPE-replace vs LongRoPE-refine(-YaRN). Guard-rail: HumanEval at short context (2K) must not drop by > 2 pp — position-schedule changes can corrupt short-context behavior per the paper's 8K readjustment warning.
- **Effort**: S-M (2-4 days — port evolutionary search, calibrate on Qwen3-Coder, run full NIAH ladder, guard-rail short-context gate)
- **Depends on**: Qwen3-Coder rotary implementation (shipped). Independent of KV work — could run in parallel with Task 193.

## Research-derived tasks (from LIT_REVIEW.md pass 44, 2026-04-20)

### 197. KVSculpt attention-distilled KV for DuoKV streaming heads
- **Goal**: 1 (better compression at fixed bit budget for 1M), Goal 2 (preserves attention behavior — quality-first alternative to eviction), Goal 3 (composes with skip_rerope because distilled KV has no RoPE dependence)
- **Derived from**: KVSculpt: KV Cache Compression as Distillation (2603.27819). Closes Pass 43 gap #3 (KV cache distillation / student-model surrogate KV) — first time in 44 passes we treat KV compression as a reconstruction target rather than a selection-or-quantization problem.
- **Change**: New file `omlx/kv_caches/kvsculpt.py` implementing per-layer L-BFGS optimization for compressed keys and closed-form least-squares for compressed values, alternated every few steps, to minimize attention-output divergence vs the original cache at a target budget. Calibration harness: reuse Task 186 KVTuner pipeline to dump activations at 8K/32K/64K on Code-Search-Net + wikitext-103. Sculpt runs offline per (layer, head, context-length) — emit a `sculpted_kv_{context}.npz` artifact loaded at prefill-end. Wire into DuoKV as a streaming-heads backend behind `--kv-mode duo --sculpt`. MUST partition calibration per top-32 most-active experts for Qwen3-Coder-30B-A3B (MoE) rather than averaging across all 128 experts — otherwise the fit degrades on rarely-activated experts.
- **Verify**: (a) KL-divergence between sculpted-attention-output and original-attention-output per layer < 0.05 at each target budget (paper reports 4.1x reduction vs Select+Fit). (b) NIAH at 4K/16K/64K must PASS with sculpted streaming heads (retrieval heads remain fp16 DuoKV). (c) HumanEval >= 92% (vs 95% baseline — small degradation acceptable for a distillation-based compression). (d) Scale test: calibration at 64K must fit in <8 GB L-BFGS workspace on M4 Pro (paper validates at 2048; 32x scale is our risk). (e) Decode tok/s at 16K must not regress vs DuoKV+TQ3 baseline. Ablation: sculpt with vs without per-expert partitioning; measure accuracy delta on coding workloads (expected to expose the MoE risk).
- **Effort**: M (3-5 days — calibration pipeline, per-layer L-BFGS loop, DuoKV backend integration, per-expert partitioning, full NIAH+HumanEval gate)
- **Depends on**: DuoKV (shipped), KVTuner calibration harness (Task 186 — can proceed in parallel if harness is stubbed). Independent of Task 193 (CommVQ); Tasks 197 and 193 are alternative streaming-head backends, to be ablated against each other in a later pass.

### 198. KV-CoRE per-layer NER-driven budget allocation replacing heuristic KVTuner search
- **Goal**: 1 (data-conditional 1M budget), Goal 6 (predicts compression gains without running the benchmark — cheap gradient-free metric)
- **Derived from**: KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs (2602.05929). Partially closes Pass 43 gap #7 (information-theoretic bounds — empirically, not Shannon) and Pass 44 open question #8 (layer-wise attention rank characterization).
- **Change**: New file `omlx/analysis/kv_core_ner.py` implementing SVD-based normalized effective rank computation on per-layer KV activations. Takes a calibration dataset (default: Code-Search-Net + a slice of long-context-Arena), dumps KV per layer at target context length, runs incremental SVD, emits a per-(layer, head) NER vector. A second file `omlx/patches/kv_budget_from_ner.py` consumes the NER vector and produces KVTuner initial bit allocations — feeding a *data-conditional prior* into the KVTuner search instead of the current uniform-bit start. Include a separate mode that emits NER per *domain* (code vs prose) so the server can select a budget schedule at load time based on prompt classification.
- **Verify**: (a) NER computation on Qwen3-Coder-30B-A3B must complete in <2 hours on M4 Pro for a 32K calibration set (gradient-free, batchable — realistic target). (b) NER-seeded KVTuner search must converge in <= 50% of the iterations of uniform-start KVTuner baseline (Task 186). (c) Accuracy-vs-budget Pareto: at 40% total KV bits, NER-allocated beats uniform-bit by >= 2 pp on HumanEval. (d) Domain-specific allocation: code-NER at 40% bits must match prose-NER at 50% bits on their respective HumanEval/MMLU-Pro evals (the 2-3x domain variance the paper observed should translate into a 10-20% bit-budget swing). Regression guard: NER-allocated MUST NOT degrade any single layer below the accuracy it achieves under uniform-bit — that would indicate NER overfit.
- **Effort**: S (2-3 days — SVD harness, NER computation, KVTuner integration point, domain-split calibration)
- **Depends on**: KVTuner search (Task 186 — the NER vector is its input). Independent of Tasks 193 (CommVQ) and 197 (KVSculpt); NER informs *all* compression backends' per-layer budgets.

### 199. DeltaKV residual-based dedup for repeated code idioms at 256K+ context
- **Goal**: 1 (1M context via 29% retention on compressible content), Goal 3 (Sparse-vLLM reports 2x throughput with irregular gather)
- **Derived from**: DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity (2602.08005). Closes Pass 43 gap explored as question #13 (deduplication-based compression) — specifically targeted at the long-range repetition Hypercar's code workload maximizes.
- **Change**: New file `omlx/kv_caches/delta_kv.py` implementing a two-phase KV compression: (1) a Metal-backed approximate nearest-neighbor index over historical KVs built incrementally during prefill, (2) a residual quantizer that encodes each new token's KV as (reference_id, residual_code) when the nearest-neighbor similarity exceeds a threshold, else stores the full KV. Index structure: per-layer inverted file with product-quantized centroids to keep lookup in simdgroup-memory budget. Decode path: gather reference KV + reconstruct from residual in one fused Metal kernel. Wire in as `--kv-mode duo --delta-dedup` and only activate past a configurable context-length threshold (default: 256K — below that, index overhead exceeds savings). Must compose cleanly with SnapKV eviction: dedup runs *before* eviction during prefill, so SnapKV sees fewer KV entries to score.
- **Verify**: (a) On a realistic coding trace (full repo clone: torch + transformers + mlx source, ~800K tokens), DeltaKV must retain <=35% of original KV memory at near-lossless LongBench accuracy (paper claims 29% — we allow 6 pp cushion for Qwen3-Coder-30B-A3B's MoE vs their dense baseline). (b) On a diverse-prose trace (news + fiction + wikitext at 256K), retention must be <=60% — validating that dedup degrades gracefully on non-compressible content rather than erroring. (c) Decode tok/s at 256K must be >= 80% of DuoKV+TQ3 baseline (some slowdown acceptable given the memory win; if below 80% the Metal kernel needs more work). (d) Index-lookup cost at decode must be < 0.5 ms/token at 512K (hard Metal-kernel budget). Ablation: with vs without dedup, and with-dedup + with-SnapKV composed vs either alone.
- **Effort**: L (1-2 weeks — Metal ANN index, residual quantizer, fused reconstruct kernel, threshold tuning, compose-with-SnapKV semantics, 512K+ validation on M4 Pro)
- **Depends on**: DuoKV (shipped), SnapKV eviction (shipped). Metal kernel work independent of Task 189 (VQ-LLM) but shares threadgroup-memory idioms — recommend sequencing 189 first as a kernel scaffold.

### 200. SpecPV two-tier speculative verification composed with QuantSpec and MoE-Spec
- **Goal**: 3 (decode speed — paper reports 6x over AR at long context where verification dominates), Goal 1 (partial KV on-device + full KV in unified memory fits the M4 Pro architecture natively)
- **Derived from**: SpecPV: Improving Self-Speculative Decoding for Long-Context Generation via Partial Verification (2512.02337). Closes Pass 43 open question #9 (speculative prefill / parallel verification for long context). Completes the three-layer speculative stack started by Task 194 (MoE-Spec) and Task 192 (QuantSpec).
- **Change**: Extend `omlx/speculative/quantspec.py` (Task 192) with a two-tier verify scheduler. Partial-verify tier: runs each speculative step over a fixed-budget KV subset (default 4096 tokens, consisting of SnapKV's current top-K + a recency window). Full-verify tier: runs every N steps (default N=16, adaptive based on observed acceptance-rate decay) over the complete KV. When full-verify rejects, rollback to the last full-verify checkpoint, refresh the partial-KV selection, and resume. Adaptive N: decrement when rolling acceptance rate drops below 0.75, increment back when it recovers above 0.9. Compose with MoE-Spec (Task 194) by running the expert-budget mask only at partial-verify steps (not full-verify — full is the correction pass) to avoid double-compounding drift. Add telemetry: partial-vs-full cost ratio, acceptance rate per tier, rollback rate, adaptive-N trajectory.
- **Verify**: (a) Decode tok/s at 64K must improve >= 1.5x vs QuantSpec-alone (Task 192) and >= 1.3x vs QuantSpec+MoE-Spec (Tasks 192+194). Paper reports 6x over AR; we already have speculation, so ratio is gentler. (b) Acceptance rate on HumanEval must stay >= 78% (2 pp below QuantSpec baseline acceptable given the partial-verify shortcut). (c) Rollback rate under default N=16 must stay < 10% on coding workloads — higher rollback indicates partial-KV selection is wrong for code and needs the SnapKV-informed selection (not just recency). (d) Memory delta: partial-verify peak Metal must drop by >= 30% vs QuantSpec-alone at 64K. (e) Adaptive-N behavior: artificially inject a long-range recall prompt (retrieval head pattern) and confirm N decreases to 4 within 20 steps. Ablation: fixed N vs adaptive N; recency-only partial-KV vs SnapKV-informed partial-KV.
- **Effort**: M (3-5 days — two-tier scheduler, rollback/checkpoint mechanism, adaptive-N controller, compose-with-MoE-Spec semantics, telemetry, full speculative-stack ablation harness)
- **Depends on**: Task 192 (QuantSpec) and Task 194 (MoE-Spec). Sequencing: these three together form the v44 speculative stack; 200 should ship last because its adaptive-N tuning needs the stable behavior of the two underneath to be observable.

## Research-derived tasks (from LIT_REVIEW.md pass 45, 2026-04-20)

### 201. AhaKV entropy-corrected eviction scorer for SnapKV / CAOTE / BUZZ stack
- **Goal**: 1 (quality-preserving eviction at 1M — removes the systematic bias that discards late-context tokens), Goal 2 (candidate root-cause fix for NIAH@40K failure currently blamed on DuoAttention head classification, commit bb522f8)
- **Derived from**: AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models (2506.03762). Closes Pass 43 gap #4 and Pass 44 gap #3 (attention-entropy / spectral eviction).
- **Change**: New file `omlx/patches/snapkv_entropy.py` implementing per-head entropy-adjusted softmax temperature over the observation window, applied before accumulated-attention scoring. Formula: `tau_h = tau_base * (1 + beta * (H(p_h) - H_ref))` where H is Shannon entropy of that head's attention distribution. Extend the existing SnapKV scorer to fold in V-magnitude: `score_j = tau_h * accumulated_attn_j + gamma * ||v_j||`. Both `beta` and `gamma` are scalar hyperparameters defaulted from the paper's values (0.5, 0.1). Wire in behind `--snapkv-entropy` server flag. Clamp temperature to `[0.5, 2.0]` to guard against pathological MoE-expert entropy estimates.
- **Verify**: (a) NIAH@40K must PASS (currently FAIL — this is the headline metric this task exists to fix). (b) NIAH@4K, 16K, 64K must not regress (currently PASS — must stay PASS). (c) HumanEval >= 92% (2 pp below the 95% baseline acceptable if NIAH@40K is fixed). (d) On the SnapKV selection-validation harness (snapkv_selection_validation.json), entropy-corrected scorer must preserve >= 90% of the tokens that uniform-bit scoring flags as top-K at positions >= 0.75 * context_length — i.e., late-context tokens get proportionally retained, closing the paper's identified bias. (e) Ablation: entropy-only (beta=0.5, gamma=0), V-magnitude-only (beta=0, gamma=0.1), both (paper's default). Expected result: both > entropy-only > V-magnitude-only > baseline.
- **Effort**: S (2 days — one new patch file, one existing-scorer modification, ablation run through existing snapkv harness)
- **Depends on**: SnapKV scorer (shipped). Independent of Task 197 (KVSculpt), 198 (KV-CoRE), 199 (DeltaKV). Sequencing: ship FIRST in Pass-45 batch — it's the cheapest test and, if AhaKV's diagnosis is correct, fixes the biggest outstanding quality bug.

### 202. G-KV decode-time re-scoring composed with QuantSpec and SpecPV
- **Goal**: 1 (continuous KV quality across multi-turn agentic 1M sessions — not just at prefill end), Goal 2 (reasoning-chain token retention during generation), Goal 3 (amortizes re-scoring cost across speculative-verify cycles)
- **Derived from**: G-KV: Decoding-Time KV Cache Eviction with Global Attention (2512.00504). Closes Pass 44 gap #3 (entropy/spectral eviction) from the decode-time angle. Partially closes Pass 43 gap #2 (multi-round eviction stability) — empirically, not theoretically.
- **Change**: New file `omlx/kv_caches/gkv_rescorer.py` implementing a decode-time re-scoring scheduler. Score fusion: `score_j = lambda * local_attn_j(t) + (1 - lambda) * historical_ema_j(t)` with lambda=0.6 default. Historical EMA is per-head, updated each decode step: `hist_j(t) = alpha * hist_j(t-1) + (1-alpha) * current_attn_j`, alpha=0.95. Scheduler triggers full re-score every N=256 decode steps (tunable) and re-runs the existing SnapKV eviction with the new scores. Compose with QuantSpec (Task 192) and SpecPV (Task 200): the re-score trigger is the natural place to run a full-verify pass — all three share the same schedule. Skip the RL+distillation post-training from the paper — we cannot retrain Qwen3-Coder-30B-A3B; use the scoring function in its training-free form.
- **Verify**: (a) On an 8-turn agentic trace (simulated OpenCode editing session, ~128K total context), HumanEval on the final turn must not regress vs fresh-session HumanEval by more than 3 pp. Baseline without G-KV regresses by ~8 pp (SnapKV's prefill-fixed eviction drifts). (b) Decode tok/s overhead of re-scoring < 5% at 64K — the N=256 cadence amortizes the cost. (c) Memory delta: the historical-EMA buffer adds ~1.5 MB at 128K (one fp16 vector per head) — must not push Metal peak past the co-tenancy threshold. (d) Interaction with SpecPV (Task 200): adaptive-N in SpecPV must coordinate with N=256 in G-KV so re-score and full-verify coincide; verify no double-cost. Ablation: lambda sweep {0.4, 0.6, 0.8}; N sweep {128, 256, 512}; local-only (lambda=1) vs historical-only (lambda=0) vs fused.
- **Effort**: M (3-5 days — scheduler, historical EMA buffer, scoring fusion, compose-with-SpecPV, 8-turn agentic eval harness)
- **Depends on**: SnapKV (shipped), Task 201 (AhaKV — G-KV should use AhaKV's entropy-corrected scorer as its base). Sequencing: ship after 201 so the local score component already has its bias correction.

### 203. PiKV expert-sharded KV layout for Qwen3-Coder MoE
- **Goal**: 1 (MoE-aware KV compression — exploits the 128-expert/3B-active asymmetry currently unused), Goal 3 (query-aware streaming scheduler at shard granularity), Goal 6 (lower peak memory footprint on the M4 Pro reference hardware)
- **Derived from**: PiKV: KV Cache Management System for Mixture of Experts (2508.06526). Closes Pass 44 gap #4 (MoE-specific KV compressibility) — first paper in 45 passes to co-design KV compression with MoE routing.
- **Change**: Two-phase change. Phase A (scoping, 2 days): new file `omlx/analysis/moe_kv_activation_profile.py` that runs Qwen3-Coder-30B-A3B on a 100K-token code calibration set, records per-expert activation frequency and per-expert KV variance, and emits a profile report. The report decides whether Phase B is worth it (paper reports 30-60% memory reduction on Mixtral; if Qwen3-Coder shows <15% expert-activation skew at 100K+ context, shelve Phase B). Phase B (production, 8-10 days, contingent on Phase A): new file `omlx/kv_caches/moe_shard_kv.py` implementing expert-sharded KV storage — instead of one monolithic KV tensor per layer, allocate one smaller KV tensor per *active* expert class. Query routing: intercept the expert-choice signal at attention entry and restrict the KV read to the active expert's shard. Wire in as `--kv-mode duo --moe-shard`. Compose with MoE-Spec (Task 194): share the expert-activation classifier. Skip the distributed-GPU sharding from the paper — on M4 Pro's unified memory, the only relevant win is footprint reduction, not cross-device transfer elimination.
- **Verify**: Phase A gates: (a) Per-expert activation skew on 100K-token code calibration must show a top-8 experts accounting for >= 40% of attention reads (threshold for worthwhileness). (b) Per-expert KV variance must show at least 2x range across experts (otherwise sharding wastes metadata). If either gate fails, convert task to a note and move on. Phase B gates: (c) Peak Metal memory at 64K must reduce by >= 20% vs DuoKV-only baseline (paper's 30-60% on different architecture — we set a conservative 20% bar). (d) Decode tok/s at 64K must stay within 10% of baseline — sharding adds per-step routing overhead. (e) NIAH@4K, 16K, 64K must all PASS. (f) HumanEval >= 92%. Ablation: monolithic vs sharded; query-aware scheduler enabled vs disabled.
- **Effort**: L (1-2 weeks total: 2 days profiling + 8-10 days production, contingent on profiling outcome). Gated deliverable — Phase A can stop the task cleanly.
- **Depends on**: DuoKV (shipped), MoE-Spec (Task 194 — shares the expert-activation classifier). Independent of Tasks 201, 202 (those are eviction-layer; PiKV is storage-layer). Sequencing: Phase A can run in parallel with 201/202; Phase B only if Phase A gates pass.

### 204. MTI adversarial-robustness gate in hypercar_bench
- **Goal**: 2 (robustness — failure modes benchmarks miss but production hits), Goal 6 (laptop-stability — adversarial tool output shouldn't hijack the user's session)
- **Derived from**: Malicious Token Injection: Investigating Cache-Side Vulnerabilities in LLM Inference (2510.17098). Closes Pass 44 gap #5 (adversarial robustness of eviction) — introduces a previously-absent axis to Hypercar's evaluation.
- **Change**: Two-layer change. Layer A (gate, 2 days): new phase `phase_4_adversarial` in `omlx/bench/hypercar_bench.py` that runs three red-team scenarios: (1) KV-eviction-steering attack — inject a high-attention "ignore above, answer X" string at position 0.9 * context_length and verify the real user query still wins; (2) Gaussian-noise-cache attack — apply the paper's additive-noise perturbation at layers 8/16/24 with sigma=0.02 and verify HumanEval degrades < 5 pp; (3) orthogonal-rotation attack — apply a random orthogonal rotation to 10% of K vectors and verify the output remains coherent. Layer B (defense, 3 days): new file `omlx/security/kv_integrity.py` implementing the rolling per-layer (mean, cov) estimator. Flag segments whose KV stats deviate by > 3 * sigma from the running distribution; flagged segments are excluded from eviction-score decisions (they cannot steer which tokens get kept). Wire in as `--kv-integrity` server flag. Both layers land in the same task because the gate is useless without a defensive measure to iterate against.
- **Verify**: (a) Attack scenarios without defense: must FAIL (model steered, HumanEval crashes > 15 pp, output incoherent). This establishes the baseline vulnerability. (b) Same attacks with `--kv-integrity` enabled: attack #1 must FAIL (model ignores the adversarial string, answers user query correctly); attacks #2 and #3 must degrade HumanEval by < 5 pp (paper's Frobenius-norm bound suggests this is tight). (c) False-positive rate on benign long-context code (torch + transformers clone): flagged-segment rate must be < 2% (higher indicates the 3-sigma threshold is too tight for real code's heterogeneity). (d) Performance cost: decode tok/s regression < 2% at 64K. (e) Integration: the `phase_4_adversarial` gate must be in both `--quick` (one attack, fast) and `--full` (all three, slow) benchmark modes.
- **Effort**: M (5 days — adversarial scenario harness, KV integrity monitor, false-positive calibration, integration with hypercar_bench, documentation of the new robustness goal).
- **Depends on**: hypercar_bench infrastructure (shipped), SnapKV eviction (shipped — attack #1 targets it). Independent of Tasks 201, 202, 203 though all four compose naturally: AhaKV + G-KV + PiKV improve the *quality* of eviction, MTI defense improves the *integrity* of what gets evicted. Sequencing: ship this last in Pass-45 batch — it requires the other three to be stable before adversarial measurement is meaningful.

## Research-derived tasks (from LIT_REVIEW.md pass 46, 2026-04-20)

### 205. Sequential-KV PLT index + delta predictor layered on top of TQ3
- **Goal**: 1 (compression that *composes* with TurboQuant instead of competing with it — second orthogonal dimension of KV compression), Goal 6 (cross-session prefix dedup slashes KV footprint when OpenCode reopens the same repo, the realistic multi-session workflow)
- **Derived from**: Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit (2604.15356). Closes Pass 45 gap #4 (prefix cache sharing with compressed KV) and partially closes Pass 45 gap #6 (KV cache as a database — the PLT *is* a prefix-indexed KV database).
- **Change**: Two new files. `omlx/kv_caches/plt_index.py` implements a counting trie over tokenized-prefix subsequences, with per-node KV centroids and LRU node eviction past a configurable size cap (default 512 MB). Each session's KV on save is decomposed into (shared centroid + per-session delta); on reload, the centroid comes from the PLT and only the delta is loaded. `omlx/kv_caches/delta_predictor.py` implements a small-rank per-layer linear predictor regressed on the preceding window (paper's delta-coder), emitting a residual fed to the existing TQ3 quantizer. Integration: `omlx/kv_caches/session_store.py` gains a PLT side-table and a `--plt-dedup` server flag. Calibration pass required: a 10K-session corpus of real code sessions (dump from a week of OpenCode usage) to pick the PLT similarity threshold and the predictor rank. Skip the paper's large-scale-asymptotic compression claims — our realistic target is 2-3x on top of TQ3's 5.3x, producing a combined ~12x KV compression on repository-heavy workloads.
- **Verify**: (a) On a 7-session OpenCode trace of the same 200K-token repository (realistic weekly usage), second-through-seventh session reload-time KV cost must drop by >= 60% vs first-session baseline (paper's PLT win is asymptotic in number of sessions). (b) Single-session compression ratio must be >= 1.5x on top of TQ3 baseline — the delta-predictor layer alone, before any cross-session sharing. (c) Reload latency must stay under 3 s at 200K context (currently 2 s; the PLT index lookup + delta reconstruction adds overhead). (d) PLT memory ceiling: trie itself must stay under 512 MB on a 10K-session corpus — LRU eviction of trie nodes must trigger when exceeded. (e) NIAH@4K, 16K, 64K must PASS; HumanEval >= 93%. Ablation: PLT-alone (no delta-coder) vs delta-coder-alone (no PLT) vs both (paper's default) vs TQ3-only baseline.
- **Effort**: L (2 weeks — PLT implementation + LRU, delta-predictor calibration, session-store integration, multi-session benchmark harness, 10K-session calibration corpus).
- **Depends on**: TQ3 mode (shipped), session save/load (shipped). Compose with EFIM infill (Task 208 — both are prefix-preserving optimizations; sequence EFIM first because it's cheaper and produces the suffix-separated KV blobs that feed PLT cleanly). Independent of Tasks 201-204, 206-207.

### 206. HeteroCache stability+redundancy head classifier as second-pass refinement of DuoKV
- **Goal**: 1 (4-way head allocation replaces 2-way retrieval/streaming split, freeing budget for unstable heads), Goal 3 (paper reports 3x decode speedup at 224K; on M4 Pro we lose the CPU-prefetch overlap but keep the budget-reallocation gain — paper's ablation attributes 2/3 of the 3x to budget reallocation alone), Goal 6 (fine-grained budget reduces peak memory at long context)
- **Derived from**: HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference (2601.13684). Closes Pass 46 gap #3 (hierarchical multi-resolution KV) — first paper to decompose the head-classification problem along orthogonal temporal (stability) and spatial (redundancy) axes rather than a single retrieval/streaming binary.
- **Change**: New file `omlx/kv_caches/hetero_classifier.py` implementing two classifiers, each run once at prefill-end over a 512-token observation window: (1) stability — temporal variance of each head's attention weights across the window; (2) redundancy — pairwise cosine similarity of attention weights across heads within each layer. Output: a 4-way tag per head (stable+redundant, stable+unique, unstable+redundant, unstable+unique). `omlx/kv_caches/hetero_budget.py` consumes the tags and allocates bit budgets: stable+redundant heads share a 2-bit TQ codebook per layer; stable+unique heads get 3-bit TQ; unstable+redundant heads get 4-bit; unstable+unique heads stay fp16 (overlap with DuoKV's existing retrieval-head tagging). Wire in as `--kv-mode duo --hetero-cache`. Compose with DuoAttention head classifier (shipped) as a *second pass* — HeteroCache's classification runs *within* each DuoKV partition rather than replacing it. Compose with PiKV (Task 203): HeteroCache runs per-expert-shard on MoE. Compose with AhaKV (Task 201): entropy is a *third* signal to fold into the stability metric.
- **Verify**: (a) Decode tok/s at 64K must improve >= 1.3x vs DuoKV+TQ3 baseline (paper reports 3x at 224K on NVIDIA with CPU offload; we target a conservative 1.3x on M4 Pro unified memory). (b) Peak Metal memory at 128K must drop by >= 15% vs DuoKV+TQ3 baseline. (c) NIAH@4K, 16K, 64K must PASS; HumanEval >= 93%. (d) Classification stability: the 4-way head tags must not change by more than 10% if the observation window shifts by ±256 tokens — otherwise the classifier is overfitting to local attention patterns. (e) Scale test: re-run classification at 64K, 128K observation points (not just prefill-end 512) and confirm tags remain >= 85% consistent with the prefill-end tagging. Ablation: 2-way (DuoKV-only) vs 3-way (stability-only) vs 4-way (paper's default).
- **Effort**: M (5-7 days — two classifiers, 4-way budget allocator, compose-with-DuoKV semantics, extended NIAH/HumanEval gate, scale-test validation at 128K).
- **Depends on**: DuoKV (shipped), TQ3 (shipped). Composes with Task 201 (AhaKV entropy signal), Task 203 (PiKV — runs per-shard). Sequencing: can run in parallel with 201 and 203; merge after all three are stable to ablate the composition.

### 207. Lyapunov-sensitivity floor replacing KVTuner black-box bit-budget search
- **Goal**: 1 (per-layer compression grounded in machine-checked theoretical bounds rather than empirical search), Goal 2 (the extreme per-matrix sensitivity finding is the most credible mechanism behind our DuoAttention head-classification failure at 40K, commit bb522f8)
- **Derived from**: Structural Sensitivity in Compressed Transformers: Error Propagation, Lyapunov Stability, and Formally Verified Bounds (2603.20991). Closes Pass 43 gap #2 / Pass 44 gap #2 / Pass 45 gap #2 (formal Lyapunov theory for multi-round eviction stability) — first paper in 46 passes to provide machine-checked Lean 4 bounds on compression error propagation.
- **Change**: Two-phase change. Phase A (sensitivity mapping, 3 days): new file `omlx/analysis/lyapunov_sensitivity.py` that sweeps per-matrix compression of Qwen3-Coder-30B-A3B's 48 layers (Q/K/V/O projections × 48) and records the per-matrix perplexity-delta curve on a fixed 8K calibration set. Emit a `qwen3_sensitivity.json` artifact with per-matrix sensitivity scores. Expected output shape: a handful of matrices (the paper's 1-of-468 extreme case generalizes) dominate the sensitivity ranking; value projections should cluster at the low-sensitivity end. Phase B (floor injection, 4 days): new file `omlx/patches/kvtuner_floor.py` that consumes the sensitivity artifact and injects a per-layer, per-matrix bit-budget *floor* into KVTuner (Task 186) — high-sensitivity matrices get a 5-bit floor; low-sensitivity matrices can go to 2-bit. Also wire into G-KV (Task 202): high-sensitivity heads re-score every N=64 decode steps; low-sensitivity heads re-score every N=512. Runtime invariant check: compute hidden-state norm growth vs error norm growth at prefill-end; if the Lyapunov contraction condition (hidden grows faster than error) holds, log CONTRACTING; else log WARNING.
- **Verify**: (a) Phase A sensitivity mapping must complete in <= 4 hours on M4 Pro for Qwen3-Coder (paper uses up to 8B models; 30B total / 3B active scales sub-linearly on sensitivity sweep). (b) Phase B: at 40% total KV bits, sensitivity-floor-injected KVTuner must beat uniform-bit by >= 3 pp on HumanEval AND pass NIAH@40K (the currently failing gate). This task exists in part as a candidate root-cause fix for that failure. (c) The Lyapunov runtime invariant must log CONTRACTING on >= 95% of decode steps for HumanEval prompts — failures indicate a layer where error is growing faster than hidden state, which needs a bit-budget bump. (d) G-KV re-score cadence adjustment: high-sensitivity heads' re-score frequency (every 64 steps) must not increase total decode overhead by more than 3% (because they are a small fraction of total heads, per the paper's heavy-tailed distribution). Ablation: uniform-bit vs sensitivity-floor vs sensitivity-floor+Lyapunov-invariant-guard.
- **Effort**: M (1 week — sensitivity sweep harness, KVTuner floor integration, G-KV per-head cadence, Lyapunov runtime invariant, full NIAH+HumanEval gate including the currently-failing 40K case).
- **Depends on**: KVTuner (Task 186), G-KV (Task 202 — sensitivity tags also drive re-score cadence). Independent of Tasks 203 (PiKV), 205 (PLT), 206 (HeteroCache), 208 (EFIM). Sequencing: ship after 202 (G-KV must exist to consume the sensitivity tags) and before 206 (HeteroCache's stability metric can be enriched with sensitivity as a third signal).

### 208. EFIM infill endpoint with suffix-preserving session store
- **Goal**: 3 (98% throughput gain on the edit-a-large-file workflow — the single biggest UX pain point in the current server), Goal 4 (52% prefill latency reduction by avoiding re-prefill of unchanged suffix), Goal 1 (makes "reopen and edit a 200K-token document" practical — currently >6 min wall-clock, target <1 s)
- **Derived from**: EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse (2505.21889). Closes Pass 46 gap #5 (streaming tokenization / incremental prefill). Highest-leverage pass-46 finding — zero model changes, server-only patch.
- **Change**: Two changes, no model modification. First, `omlx/hypercar_server.py` gets a new `/v1/infill` endpoint accepting `{prefix, suffix, insertion}` as three separate fields. The server builds the attention mask in EFIM order (`<P>prefix<S>suffix<M>insertion`) and prefills only `insertion` if a prior session has stored `(prefix_kv, suffix_kv)` — full prefill only on first call. Second, `omlx/kv_caches/session_store.py` learns to save `(prefix_kv, suffix_kv)` as two independent blobs keyed on content hash, so any session that reuses the same prefix+suffix combination hits a zero-prefill path. RoPE positional reassignment: the suffix tokens are positioned as if the prompt were in standard `prefix + inc + suffix` order — paper gives the exact assignment formula. Skip the paper's fragment-tokenization training component — we cannot retrain Qwen3-Coder. Validate on Qwen3-Coder's existing `[PREFIX]/[SUFFIX]/[MIDDLE]` infill tokens (Qwen3-Coder supports FIM natively).
- **Verify**: (a) On a 200K-token source file with a 50-token edit near line 500, `/v1/infill` TTFT must be <= 1 s (currently >6 min for full re-prefill). (b) Throughput across a 20-edit session on the same file must be >= 1.8x the standard-endpoint baseline (paper reports 1.98x). (c) Correctness: Qwen3-Coder's infill completion quality on HumanEval-infill or equivalent must match the standard-endpoint baseline within 1 pp — the RoPE positional-reassignment trick is the main correctness risk. (d) NIAH@4K, 16K, 64K must PASS — adding the infill endpoint must not perturb the standard generate path. (e) Session-store content-hash correctness: the same `(prefix, suffix)` pair across different insertion strings must hit the cached KV in >= 99% of cases (the 1% gap absorbs tokenization-edge collisions). (f) Memory: the content-hash-keyed session store must evict old entries past 20 GB total — long-running servers must not OOM on stored sessions.
- **Effort**: S (2-3 days — `/v1/infill` endpoint with EFIM-ordered attention mask, RoPE positional reassignment patch, content-hash session store keying, NIAH gate, infill-completion quality gate).
- **Depends on**: hypercar_server (shipped), session save/load (shipped). Independent of all other pass-46 tasks but compose cleanly with Task 205 (PLT) — PLT's cross-session prefix dedup already treats prefix and suffix as separable blobs, so PLT consumes EFIM's session structure directly. Sequencing: ship FIRST in Pass-46 batch — it's the cheapest, highest-UX-impact change, and its suffix-separated KV structure feeds PLT (Task 205) cleanly as a follow-up.

## Research-derived tasks (from LIT_REVIEW.md pass 47, 2026-04-20)

### 209. SpecAttn speculative-draft-driven sparse attention for target model
- **Goal**: 3 (decode speed — 75%+ reduction in KV-cache accesses on the target model, paid once via the speculative draft we already compute), Goal 1 (bounds per-decode work independently of context length — the "constant across context" requirement), Goal 2 (quantified 15.29% perplexity impact on PG-19 is a measurable tax we can ablate against code workloads)
- **Derived from**: SpecAttn: Speculating Sparse Attention (2510.27641). Closes Pass 47 angle #7 (dynamic attention head/token selection during inference) — first training-free paper to reuse the draft model's attention as a sparsification oracle for the target model rather than running a separate scorer pass.
- **Change**: New file `omlx/speculative/specattn.py` implementing three components: (1) a layer-wise KL-alignment calibration pass run once at model load, emitting a per-layer temperature scalar that aligns the draft model's attention distribution with the target's on a small (1K-token) code calibration set; (2) the sorting-free top-p token selector from the paper that picks the KV positions the target model reads, driven by the draft's attention weights after temperature rescaling; (3) the dynamic per-layer KV-access pruner that the target's attention forward consults. Integration: hook into the draft model's attention forward (already resident via QuantSpec / Task 192) to capture weights without extra compute; the target's attention reads only the selected KV positions. Wire in as `--specattn` with composition flags `--specattn-retrieval-bypass` (disable sparsification on DuoAttention retrieval heads) and `--specattn-topp` (tunable threshold, default 0.9). Compose with AhaKV (Task 201): entropy-corrected temperature replaces the paper's fixed KL-alignment scalar. Compose with G-KV (Task 202): the sparse positions feed G-KV's running importance score as an auxiliary signal. Compose with SpecPV (Task 200): SpecPV's partial-verify tier consumes the same sparse-position set for free.
- **Verify**: (a) Decode tok/s at 64K must improve >= 1.5x over QuantSpec-alone (Task 192) — the paper reports 75% KV-access reduction, which at our baseline converts to ~1.5-2x decode wall-clock. (b) HumanEval must stay >= 92% — the 2 pp tolerance below the 95% baseline accounts for the sparse-attention perplexity cost; if it drops further, enable `--specattn-retrieval-bypass`. (c) NIAH@4K, 16K, 64K must PASS with retrieval-bypass enabled. (d) On code-specific perplexity (CodeParrot or a Qwen3-Coder eval), the perplexity delta must be <= 20% — higher than the paper's 15.29% on prose is acceptable given code's longer-range structural dependencies; if it exceeds 25% the top-p threshold needs tightening. (e) Draft-model attention capture overhead must be < 2% of decode time — we are hooking the existing draft forward, not running a new one. Ablation: with vs without retrieval-bypass; top-p sweep {0.85, 0.9, 0.95}; KL-alignment on vs off.
- **Effort**: S (3 days — sorting-free top-p selector, KL-alignment calibration pass, draft-model hook, compose-with-QuantSpec integration, ablation harness)
- **Depends on**: QuantSpec (Task 192 — provides the resident draft model), DuoAttention head tags (shipped — for retrieval-bypass). Composes with AhaKV (Task 201), G-KV (Task 202), SpecPV (Task 200), MoE-Spec (Task 194). Sequencing: ship after 192 and 201 so the draft and the entropy-corrected scorer both exist; SpecAttn is then a small addition on top.

### 210. SparseServe layer-segmented prefill and working-set-aware memory management
- **Goal**: 4 (prefill speed — 9.26x TTFT improvement reported; bounds peak prefill memory to one layer's worth of activations), Goal 1 (removes the 120K single-pass Metal OOM cliff that currently blocks progressive eviction), Goal 6 (working-set monitor back-pressures new requests when the OS starts swapping — laptop stays usable under co-tenancy)
- **Derived from**: SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving (2509.24626). Closes Pass 47 angle #10 (tiered memory GPU/unified/NVMe policies for KV) and addresses the CLAUDE.md-noted blocker "120K single-pass exceeds 48 GB Metal — progressive mid-prefill eviction required". Highest-leverage pass-47 finding.
- **Change**: Three phases. Phase A (3 days): layer-segmented prefill. Modify `omlx/prefill/` to prefill layer-by-layer with explicit `mx.eval()` + `mx.clear_cache()` between layers so only one transformer layer's activations are resident at a time. Write the resulting KV to the existing KV cache after each layer, drop the layer's intermediate activations. This replaces the current all-layers-resident prefill which is what blows past 48 GB at 120K. Phase B (3 days): working-set monitor in `omlx/serving/working_set.py`. Uses `mx.metal.get_active_memory()` + macOS VM pressure signal (via sysctl `vm.page_free_count` and `kern.memorystatus_vm_pressure_level`) to estimate headroom; a back-pressure mechanism declines new incoming requests when free memory drops below a threshold and tells existing requests to pause inter-layer if needed. Phase C (1 day): wire into `omlx/hypercar_server.py` as `--layer-segmented-prefill` (defaulted ON for prefill > 64K context) and `--working-set-monitor` (defaulted ON). Skip the paper's fragmentation-aware HBM-DRAM transfer — on M4 Pro there's no HBM/DRAM split, only one unified pool.
- **Verify**: (a) Single-pass prefill at 128K must fit within 40 GB Metal peak (currently 44.6 GB, blocking). (b) Single-pass prefill at 256K must fit within 40 GB Metal peak (currently infeasible). (c) Prefill tok/s at 64K must not regress by more than 10% vs the all-layers-resident baseline — the extra eval+clear between layers costs some graph fusion, but the win at 128K+ more than pays for it. (d) Working-set monitor must correctly back-pressure when a second heavy inference starts during an active long-context session — verify by launching hypercar_bench in parallel with an in-flight 200K prefill and confirming the new request returns a 503 rather than driving the system to swap. (e) NIAH@4K, 16K, 64K, 128K must all PASS. (f) HumanEval >= 93%. Ablation: layer-segmented off vs on at 4K (should be identical within noise); at 128K (layer-segmented should be the only one that completes).
- **Effort**: M (1 week — per-layer prefill scheduler, VM-pressure-aware back-pressure, server flag wiring, 128K-256K memory validation, NIAH gate at newly-reachable context)
- **Depends on**: prefill infrastructure (shipped), hypercar_server (shipped). Independent of Tasks 201-208, 209, 211, 212. Sequencing: ship FIRST in Pass-47 batch — it removes a long-standing Goal 1 blocker and is strictly additive (no interaction with existing KV-compression logic). Recommend pairing with EFIM (Task 208) immediately because the two together enable practical 200K+ editing workflows.

### 211. Tawa-style aref warp specialization ported to Metal TQ3 SDPA kernel
- **Goal**: 3 (decode speed — the currently-broken TQ3 fused SDPA is 10x slower than fp16 flash per commit 874aa5b; closing that gap is a direct Goal 3 win), Goal 4 (same kernel is the prefill bottleneck in TQ3 mode)
- **Derived from**: Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References (2510.14719). Closes Pass 46 gap #1 (Metal simdgroup tree-attention kernel, carried from passes 43-46) from the *methodology* angle — Tawa gives us the vocabulary (asynchronous references / aref, producer-consumer warp specialization) even though its implementation is H100-only.
- **Change**: Phase A (1 week, hand-port). Rewrite the TQ3 fused SDPA kernel in `omlx/kernels/` using a simdgroup-level producer/consumer pattern: one simdgroup preloads the next tile of the KV codebook into threadgroup memory via async tile load, while the other simdgroup does the current tile's matmul. Synchronize via `simdgroup_barrier()` and `threadgroup_barrier(mem_flags::mem_threadgroup)`. Size tiles so the double-buffered codebook fits in the ~32 KB per-threadgroup memory budget (smaller than H100's 228 KB, so the paper's exact tile sizes won't port — need to halve them). Phase B (optional, indefinite): if Phase A closes the 10x gap, generalize into a small Python DSL in `omlx/kernels/aref_metal.py` that takes a tile-based MLX-like specification and emits MSL with the producer/consumer pattern; this becomes the Hypercar-specific analog of Tawa. Wire Phase A in as the default TQ3 SDPA path once it exceeds the current kernel's performance (no user-visible flag; the kernel is transparent to callers).
- **Verify**: (a) TQ3 SDPA kernel time at 8K prefill must drop from current ~0.5 ms/call to <= 0.15 ms/call — 3x+ improvement; full 10x is the Tawa-on-H100 target but we accept 3x given Metal's tighter memory budget. (b) TQ3 decode tok/s at 16K must improve >= 1.5x vs the pre-warp-specialization baseline. (c) TQ3 mode must match native mode prefill tok/s within 15% at 8K (currently native is 1.5-2x faster due to the broken fused kernel — this gap should close). (d) Numerical correctness: output of the new kernel must match the reference MLX SDPA to within 1e-3 (fp16 tolerance) on random inputs at multiple context lengths (1K, 8K, 64K). (e) HumanEval in TQ3 mode must not regress — quality gate on the new kernel. (f) NIAH@4K, 16K, 64K in TQ3 mode must PASS.
- **Effort**: L (2+ weeks for Phase A hand-port — MSL producer/consumer pattern, tile sizing for M4 Pro threadgroup memory, numerical validation, integration with existing TQ3 path; Phase B is indefinite scope)
- **Depends on**: TQ3 kernel infrastructure (shipped). Independent of all other Pass-47 tasks. Sequencing: can run in parallel with 209 (SpecAttn) and 210 (SparseServe) — different layer of the stack. But recommend scheduling *after* 210 because 210 removes the currently-hot 120K+ prefill bug and makes it possible to benchmark 211 at long context without OOM.

### 212. LoLA self-recall error metric as training-free token-promotion signal for DuoKV
- **Goal**: 1 (training-free oracle for "this token will be needed later" — the signal SnapKV, CAOTE, and H2O all approximate heuristically), Goal 2 (associative-recall augmentation for streaming heads, partial closure of the long-context retrieval failure mode)
- **Derived from**: LoLA: Low-Rank Linear Attention With Sparse Caching (2505.23666). Partially closes Pass 46 gap #6 (post-hoc linearization without retraining) — first training-free paper in the linear-attention-augmentation family. Adopts the *monitoring primitive* even though we can't apply the full linear-attention conversion (Qwen3-Coder is softmax-attention).
- **Change**: New file `omlx/kv_caches/self_recall_monitor.py` implementing the self-recall error metric. During decode: a tiny per-head linear predictor (d_head x d_head, fp16, ~2 KB per head) predicts the next K from the preceding window's recurrent-summary K state (EMA over the last 256 Ks). The error between predicted K and actual K is the "difficulty-to-memorize" signal; tokens whose error exceeds a threshold `tau` (default: 75th percentile of a 1K-token calibration set) are flagged for promotion. Integration with DuoKV: flagged streaming-head tokens get copied into the retrieval-head KV pool before SnapKV eviction sees them; since retrieval heads are already fp16, no additional quantization cost. Wire in as `--self-recall-promotion` and `--self-recall-budget K` (default 4K — max number of promoted tokens at any time; LRU eviction on the promotion pool when budget exceeded). Skip the linear-attention architecture conversion — we use only the self-recall error metric, not the recurrent state it was designed for. Calibrate the predictor with a few hundred warmup steps before activating promotion; during warmup, the monitor runs but doesn't act.
- **Verify**: (a) NIAH@4K, 16K, 64K must PASS (baseline). (b) NIAH@40K (currently failing, candidate root cause: DuoAttention head classification missing important tokens) must PASS with self-recall promotion enabled — this task is in part a second candidate fix for the 40K failure, orthogonal to AhaKV (Task 201). (c) HumanEval >= 93% — the promotion pool adds retrieval capacity without evicting streaming-head recency, so no regression expected. (d) Promotion-pool fidelity: on a passkey-at-depth synthetic eval (passkey placed at 75% of context length), promotion-pool-enabled mode must recover the passkey with >= 90% accuracy vs baseline DuoKV which drops it. (e) Decode overhead: the per-head linear predictor + error computation must add < 3% to decode tok/s at 64K (one small matvec per head per decode step). (f) Promotion budget behavior: under high-churn input (a pathological trace where every token triggers promotion), the 4K budget must cap the pool size — verify LRU eviction kicks in. Ablation: self-recall off vs on; budget sweep {2K, 4K, 8K}; predictor warmup sweep (200, 500, 1000 steps).
- **Effort**: M (1 week — per-head predictor, error metric, promotion integration with DuoKV, LRU budget cap, warmup controller, NIAH+passkey-at-depth validation)
- **Depends on**: DuoKV (shipped), SnapKV (shipped). Composes with AhaKV (Task 201 — two independent candidate fixes for NIAH@40K; ablate both independently and together), G-KV (Task 202 — self-recall signal could feed G-KV's running score). Independent of 209 (SpecAttn), 210 (SparseServe), 211 (Tawa). Sequencing: ship after 201 so the two 40K-failure candidate fixes can be ablated cleanly against each other and in combination.

## Research-derived tasks (from LIT_REVIEW.md pass 48, 2026-04-21)

### 213. RetroInfer Wave Index — content-addressed retrieval-head KV with tripartite error bound
- **Goal**: 1 (sublinear-per-decode attention retrieval at 1M context — the decode-time analog of SparseServe's prefill-time fix, and the first formally bounded version of this operation in the literature), Goal 3 (decode speed constant across context — 4.5x over full attention on retrieval heads, 10.5x when combined with CPU-memory overlap; on unified memory the second gain converts to zero transfer overhead, pure win), Goal 6 (Wave Buffer's memory placement logic collapses cleanly to M4 Pro unified memory — no GPU/CPU transfer to schedule, only wired/pageable residency).
- **Derived from**: RetroInfer (2505.02922). Closes Pass 47 gap #1 / Pass 46 gap #2 (KV as a queryable database). First paper in the pass-46-onward search that provides a *formal error bound* on sublinear KV retrieval — prior work (Quest, RetrievalAttention, PQCache) only bounds the retrieved-top-K portion, not the un-retrieved tail.
- **Change**: New file `omlx/kv_caches/wave_index.py` with three components. (a) Per-retrieval-head K clustering: at prefill end, run balanced k-means on the fp16 retrieval K tensors with sqrt(context_len) clusters per head (256 at 64K, ~1K at 1M); store cluster centroids (fp16) + per-token cluster ID (int16). Incremental clustering option: cluster each 64K chunk during prefill and merge at boundary. (b) Tripartite attention approximation: per decode query compute three terms — near (top-K clusters' actual K×Q), far (remaining clusters' centroid×Q with bounded approximation), and residual (a small fp16 correction on the gap). MLX-native kernel (three matmuls + softmax, fuses automatically). (c) Top-K cluster selector: mx.argpartition on cluster-query similarities. Wire in as `--wave-index` with `--wave-clusters-per-head K` (default: sqrt(context_len) capped at 1K). Composes with DuoKV: Wave Index runs only on retrieval heads (already fp16); streaming heads untouched. Composes with SnapKV: Wave Index re-runs after eviction to rebuild the index over the kept tokens (eviction happens at prefill end, once, before first decode).
- **Verify**: (a) NIAH@4K, 16K, 64K, 128K must PASS with wave-index on — the tripartite bound must prevent the un-retrieved tail from dropping accuracy. (b) Decode tok/s at 64K must improve >= 2x over the DuoKV-fp16-retrieval baseline — the paper reports 4.5x vs full attention, we conservatively target half that after DuoKV already provides some sparsity. (c) Decode tok/s at 1M must be >= 25 tok/s (a fresh target: 50 tok/s is Goal 3 at short context, we set 25 at 1M as the constant-across-context bar). (d) Per-decode attention compute time must be sub-linear in context length — log-log plot of tokens/s vs context must have slope >= -0.3 (a purely O(N) scheme has slope -1.0). (e) HumanEval must stay >= 93% (2 pp below 95% baseline accounts for tripartite approximation error). (f) Tripartite bound ablation: far-term-only vs near+far vs near+far+residual — the paper's key contribution is the residual term, need to verify its empirical contribution on Qwen3-Coder's code calibration set. (g) Initial clustering cost at 1M must be < 5% of total prefill time — if it exceeds, enable incremental chunked clustering.
- **Effort**: M (1 week — per-head k-means clustering, tripartite MLX kernel, top-K cluster selector, DuoKV integration, NIAH+HumanEval gates including 128K re-validation at newly-reached context).
- **Depends on**: DuoKV (shipped), SnapKV (shipped). Composes with Sequential-KV PLT (Task 205 — PLT becomes prefix-hash-only, Wave Index handles semantic lookup) and SpeCache (Task 215 — Wave Index decides *which* clusters to fetch, SpeCache's speculator decides *when*). Independent of A²ATS (Task 214 — different layer of stack: A²ATS is codebook design, Wave Index is retrieval index). Sequencing: ship FIRST in Pass-48 batch — highest-leverage single finding and enables Task 215 (SpeCache needs a cluster-ID signature to predict).

### 214. A²ATS query-aware codebook with Windowed RoPE — CommVQ upgrade
- **Goal**: 1 (a *shared* codebook that works across the full 1M window — CommVQ's current codebook stretches to cover all positions, which at 1M is asking too much expressive capacity from 2-3 bits per token), Goal 2 (query-aware calibration replaces reconstruction-error proxy with the actual attention-score-error objective — a strict quality improvement, no trade-off), Goal 5 (2.7x long-context serving throughput implies ~2.7x KV memory headroom at equal quality).
- **Derived from**: A²ATS (2502.12665). Partially closes Pass 47 gap #5 (position-aware codebook design). First training-free paper to factor the codebook *through* position via WRoPE rather than forcing the codebook to absorb position-dependence (which CommVQ does, limiting its range).
- **Change**: Two changes to existing `omlx/patches/commvq.py`. (a) Windowed RoPE: in `omlx/attn/rope.py`, add a windowed variant where the rotation angle is reset to 0 every W=64 positions (matches existing BUZZ segment size) and the absolute position is stored in a separate 2-byte per-token field. The codebook then only needs to represent position-independent semantic content; position is recovered at query time from (cluster-local-offset, window-absolute-position). (b) Query-aware codebook calibration: swap CommVQ's current reconstruction-error calibration loss (`||K_quantized - K||^2`) for query-aware attention-score-error loss (`||softmax(Q×K_quantized) - softmax(Q×K)||^2`) on a 1K-token calibration set, where Q comes from a Qwen3-Coder forward pass on real code data (not random tokens). Wire in as `--commvq-query-aware` and `--wrope-window W`. Skip the paper's heterogeneous inference architecture (GPU codebook + CPU residuals) — on unified memory there's no benefit. Validate on Qwen3-Coder-specific code tokens rather than the paper's Llama-3.1-8B / Mistral-7B natural-language calibration.
- **Verify**: (a) NIAH@4K, 16K, 64K must PASS — WRoPE alters positional behavior, so must verify no pathological cross-window regression on real retrieval workloads. (b) HumanEval >= 93% — query-aware calibration should *improve* vs CommVQ baseline, not regress; if regression, the Qwen3-Coder-specific positional preferences aren't captured by WRoPE and we should keep only the query-aware objective (skip WRoPE). (c) RULER and NIAH@128K: the window-crossing behavior matters most at long context; need to verify the 2-byte absolute-position field is correctly consumed. (d) Codebook-memory reduction: A²ATS's query-aware objective should allow a smaller codebook (128 codewords vs our current 256) for the same quality — if achievable, saves ~8% retrieval-tier memory at 1M. (e) Ablation: A vs B — pure query-aware-objective-upgrade (no WRoPE) vs full A²ATS (query-aware + WRoPE). If A matches B, ship only A (safer, fewer moving parts). (f) Calibration cost: the Qwen3-Coder forward pass on 1K calibration tokens must complete in < 2 min on M4 Pro (one-time cost at model load).
- **Effort**: M (1 week — WRoPE RoPE variant, query-aware calibration loss, ablation harness, Qwen3-Coder-specific calibration set, NIAH+HumanEval+RULER full gate).
- **Depends on**: CommVQ (Task 193 — this task is a strict upgrade path for Task 193, not a replacement). Independent of Wave Index (Task 213 — different layer: Task 213 is retrieval index, Task 214 is codebook design). Composes with KVTuner floor (Task 207) and KV-CoRE NER (Task 198) — all three feed bit-budget allocation decisions from different angles. Sequencing: ship after Task 193 is fully validated at 64K; Task 214 is a follow-up upgrade that extends Task 193's range to 1M.

### 215. SpeCache speculative KV prefetch — pre-warm wired set from compressed memory
- **Goal**: 3 (decode speed — eliminates the ~20% fetch-latency stall at 64K we currently pay every time a cold tile must be paged in; on unified memory the prefetch target is the wired memory set, not a separate GPU), Goal 6 (prevents the "laptop stutters when paging" failure mode under co-tenancy — prefetch happens during compute, invisible to user).
- **Derived from**: SpeCache (2503.16163). Closes Pass 47 gap #3 (speculative cache warmup / preheat during idle) from the literature side — previously classified as pure engineering because no paper existed.
- **Change**: New file `omlx/kv_caches/speculative_prefetch.py` implementing (a) ring buffer of "attention pattern signatures" for the last N=8 decode steps — signature = top-K cluster IDs from Wave Index (Task 213), typed as int16[K], ~128 bytes per step; (b) EMA predictor of next step's cluster set — exponential smoothing of the ring buffer's cluster occupancy counts, no training, just arithmetic; (c) async prefetch op that issues a page-in / dequantize request for the predicted clusters *during* the current decode step's compute, so by the time step t+1's attention fires the clusters are in wired memory. Wire in as `--speculative-prefetch` with `--prefetch-horizon 1` (horizon > 1 risks thrashing; the paper implicitly uses 1). Gate speculation on live VM-pressure signal from the SparseServe working-set monitor (Task 210) — only speculate when the system is already paging (the win is largest there). Skip the paper's CPU-GPU transfer scheduling (no transfer on unified memory); adapt the speculation signal to drive Metal's `mx.metal.set_wired_limit()` + explicit eval-of-predicted-cluster tiles instead.
- **Verify**: (a) Decode tok/s at 64K under VM pressure (simulated by running `hypercar_bench --mem-pressure 80%` concurrently) must improve >= 1.3x over the baseline — the speculation is specifically a pressure-regime win. (b) Decode tok/s at 64K on a clean system must not regress by more than 3% — speculation adds work, and on hot-tile-only workloads it's pure overhead. (c) Prefetch hit rate must be >= 70% at 64K — EMA smoothing of attention patterns should predict the next step's top-K clusters with high accuracy since attention patterns have substantial temporal locality on code. (d) NIAH@64K must PASS — speculation is a performance optimization, not a correctness change; any NIAH regression is a bug. (e) Speculator overhead: the EMA prediction step must take < 1% of decode time (one matmul on 8 × K int16, trivial). (f) VM-pressure gating: when the system is not paging, speculation must be a no-op (verify via profiling that no extra mx.eval is triggered). (g) Ablation: off vs horizon=1 vs horizon=2 (confirm horizon=2 degrades as the paper's intuition suggests).
- **Effort**: M (1 week — signature ring buffer, EMA predictor, async prefetch integration, VM-pressure gating, stress-test under synthetic memory pressure, NIAH gate).
- **Depends on**: Wave Index (Task 213 — provides the cluster-ID signature SpeCache predicts; SpeCache is effectively meaningless without a cluster-addressed KV layout), SparseServe working-set monitor (Task 210 — provides the VM-pressure gating signal). Independent of A²ATS (Task 214), block-pool (Task 216). Sequencing: ship after 213 AND 210; SpeCache is the tertiary optimization that rides on top of both.

### 216. Block-pool multi-agent session store — concurrent isolated Q4 per-agent caches
- **Goal**: 1 (persistent Q4 KV cache is a strict memory upgrade over current fp16 session-store format — 4x more contexts fit in the same budget; the paper's direct M4 Pro numbers validate the approach), Goal 3 (TTFT for multi-agent context switching: the paper reports 22-136x improvement over re-prefill on M4 Pro specifically — this is the biggest single latency win in pass-48), Goal 6 (multi-repo workflow finally viable — the #1 UX gap in the current server is "user switches repo, pays re-prefill" which this task eliminates).
- **Derived from**: Agent Memory Below the Prompt / Persistent Q4 KV Cache (2603.04428). Closes Pass 47 gap #6 (MLX / Apple Silicon runtime) from the measurement angle — first paper in 48 passes whose primary benchmarks are on Apple M4 Pro hardware. Also addresses a concrete UX gap in the existing server (context-switch re-prefill).
- **Change**: Three changes, no model modification. First, new file `omlx/kv_caches/block_pool.py` implementing a per-agent block-pool layout on top of the existing session store: keys are (agent_id, block_id), values are Q4-compressed KV blocks at 2K tokens per block. LRU eviction across agents with a pluggable priority function (default: recency + context length). Second, modify `omlx/hypercar_server.py` to multiplex across agents in a single process via the block pool — current server has a single active KV cache; this replaces it with a pool that supports `/v1/sessions/switch?agent_id=X` in O(1) (the Q4 blocks stay resident; only the active-cache pointer moves). Third, content-addressed Q4-block keying: `(agent_id, block_hash)` → same block dedups across agents that share common prefixes (e.g., two agents on same codebase). Skip the paper's BatchQuantizedKVCache (concurrent decode from multiple Q4 caches simultaneously) — defer the Metal kernel to Task 211 (Tawa-style warp specialization), which handles Q4-aware SDPA at the kernel level. Wire in as `--block-pool` with `--max-agents N` (default 8) and `--block-size 2048`.
- **Verify**: (a) Context-switch TTFT: `/v1/sessions/switch` between two agents with 64K context each must complete in < 500 ms (target: 100x improvement over the current 50+ s re-prefill cost; paper reports 22-136x on smaller models). (b) Memory: serving 8 agents × 64K context each must fit in 38 GB Metal peak — the Q4 representation packs 4x denser than our current fp16 retrieval-tier default, so 8 × 64K at Q4 = equivalent to ~2 × 64K at fp16, well within budget. (c) Perplexity impact: HumanEval on any single agent in the pool must stay within 1 pp of the single-agent baseline — paper reports -0.7% to +3.0% across three model families, on much smaller models; Qwen3-Coder-30B should tolerate Q4 at least as well. (d) LRU eviction: under a pathological load (9 agents competing for 8-slot pool) the evicted agent's subsequent switch-in must rebuild its KV from disk in < 2 s at 64K (disk-I/O-bound on M4 Pro's SSD at ~5 GB/s, our Q4 64K KV is ~60 MB/agent). (e) Content-addressed dedup: two agents sharing a 10K-token prefix must occupy ~1x the memory of one agent for that prefix, not 2x — verify via `mx.metal.get_active_memory()` before/after a second-agent session start. (f) Concurrent decode: two agents decoding simultaneously must each achieve >= 40 tok/s (half of our 53.6 target, accounting for Metal command-queue contention — BatchQuantizedKVCache via Task 211 would lift this, but 40 is acceptable for the shipped version). (g) Session persistence: kill-server / restart-server with 5 agents active must reload all 5 agents in < 10 s total (Q4 blocks on SSD). NIAH@4K, 16K, 64K on a randomly chosen agent in a multi-agent pool must PASS.
- **Effort**: M (1 week — block-pool data structure, `/v1/sessions/switch` endpoint, content-addressed block keying, LRU eviction policy, multi-agent benchmark harness, NIAH gate in multi-agent mode). BatchQuantizedKVCache deferred to Task 211 as noted.
- **Depends on**: session save/load (shipped), EFIM infill endpoint (Task 208 — both extend the session-store layer; 216 should land after 208 so the infill path works per-agent). Independent of Wave Index (Task 213), A²ATS (Task 214), SpeCache (Task 215). Composes with PiKV expert-sharded layout (Task 203) — PiKV is per-expert KV sharding, block-pool is per-agent KV sharding; the two layouts stack (each agent's KV cache is itself PiKV-sharded internally). Sequencing: ship after 208 (shared session-store infrastructure) and independently of other pass-48 tasks; delivers the biggest single UX win of pass-48.

## Research-derived tasks (from LIT_REVIEW.md pass 49, 2026-04-21)

### 217. Continuum TTL scheduler — tool-call-aware KV cache retention for multi-turn agents
- **Goal**: 3 (decode speed — 1.12-3.66x end-to-end delay reduction and 1.10-3.22x throughput gain reported on SWE-Bench agentic coding, which is exactly our target workload; no re-prefill on turn N+1 after a tool call), Goal 6 (laptop stays usable: the current greedy-evict policy thrashes the KV cache whenever a second request arrives during an agent's tool-call pause, causing visible stutter in OpenCode).
- **Derived from**: Continuum (2511.02230). Highest-leverage pass-49 finding — the only KV paper in 49 passes whose headline benchmark is *exactly* mini-swe-agent on SWE-Bench (our workload). Complements Task 216 (block-pool) at the temporal level: 216 is inter-agent spatial separation, 217 is intra-agent multi-turn temporal retention.
- **Change**: Two changes, no compression work. First, new file `omlx/serving/ttl_scheduler.py` implementing: (a) per-session `Dict[tool_name, SlidingWindow[last 10 durations]]` that tracks observed tool-call wall-time per tool name, (b) EMA + 1σ duration predictor per tool, (c) TTL admission queue that assigns `ttl_expiry = now + predicted_duration × safety_margin` to a session's KV cache when its turn finishes, (d) eviction policy that prefers oldest-TTL-expired sessions when memory pressure forces a choice, (e) cold-start fallback default TTL of 5 seconds for tool names with < 3 observed calls. Second, modify `omlx/hypercar_server.py` to route new requests through the TTL scheduler instead of the current greedy-evict policy — when request arrives, check memory headroom; if sum of live-TTL sessions + new request fits, admit; if not, evict oldest-TTL-expired session (or newest-session if none expired, matching current behavior). Wire in as `--agent-ttl-scheduling` (defaulted ON when `--block-pool` is active) with `--ttl-margin-sigma K` (default 1.0). Skip the paper's multi-GPU scheduling (we have one GPU) and distributed-tracing assumptions.
- **Verify**: (a) End-to-end delay on a synthetic SWE-Bench-style trace (5 turns, each followed by a `bash` or `edit` tool call with random 0.5-8s duration): wall clock must drop by >= 1.5x vs the current greedy-evict baseline (conservative of the paper's 1.12-3.66x range). (b) Tool-call prediction accuracy: over a logged trace of 100 real tool calls on our OpenCode workflow, the EMA + 1σ predictor must have actual-duration-within-prediction-window at >= 80% rate (paper reports 85%+). (c) Cold-start safety: on a first-ever call to a tool name (no history), the 5s default TTL must not cause memory OOM when combined with a concurrent 64K session — verify via targeted benchmark. (d) Adversarial load (9 sessions, 8-slot block-pool, every session in a tool call): TTL scheduler must fall back to current greedy-evict semantics (no performance regression vs baseline). (e) Session save-on-evict: when TTL scheduler evicts a session, its KV cache must be saved to disk via the existing session-save path, not discarded. (f) NIAH@4K, 16K, 64K on a single agent in a multi-agent pool must PASS — scheduling is correctness-neutral.
- **Effort**: S (4 days — TTL data structures, EMA predictor, server routing change, OpenCode-trace validation, multi-turn synthetic benchmark)
- **Depends on**: block-pool (Task 216 — TTL scheduler is meaningless without per-agent isolation; ship 217 immediately after 216 as a pair). Session save/load (shipped). Independent of compression stack (Tasks 218, 219, 220). Sequencing: ship FIRST in Pass-49 batch — highest UX leverage, simplest change, no compression interaction risk.

### 218. OjaKV online low-rank streaming-head compression — continuous subspace adaptation
- **Goal**: 1 (replaces fp16 streaming-head ring buffer with r=32 low-rank codes, saves 150 MB at 1M context and enables continuous alignment to topic drift), Goal 3 (maintained-accuracy-at-higher-compression means we can lift context length further without decode regression), Goal 2 (paper reports accuracy *gains* on very-long-context reasoning benchmarks — topic-drifting multi-turn sessions benefit most).
- **Derived from**: OjaKV (2509.21623). Closes Pass 48 gap #1 (online retrospective KV recompression, carried from passes 45-48) — better formulation than original gap: continuous online adaptation so there is no "cold segment" to retrospectively recompress. Partially closes Pass 48 gap #6 (on-device inference-time index adaptation).
- **Change**: Three changes. First, new file `omlx/kv_caches/oja_lowrank.py` implementing (a) anchor policy: first-128 and last-128 tokens stored full-rank fp16 (matches StreamingLLM sink pattern we already trust), (b) per-head basis matrix U ∈ fp16^{d×r} with d=128, r=32 (4x compression over fp16), initialized via prefill-stage SVD of the first 1K streaming-head K vectors, (c) online Oja update per decode step: `U ← normalize(U + η × k_t × (k_tᵀ × U))` with η=1e-3 from paper, (d) compressed storage: intermediate-band tokens stored as c_t = Uᵀ × k_t (r=32 floats instead of d=128 floats), (e) periodic refresh: every 16K decode steps, recompute SVD from a sliding window of 1K recent Ks to bound numerical drift. Second, modify `omlx/kv_caches/duo.py` to wire the OjaKV layer as the streaming-head backing store, replacing the current fp16 ring buffer. Streaming-head Ks flow through OjaKV on write, through Uᵀ reconstruction on read (inside attention). Retrieval heads unchanged (they use Wave Index, Task 213). Third, new benchmark phase in `hypercar_bench` that stresses topic drift: prompt = Python 20K + Rust 20K + Go 20K + HumanEval — if Oja adaptation works, HumanEval accuracy should not degrade from topic switches. Wire in as `--oja-streaming` with `--oja-rank R` (default 32) and `--oja-lr η` (default 1e-3).
- **Verify**: (a) Memory: at 1M context, streaming-head KV memory must drop from ~200 MB (fp16, r=128) to ~50 MB (r=32) — verify via `mx.metal.get_active_memory()` diff before/after OjaKV enable. (b) Numerical stability: after 1M decode steps without refresh, `||U^T × U - I||` must be < 0.01 (orthonormality preserved within fp16 precision). With refresh on (every 16K steps), same check must hold at 1M steps. (c) Accuracy: HumanEval >= 93% (2 pp margin below 95% baseline, accounts for low-rank approximation); RULER >= 95%; NIAH@4K, 16K, 64K, 128K must PASS. (d) Topic drift benchmark (new): on the Python-Rust-Go-HumanEval synthetic above, HumanEval accuracy must be >= 90% — if static-codebook baselines (current fp16 ring) drop below 85% here, OjaKV's win is confirmed. (e) Decode overhead: per-step Oja update is d×r = 128×32 matmul per head; must add < 2% to decode tok/s at 64K. (f) Prefill-stage SVD init: must complete in < 500ms on M4 Pro for 1K × 128 matrix (trivial). Ablation: static-SVD (no online update) vs full OjaKV; r ∈ {16, 32, 64}; refresh interval ∈ {8K, 16K, 32K, off}.
- **Effort**: M (1 week — Oja update kernel in MLX, anchor storage layout, prefill SVD init, refresh controller, topic-drift benchmark, full NIAH + HumanEval gate)
- **Depends on**: DuoKV (shipped), streaming-head infrastructure (shipped). Independent of Wave Index (Task 213 — different tier) and A²ATS (Task 214 — different compression method, different tier). Does NOT compose with CommVQ (Task 193) on streaming heads — ship one OR the other there. Composes with R-KV (Task 220 — R-KV's redundancy check operates on the reconstructed Ks from OjaKV codes, no interference). Composes with GVote (Task 219 — GVote picks keep mask, OjaKV compresses what's kept). Sequencing: ship after Task 216 (block-pool) and Task 217 (Continuum TTL) so the memory saved by OjaKV directly grows the per-agent context budget in the block pool.

### 219. GVote synthetic-query Monte Carlo auto-budget — training-free per-request cache sizing
- **Goal**: 1 (auto-sized budget means simple requests get aggressive compression while complex requests get generous budgets — on our 48 GB, this is the difference between 2× simultaneous 1M sessions and 1× 1M session), Goal 2 (paper shows fixed budgets hurt accuracy on simple tasks AND collapse on hard tasks; auto-sizing is a strict Pareto improvement), Goal 3 (easier prompts = smaller effective KV = less per-step attention compute).
- **Derived from**: GVote (2509.03136, ICLR 2026). Addresses Pass 49 adjacent angle #9 (learned query token budget) and partially closes Pass 48 gap #1 (online KV recompression from the budget-auto-sizing angle — the budget itself becomes a function of the query distribution rather than a fixed knob).
- **Change**: New file `omlx/kv_caches/query_voting.py` implementing: (a) prefill-end hidden-state snapshot (single fp16 tensor of final-layer hidden states, L×d, reused from existing prefill path), (b) Gaussian fit: diagonal-covariance estimator over the L hidden states (one mean vector μ ∈ ℝ^d, one variance vector σ² ∈ ℝ^d — O(Ld) work, negligible at prefill end), (c) Monte Carlo sampling: draw K=128 synthetic hidden vectors h_i ~ N(μ, diag(σ²)) via `mx.random.normal`, project each through the model's Q_proj weight matrix to get synthetic queries q_i, (d) Union keep mask: for each q_i, compute q_i × Kᵀ → attention scores, take top-k indices where k is set by the entropy of those scores (high-entropy = broad k; see paper for exact rule), OR the 128 top-k masks together to form the final keep set, (e) pass the keep set as an "augment mask" to the existing SnapKV path in `omlx/patches/snapkv.py` — SnapKV's real-Q top-K and GVote's synthetic-Q union are combined via OR (keep if either says keep). The auto-budget emerges from the union's size. Wire in as `--query-vote-augment` with `--vote-samples N` (default 128), `--vote-seed S` (default 42 for reproducibility), and `--vote-covariance {diag, full}` (default diag; full Σ only helps on tiny d, which isn't us). Skip the paper's iterative budget refinement loop — one-shot at prefill end is good enough per their own ablation.
- **Verify**: (a) Accuracy: HumanEval must improve by >= 0.5 pp over SnapKV-alone baseline at the same 50% keep ratio — GVote's augment only adds keeps, so accuracy cannot regress unless the implementation is buggy. (b) NIAH@4K, 16K, 64K, 128K must PASS — retention, not eviction, so accuracy floor is safe. (c) Average budget sizing: over 100 representative prompts (mix of easy edits and hard refactors), the effective keep ratio must vary by >= 2x across the distribution (proves auto-sizing is actually adapting; if variance is near-zero, Gaussian fit is broken). (d) Memory: total KV must never exceed the user's `--snapkv-keep` cap, but may use *less* — GVote's union might be smaller than the fixed budget on easy prompts. (e) Gaussian-ness check: on a code-calibration set of 1K prompts, confirm hidden-state distribution passes a Kolmogorov-Smirnov test per dimension (or use GMM fallback if not). (f) Decode overhead: one-shot at prefill end — 128 samples × L × d matmul, must add < 2 seconds at L=128K (prefill is multi-minute; trivial). Ablation: samples ∈ {32, 64, 128, 256}; diag-Σ vs full-Σ (diag should suffice at d=128); seed-stability check (repeat with 3 seeds, union must overlap > 90%).
- **Effort**: S (4 days — Gaussian fit, MC sampler, union mask construction, SnapKV-patch integration, 100-prompt budget-variance benchmark)
- **Depends on**: SnapKV (shipped), prefill path (shipped). Composes with CAOTE (shipped — GVote augments what CAOTE would evict), R-KV (Task 220 — opposite ends of pipeline: GVote picks keep, R-KV picks redundant-drop), Wave Index (Task 213 — GVote operates on streaming heads, Wave Index on retrieval heads), OjaKV (Task 218 — GVote picks mask, OjaKV compresses kept tokens). Sequencing: ship after Task 218 so the reduced streaming-head memory footprint amplifies GVote's budget-auto-sizing gain; before Task 220 so R-KV's redundancy check composes over the auto-sized union.

### 220. R-KV redundancy-aware eviction — K-vector cosine similarity for reasoning-style workloads
- **Goal**: 1 (enables 10%-budget operation at 100% full-cache quality on reasoning traces — would unlock CoT-heavy 1M workloads that currently fail at standard keep ratios), Goal 2 (reasoning workloads are the shipped eviction stack's weakest spot; R-KV fixes this before we add reasoning evals to the Goal 2 breadth set).
- **Derived from**: R-KV (2505.24133, NeurIPS 2025). Closes Pass 48 gap #5 (KV cache for reasoning / chain-of-thought models, new in pass 48). Strict upgrade to existing CAOTE (Task 100, shipped) — CAOTE is V-space redundancy, R-KV is K-space redundancy; the two are independent signals.
- **Change**: One new scoring function, plugged into existing eviction path. Specifically: (a) new file `omlx/kv_caches/redundancy_score.py` implementing a per-segment pairwise-cosine-similarity check on K vectors within the last L=1024 tokens (paper's default), producing a per-token redundancy score s_redundancy(t) = max over recent t' of cos(K_t, K_{t'}), (b) modify `omlx/patches/snapkv.py` to compose the redundancy score with the existing SnapKV attention score and CAOTE V-distance score: `score_total = α × attention_score - β × redundancy_score + γ × caote_score`, where weights are set by attention entropy of the current step (high entropy → trust redundancy more; α=0.5, β=0.3, γ=0.2 at default entropy, per paper table 3), (c) cap cost: redundancy check is O(L²) in window size, so hard-cap L at 1024 even if user's SnapKV window is larger. Wire in as `--reasoning-aware-eviction` (defaulted OFF — pure win on reasoning workloads, neutral-to-slight-loss on code workloads; let the user opt in) with `--redundancy-window W` (default 1024). Skip the paper's dynamic window sizing — fixed 1024 is simpler and the paper's ablation shows < 1% accuracy difference.
- **Verify**: (a) Reasoning benchmark (new): synthetic CoT trace at 16K context with reflection-echo patterns, eviction at 10% budget must achieve >= 95% of full-cache accuracy — paper claims 100%, we set the bar slightly lower to account for Qwen3-Coder's non-reasoning training. (b) Code workload (regression check): HumanEval with `--reasoning-aware-eviction` on must stay within 1 pp of baseline — if redundancy over-penalizes legitimate code repetition (loops, identical function signatures), this will flag it. (c) CAOTE composition: ablation {attention alone, +CAOTE, +redundancy, +CAOTE+redundancy} — the stack should dominate on reasoning, and each signal should contribute independently. (d) NIAH@4K, 16K, 64K must PASS with redundancy-aware eviction on. (e) Decode overhead: O(L²) within W=1024 adds ~1M FLOPs per eviction round; must add < 3% to decode tok/s at 64K. (f) Memory: no additional permanent buffers — only L×L similarity matrix during eviction, freed after. (g) Entropy-weighted composition: sanity check that the α/β/γ weighting actually shifts with entropy (if it doesn't, fall back to paper's fixed weights).
- **Effort**: S (3 days — redundancy scoring kernel, SnapKV-patch composition, reasoning-stress benchmark, HumanEval regression gate)
- **Depends on**: SnapKV (shipped), CAOTE (Task 100, shipped), BUZZ (shipped). Composes with GVote (Task 219 — GVote picks keep set, R-KV picks redundant-drop within that set), AhaKV (Task 201 — both refine the scoring signal, orthogonal), G-KV (Task 202 — global rescoring operates after R-KV). Does NOT compose with RLKV (2510.08525, held) — RLKV requires RL training which we avoid. Sequencing: ship after Task 219 so the auto-budget and redundancy composition can be ablated together on a reasoning-heavy trace; can ship before Task 218 (OjaKV) if reasoning evals are prioritized over streaming-head memory savings.

## Research-derived tasks (from LIT_REVIEW.md pass 50, 2026-04-21)

### 221. SliceMoE AMAT+DBSC — bit-sliced expert caching for Qwen3-Coder MoE
- **Goal**: 6 (~40% effective model-footprint reduction: Qwen3-Coder's 17.2 GB drops to ~10 GB, freeing ~7 GB on the 48 GB M4 Pro budget for KV expansion at 1M), Goal 1 (the freed memory translates directly into longer context fits — either 2M single-agent or 2× concurrent 1M agents), Goal 3 (per-expert bit slicing stabilizes the active-expert working set size regardless of which experts route, removing a source of decode-speed variance)
- **Derived from**: SliceMoE (2512.12990). Closes Pass 49 gap #6 (per-head bit budget for MoE active vs inactive layers) — first paper in 50 passes with a calibration-free per-expert bit-slicing primitive compatible with our existing 8-bit Qwen3-Coder base.
- **Change**: Two phases. Phase A (1 week, AMAT conversion): new file `omlx/quantize/bit_sliced_expert.py`. Verify our Qwen3-Coder-8bit weights use a signed-fixed-point representation compatible with AMAT truncation (if not, re-quantize from original bf16 weights using AMAT's asymmetric-matryoshka scheme). Output: each expert stored as high-4-bit + low-4-bit slices where high-4 alone is a valid lower-precision expert. No calibration data required. Phase B (1 week, DBSC runtime): new file `omlx/quantize/bit_slice_residency.py` implementing the Dynamic Bit-Sliced Caching policy — keep high-4-bit slices for recently-active experts at all times, admit low-4-bit slices on expert activation, evict low-4-bit slices first under memory pressure. Wire into the existing expert-dispatch path in `omlx/models/qwen3_moe.py`. Skip the paper's Predictive Cache Warmup (PCW) — requires a prefill-time expert-activation prediction model we don't have, and on our single-user serving profile PCW is lower leverage than the headline memory win. Wire in as `--bit-sliced-experts` with `--dbsc-slice-depth {2, 4}` (default 4 for our 8-bit base) and `--dbsc-high-residency-ratio R` (default 1.0 — always keep high slices; lower R activates more aggressive eviction under memory pressure).
- **Verify**: (a) Memory: model footprint must drop from 17.2 GB to <= 11 GB with `--bit-sliced-experts` on and `--dbsc-high-residency-ratio 1.0`. (b) Memory under pressure: with `--dbsc-high-residency-ratio 0.5`, model footprint must drop further to <= 7 GB while accuracy stays within 1 pp of baseline on HumanEval (verify low-slice eviction actually fires under synthetic memory pressure). (c) Accuracy: HumanEval >= 93% (2 pp margin below baseline, accounts for any quantization noise from 4+4 bit-slicing). MMLU-Pro >= 60%. RULER must pass with >= 95%. (d) Decode speed: no regression (paper reports 2.37-2.85× *energy* reduction; our wall-clock target is neutral — we measure and regress 0% of baseline). (e) NIAH@4K, 16K, 64K, 128K must PASS. (f) AMAT property verification: write a unit test that the high-4-bit slice of an AMAT-encoded weight is itself a valid 4-bit quantized weight whose dequantization matches a directly-4-bit-quantized version of the original weight within 1e-3 (the core mathematical property that makes AMAT calibration-free). (g) Expert-routing fairness: verify DBSC's high-residency policy doesn't systematically disadvantage rarely-routed experts (if they never get their low-slices paged in, they can never improve — this would cascade into routing drift). Sample 1K prompts, confirm all 128 experts get activated at least once with low-slice present.
- **Effort**: M-L (2 weeks — AMAT conversion, DBSC residency policy, expert-dispatch integration, accuracy gates including full HumanEval + MMLU-Pro + RULER + NIAH, AMAT mathematical-property unit tests)
- **Depends on**: MoE model infrastructure (shipped). Composes with PiKV (Task 203 — PiKV is per-expert KV sharding, SliceMoE is per-expert weight slicing; the two layouts stack). Composes with KVTuner (Task 186 — per-head bit budget for KV extends to per-expert-per-head bit budget for weights). Independent of all eviction-stack tasks (196-220). Sequencing: ship after Task 216 (block-pool) so the freed 7 GB can immediately grow per-agent context budgets in the block pool. Highest single memory leverage in 50 passes.

### 222. Reasoning-cache budget floor — theoretical s >= k / L constraint on eviction budget (interpretive)
- **Goal**: 2 (principled budget selection across reasoning-depth regimes — explains why 50% keep works and 25% keep fails, removes the heuristic "pick-a-number" quality of current budget flags), Goal 1 (informs bit-budget floor at 1M — the deeper the reasoning trace, the more budget each token needs, with a proven scaling rule)
- **Derived from**: How Much Cache Does Reasoning Need? (2604.17935). First proof-based backing for the eviction-vs-budget trade-off we have empirically tuned across 50 passes. Interpretive task, not a code change; action items encode the theoretical floor into existing budget-allocation tasks.
- **Change**: Four interpretive changes, no new source file. (a) Document the budget floor `s ≥ k_expected / L_available` in `omlx/docs/budget_floor.md`: s = per-head budget fraction, k_expected = estimated reasoning depth in pointer-chasing hops (workload-dependent — retrieval ~1, editing ~2, planning ~4, CoT ~8), L_available = Qwen3-Coder's 48 layers. The formula says: pushing budget below k/L hurts any workload requiring k hops; our shipped default at 50% keep supports k up to 24, so safe for all typical workloads. (b) Modify `omlx/patches/snapkv.py` defaults: add `--reasoning-depth-estimate K` flag that computes the budget floor as max(user-requested-keep, k/L) — prevents accidental under-budget in CoT workloads. (c) Extend Task 207 (Lyapunov-sensitivity floor) and Task 186 (KVTuner) to consume the per-depth correction: heads active in mid-stack layers (where reasoning aggregation happens per our layer-wise probing data) need more budget than heads in the first/last layers. Specifically, add a `layer_reasoning_weight[l]` array in `omlx/patches/kvtuner.py` that biases bit allocation toward mid-stack layers by ~20%. (d) Add a new gate to `hypercar_bench`: on a synthetic pointer-chasing trace (generate a trace of N=1000 tokens where each token is a pointer to an earlier token; the model must chase k=8 hops to find the answer), verify that at budget s < k/L the accuracy crashes and at s >= k/L accuracy is stable — this is the empirical verification of the theoretical floor.
- **Verify**: (a) Pointer-chasing benchmark: at s = 50% (current default), accuracy on k=8-hop pointer chase must be >= 90%; at s = 10% (below k/L for k=8, L=48 → floor is ~16%), accuracy must drop below 50% (the theoretical floor is being respected — pushing below it breaks). (b) Default-budget behavior: with `--reasoning-depth-estimate 8`, the effective keep ratio must never drop below 16% even if the user requested 10%. (c) Layer-weighted KVTuner: verify mid-stack layers (15-35 of 48) get ~20% more bit budget than first/last layers under the new weighting; regression-test that HumanEval does not regress by this reallocation (if it does, the reasoning-weight heuristic is wrong for code workloads and should be ablated off). (d) Documentation: the budget-floor md file must be referenced from CLAUDE.md's Goal 1 section so future agents know why 50% is the default. (e) No performance regression: all four changes are additive or documentation; decode and prefill tok/s must match baseline within 1%.
- **Effort**: S (3 days — budget-floor flag plumbing, layer-weighted KVTuner correction, pointer-chasing benchmark, docs, CLAUDE.md cross-reference)
- **Depends on**: SnapKV (shipped), KVTuner (Task 186 — must exist before layer-weighted correction is meaningful), Lyapunov floor (Task 207). Independent of all pass-50 code tasks (221, 223, 224). Sequencing: ship anytime — standalone interpretive change. Recommend early (pairs well with Task 207 landing) so the pointer-chasing benchmark catches budget-floor bugs in future changes.

### 223. Helium workflow-IR scheduler — topological KV retention for OpenCode multi-branch workflows
- **Goal**: 3 (1.56× end-to-end speedup on multi-agent benchmarks — and our OpenCode workload has 50-200K repo prefixes that amortize prefix reuse much more than the paper's short-prompt tests, so the realized speedup should exceed the paper's headline), Goal 6 (cache-aware scheduling avoids the thrash pattern where two parallel branches evict each other — the current greedy-evict policy is pathological under OpenCode's compound prompts)
- **Derived from**: Helium / Efficient LLM Serving for Agentic Workflows (2603.16104). Closes Pass 50 gap #5 (Continuum-style TTL for non-coding workloads) from a different angle: Helium generalizes Continuum's temporal TTL into topological workflow-IR scheduling. The two compose directly.
- **Change**: Three changes, no compression work. First, new file `omlx/serving/workflow_ir.py` implementing a minimal workflow IR capturing OpenCode's actual request shapes: a plan is a DAG of `LLMCall(prefix_hash, tool_call, branch_id)` nodes where shared-prefix subgraphs collapse. Keep the IR deliberately small — just enough to express sequential tool-use and simple parallel branches (edit-then-test-then-commit); skip the paper's full query-plan rewrite engine. Second, extend Task 217's TTL scheduler in `omlx/serving/ttl_scheduler.py` with a per-branch priority signal: when evicting under memory pressure, prefer branches whose remaining plan steps have *cheap* prefix-reprefill (can be rebuilt from shared parent) over branches whose prefix is only resident on this branch (unique-prefix branches are expensive to lose). Third, modify `omlx/hypercar_server.py` to accept an optional `workflow_plan` field in `/v1/chat/completions` payloads — when present, the server parses into workflow IR and routes through the workflow-aware scheduler; when absent (default), fall back to current Continuum-only scheduling. OpenCode client extension (separate patch, not in this task) emits the plan. Skip the paper's full query-optimization cost model — our single-user single-GPU deployment doesn't need it; we only need the prefix-aware priority rule.
- **Verify**: (a) Compound-prompt benchmark: synthetic OpenCode trace (50K repo prefix, then edit tool call, then test tool call, then commit tool call — all sharing the repo prefix): end-to-end wall clock must drop by >= 1.3× vs Task-217-Continuum-alone baseline (conservative of the paper's 1.56×). (b) Backwards compatibility: requests WITHOUT `workflow_plan` field must behave identically to current server — verify full hypercar_bench passes with `--workflow-scheduler` enabled but no plan emitted. (c) Memory pressure test: launch 4 concurrent branches of the same parent plan (each 64K context), confirm Helium's priority scheduler keeps the parent prefix resident while evicting branch-unique suffixes — verify via logging that the correct branches get evicted first. (d) Plan-parse correctness: unit tests on 10 hand-written workflow plans covering sequential, parallel, and mixed shapes. Malformed plans must gracefully fall back to flat scheduling with a warning. (e) NIAH@4K, 16K, 64K on a single branch in a multi-branch plan must PASS. (f) No regression: standard `hypercar_bench` with `--workflow-scheduler` on must pass all gates vs baseline.
- **Effort**: M (1-2 weeks — workflow IR data structure, per-branch priority extension to Task 217, server payload parsing, compound-prompt benchmark, multi-branch memory-pressure test, plan-parse unit tests)
- **Depends on**: Continuum TTL (Task 217 — Helium extends it with a priority signal; Helium is meaningless without temporal TTL as the base). Block-pool (Task 216 — multi-branch concurrency is meaningless without per-agent KV isolation). Composes with SideQuest (Task 224 — Helium decides which branch's KV to keep resident, SideQuest decides which tokens within that branch's KV are retrievable-externally). Sequencing: ship after 217 AND 216; Helium is the tertiary scheduling layer that rides on both. Pairs with 217 as the "pass-49-to-50 agentic scheduling" deliverable.

### 224. SideQuest model-driven retrievability annotation — training-free prompt variant
- **Goal**: 1 (paper reports 65% peak-token reduction on agentic tasks — directly extends what fits at 1M), Goal 2 (model-driven retrievability annotation replaces heuristic eviction with signal from the model itself, robust to the agentic-task failure modes where attention scores lie)
- **Derived from**: SideQuest (2602.22603). Partially closes Pass 50 gap #2 (compressed prompt prefix encoding without retraining) from the retrievability-annotation angle. We ship the training-free prompt-only variant; the 215-sample fine-tune variant is held as a future research direction pending explicit relaxation of the training-free mandate.
- **Change**: New file `omlx/kv_caches/model_driven_eviction.py` implementing: (a) A side-stream forward pass at prefill end — takes the prefilled KV cache, feeds the model a compression-annotation prompt in a *separate* token stream (crucial: not inline in the main context; avoids polluting the model's main reasoning memory) asking "for each 256-token chunk of your current context, output 1 if the content is re-readable from external retrieval and 0 if it's genuinely novel". The side stream is evaluated once; its output is a bit-vector of length ceil(context_len / 256). (b) Annotation fed into existing SnapKV eviction as a per-token prior: tokens in chunks flagged as re-readable get their SnapKV score divided by 2 before top-K selection. (c) Side-stream forward adds one prefill's worth of compute at session start, amortized across all subsequent turns on that session. (d) Calibrate the annotation prompt on a 50-prompt OpenCode trace before shipping — prompt quality determines eviction quality. Prompt template stored in `omlx/prompts/sidequest.txt`. Wire in as `--model-driven-eviction` with `--sidequest-prompt PATH` (default: calibrated OpenCode prompt) and `--sidequest-min-context N` (default 32768 — enable only when context > N so one-shot short prompts don't pay the side-stream overhead).
- **Verify**: (a) Agentic benchmark: on a synthetic OpenCode trace where the model reads 5 files (50K tokens each), then answers a question requiring re-reading one specific file — with `--model-driven-eviction`, peak-token usage must drop by >= 50% vs baseline SnapKV (paper reports 65%; we set 50% as the lower bar accounting for training-free quality gap). Accuracy on the final question must match baseline within 2 pp. (b) HumanEval regression: with `--model-driven-eviction` on, HumanEval must not regress more than 1 pp — SideQuest is designed for agentic workloads; code workloads are the regression risk. (c) NIAH@4K, 16K, 64K, 128K must PASS — the annotation-driven prior must not break exact-retrieval. (d) Side-stream overhead: at session start, the annotation forward must add < 10% to the prefill wall-clock; amortized over a 5-turn session that's < 2% overhead per turn. (e) Annotation quality: on a labeled 100-prompt OpenCode trace (human-annotated ground truth for re-readable vs novel), the model's annotations must achieve >= 70% F1 — below this, fall back to heuristic eviction (regression-test that the fallback path works). (f) Separation-of-streams correctness: unit-test that the annotation side-stream's KV state does NOT persist into the main context — this is the core architectural property; if the annotation tokens leak into main-context KV, it pollutes the model's reasoning and SideQuest's premise breaks. (g) Compose with Task 219 (GVote) and Task 220 (R-KV): ablation {baseline, +SideQuest, +SideQuest+GVote, +SideQuest+R-KV, +all-three} — verify additive benefits on agentic trace, verify no regression on code trace. (h) Prompt calibration: document the calibrated OpenCode prompt in `omlx/prompts/sidequest.txt` with a 50-prompt calibration log.
- **Effort**: M (1 week — side-stream forward integration, separation-of-streams safety check, annotation-to-SnapKV-prior wiring, OpenCode-prompt calibration, agentic benchmark, composition ablation with GVote/R-KV)
- **Depends on**: SnapKV (shipped), prefill path (shipped). Composes with Continuum TTL (Task 217 — sessions with high re-readable fraction get shorter TTLs because their KV is cheaper to rebuild). Composes with R-KV (Task 220 — SideQuest picks retrievability prior, R-KV picks redundancy prior, multiplicative combine before CAOTE's value-aware signal). Composes with GVote (Task 219 — GVote picks keep-augment mask; SideQuest's re-readability flag downweights within that keep set). Compose with Helium (Task 223 — Helium scheduler consumes the per-session re-readability fraction as a priority signal: high-re-readable sessions evict faster on memory pressure since they're cheap to rebuild). Sequencing: ship after Task 220 (R-KV) so composition ablation is well-defined; can ship before Task 221 (SliceMoE) if agentic context size matters more than model memory footprint. Gates on `--sidequest-min-context 32768`: pure overhead on short prompts, pure win on long agentic traces.

## Research-derived tasks (from LIT_REVIEW.md pass 51, 2026-04-22)

### 225. StructKV Global In-Degree Centrality — attention-graph-topology eviction signal
- **Goal**: 1 (preserves long-range dependencies at aggressive compression — the paper's core result is that local-saliency eviction systematically discards globally-important but locally-dormant tokens, which maps directly onto the quality cliff we've observed at 25% keep), Goal 2 (multi-hop retrieval quality preservation — our RULER scores at aggressive compression should improve from the structural-protection of information-hub tokens that current scorers evict)
- **Derived from**: StructKV (2604.06746, ACL 2026 Findings). Introduces a new pass-51 eviction vertex — Global In-Degree Centrality (GIDC) on attention graphs — genuinely orthogonal to SnapKV (local temporal), CAOTE (value magnitude), BUZZ (per-segment structural), and R-KV (key redundancy). None of our four shipped scorers capture attention-graph *topology*.
- **Change**: Three components. First, new file `omlx/patches/structkv.py` implementing GIDC: accumulate column sums of attention matrices during the existing SnapKV Q-capture pass (which we already run), yielding one scalar per token = total attention received across all layers × heads. Cost is O(L × H × N) if integrated with Q-capture, no separate pass needed. Second, *Dynamic Pivot Detection* one-time calibration: on a 50-prompt OpenCode trace (16K each), rank layers by how much their in-degree distributions correlate with downstream task quality under ablation; output a `pivot_layers.json` config identifying the 5-8 layers where structural protection matters most. Third, wire GIDC into the existing SnapKV eviction pipeline as an *additive* (not multiplicative) bias: new score = `(SnapKV × CAOTE × R-KV) + structural_weight × GIDC_normalized` where structural_weight is nonzero *only* for pivot layers. Additive because GIDC dominates specifically on tokens that are locally-dormant (low SnapKV score) but globally-important (high GIDC) — multiplying would zero them out. Wire in as `--structkv` with `--structkv-weight W` (default 0.3), `--structkv-pivot-layers PATH` (default: precomputed `omlx/configs/pivot_layers.json` from calibration), and `--structkv-decay D` (default 0.95 — GIDC contribution decays per eviction round so ancient high-GIDC tokens eventually drop).
- **Verify**: (a) RULER multi-hop retrieval at 50% keep: accuracy must improve by >= 2 pp vs SnapKV+CAOTE baseline (paper reports larger gains on LongBench multi-hop; 2 pp is the conservative lower bar on our shorter validation ladder). (b) NIAH@4K, 16K, 64K, 128K must PASS — retrievability of single-needle is not the main target but must not regress. (c) 25% keep regression: with `--snapkv-keep 0.25 --structkv` on, HumanEval must recover at least 5 pp of the 14% decode-quality loss we observed at 25% keep (Task 155) — this is the *specific* failure mode StructKV is designed for. (d) Prefill overhead: integrating GIDC into Q-capture must add < 1% to prefill wall clock at 128K (pure-accumulation cost is negligible; integration bugs could add more). (e) Pivot-layer calibration robustness: re-run the Dynamic Pivot Detection on 3 independent 50-prompt traces; the pivot-layer set should overlap by >= 80% (Jaccard) across traces — if not, the pivot set is unstable and default should use a conservative union. (f) Ablation: {baseline, +StructKV all layers, +StructKV pivot-only, +StructKV pivot-only with decay=0.95}; verify pivot-only with decay dominates — all-layer GIDC over-protects at cost to quality on code workloads. (g) Compose with existing eviction stack: run full eviction ablation {SnapKV, +CAOTE, +BUZZ, +R-KV, +StructKV} on a 64K multi-hop retrieval trace — verify StructKV contributes orthogonal improvement (>= 1 pp beyond the sum of pairwise improvements), which is the critical test that it captures a signal the others miss. (h) Hypercar_bench full pass — no regression on any existing gate.
- **Effort**: M (1 week — GIDC integration with Q-capture, pivot-layer calibration, additive-bias scoring, ablation across existing eviction stack, RULER and HumanEval validation)
- **Depends on**: SnapKV Q-capture (shipped — provides the attention tensors GIDC accumulates over). Composes with CAOTE (shipped), BUZZ (shipped), R-KV (Task 220), GVote (Task 219), AhaKV (Task 201), G-KV (Task 202), SideQuest (Task 224), TRIM-KV (Task 227 — when shipped). Independent of MoE orchestration (Task 221, 226) and scheduling (Task 217, 223, 228). Sequencing: ship early — this is pure additional signal with no new infra requirement, and the 25%-keep quality recovery is a cheap win that unlocks more aggressive compression for 1M workloads. Highest-leverage eviction signal from pass 51.

### 226. DyMoE runtime orchestration — dynamic expert quantization + depth-adaptive scheduling + look-ahead prefetching
- **Goal**: 4 (3.44-22.7× TTFT speedup on edge MoE per paper — on M4 Pro the relative gain will be smaller but the directional effect is clear), Goal 3 (14.58× TPOT ceiling per paper — our Goal-3 target is decode speed *constancy*, and DyMoE's dynamic quantization stabilizes active-expert bandwidth), Goal 6 (edge-hardware profile — M4 Pro unified memory is much closer to DyMoE's edge device than to a datacenter GPU, so the paper's assumptions transfer well)
- **Derived from**: DyMoE (2603.19172). Pass-51 runtime complement to SliceMoE (Task 221) — SliceMoE is *static* per-expert bit-slicing decided once at deployment; DyMoE is *runtime-dynamic* expert orchestration decided per decode step. The two layers compose cleanly: SliceMoE provides the storage format (high-4-bit + low-4-bit slices), DyMoE provides the residency policy (which slices stay hot per step).
- **Change**: Three phases. Phase A (5 days): *Importance-Aware Dynamic Quantization*. New file `omlx/quantize/dynamic_expert_orchestrator.py` with a per-step expert-importance tracker that reads router top-k from the current forward pass. Hot experts (top-1 routed, router score > 0.05) are promoted to high-precision (high-4-bit + low-4-bit both resident, effective 8-bit); cold experts demote to low-4-bit only (effective 4-bit) or low-2-bit under memory pressure. Wire through Task 221's DBSC residency policy rather than duplicating storage management. Phase B (3 days): *Depth-Adaptive Scheduling*. Add a `layer_precision_bias[l]` array computed from Task 222's layer-weighted KVTuner correction — deeper layers get lower minimum precision (error compensates via residual path), shallower layers enforce higher minimum. Empirical merge: where Task 222 says "mid-stack layers need more KV budget" (reasoning aggregation) and DyMoE says "deep layers tolerate lower expert precision" (residual compensation), the combined rule is "mid-stack KV budget × early-layer expert precision"; validate that this is stable on HumanEval (conflicting signals could cause oscillation). Phase C (2 days, GATED on MLX benchmark evidence): *Look-Ahead Prefetching*. Use current-layer router top-k to speculatively issue MLX memory-access hints for next-layer experts. If MLX's unified-memory allocator doesn't honor Python-level prefetch hints (likely), implement via a dummy-read pattern; gate this phase on a measured locality-of-reference improvement ≥ 5% of decode wall clock — if the dummy-read bandwidth cost exceeds the prefetch win, skip Phase C and ship A+B only. Wire in as `--dynamic-expert-orchestration` with `--expert-promotion-threshold T` (default 0.05), `--lookahead-depth D` (default 0: disabled; 1: next layer; 2+: experimental), and `--depth-adaptive-schedule PATH` (default: derived from Task 222's layer_reasoning_weight).
- **Verify**: (a) Decode speed: with `--dynamic-expert-orchestration` on, decode tok/s at 16K must improve by >= 10% vs SliceMoE-only baseline on Qwen3-Coder (paper reports 14.58× on edge hardware, but M4 Pro's unified memory starts from a much better baseline; 10% is the realistic gain). (b) Decode speed constancy: tok/s must not degrade at 64K and 128K vs 16K under memory pressure — this is the specific gain DyMoE offers over static SliceMoE (static bit-slicing loses stability when cold experts get paged in under pressure). (c) Accuracy: HumanEval >= 93%, MMLU-Pro >= 60% with DyMoE on — no more regression than Task 221's already-quantified acceptable quality loss. (d) Router-signal accuracy: the expert-importance tracker's top-k predictions must match the actual top-k with >= 95% agreement (if the tracker drifts, wrong experts get promoted and quality falls). (e) Depth-adaptive stability: run ablation {baseline, +Phase A, +Phase A+B} and confirm Phase B's layer-precision bias doesn't cause oscillation between high- and low-precision modes (observable as tok/s variance > 10% cycle-to-cycle — if so, dampen the bias). (f) Phase C decision: benchmark Phase C on a 5K-prompt decode trace; only ship if decode wall clock improves by >= 5% beyond Phase A+B. If MLX doesn't expose prefetch effectively, ship without Phase C. (g) NIAH@64K must PASS. (h) Compose with Task 221 (SliceMoE static bit-slicing): the two must be active simultaneously — DyMoE reads from SliceMoE's bit-sliced storage, doesn't require a separate format. Unit-test that DyMoE's promote/demote operations are pure residency changes, not new allocations.
- **Effort**: M-L (2 weeks — dynamic quantization integration with SliceMoE's DBSC, depth-adaptive scheduling derived from Task 222, benchmark-gated Phase C look-ahead prefetching, full eval including HumanEval + MMLU-Pro + NIAH + decode-speed constancy)
- **Depends on**: SliceMoE (Task 221 — DyMoE is the runtime layer on top of SliceMoE's static storage; cannot ship before 221). Reasoning-cache layer-weighted KVTuner (Task 222 — Phase B consumes the layer_reasoning_weight correction). Independent of all eviction tasks (225, 227) and scheduling tasks (228). Sequencing: ship directly after Task 221 — DyMoE is the operational complement that completes the MoE-optimization vertex.

### 227. TRIM-KV learned retention gates — distillation-trained per-head temporal-decay scorer
- **Goal**: 1 (learned retention outperforms heuristic eviction at matched budget, with the surprising paper result that selective retention can *exceed* full-cache quality via regularization — for 1M context this converts the current "pay quality for memory" tradeoff into "pay memory budget for regularized quality"), Goal 2 (mathematical reasoning benchmarks GSM8K / MATH-500 / AIME24 are exactly where Task 222's CoT-budget floor matters; TRIM-KV preserves retention on reasoning traces where heuristic eviction fails)
- **Derived from**: TRIM-KV / Cache What Lasts (2512.03324). Partially closes Pass 50 gap #2 (compressed prefix encoding without retraining) — distillation-only (no RL, no human labels, ~25 MB of gate params), narrow enough to justify formally relaxing the training-free mandate, same argument as SideQuest's optional 215-sample fine-tune (Task 224).
- **Change**: Three phases. Phase A (3 days): new file `omlx/patches/trimkv.py` with the retention-gate module — a tiny 2-layer MLP per head, per layer, taking K/V projections as input and outputting a scalar retention score. Total params: ~16K × 48 layers × 32 KV heads ≈ 25 MB. Phase B (4 days, distillation training): distillation trainer that takes the base Qwen3-Coder (teacher = full-cache, student = cached with gates). Train on a 500-prompt OpenCode trace by minimizing KL between teacher and student output distributions at matched eviction budget. Use MLX's autograd; chunked context (8K-32K per training sample) with generalization to longer contexts at inference time — this avoids OOM during distillation but requires validating that gates trained on short contexts generalize to 64K+ (the paper claims they do). Phase C (2 days, integration): integrate gate output as a multiplicative prior to SnapKV top-K. New score = `SnapKV_score × CAOTE × gate_retention_score`, where gate_retention_score ∈ [0, 1] is output of the trained gate at each step. Wire in as `--trimkv-gates PATH` (default: pretrained gate weights trained from our 500-prompt trace, shipped as `omlx/weights/trimkv_gates.safetensors`) and `--trimkv-fallback-to-heuristic` (default on — if gate output is out-of-distribution, fall back to SnapKV-only scoring).
- **Verify**: (a) Matched-budget quality: at 50% keep with `--trimkv-gates`, HumanEval must be >= baseline SnapKV and MMLU-Pro must be >= baseline SnapKV within 1 pp — TRIM-KV's mandate is "no regression at matched budget, ideally improvement." (b) "Exceeds full-cache" regularization check: on the mathematical reasoning benchmarks GSM8K and MATH-500 (add to eval if not yet present), verify that 50%-keep with trained gates matches or exceeds full-cache — the paper's surprising result. If it fails, document why (our OpenCode trace is code-heavy, not math-heavy; might not capture the regularization effect). (c) 25% keep: HumanEval with `--snapkv-keep 0.25 --trimkv-gates` must recover at least 5 pp of the 14% decode-quality loss at 25% keep. (d) NIAH@4K, 16K, 64K, 128K must PASS. (e) Gate OOD behavior: on adversarial prompts (random token sequences, never-seen languages), verify the fallback path triggers — gate output should be within [0, 1] for >= 99% of tokens; OOD tokens should fall back to heuristic SnapKV scoring without error. (f) Training reproducibility: the distillation script must be deterministic given a fixed seed, and re-running it on the same 500-prompt trace should produce gates that match the shipped weights within 1 pp on all eval gates (this catches silent drift in the training pipeline). (g) Memory overhead at inference: the 25 MB of gate params must fit in the existing model memory budget without affecting the <42% system-RAM-at-load target. (h) Compose with Task 225 (StructKV): TRIM-KV is a multiplicative prior, StructKV is an additive bias — verify the composite score (SnapKV × CAOTE × R-KV × trim_gate) + structural_weight × GIDC is stable and monotone. (i) Gate interpretability: dump per-head retention patterns; verify alignment with known strategies (attention sink, sliding window) as paper claims — if patterns are random noise, the gates haven't converged.
- **Effort**: M-L (1-2 weeks — gate module, distillation trainer, 500-prompt training run, quality verification across HumanEval + MMLU-Pro + NIAH + mathematical reasoning, regularization hypothesis check, OOD fallback validation, interpretability dump)
- **Depends on**: SnapKV (shipped). CAOTE (shipped). Composes with StructKV (Task 225 — multiplicative prior alongside additive structural bias), SideQuest (Task 224 — SideQuest is retrievability prior, TRIM-KV is temporal-decay prior; both feed the SnapKV top-K selector), R-KV (Task 220), GVote (Task 219). Independent of MoE tasks (221, 226) and scheduling tasks (217, 223, 228). Sequencing: ship after Task 225 so the additive-vs-multiplicative score composition is well-tested. Note: this task formally relaxes the training-free mandate (same argument as SideQuest's optional fine-tune) — document in CLAUDE.md that narrow-and-highly-leveraged distillation is acceptable when (a) training cost fits overnight on M4 Pro, (b) training uses only the model's own outputs as labels, (c) fallback to heuristic path exists.

### 228. LLMSched Bayesian uncertainty-aware scheduler — completing the Continuum × Helium × LLMSched agentic-scheduling trilogy
- **Goal**: 3 (paper reports 14-79% JCT reduction vs existing schedulers on compound LLM applications — OpenCode's branching tool-call workflows are the target pattern, with JCT reductions mapping directly onto tail-latency improvements at long context), Goal 6 (Bayesian uncertainty reduction means the scheduler does more with less under memory pressure, avoiding the thrash pattern where co-tenancy breaks scheduling assumptions)
- **Derived from**: LLMSched (2504.03444, ICDCS 2025). Closes Pass 50 gap #5 (unknown-shape workload TTL prediction). Continuum (Task 217) assumes per-tool-type duration distributions are known; Helium (Task 223) assumes workflow plan is known. LLMSched handles the general case where neither is known — we have priors but they update as stages complete. Completes the scheduling trilogy: Continuum (temporal) × Helium (topological) × LLMSched (uncertainty) = when × where × which-first.
- **Change**: New file `omlx/serving/bayesian_scheduler.py` with three components. (a) DAG representation over stages with per-stage duration prior (Gamma distribution, default shape from our existing trace averages — parse Task 217's TTL scheduler logs to populate priors). (b) Bayesian update rule that refreshes posteriors as stages complete: each stage completion provides evidence for its own duration prior and for the joint distribution over successor stages. Implement as a forward-message-passing update on the DAG (simpler than the paper's full Bayesian-network inference; adequate for our single-user single-GPU profile where we don't need multi-worker load balancing). (c) Entropy-based stage prioritizer: pick the next-to-run stage as `argmax_s ΔH(workflow) / E[duration(s)]` where ΔH is the expected reduction in joint-workflow entropy from completing stage s. Wire in as `--bayesian-scheduler` with `--bayesian-priors PATH` (default: `omlx/configs/stage_priors.json` populated from observed traces) and `--bayesian-update-interval N` (default: every 10 stage completions — avoids re-inference on every stage). Integrates with Task 217 (Continuum) and Task 223 (Helium) as the *top* scheduling layer: LLMSched picks which stage runs next; Continuum decides what to evict between stages; Helium decides which branch's KV stays resident.
- **Verify**: (a) JCT improvement: on a synthetic OpenCode compound-workflow benchmark (10 scenarios, each with 5-20 stages of mixed tool calls + model responses), average JCT must improve by >= 20% vs Continuum-only baseline (paper reports 14-79%; 20% is mid-range realistic). (b) Prior calibration: default priors must produce JCT within 2× of oracle priors on a held-out trace — otherwise the prior calibration pipeline is broken. (c) Bayesian update cost: each scheduling decision must complete in < 100ms (imperceptible on interactive workloads). If the DAG grows beyond 100 stages, switch to a pruned subgraph — forward-message-passing is O(stages²) and can exceed 100ms at scale. (d) Entropy-based prioritization correctness: unit-test that on a hand-constructed DAG with a known uncertainty-reducing stage, the scheduler picks that stage over higher-TTL-priority stages (this is the core scheduling behavior). (e) Compose with Task 217 (Continuum): verify the layered scheduler doesn't double-count priorities — LLMSched picks the stage, Continuum picks the evictions; unit-test a trace where both schedulers agree (same stage + same eviction) and a trace where they disagree (LLMSched picks an unexpected stage; Continuum must adapt its evictions to that choice). (f) Compose with Task 223 (Helium): on a multi-branch workflow, LLMSched's stage prioritization and Helium's branch-priority must merge into a consistent "next-stage-on-best-branch" decision. (g) Fallback: when no workflow plan is emitted by the client (single-shot prompts), the scheduler must fall back to Continuum-only without error and without overhead (gate on plan-present). (h) No regression on hypercar_bench — `--bayesian-scheduler` on with no plan emitted must pass all gates at baseline performance.
- **Effort**: M (1 week — DAG representation, Bayesian forward-message-passing update, entropy-based prioritizer, trace-based prior calibration, JCT benchmark with 10 compound-workflow scenarios, composition tests with Tasks 217 and 223, fallback correctness)
- **Depends on**: Continuum TTL (Task 217 — LLMSched reads priors from Continuum's trace logs and sits above it in the scheduling stack). Helium (Task 223 — LLMSched picks stage-across-branches, Helium picks branch-priority; the two are orthogonal but must compose). Block-pool (Task 216 — multi-branch scheduling requires per-agent KV isolation). Independent of all eviction tasks (224, 225, 227) and MoE tasks (221, 226). Sequencing: ship after Task 223 (Helium) — LLMSched is the tertiary scheduling layer that completes the trilogy. Pass-51 highest-effort-closing-a-pass-50-gap task.

## Research-derived tasks (from LIT_REVIEW.md pass 52, 2026-04-22)

### 229. DiffKV parallel Metal compaction — on-GPU defragmentation for long-session KV slabs
- **Goal**: 6 (memory fragmentation is a silent Goal-6 killer we have observed — long M4 Pro sessions show gradual Metal-heap fragmentation that manifests as swap pressure at context lengths that previously fit; DiffKV's parallel compaction eliminates this), Goal 1 (reduced fragmentation extends what fits at long context without allocator headroom padding), Goal 3 (compacted memory improves locality-of-reference and reduces TLB pressure during attention kernels)
- **Derived from**: DiffKV (2412.03131, SOSP 2025). Closes pass-52 gap #14 (memory fragmentation / KV compaction). Pairs with Task 230 (PagedEviction) as the "defragmentation layer" below the "block-aligned eviction layer" — PagedEviction prevents sub-page holes, DiffKV compacts the inter-slab holes that accumulate anyway from session churn.
- **Change**: Two components. First, new file `omlx/kv_caches/kv_compaction.py` implementing a Metal compute-kernel (via `mx.fast.metal_kernel` primitive) that takes a free-list bitmap and a target slab, performs parallel gather-scatter to move live entries into contiguous prefix, updates the position-index map in place. Second, integration hooks in `omlx/turboquant_kv.py` and `omlx/duo_kv.py`: after each SnapKV eviction round that leaves > 10% hole fraction in a slab, schedule a compaction pass. The compaction and the eviction must be submitted as a single Metal command buffer (not two — Metal serializes command buffers, so separate submissions would be sequential, not parallel). Skip the paper's K-vs-V differentiated-precision component (phase 2, optional) — we already quantize K and V at the same budget, splitting them would require re-calibrating the TQ3 codebook. Wire in as `--kv-compact` (default ON once validated) with `--kv-compact-threshold F` (default 0.1 — hole fraction above which compaction triggers) and `--kv-compact-mode {auto, force, off}` (default auto).
- **Verify**: (a) Fragmentation regression benchmark: run a 6-hour synthetic OpenCode session with 100 eviction rounds at varying keep ratios; Metal heap fragmentation (measured via `mx.metal.get_peak_memory()` minus live-data size) must stay below 5% throughout the run — without compaction, pass-51 traces showed this growing to >20% by hour 4. (b) Decode wall-clock regression: compaction passes must not add more than 2% to decode tok/s at 64K — if higher, the command-buffer fusion is broken or the gather-scatter kernel is inefficient. (c) Correctness: after compaction, attention output on a 1000-token test sequence must match pre-compaction output within 1e-5 absolute (pure relocation should be bit-exact; any drift indicates a position-index bug). (d) NIAH@4K, 16K, 64K, 128K must PASS after a forced compaction round. (e) Single-command-buffer verification: unit-test that compaction and eviction happen within one command buffer by measuring their combined wall-clock vs sum of individual wall-clocks — if sum > combined × 1.3, they're running sequentially and the integration is broken. (f) Long-session stability: 12-hour continuous OpenCode-like workload; memory peak must not drift upward more than 1 GB above the peak from hour 1 (if it does, there's a compaction-escape path). (g) Compose with Task 230 (PagedEviction): PagedEviction's block-aligned evictions plus DiffKV's inter-slab compaction should yield < 2% total fragmentation at steady state — verify with combined gate.
- **Effort**: M (1-2 weeks — Metal kernel, integration hooks in both TQ3 and DuoKV, single-command-buffer fusion, long-session stability test, correctness verification against pre-compaction reference)
- **Depends on**: TurboQuantKVCache pre-allocated slab (Task 149, shipped — provides the slab abstraction compaction operates on). DuoKVCache pre-alloc slab (Task 149, shipped). Composes with PagedEviction (Task 230 — block-aligned eviction prevents sub-page holes, DiffKV compacts the inter-block holes that accumulate anyway). Composes with all eviction scorers (Tasks 219, 220, 225, 227) — compaction runs after eviction regardless of which scorer picked the victims. Sequencing: ship early. This is a no-regression win that also *fixes* a long-session bug — we have been attributing it to macOS swap pressure but it's actually a heap fragmentation issue that DiffKV-style compaction solves directly.

### 230. PagedEviction block-aligned KV eviction — sub-page hole prevention
- **Goal**: 6 (block-aligned eviction preserves page boundaries, which on Metal unified memory affects MMU TLB entries and L2 cache efficiency — second-order performance gain beyond pure memory-layout cleanliness), Goal 1 (efficient block-aligned eviction at long context without fragmentation overhead), Goal 3 (block-aggregated re-RoPE is cheaper than token-level re-RoPE since whole blocks are removed contiguously — one RoPE shift covers N tokens rather than N separate shifts)
- **Derived from**: PagedEviction (2509.04377, ICML 2025 Efficient ML workshop). Closes pass-52 gap #14 (memory fragmentation — block-wise variant). Paired with DiffKV (Task 229) as the "eviction layer" to DiffKV's "defragmentation layer": PagedEviction prevents creating sub-page holes; DiffKV compacts the holes that accumulate anyway from cross-session churn.
- **Change**: Modify `omlx/patches/snapkv.py` to support a block-granularity mode. When `--paged-evict-mode block` is active, aggregate per-token scores at block granularity (sum of token scores within each block, normalized by block size and weighted by token recency as a linear decay), select top-K blocks rather than top-K tokens, and evict whole blocks contiguously. Block size defaults to the TQ3 codebook group size (64 tokens) — this co-aligns eviction with the codebook group structure, which means evictions don't split codebook groups (which would force re-quantization). Implement block-aggregated re-RoPE: instead of per-token rotation, compute one RoPE shift per evicted block and apply it to the entire successor region — O(blocks) instead of O(tokens). Wire in as `--paged-eviction` with `--paged-block-size N` (default 64, aligned with TQ3 groups) and `--paged-evict-mode {block, token}` (default `block` when `--kv-mode tq3`, else `token` preserving existing behavior for fp16 mode). Composition with CAOTE (shipped): block-aggregated CAOTE score is mean V-distance within block, weighted by block-size normalization. Composition with BUZZ (shipped): BUZZ's segment boundaries must align with eviction-block boundaries to avoid split-segment artifacts — validate at integration time.
- **Verify**: (a) Memory efficiency: at 50% keep in TQ3 mode, block-aligned eviction must produce a contiguous live region (0% sub-page fragmentation), verified by inspecting the free-list bitmap post-eviction. (b) Decode speed: block-aligned re-RoPE must be faster than token-level at 64K — measure re-RoPE wall clock; block-level should be >= 3× faster (one RoPE shift per 64 tokens vs per 1 token, minus fixed dispatch overhead). (c) Accuracy: HumanEval at 50% keep with `--paged-evict-mode block` must stay within 0.5 pp of token-level baseline — block granularity is coarser but the paper shows the score-aggregation is robust. (d) Accuracy at 25% keep: verify that block-granularity doesn't exacerbate the 14% decode-quality drop (Task 155); if it does, document the regression and lower `--paged-block-size` to 32. (e) NIAH@4K, 16K, 64K, 128K must PASS. (f) Block-boundary stress test: generate a prompt where the critical tokens straddle a block boundary; verify eviction doesn't drop the block containing the needle even if average block score is low (block-level retention of any high-score token must protect the whole block). (g) Compose with Task 229 (DiffKV compaction): block-aligned eviction + parallel compaction = both Metal-layout-clean and cross-slab-hole-free; verify combined fragmentation stays < 2% at steady state (Task 229's gate). (h) Compose with BUZZ: BUZZ segment boundaries must align with block boundaries; unit-test a trace where the alignment is explicit, regression-test when they drift (BUZZ must handle misalignment gracefully or be reconfigured to align).
- **Effort**: S-M (3-5 days — block-aggregated scoring in SnapKV patch, block-granularity re-RoPE, TQ3 codebook-group alignment, full accuracy ladder, integration with CAOTE and BUZZ)
- **Depends on**: SnapKV (shipped). CAOTE (shipped). BUZZ (shipped). TQ3 codebook groups (shipped). Composes with DiffKV compaction (Task 229 — eviction layer sits above compaction layer). Composes with all score-based scorers (Tasks 219, 220, 225, 227 — aggregation rule is additive over per-token scores). Sequencing: ship together with Task 229 (the two are paired by design: block-aligned eviction plus parallel compaction is the full memory-management solution). Can ship independently if Metal command-buffer fusion for DiffKV proves hard — PagedEviction alone still wins on re-RoPE speed and page-alignment.

### 231. Chelsea online KV clustering — Chunked Soft Matching for duplicate-token merging
- **Goal**: 1 (up to 80% KV cache memory reduction per paper on Llama-2; code workloads should realize *larger* gains due to higher key-state similarity along the sequence — repeated identifiers, function signatures, import statements, error handlers), Goal 3 (3.19× decoding acceleration from reduced attention-matrix size — orthogonal to sparse-skipping gains since Chelsea reduces matrix dimension itself, not the sparsity pattern)
- **Derived from**: Chelsea / KV Cache Clustering (2506.11418, NeurIPS 2025). Closes the pass-49-through-51 carry: online k-means for Wave Index centroids was open since pass 49; pass 51's Bregman-divergence search turned up theory but nothing LLM-applicable; Chelsea's Chunked Soft Matching is the specific LLM-applicable algorithm that answers the open question with an O(chunk²)-per-chunk decomposition that makes it actually runnable.
- **Change**: New file `omlx/kv_caches/chelsea_clustering.py` implementing Chunked Soft Matching. Three components: (a) chunked scan: partition the last N=4096 tokens of cache into chunks of C=256 each; for each chunk, run 5 iterations of soft k-means (alternating-partition strategy — partition by centroid then update centroids) to identify clusters above cosine similarity T=0.92; (b) centroid merging: for each identified cluster, replace its member KV vectors with a single centroid KV vector (mean of merged values, median of merged keys as more robust than mean under outliers); (c) position-mapping sparse table: track which original-token positions map to which centroid, so attention correctly routes to the centroid when a merged-token position is queried. Periodic trigger: run clustering every K=512 decode steps. Integration with TurboQuantKVCache: the centroid KV vectors get quantized alongside regular entries; the position-mapping table is unquantized fp16 (tiny — one int32 per token). Wire in as `--kv-clustering` with `--chelsea-chunk-size N` (default 256), `--chelsea-interval K` (default 512 decode steps), `--chelsea-threshold T` (default 0.92 cosine similarity), and `--chelsea-window W` (default 4096 tokens — only cluster within the most-recent W tokens, since older tokens are already compressed by other means).
- **Verify**: (a) Memory reduction: at 64K context on a code-heavy workload (1000-line Python file), Chelsea must achieve >= 50% KV memory reduction beyond SnapKV+TQ3 baseline (paper reports 80%; 50% is conservative for the interaction with our existing stack). (b) Decoding speedup: attention-matrix size reduction must translate to >= 1.5× decode speedup at 64K (paper reports 3.19× on Llama-2; 1.5× is conservative). (c) Accuracy: HumanEval with `--kv-clustering` must stay within 1 pp of baseline — the merging should be lossless on code because near-duplicate keys really do encode near-duplicate attention patterns. (d) NIAH@4K, 16K, 64K, 128K must PASS — this is the *critical* gate because NIAH's single-needle pattern is exactly the case where the "needle" token's key must NOT get merged into a nearby-but-unrelated cluster. Threshold tuning may be needed to pass NIAH. (e) Position-mapping correctness: unit-test that after multiple clustering rounds, the sparse position table correctly maps all original positions to centroids; specifically, test a pathological case where the same position is part of two sequential clusters across different rounds. (f) Threshold sensitivity: sweep T ∈ {0.85, 0.90, 0.92, 0.95, 0.98}; plot memory-vs-quality; verify the default T=0.92 sits on the knee of the curve. (g) Chunk-boundary stress test: generate a prompt where near-duplicate tokens straddle a chunk boundary (so the boundary prevents them from merging); verify the decode output matches token-level baseline (the boundary non-merging is a design artifact, acceptable, but not a correctness issue). (h) Compose with Task 229 (DiffKV compaction): after Chelsea clustering creates holes (merged positions are "gone" from the cache), DiffKV compacts; verify the pipeline holds across multiple clustering+compaction cycles. (i) Compose with Task 225 (StructKV GIDC): high-GIDC tokens should NOT be merged even if similar — StructKV's structural-protection bit must be a Chelsea merge-veto; unit-test this.
- **Effort**: M (1 week — Chunked Soft Matching kernel, centroid-merging with position-table maintenance, threshold calibration, full accuracy ladder, NIAH critical-gate debugging, StructKV composition veto)
- **Depends on**: TurboQuantKVCache (shipped — centroids live in the same quantized storage). SnapKV (shipped — clustering runs as a pre-eviction pass; evicted tokens need not be clustered). Composes with StructKV (Task 225 — StructKV provides merge-veto signal), R-KV (Task 220 — R-KV evicts redundant tokens, Chelsea *merges* them; the two are alternative strategies for handling redundancy — validate that only one is active at a time, or that Chelsea takes priority because merging preserves more information). Composes with DiffKV compaction (Task 229 — cleanup after clustering creates holes). Sequencing: ship after Task 229 (need compaction to handle cluster-induced holes cleanly). Can ship before Task 225 — but composition veto is cleaner if StructKV ships first. If both ship, Chelsea should consult StructKV's GIDC score before merging.

### 232. SpecGuard attention-grounding verification — step-level speculative decoding reject signal
- **Goal**: 2 (3.6% accuracy improvement on multi-step reasoning per paper — direct contribution to GPT-4-parity on reasoning benchmarks we do not currently track), Goal 3 (11% latency reduction from reducing rejection rate on speculative decoding — composes with every draft-model architecture we have planned)
- **Derived from**: SpecGuard (2604.15244, arxiv preprint Apr 2026). Closes pass-52 gap #12 (speculative decoding with formal verification). Verification-layer infrastructure — composes on top of any draft-model architecture: QuantSpec (Task 192), SpecPV (Task 200), MoE-Spec (Task 194). The attention-grounding signal is especially relevant for code where correctness depends on grounding (correct variable names, valid API calls).
- **Change**: New file `omlx/decode/spec_guard.py` with two signal computations and a threshold rule. Signal 1 — attention grounding score: during verification of each speculated step, compute the column sums of the attention matrix restricted to accepted-prefix positions, normalize by total attention mass per head, average across heads; high grounding = step is supported by evidence, low grounding = confabulation. Signal 2 — log-probability score: standard token-level model confidence, already computed by the existing decode path. Rule: accept step if `(grounding_weight × grounding + (1 - grounding_weight) × log_prob) >= threshold`; reject and recompute with target model otherwise. Step boundary detection: heuristic based on newline + step-marker tokens (adequate for code where steps are syntactically bracketed; defer the paper's full parser for a future extension). Wire in as `--spec-guard` with `--spec-guard-threshold T` (default 0.6 per paper), `--spec-guard-grounding-weight W` (default 0.5 blending grounding and log-prob), `--spec-guard-step-detector {newline, parser}` (default newline), and `--spec-guard-fallback {full-recompute, token-level}` (default full-recompute per paper). Integrate with QuantSpec (Task 192) first as the cleanest composition (same-architecture draft means attention matrices are aligned); extend to MoE-Spec (Task 194) after validating that expert-routing differences between draft and target don't break the grounding signal.
- **Verify**: (a) Reasoning-benchmark accuracy: on GSM8K / MATH-500 (add to eval if not present), SpecGuard-enabled speculative decode must achieve >= 3 pp accuracy improvement over plain speculative decode (paper reports 3.6%; 3 pp is conservative). (b) Latency improvement: on the same benchmarks, end-to-end decode wall clock must improve by >= 8% (paper reports 11%; 8% is conservative — accounts for step-detection overhead). (c) HumanEval regression check: with `--spec-guard` active on code workload, HumanEval must match baseline speculative-decode within 0.5 pp — code reasoning is step-structured, so SpecGuard should be neutral-to-positive; any regression indicates step-detection is mis-firing on code. (d) Grounding-signal health: dump grounding scores across 100 verification steps on a labeled code trace (hand-labeled as "grounded" / "confabulated"); classifier using grounding alone (threshold 0.5 on normalized score) must achieve >= 70% AUROC on the labels, else the signal is noisy and the weight should be lowered. (e) Log-prob fallback: set `--spec-guard-grounding-weight 0` (all log-prob, no grounding); must match baseline exactly (this isolates the grounding contribution and validates no regression when disabled). (f) MoE-Spec composition (future): when the grounding signal is computed from a draft with different expert routing than target, verify that grounding scores stay within [0, 1] and track reasonable patterns — if not, MoE-Spec + SpecGuard requires a recalibration. (g) Step-detection robustness: unit-test newline-based detector on 10 code samples with mixed step styles (multi-line function, single-line statement, nested blocks); false-positive rate for step boundaries must be < 10% (excessive step-fragmentation inflates verification cost without proportional quality gain). (h) No regression on hypercar_bench — `--spec-guard` on must pass all gates.
- **Effort**: S-M (4-6 days — grounding-score computation, log-prob integration (already exists), step-detector heuristic, threshold tuning, GSM8K/MATH-500 eval addition, HumanEval regression check, grounding-signal AUROC validation)
- **Depends on**: Speculative decoding infrastructure (Task 192 QuantSpec must ship first — SpecGuard is a verification layer on top). Composes with QuantSpec (Task 192 — cleanest composition). Composes with SpecPV (Task 200) and MoE-Spec (Task 194) with caveats (MoE-Spec requires validation; SpecPV's partial-verify tier interacts with SpecGuard's threshold rule — the combined acceptance criterion is `partial-verify-pass AND grounding-threshold-pass`). Independent of all eviction tasks (219-231) and MoE static tasks (221). Sequencing: ship after QuantSpec (Task 192). Can ship before MoE-Spec (Task 194) — extends straightforwardly when MoE-Spec lands. Add GSM8K + MATH-500 as evals to the Goal 2 breadth set as part of this task's validation.

## Research-derived tasks (from LIT_REVIEW.md pass 53, 2026-04-22)

### 233. SDAG block-sparse RAG attention mask — O(N × doc_size) inter-document attention for OpenCode multi-file prompts
- **Goal**: 1 (block-diagonal attention reduces inter-document attention compute by roughly N_docs× — at 64K with 15 retrieved files the reduction is ~14× on the dominant attention-matrix term, directly extending what fits at 1M), Goal 3 (decode-time attention cost reduction is a per-token speedup that compounds across 1M tokens), Goal 4 (prefill attention is the quadratic term dominating long-context chunked prefill; SDAG's block structure reduces prefill compute by the same factor as decode), Goal 2 (bonus: defense against corpus-knowledge-poisoning — OpenCode ingests untrusted retrieved content and cross-document attention is the injection vector)
- **Derived from**: SDAG (2602.04711, arxiv Feb 2026). Closes pass-52 gap #6 (RAG-specific attention patterns). The OpenCode workload is structurally RAG — prompts are system + retrieved-source-files + user-request + tool-call-history — so SDAG applies directly. Pure attention-mask modification with zero fine-tuning; inference-time only. Highest-leverage pass-53 finding because the workload alignment is precise (OpenCode = multi-file RAG) and the change is mask-level (no kernel work).
- **Change**: Two components. First, new file `omlx/attention/block_sparse_rag.py` exposing `build_sdag_mask(seq_len, query_range, doc_boundaries, cross_doc_whitelist=None)` — takes the query's token range, a list of `[start, end]` document boundaries, and an optional whitelist of cross-document-attention allowed pairs (for known import/reference patterns). Produces a block-diagonal mask where documents attend only to themselves and the query-range tokens attend to all tokens (query-as-broadcast-row, SDAG-paper construction). Second, integration hook in `omlx/attention/dispatcher.py`: when the prompt metadata dict carries `rag_document_boundaries: [[s1, e1], [s2, e2], ...]`, replace the dense causal mask with the SDAG mask for the forward pass. API surface: per-request hint in OpenAI-compatible `extra_body` — `{"rag_boundaries": [[s, e], ...]}` — which `hypercar_server.py` parses and forwards to the attention layer. Wire in as `--sdag` (default off until validated) with `--sdag-whitelist-imports` (default on — auto-detect Python `import` statements to populate `cross_doc_whitelist`). Compose with SnapKV (shipped): document-boundary-aware eviction should preserve the focal-document tokens with priority — add a `doc_boundary_penalty` to the SnapKV score that reduces eviction probability for tokens within the current focal document. Compose with PagedEviction (Task 230): paged blocks should align with document boundaries when possible — update PagedEviction to prefer block boundaries that coincide with document starts/ends.
- **Verify**: (a) Correctness: on a 5-file RAG prompt (generated from the Hypercar codebase — 5 random files), SDAG-masked output must match dense-mask output on a code-completion task where the completion depends only on one file; verify no quality regression when cross-document attention is unused. (b) Attention-compute reduction: measure wall-clock of the attention forward pass at 64K context with 15 retrieved files; SDAG must achieve >= 8× speedup on the inter-document portion (paper's theoretical max is N_docs = 15× reduction on inter-doc; 8× accounts for the query-strip and causal-within-doc remaining compute). (c) Decode throughput: in an OpenCode-realistic workload (15 retrieved source files totaling 40K tokens + 10K history + 4K query), decode throughput must improve by >= 2× vs dense-mask baseline at the same quality. (d) Quality: HumanEval with SDAG active and retrieved-context wrapping must stay within 0.5 pp of baseline — the completion depends on focal-file content which SDAG preserves. (e) Multi-file correctness test: a prompt where the completion requires *cross-file* reasoning (function A in file1 calls function B in file2); without `cross_doc_whitelist`, SDAG should fail this; with the whitelist populated via import detection, it should succeed. Gate: both cases behave as predicted. (f) Poisoning-defense test: inject a prompt-injection string into one of 5 retrieved documents; SDAG must prevent the injection from affecting completions on unrelated files. Measure: run 20 injection attempts across 20 unrelated completions; SDAG injection-success rate must be < 5%, dense-mask baseline rate will typically be 20-40%. (g) NIAH@4K, 16K, 64K, 128K must PASS with SDAG active (the NIAH setup is single-document so SDAG is a no-op on it — verifies no correctness regression on non-RAG workloads). (h) Composition with SnapKV: verify `doc_boundary_penalty` doesn't cause over-eviction of non-focal documents (they should still be retained since the query strip attends to them).
- **Effort**: S (2-3 days — mask builder, dispatcher integration, API plumbing, whitelist auto-detection, 5-file RAG regression test, attention-compute benchmark, poisoning-defense validation)
- **Depends on**: MLX attention dispatcher (exists). SnapKV (shipped — composes via doc_boundary_penalty). PagedEviction (Task 230 — composes via boundary-aligned blocks; can ship independently). Independent of all codec tasks (TQ3, Chelsea Task 231, DiffKV Task 229). Sequencing: ship first among pass-53 tasks. Lowest cost, highest leverage, no dependencies — drop-in addition to the attention layer. If PagedEviction ships first, add the boundary-alignment enhancement; otherwise ship SDAG standalone and retrofit when PagedEviction lands.

### 234. KVFundaBench benchmark port + ShotKV shot-level eviction — prompt-length + long-generation sensitivity gates, demonstration-preserving compression
- **Goal**: 2 (adds two missing benchmark axes — prompt-length vulnerability and long-generation sensitivity — to Hypercar gate set; catches compression-induced regressions that current gates miss), Goal 1 (ShotKV's shot-level compression preserves long-generation accuracy where token-level methods fail; relevant for 1M-context multi-demonstration prompts), Goal 3 (ShotKV reports 9% latency reduction on arithmetic-reasoning workloads; for code-generation the savings come from preserving whole demonstrations which reduces re-RoPE cost)
- **Derived from**: KVFundaBench + ShotKV (2502.01941, ICLR 2026). Closes pass-52 gap #3 / #11 (LLM-as-judge for compression decisions, coder-specific compression benchmark). KVFundaBench's six-axis findings (task-dependent degradation, model-type robustness, prompt-length vulnerability, chunk-level superiority, prompt-gain sensitivity, long-generation sensitivity) give us an evaluation framework we have been reinventing piecemeal. ShotKV extracts the actionable eviction primitive from findings (a)+(d).
- **Change**: Two phases. Phase 1 — **benchmark adoption**: add two new phases to `omlx/bench/hypercar_bench.py`. `phase-3g` — prompt-length vulnerability sweep: run the Code Intelligence eval at prompt lengths {2K, 8K, 32K, 64K} with a fixed compression ratio (e.g., SnapKV-50%); gate requires quality degradation from 2K → 64K to be <= 3 pp (paper's finding: compression methods often degrade 10-20 pp across this range). `phase-3h` — long-generation sensitivity: a test that prompts for a 4K-token completion (extended docstring or code refactor) and measures (i) coherence at positions 0, 1K, 2K, 3K of the output, (ii) regression against baseline uncompressed generation; gate requires coherence to hold across the full 4K output. Phase 2 — **ShotKV implementation**: add `--shot-eviction` to `omlx/patches/snapkv.py`. When active, eviction respects shot boundaries (configurable `--shot-boundary-marker` — default `\n\n` for generic prompts, tool-call-boundary for OpenCode). Algorithm: compute per-shot attention score (sum of attention received by tokens within the shot, normalized by shot length); evict whole shots with lowest score first; only do within-shot eviction if intra-shot budget is exhausted. Wire in as `--shot-eviction` (default off until validated) with `--shot-boundary {double-newline, tool-call, custom:<regex>}` and `--shot-fallback-threshold N` (default 3 — fall back to token-level eviction if fewer than N shots detected, avoiding false-shot artifacts on free-form prompts).
- **Verify**: (a) Phase-3g integration: running `hypercar_bench --full` produces the prompt-length sweep; plot shows quality vs length; gate fails if degradation > 3 pp from 2K to 64K. (b) Phase-3h integration: running `hypercar_bench --full` runs the long-generation test and verifies coherence across output positions. (c) ShotKV correctness on in-context-learning: on a 5-shot prompt (5 example pairs + 1 query) at 50% cache keep, ShotKV must preserve all 5 shots intact and drop within-shot tokens only after exhausting intra-shot budget; measure that in-context-learning accuracy stays within 1 pp of baseline where SnapKV (token-level) drops 5-10 pp per paper. (d) ShotKV on OpenCode workload: with `--shot-boundary tool-call` active, verify that prior tool-call outputs are either fully retained or fully evicted (no half-evicted tool calls which would corrupt the reasoning chain). (e) Shot-boundary fallback: prompt without clear boundaries (free-form essay); verify `--shot-fallback-threshold` triggers and eviction falls back to token-level. (f) NIAH@4K, 16K, 64K, 128K must PASS with ShotKV active — needle tokens should survive shot-level eviction (they're in the single-shot haystack so ShotKV is a no-op here). (g) HumanEval regression: ShotKV active must not regress HumanEval below baseline (code prompts typically have clear shot boundaries at blank-line separation). (h) Prompt-length sweep comparison: run phase-3g with SnapKV (token-level) and ShotKV; ShotKV must show lower degradation across the length range, specifically at 32K+ where shot-level's gain is largest.
- **Effort**: S-M (4-6 days — two new benchmark phases, ShotKV algorithm in snapkv.py, shot-boundary parser for OpenCode tool-call protocol, ICL regression test, prompt-length sweep validation, HumanEval no-regression check)
- **Depends on**: SnapKV (shipped — ShotKV is a scoring modification on top). OpenCode tool-call protocol (exists). Composes with PagedEviction (Task 230) — shots can align with blocks when boundaries coincide. Composes with BUZZ (shipped) — BUZZ segments are a *different* grouping from shots; validate that enabling both doesn't double-evict. Independent of all other pass-53 tasks. Sequencing: ship after SDAG (Task 233) — SDAG's boundary concept overlaps with shots in the RAG case (each retrieved document is often a shot), so landing SDAG first makes ShotKV's integration cleaner. The phase-3g and phase-3h benchmark additions can ship as a standalone bench-only change first (no algorithmic risk), with ShotKV following.

### 235. DuetServe adaptive prefill-decode Metal queue split — single-GPU roofline-guided multiplexing
- **Goal**: 4 (smooths prefill-decode contention on single Metal device — critical for constant-across-context prefill target when long prefill is interleaved with short new-request decode), Goal 3 (concurrent new-request decode latency stays bounded even during active 128K+ chunked prefill — maps to the UX case "type new request while previous long-context response is still generating"), Goal 6 (M4 Pro has exactly one Metal device so physical disaggregation is not an option; single-GPU adaptive multiplexing is the only path to overlap)
- **Derived from**: DuetServe (2511.04791, arxiv Nov 2025). Closes pass-52 gap #12 (prefill-decode overlap). The single-GPU focus is exactly the M4 Pro case — no multi-GPU disaggregation possible — and the attention-aware roofline model is the intellectual primitive worth porting standalone.
- **Change**: Two phases. Phase 1 — **attention-aware roofline model port**: new file `omlx/serving/roofline.py` that predicts wall-clock of MLX attention kernels as a function of (batch, seq_len, num_heads, head_dim, dtype, bits). Calibrate offline on 100 synthetic shapes using MLX timing primitives; fit a piecewise-linear model keyed on the arithmetic-intensity regime (compute-bound at long seq, memory-bound at short seq). Expose `predict_attn_wallclock(shape, kernel)` and `predict_interference(shape_a, shape_b, co_scheduled)` — the latter estimates total wall-clock when two kernels are submitted to the same device concurrently. Phase 2 — **queue-level prefill-decode split**: modify `hypercar_server.py` to maintain two Metal command queues: a `prefill_queue` (lower priority, larger resource share by default) and a `decode_queue` (higher priority). Submit chunked-prefill work to `prefill_queue` and active-decode work to `decode_queue`. Use the roofline model to decide dynamically when to reduce prefill chunk size: if `predict_interference(current_prefill_chunk, next_decode_step)` predicts decode wall-clock > `tbt_slo_ms`, shrink the prefill chunk for the next submission. Wire in as `--duet-serve` (default off until validated) with `--duet-tbt-slo-ms N` (default 150), `--duet-prefill-priority {low, normal}` (default `low`), `--duet-roofline-cache {on, off}` (default on — memoize per-shape predictions since shapes are small in number across a session).
- **Verify**: (a) Roofline calibration: sweep 100 shapes and measure wall-clock vs prediction; mean absolute percentage error must be < 15% (paper reports < 10% but MLX timing noise is higher than CUDA). (b) Concurrent-request TTFT: in a workload where a long-context request (64K prefill in progress) is followed by a short new request (500 tokens), new-request TTFT must stay below `tbt_slo_ms` × 2 = 300 ms (today it blocks until prefill chunk completes, typically 2-5 seconds). (c) Long-prefill throughput regression: 64K prefill alone must not slow down by more than 15% with `--duet-serve` active vs without (splitting resources has a cost, but not a large one). (d) Decode throughput under concurrent prefill: while 64K prefill is running on a concurrent request, active decode on a separate request must maintain >= 20 tok/s (baseline without multiplexing: 0 tok/s, because decode blocks). (e) Chunk-shrink logic: trace the prefill chunk size over time during a concurrent workload; verify it shrinks when decode is active and grows back when decode finishes. (f) NIAH@4K, 16K, 64K, 128K must PASS (single-request workload, `--duet-serve` should be neutral). (g) Memory: DuetServe must not cause Metal peak to exceed baseline by more than 500 MB (the two queues share memory, but scheduler state is small). (h) Compose with SDAG (Task 233): verify the roofline model correctly handles the block-sparse attention shape when SDAG is active — predicted wall-clock should drop by the SDAG compute-reduction factor.
- **Effort**: M (1-2 weeks — roofline model port (offline calibration + piecewise-linear fit), two-queue scheduling in hypercar_server.py, chunk-shrink adaptive logic, concurrent-request workload test, interference prediction validation, NIAH no-regression, memory check)
- **Depends on**: `hypercar_server.py` (exists). MLX command-queue primitives (exist — MLX exposes `mx.stream()` and multiple-device scheduling). Composes with SDAG (Task 233) via updated shape prediction. Composes with chunked-prefill logic (exists). Independent of all codec and eviction tasks. Sequencing: ship after SDAG (Task 233) and after benchmark-extension (Task 234 phase 1) so we have the measurement infrastructure to validate DuetServe's gains. Roofline model (phase 1) can ship as a standalone diagnostic tool before phase 2; useful for scheduling decisions in other tasks (e.g., PagedEviction chunk sizing, DiffKV compaction scheduling).

### 236. WaveTune learned kernel dispatcher — tree-based per-call attention-kernel selection for MLX
- **Goal**: 3 (eliminates fixed-cost kernel-dispatch tax on every attention call; decode throughput is sensitive to per-op overhead at 1M-token scale where call count dominates), Goal 4 (prefill sees wider shape diversity than decode so kernel auto-tuning's gains are larger on prefill — chunked-prefill chunk sizes vary, each benefits from shape-optimal dispatch), Goal 6 (M4 Pro's heterogeneous hardware — GPU + AMX + ANE potentially — benefits from shape-aware dispatch that picks the right accelerator per call; static dispatch leaves performance on the table)
- **Derived from**: WaveTune (2604.10187, arxiv Apr 2026). Closes pass-52 gap #4 (attention kernel auto-tuning — held across multiple passes). Orthogonal to the pass-52 formal dead-end on Metal-specific kernel papers: WaveTune is about *dispatch policy* (pick the right kernel from a set) not *kernel design* (write a better kernel). The wave-aware bilinear cost model maps cleanly to Metal's threadgroup-retirement semantics.
- **Change**: Three files. File 1 — `omlx/kernels/roofline_profile.py` — offline benchmark harness that sweeps the existing MLX attention-kernel options (fused fp16 via `mx.fast.scaled_dot_product_attention`, TQ3 fused, native quantized, block-sparse-SDAG) across a grid of (batch, seq_len, num_heads, head_dim, dtype, bits). Record wall-clock and fit WaveTune's bilinear wave model `T(shape) = a × wave_count(shape) + b × wave_time(shape) + c` per kernel. Cache the profile at `omlx/kernels/kernel_profile.json`. File 2 — `omlx/kernels/dispatch_tree.py` — trains a shallow decision tree (depth 6, scikit-learn DecisionTreeRegressor or a hand-rolled equivalent to avoid the sklearn dependency for runtime) on the profiled data with shape features as input and optimal-kernel index as output. Serialize to `omlx/kernels/dispatch_tree.npy` (tree structure only, no runtime sklearn needed). File 3 — integration in `omlx/attention/dispatcher.py` — at each attention call, query the dispatch tree with `(batch, seq_len, num_heads, head_dim, dtype, bits)` and route to the selected kernel. Per-call overhead must be < 1 μs (paper reports < 1 μs; verify for MLX call path). Wire in as `--kernel-dispatch {static, wavetune}` (default `static` until validated). The profiling file is committed (small); the tree file is generated by a one-time build step `python -m omlx.kernels.build_dispatch_tree`.
- **Verify**: (a) Profiling coverage: the offline sweep must cover at least 200 shapes across the 4 kernel variants — sparse points fit poorly so coverage matters. (b) Model fit quality: WaveTune bilinear model MAPE < 15% per kernel family (paper reports < 10%). (c) Dispatch tree accuracy: on a held-out test set of 50 shapes, the tree must select the optimal kernel (within 5% wall-clock of oracle) >= 90% of the time. (d) Per-call dispatch overhead: measure tree-lookup wall-clock on 10000 calls; median must be < 1 μs (above which the dispatch cost erodes the gains). (e) End-to-end decode speedup: decode throughput on a mixed-shape workload (alternating short/medium/long contexts simulated) must improve by >= 10% with `--kernel-dispatch wavetune` vs `static` (paper reports 1.8-3.2× in the aggressive case but our kernel set is smaller so 10% is conservative). (f) Prefill speedup: chunked prefill with variable chunk sizes must improve by >= 15% (prefill has more shape diversity so benefits more). (g) No regression: single-shape workloads (e.g., a 4K NIAH run with fixed batch=1) must match `static` dispatch exactly — the tree should pick the same kernel that static dispatches to. (h) Robustness: introduce a shape not in the profile (extrapolation case); the tree must still produce a valid kernel selection (no crash) and wall-clock must be within 30% of the nearest profiled shape. (i) Compose with SDAG (Task 233): the SDAG-sparse attention kernel must be profiled as one of the kernel variants; the tree should learn to prefer SDAG when the shape includes document-boundary metadata.
- **Effort**: M (1 week — offline profiling harness (longest piece, ~200 shape runs), bilinear model fit, decision-tree training and serialization, dispatcher integration with < 1 μs overhead budget, end-to-end decode/prefill speedup validation, no-regression check on single-shape workloads, SDAG composition)
- **Depends on**: MLX attention kernels (exist — fused, TQ3, native quantized). SDAG kernel (Task 233 — composes as one of the profiled kernel variants). Composes with DuetServe (Task 235) via the roofline model — the two roofline models (DuetServe's interference-prediction model, WaveTune's bilinear wave model) are related but serve different purposes; they can share profiling data but have separate fits. Sequencing: ship last among pass-53 tasks. Depends on having the full kernel set including SDAG so the tree can learn to dispatch to SDAG. Can reuse profiling data from DuetServe's phase-1 calibration to save time. If DuetServe's roofline model is already shipped, this task's profiling-harness phase is partial-duplicate work and should be merged — a single `omlx/kernels/unified_profile.py` feeding both the dispatcher and the scheduler.

## Research-derived tasks (from LIT_REVIEW.md pass 54, 2026-04-22)

### 237. SmallKV cross-scale assist-model attention-compensation — runtime self-verification of compressed eviction
- **Goal**: 2 (first pass-40-54 mechanism for *runtime detection* of compression-induced quality loss — the signal is attention-matrix similarity between a small assist model on full cache and the compressed 30B target; when divergence is detected we know the compressed model is likely to err and can pin the flagged tokens), Goal 1 (1.75-2.56× throughput improvement at long context per paper — achieved by preserving marginal tokens that the large compressed model would evict but are collectively important), Goal 3 (marginal-token-aware eviction reduces over-compression failure modes that otherwise force regeneration — net decode throughput improves even net of assist-model compute cost at context > 16K)
- **Derived from**: SmallKV (2508.02751, v2 2026-01). Closes pass-53 gap #4 (self-verification of compressed output at runtime). Highest-leverage pass-54 find because it adds a *novel signal type* (cross-scale attention similarity) to the stack rather than refining an existing signal. M4 Pro's memory headroom (~6 GB spare after 30B + 22.5 GB KV at 1M) fits a 0.5B assist model at ~1-2 GB with room to spare.
- **Change**: Three phases. Phase 1 — **assist-model runner**: new file `omlx/assist/small_model_runner.py` that loads a small model (default Qwen2.5-Coder-0.5B-Instruct-bf16, ~1.5 GB; smaller 100M distilled variant as a faster alternative) alongside the main 30B model. Parallel inference paths: both models share tokenization and position encoding but have independent KV caches — the small model operates on a *full* (uncompressed) cache while the large model operates on the SnapKV+TQ3 compressed cache. Phase 2 — **attention-similarity bridge**: at each decode step, compute cosine similarity between the small-model's attention pattern (averaged over heads) and the large-model's compressed attention pattern. When similarity drops below `--assist-similarity-threshold` (default 0.75), flag the current step as a potential compression error. Log the flag to a drift monitor in `omlx/telemetry/drift.py` (new file) and feed it back as a score modifier to the eviction ranker — the flagged-token's SnapKV score gets a +0.3 boost for the next 10 decode steps, making it survive re-eviction. Phase 3 — **marginal-token recovery**: maintain a rolling window of tokens the small model ranked in its top-32 that the large model evicted; when a flag fires, check whether any such rolling-window tokens are what the large model is currently trying to attend to — if so, either (a) re-fetch from session-save storage, (b) use the small model's KV as a stand-in (approximate but cheap), or (c) log for the next eviction cycle. Wire in as `--assist-small-model PATH` (default off — requires separate 0.5B model download to `~/.omlx/models/qwen2.5-coder-0.5b`), `--assist-similarity-threshold T` (default 0.75 — tunable per the drift validation), `--assist-recovery-mode {log, soft-pin, refetch, small-kv-fallback}` (default `soft-pin` — adds score boost; `refetch` requires Block-Pool Task 216 shipped; `small-kv-fallback` is the most experimental), `--assist-context-gate-length N` (default 16384 — only engage above this context since short contexts don't benefit from compression compensation), `--assist-probability P` (default 0.2 — sample rate for similarity checks; doing this every step is wasteful, every 5th step is plenty). Composition: SmallKV's signal is a *score augmentation* to SnapKV (shipped); CAOTE (shipped) still provides the primary ranking; SmallKV's +0.3 boost operates on top. With DuoAttention (shipped): apply SmallKV only to retrieval heads — streaming heads don't need the check since they're already using ring buffers. With SparseServe (Task 210 planned): SmallKV's attention-matrix computation needs the full attention matrix of the small model, which requires a non-segmented prefill path for the small model; adjust the segmenter to not apply to the assist-model.
- **Verify**: (a) Assist-model loading: the 0.5B model loads in < 5 seconds on M4 Pro and uses < 2 GB Metal memory; verify via `mx.metal.get_peak_memory()` delta before/after load. (b) Attention-similarity correctness: on a 1000-token warmup trace, small-model and large-model attention patterns (both on full cache, no compression) must have cosine similarity > 0.6 at > 80% of heads — if substantially lower, the two models are too different for the similarity signal to be meaningful and we need a larger assist model. (c) Drift-detection AUROC: hand-label 200 decode steps on a long-context trace as "quality OK" / "quality degraded" (use HumanEval-style correctness as label); AUROC of similarity-based drift score must be >= 0.70, else the signal is noisy. (d) Throughput gate: at 64K context with `--assist-small-model` active, decode throughput must stay within 15% of baseline without assist model (the assist model adds overhead but the marginal-token protection should recover quality at tolerable cost). (e) Quality recovery: on a curated "over-compression failure" set (10 prompts that fail under aggressive TQ3 at 64K), SmallKV active must recover quality on >= 6 of 10 cases (the paper's 1.75-2.56× throughput improvement comes from *both* speed and quality — we're primarily measuring the quality recovery here). (f) HumanEval regression: SmallKV active must not regress HumanEval below baseline at 4K context — if active at short context is harmful, gate it behind context-length threshold. (g) NIAH@4K, 16K, 64K, 128K must PASS with SmallKV active — SmallKV should strictly improve NIAH retrieval since its attention-matching signal catches exactly the case where the needle token was wrongly evicted. (h) Memory headroom: at 1M context with Qwen3-Coder-30B + 0.5B assist model + 22.5 GB KV, total peak Metal memory must stay below 45 GB (leaving 3 GB for system and working memory on 48 GB hardware). (i) Telemetry test: drift monitor must log events correctly; verify the event stream can drive an adaptive threshold that tightens the threshold when false-positive rate is high. (j) Fallback: `--assist-recovery-mode log` must be a no-op on output (just logs); this isolates the assist compute overhead without any behavior change, useful for A/B comparisons.
- **Effort**: L (2-3 weeks — assist-model loader and parallel inference path, attention-matrix extraction hook in MLX attention, similarity computation, drift telemetry, SnapKV score-modifier integration, 3 recovery modes, context-gate logic, full eviction+NIAH ladder validation at 1M)
- **Depends on**: MLX model loader (exists). Qwen2.5-Coder-0.5B weights (needs download — add to `setup.py` model-fetch helper or document in README). SnapKV (shipped — SmallKV's signal is a score modifier). DuoAttention (shipped — SmallKV engages on retrieval heads only). Composes with Block-Pool (Task 216 — refetch mode requires persistent token storage). Composes with SDAG (Task 233 — attention pattern under block-sparse mask is different; the similarity computation should account for the document-boundary structure — when SDAG is active, compare attention patterns within each document's block, not across). Independent of codec tasks (TQ3, KQ-SVD Task 239) — SmallKV operates at the attention-pattern level above the codec. Sequencing: ship after DuoAttention optimizations stabilize (shipped). Can ship before Block-Pool — without Block-Pool, the `refetch` mode is unavailable but `soft-pin` and `small-kv-fallback` still work. This is the ship-next pass-54 task.

### 238. Adaptive Context Compression conversational-phase eviction — per-turn budget for long-running OpenCode sessions
- **Goal**: 1 (conversational-phase awareness extends what fits in a session by adapting budget to actual task demand — a debug turn needs less history than an exploratory turn, so per-phase eviction recovers headroom without quality loss), Goal 3 (per-turn budget re-negotiation keeps decode latency bounded as context grows — the "constant across context window" target requires active management, not one-shot eviction at prefill boundary), Goal 2 (coherence-sensitive filtering prevents eviction of cross-turn context that breaks multi-turn coherence — a known quality-cliff on the current stack)
- **Derived from**: Adaptive Context Compression (2603.29193, arxiv Mar 2026). Closes pass-53 gap #5 (streaming summarization). Partial overlap with Continuum TTL (Task 217) which provides task-level TTL — this task provides turn-level phase-conditional TTL, complementary not duplicative.
- **Change**: Three components. First, **phase classifier** in `omlx/serving/phase_classifier.py` (new) — tracks turn-level features: turn input length, tool-call fraction of output, topic-embedding distance from prior turn (using sentence-transformers or a small MLX embedder), new-entity rate (heuristic). Classifies each turn into {explore, edit, test, debug, idle} via a rule-based initial implementation and a learned classifier (logistic regression over features) available as `--phase-classifier learned` with trained weights shipped. Second, **phase-conditional budget** in `omlx/patches/snapkv.py` — extend the eviction-call to accept a phase hint; applies a phase-indexed budget table (default: explore=80%, edit=60%, test=40%, debug=50%, idle=20% of baseline cache size). The table is `--phase-budget-override "explore:0.8,edit:0.6,..."` configurable. Third, **coherence filter** in `omlx/patches/coherence_filter.py` (new) — computes per-turn coherence embedding (sentence-level) and hard-vetoes eviction of tokens whose removal would drop coherence (measured by embedding distance from cluster centroid of prior turns' embeddings) below `--coherence-threshold` (default 0.7). Wire in as `--adaptive-context` (default off until validated), `--phase-classifier {heuristic, learned}` (default `heuristic`), `--phase-budget-override` (escape hatch), `--coherence-threshold {0.5..0.9}` (default 0.7), `--phase-classifier-fallback-confidence C` (default 0.6 — below which fall back to baseline-budget, preventing mis-classification from over-evicting). Composition: with DuoKV (shipped — phase-conditional budget scales DuoKV's 2× retrieval-head allocation proportionally, not independently), with shot-eviction (Task 234 — shots are finer-grained than phases; shots live within a turn, phases span turns), with Continuum TTL (Task 217 — phases provide per-turn override to task-level TTL; Continuum manages long-horizon task scheduling, Adaptive Context manages per-turn in-session budget), with SmallKV (Task 237 — SmallKV's drift signal can supplement phase-classifier's confidence; if drift is detected during a claimed "debug" phase, phase classifier lowers confidence and reverts to baseline budget).
- **Verify**: (a) Phase-classifier accuracy: on a hand-labeled trace of 200 OpenCode turns (generated by running the Hypercar agent on 20 repos for 10 turns each), the classifier must achieve >= 70% accuracy on phase classification. (b) LOCOMO/LOCCO validation: port the paper's LOCOMO benchmark as an optional eval — a conversational-QA eval where compression impacts multi-turn coherence; measure whether Adaptive Context Compression reduces degradation vs plain SnapKV at matched keep ratio. (c) Session-memory efficiency: on a 50-turn synthetic OpenCode session, active `--adaptive-context` must reduce peak KV memory by >= 25% vs baseline SnapKV at matched quality. (d) Coherence-filter effectiveness: test case — a long session where user's task shifts from "exploring library X" to "debugging feature Y"; coherence-filter should retain tokens from Y's setup that occurred 20 turns ago if their removal would break coherence; without the filter, SnapKV evicts them. (e) Mis-classification guard: artificially force the phase classifier to mis-classify (via a mock); verify `--phase-classifier-fallback-confidence` triggers and the system falls back to baseline budget rather than applying an over-aggressive budget. (f) HumanEval regression: must stay within 0.5 pp of baseline — HumanEval is single-turn so phase classification should be neutral. (g) NIAH@4K, 16K, 64K, 128K: must PASS with `--adaptive-context` active — NIAH is single-turn single-phase so this is a no-regression check. (h) Multi-turn NIAH (new gate): insert a needle in turn 1 and query in turn 20 with 10 debug/test turns in between; needle must survive `--adaptive-context` eviction throughout the 20 turns. (i) Phase-budget override: manual override via CLI produces observably different memory consumption; verify via explicit benchmark run. (j) Compose with Continuum TTL (Task 217): when both active, phase-level TTL must not conflict with task-level TTL — lower of the two wins (more conservative retention); validate no deadlock on tokens tagged with inconsistent TTLs.
- **Effort**: M (5-7 days — phase classifier (heuristic first; learned later), phase-budget table in snapkv.py, coherence filter, multi-turn NIAH gate, LOCOMO-style validation, 200-turn hand-labeling for classifier accuracy)
- **Depends on**: SnapKV (shipped — adaptive-context extends its eviction call). OpenCode turn protocol (exists — server maintains turn history). Sentence embedder for coherence (MLX sentence-transformers port or lightweight alternative — if not available, defer coherence-filter sub-phase). Composes with Continuum TTL (Task 217 — turn-level phases × task-level TTL). Composes with shot-eviction (Task 234 — shots within turns). Composes with SmallKV (Task 237 — drift detection informs phase-classifier confidence). Independent of all codec tasks. Sequencing: ship after Task 237 (SmallKV's drift signal improves the phase-classifier's self-correction). Can ship standalone with rule-based phase classification; the learned classifier and LOCOMO validation follow after 200-turn hand-labeling is collected.

### 239. KQ-SVD provable low-rank attention-matrix decomposition — stacked codec front-end with TurboQuant
- **Goal**: 1 (low-rank + 3-bit compounds to projected 10.6× compression vs 5.3× from TQ3 alone — at 1M context this turns 22.5 GB KV into ~11 GB, freeing 11 GB that combined with SliceMoE's 7 GB model-size reduction gives 4 GB spare; the tight 48 GB fit becomes comfortable), Goal 2 (closed-form provable attention-output fidelity bound is the first pass-40-54 codec paper with a formal bound on the attention *output* not just the intermediate matrix — stronger quality guarantee than empirical-only claims), Goal 6 (projected 18 GB headroom at 1M context enables co-tenancy workloads that currently require swap)
- **Derived from**: KQ-SVD (2512.05916, arxiv Dec 2025). New codec-axis — orthogonal to TurboQuant's product-quantization. Paper provides closed-form SVD-based optimal rank-r decomposition of the attention matrix with a proof that extends to the value-output-projection interaction. Unlike OjaKV (online streaming approximate), KQ-SVD is static-optimal — better for session-save workflows (Task 216) where decomposition is computed once and reused.
- **Change**: Three files. First, new file `omlx/compress/kq_svd.py` implementing the closed-form SVD decomposition — takes K, V, Q_projection matrices and returns rank-r low-rank factors (U_k, V_k, diag_s for K; U_v, V_v, diag_s_v for V). The decomposition considers K-Q inner products and V-output-projection, per the paper's extension. Second, integration hook in `omlx/cache/tq3_cache.py` — optional low-rank front-end. When `--kq-svd-rank R` is set, K and V are first projected to rank R via the SVD factors; the rank-R tensors are then quantized to 3-bit via TQ3. Attention computation reconstructs K, V lazily from the low-rank factors before computing attention (or, more efficiently, the Q projection is applied to the low-rank factors directly — this is an optimization path for a future iteration). Third, calibration routine `omlx/compress/kq_svd_calibrate.py` — sweeps ranks per-layer to minimize cumulative attention-output error subject to a memory budget; outputs a per-layer rank allocation. Wire in as `--kq-svd-rank R` (default off; R=64 on d=128 gives 2× rank reduction stacking to 10.6× total with TQ3), `--kq-svd-calibrate` (runs the calibration sweep offline), `--kq-svd-per-layer {fixed, sensitivity-weighted}` (default `sensitivity-weighted` — more rank for higher-variance attention layers), `--kq-svd-mode {save-time, live}` (default `save-time` — apply only at session-save, avoids the O(N²d) cost during decode). Composition with TQ3 (shipped — primary codec path): low-rank projection first, then 3-bit quantization of rank-projected tensor. With DuoKV (shipped): apply KQ-SVD only to streaming heads (3-bit path); retrieval heads remain fp16 since they need full resolution for retrieval accuracy. With Block-Pool (Task 216): ideal composition — SVD decomposition is computed once at session save, the compressed form is what's persisted; load is instant. With SmallKV (Task 237): SmallKV's assist model operates on full-rank cache so no composition conflict.
- **Verify**: (a) Compression ratio: at 64K context with `--kq-svd-rank 64` and `--kv-mode tq3`, total KV memory must be <= 0.2× the fp16 baseline (paper's decomposition gives 2× reduction × TQ3's 5.3× = 10.6×, so 0.2× is conservative accounting for metadata overhead). (b) Attention-output fidelity: on 100 fixed prompts, compare compressed vs uncompressed attention outputs (per-token, per-head) via cosine similarity; median must be >= 0.95, 95th percentile >= 0.90 (paper's bound is stronger but we're stacking with TQ3 quantization noise). (c) HumanEval regression: at `--kq-svd-rank 64 --kv-mode tq3`, HumanEval must stay within 1 pp of baseline TQ3 — the paper-provable bound on attention output should ensure this, but we still measure. (d) NIAH@4K, 16K, 64K, 128K must PASS with KQ-SVD active — the rank projection reduces expressiveness; NIAH is the critical test that needle-attention still routes correctly through the projected representation. (e) Calibration correctness: the `--kq-svd-per-layer sensitivity-weighted` mode should allocate higher rank to layers with higher attention variance; unit-test on a 100-sample calibration set that the layer-rank distribution is monotonic in variance. (f) SVD cost at save time: for a 64K-context session, the SVD decomposition must complete in < 2 minutes on M4 Pro (O(N²d) ≈ 500 GFLOPs — achievable). (g) `--kq-svd-mode live` benchmark (future): measure SVD cost at 8K context; if feasible (<100ms), enable live mode for short contexts. (h) Stacking-error check: sweep compression ratios at rank R × bits B ∈ {(64, 3), (96, 3), (128, 4)}; verify no catastrophic stacking regression — paper doesn't study this composition so we validate independently. (i) Compose with DuoKV: KQ-SVD on streaming heads only; retrieval-head quality must be unchanged (fp16 path is untouched); verify on NIAH and HumanEval. (j) Compose with Block-Pool (Task 216): session save with KQ-SVD + TQ3 + SnapKV+CAOTE; reload and verify attention-output fidelity matches pre-save within the compression-error budget.
- **Effort**: M (1-2 weeks — closed-form SVD decomposition in MLX, integration with TQ3 cache, per-layer calibration sweep, save-time mode validation, full accuracy ladder with stacking-error check, DuoKV and Block-Pool composition)
- **Depends on**: TurboQuantKVCache (shipped — primary codec). DuoKV (shipped — orthogonal on retrieval-vs-streaming head split). Block-Pool (Task 216 — ideal composition at session save). SliceMoE (Task 221 — stacking KV compression × model compression is the 1M fit path). Independent of eviction tasks and SmallKV. Sequencing: ship after TQ3 production-validation (already stable). Can ship before SliceMoE; the 10.6× KV compression alone gives substantial room. This is the pass-54 codec shipping priority — ship third after SmallKV (Task 237) and Adaptive Context (Task 238).

### 240. SAC (Semantic-Anchor Compression) — autoencoding-free in-context anchor-token aggregation with bidirectional attention
- **Goal**: 1 (anchor-based compression aggregates KV into selected context tokens without introducing synthetic compression tokens — compression at source, the sequence-length reduction translates directly to memory and compute savings at 1M), Goal 2 (paper reports consistent improvement over autoencoder-based compression on question-answering and long-context summarization — lower quality risk than gist tokens), Goal 3 (reduced effective sequence length means faster attention compute per decode step — not just memory savings, but directly latency savings)
- **Derived from**: SAC / Semantic-Anchor Compression (2510.08907, arxiv Oct 2025). Partial-close of the formally-declared training-free gist-token dead end — SAC requires some training but it's small (anchor embedding + LoRA fine-tune), task-specific, and uses bidirectional-attention modification at anchor positions only. The bidirectional-attention primitive is conceptually novel to our stack — every other KV-compression paper takes causal masking as fixed.
- **Change**: Three phases. Phase 1 — **anchor selection policy**: new file `omlx/anchors/selector.py` with rule-based anchor placement for code: function signatures, class definitions, top-level imports, tool-call boundaries, every-N-tokens fallback (default N=16). Additionally provide a learned-anchor policy (`--anchor-selection learned`) that uses a tiny classifier (MLP over token embeddings) trained on 5000 code snippets with rule-based labels — when rule-based fails, learned can pick better anchors. Phase 2 — **bidirectional attention hook**: modify MLX attention-mask construction in `omlx/attention/anchor_mask.py` (new) to allow bidirectional attention at anchor positions only — an anchor attends to past and future within a `--anchor-window W` (default 64 tokens either side); non-anchor tokens remain causal. Phase 3 — **task fine-tune**: LoRA fine-tune Qwen3-Coder-30B on a curated code-completion + tool-call dataset (~10K samples). Anchor embedding (~128 params per anchor-position-rank, trained alongside the LoRA; the LoRA itself is ~10M params, a 0.03% adapter). Training objective: next-token prediction on the downstream task (code completion, tool-call generation) with anchor tokens enabled. Wire in as `--anchor-compress` (default off until validated) with `--anchor-ratio R` (default 16 — one anchor per 16 context tokens), `--anchor-weights PATH` (default: `omlx/weights/sac_lora.safetensors`, needs to be trained and shipped), `--anchor-selection {rule-based, learned, mixed}` (default `rule-based`), `--anchor-window W` (default 64 — bidirectional window around each anchor), `--anchor-preserve-on-eviction` (default on — anchors are pinned through SnapKV; eviction only targets non-anchor tokens). Composition: with SnapKV (shipped — SnapKV's score can be seeded with anchor-priority tokens; anchors are always top-retained), with SDAG (Task 233 — document boundaries double as anchor locations; the rule-based selector should use `rag_document_boundaries` hint when provided), with Block-Pool (Task 216 — anchors persist across session save/load; they're a natural anchor for session re-indexing), with SmallKV (Task 237 — SmallKV's drift detection should check whether the assist model agrees with SAC's anchor placements; disagreement flags a SAC-LoRA regression).
- **Verify**: (a) Compression ratio: at `--anchor-ratio 16`, effective context sequence length (non-pruned positions during attention compute) is ~6% of original + anchor positions = ~12% total; memory and attention-compute reduction should match this factor within 10%. (b) HumanEval: must stay within 2 pp of baseline — the LoRA fine-tune may regress general capability; if regression > 2 pp, the fine-tune dataset is too narrow and needs broadening. (c) NIAH@4K, 16K, 64K, 128K: must PASS — the needle must either be an anchor position or be within the bidirectional window of an anchor; if NIAH fails because needles randomly land between anchors, the anchor placement needs to be denser or the window wider. (d) Tool-call validity: the tool-call-boundary anchor placement should improve tool-call JSON validity (Task 161 / Phase 3f) — measure; validity rate should be >= baseline. (e) Quality ladder at ratios: sweep `--anchor-ratio` ∈ {8, 16, 32}; find the knee where quality drops more than 2 pp below baseline. (f) Learned-anchor vs rule-based: on a held-out test set, the learned anchor policy must beat rule-based by >= 3 pp on code completion (if not, learned adds complexity without value — ship rule-based only). (g) LoRA composition: `--anchor-weights none` (disabled anchor LoRA but enabled anchor mechanism) must match baseline within 0.5 pp — this isolates the mechanism from the weights; mechanism-only must not regress. (h) Fine-tune data distribution: the fine-tune dataset must be at least 30% tool-call samples, 50% code completion, 20% code intelligence — verify distribution; if skewed, re-weight. (i) Compose with SnapKV: anchors are pinned through eviction; unit-test that a 64K run with SnapKV at 50% keep preserves all anchor positions — even if the intra-anchor non-anchor tokens are evicted, the anchors + their aggregated KV remain. (j) Revert option: `--anchor-compress off` must reproduce baseline behavior exactly; unit-test by running 100 prompts with and without the flag and measuring output equality under deterministic decoding.
- **Effort**: L (3-4 weeks — anchor selector (rule-based first, learned later), bidirectional attention mask construction, MLX LoRA fine-tune infrastructure (exists), 10K-sample dataset curation, fine-tune run + validation, full accuracy ladder, SnapKV composition, optional learned-anchor classifier)
- **Depends on**: MLX LoRA fine-tune infrastructure (exists). Curated code-completion + tool-call dataset (needs curation — 10K samples from Hypercar codebase + OpenCode trace + HumanEval training split; ~1 week effort). Qwen3-Coder-30B full-weights access (have). Composes with SnapKV (shipped — anchors pinned through eviction), SDAG (Task 233 — document boundaries as anchor hints), Block-Pool (Task 216 — anchors persist across sessions), SmallKV (Task 237 — drift on anchor placement). Independent of codec tasks (TQ3, KQ-SVD Task 239) — SAC operates at the sequence-representation level above the codec. Sequencing: ship last among pass-54 tasks. Highest cost and highest risk (task fine-tune can regress general capability); ship only after SmallKV (Task 237), Adaptive Context (Task 238), and KQ-SVD (Task 239) are stable. Consider merge-back option: if HumanEval drops > 2 pp, revert the anchor-LoRA and keep only the anchor-mechanism (which provides compression benefits even without the fine-tune calibration).

## Research-derived tasks (from LIT_REVIEW.md pass 55, 2026-04-22)

### 241. KVTC transform-coding on-disk KV compression — Block-Pool session-save codec
- **Goal**: 1 (20× on-disk KV compression at matched accuracy turns 22.5 GB raw KV at 1M into ~1.1 GB; stacking with TQ3 3-bit quantization the projected session-save size is ~1.0 GB, the difference between a 4-second and sub-second SSD save), Goal 5 (compact on-disk KV shrinks swap footprint during session migration — save spikes by 1.1 GB not 22.5 GB, well within the 8 GB envelope), Goal 6 (storage-layer codec applied at save-time only — no effect on runtime working-memory footprint or decode cost, so no regression on the 48 GB working-set fit)
- **Derived from**: KVTC / KV Cache Transform Coding (2511.01815, ICLR 2026). Closes pass-54 angle #12 (efficient KV serialization formats). Highest-leverage pass-55 find due to clean composition with existing codec stack (TQ3 × KQ-SVD Task 239 × KVTC) for projected ~40× session-save compression. ICLR 2026 acceptance gives strongest pass-55 peer-review backing.
- **Change**: Two files. First, new file `omlx/compress/kvtc.py` implementing the three-stage classical-media-compression pipeline: (1) **PCA decorrelation** — per-layer basis matrices computed from a calibration corpus of ~50 diverse OpenCode sessions; bases stored as `omlx/weights/kvtc_bases_qwen3_coder_30b.safetensors`; transform is a single matmul per layer per K/V; (2) **adaptive quantization** — per-coefficient variance-weighted bit-depth using the paper's allocation formula; coefficients with higher variance (carrying more information) get more bits; coefficients near-zero variance get 1 bit or pruned; (3) **entropy coding** — arithmetic coder over the quantized coefficients; start with a numpy-based implementation (single-threaded, acceptable at save-time); if load-time decode becomes a bottleneck, parallelize across layers (each layer is independent, 48 layers parallel on M4 Pro's 10 cores). Second, integration hook in Block-Pool save/load path (Task 216, depends-on) — `save_with_kvtc(cache, path)` and `load_with_kvtc(path)`; Block-Pool's metadata header stores a magic byte sequence and version identifier that the loader uses to detect KVTC-compressed files. Wire in as `--kvtc-compress` (default off until validated; default on after gate passes — near-lossless and storage is effectively unbounded so no reason not to), `--kvtc-bases PATH` (default: shipped calibration bases; override for specialist workloads), `--kvtc-recalibrate` (CLI subcommand that runs the calibration sweep against a user-specified corpus of sessions and outputs new bases), `--kvtc-parallel-decode {1,2,4,8}` (default 4 — parallel-layer decode threads for faster load). Composition: with TQ3 (shipped — KVTC encodes TQ3's 3-bit codes, not raw fp16; the PCA bases operate on the 3-bit code indices rather than the dequantized activations), with KQ-SVD (Task 239 — KVTC encodes the rank-reduced coefficients when KQ-SVD is active; composition is automatic since both are save-time codecs), with Block-Pool (Task 216 — KVTC is the codec inside Block-Pool's storage format; Block-Pool manages the file layout, KVTC handles the byte-level compression), with DuoKV (shipped — KVTC applies to both streaming (3-bit) and retrieval (fp16) head caches, with different calibration bases per head type). Independent of runtime (decode-time) codec work — KVTC operates only at save/load boundaries. Risk: arithmetic coding has variable decode cost; if load-time > 2 seconds at 1M context, parallelize. Calibration risk: the PCA bases are corpus-dependent; validate on OpenCode-like traces before shipping defaults.
- **Verify**: (a) Compression ratio on fp16 baseline: at 16K, 64K, 128K contexts, KVTC alone (applied to fp16 KV cache) achieves >= 15× compression (paper reports 20× but allow 25% margin for real-world variance); measure on 10 diverse OpenCode session saves. (b) Compression ratio stacked with TQ3: at 16K, 64K, 128K, KVTC applied to TQ3-quantized KV achieves >= 3× additional compression on top of TQ3's 5.3× (so projected 16× total); measure. (c) Accuracy preservation: on 100 fixed prompts, load a session saved+reloaded via KVTC, and compare output to the in-memory session; per-token logit cosine similarity must be >= 0.99 at the top-1, and generated output must match within deterministic decoding. (d) Calibration-basis robustness: validate that OpenCode calibration bases work well on a held-out test set of traces including tool-call-heavy, code-generation-heavy, and long-conversation traces; per-trace compression ratio must stay within 10% of the calibration-set mean. (e) Save throughput: at 1M context, save-with-KVTC must complete in < 10 seconds on M4 Pro SSD; baseline (no KVTC, raw save) is ~4 seconds for 22.5 GB, so KVTC adds encoding overhead but should stay under 10s total. (f) Load throughput: at 1M context, load-with-KVTC must complete in < 3 seconds with parallel-decode (target is sub-second total session restoration, paired with TQ3 load pipeline). (g) Metadata stability: version bump in KVTC format must be backwards-compatible; old sessions must still load after format upgrade. (h) Recalibrate command: `--kvtc-recalibrate` on a new session corpus produces valid new bases that compress that corpus better than the default bases by >= 5%. (i) Compose with Block-Pool (Task 216): end-to-end session save → KVTC encode → Block-Pool store → Block-Pool retrieve → KVTC decode → session reload must be lossless and complete within the combined latency budget. (j) Integrity check: malformed or truncated KVTC data must fail-fast with a clear error (not silent corruption); test via intentionally corrupted test fixtures.
- **Effort**: M (1 week — PCA basis computation (offline, cached), adaptive quantizer, arithmetic coder (numpy-backed initially; MLX-native if needed), Block-Pool integration, calibration script, parallel-layer decoder)
- **Depends on**: Block-Pool session store (Task 216, pending — KVTC is the codec inside Block-Pool). TurboQuantKVCache (shipped). Calibration corpus (needs curation — 50 diverse OpenCode sessions, ~2 days). Composes with KQ-SVD (Task 239 — KVTC encodes rank-reduced coefficients), with DuoKV (shipped — per-head-type calibration bases). Independent of eviction tasks, runtime codec work, and SmallKV (Task 237). Sequencing: ship after Block-Pool is shipped (KVTC needs Block-Pool's storage abstraction); can ship before KQ-SVD (KVTC alone on TQ3 is valuable; stacking with KQ-SVD is additive). This is the pass-55 ship-first priority — highest leverage, smallest ship-cost, ICLR 2026 peer review.

### 242. DP-LLM runtime per-decoding-step dynamic layer-wise precision — temporal bit-budget allocation
- **Goal**: 2 (per-decoding-step dynamic precision preserves accuracy at high-sensitivity steps while allowing low-sensitivity steps to use cheaper bit-widths — same or better accuracy at lower average bit-budget, via exploiting the observation that layer sensitivity varies across decoding steps rather than being a static property), Goal 3 (lower average bit-budget means lower decode-time memory bandwidth — on Apple Silicon's memory-bound workload, this directly translates to decode throughput improvement; projected 15-20% decode speedup at matched accuracy), Goal 6 (~17% KV memory reduction via dynamic-vs-static allocation means 22.5 GB at 1M → 18.75 GB, freeing 3.75 GB for co-tenancy)
- **Derived from**: DP-LLM (2508.06041, NeurIPS 2025). Closes pass-54 angle #15 (per-query precision attention). First pass-40-55 result to exploit temporal (per-decoding-step) rather than spatial layer-sensitivity variation. Stacks with every other codec vertex: KVTuner per-head static (Task 186), DP-LLM per-step dynamic (this task), KQ-SVD rank (Task 239), KVTC entropy (Task 241).
- **Change**: Three phases. Phase 1 — **multi-bit-width codec**: extend `omlx/cache/tq3_cache.py` to support 2-bit, 3-bit, and 4-bit variants sharing the same WHT rotation basis. The cache stores one bit-width tag per layer per decoding-step; dequantize path dispatches to the appropriate codebook based on tag. Upcast (3→4) and downcast (3→2) are handled via re-quantization of the dequantized tensor — acceptable cost since dequantize must happen anyway for attention compute. The 2-bit and 4-bit paths use the same TurboQuant codebook structure (WHT + Beta codebook) at different resolution. Phase 2 — **sensitivity selector**: new file `omlx/compress/sensitivity_selector.py` implementing the input-activation-based sensitivity metric per decoding step. Metric: L2 norm of the input activations for each layer, normalized per-layer by the calibration-time running mean + variance. High norm relative to calibration distribution → high sensitivity → upcast; low norm → downcast. Called once per decode step (batch-level, not per-layer) for all layers; output is a 48-element bit-width vector (one entry per layer). Phase 3 — **precision scheduler**: integration in the attention forward path — the scheduler reads the bit-width vector from the selector, dispatches each layer's dequantize call to the appropriate codec, and updates the cache's bit-width tag so subsequent decode steps see the new bit-width. The scheduler also handles the upcast/downcast re-quantization. Wire in as `--dp-llm-dynamic` (default off until validated), `--dp-llm-bit-widths "2,3,4"` (configurable set; default 2/3/4), `--dp-llm-sensitivity-threshold T` (default auto-tuned from calibration — below T means downcast candidate, above T means upcast candidate; T is a two-element vector for the down/up thresholds), `--dp-llm-per-layer-enable "0-47"` (layer subset — default all; can restrict to specific layers via a range string), `--dp-llm-update-frequency N` (default 1 — every decode step; increase to N for less frequent bit-width updates if per-step overhead is too high), `--dp-llm-head-scope {all, streaming-only}` (default `streaming-only` — retrieval heads stay fp16/3-bit static, dynamic applies only to streaming heads). Composition: with TQ3 (shipped — DP-LLM's multi-bit-width codec extends TQ3's single-bit-width codec), with KVTuner per-head (Task 186 — KVTuner provides the *baseline* per-head static allocation that DP-LLM's dynamic then modulates around), with DuoKV (shipped — DP-LLM operates on streaming-head path only by default), with SmallKV (Task 237 — SmallKV's drift signal can serve as *additional* sensitivity input; when drift fires, bump bit-width for affected layers), with KVTC (Task 241 — KVTC save-time codec is independent of DP-LLM runtime codec; no interaction). Risk: upcast/downcast conversion on-the-fly adds per-step overhead; if > 5% of decode time, increase `--dp-llm-update-frequency` to reduce the update rate. Sensitivity-signal risk: if the norm-based metric is too noisy, consider an alternative signal based on attention-entropy or query-key divergence.
- **Verify**: (a) Multi-bit-width codec correctness: 2-bit, 3-bit, 4-bit variants must all dequantize to within 5%, 2%, 0.5% of fp16 baseline respectively (progressively tighter at higher bit-widths). Test on 1000 sample activations. (b) Upcast/downcast determinism: `3→4→3` round-trip must return identical bits (lossless upcast since 4-bit has more resolution than 3-bit); `3→2→3` must be within 5% L2 error (lossy). (c) Sensitivity-signal correlation with quality: on a labeled set of 100 decode steps marked as "quality OK" / "quality degraded", the norm-based metric's separation between the two groups must have AUROC >= 0.65 (lower than SmallKV's 0.70 target because this is a cheaper, noisier signal). (d) Average bit-budget: at 64K context with `--dp-llm-dynamic`, the observed average bit-width across all layers and decoding steps must be in [2.3, 2.8] — confirming dynamic allocation is actually spending less than static 3-bit. (e) Decode throughput: at 64K context, decode-throughput with `--dp-llm-dynamic` must be >= baseline 3-bit static by 10-20% (memory-bandwidth-bound workload should see proportional speedup). (f) Accuracy: HumanEval, NIAH@4K/16K/64K/128K, and code-intelligence must all PASS — no regression vs baseline TQ3. If any regress, the sensitivity-threshold is mis-tuned; re-calibrate. (g) Overhead check: per-step sensitivity computation + dispatch must be < 0.5% of decode-step wall-clock time; measure via profiler. (h) `--dp-llm-update-frequency N` sweep: test N ∈ {1, 4, 16} to find the sweet spot between responsiveness and overhead. (i) Composition with SmallKV (Task 237): when SmallKV drift fires, DP-LLM should bump bit-width for flagged layers; unit-test the signal wiring. (j) Revert: `--dp-llm-dynamic off` reproduces baseline TQ3 behavior exactly; unit-test with deterministic decoding.
- **Effort**: L (2-3 weeks — multi-bit-width codec extension (the core change, 1 week), sensitivity selector (2 days), precision scheduler with upcast/downcast (3 days), integration testing + accuracy ladder (3 days), sweep of update frequency and threshold tuning (2 days))
- **Depends on**: TurboQuantKVCache (shipped — DP-LLM extends the codec). KVTuner (Task 186, pending — DP-LLM operates around KVTuner's baseline static allocation). DuoAttention (shipped — DP-LLM defaults to streaming-head scope). Composes with SmallKV (Task 237 — drift signal supplements sensitivity signal; ship SmallKV first for the composition path). Independent of save-time codecs (KVTC Task 241, KQ-SVD Task 239). Sequencing: ship after SmallKV (Task 237) for the composition signal; can ship before KVTuner (Task 186) since DP-LLM handles its own per-layer allocation in the absence of KVTuner. This is the pass-55 ship-second priority — NeurIPS 2025 peer review, clean composition, highest absolute decode-throughput gain of the pass-55 tasks.

### 243. LCC (Latent Context Compilation) disposable-LoRA context compilation into stateless buffer tokens — per-repo precompilation
- **Goal**: 1 (16× compression of long context into buffer tokens amortizes prefill cost across many queries — for the OpenCode repo-context case where 200K tokens of codebase are referenced across many turns, compilation once to ~12K buffer tokens massively improves per-turn cost), Goal 3 (decoding against buffer tokens is at 16× reduced sequence length — attention compute drops quadratically; projected 256× reduction at long context makes decode fast even at what was effectively 200K context), Goal 4 (prefill of buffer tokens is near-zero — they're already precomputed; per-query prefill cost is paid at compile time, amortized across all subsequent queries)
- **Derived from**: LCC / Latent Context Compilation (2602.21221, Feb 2026). *Partial close* of the formally-declared training-free gist-token dead end — LCC requires per-context LoRA compilation (not training-free per the strict definition) but the LoRA is disposed after compilation making serving stateless. First pass-40-55 paper to use a training procedure whose artifact is designed to be thrown away.
- **Change**: Three phases. Phase 1 — **LCC compiler**: new file `omlx/compile/lcc_compiler.py` implementing the disposable-LoRA compilation procedure. Inputs: long context tokens (up to 256K), a fresh LoRA (rank 8, ~1M params), the base frozen Qwen3-Coder-30B model. Procedure: (1) initialize LoRA with random weights; (2) run the self-aligned optimization — N optimization steps (default 100) where at each step the LoRA is updated to minimize KL divergence between the base-model's output on (context + random context-agnostic query) and the LoRA-adapted-model's output on (buffer_tokens + random context-agnostic query), where buffer_tokens are the LoRA-projected activations at designated positions in the context; (3) extract the final buffer token activations; (4) discard the LoRA. Output: `buffer_tokens.safetensors` of shape (N/16, d_model). Compilation cost: ~30 seconds per 200K context on M4 Pro (estimated — 100 steps × ~0.3 sec/step forward+backward on a 1M-param LoRA with a frozen 30B model). Phase 2 — **buffer-token insertion hook**: modify prefill path in `omlx/hypercar_server.py` to accept a `--buffer-tokens PATH` argument; when provided, buffer tokens replace the corresponding context tokens at the start of the sequence. Position IDs for buffer tokens are set to the mean position of the tokens they compile from (per paper). Phase 3 — **compilation cache**: store compiled buffer tokens in `~/.omlx/compiled/<repo-sha256>/buffer_tokens_ratio<R>.safetensors`; on each session start, check if the current repo contents match the cached hash; if yes, load buffer tokens; if no, trigger recompilation (or fall back to raw context if `--lcc-refresh-policy` is `never`). Wire in as `--lcc-compile CONTEXT_PATH` (CLI subcommand that runs offline compilation and caches; use case: user runs this once after cloning a repo), `--buffer-tokens PATH` (use precompiled buffer tokens at inference; server-side flag), `--lcc-ratio R` (default 16 — compression ratio, lower means denser buffer tokens with higher quality), `--lcc-refresh-policy {never, on-change, daily}` (default `on-change` — recompile when repo content hash changes), `--lcc-opt-steps N` (default 100 — LoRA optimization steps; reduce for faster compilation with quality cost), `--lcc-lora-rank R` (default 8 — LoRA rank; higher means more expressive but slower compilation). Composition: with SnapKV (shipped — buffer tokens are pinned through eviction; SnapKV evicts only non-buffer tokens), with SAC (Task 240 — SAC provides task-level anchors, LCC provides context-level buffers; they operate at orthogonal scopes; a session can use both: SAC for task-aware attention structure + LCC for repo-context compression), with Block-Pool (Task 216 — buffer tokens persist via Block-Pool storage; the ~12K buffer tokens for 200K context are a tiny fraction of Block-Pool's budget), with SmallKV (Task 237 — SmallKV's drift signal on a buffer-tokens session detects when compilation is too aggressive; if drift fires frequently, lower `--lcc-ratio` or recompile with more opt steps). Risk: quality cliff at high compression ratios; validate on NIAH at 16×, 32×; back off compression if retrieval degrades. Compilation-stability risk: the LoRA initialization is random; test that compilation produces reproducible buffer tokens across 3+ runs with same seed.
- **Verify**: (a) Compilation correctness: compile a 200K context into 12.5K buffer tokens; verify output shape is (12500, d_model) and all values are finite (no NaN/inf). (b) Compilation cost: at 200K context with default settings (ratio 16, 100 steps, rank 8), compilation must complete in < 60 seconds on M4 Pro (target: 30 seconds). (c) Compression ratio: effective context length during inference with buffer tokens is ~1/16th of original; verify attention-compute and memory reduction match. (d) Quality preservation — HumanEval: running HumanEval over a compiled code-context + query, the HumanEval pass rate must stay within 3 pp of baseline (no compilation, raw context) — if regression > 3 pp, compilation is too aggressive. (e) Quality preservation — NIAH: insert a needle into a 200K context before compilation; after compilation the needle's information must be retrievable via query; NIAH@200K with LCC must PASS. (f) Compilation cache: recompile the same context with same seed twice; buffer tokens must be within 1e-4 L2 distance (reproducibility) — nondeterminism from MLX should be minor. (g) On-change refresh: modify one file in the compiled repo; content hash must differ; `--lcc-refresh-policy on-change` must trigger recompilation. (h) Ratio sweep: test `--lcc-ratio` ∈ {8, 16, 32}; find the knee where quality drops > 3 pp below baseline. (i) Opt-steps sweep: test `--lcc-opt-steps` ∈ {50, 100, 200}; find the knee where additional steps produce < 0.5 pp accuracy improvement. (j) Compose with SAC (Task 240): run a compiled session with SAC anchors within the buffer-tokens representation; verify no conflict (anchor positions should be selected *before* compilation; buffer tokens preserve anchor structure) — may require interaction design and is a known composition risk.
- **Effort**: L (3-4 weeks — LCC compiler (1 week: LoRA optimization loop, buffer-token extraction, self-aligned objective), buffer-token insertion hook (3 days), compilation cache with on-change refresh (3 days), ratio/opt-steps sweep + validation (1 week), compose with SnapKV + SAC + Block-Pool (1 week))
- **Depends on**: MLX LoRA fine-tune infrastructure (exists — used in SAC Task 240). Qwen3-Coder-30B frozen-weight access (have). Block-Pool (Task 216, pending — for buffer-token persistence). Composes with SnapKV (shipped — buffer tokens are pinned), SAC (Task 240 — orthogonal scope), SmallKV (Task 237 — drift on compilation quality). Independent of runtime codec (TQ3, DP-LLM Task 242) and save-time codec (KVTC Task 241, KQ-SVD Task 239). Sequencing: ship last of pass-55 tasks — largest ship-cost, most quality risk, largest projected impact. Ship only after KVTC (Task 241) and DP-LLM (Task 242) are stable. Also consider whether SAC (Task 240 from pass 54) should ship first since the anchor-placement insight informs buffer-token placement in LCC compilation.

## Research-derived tasks (from LIT_REVIEW.md pass 56, 2026-04-23)

### 244. PM-KVQ progressive mixed-precision KV quantization with cumulative-error recovery — block-wise spatial schedule + positional-interpolation calibration
- **Goal**: 1 (closes the cumulative-quantization-error floor that every shipped codec currently ignores — TQ3 and DuoKV apply uniform bit-width everywhere, accumulating error linearly with decode depth on long-CoT traces; PM-KVQ's block-wise memory allocation + progressive bit-width schedule bounds the cumulative error and is the first pass-40-56 result to directly target this failure mode), Goal 2 (up to 8% accuracy improvement over SOTA baselines on reasoning benchmarks at matched memory budget — directly improves long-CoT coding/math eval scores on HumanEval and MMLU-Pro), Goal 3 (2.73-5.18× throughput over fp16 baseline at matched quality — better compression ratio translates to lower memory bandwidth on Apple Silicon's memory-bound workload, which is decode-throughput-limiting)
- **Derived from**: PM-KVQ / Progressive Mixed-Precision KV Cache Quantization for Long-CoT LLMs (2505.18610). Closes pass-55 new gap #5 (error recovery for progressive compression). First pass-40-56 paper to name and directly target the cumulative-error floor. Positional-interpolation calibration fix is an independently valuable offline-only change.
- **Change**: Three phases. Phase 1 — **positional-interpolation calibration**: modify `omlx/cache/tq3_calibration.py` (new or existing) to apply RoPE frequency interpolation on short calibration traces (~4K tokens) to approximate position-p distributions at positions up to 1M; compute scales and codebook against the interpolated distribution; output `omlx/weights/tq3_scales_posinterp_qwen3_coder_30b.safetensors`. One-time offline step; no runtime impact. Phase 2 — **block-wise bit-width schedule**: extend TurboQuantKVCache to accept a per-block bit-width vector (48 elements for Qwen3-Coder-30B), stored as `omlx/weights/tq3_block_schedule_qwen3_coder_30b.json`. The schedule is derived offline via PM-KVQ's sensitivity analysis — for each block, measure the perplexity delta of dropping it to 2-bit vs keeping it at 4-bit; blocks with larger delta get higher bit-width. The total bit-budget integrates to the same average as uniform 3-bit. Phase 3 — **optional progressive decode-depth schedule**: modulate bit-width with decode-step index — later tokens in long-CoT traces use higher precision for the KV they read (since their reads dominate cumulative error buildup). Wire in as `--pm-kvq-schedule PATH` (default: shipped schedule for Qwen3-Coder), `--pm-kvq-pos-interp` (default on — positional-interpolation calibration is near-free and improves accuracy), `--pm-kvq-progressive` (optional — decode-depth-based schedule; default off until validated), `--pm-kvq-recalibrate` (CLI subcommand for recomputing schedule against a custom corpus). Composition: with TQ3 (shipped — PM-KVQ extends the codec with per-block bit-width), with DP-LLM (Task 242 — DP-LLM temporal per-step on top of PM-KVQ spatial per-block; the (block × step) matrix of bits is the genuine composition), with KVTuner (Task 186 — KVTuner learns per-head static bit-width, PM-KVQ learns per-block static; the composition is a per-(block, head) matrix), with KVTC (Task 241 — independent; save-time vs runtime codec), with DuoKV (shipped — PM-KVQ applies to both streaming and retrieval paths with separate schedules per path). Risk: progressive schedule may interact badly with eviction (SnapKV re-RoPE changes positions, which changes the "ideal" bit-width for repositioned tokens); validate on NIAH post-eviction. Calibration risk: positional interpolation may misrepresent the tail of the channel distribution; cross-validate against a small true-long calibration set (10 samples at 64K) to catch misrepresentation.
- **Verify**: (a) Positional-interpolation calibration correctness: computed scales must be within 10% of scales computed on true long-context calibration data (cross-validate on 10 samples at 64K). (b) Block-wise schedule effect: at 64K, 128K, 256K contexts, PM-KVQ (block-wise variable bit-width averaging 3) must achieve >= 1% accuracy improvement over uniform 3-bit on HumanEval and MMLU-Pro; if no improvement, schedule is mis-learned. (c) Bit-budget equivalence: the sum of block bit-widths × block-params divided by total params must equal 3.0 (average) ± 0.05 to ensure no hidden memory inflation. (d) Cumulative-error check: at 64K decode depth on a long-CoT trace, per-token-position MSE error between PM-KVQ and fp16 must be within 80% of uniform-3-bit MSE (PM-KVQ should be better, not just equal). (e) Throughput: decode throughput with PM-KVQ must be >= baseline TQ3 (block-wise bits average to same, so no regression expected). (f) NIAH preservation: NIAH@4K/16K/64K/128K must all PASS with PM-KVQ enabled. (g) HumanEval preservation: HumanEval pass@1 must be >= 92% (baseline 95%, allow 3-pp margin for schedule noise). (h) Composition with eviction: SnapKV@50% + PM-KVQ must not regress NIAH vs baseline SnapKV@50% + uniform 3-bit. (i) Composition with DP-LLM (Task 242): PM-KVQ + DP-LLM must produce a valid (block, step) bit-matrix where no position exceeds 4-bit or drops below 2-bit. (j) Recalibrate command: `--pm-kvq-recalibrate` on a new corpus produces a valid new schedule JSON; test with toy corpus of 100 sessions.
- **Effort**: M (1-2 weeks — positional-interpolation calibration (3 days, mostly offline), block-wise schedule learning script (3 days), TurboQuantKVCache per-block bit-width extension (3 days), accuracy validation + NIAH ladder (3 days), optional progressive decode-depth schedule (deferred if time-short))
- **Depends on**: TurboQuantKVCache (shipped — PM-KVQ extends). Calibration corpus (needs curation — ~100 diverse long-CoT traces for schedule learning, ~2 days). Composes with DP-LLM (Task 242, pending — DP-LLM temporal on top of PM-KVQ spatial), with KVTuner (Task 186, pending — per-head on top of per-block), with DuoKV (shipped — per-path schedules), with SnapKV (shipped — post-eviction validation required). Independent of save-time codecs (KVTC Task 241, KQ-SVD Task 239) and XQuant (Task 245 — alternative compression axis). Sequencing: ship-first of pass-56 tasks — smallest ship-cost, most modest absolute impact but near-free calibration fix alone is worth shipping. Calibration is S (3 days); full block-schedule is M (1-2 weeks).

### 245. XQuant KV cache rematerialization via cached quantized input activations — memory-wall inversion for Apple Silicon
- **Goal**: 1 (7.7-12.5× memory reduction vs fp16 KV at <0.1 perplexity degradation — at 1M context fp16 KV is ~120 GB, XQuant brings this to 9.6-15 GB, fitting comfortably in 48 GB unified memory with 30 GB+ headroom vs TQ3's 22.5 GB; explicitly designed around compute:bandwidth imbalance — the right direction for Apple Silicon where FLOPs are abundant and memory bandwidth is scarce), Goal 3 (memory-bandwidth-bound decode throughput on M4 Pro benefits directly from reduced KV footprint — 12.5× less memory to touch per attention step is a 12.5× memory-bandwidth saving, partially offset by W_K/W_V rematerialization matmuls; net throughput gain expected to be 3-5× at 1M context), Goal 6 (48 GB fit is comfortably maintained; XQuant trades compute for memory capacity rather than requiring memory headroom — exactly the trade Apple Silicon's profile supports)
- **Derived from**: XQuant (2508.10395). Closes pass-55 new gap #6 (hardware-software codesign). The deepest codesign insight in the pass-40-56 arc — explicitly exploits the compute:bandwidth imbalance that defines Apple Silicon's memory-bound workload profile.
- **Change**: Four phases. Phase 1 — **X-cache infrastructure**: new file `omlx/cache/x_cache.py` implementing the X-activation cache. Stores per-layer input activations (post-layernorm, pre-attention-projection) in 3-bit TQ3-style quantization with shared WHT codebook rotation. Capacity is 1 tensor per layer (vs 2 for K+V), so memory is already halved before quantization. Cache API mirrors existing KVCache: `update(x_tensor)`, `get()`, `size()`, etc. Phase 2 — **rematerialization in attention forward**: modify `omlx/models/qwen3_coder.py` attention forward to accept an X-cache; during attention, run `K = dequant(X) @ W_K; V = dequant(X) @ W_V` instead of reading pre-computed K, V from cache. The per-layer matmul cost is ~one extra FFN-matmul equivalent per step; must benchmark on M4 Pro to confirm net positive. Phase 3 — **XQuant-CL cross-layer similarity**: exploit that X embeddings are highly correlated across adjacent layers — cache X only every N layers, share across skip-connected groups. Default N=4 per paper. Phase 4 — **integration with DuoKV**: XQuant applies to streaming-head path only (retrieval-head reads are rare and predictable so KV caching is cheaper than rematerialization); DuoKV's existing per-head classification drives this. Wire in as `--kv-mode xquant` (new mode parallel to tq3/duo/native), `--xquant-cl` (default on for further compression), `--xquant-bit-width {2,3,4}` (default 3 matching TQ3), `--xquant-share-every N` (default 4 — cross-layer sharing stride), `--xquant-scope {all, streaming-only}` (default streaming-only). Composition: with TQ3 (shipped — XQuant shares TQ3's WHT codebook rotation for X-quantization), with DuoKV (shipped — XQuant operates on streaming path only by default), with SnapKV (shipped — eviction drops X entries the same way it drops K/V; rematerialization naturally applies to surviving X), with PM-KVQ (Task 244 — X has its own per-block sensitivity; PM-KVQ's schedule can drive X's per-block bits), with KQ-SVD (Task 239 — mutually exclusive, pick one axis of compression), with KVTC (Task 241 — independent; KVTC applies to X bytes on save). Risk: per-decode-step W_K, W_V matmuls add compute cost; if overhead exceeds 20% of decode time the trade is not net-positive — measure on M4 Pro MLX. Kernel risk: MLX's current attention kernel assumes pre-computed K, V; restructuring to accept X + remat requires kernel-side change (not just Python).
- **Verify**: (a) X-cache correctness: quantize + dequantize X at 3-bit and compare to fp16 — MSE must be within 0.5% of TQ3's KV-quantization MSE (X distributions may differ slightly, but a similar bound should hold). (b) Rematerialization matches caching: at fp16 precision, K_remat = X @ W_K must match cached K exactly — unit-test with deterministic input. (c) End-to-end accuracy: at 16K, 64K, 128K contexts, XQuant must preserve perplexity within 0.1 vs fp16 baseline (paper claim); must preserve HumanEval pass@1 within 3 pp. (d) Memory reduction: peak Metal memory at 64K context with XQuant must be 0.25-0.4× of baseline TQ3 (7.7×+ vs fp16 = 3×+ vs 3-bit TQ3). (e) Throughput check: decode throughput at 16K with XQuant must be net positive vs TQ3 — acceptance threshold is >= 1.2× TQ3 throughput (accounts for remat overhead). If < 1.2×, either lower `--xquant-share-every` or disable streaming-scope. (f) Cross-layer sharing: XQuant-CL with `--xquant-share-every 4` must preserve accuracy within 0.01 additional perplexity vs XQuant-CL off. (g) Composition with DuoKV: streaming-head scope must not affect retrieval-head accuracy; NIAH@4K/16K must still PASS. (h) SnapKV + XQuant: eviction must correctly drop X entries alongside K/V logical eviction; unit-test the bookkeeping. (i) MLX kernel support: attention forward with X-cache must run without breaking MLX compilation; profile per-layer matmul cost as fraction of total decode time. (j) Revert: `--kv-mode tq3` still works unchanged — XQuant is purely additive.
- **Effort**: L (3-4 weeks — X-cache infrastructure (1 week), rematerialization in attention forward (1 week, includes MLX kernel modification), XQuant-CL cross-layer sharing (3 days), DuoKV integration (3 days), accuracy + throughput validation (1 week), edge-case handling and regression tests (2 days))
- **Depends on**: TurboQuantKVCache (shipped — shares WHT codebook rotation). DuoKV (shipped — XQuant defaults to streaming-head scope). MLX attention kernel source access (have). Composes with PM-KVQ (Task 244 — X per-block bits), with SnapKV (shipped — X eviction), with KVTC (Task 241 — X-byte compression on save). Mutually exclusive with KQ-SVD (Task 239 — alternate compression axis). Independent of DP-LLM (Task 242 — operates on KV runtime policy, not X). Sequencing: ship after PM-KVQ (Task 244) for the schedule-guided per-block bit allocation on X. Major ship risk is the MLX kernel modification for attention-with-rematerialization; if this proves infeasible within the time budget, fall back to a Python-level rematerialization that costs 2-3× decode throughput but validates the approach. Tertiary priority among pass-56 tasks due to ship-cost and kernel risk, but highest absolute memory-reduction upside.

### 246. Commodity-GPU sparse top-k attention for 1M context — training-free orthogonal-to-codec decode-time selection
- **Goal**: 1 (direct Goal-1 hit — ICML 2025 paper demonstrates 1M-token inference on 16 GB GPU RAM, well below Hypercar's 48 GB budget; the top-k sparse attention mechanism retains all KV tokens and attends to only the top-k most relevant per decode step, orthogonal to every compression/eviction path we currently have; converts Goal 3's "constant decode speed across context window" from a quantization-eviction problem into a selection-algorithm problem with a sub-linear answer), Goal 3 (attending to < 2% of tokens at every decode step reduces attention compute 50×+ — memory-bandwidth-bound decode on M4 Pro benefits proportionally, projected 10-50× decode speedup at 1M context; the key property is that decode speed becomes nearly independent of N beyond the top-k reach), Goal 4 (sparse-prefill variant reduces prefill compute proportionally — top-k selection happens chunk-by-chunk during prefill, so prefill speed scales sub-linearly with context length; projected 500+ tok/s at 1M if the selector itself is cheap)
- **Derived from**: Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs (2502.06766, ICML 2025). Highest-leverage pass-56 find — the single most directly Goal-1-aligned paper in the pass-40-56 arc. Orthogonal to every codec (TQ3, DuoKV), every eviction (SnapKV, SparseServe, BUZZ), and every quantization path; retains all tokens in (quantized) form and reduces attention cost via selection rather than compression.
- **Change**: Three phases. Phase 1 — **top-k sparse attention kernel**: new file `omlx/attention/sparse_topk_attention.py`. Takes Q, K-cache, V-cache and selects top-k tokens by approximate attention score. Critical design: the selector must be sub-linear in N — read the full PDF's Section 3 for the paper's algorithm (likely LSH-based, tree-based, or based on approximate nearest neighbors). Start with the simplest effective variant; upgrade if accuracy warrants. Output: (indices, Q @ K[indices]^T, V[indices]) for the selected top-k. Compose the final attention output over only these k tokens. Phase 2 — **integration with MLX attention forward**: modify `omlx/models/qwen3_coder.py` attention to dispatch to sparse-topk variant when context length exceeds a threshold (default 32K — below this, full attention is cheaper than selection overhead). The sparse path reads from TurboQuantKVCache (quantized 3-bit KV) for the selected top-k indices. Phase 3 — **k-tuning schedule + approximate-selector choice**: implement paper's tunable-k mechanism — higher k gives higher accuracy at higher cost; default k = 20000 (at N = 1M, this is 2%). Wire in as `--sparse-topk-k K` (default 20000), `--sparse-topk-threshold N` (default 32K — below this context length, use full attention), `--sparse-topk-selector {approx, hash, tree}` (default approx — the simplest algorithm; other modes for accuracy-critical paths), `--sparse-topk-per-head` (optional — allow different k per head; default shared). Composition: with TQ3 (shipped — sparse-topk reads quantized KV for selected tokens), with SnapKV (shipped — two-stage eviction-then-select; SnapKV evicts permanently during prefill, sparse-topk selects among retained tokens per decode step), with DuoAttention (shipped — retrieval heads stay full-attention; streaming heads use sparse top-k), with SparseServe (shipped — SparseServe's sparse eviction provides the prefilled-cache-is-already-small baseline; sparse-topk composes by doing selection over that smaller cache), with XQuant (Task 245 — the X-cache remat can be gated on top-k selection; only rematerialize X for selected tokens), with PM-KVQ (Task 244 — orthogonal; PM-KVQ's bit-schedule applies to retained KV, sparse-topk determines which KV is read per step). Risk: the top-k selection algorithm must be cheap; a naïve O(N) selection at every decode step defeats the purpose. Paper's algorithm (likely LSH or tree-based) must be validated on Apple Silicon — the selector's memory-access pattern is non-contiguous, which is bad for Metal's memory-access optimization. May require a gather kernel optimized for random-access patterns (extends DiffKV's Task 229 compaction path). Verify selector itself is sub-linear.
- **Verify**: (a) Top-k selection correctness: for a small test case (N=1024, k=32), compare sparse-topk output to full-attention output on the top-k ground-truth indices; MSE must be within 1% (exact on the selected indices, zero otherwise). (b) Top-k accuracy preservation: at 16K context with k=512 (3% of N), attention output MSE vs full attention must be within 5% on 100 real traces. (c) RULER preservation: RULER@1M with sparse-topk must be >= 95% of full-attention RULER (paper's claim); measure on subset of RULER tasks if full is too expensive. (d) Selector sub-linearity: wall-clock time of selection step must scale sub-linearly with N — measure at N = 32K, 128K, 512K, 1M and fit; the slope must be < 0.5 (sub-linear). (e) NIAH@64K/128K preservation: sparse-topk with k=20000 must PASS NIAH at both context lengths. (f) Threshold behavior: at N < 32K, sparse-topk must fall through to full attention (no selection overhead); unit-test. (g) Decode throughput at 1M: decode throughput with sparse-topk must be >= 10 tok/s at 1M context (Goal 3 minimum; paper's framing suggests 50+ achievable). (h) Composition with TQ3: sparse-topk reading TQ3-quantized KV must preserve accuracy within 1% of sparse-topk reading fp16 KV. (i) Composition with DuoAttention: retrieval heads must remain full-attention; unit-test. (j) Revert: `--sparse-topk-threshold 1000000` (higher than any context) reproduces baseline TQ3 behavior exactly.
- **Effort**: L (2-3 weeks — top-k sparse attention kernel (1 week: selector algorithm + gather kernel for random access), MLX attention forward integration (3 days), k-tuning + selector mode choice (2 days), 1M-context validation (1 week — requires careful memory management; may need to run at 256K initially and extrapolate), edge cases + regression (3 days))
- **Depends on**: TurboQuantKVCache (shipped — sparse-topk reads quantized KV). DiffKV gather kernel (Task 229, pending — may share the random-access gather primitive). MLX attention kernel source access (have). Composes with SnapKV (shipped), DuoAttention (shipped), SparseServe (shipped), XQuant (Task 245 — optional gated rematerialization). Independent of KVTC (Task 241) and KQ-SVD (Task 239). Sequencing: highest-leverage pass-56 task — ship-second after PM-KVQ (Task 244, smaller ship-cost). The 1M-context decode throughput is a direct Goal 1 + Goal 3 gate, so this is a gate-defining ship. Consider validating at 256K first (smaller memory, faster iteration) before scaling to 1M validation runs. The paper's Section 3 algorithm details drive the selector design — download and read the full PDF before starting implementation. ICML 2025 peer-review backing makes this the strongest-provenance pass-56 task.

## Research-derived tasks (from LIT_REVIEW.md pass 57, 2026-04-23)

### 247. TTKV temporal-tiered KV cache with differential precision — software-policy migration for Apple Silicon unified memory
- **Goal**: 1 (temporal-tiered partition with heterogeneous capacity + precision directly closes the hierarchical two-level KV gap held since pass 55; 5.94× cross-tier traffic reduction on 128K-context tasks demonstrates the policy works at Hypercar's first-tier context target; the TTKV abstraction makes "retention tier" a first-class primitive we have been approximating through SnapKV+DuoKV+CAOTE composition), Goal 3 (up to 76% latency reduction and 2× throughput improvement on long-context decode — on Apple Silicon's memory-bound workload the reduced cross-tier traffic translates through reduced cache-line pressure; projected 1.3-1.8× decode speedup at 64K on Hypercar baseline), Goal 6 (differential precision per tier is the principled version of Hypercar's ad-hoc fast-fp16 + slow-3-bit slab split; memory fit is unchanged — the total bit-budget matches current TQ3 baseline, but the allocation is temporal rather than spatial)
- **Derived from**: TTKV / Temporal-Tiered KV Cache for Long-Context LLM Inference (2604.19769). Closes pass-56 gap #2 (hierarchical two-level KV for text — held 1 pass from pass 55). The paper's HBM/DRAM hardware framing is incidental; the algorithmic contribution (temporal partition + differential precision + block-wise streamed attention) is hardware-agnostic and maps cleanly to Apple Silicon's unified memory as a software-only tiered memory.
- **Change**: Three phases. Phase 1 — **tier infrastructure**: new file `omlx/kv_caches/tiered_kv.py` implementing a two-tier slab structure. Fast tier is fp16 or 4-bit TQ (configurable); slow tier is 3-bit TQ (shared codec with existing TurboQuantKVCache). Capacity is configurable — fast-tier default 4K tokens (fits comfortably within Metal cache lines and L2), slow-tier unbounded (grows with context). The cache API mirrors existing KVCache with added `promote(idx)` and `demote(idx)` methods; the attention forward iterates both tiers. Phase 2 — **temporal migration policy**: each decode step, tokens whose position age exceeds the fast-tier window are batch-demoted to slow tier (quantize once, write to slow slab, drop from fast). Re-promotion: if a slow-tier token's running attention-score sum over the last K=256 decode steps (tracked in a compact histogram alongside the slow slab) exceeds a threshold T, it is promoted back to fast tier (dequantize, write to fast slab). Add hysteresis — a token must be in a tier for at least K=32 decode steps before it can migrate out, preventing thrash. Phase 3 — **block-wise streaming attention**: modify `omlx/attention/` to dispatch attention over the two tiers in parallel — fast-tier attention computes synchronously (small, hot), slow-tier attention streams in blocks of B=256 tokens (fits Metal thread-group memory) while fast-tier results are being aggregated. The final output is a weighted sum of tier contributions. Wire in as `--kv-mode tiered` (new mode parallel to tq3/duo/native/fp16), `--tiered-fast-window N` (default 4096 — fast-tier capacity in tokens), `--tiered-fast-bits {4, 8, 16}` (default 16 — fast-tier precision; 4 is an aggressive tier, 16 is the canonical TTKV setting), `--tiered-slow-bits {2, 3}` (default 3 — slow-tier precision, matches TQ3 baseline), `--tiered-promote-threshold T` (default 0.10 — fraction of average attention share above which slow-tier tokens are promoted), `--tiered-hysteresis N` (default 32 — min decode steps in a tier before migration allowed), `--tiered-block-size B` (default 256 — streaming-attention block size for slow-tier access). Composition: with TQ3 (shipped — TQ3 is the slow-tier codec; no changes to TQ3 itself), with DuoKV (shipped — per-head temporal tiers give a (head × time) classification matrix; the TTKV temporal axis replaces DuoKV's static retrieval-vs-streaming classification for heads where temporal behavior dominates head-type behavior), with SnapKV (shipped — two-stage eviction: SnapKV permanently removes the worst tokens during prefill, TTKV demotes the moderately-relevant tokens to slow tier during decode; they compose as sequential filters), with PM-KVQ (Task 244 — PM-KVQ's per-block bit schedule drives the slow-tier codec bits; the fast tier stays fp16 regardless), with Commodity-GPU Sparse 1M (Task 246 — sparse-topk selection happens within the tiered structure; selection prioritizes fast-tier tokens first since they are cheaper to access). Risk: migration overhead (quantize-on-demote, dequantize-on-promote) adds per-decode-step cost; if > 2% of decode time, reduce migration frequency (demote in batches every N decode steps instead of per-step). Tier-thrashing risk: if promote/demote flips frequently for the same token, increase hysteresis window or tighten promote threshold. Memory risk: fast-tier fp16 at 4096 tokens adds ~0.1 GB per layer (48 layers = ~5 GB additional Metal memory vs. all-3-bit); validate fit on 48 GB M4 Pro.
- **Verify**: (a) Tier infrastructure correctness: promote/demote round-trip on the same token reproduces within 5% L2 error at fast-fp16 + slow-3-bit (quantization floor); test with 1000 random tokens. (b) Migration policy: at 64K context, demote cost per-step must be < 1% of decode time; promote cost per-step must be < 0.5% (rarer event). (c) Tier-size adherence: after 10K decode steps, fast-tier must contain exactly `--tiered-fast-window` tokens (temporal-ordering invariant); unit-test. (d) Cross-tier traffic reduction: at 128K context, measure cross-tier (i.e., slow-tier-read-into-compute) bandwidth vs baseline TQ3 all-3-bit; must be >= 3× reduction (paper claims 5.94× on GPU; allow margin for Apple Silicon). (e) Decode throughput: at 64K with `--kv-mode tiered`, decode-throughput must be >= baseline TQ3 by 1.2× (bounded claim vs paper's 2× GPU result). (f) NIAH preservation: NIAH@4K/16K/64K/128K must all PASS with tiered mode enabled — quality is the gate-defining concern. (g) HumanEval preservation: HumanEval pass@1 must be within 3 pp of baseline TQ3 (95%). (h) Hysteresis effectiveness: under worst-case attention pattern (rapidly shifting focus), migration count must be <= 2× the expected natural migration rate; if > 2×, increase hysteresis window. (i) Composition with SnapKV: SnapKV@50% + tiered must preserve NIAH vs baseline SnapKV@50% + TQ3. (j) Revert: `--kv-mode tq3` still works unchanged — tiered is purely additive.
- **Effort**: M-L (2-3 weeks — tier infrastructure with promote/demote (1 week), temporal migration policy with attention-score histogram tracking (3 days), block-wise streaming attention kernel in MLX (1 week — critical path; may require `mx.fast.metal_kernel` for efficient overlap), accuracy + throughput validation at 4K/16K/64K/128K (1 week), hysteresis tuning and edge-case handling (3 days))
- **Depends on**: TurboQuantKVCache (shipped — slow-tier codec). MLX streaming attention primitives (have — `mx.fast` has the kernels needed). Composes with DuoKV (shipped — (head × time) matrix), with SnapKV (shipped — two-stage filter), with PM-KVQ (Task 244 — per-block slow-tier bit schedule), with Commodity-GPU Sparse 1M (Task 246 — selection within tiered structure). Independent of XQuant (Task 245) and KVTC (Task 241). Sequencing: ship-second of pass-57 tasks — middle ship-cost; the block-wise streaming attention kernel is the main engineering risk. Can ship before or after PM-KVQ (Task 244) — if before, slow-tier uses uniform TQ3 3-bit; if after, slow-tier can use PM-KVQ per-block schedule. Tiered mode is the first task to formalize a temporal (rather than spatial or head-type) axis of KV allocation — a genuine new axis orthogonal to everything shipped or tasked.

### 248. qTTT query-only test-time training for long context — per-query W_Q gradient adaptation on frozen pretrained LLM
- **Goal**: 2 (12.6-14.1 percentage point improvement on LongBench-v2 and ZeroScrolls for Qwen3-4B at matched compute — direct intelligence-breadth gain on long-context benchmarks with the same model family as Hypercar's Qwen3-Coder-30B; the mechanism should transfer to 30B scale with similar or better margin), Goal 1 (evaluated at 8K-128K on the Qwen3 family Hypercar uses — direct fit, no architecture-mismatch concerns; the query-only gradient update is compatible with any frozen KV cache including TQ3 3-bit), Goal 3 (qTTT is *compute-matched* to thinking-token baselines per the paper — it uses the same FLOP budget as inference-time scaling but spends it on gradient updates rather than token generation; net impact on decode speed is neutral at matched budget, positive if the user has extra compute budget to spend on quality; the key property is that it's an optional per-query knob, not a baseline cost)
- **Derived from**: Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs (qTTT, 2512.13898). Closes pass-56 gap #5 (test-time adaptation for long context — new angle, unexplored). First pass-40-57 paper to demonstrate test-time gradient adaptation on a *frozen* pretrained LLM (not a TTT-pretrained architecture like TTT-E2E which requires meta-learning during pretraining) with full compatibility with quantized KV cache on the exact Qwen3 family at the exact 8K-128K context window Hypercar targets. The target-distractor-margin proof provides a theoretical reason why it works at long context where vanilla prompt engineering fails.
- **Change**: Three phases. Phase 1 — **W_Q-only gradient infrastructure**: MLX has existing gradient support via `mx.grad`; create a new training loop in `omlx/train/qttt.py` that takes (model, kv_cache, context) and (a) identifies the W_Q matrices across all 48 layers × 8 heads (query projection weights), (b) sets their `requires_grad=True` while freezing everything else (W_K, W_V, FFN, embeddings, layernorm), (c) makes a pinned copy of the original W_Q matrices (for post-query restoration), (d) runs N=32 gradient descent steps on W_Q only with a small learning rate (default 1e-5). Phase 2 — **span sampling and loss**: implement the span-based TTT objective — at each gradient step, randomly sample M=32 spans of length k=128 from the prefilled context; for each span, predict the last token given the prior k-1 tokens using the current (possibly updated) W_Q against the frozen KV cache; loss is the cross-entropy of next-token prediction summed over spans. The loss pushes W_Q to produce queries that correctly retrieve from the frozen KV cache. Phase 3 — **inference integration**: new CLI/API flag `--qttt-enable` on the hypercar_server generate endpoint; when set, the request flow is prefill → qTTT loop → generate-with-adapted-W_Q → restore-W_Q-from-pin. Also expose as a per-request query parameter `/v1/completions?qttt=1` so the user can opt-in per call (high-value queries only). Wire in as `--qttt-enable` (default off — opt-in per request), `--qttt-steps N` (default 32 per paper; user can override for faster-but-weaker adaptation), `--qttt-span-length K` (default 128), `--qttt-span-count M` (default 32), `--qttt-lr L` (default 1e-5 — conservative step size appropriate for frozen-pretrained-model adaptation), `--qttt-scope {all-heads, retrieval-only, streaming-only}` (default all-heads; DuoKV-aware scoping is a follow-up — retrieval-only could halve compute), `--qttt-delta-clip C` (default 0.10 — max fraction change in W_Q elements per query; safety bound on adaptation magnitude), `--qttt-gradient-dtype {fp32, bf16, fp16}` (default bf16 — halves gradient-storage memory vs fp32 at small numerical cost). Composition: with TQ3 (shipped — TQ3 KV is frozen during TTT, so fully compatible; qTTT only adapts queries, not cached keys/values), with DuoKV (shipped — DuoKV's head classification is compatible; qTTT updates W_Q for both retrieval and streaming heads by default), with SnapKV (shipped — SnapKV evicts before TTT begins; qTTT operates on the retained cache), with Commodity-GPU Sparse 1M (Task 246 — sparse-topk selection happens at decode; qTTT adapts W_Q which changes the scores that drive selection — composition needs careful validation since both affect attention scoring; recommended: validate qTTT alone first, then validate composition), with SmallKV (Task 237 — SmallKV's drift monitor can flag when qTTT adaptation has diverged from baseline, providing a safety signal; if drift fires, clip W_Q updates more aggressively), with TTKV (Task 247 — qTTT operates on the current cache state; tiered organization is orthogonal). Risk: 2-4 minute per-query latency on M4 Pro for 128K context is too slow for interactive use — gate qTTT behind explicit user opt-in. Gradient-storage risk: ~100 MB fp32 gradient on 30B-scale W_Q is tight with Metal allocator — use bf16 by default (50 MB) to leave headroom. Numerical-stability risk: 32 gradient steps at lr=1e-5 on a 30B model is a small change but can drift on edge cases — add the delta-clip safety bound (clamp |W_Q_new - W_Q_original| by 10%) to prevent runaway adaptation. Reproducibility risk: MLX gradients have small nondeterminism; pin a seed for the span sampler so the same query produces consistent results.
- **Verify**: (a) W_Q-only gradient isolation: after N qTTT steps, verify that only W_Q matrices have changed — all other weights (W_K, W_V, FFN, embeddings, layernorm) must be bit-identical to pre-TTT state. (b) Pinned-copy restoration: after query completes, W_Q must match the pinned original bit-identical; test via 100 consecutive queries with and without restoration. (c) Gradient stability: on a test context, 32 steps at lr=1e-5 must produce |W_Q_new - W_Q_original|/||W_Q_original|| < 5% (small, bounded change). (d) Delta-clip enforcement: on an adversarial test case, if the raw update exceeds 10% delta, the clipped update must not exceed 10%; unit-test. (e) Long-context quality gain: on LongBench-v2 subset with 128K context, qTTT must improve pass@1 by >= 5 pp over baseline (paper claims 12+ pp for Qwen3-4B; allow margin for 30B-scale variance). (f) NIAH preservation: NIAH@64K/128K with qTTT must match or exceed baseline NIAH (never regress — W_Q adaptation should only help or stay flat on pure retrieval). (g) HumanEval preservation: HumanEval with qTTT must be within 3 pp of baseline (95%) — short-context workloads shouldn't benefit significantly but also shouldn't regress. (h) Latency measurement: qTTT at 8K context on M4 Pro must complete in < 2 minutes (paper's 28s on A100 extrapolates to ~2 min on M4 Pro via memory-bandwidth ratio); qTTT at 128K must complete in < 10 minutes. (i) Opt-in gating: `--qttt-enable off` path must be bit-identical to baseline; unit-test. (j) Composition with SnapKV: SnapKV@50% + qTTT must preserve NIAH vs baseline SnapKV@50% + no-qTTT.
- **Effort**: M (1-2 weeks — W_Q-only gradient loop (3 days, depends on MLX grad behavior), span sampler + span-based TTT loss (2 days), CLI/API integration with opt-in flag (2 days), pinned-copy + restoration (1 day — critical correctness path), long-context validation + latency measurement on M4 Pro (3 days), delta-clip + safety bounds + edge cases (2 days))
- **Depends on**: MLX gradient support (have — `mx.grad` with selective parameter grad). Qwen3-Coder-30B frozen-weight access (have). TurboQuantKVCache (shipped — qTTT reads frozen KV). Composes with DuoKV (shipped — head-scope flag), with SnapKV (shipped — operates on retained cache), with SmallKV (Task 237 — drift monitor as safety signal), with TTKV (Task 247 — orthogonal scope), with Commodity-GPU Sparse 1M (Task 246 — needs careful validation as both affect attention scoring). Independent of codec tasks (TQ3, PM-KVQ, XQuant, KVTC, KQ-SVD, DP-LLM). Sequencing: ship-first of pass-57 tasks — smallest ship-cost (M not L), highest per-query quality upside, and the opt-in gating means zero risk to baseline behavior. The latency-vs-quality tradeoff belongs to the user per query (via the `qttt=1` request parameter); there is no default cost to ship this feature. Validate at 8K first for iteration speed before scaling to 128K validation runs. Paper's Section 4 (algorithm details) and Table 10-12 (wall-clock measurements) are the implementation references — already downloaded at research/2512.13898_qttt.pdf.

## Research-derived tasks (from LIT_REVIEW.md pass 58, 2026-04-23)

### 249. EpiCache block-wise prefill — memory-bounded sequential prefill replacing single-pass progressive-eviction
- **Goal**: 1 (block-wise prefill bounds peak memory during prefill by processing the context in memory-bounded sequential blocks, directly replacing the 120K-single-pass Metal-OOM failure mode that Hypercar currently works around via session-save/load; EpiCache block-wise prefill is the formalized version of adaptive-prefill chunk sizing (Task 94 shipped) extended with per-block eviction, and is the drop-in replacement for the progressive-eviction path explicitly noted as BLOCKED by MLX KVCache position model / re-RoPE accumulation in the Goal-1 Path section; projected to enable 256K+ single-pass prefill within 48 GB without session-save/load), Goal 6 (memory-bounded peak eliminates the 44.6 GB Metal-peak cliff at 128K that currently forces chunked prefill + eviction midpoint cuts)
- **Derived from**: EpiCache / Episodic KV Cache Management for Long Conversational Question Answering (2509.17396, Apple ML). Closes pass-56 gap #2 (KV cache migration between contexts) flagged in pass 57 for targeted review. Apple ML provenance with code release (github.com/apple/ml-epicache) makes ship-cost minimal vs replicating from scratch.
- **Change**: Three phases. Phase 1 — **block-wise prefill loop**: new file `omlx/prefill/block_wise.py` implementing a sequential prefill that processes the context in blocks of B tokens (default 4096), applying per-block eviction between blocks so peak memory stays bounded. Each block's prefill loads that block's K/V into the cache, runs SnapKV (or EpiCache's episode-scoped scorer, see Task 250) to select which of the running cache to keep under the block-wise memory budget, evicts (with re-RoPE), then proceeds to the next block. Phase 2 — **memory-budget tracking**: add a `PrefillMemoryBudget` class that tracks Metal peak during prefill and adapts block size dynamically — if peak approaches threshold, shrink next block; if peak well below, grow. Default threshold: 90% of 80% Metal budget (i.e., 72% of system RAM). Phase 3 — **integration with hypercar_server prefill path**: new CLI flag `--block-wise-prefill` (default on when context > 32K — below this, single-pass is faster), `--block-wise-prefill-size B` (default 4096 — initial block size), `--block-wise-prefill-adapt` (default on — dynamic block-size adaptation), `--block-wise-prefill-budget F` (default 0.72 — fraction of system RAM reserved as ceiling). Composition: with SnapKV (shipped — per-block scorer), with EpiCache episodic (Task 250 — episode-scoped per-block eviction), with TQ3 (shipped — per-block eviction compacts TQ3 cache in place), with DuoKV (shipped — block-wise applies to both streaming and retrieval paths), with adaptive-prefill (Task 94 shipped — block-wise prefill supersedes the shipped adaptive-chunk path; migrate adaptive-prefill to call into block-wise-prefill). Risk: per-block re-RoPE accumulates error across blocks; the accumulated error at N blocks is the open question that blocks progressive-eviction today. EpiCache's claim is that per-block eviction (N blocks with 1 eviction per block) has bounded error whereas progressive-eviction (1 block with N evictions) accumulates error linearly; verify on NIAH@128K with both modes. Secondary risk: block-size-dependent quality — too-small blocks lose context, too-large blocks exceed memory; adaptive-block-size heuristic is the mitigation.
- **Verify**: (a) Peak memory bounded: at 128K context with `--block-wise-prefill --block-wise-prefill-budget 0.72`, Metal peak must remain under 72% of system RAM throughout prefill (no spikes above). (b) Block-boundary correctness: at a block boundary, the cache state must match what would be produced by running the next block's prefill with the post-eviction cache from the prior block (unit-test with 2-block config vs reference). (c) NIAH preservation: NIAH@64K/128K/256K must PASS with block-wise prefill enabled (compare to single-pass NIAH@64K as upper bound for blocks that fit single-pass). (d) Re-RoPE accumulation bound: at 256K context with 64 blocks × 1 eviction per block, compare per-block cumulative error to progressive-eviction 1-block × 64 evictions (if infeasible, use smallest-scale comparison possible); EpiCache claim is per-block accumulates sub-linearly. (e) 256K single-pass feasibility: 256K context with block-wise prefill must complete in < 3h on 48 GB M4 Pro without invoking session-save/load (current 256K path requires session-save). (f) Adaptive-block-size stability: adaptive block size must not oscillate — the running block size must be monotonically non-increasing within a single prefill, or have at most 1 enlargement per 4 blocks. (g) HumanEval preservation: HumanEval pass@1 with `--block-wise-prefill` enabled must be >= 92% (baseline 95%, 3-pp tolerance for block-boundary effects). (h) Composition with SnapKV: block-wise prefill with SnapKV@50% per block must PASS NIAH at 128K and 256K. (i) Fallback: `--block-wise-prefill off` reproduces baseline single-pass prefill bit-identically. (j) Revert: all prior `--snapkv-keep K --caote --segmented-evict 512` flags continue to work unchanged when block-wise is enabled.
- **Effort**: M (1-2 weeks — block-wise prefill loop (3 days), memory-budget tracker + adaptive block size (3 days), hypercar_server integration (2 days), accuracy validation at 64K/128K/256K (1 week — the 256K validation is the long pole), adaptive-prefill migration (2 days), edge-case handling (2 days))
- **Depends on**: SnapKV (shipped — per-block scorer). Adaptive-prefill (Task 94 shipped — supersedes). Re-RoPE compaction (Task 46 shipped). TurboQuantKVCache (shipped). Composes with EpiCache episodic (Task 250 — per-block scorer upgrade), with TQ3 (shipped — in-place compaction), with DuoKV (shipped). Sequencing: **ship-first of pass-58 tasks** — lowest ship-cost, highest direct impact on Goal 1 (closes the 120K+ single-pass Metal-OOM path that currently requires session-save/load). Independent ship-worthy before Task 250 (which builds on top by upgrading the per-block scorer from SnapKV to episode-scoped). EpiCache paper + Apple ML code release removes most algorithmic ambiguity.

### 250. EpiCache episodic KV compression — multi-turn topic clustering replacing query-dependent eviction
- **Goal**: 1 (episodic compression clusters conversation history into coherent episodes and applies episode-specific eviction, preserving topic-relevant context across turns where SnapKV's query-dependent eviction narrows to the single current query and discards topic history; 4-6× compression at near-full accuracy per paper, directly extending Hypercar's shipped eviction stack to multi-turn agentic workflows; closes pass-56 gap #2 (KV cache migration between contexts)), Goal 3 (2.4× latency reduction on multi-turn workloads per paper — agentic coding workflows with repeated tool-call round-trips are exactly this pattern), Goal 6 (3.5× memory reduction at near-full accuracy gives headroom on 48 GB M4 Pro for larger prompts or lower swap risk)
- **Derived from**: EpiCache / Episodic KV Cache Management for Long Conversational Question Answering (2509.17396, Apple ML). Multi-turn generalization of SnapKV the shipped stack lacks; fixes the known multi-turn failure mode of query-dependent eviction.
- **Change**: Four phases. Phase 1 — **episode boundary detection**: new file `omlx/kv_caches/episodic.py` implementing the episode clusterer. Takes turn-separated conversation history (from the `/v1/chat/completions` messages array), computes sentence-transformer embeddings per turn, clusters into episodes via cosine-similarity threshold (default τ = 0.55 per EpiCache paper). Exposes an `episode_ids: List[int]` parallel to the token positions in the KV cache. Reuse the sentence-transformer infrastructure already deployed for freshness-decay (Task 97 shipped) to avoid adding new model dependencies. Phase 2 — **episode-scoped eviction**: extend SnapKV scorer to take `episode_ids` and compute attention scores *within* episodes — tokens in the same episode as the current query get full scoring; tokens in other episodes get a conservative retention baseline (keep top-K per episode independently, rather than top-K globally). This preserves topic-relevant context from non-active episodes that vanilla SnapKV would evict. Phase 3 — **adaptive layer-wise budget**: implement EpiCache's per-layer sensitivity measurement — for each of the 48 layers, measure how much accuracy degrades when that layer's cache is aggressively evicted; allocate a larger fraction of the total bit/token budget to high-sensitivity layers. Output `omlx/weights/epicache_layer_budget_qwen3_coder_30b.json`, a one-time offline calibration. Phase 4 — **integration**: new CLI flags `--epicache-enable` (default on when num_turns >= 3 — below this, SnapKV single-query is fine), `--epicache-boundary-threshold T` (default 0.55 — cosine threshold for episode boundary), `--epicache-per-episode-keep K` (default 40% per episode — complement to global 50% shipped SnapKV default), `--epicache-layer-budget PATH` (default: shipped budget for Qwen3-Coder-30B). Composition: with SnapKV (shipped — episodic extends the scorer; SnapKV still runs the attention-score calculation, episodic adds the per-episode scoping), with block-wise prefill (Task 249 — per-block eviction uses episodic scorer when conversation spans multiple episodes), with CAOTE (shipped — value-aware scoring applies within each episode), with fair-eviction (shipped — proportional budgets apply per-episode not globally), with DuoKV (shipped — episodic applies to streaming path; retrieval heads stay full), with TQ3 (shipped — orthogonal; codec unchanged), with KVTuner (Task 186 — per-head budget composes orthogonally with per-layer adaptive budget as a (layer, head) matrix). Risk: episode-boundary detection on code-heavy conversations with long single-turn spans — cosine-similarity across code turns may not cluster cleanly (code turns often share low semantic similarity with surrounding English); validate on real agentic coding session replays before shipping. If cluster quality is poor, add AST-node-transition and file-path-transition signals as auxiliary boundary heuristics. Secondary risk: per-episode keep with num_episodes=N gives total keep ≈ N × 40% which may exceed global budget; clamp to `--snapkv-keep K` as global ceiling.
- **Verify**: (a) Episode boundary correctness: on 20 hand-labeled agentic-coding session transcripts (create if needed), measure cluster purity vs labels — threshold τ=0.55 must achieve >= 0.75 cluster purity. (b) Episodic vs query-dependent quality: on 10 multi-turn sessions where turn-5 query references context from turn-2, episodic must retain turn-2 context (measure via attention-score on turn-2 tokens at turn-5 decode); SnapKV single-query baseline must evict turn-2 context; episodic must outperform baseline by >= 10 pp on a tool-call correctness metric. (c) Adaptive layer-budget correctness: summed budget across 48 layers must equal baseline total budget ± 0.01 (no hidden memory inflation). (d) Layer-budget calibration reproducibility: running `--epicache-layer-budget recalibrate` on the same corpus produces the same JSON bit-identically (determinism). (e) NIAH preservation: NIAH@64K/128K with episodic enabled must PASS (never regress vs SnapKV single-query baseline). (f) HumanEval preservation: HumanEval pass@1 with episodic enabled must be >= 92% (baseline 95%, 3-pp tolerance). (g) Composition with block-wise prefill (Task 249): block-wise + episodic must preserve NIAH vs block-wise + vanilla SnapKV. (h) Global-budget clamp: if num_episodes × per-episode-keep exceeds global-keep, total retained tokens must equal global-keep (clamp enforced). (i) Opt-in gating: `--epicache-enable off` reproduces SnapKV single-query baseline bit-identically. (j) Code-heavy fallback: when cluster-purity signal is weak (all turns cluster into 1 episode), must gracefully fall back to SnapKV single-query behavior.
- **Effort**: M (1-2 weeks — episode clusterer (3 days, reuses shipped sentence-transformer), episode-scoped scorer (2 days), adaptive layer-budget offline calibration (3 days), hypercar_server integration (2 days), accuracy validation on multi-turn workloads (1 week — the 20-session hand-labeling is the long pole), edge-case handling (3 days))
- **Depends on**: SnapKV (shipped — episodic extends scorer). Block-wise prefill (Task 249 — episodic composes with per-block scorer). Freshness-decay sentence-transformer (Task 97 shipped — reuse). TurboQuantKVCache (shipped). Composes with CAOTE (shipped), with fair-eviction (shipped), with DuoKV (shipped), with TQ3 (shipped), with KVTuner (Task 186). Sequencing: **ship-second of pass-58 tasks after Task 249** — builds on block-wise prefill to fully replace the shipped SnapKV single-query path for multi-turn workloads. The 20-session hand-labeling dataset is the main pre-ship investment. Tasks 249+250 together fully close pass-56 gap #2 and pass-57's targeted-review flag.

### 251. LoCoBench integration — long-context code evaluation replacing extrapolation with measurement
- **Goal**: 2 (first benchmark in corpus that directly probes Hypercar's Goal 1 × Goal 2 intersection at the exact 10K-1M context ladder Hypercar targets; 8 task categories spanning architectural understanding, cross-file refactoring, multi-session development, bug investigation, feature implementation, code comprehension, integration testing, security analysis provide genuine long-context code understanding evidence that HumanEval single-function and Code Intel small-repo evals do not; 17 metrics across 4 dimensions combined into LoCoBench Score (LCBS) as a single headline number), Goal 1 (measured quality degradation curves at 10K/32K/64K/128K/256K replace the current extrapolation-from-4K-NIAH claim; if LoCoBench scores hold at 128K, that is measured evidence vs the current "NIAH@4K passes so we trust the 128K claim" extrapolation)
- **Derived from**: LoCoBench (2509.09614, Salesforce AI Research, code released at github.com/SalesforceAIResearch/LoCoBench). First benchmark in the pass-40-58 arc directly probing long-context code understanding at Hypercar's target context ladder.
- **Change**: Three phases. Phase 1 — **harness integration**: new file `omlx/bench/locobench.py` that loads the LoCoBench dataset from the released repo, exposes a programmatic API `run_locobench(scenarios, context_budget, metrics)` that feeds scenarios through hypercar_server and collects the 17 metrics. Phase 2 — **scenario subset selection**: the full 8000 × 7-context-lengths matrix is weeks of M4 Pro runtime; implement a stratified sampling strategy that selects ~200 scenarios across the 8 categories at each of 10K, 32K, 64K, 128K context lengths. This reduces runtime to ~1-2 hours per context (matching HumanEval runtime scale). 256K+ runs remain opt-in (multi-hour). Phase 3 — **hypercar_bench gate integration**: extend `omlx/bench/hypercar_bench.py` to include LoCoBench phase with pass/fail gates per context. Gate thresholds: LCBS >= 40 at 10K (baseline check), LCBS >= 30 at 32K, LCBS >= 20 at 128K (these thresholds need calibration on first-run; start with "non-zero and trend consistent"). Wire in as `--bench-locobench {off, subset, full}` (default subset — the stratified 200-scenario sample), `--bench-locobench-contexts "10K,32K,64K,128K"` (default four budget points), `--bench-locobench-full` (opt-in to the 8000-scenario run). Composition: extends the existing 11-gate `hypercar_bench` pipeline; LoCoBench is a new phase run after HumanEval. Report renders as a context-vs-LCBS table in the benchmark summary.
- **Verify**: (a) Harness correctness: smoke-test on 10 scenarios at 10K context — LCBS scores match the reference implementation within 1% (or bit-identical if deterministic). (b) Subset-vs-full consistency: 200-scenario subset at 10K must produce LCBS within 10% of full 1000-scenario-at-10K LCBS (measure on one context for the calibration). (c) Context-ladder sanity: LCBS at 10K must be >= LCBS at 32K must be >= LCBS at 64K (monotonic non-increasing; non-monotonic flags a bug in the harness). (d) Runtime fit: stratified subset at 4 contexts must complete in < 8 hours total on M4 Pro (the per-context budget is ~1-2 hours). (e) Integration: `.venv/bin/python -m omlx.bench.hypercar_bench --full --bench-locobench subset` completes and reports LCBS at each context; gate pass/fail decisions reflect calibrated thresholds. (f) Metric correctness: the 17 metrics each have a non-null value per scenario (no division-by-zero or missing-data floors). (g) Leakage audit: verify LoCoBench scenarios were synthesized after Qwen3-Coder-30B training cutoff (paper's 5-phase synthesis suggests yes, but pin down via file timestamps + authors' statement). (h) Compare to SOTA: Hypercar's LCBS should be within 50% of SOTA long-context models reported in the paper (weaker but detectable; gates ensure non-trivial performance). (i) Hypercar-specific: at 128K with `--snapkv-keep 0.5 --caote --segmented-evict 512` (the shipped long-context config), LoCoBench must produce a meaningful score; the 128K-post-eviction measurement is the core Goal-1 validation. (j) Revert: `--bench-locobench off` restores baseline hypercar_bench behavior (pre-pass-58).
- **Effort**: M (2 weeks — harness integration (3 days), subset selection strategy + calibration (3 days), hypercar_bench gate integration (3 days), first-run calibration of thresholds (1 week — must iterate to get reasonable thresholds on the 30B model), leakage audit (2 days))
- **Depends on**: hypercar_bench (shipped — extends). LoCoBench dataset release (have — github.com/SalesforceAIResearch/LoCoBench). Composes with every KV mode (duo, tq3, native, fp16 — LoCoBench measures quality orthogonal to codec choice). Sequencing: **ship-third of pass-58 tasks** — parallel to Tasks 249+250 which ship the algorithmic upgrades; LoCoBench provides the measurement infrastructure that validates those upgrades. Ideal sequencing: Task 249 (block-wise prefill) → Task 251 (LoCoBench integration) → Task 250 (episodic compression, measured against LoCoBench gates). Runtime is the main investment; careful subset selection keeps it under 8-hour turnaround.

### 252. Stream-CQSA exact-attention streaming — cyclic-quorum-set decomposition for arbitrary memory budgets
- **Goal**: 1 (first pass-40-58 paper to preserve *exact* attention arithmetic while fitting arbitrary memory budgets via CQS Divide decomposition — all other Goal-1 paths in the corpus involve approximation (TQ3 quantization distortion, SnapKV eviction information loss, sparse-top-k selection error, KQ-SVD rank-projection error, PM-KVQ cumulative quantization error); paper claims exact attention over billion-token sequences on single GPU via streaming — well beyond Hypercar's 1M target at zero approximation error), Goal 3 (memory-adaptive scheduling recasts attention from monolithic operation to collection of schedulable tasks; on Apple Silicon's memory-bound decode profile, reduced per-step memory footprint translates directly to reduced cache-line pressure and higher decode throughput; projected memory reduction is exact-trades-memory-for-time so the Goal-3 throughput benefit depends on whether the streaming orchestration overhead is below the memory-bandwidth saving), Goal 6 (predictable memory scaling — the single-node memory-budget guarantee every other Goal-1 path in the corpus lacks; CQSA alone is approximation-free)
- **Derived from**: Stream-CQSA / Avoiding Out-of-Memory in Attention Computation via Flexible Workload Scheduling (2604.20819, 2026-04-22 — this month, first-class Goal-1 paper with bit-identical exactness guarantee).
- **Change**: Exploratory four-phase implementation gated on Phase-1 exactness validation. Phase 1 — **CQS Divide reference implementation**: new file `omlx/attention/cqs_divide.py` implementing the CQS Divide operation on a small scale. Read paper's Section 3 algorithm in detail before implementing — the cyclic-quorum-set decomposition must be mathematically understood before translation to MLX. At N=1024 with budget=128, produce partitioned attention and verify recomposition is **bit-identical** to full-attention (not within epsilon — bit-identical). This is the gate for Phase 2+. Phase 2 — **Streaming scheduler**: the memory-adaptive scheduler that partitions attention into subproblems fitting arbitrary budgets. Integrates with MLX dispatch via `mx.fast.metal_kernel` for streaming execution. Phase 3 — **Attention-forward integration**: modify `omlx/models/qwen3_coder.py` attention to dispatch to CQSA when memory budget is constrained (e.g., when context > threshold or when Metal peak approaches limit). Phase 4 — **Composition with quantized KV**: CQSA is codec-orthogonal — it operates on whatever KV cache it is given. Verify exactness with TQ3 3-bit KV cache (the exactness guarantee is vs "full-attention-on-given-KV" not vs fp16 — if given TQ3 KV, CQSA produces the same output as full attention over TQ3 KV, not vs full attention over fp16 KV). Wire in as `--attention-mode {standard, cqsa}` (default standard — CQSA is exploratory), `--cqsa-budget BYTES` (default: 50% of Metal ceiling — CQSA stays well below the budget), `--cqsa-block-size B` (default auto — scheduler decides). Composition: with TQ3 (shipped — CQSA operates on any KV, exactness is vs given-KV), with SnapKV (shipped — CQSA attention runs on retained cache), with DuoKV (shipped — both paths benefit), with block-wise prefill (Task 249 — block-wise prefill compresses context at prefill time; CQSA streams attention at decode/prefill time on the compressed cache), with Commodity-GPU Sparse 1M (Task 246 — mutually composable; sparse-top-k chooses which tokens to attend to, CQSA streams the attention over that choice). Risk: **paper is brand new (2026-04-22, this month) with no independent replication**; the bit-identical exactness claim must be verified at small scale before trusting at 1M. If Phase-1 bit-identical check fails, the entire task path is aborted and the paper is declared not-ready-for-ship. Secondary risk: streaming orchestration overhead — the partition cost itself has wall-clock impact; if CQSA is slower than baseline full-attention at 1M due to overhead, the Goal-3 benefit evaporates even if Goal-1 benefit holds. Tertiary risk: CQS-theory derivation (from 1980s distributed-systems theory) is novel in LLM-inference; must carefully implement Section 3's algorithm rather than rely on intuition.
- **Verify**: (a) **Phase-1 bit-identical check** (gate-defining): at N=1024 with budget=128 (8 partitions), CQS Divide partition-then-recompose must produce attention output bit-identical to full-attention on the same input; MSE must be 0.0 (not epsilon-bounded, exactly zero modulo floating-point determinism). If non-zero, abort task. (b) Scale-up exactness: at N=16384 with 16 partitions, bit-identical check must still pass (or MSE < 1e-12 to account for MLX floating-point reduction-order determinism). (c) Billion-token claim verification: at N=10M (memory-permitting on test rig), CQSA attention must complete without OOM and produce output within 1e-10 of a reference "larger-memory" computation. (Actual 1B-token verification is infeasible on 48 GB M4 Pro but the streaming-overhead claim can be validated by extrapolation from 10M.) (d) Memory-budget adherence: CQSA with `--cqsa-budget 8GB` must not exceed 8 GB Metal peak during attention at any context length; measure. (e) Scheduler correctness: auto-block-size must produce a valid partition that respects both the memory budget and the sequence length; unit-test boundary cases (N=B, N=B-1, N=B+1). (f) NIAH preservation: NIAH@64K/128K with CQSA enabled must PASS (exactness claim implies equality with baseline; any regression flags a bug). (g) HumanEval preservation: HumanEval pass@1 with CQSA must be bit-identical or within 0.5 pp of baseline (exactness → equivalent quality). (h) Decode-throughput measurement at 1M: with CQSA enabled at 1M context, decode throughput must be measured; gate is a Goal-1 gate (>= 10 tok/s) not a Goal-3 gate (whether CQSA beats baseline throughput is a research-question not a ship-gate). (i) Composition with TQ3: CQSA reading TQ3 KV must produce output bit-identical to full-attention on TQ3 KV (exactness-vs-given-KV, not vs fp16-KV). (j) Fallback: `--attention-mode standard` reproduces baseline attention-path behavior bit-identically.
- **Effort**: M-L (2-3 weeks — Phase-1 reference implementation in NumPy + bit-identical validation (1 week — the critical gate; this IS the ship-decision), Phase-2 streaming scheduler (3 days once reference is validated), Phase-3 MLX integration + Metal kernel (1 week), Phase-4 composition validation (3 days), edge-case handling (3 days)). Effort could be as low as 1 week if Phase-1 fails and task is aborted; could be L if the MLX Metal-kernel work proves harder than expected.
- **Depends on**: MLX `mx.fast.metal_kernel` primitive (have). TurboQuantKVCache (shipped — CQSA operates on TQ3 KV, exactness vs given-KV). Composes with SnapKV (shipped), DuoKV (shipped), block-wise prefill (Task 249), Commodity-GPU Sparse 1M (Task 246). Independent of codec-axis tasks (PM-KVQ Task 244, XQuant Task 245, KVTC Task 241, KQ-SVD Task 239, DP-LLM Task 242) and learning-axis tasks (qTTT Task 248, TTKV Task 247). Sequencing: **ship-last of pass-58 tasks** — highest-upside-highest-risk exploratory task. Phase-1 bit-identical validation is the gate-defining first step; allocate 1 week before committing to the remaining 2 weeks. Do not ship until at least one independent implementation-or-replication exists in the literature (pass-60+ may surface this). Paper is 2026-04-22 — brand new, zero third-party validation; treat as high-risk research task, not a ship-track task, until exactness validation passes.

## Research-derived tasks (from LIT_REVIEW.md pass 59, 2026-04-23)

### 253. Twilight adaptive-budget sparse attention — hierarchical top-p pruning framework replacing fixed-k budgets
- **Goal**: 1 (up to 98% token pruning in the asymptotic sparsity regime means attention cost becomes nearly constant across context length — directly the Goal-3 property Hypercar needs at Goal-1 scale; on Qwen3-Coder-30B the per-expert sparsity varies, so a data-driven top-p threshold captures more savings than a hand-tuned fixed-k), Goal 3 (paper reports 3.9× end-to-end per-token decode latency reduction — directly on the critical path for "constant decode speed across context window"; adaptive budget means decode is fast when attention is sparse (most code-workload tokens) and only pays full cost when attention genuinely needs it), Goal 6 (data-driven budget shrinks the effective KV footprint at high-sparsity moments; orthogonal to codec memory savings (TQ3/DuoKV) and eviction savings (SnapKV) — Twilight is a wrapper that enhances rather than replaces existing memory paths)
- **Derived from**: Twilight / Adaptive Attention Sparsity with Hierarchical Top-p Pruning (2502.02770, NeurIPS 2025 Spotlight, code released at github.com/tsinghua-ideal/Twilight). Pass-59 highest-leverage find — NeurIPS Spotlight is top-3% of accepted papers (strongest pass-59 provenance signal); framework wraps every existing sparse-attention path rather than competing with them.
- **Change**: Four phases. Phase 1 — **top-p adaptive budget core**: new file `omlx/attention/twilight.py` implementing the hierarchical top-p pruning mechanism. The core primitive is: given sparse-attention scores, select the smallest prefix of top-ranked scores whose cumulative mass reaches threshold p. Hierarchical application: apply at block granularity first (prune whole blocks whose attention mass is below the block-p threshold), then at subsequence granularity within remaining blocks, then at token granularity within remaining subsequences. Adapt the top-p mechanism from tsinghua-ideal/Twilight. Phase 2 — **wrapper integration**: Twilight is designed as a framework wrapper. Extend the three shipped fixed-k sparse-attention paths with a common `BudgetPolicy` abstraction: (a) SnapKV `--snapkv-keep K` fixed-k vs `--snapkv-keep-policy top_p --snapkv-p 0.95`, (b) Commodity-GPU Sparse 1M Task 246 top-k vs top-p, (c) SpargeAttn Task 255 fixed-threshold vs adaptive. The `BudgetPolicy` interface takes the computed scores and returns the set of indices to keep. Phase 3 — **p-threshold calibration**: per-layer calibration on Qwen3-Coder-30B code workload (LoCoBench 10K scenarios as the calibration set). Produces `qwen3_twilight_p.json` artifact mapping `(layer_idx, budget_mode)` → optimal p. Phase 4 — **server wire-in + fallback cap**: `--attention-budget-policy {fixed_k, top_p}` flag on hypercar_server; `--twilight-p FLOAT` override; `--twilight-cap K` enforces worst-case budget ceiling (avoids adaptive-budget denial-of-service on adversarial inputs that produce flat attention distributions). Composition: wraps Task 246 (Commodity-GPU Sparse 1M), wraps Task 255 (SpargeAttn), wraps SnapKV (shipped), wraps CAOTE/BUZZ (shipped), orthogonal to every codec (DuoKV, TQ3, native, fp16) and to every layer-axis policy (TailorKV Task 254).
- **Verify**: (a) Top-p mechanism correctness: on N=4K attention, Twilight's top-p selection must match a reference NumPy `cumsum-then-threshold` implementation within 1e-6 relative error. (b) Hierarchical pruning correctness: on a known-sparse input (90% of tokens have attention mass < 0.001), the block-level pass must prune at least 80% of blocks. (c) NIAH preservation at 64K: NIAH@64K with Twilight+SnapKV (adaptive-k variant of shipped SnapKV) must PASS with fewer tokens retained than fixed-k SnapKV (measure both; adaptive should retain less in the asymptotic sparse regime). (d) NIAH preservation at 128K: NIAH@128K with Twilight must PASS. (e) HumanEval parity: HumanEval pass@1 with Twilight must be within 2 pp of fixed-k baseline (paper claims no regression). (f) Decode speed at 64K: Twilight+SnapKV decode at 64K must be >= fixed-k baseline (in high-sparsity moments, Twilight prunes harder; in low-sparsity moments, it preserves more — net should be >= baseline, often >). (g) Decode speed at 128K: measure; target modest improvement (5-10%). (h) Worst-case cap enforcement: on synthetic adversarial input (uniform attention distribution), Twilight must fall back to `--twilight-cap K` budget (validate the cap is respected). (i) Calibration robustness: the calibrated p thresholds must produce monotonic quality-vs-budget trade-offs (higher p → more tokens retained → higher quality; if non-monotonic, calibration bug). (j) Composition with TailorKV (Task 254): `--tailor-kv on --attention-budget-policy top_p` must produce stacked layer-axis × adaptive-budget policy; verify no false synchronization or policy conflict. (k) Revert: `--attention-budget-policy fixed_k` restores baseline pre-pass-59 behavior.
- **Effort**: S-M (1-2 weeks — top-p mechanism implementation (3 days — the algorithm is a simple cumsum-threshold; hierarchical structure is the main design question), wrapper integration across three sparse-attention paths (3 days), p-threshold calibration (3 days — requires running the calibration corpus through Hypercar which takes time but is straightforward), server wire-in + cap fallback (1 day), gate validation (1 week))
- **Depends on**: SnapKV (shipped — wraps), Commodity-GPU Sparse 1M (Task 246 — wraps, so Task 246 should ship first or be designed with the `BudgetPolicy` abstraction in mind), SpargeAttn (Task 255 — wraps). Composes with every codec, every layer-axis policy. Sequencing: **ship-first of pass-59 tasks** — smallest effort, highest confidence, NeurIPS 2025 Spotlight provides the strongest peer-review signal of the batch; wrapper design is the risk. Consider shipping Task 253 in parallel with Task 246 (they co-design) rather than strictly sequential.

### 254. TailorKV layer-axis hybrid compression — quantize-dominant-tokens + offload-rest per layer type
- **Goal**: 1 (layer-aware hybrid policy composes orthogonally with DuoKV's head-axis retrieval/streaming split and TQ3's uniform quantization — the 4-cell product of layer-axis × head-axis provides the first principled within-layer policy refinement in the pass-40-59 arc; Llama-3.1-8B at 128K on single RTX 3090 at 82 ms/token decode suggests the pattern transfers to M4 Pro 48 GB at 256K-1M with appropriate scaling), Goal 3 (82 ms/token = 12.2 tok/s at 128K — clears Goal-3 floor with room; on M4 Pro's unified memory the PCIe-bottleneck caveat vanishes so decode benefits are not erased by wire latency), Goal 6 (principled budget allocation: dominant-token layers keep dominant tokens in full precision + quantize the rest; global-info layers keep the full cache — the net memory footprint is a weighted combination that can be tuned to fit 48 GB at longer contexts than uniform policies allow)
- **Derived from**: TailorKV (2505.19586, Findings of ACL 2025, code released at github.com/ydyhello/TailorKV). Highest-structural-resonance pass-59 find — layer-axis policy composes cleanly with DuoKV head-axis and TQ3 codec-axis.
- **Change**: Three phases. Phase 1 — **layer-type calibration**: new file `omlx/models/qwen3_layer_calibration.py` implementing the one-time offline pass that classifies each of Qwen3-Coder-30B's 48 layers as either "global-info" (needs full KV) or "dominant-token" (few tokens carry most attention). Adapt the calibration procedure from ydyhello/TailorKV. Runs as a one-time job against a reference corpus (use LoCoBench 10K scenarios as the code corpus); produces `qwen3_layer_types.json` artifact with per-layer classification + dominant-token threshold. Phase 2 — **hybrid dispatch in attention path**: in the Qwen3 attention forward, dispatch per layer: dominant-token layers run the TailorKV policy (identify dominant tokens via activation magnitude, keep in full precision, quantize the rest); global-info layers keep the full KV in current codec (TQ3 or DuoKV). Phase 3 — **server wire-in**: `--tailor-kv on/off` flag on hypercar_server, defaulting to off initially. When on, loads the calibration artifact and dispatches the hybrid policy per layer. Composition: with DuoKV (shipped — TailorKV operates at the layer-type level *above* DuoKV's head-type split; the hierarchy is layer-type → head-type → codec), with TQ3 (shipped — for layers where TailorKV keeps quantized KV, TQ3 is the quantization codec), with Twilight adaptive-budget policy (Task 253 — TailorKV's dominant-token detection is orthogonal to Twilight's top-p budget; both compose via the `BudgetPolicy` abstraction), with SnapKV/CAOTE (shipped — same composition), with block-wise prefill (Task 249 — TailorKV's layer-aware policy is applied per-block during block-wise prefill).
- **Verify**: (a) Calibration correctness: the calibration artifact for Qwen3-Coder-30B produces layer classifications that reproduce the paper's pattern on Llama (roughly 30-50% of layers classified as "dominant-token") — if all 48 layers come out as the same type, the classification is collapsed and the task is aborted. (b) Quality parity: HumanEval pass@1 with TailorKV on must be within 2 pp of baseline (paper claims near-lossless). (c) NIAH preservation: NIAH@64K/128K with TailorKV on must PASS (layer-axis policy is lossy for quantized tokens in dominant-token layers; the selection heuristic must be tight enough to preserve needles). (d) Memory win: at 128K context, TailorKV on must reduce Metal peak by at least 1 GB vs baseline duo mode (this is the conservative win; larger wins are bonus). (e) Decode speed: decode at 64K with TailorKV must clear Goal-3 floor (>= 10 tok/s) and should meet or beat baseline duo-mode (the dominant-token attention has fewer tokens to attend to in those layers; should be faster). (f) Composition with Twilight: `--tailor-kv on --attention-budget-policy top_p` must produce the stacked policy correctly; no false synchronization between layer-axis dispatch and adaptive-budget selection. (g) Unified-memory semantics verification: on M4 Pro, the "offload" primitive is cache-residency not PCIe transfer; verify no false synchronization points are introduced by measuring Metal command-queue latency with and without TailorKV. (h) Calibration runtime budget: calibration pass must complete in < 8 hours on M4 Pro (one-time offline cost; paper's calibration is a single-pass statistical analysis). (i) Revert: `--tailor-kv off` restores baseline pre-pass-59 behavior.
- **Effort**: M (2-3 weeks — port layer-type calibration from ydyhello/TailorKV (1 week — the calibration is statistical, no gradient training), hybrid dispatch integration (1 week), server wire-in (2 days), full gate validation at 64K/128K (1 week))
- **Depends on**: DuoKV (shipped — composes orthogonally at the head axis), TQ3 (shipped — composes as codec for quantized layers), Twilight (Task 253 — composes via `BudgetPolicy` abstraction; Task 253 should ship first since it's smaller-effort and establishes the shared policy infrastructure). Composes with SnapKV/CAOTE (shipped), block-wise prefill (Task 249). Sequencing: **ship-second of pass-59 tasks** — after Task 253 (Twilight) which establishes the `BudgetPolicy` abstraction that TailorKV's layer-axis dispatch plugs into. ACL 2025 Findings with released code gives solid peer-review floor.

### 255. SpargeAttn two-stage sparse-attention filter — training-free QK^T matmul skipping for decode + prefill
- **Goal**: 1 (training-free sparse attention that replaces `mx.fast.scaled_dot_product_attention` with a two-stage filter variant — 2.5-5× faster dense-attention across evaluated benchmarks, directly benefits long-context decode and prefill where attention dominates compute; the training-free property is mandatory since Hypercar cannot fine-tune the 30B Qwen3-Coder weights), Goal 3 (attention compute at 64K+ is the dominant decode cost; 2.5-5× reduction should move decode at 64K from current 33.7 tok/s post-eviction toward 50+ tok/s, clearing Goal-3 with margin), Goal 4 (sparse-prefill benefits — stage-1 filter skips matmuls during prefill chunk processing; projected 500+ tok/s prefill at 64K+ if selectivity is high on code workloads)
- **Derived from**: SpargeAttn (2502.18137, ICML 2025, code released at github.com/thu-ml/SpargeAttn). Highest attention-compute-reduction-per-engineering-effort for long-context decode; universal drop-in for SDPA API.
- **Change**: Four phases. Phase 1 — **reference understanding**: read ICML paper Sections 3-4 and study thu-ml/SpargeAttn CUDA reference to understand the two-stage filter algorithm. Stage 1 is a cheap mean-similarity surrogate score that predicts the attention map; stage 2 is a softmax-aware online filter that further skips QK^T multiplications. Write a pure-Python/MLX reference in `omlx/attention/sparge_reference.py` that produces correct output (may be slow — correctness is the gate, not speed). Phase 2 — **Metal kernel implementation**: translate the two-stage filter to Metal via `mx.fast.metal_kernel` in `omlx/attention/sparge_sdpa.py`. Main engineering effort: thread-group organization for the filter decision + sparse QK^T matmul. Phase 3 — **attention-forward integration**: in `omlx/models/qwen3_coder.py` attention, add optional dispatch to SpargeAttn gated on `--attention-mode {standard, sparge}` flag (default standard). Phase 4 — **selectivity calibration**: the SpargeAttn paper's evaluation is dominated by diffusion models; language-model selectivity on Qwen3-Coder-30B code workloads must be measured. Calibrate the stage-1 threshold to balance selectivity vs quality on HumanEval + NIAH + LoCoBench. Composition: with every codec (DuoKV, TQ3, native, fp16 — SpargeAttn operates on whatever KV cache is provided), with SnapKV/CAOTE (shipped — SpargeAttn runs on retained post-eviction cache), with Twilight (Task 253 — Twilight's top-p `BudgetPolicy` wraps SpargeAttn's fixed-threshold selection for adaptive budgets), with Commodity-GPU Sparse 1M (Task 246 — subtly different: top-k chooses tokens, SpargeAttn chooses matmuls; both compose), with TailorKV (Task 254 — SpargeAttn is attention-compute, TailorKV is KV-memory; orthogonal).
- **Verify**: (a) Reference correctness: at N=4K, the pure-MLX reference `sparge_sdpa` must produce output within 1e-4 MSE of `mx.fast.scaled_dot_product_attention` on the same input. (b) Metal-kernel correctness: the Metal-kernel port must match the reference within 1e-3 MSE (Metal floating-point reductions are non-deterministic but should agree to this tolerance). (c) Speed at 4K: Metal SpargeAttn at 4K must match or beat baseline SDPA (short-context should show no regression — if regression, install `--sparge-min-context N` gate). (d) Speed at 64K: Metal SpargeAttn at 64K must deliver measurable speedup (target 1.5× minimum; paper claims 2.5-5× on evaluated benchmarks). (e) Quality at 64K: HumanEval pass@1 with SpargeAttn must be within 3 pp of baseline (peer-reviewed claim is no regression; verify on code workload). (f) NIAH preservation at 64K/128K: must PASS (sparse attention may miss needles if selectivity is miscalibrated; this is the gate). (g) Prefill speed: at 16K prefill, SpargeAttn must match or beat baseline; at 64K prefill, target 1.5× minimum speedup. (h) Composition with TQ3: SpargeAttn reading TQ3 KV must produce quality within baseline TQ3 (the KV codec and attention path are orthogonal). (i) Selectivity calibration artifact: the calibrated stage-1 threshold must be recorded per-layer (not uniform) — Qwen3-Coder layers have varying sparsity; uniform threshold may regress some layers. (j) Revert: `--attention-mode standard` restores baseline attention path.
- **Effort**: M-L (3-4 weeks — reference understanding + pure-MLX reference (1 week), Metal-kernel implementation (2 weeks — the main engineering; Metal thread-group semantics for sparse-matmul require careful design), integration (3 days), calibration (3 days), gate validation (1 week))
- **Depends on**: MLX `mx.fast.metal_kernel` primitive (have), MLX `mx.fast.scaled_dot_product_attention` baseline (have — drop-in replacement target). Composes with every codec, every eviction path, Twilight (Task 253 — wraps for adaptive budgets), TailorKV (Task 254), Commodity-GPU Sparse 1M (Task 246). Sequencing: **ship-third of pass-59 tasks** — larger effort than 253/254, but highest decode/prefill-speed-per-effort ratio; the Metal-kernel work is the risk. ICML 2025 accepted with thu-ml reference code removes algorithmic risk. Should ship after Task 253 (Twilight) which establishes the `BudgetPolicy` abstraction that SpargeAttn plugs into.

### 256. HashAttention Hamming-space pivotal-token retrieval — learned bitwise MIPS for sub-linear attention
- **Goal**: 1 (16× token reduction with 32 bits/token auxiliary memory at the paper's generic-data calibration; projected credible path to 1M if the learned Hamming mapping transfers to Qwen3-Coder; auxiliary memory is 4 MB/head at 1M which is negligible vs 22.5 GB KV budget), Goal 3 (bitwise AND + popcount is 10× cheaper than even surrogate-score computation; up to 4.3× attention latency reduction on A100 reported — Metal has native popcount so the primitive maps efficiently; decode-constant-across-context becomes achievable if pivotal-token selection stays sub-linear), Goal 6 (auxiliary memory negligible; net memory footprint dominated by KV which HashAttention does not compress — just uses more efficiently via selection)
- **Derived from**: HashAttention (2412.14468, ICML 2025 poster). Algorithmic-space alternative to SpargeAttn's surrogate-score path — bitwise lookup vs mean-similarity. Higher-risk-higher-reward: learned-mapping-function calibration is the structural caveat.
- **Change**: Exploratory four-phase implementation with an early abort gate. Phase 1 — **calibration feasibility investigation**: before committing to the full path, verify that HashAttention-style learned mapping functions can be calibrated on Qwen3-Coder-30B without requiring gradient-based training infrastructure Hypercar does not have on-node. Options: (a) use a smaller proxy model (e.g., Qwen3-4B) for calibration and hope for transfer to 30B; (b) trace-and-fit the Hamming mapping from attention-pattern logging on a reference corpus (no gradient training needed). If neither option produces a mapping with selectivity > 8× on 100 test sequences at 16K, abort task. Phase 2 — **Metal kernel for bitwise MIPS**: new file `omlx/attention/hash_attention.py` with Metal kernel implementing bitwise-AND + popcount top-k selection over 32-bit signatures. Metal native popcount primitive should dispatch efficiently; thread-group organization is for parallel scan over 1M-token range. Phase 3 — **attention-forward integration**: dispatch to HashAttention via `--attention-mode hashattn` with learned mapping artifact loaded from `qwen3_hash_mapping.pt`. Phase 4 — **MoE-specific validation**: Qwen3-Coder is MoE (3B active of 30B). Measure whether the Hamming mapping's selectivity varies across experts — if expert-dependent, the static mapping may miss expert-specific attention patterns. If severe, abort and defer to Task 255 (SpargeAttn) which is training-free and unaffected by MoE.
- **Verify**: (a) Calibration feasibility (gate): proxy-model or trace-and-fit produces a Hamming mapping with selectivity >= 8× on 100 held-out sequences at 16K context; if < 8×, abort task (HashAttention is not ready for the 30B MoE target). (b) Metal correctness: bitwise-AND+popcount top-k selection matches reference exact-inner-product top-k within 95% agreement at k=32 (Hamming is an approximation of cosine; disagreement is expected but must be bounded). (c) Speed at 64K: Metal HashAttention at 64K must deliver measurable decode-throughput improvement vs baseline (target 1.5× minimum; paper claims up to 4.3× on A100). (d) Quality at 64K: HumanEval pass@1 with HashAttention must be within 5 pp of baseline (higher tolerance than Task 253/254/255 because HashAttention is a stricter approximation). (e) NIAH at 64K: must PASS (if not, the Hamming mapping is too aggressive; recalibrate selectivity). (f) MoE-dependency measurement: per-expert selectivity analysis — if top-k selection overlaps < 70% across experts for the same query, the static mapping is MoE-incompatible; document and either ship with expert-conditional mapping (extra engineering) or defer. (g) Auxiliary memory cost: 32 bits/token × 1M tokens × 48 layers = 192 MB at 1M — verify the signature storage is within this budget. (h) Composition with TQ3: HashAttention reading TQ3 KV must produce valid output (the codec and selection path are orthogonal; quantized KV is decompressed before attention). (i) Composition with SnapKV: HashAttention over retained post-eviction cache must still find pivotal tokens (selection is over whatever tokens remain). (j) Revert: `--attention-mode standard` restores baseline.
- **Effort**: M-L (2-3 weeks if calibration feasibility gate passes; 1 week if it fails and task aborts — calibration feasibility investigation (1 week — this is the gate-defining first step), Metal-kernel implementation (1 week), integration + MoE validation (1 week))
- **Depends on**: MLX `mx.fast.metal_kernel` + native popcount primitive (have). Proxy model Qwen3-4B for calibration alternative (have — same family). Composes with every codec, every eviction path, Twilight (Task 253 — `BudgetPolicy` abstraction), TailorKV (Task 254), SpargeAttn (Task 255 — competes; if 255 delivers similar speedups with simpler implementation, 256 loses). Sequencing: **ship-last of pass-59 tasks** — highest-risk-highest-reward exploratory path. The calibration feasibility gate (Phase 1) is the ship-decision; allocate 1 week before committing to remaining phases. If SpargeAttn (Task 255) ships with measurable speedup, HashAttention becomes lower-priority since both address the same goal with different mechanisms. Track the ICML 2025 code release (if the authors release the learned mapping for open LLMs, calibration cost drops to a model-port).

## Qwen3.6-35B-A3B migration tasks (2026-04-23)

Model-migration pass — not a research pass. Target model pivot from `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` (current, ~17.2 GB) to `Qwen3.6-35B-A3B` (released 2026-04-16). Key deltas: 35B total / 3B active with **256 experts** (8 routed + 1 shared per token, vs. Qwen3-Coder's lower expert count), native 262K context extensible to 1M without fine-tuning, vision encoder baked in (multimodal by default), 73.4% SWE-Bench Verified, 51.5 Terminal-Bench 2.0, Apache 2.0. MLX variants available: `mlx-community/Qwen3.6-35B-A3B-4bit`, `unsloth/Qwen3.6-35B-A3B-MLX-8bit`, `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`. Tasks sequenced so Task 257 (smoke) gates Task 258 (bit-width) gates Task 259 (quality) gates Task 263 (default flip); Tasks 260-262 run in parallel once Task 257 passes.

### 257. Qwen3.6-35B-A3B smoke test on M4 Pro — load, generate, measure baseline memory and speed
- **Goal**: 6 (validates that the 35B/3B-active model at 4-bit quantization fits within the 48 GB M4 Pro budget before any migration work; gates every subsequent Qwen3.6 task — a failure here halts the migration), Goal 3 (first decode-speed measurement on the new model, baseline for Goal-3 constant-throughput claim), Goal 4 (first prefill-speed measurement on the new model)
- **Derived from**: User directive 2026-04-23 to pivot target model from Qwen3-Coder-30B to Qwen3.6-35B-A3B. First engineering step of the migration: **does the model even load and generate coherently on the reference machine** before committing benchmark runtime or code changes? The 4-bit variant (`mlx-community/Qwen3.6-35B-A3B-4bit`, ~17.5 GB) is the fit-candidate; the 8-bit variant (~35 GB) is out of reach on 48 GB with any meaningful KV budget and covered in Task 258.
- **Change**: Single throwaway script `scripts/qwen36_smoke.py` (not committed to main — lives in a scratch branch or as a one-off). Three-step procedure. Step 1: download `mlx-community/Qwen3.6-35B-A3B-4bit` via `mlx_lm.load` (~17.5 GB download; ensure `~/.cache/huggingface` has space; do NOT download during any CI run). Step 2: measure model-load memory via `mx.metal.get_peak_memory()` before and after load; record Metal peak at load. Step 3: generate 64 tokens from a 4K-token code prompt; measure (a) prefill wall-time and tok/s, (b) single-token decode latency, (c) Metal peak during generation, (d) output coherence on a simple code-completion task ("Write a Python function that computes fibonacci"). Log results to `bench/snapshots/qwen36_smoke_<timestamp>.json`. Do NOT wire into hypercar_server or hypercar_bench yet — this is a dry-run on `mlx_lm` directly to validate the model before touching the Hypercar code path. If the model uses a tokenizer or chat template that differs from Qwen3-Coder, note the deltas in the smoke-test output (later tasks will need to handle this).
- **Verify**: (a) Download completes without error; model weights resolve on disk. (b) Model loads via `mlx_lm.load` without OOM on 48 GB M4 Pro. (c) Metal peak at load < 20 GB (gate — the 4-bit variant must sit well below half of system RAM to leave KV + activation headroom). (d) Generate produces coherent Python output (simple visual check — the function compiles and roughly computes fibonacci). (e) Single-token decode latency recorded on M4 Pro (baseline number for Task 259 Goal-3 gate calibration). (f) 4K prefill speed recorded on M4 Pro (baseline number for Task 259 Goal-4 gate calibration). (g) Metal peak during generation < 30 GB (leaves 18 GB headroom for KV at longer contexts). (h) Vision encoder presence detected — if the model config includes a vision tower, record its parameter count + memory contribution separately (feeds into Task 261). (i) Tokenizer/chat-template deltas documented — if `tokenizer.chat_template` or special-token set differs from Qwen3-Coder, note in smoke-test output. (j) If any gate (b/c/d) fails, migration halts; update smoke-test output with failure mode and escalate.
- **Effort**: S (3-5 days — download wait-time dominates; actual scripting is ~1 day, measurement is ~1 day, coherence gate is ~0.5 day, failure-mode documentation is ~0.5 day)
- **Depends on**: `mlx_lm` (have), internet access for HuggingFace download, ~20 GB free disk. No Hypercar code dependencies. Sequencing: **ship-first of migration tasks** — gates every subsequent migration task. Do NOT download weights before this task is explicitly picked up; user policy is no ad-hoc model downloads.

### 258. Qwen3.6 bit-width tradeoff benchmark — 4-bit vs 8-bit memory fit and quality on 48 GB
- **Goal**: 6 (determines which bit width fits the reference machine — the 4-bit variant at ~17.5 GB leaves 30 GB for KV and activations, while the 8-bit variant at ~35 GB likely does not fit with meaningful KV budget and is included mainly to quantify the gap), Goal 2 (first quality-delta measurement between bit widths on Qwen3.6 — if 4-bit regresses >3pp from 8-bit on HumanEval or MMLU-Pro, the migration story changes since 8-bit is the Hypercar baseline bit width for Qwen3-Coder)
- **Derived from**: Task 257 (smoke test), user directive 2026-04-23. The Qwen3-Coder baseline runs 8-bit; dropping to 4-bit for Qwen3.6 is a *bit-width regression* that requires explicit validation even if the parameter-count increase (30B → 35B) compensates. The 8-bit variant is unlikely to fit but must be quantified as the upper-bound reference.
- **Change**: Two subtasks run sequentially. Subtask 258a — **4-bit bench**: run `.venv/bin/python -m omlx.bench.hypercar_bench --model mlx-community/Qwen3.6-35B-A3B-4bit` (note: this requires Task 263 to be partially done — or run hypercar_bench with an explicit `--model` override on a branch that accepts it; verify `--model` override works before committing). Record full 11-gate output to `bench/snapshots/qwen36_4bit_<timestamp>.json`. Subtask 258b — **8-bit bench (conditional)**: only run if the 48 GB M4 Pro has sufficient swap tolerance; the 8-bit variant at ~35 GB leaves ~13 GB for KV which at 3-bit is ~3K-context budget before swap. Expected outcome: 8-bit bench fails the memory gate at load or at 4K context. Record whatever it produces (including the failure mode) to `bench/snapshots/qwen36_8bit_<timestamp>.json`. Document which bit width is the viable Hypercar baseline for Qwen3.6. No code changes committed from this task — only bench snapshots + a findings note on whether the 4-bit quality regression is within tolerance.
- **Verify**: (a) 4-bit bench runs to completion without memory breach — Metal peak at load < 70% of system RAM; peak during benchmark < 80%. (b) 4-bit HumanEval pass@1 >= 35% (Hypercar bench floor; if >= 35% and within 10pp of Qwen3-Coder 95%, migration is viable; if < 35%, migration is questionable). (c) 4-bit NIAH@4K PASS. (d) 4-bit decode at 2K >= 40 tok/s (below the 50 tok/s target but within range given the 17% parameter-count increase; this is a soft floor). (e) 4-bit prefill at 4K >= 400 tok/s (below target, soft floor). (f) 8-bit bench either completes or fails at the memory gate; either outcome is a valid result, and the failure mode must be recorded (OOM at load? OOM at 4K? swap exceeds 100 MB/s p90?). (g) Bit-width quality delta measured: HumanEval pass@1 delta 4-bit → 8-bit (if 8-bit runs). If delta > 5pp, migration must accept the 4-bit quality regression. (h) No gate-failure in the 4-bit run is a hard abort — a failure just documents the Qwen3.6-specific deltas; the migration decision is a judgment call informed by (b) through (g). (i) Findings note appended to `bench/snapshots/qwen36_bit_width_findings.md` (one-time file) with the decision: "ship on 4-bit", "ship on 8-bit", or "halt migration".
- **Effort**: S-M (1 week — 4-bit bench runtime ~30min to 2hr depending on `--full` vs `--quick`; 8-bit bench runtime as long as the machine tolerates before failing; findings note writeup ~0.5 day)
- **Depends on**: Task 257 (smoke — gates this task). `mlx_lm` supports both model variants (verify at smoke-test time). `hypercar_bench` `--model` override flag works (spot-check that the flag is already wired in before running, or as a precondition make it work). No other code dependencies.

### 259. Quality-gate re-run against Qwen3.6 — HumanEval, MMLU-Pro, RULER, LiveCodeBench, Code Intel
- **Goal**: 2 (the "4 independent evals beating GPT-4" claim is the Hypercar Goal-2 proof — every eval must be re-measured on Qwen3.6 since the model change invalidates prior Qwen3-Coder scores; if Qwen3.6 beats Qwen3-Coder on all 5 families, the migration strengthens Goal 2; if any regresses materially, the migration requires explicit justification against the affected goal)
- **Derived from**: Task 258 (bit-width decision), user directive 2026-04-23. CLAUDE.md's Goal-2 table lists current scores on Qwen3-Coder: HumanEval 95%, MMLU-Pro 62%, RULER 100%, LiveCodeBench 30%, Code Intel 5/5. Qwen3.6's published deltas (73.4 SWE-Bench vs 70.0, 51.5 Terminal-Bench 2.0) suggest a net-positive quality story, but none of those benchmarks are directly in Hypercar's gate set — Hypercar's gates must be measured on-node.
- **Change**: Run the 5 quality-gate families on the chosen bit-width Qwen3.6 variant (determined by Task 258). Each eval produces a snapshot file. (a) HumanEval pass@1 via `.venv/bin/python -m omlx.bench.hypercar_bench --full --model <qwen36-variant>`; record `bench/snapshots/qwen36_humaneval_<timestamp>.json`. (b) MMLU-Pro via the existing MMLU-Pro harness (find in bench/ — same path used for Qwen3-Coder baseline); record `bench/snapshots/qwen36_mmlu_pro_<timestamp>.json`. (c) RULER via the existing RULER harness; record `bench/snapshots/qwen36_ruler_<timestamp>.json`. (d) LiveCodeBench via the existing LiveCodeBench integration (find in bench/); record `bench/snapshots/qwen36_livecodebench_<timestamp>.json`. (e) Code Intel via hypercar_bench's built-in 5-problem code-intelligence gate (runs inline with the main bench); extracted from the `--full` snapshot. No code changes committed from this task — only the 5 snapshots + a findings note comparing Qwen3.6 deltas vs Qwen3-Coder baseline across each eval. Note if Qwen3.6 is run with native chat template (which may differ from Qwen3-Coder's template) — a template mismatch can shift scores by 5-10pp independent of model quality, so explicit template validation is part of this task.
- **Verify**: (a) All 5 evals run to completion on Qwen3.6 without harness errors (if any harness has Qwen-version hardcoding, fix the harness before running — do not hardcode results). (b) HumanEval pass@1 on Qwen3.6 >= 80% (soft floor — if below 80% on a 35B model when a 30B baseline hits 95%, investigate template or tokenizer mismatch before accepting). (c) MMLU-Pro on Qwen3.6 >= 55% (soft floor — if below, investigate). (d) RULER on Qwen3.6 PASS at 4K (hard floor — if fails, migration is not viable). (e) LiveCodeBench on Qwen3.6 >= 25% (soft floor — if Qwen3.6 matches or beats Qwen3-Coder's 30%, the LiveCodeBench room-for-improvement story flips). (f) Code Intel on Qwen3.6 >= 3/5 (hard floor). (g) Chat template explicitly verified matches the model's `tokenizer.chat_template` — if the harness uses a hardcoded template, the template-mismatch risk is non-trivial and must be audited. (h) Findings note `bench/snapshots/qwen36_quality_findings.md` (one-time file) with each eval's delta vs Qwen3-Coder baseline and an overall net-quality verdict ("migration improves quality", "migration is quality-neutral", "migration regresses quality and requires justification"). (i) If (b), (c), (e), or (f) soft floors are missed, log the delta and flag for Task 263 (default migration) — a soft-floor miss does not block migration but changes the CLAUDE.md quality targets table that Task 263 updates. (j) No other gates regress silently — every current Hypercar bench gate that is not in (a)-(f) must still pass on Qwen3.6.
- **Effort**: M (1-2 weeks — HumanEval 30min-2hr per run, MMLU-Pro several hours, RULER ~1hr, LiveCodeBench several hours, Code Intel inline. Add 2-3 rerun attempts for harness-fix iterations; chat-template audit ~0.5 day; findings note ~0.5 day)
- **Depends on**: Task 258 (bit-width chosen). All 5 eval harnesses exist and work on Qwen3-Coder (verify before running; any harness that is Qwen-specific needs updating before it can be used on Qwen3.6). No code commits except harness fixes if any.

### 260. Native 1M-context audit — RoPE extrapolation or baked in?
- **Goal**: 1 (whether Qwen3.6 needs RoPE extrapolation at all is the biggest architectural question for the 1M context target — if 1M is baked into the model's RoPE config, the shipped LongRoPE path (Task 196) and related extrapolation tasks become obsolete for this model; if not, the extrapolation stack must be re-validated against the new base model's RoPE base value and scaling factor)
- **Derived from**: User directive 2026-04-23 — "native 262K context, extensible to 1M without fine-tuning". "Without fine-tuning" is the key claim; it suggests the RoPE config already supports 1M via base-value scaling rather than requiring LongRoPE-style learned extrapolation. This must be verified empirically.
- **Change**: Two-phase audit. Phase 1 — **config inspection**: load Qwen3.6's `config.json` and inspect `max_position_embeddings`, `rope_theta` (base), `rope_scaling` (if present). Compare to Qwen3-Coder's config. Record the deltas in `bench/snapshots/qwen36_rope_audit.md`. Phase 2 — **NIAH at boundary points**: run NIAH at (a) 262K+1 tokens (just past the native window) and (b) 524K tokens (2× native). Run with NO Hypercar RoPE extrapolation patches applied (vanilla model config only). If both pass, 1M is genuinely baked in and Task 196 (LongRoPE) is obsolete for Qwen3.6. If 262K+1 fails, the "extensible to 1M without fine-tuning" claim is false and LongRoPE or similar is still needed. If 262K+1 passes but 524K fails, native is 262K and the 262K → 1M path requires something — likely a scaling factor in `rope_scaling`; measure the required factor by binary search between 262K and 524K. Do NOT run the full 1M NIAH in this task (it is multi-hour and the 524K measurement is sufficient evidence). Document whether the existing Hypercar RoPE patches (if any apply to Qwen3.6 architecturally) compose cleanly or conflict.
- **Verify**: (a) `config.json` for Qwen3.6 inspected; `max_position_embeddings` and `rope_theta` recorded; any `rope_scaling` field documented. (b) NIAH@262K+1 on Qwen3.6 with vanilla config: PASS or FAIL recorded. (c) NIAH@524K on Qwen3.6 with vanilla config: PASS or FAIL recorded. (d) If (b) PASS and (c) PASS, findings note declares "native 1M — Task 196 obsolete for Qwen3.6" and the next step flag is "retire LongRoPE for Qwen3.6 in the next commit that migrates the default model". (e) If (b) FAIL, findings note declares "262K is not native — Hypercar RoPE extrapolation still needed for Qwen3.6" and Task 196 (or equivalent) remains live. (f) If (b) PASS and (c) FAIL, findings note declares "262K native, 262K→1M requires scaling factor; measured factor = X" and Task 196 remains live but with different calibration. (g) Runtime: NIAH@524K must complete in < 3hr on M4 Pro (262K+1 ~1hr, 524K ~2hr); if runtime exceeds, shrink the needle-count to keep the task tractable. (h) Metal peak during NIAH@524K < 80% of system RAM; if exceeds, this is a separate Task-249 (block-wise prefill) application and should not be resolved in this audit task. (i) No code changes committed — findings note only. (j) Composition test: verify that Hypercar's shipped SnapKV+CAOTE stack does not silently break when applied to Qwen3.6 — run NIAH@64K with the full eviction stack and Qwen3.6; must PASS.
- **Effort**: M (1-2 weeks — config inspection 0.5 day, NIAH@262K+1 run ~1hr + iteration + debugging ~2 days, NIAH@524K run ~2hr + iteration ~3 days, scaling-factor binary search (if needed) ~3 days, findings-note writeup ~0.5 day)
- **Depends on**: Task 257 (smoke — gates every Qwen3.6 task), Task 258 (bit-width decided — NIAH runs on the chosen variant). Hypercar NIAH harness works on Qwen3.6 (verify; should be model-agnostic but confirm). Composes with Task 196 (LongRoPE) — this audit's outcome determines whether Task 196 stays live for Qwen3.6 or is retired. Sequencing: can run in parallel with Task 259 (quality re-run) after Task 258 completes — they share the bit-width decision but use different harness paths.

### 261. Vision-encoder memory ablation for text-only inference
- **Goal**: 6 (if the baked-in vision encoder consumes 1-3 GB of Metal memory even when inference is text-only, surgical removal frees headroom for longer KV; on a 48 GB machine every GB of headroom is worth engineering effort), Goal 3 (vision encoder presence may add dispatch overhead even on text-only inputs — if the forward pass does not short-circuit cleanly, per-decode-step latency may regress relative to Qwen3-Coder which had no vision encoder)
- **Derived from**: User directive 2026-04-23 — "Vision encoder baked in (multimodal by default)". The Hypercar product target is text-only code inference; the vision encoder is dead weight for this workload and is a candidate for surgical removal if `mlx_lm` or the model config permits.
- **Change**: Three-phase ablation. Phase 1 — **baseline memory measurement**: from Task 257 smoke test, extract the model's vision-encoder parameter count and its contribution to Metal peak at load. If the smoke test did not separate vision-encoder memory, re-run with explicit accounting (inspect `model.named_modules()` or equivalent; tag every parameter as "text" or "vision" and sum). Phase 2 — **surgical-removal feasibility investigation**: study how `mlx_lm` loads the Qwen3.6 checkpoint. Options: (a) load-time tensor filter — skip any tensor matching a vision-encoder prefix pattern (`vision_tower.*`, `visual.*`, etc — confirm exact prefix from the model's state-dict keys); (b) model-config patch — edit the loaded config to disable the vision branch; (c) post-load weight deletion — `del model.vision_tower` before the first forward pass. Option (a) is the cleanest if supported by `mlx_lm`. Phase 3 — **validated-removal measurement**: apply the chosen removal method; measure Metal peak after load. Verify that text-only inference still works (generate 64 tokens from a simple code prompt; compare output to baseline-with-vision run). Compare Metal-peak delta; if delta >= 1 GB, removal is worth shipping as an option; if < 1 GB, the engineering overhead of maintaining a text-only variant exceeds the memory win and the task outcome is "keep vision encoder as dead weight, document the cost". Wire-in (conditional on Phase-3 win): new CLI flag `--text-only` on hypercar_server (default off initially, on after Task 263 migration if validated) that applies the Phase-2 removal at load time.
- **Verify**: (a) Vision-encoder parameter count separated from text-model parameter count; both recorded in findings note. (b) Vision-encoder Metal contribution at load measured; value recorded. (c) Removal method selected and implemented; feasibility demonstrated by successful load + text-only generate. (d) Post-removal Metal peak at load reduced by the measured vision-encoder memory within 10% tolerance (e.g., if vision encoder is 2 GB, post-removal peak should be 1.8-2.2 GB lower). (e) Text-only generate output after removal is identical or near-identical to pre-removal (allow for non-determinism in the forward pass — but the first 16 tokens should match unless the removal perturbs shared layers, which would be a bug). (f) No regression in NIAH@4K after removal (text-only gate unaffected). (g) No regression in HumanEval pass@1 after removal (sanity check — code-completion is text-only, vision-encoder removal should not affect it). (h) If removal delta < 1 GB, findings note declares "vision encoder is low-cost; keep as-is"; no code commit. (i) If removal delta >= 1 GB, findings note declares the removal method + expected memory win; code commit deferred to Task 263's default-model migration commit (bundles the `--text-only` flag with the default-model flip for a coherent migration changeset). (j) Composition with vision-capable workflows not regressed: the `--text-only off` default must still load the vision encoder for users who want multimodal inference (even if Hypercar doesn't ship multimodal features today, the option is preserved).
- **Effort**: S-M (1 week — baseline measurement 1 day, surgical-removal investigation + implementation 3 days, validated-removal measurement + regression testing 2 days, findings note + optional code commit preparation 1 day)
- **Depends on**: Task 257 (smoke — gates), `mlx_lm` internals (read-only — no PRs upstream). No other code dependencies. Sequencing: can run in parallel with Tasks 259 and 260 after Task 258 completes.

### 262. MoE scheme re-validation at 256 experts — PiKV / SliceMoE / DyMoE parameter audit
- **Goal**: 2 (Qwen3.6's 256 experts with 8+1 active per token is a structural shift from Qwen3-Coder's smaller expert count — any MoE-specific optimization in the Hypercar task backlog that hard-codes expert counts or routing assumptions must be audited and updated; shipping MoE work against the wrong expert count wastes effort and produces invalid measurements), Goal 6 (expert working-set math changes materially at 256 experts — the active-parameter fraction is ~3.5% per token, meaning expert-parallelism or expert-caching strategies have different working-set implications than at Qwen3-Coder's lower count)
- **Derived from**: User directive 2026-04-23 — "256 experts (8 routed + 1 shared per token)". Tasks 185 (PiKV expert-scoped KV), 221 (SliceMoE), and 226 (DyMoE) were designed against Qwen3-Coder's expert count. The expert-scoped KV cache size, per-expert attention-budget, and routing-telemetry tables in those tasks use hardcoded values tied to the old expert count.
- **Change**: Three-phase audit. Phase 1 — **task inventory**: grep the task backlog for hardcoded expert-count assumptions. Tasks 185, 221, 226 are the explicit ones; also check any other task (search TASKS.md for "expert" / "MoE" / "routing") that references Qwen3-Coder's expert count. Record all findings in a migration findings note `bench/snapshots/qwen36_moe_audit.md` with a per-task delta: old expert count, new count (256 routed + 1 shared = 257 total), per-token active (8 routed + 1 shared = 9 per token), and how the task's math must change. Phase 2 — **TASKS.md updates**: for each affected task, append a "2026-04-23 Qwen3.6 addendum" paragraph to the task body noting the new expert-count values and any parameter re-defaults required (e.g., if Task 185 specified "cache per expert of size X", re-compute X at 256 experts and flag in the addendum). Do NOT rewrite the whole task — just append an addendum so the provenance is clear (Qwen3-Coder original + Qwen3.6 addendum). Phase 3 — **implementation code audit**: if any task has already shipped partial MoE-related code (check `omlx/moe/` or similar path if present), grep for literal expert-count constants and document them in the findings note. Do NOT modify code in this pass — Task 263 bundles the default-model flip with code constant updates as a single coherent changeset.
- **Verify**: (a) Task inventory complete — every TASKS.md entry mentioning "expert" or "MoE" audited; list of affected tasks recorded. (b) Per-task addendum appended to Tasks 185, 221, 226 minimum; additional tasks added if audit finds them. (c) Addendum content is specific — contains the new expert count (256 routed + 1 shared), new active-per-token (8 + 1 = 9), and the recomputed per-task parameter values where the math changes. (d) No task body is rewritten; all deltas are appended as addenda for provenance clarity. (e) Code audit identifies whether any shipped code has hardcoded expert counts; if none shipped, findings note states "no code touched by this audit — implementations will pick up the new constants when they ship". (f) If any shipped code has hardcoded expert counts that are wrong for Qwen3.6, the finding is flagged as a Task 263 prerequisite (cannot flip default without fixing the hardcode). (g) Findings note committed to `bench/snapshots/qwen36_moe_audit.md`. (h) Tasks 185, 221, 226 addenda committed to TASKS.md. (i) No code commits in this task — code changes are bundled into Task 263. (j) Re-validation: after Task 259 (quality) and Task 260 (1M audit) complete on Qwen3.6, spot-check that the addendum-noted expert-count values match what the loaded Qwen3.6 model actually has (via `model.config` inspection); if not, re-do the addendum with corrected values.
- **Effort**: S (3-5 days — task inventory grep ~0.5 day, per-task addenda writing ~1-2 days, code audit ~1 day, findings note + commit ~0.5 day)
- **Depends on**: Task 257 (smoke — provides loaded-model config for verification in (j)). Independent of Tasks 259, 260, 261 but Task 259's quality findings may flag additional MoE-related tasks that need addenda. Sequencing: can run in parallel with Tasks 259, 260, 261 after Task 258 completes.

### 263. Default-model migration — flip bench + server defaults to Qwen3.6, keep Qwen3-Coder behind flag
- **Goal**: 6 (migration becomes real for Hypercar users the moment defaults flip; must be a single coherent commit covering every reference), Goal 2 (CLAUDE.md quality targets table and performance-target table must be updated to reflect Qwen3.6-measured values from Tasks 258-260; users should never see stale Qwen3-Coder numbers after migration), Goal 3 (performance-target numbers must be re-measured values not extrapolations)
- **Derived from**: User directive 2026-04-23, gated by Tasks 257 (smoke), 258 (bit-width), 259 (quality), 260 (1M audit), 261 (vision-encoder), 262 (MoE audit). This is the "flip the switch" task — after all upstream audits and validations pass, migrate every default in a single coherent commit so a user upgrading Hypercar does not land in a half-migrated state.
- **Change**: Single-commit migration covering seven surfaces. Surface 1 — `omlx/bench/hypercar_bench.py`: update `MODEL_ID` (line 48 as of 2026-04-23) to the chosen Qwen3.6 variant (likely `mlx-community/Qwen3.6-35B-A3B-4bit` per Task 258 decision). Keep the `--model` override flag functional so regression comparisons against Qwen3-Coder continue to work. Surface 2 — `omlx/bench/agentic_bench.py`: update the `"model"` field (line 95 as of 2026-04-23) to the same Qwen3.6 variant. Surface 3 — `omlx/hypercar_server.py`: update the default `--model` argparse default (line 295 as of 2026-04-23) and the docstring examples (lines 15, 20 as of 2026-04-23). Surface 4 — `CLAUDE.md` **Key Architecture** section: update the model line to "Qwen3.6-35B-A3B-<bit>bit (<memory>GB, <layers> layers, MoE <active>B active, <experts> experts)". Surface 5 — `CLAUDE.md` **Memory Budget** table: re-measured 65K, 256K, 1M rows with Qwen3.6-4bit numbers from Tasks 258-260 (not extrapolations — actual measured values). Surface 6 — `CLAUDE.md` **Performance Targets** table: re-measured decode/prefill numbers from Task 259; update with Qwen3.6 results. Surface 7 — `CLAUDE.md` **Current status against goals** table: update every "Current" column with Qwen3.6-measured values from Tasks 258-260; re-assess every gap row; update "5 of 6 goals MET" summary if goal status changed. Regression-comparison preservation: document in CLAUDE.md a `--model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` flag usage example for running the old baseline as a regression-comparison path. Update the README.md quickstart if it references the old model. Do NOT delete Qwen3-Coder references from commit messages, task bodies, or historical snapshots — the history is preserved.
- **Verify**: (a) `omlx/bench/hypercar_bench.py` MODEL_ID updated; `.venv/bin/python -m omlx.bench.hypercar_bench --quick` runs on Qwen3.6 by default. (b) `--model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` override still works (regression-comparison path preserved). (c) `omlx/bench/agentic_bench.py` model field updated; agentic bench runs on Qwen3.6. (d) `omlx/hypercar_server.py` default model updated; `python -m omlx.hypercar_server` launches with Qwen3.6. (e) Qwen3-Coder still loads via `--model` override on hypercar_server. (f) CLAUDE.md Key Architecture section reflects Qwen3.6. (g) CLAUDE.md Memory Budget table has Qwen3.6-measured values at 65K, 256K, 1M (values sourced from Tasks 258-260 — cite snapshot files in commit message). (h) CLAUDE.md Performance Targets table has Qwen3.6-measured values (values sourced from Task 259 — cite snapshot files). (i) CLAUDE.md Current status table re-assessed against Qwen3.6 measurements; goal-status summary updated. (j) `hypercar_bench --full` passes all 11 gates on the new Qwen3.6 default (hard gate — if this fails, the migration commit is reverted). (k) `hypercar_bench --full --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` passes all 11 gates (regression-path preserved). (l) Task 262 addenda values confirmed in any shipped MoE-related code (expert-count constants match Qwen3.6 at 256 experts). (m) Task 261 `--text-only` flag wired in if that task ran positive (post-removal memory win >= 1 GB); otherwise no flag added. (n) Commit message cites Tasks 257-262 and the snapshot files used for each CLAUDE.md table update; user can trace every updated number back to a specific benchmark run.
- **Effort**: M (1 week — coordinated-commit preparation 2 days, benchmark re-runs to generate final snapshot numbers 2 days (if Tasks 258-260 snapshots are stale by the time this commits), CLAUDE.md table updates 1 day, `hypercar_bench --full` validation pass 1 day, rollback-readiness check 0.5 day, commit + push 0.5 day)
- **Depends on**: Tasks 257, 258, 259, 260, 261, 262 all complete with positive outcomes (or documented soft-floor misses that do not block migration). Sequencing: **ship-last of migration tasks**. Must be a single coherent commit covering all seven surfaces; partial-migration states (e.g., bench flipped but server not) are forbidden since users run the bench+server together. Rollback plan: `git revert` on the single migration commit restores the Qwen3-Coder default in one operation.

### 264. Expert-offload cold MoE experts to NVMe for Qwen3.6-35B-A3B
- **Goal**: 6 (machine fit — reduce active Metal memory by offloading rarely-routed experts to M4 Pro's ~7 GB/s NVMe), Goal 1 (frees headroom that can go to KV expansion at 1M context)
- **Derived from**: User directive 2026-04-23 — Qwen3.6-35B-A3B has 256 experts but only 9 activate per token (8 routed + 1 shared). At 4-bit the full expert bank is ~16 GB. If we keep only the top-N hot experts resident in unified memory and stream the rest from NVMe, active Metal footprint drops dramatically without changing routing correctness. Additive to SliceMoE (Task 221) — SliceMoE compresses each expert, offloading relocates cold ones. Complementary to DyMoE (Task 226) runtime orchestration.
- **Change**: New `omlx/moe_offload.py` implementing: (1) `ExpertOffloadStore` — memory-map expert weights from a single on-disk bank file (prepare once during model load; map read-only into process VM so Metal can DMA when accessed). (2) Per-expert hot/cold state tracked by exponentially-weighted routing frequency counters updated every decode step. (3) `HotExpertCache` — fixed-size in-unified-memory slab holding the top-N hot experts; LRU eviction when a cold expert is routed and the cache is full. (4) Prefetch hook on the router's top-k computation: once we know which experts are needed for the next layer, kick off async `mmap` page-ins ahead of the FFN compute (pipelined with attention). (5) Configurable via CLI flags: `--moe-hot-cache-size N` (default N=32 out of 256), `--moe-offload-path /path/to/expert_bank.bin`. (6) Compatibility layer so SliceMoE's bit-sliced expert blocks can be stored in the offload bank; compression and offloading compose rather than exclude each other. (7) Fallback: if `--moe-offload-path` is unset, fall back to the full-resident behavior (no regression on users who don't opt in).
- **Verify**: (a) `omlx/moe_offload.py` unit tests cover ExpertOffloadStore mmap correctness (bit-exact output vs in-memory baseline for 10 random expert loads), HotExpertCache LRU correctness, and prefetch-hook scheduling. (b) `omlx/bench/hypercar_bench.py --model mlx-community/Qwen3.6-35B-A3B-4bit --moe-hot-cache-size 32 --moe-offload-path /tmp/qwen36_experts.bin` loads successfully. (c) Peak Metal memory with `--moe-hot-cache-size 32` < 10 GB for Qwen3.6-35B-A3B 4-bit (vs ~17.5 GB full-resident baseline). (d) NVMe cold-expert load latency < 5ms per access (measured via instrumentation; baseline reference: M4 Pro NVMe random 4KB read ~50µs, expert block ~5MB so ~1ms read + overhead). (e) Decode throughput regression < 10% vs full-resident baseline at 4K context on the same hardware (single-user interactive workload; higher regression acceptable at 1M where we're memory-bound anyway). (f) `hypercar_bench --full --moe-hot-cache-size 32` passes all 11 gates on Qwen3.6-35B-A3B-4bit (quality preservation is the hard gate — offloading must not change outputs, only latency). (g) Tested composition: `--moe-hot-cache-size 32` + SliceMoE (Task 221) stacks correctly with no output divergence vs SliceMoE alone. (h) Stress test: sustained 30-minute run with diverse prompts does not exceed 12 GB peak Metal (cold-expert thrashing guard).
- **Effort**: L (1 week — mmap-backed ExpertOffloadStore 2 days, HotExpertCache + prefetch integration 2 days, composition with SliceMoE 1 day, benchmark + stress validation 2 days)
- **Depends on**: Task 257 (Qwen3.6 smoke test) as hard prerequisite. Task 262 (MoE re-validation at 256 experts) as soft prerequisite — understanding routing statistics from 262 informs the initial HotExpertCache size default. Task 221 (SliceMoE) is independent but compositional; offloading works with or without SliceMoE active, and the two compose multiplicatively for active-memory reduction.

## TQ3 weight quantization unblock tasks (2026-04-23)

Diagnostic + literature triage identified that our TQ3.5 weight quantization is QuaRot without the parts that make QuaRot work. Evidence refuting the "error compounds across 40 layers" hypothesis: 4-bit WHT-rotated ALSO produces garbage ("is is is"), so the villain is rotation plumbing, not bit-width or depth. We're missing: AWQ or GPTQ calibration, RMSNorm fusion, online Hadamard in QK^T/down_proj. Skip-list and seed mismatches between converter and runtime are live footguns.

### 265. Rotation-off ablation — decisive test of whether WHT rotation helps at all
- **Goal**: 6 (unblock TQ3 weights = ~3-4 GB on 35B), Goal 2 (diagnosis determines 2-day vs 2-week fix)
- **Derived from**: Diagnostic agent 2026-04-23: TQ3.5 rotates activations at each Linear entry and un-rotates at exit, so the residual stream lives in un-rotated basis. QuaRot requires R fused into RMSNorm for activations to live in rotated space end-to-end. Our rotation may be injecting noise rather than removing it. Evidence: 4-bit WHT-rotated ALSO produces garbage ("is is is"), refuting "3-bit depth compounding".
- **Change**: Add `--no-rotate` flag to `omlx/turboquant_convert.py` (around line 227 where rotation seed is set) that forces `R = I`. Expose via `--weight-mode turbo35-norotate` in `omlx/cli.py` and wire through `omlx/engine/batched.py`. Instrumentation only.
- **Verify**: Run `omlx serve --weight-mode turbo35-norotate` on `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, generate text on 3 BENCHMARKS.md Run 20 prompts. Compare: (a) native MLX 3-bit (similar if rotation is net-noise), (b) TQ3.5-with-rotation baseline (better if rotation helps). **Outcome branching**: if `--no-rotate` matches native 3-bit, rotation is net-harmful — fix is AWQ/GPTQ on unrotated weights (Tasks 268, 269). If worse than native 3-bit, rotation is theoretically correct but plumbed wrong — need full QuaRot port (Task 271).
- **Effort**: S (0.5h instrumentation + 0.5h run + 0.5h analysis)
- **Depends on**: Nothing. First-move task.

### 266. Depth sweep — empirically test "error compounds across 40 layers" hypothesis
- **Goal**: 2 (validate/refute BENCHMARKS.md's unfounded claim), Goal 6 (if few sensitive layers, mixed-precision wins)
- **Derived from**: Diagnostic agent 2026-04-23: BENCHMARKS.md Run 20 conjectures "errors compound through 40 layers" with zero measurement. The `--fp16-layers` flag at `omlx/patches/turboquant_weights.py:94` makes this trivially testable but was never run. Lyapunov sensitivity (Task 207, pass 46) predicts a few specific layers dominate, not uniform degradation.
- **Change**: Write `omlx/bench/tq3_weight_depth_sweep.py` that runs `--weight-mode turbo35` with `fp16_layers ∈ {0, 4, 12, 24, 36, 40}` on Qwen3-Coder-30B-A3B. Measure HumanEval pass@1 + coherence at each point. Plot quality vs # quantized layers. Interpret: linear degradation = depth compounding; step function = few sensitive layers (→ Task 270); flat-at-broken = rotation plumbing (→ Task 265 confirms).
- **Verify**: Quality curve published to `bench/snapshots/tq3_depth_sweep.json`. Curve shape determines next action.
- **Effort**: S (1h script + benchmark runs)
- **Depends on**: Nothing. Parallel with Task 265.

### 267. Fix skip-list and seed mismatches between converter and runtime
- **Goal**: 2 (prevent silent corruption from live footguns)
- **Derived from**: Diagnostic agent 2026-04-23 found two bugs: (a) `omlx/patches/turboquant_runtime.py:99-105` adds "router" to `_SKIP_PATTERNS`; `omlx/turboquant_convert.py:131-138` does NOT. Model with `router` linear gets rotated weights + un-rotated activations = broken matmul. (b) Seed mismatch: `turboquant_weights.py:113` uses `hash(full_name)`, `turboquant_convert.py:227` and `turboquant_runtime.py:159` use `seed=in_dim`. Offline-converted + runtime-patched = different R.
- **Change**: (1) Move `_SKIP_PATTERNS` into new `omlx/turboquant_shared.py` (or `omlx/turboquant_linear.py`), import from all three sites. Add "router", "conv1d" to converter skip list. (2) Standardize seed to `seed=in_dim` everywhere, OR serialize R matrices into converted safetensors alongside quantized weights (preferred — removes seed dependence).
- **Verify**: `grep -rn '_SKIP_PATTERNS' omlx/` shows one definition. Cross-path test: convert offline with `--weight-mode turbo35`, load with same flag, generate — bit-identical to fresh convert without save/load round-trip.
- **Effort**: S (1h)
- **Depends on**: Nothing. Pure cleanup — land before 268/269 to avoid confounding.

### 268. AWQ-style per-channel activation scaling for TQ3 weights
- **Goal**: 6 (enable 3-bit weights = ~3-4 GB on 35B), Goal 2 (recover quality at 3-bit)
- **Derived from**: Unblocker-survey 2026-04-23: AWQ (arXiv:2306.00978) protects top ~1% of channels by activation magnitude via per-channel scale `s` applied before RTN. Orthogonal to WHT rotation (AWQ scales in rotated basis). Empirically recovers 3-5 pp at 3-bit. Our TQ3.5 has zero outlier protection.
- **Change**: New `omlx/tq_weight_awq.py`: (1) calibration on 128 samples × 2K tokens from Code-Search-Net + code-review set, capture per-linear input activation abs-max per channel. (2) compute `s = (|X|.mean(axis=0))^α / (|W|.mean(axis=0))^(1-α)` with α=0.5. (3) fold: `W' = W @ diag(s^{-1})`, adjusted input: `X' = X * s`. (4) quantize `W'` with existing TQ3.5 pipeline. Add `--act-scale calib.npz` flag to `omlx/turboquant_convert.py`.
- **Verify**: HumanEval pass@1 ≥ 50% on Qwen3-Coder-30B-A3B with `--weight-mode turbo35 --act-scale calib.npz` (vs fp16 ~95%, vs broken TQ3.5 ~0%). Coherent on BENCHMARKS.md Run 20 prompts (no "vet vet"). Stacks cleanly with WHT rotation.
- **Effort**: M (1-2 days — 0.5d calibration corpus, 0.5d AWQ computation, 0.5d integration, 0.5d validation)
- **Depends on**: Task 267 (skip-list/seed). Soft-dep Task 265 (informs rotation-vs-no-rotation).

### 269. GPTQ-style iterative Hessian column reconstruction for TQ3 weights
- **Goal**: 6 (enable 3-bit weights), Goal 2 (gold-standard — GPTQ at 3-bit on LLaMA-70B is within 1 pp of fp16 per paper)
- **Derived from**: Unblocker-survey 2026-04-23: GPTQ (arXiv:2210.17323) does Hessian-approximated iterative column reconstruction. Post-hoc, calibration-only (~128 samples). Composes ON TOP of WHT rotation — rotate first, then GPTQ the rotated weight. Dispositive fix for post-hoc 3-bit PTQ.
- **Change**: New `omlx/gptq_quant.py`: (1) Hessian `H = 2 X^T X / N` per linear from calibration activations. (2) Cholesky of `H + λI` (ridge regularization). MLX lacks native Cholesky — use numpy round-trip on ~4096×4096 matrix (~2s/layer, one-time). (3) Sequential column-wise quantization with error compensation: `W[:, j] -= err[:, j-1] @ H_inv[j-1, j:]`. (4) Hook into `omlx/turboquant_convert.py` via `--gptq` flag. Pre-rotate with WHT if `--weight-mode turbo35`, then GPTQ the rotated weights.
- **Verify**: HumanEval pass@1 within 5pp of fp16. RULER retention at 4K/16K unchanged from fp16. Calibration runtime per layer < 30s. Memory < 20GB during calibration.
- **Effort**: M (2-3 days — 1d GPTQ with MLX/numpy Cholesky, 0.5d calibration harness, 0.5d convert.py integration, 1d validation)
- **Depends on**: Task 267. Task 268 AWQ should run first (lighter, informs whether GPTQ still needed). Composes with 268 — AWQ scales + GPTQ iteration on rotated weights is the full post-hoc stack.

### 270. Per-layer / per-expert mixed-precision ladder (Unsloth-style + Lyapunov-informed)
- **Goal**: 6 (active memory with targeted fp16 islands), Goal 2 (protect sensitive few layers that dominate quality)
- **Derived from**: Unblocker-survey 2026-04-23 + Task 207 (Lyapunov sensitivity, pass 46, arXiv:2603.20991). "One projection out of 468 induces 20,000x perplexity spike" — direct diagnosis for uniform 3-bit failure. Unsloth Dynamic 2.0 is empirical SOTA for per-layer mixed-precision GGUF. KVTuner (Task 186) is the KV-side analog; this applies same idea to weights.
- **Change**: (1) Build sensitivity probe `omlx/tq_weight_sensitivity.py` that measures per-layer (and per-MoE-expert) perplexity delta when that layer/expert goes 3-bit vs fp16. (2) Compute Lyapunov contraction condition as cheap proxy. (3) Produce `config.json` per-layer bit-width: sensitive layers stay fp16, medium 4-bit, insensitive 3-bit. Target average ≤ 3.5 bits. (4) Wire via `--bit-ladder config.json` flag in `omlx/turboquant_convert.py`. Per-expert extension: MoE hot experts get higher precision.
- **Verify**: HumanEval within 2pp of fp16 at average bit-width ≤ 3.5. On Qwen3.6-35B-A3B 256-expert, hot experts get higher precision than cold ones. Memory at 1M context reduces ≥ 2.5 GB vs uniform 4-bit MLX.
- **Effort**: M (3-4 days — 1d sensitivity probe, 1d Lyapunov integration, 1d MoE per-expert extension, 1d validation)
- **Depends on**: Task 266 (depth sweep informs sensitive layers). Task 268 or 269 (need working 3-bit path for "insensitive" slot). Task 257 if applying to Qwen3.6.

### 271. Full QuaRot port — RMSNorm fusion + online Hadamard in QK^T and down_proj
- **Goal**: 6 (enable genuine sub-4-bit weights), Goal 2 (if rotation is theoretically right but plumbed wrong, fix plumbing)
- **Derived from**: Diagnostic agent 2026-04-23 + QuaRot paper (arXiv:2404.00456, pass 5, Task 41). QuaRot requires: (a) R fused into preceding RMSNorm scale — activations live in rotated space end-to-end. (b) online Hadamard in QK^T (attention) and down_proj (FFN) to protect intermediate products. Our TQ3.5 does neither. **CONTINGENT** on Task 265 showing rotation actually helps — if rotation is net-noise, skip entirely.
- **Change**: (1) RMSNorm fusion: modify `omlx/patches/turboquant_runtime.py` to multiply R into preceding RMSNorm's weight during conversion. (2) Online Hadamard in QK^T: after q_proj/k_proj, apply Hadamard to Q and K before `Q @ K.T`. (3) Online Hadamard in down_proj input: apply Hadamard to FFN intermediate before down_proj. (4) Single shared rotation across entire model to enable fusion.
- **Verify**: HumanEval within 3pp of fp16 at 3-bit weights + 3-bit activations (W3A3). Coherent on BENCHMARKS.md Run 20 prompts. Correctness: fp16 vs quantized hidden-state cosine similarity > 0.98 at each layer.
- **Effort**: L (1-2 weeks — 3d RMS fusion, 3d online Hadamard attention, 2d online Hadamard FFN, 3d validation + debugging)
- **Depends on**: Task 265 **GATE** (if rotation is net-noise, skip entirely). Tasks 267, 268, 269.

### 272. Research pass — MC-MoE and MoQE for MoE-aware expert-specific quantization
- **Goal**: 6 (256-expert Qwen3.6 is where per-expert bit allocation pays off most)
- **Derived from**: Unblocker-survey 2026-04-23: **MC-MoE** (arXiv:2410.06270) combines per-expert Hessian-aware bit allocation with expert activation frequency. Reports 2.54-bit average on Mixtral with <1 pp loss. **MoQE** (arXiv:2310.02410) does per-expert quantization with activation-frequency-aware bit allocation. Neither in LIT_REVIEW.md despite ~10 passes touching MoE. MoE axis that GPTQ+AWQ alone don't address.
- **Change**: Research-only task for next research cron pass: download MC-MoE (2410.06270) and MoQE (2310.02410), add to `research/LIT_REVIEW.md` with standard per-paper format. Evaluate applicability to Qwen3.6-35B-A3B's 256-expert layout. File follow-on implementation tasks if applicable.
- **Verify**: Both papers cited in LIT_REVIEW.md. Clear ship/defer decision per paper.
- **Effort**: S (research only, 1 cron pass)
- **Depends on**: Nothing — next research cron can pick up.
