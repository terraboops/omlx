# oMLX Hypercar

## Project Overview

oMLX Hypercar — high-performance local LLM inference on Apple Silicon with compressed KV caches. Serves Qwen3.6-35B-A3B (Tier S2 hybrid SSM+attention) via OpenAI-compatible API; bench-validated to 512K context on 48 GB, 1M is the open Goal 1 frontier. Earlier Qwen3-Coder-30B-A3B path remains supported as a `--model` override.

## Hardware Target

Reference machine: **16" MacBook Pro (Nov 2024) — Apple M4 Pro, 48 GB unified memory, macOS Tahoe 26.2**.
All performance targets below MUST hold on this hardware with the laptop under normal desktop load
(browser, editor, Claude Code session). No "cold-boot" excuses.

## Hypercar Goals (North Star)

These are the non-negotiable targets that define "Hypercar-class" inference. Every optimization,
refactor, and architectural decision should be measured against them. A change that regresses
any of these — even to improve another — needs explicit justification.

| # | Goal | Target | Why it matters |
|---|------|--------|----------------|
| 1 | **Context window** | 1M tokens | Whole-repository reasoning without chunking |
| 2 | **Intelligence** | Matches or beats GPT-4, proven on **4 independent evals** | One eval is a lucky prompt; four is a claim |
| 3 | **Decode speed** | ≥ 50 tok/s, **constant across context window** | Feels interactive at 2K AND at 1M — not a cliff |
| 4 | **Prefill speed** | ≥ 500 tok/s, **constant across context window** | Re-reading 200K of code shouldn't take a coffee break |
| 5 | **Swap pressure** | p90 sustained swap I/O < 100 MB/s (N≥8 runs) | Swap depth alone misses throughput; 100 MB/s leaves 4x headroom vs M4 Pro's ~430 MB/s floor |
| 6 | **Machine fit** | Runs comfortably on the M4 Pro 48GB reference machine | Laptop stays usable while inference runs |

### Current status against goals (as of 2026-04-25, post N=3 variance series)

| # | Goal | Current | Gap |
|---|------|---------|-----|
| 1 | 1M context | **VALIDATED TO 512K on Qwen3.6** (2026-05-03, `--kv-mode native --kv-bits 4 --prefill-chunk 0` adaptive controller). NIAH @ 512K PASS, Metal peak 47 GB, 111 min wall. Trajectory: 256K → 384K (fixed chunk=1024) → 458K (stuck) → 512K (adaptive controller via `omlx/patches/adaptive_prefill.py`). Goal 1 reach is now at 50% of 1M target by context length. Earlier path on Qwen3-Coder reached 128K (fp16+SnapKV). | 1M needs algorithmic memory reduction (KVLinC 2-bit, online SnapKV during prefill); chunk-tuning alone hit physical-RAM ceiling. |
| 2 | 4 independent evals beating GPT-4 | **HumanEval 95%**, Code Intel 5/5, RULER 100%, **MMLU-Pro 62%**, LiveCodeBench 40% (8/20, +1 PASS via retry — Tasks 254/255/258). All N=3 runs reproduce these numbers EXACTLY. | LiveCodeBench ceiling — sampling axes exhausted on Qwen3-Coder. Qwen3.6 migration is the biggest remaining lever. See `research/qwen3_coder_cycle_exhaustion.md`. |
| 3 | 50 tok/s decode constant | **Short context MET** (smoke 53 tok/s). **Decode scaling**: 40/34/26/18/10 tok/s at 2K/4K/8K/16K/32K (Task 290). Decode CV is **0.5%** intra-run (Task 288); steady-state is reproducible. | 16K+ below target. **Component breakdown (Task 339)**: at decode T_kv=16K, projections+norms+router+RoPE are **53% of decode budget**, MoE gather_mm is 25%, attention SDPA is only **19%**, lm_head is 3%. Long-believed "attention is the bottleneck" framing was wrong; mlx_lm already gather_mm-fuses MoE and projections+norms+RoPE dominate. **Three cheap-fusion levers EMPIRICALLY FALSIFIED**: Fix 2 streaming/retrieval split (Tasks 277+319-323), custom Metal int4-fused SDPA (Tasks 332-334, 2-5× SLOWER than MLX MPSGraph reference), `mx.compile` auto-fusion (Task 340, no signal above 50% noise floor). **Speculative decoding (Task 341) is the only remaining cheap lever** — mlx_lm 0.31.2 ships first-class API; predicted at α=0.5, N=8: ~3× speedup → 49 tok/s ⇒ Goal 3 nearly MET. Probe ready at `scripts/probe_speculative_decoding.py`; user-invoked measurement pending. Wall-time variance is bimodal but decode tok/s itself is NOT — variance is in prefill phases only. |
| 4 | 500 tok/s prefill constant | **Duo: 816 tok/s at 4K — GOAL MET**. 522 tok/s @ 16K (104%) — also met. 339 tok/s @ 32K. TQ3: 672 tok/s at 8K. **Component breakdown (Task 351)**: at 32K prefill, attention SDPA = **82% of budget**, MoE = 12%, projections+norms = 6%. Path to closing 32K gap: enable MInference sparse-attention (`--prefill-sparse minference`, already implemented) — predicted 575 tok/s at 50% sparsity. | 32K prefill below target; **clear path via MInference** (user-invoked validation pending). Unlike Goal 3, Goal 4 has an architectural lever ready. |
| 5 | Swap p90 < 100 MB/s | **GOAL MET fast-mode** (3-4 GB peak). Slow-mode: 10 GB peak (still under 12.9 GB limit). | Co-tenancy can flip into slow mode; mode flag should be tracked alongside runs. |
| 6 | 48GB M4 Pro fit | Duo: 35.1 GB peak during MMLU-Pro (exact across N=3 runs); 40.7 GB peak at 32K decode (efficiency_profile). — **PASS** | 32K decode peak is at 99% of ceiling; any further state-adding optimizations need to be memory-neutral. |

### Bench reproducibility (variance series, N=3, Tasks 286/287/291)

**Quality is rock-solid reproducible**: all 4 quality gates exact-match across N=3 runs. Code Intel 5/5, MMLU-Pro 62/100, LCB 8/20, HumanEval 19/20.

**Wall-time is bimodal**:

| Run | Mode | Total | Swap peak |
|---|---|---:|---:|
| #1 (286) | fast | 46.1 min | 3.6 GB |
| #2 (287) | slow | 66.6 min | 10.4 GB |
| #3 (291) | fast | 47.3 min | 3.1 GB |

Discriminator: swap peak < 5 GB → fast; > 8 GB → slow. Slow mode is +44% wall-time vs fast. Fast-mode runs are within 3% of each other. Decode tok/s itself doesn't bimodal; the variance is concentrated in prefill-heavy phases (RULER +124%, NIAH +83% in slow vs fast). For honest regression detection, samples must be stratified by mode before aggregation.

### Decode scaling curve (Tasks 264/267/271/290, steady-state)

| Context | DuoKV (bench mode) | Raw MLX (no patches) | Metal peak |
|--------:|------------------:|--------------------:|-----------:|
| 2K      | 46.65 tok/s       | 46.65               | 33.9 GB    |
| 4K      | 43.86             | 43.86               | 34.9 GB    |
| 8K      | **22.22** (post-271) | 39.17            | 37.0 GB    |
| 16K     | **16.11** (post-271) | 27.75            | 41.1 GB    |
| 32K     | **10.21** (Task 290) | —                | 40.7 GB    |

DuoKV adds ~50% per-step tax at long context; Tasks 269/271 reduced the
Python-overhead component by hoisting trim-path indices/masks to a
module-level cache. The remaining tax is in the gather/mask MLX ops
themselves (O(T_total) per call).

**Task 265 Fix 2 (streaming/retrieval split) — DEPRIORITIZED** after
Tasks 277 + 319-323 audit chain. Original projection was decode@16K
~25-28 tok/s, closing most of the Goal 3 gap. Reality: (a) Fix 2 v2
shipped under `--kv-mode duo-split` and REGRESSED 8-20% at 8K-16K
decode because the fixed cost of two SDPA calls + assembly exceeds
attention compute savings at L_q=1 (Task 277); (b) per-layer KV-head
distribution is far more retrieval-heavy than the 58.92% global
Q-streaming fraction implies — 58% of layers have ZERO streaming KV
heads, so Fix 2's average attention-compute saving is only ~11%/layer
at 16K, not the ~50% projected (Task 323); (c) the dilution Fix 2
would fix is direction-preserving (cos > 0.99 at decode, cos > 0.83
p10 at prefill), so it's a magnitude-amplification fix, not a
correctness fix (Tasks 319-321).

**Task 281 (Open-TQ-Metal port) — REFRAMED as memory-only opt-in**
after Phase 1 probe sequence (Tasks 327-334). Original framing was
"Goal 3 long-context perf path." Reality: Phase 1 probes built the
combined simdgroup × split-K kernel in 8 cycles and verified
correctness end-to-end (cos > 0.9999, rel_err < 0.05). But Tasks
332-334 perf-measured the kernel at 0.19-0.48× of MLX MPSGraph SDPA
at all sizes (T_kv ∈ {64, 256, 1024, 4096, 16384}) — the kernel is
2-5× SLOWER than the reference Hypercar already uses. The paper's
48× claim is vs a non-fused fp32-dequant baseline; against MLX's
MPSGraph fast path, custom kernels are at a structural disadvantage.
**Net: Task 281 should ship as `--kv-mode duo-int4-fused` opt-in for
long-context (64K+) where 3.2× KV memory savings unlock paths that
today require swap. Do NOT make default — would regress decode
2-3× at 16K.** Spikes 316/318 PASSED + Phase 1 fully verified;
estimated Phase 2 effort 3-4 weeks. See
`research/design_notes/int4_fused_sdpa_phase2.md` for the full plan.

**Goal 3 component-cost reframing (Task 339, 2026-04-26).** Direct
measurement of decode wall-time at production shapes (Qwen3-Coder-30B-
A3B-Instruct-8bit, T_q=1, fp16) shows the long-believed "attention
dominates" framing is wrong. At T_kv=16K (in decreasing share):

- Projections + norms + router + RoPE: **53.2%** ← LARGEST
- MoE gather_mm (8 active experts): 25.2%
- Attention SDPA: 18.6%
- lm_head: 3.1%

Sequential-MoE anti-pattern (1.22 ms/layer) vs `mx.gather_mm` real path
(0.46 ms/layer) confirms mlx_lm already fuses experts — "fuse experts
into one kernel" is NOT a Goal 3 lever (already done). The 53%
projections+norms share is dominated by per-call dispatch overhead +
intermediate materialization across 192 small kernels per token
(RMSNorm × 2 + QKV + O + RoPE + router, × 48 layers). Probe at
`scripts/probe_decode_component_cost.py`. Full breakdown in
`research/decode_component_cost_breakdown.md`.

**Cheap-fusion levers FALSIFIED (Task 340, 2026-04-27).** `mx.compile`
on the (rms_norm → qkv-proj → rope) chain at decode shapes shows NO
meaningful speedup: 3-run variance (baseline 0.22-0.33 ms, compiled
0.21-0.36 ms) puts the signal well below the 50% run-to-run noise
floor. Initial 5-iter warmup test reported "1.65× speedup" — caught
by the 30-iter steady-state re-run as JIT-cache cold-start artifact.
**Methodology lesson now codified in `feedback_perf_microbench_first.md`**:
default to 30 warmup + 60 timed iters for any MLX microbench; always
run 3+ repeats and check signal exceeds run-to-run variance before
claiming a result. Probe at `scripts/probe_decode_kernel_fusion.py`.
Negative-result writeup in `research/decode_kernel_fusion_validation.md`.

**Speculative decoding (Task 341, 2026-04-27) is the only remaining
cheap Goal 3 lever.** mlx_lm 0.31.2 ships first-class API support
(`stream_generate(model, tokenizer, prompt, draft_model=...)` with
`GenerationResponse.from_draft` per token) — no Hypercar-specific
integration needed for the basic path. Predicted speedup math
(T_eff = (T_draft × N + T_main) / (1 + α × N)) at T_main=62.5 ms
(16K decode), T_draft=5 ms (small drafter):

| α | N=4 | N=8 |
|---:|:---:|:---:|
| 0.5 | 2.27× | **3.05×** |
| 0.7 | 2.88× | 4.02× |

Goal 3 (50 tok/s) requires ≥3.1× speedup — predicted achievable at
α≥0.5 with N=8. UAG-MLX-LM (research pass 65) measured α≈0.4 on
structured text; coding workloads should score HIGHER given token-
level redundancy. **Recommended drafter**: `mlx-community/Qwen2.5-
Coder-1.5B-Instruct-4bit` (same Qwen tokenizer family, REQUIRED;
~1 GB VRAM addition). Probe at `scripts/probe_speculative_decoding.py`
(heavy — user-invoked, not cron). Strategy in
`research/speculative_decoding_lever.md`.

**Goal 3 lever ranking (post-Task 341)**:

| Lever | Status | Predicted Goal 3 contribution |
|---|---|---|
| **Speculative decoding** | PROBE READY (Task 341), SERVER INTEGRATED (Task 348) | At α=0.5, N=8: 3.05× ⇒ 49 tok/s |
| ~~mx.compile fp16 fusion~~ | **FALSIFIED** (Task 340) | — |
| ~~mx.compile quantized fusion~~ | **LIKELY FALSIFIED** (Task 347 re-eval of Task 345) | — |
| ~~Custom Metal kernel fusion~~ | **FALSIFIED** (Tasks 332-334) | — |
| ~~MoE expert prefetch~~ | **NOT IMPLEMENTABLE in pure MLX** (Task 350) | — |
| Attention path (Task 281) | Memory-only opt-in | <10% at 16K |
| Qwen3.6 migration | Download blocked (Task 293) | Indirect, untested |
| MoE optimization | gather_mm-fused already | Ceiling at 26% |

**Task 350 MoE expert prefetch closure**: Research pass 65 surfaced
UMD's "Speculating Experts" paper (arxiv:2603.19289) claiming 14% TPOT
reduction by predicting next-layer expert routing and prefetching
weights during current-layer compute. The technique requires
concurrent Metal kernel execution (current layer compute || next layer
weight load). Direct measurement: `mx.async_eval` does NOT enable
concurrent Metal execution — async/sequential ratio = 0.99 (no overlap).
MLX 0.31.2 serializes through a single Metal command queue. The lever
is closed for pure-MLX implementation. Would require lower-level Metal
API access (CAMetalCommandQueue with parallel encoders) — out of scope
for user-mode MLX. **Speculative decoding remains the SOLE remaining
cheap Goal 3 lever**, now with both probe (Task 341) and server
integration (Task 348) ready for user-invoked α measurement.

**Task 345 → 347 re-evaluation note**: Task 345 reported "MARGINAL
POSITIVE" 1.18× speedup for `mx.compile` on a 3-op quantized chain.
Task 347 retested with cross-chain JIT priming as a methodology
control — when shared kernels are pre-warmed (matching production
decode where the server runs many tokens), the speedup collapses to
**1.00× ± noise** in steady-state. Task 345's apparent positive was
likely cold-start primer benefit (each measurement was a fresh Python
process), not real fusion gain. **Speculative decoding is now the
SOLE remaining cheap Goal 3 lever.**

**Future Goal 3 work**: user-invoked spec-decoding probe with
recommended drafter on coding workload. If α ≥ 0.5, file integration
task to add `--draft-model` flag to `hypercar_server.py`. If
0.3 ≤ α < 0.5, ship as opt-in (Task 281-style memory-mode pattern).
If α < 0.3, accept the 16K cliff as structural and focus on memory
headroom (Task 281 memory-only opt-in, KVLinC 2-bit Task 288) or
the Qwen3.6 migration (Task 253) which has different MoE profile.

**32K data point** (Task 290 via `efficiency_profile --decode-ctx 32768
--kv-mode duo --decode-n 64` post-Task-289 fix): 10.21 tok/s, Metal peak
40.7 GB (within 0.5 GB of 41.2 GB ceiling). The 16K→32K decline is
-43%; further state-adding optimization at this point must be
memory-neutral. Curve cliff continues monotonically.

**Tool consistency note (Task 289)**: `efficiency_profile` and
`hypercar_bench` now apply the same patch stack (`prefill_last_logit`)
and read decode tok/s within variance. Before Task 289, efficiency_profile
read ~30% lower than bench at 16K because the missing patch left a
~5 GB lm_head logits tensor in Metal, pushing peak to the 41 GB ceiling
and triggering paged-memory cost during decode.

### Goal 1 practical cost progression (analyst Run 78, 2026-04-17)

| Context | Metal Peak | Swap | Wall Clock | SnapKV Needed? |
|--------:|----------:|---------:|----------:|:-------------|
| 4K | 32.5 GB | 0 GB | 0.3s | No |
| 16K | 33.9 GB | 0 GB | 50s | No |
| 64K | 38.4 GB | 4.9 GB | 23 min | Recommended |
| 128K | 44.6 GB | 6+ GB | 42 min | Required (chunked prefill) |
| 256K | ~30 GB* | 0 GB* | ~3 hr | Required (fp16 + SnapKV@50%) |

*256K with SnapKV@50% keep: model 17.2 + KV 6.2 = 23.4 GB after eviction.

**Recommended mode: `--kv-mode duo`** — best quality (MMLU-Pro 62%, HumanEval 95%, LiveCodeBench
40%), zero swap. Decode tok/s: 46/44/22/16 at 2K/4K/8K/16K (post-Tasks 269+271 module-level
trim caching, +16% at 16K from 13.3 baseline). DuoKV pre-allocated slab and vectorized trim
shipped earlier. For long context (64K+): use DuoKV+SnapKV (`--snapkv-keep K`) for 47% faster decode.
SnapKV eviction works with ALL KV modes — DuoKV, fp16, native, TQ3.
TQ3 mode: 672 tok/s prefill at 8K (fused WHT quantize), good decode (50 tok/s).
Session save/load for persistent context (prefill once, reload via `/v1/sessions/load`).

**5 of 6 goals MET** in duo mode. Full `--full` benchmark ALL 11 GATES PASS (Run 74).
**Goal 1**: SnapKV validated to 128K (fp16) and 64K (TQ3 native 3-bit).
120K single-pass exceeds 48 GB Metal — progressive mid-prefill eviction required.
Progressive eviction with single midpoint cut being validated (avoids re-RoPE accumulation).
Server: `--snapkv-keep K --caote --segmented-evict 512` enables the full eviction stack.
Per-phase headroom checks (Task 86) now gracefully skip memory-hungry phases under co-tenancy.

### Goal 1 Path (KV compression — SHIPPED + VALIDATED, 2026-04-18)

SnapKV eviction stack: 9 composable layers, GPU-validated to 96K (fp16) and 64K (TQ3).

| Component | Status | What it does |
|-----------|--------|-------------|
| **SnapKV + real Q capture** (Task 46) | **SHIPPED** | Attention-guided token selection using real query projections |
| **CAOTE scoring** (Task 100) | **SHIPPED** | Value-aware eviction: `(α/(1-α)) × ||V_mean - v_j||` |
| **BUZZ segmented** (Task 98) | **SHIPPED** | Per-segment top-K preserving local structure |
| **Re-RoPE compaction** (Task 46) | **SHIPPED** | Physical token removal with RoPE position correction |
| **Freshness decay** (Task 97) | **SHIPPED** | Cosine-similarity conflict detection for multi-turn |
| **Adaptive prefill** (Task 94) | **SHIPPED** | Memory-aware chunk sizing |
| **GER safety** (Task 106) | **SHIPPED** | Hallucination cliff guard |
| **Fair eviction** (Task 107) | **SHIPPED** | Proportional partition budgets |
| **Head rebalancing** (Task 111) | **SHIPPED** | Retrieval 2×, streaming 0.5× |

**Validated context ladder (NIAH PASS at 50% keep recommended — 25% hurts decode 14%):**

| Context | Mode | Metal Peak | Metal After | Saved | Time |
|---------|------|-----------|-------------|-------|------|
| 64K | TQ3 native | 42.1 GB | 41.2 GB | 0.9 GB | 18 min |
| 96K | fp16 | 44.6 GB | 38.6 GB | 6.0 GB | 18 min |
| 128K | fp16 | 44.6 GB | 35.6 GB | 9.0 GB | 42 min |

**TQ3+SnapKV**: fp16 KV capture hooks (`install_kv_capture_hooks`) save pre-quantization
K/V during prefill. `compact_cache` uses clean fp16 for gather+re-RoPE, then does SINGLE
quantization back to 3-bit. Eliminates double-quantization noise that corrupted at 64K.

**Progressive eviction (120K+)**: BLOCKED by MLX KVCache position model. Re-RoPE
accumulates across multiple eviction rounds. Scatter-back causes Metal OOM. Needs
SparseKVCache class (L effort). Workaround: single-pass prefill + session save/load.

**256K workflow**: prefill once (~3h) → SnapKV compact → `/v1/sessions/save` → reload in 2s.
Server flags: `--snapkv-keep K --caote --segmented-evict 512`.

### Performance optimizations (analyst efficiency audit + bench-loop cycles, 2026-04-18 / 04-23 / 04-24)

16 fixes across all KV cache modes (12 from the original analyst audit + 4 from the bench-loop session 2026-04-23/24):

| Fix | Impact | What changed |
|-----|--------|-------------|
| **TQ3 fused quantize** (Task 152) | 15x short-prefill, 1.33x at 8K | WHT uses existing fused dense kernel (H is symmetric) |
| **TQ3 fused dequantize** (Task 145) | 2x per-call | All hot paths use dequantize_fused |
| **DuoKV pre-alloc slab** (Task 149) | 248x decode at 64K | Pre-allocated buffer + O(1) slice write |
| **DuoKV gather trim** (Task 142) | Eliminates per-head loop | Single take_along_axis replaces 12 intermediates |
| **StreamingKV ring vectorize** (Task 143) | O(1) ring writes | Modular arithmetic scatter |
| **skip_rerope default** (Task 168) | 4.7x post-eviction decode | 7.2→33.7 tok/s, NIAH still PASS |
| **GQA broadcast** (Task 148) | Saves 1.07 GB/layer | Q reshape replaces mx.repeat |
| **SnapKV vectorization** (Tasks 139-141,146,147) | 5.7s→<0.5s at 64K | Band mask, argpartition, scatter ops |
| **TQ3 geometric growth** (Task 144) | ~7 allocs vs ~250 at 64K | 2x buffer doubling |
| **SnapKV 50% keep** (Task 155) | 7% faster decode | 25% hurts decode 14% (re-RoPE cost) |
| **TQ3 save fix** (Task 160) | Fixes save during warmup | Quantize fp16 buffer before save |
| **Tool-call gate** (Task 161) | JSON validity tracking | Phase 3f in benchmark |
| **MMLU-Pro per-Q cleanup** (Task 259) | Prevents 100× slowdown after Q~65 | `gc.collect() + mx.clear_cache()` between questions (latent leak — every other phase had this) |
| **DuoKV trim index cache** (Task 269) | +12.8% NIAH decode @ 16K | Module-level cache for stream/retrieval Python lists, hits 48× per decode step instead of 1× |
| **DuoKV trim mx.array cache** (Task 271) | +2.9% incremental at 16K | Cache MLX-array versions of trim rows; per-layer assembly is `mx.stack(cached)` not `mx.array(list)` |
| **LCB CoT prompt + retry** (Tasks 254/255/258) | LCB 30% → 40% (+1 PASS) | Step-by-step prompt + sample-verify retry at temp=0.7; recovers Fill-the-Gaps |

## Before Every Commit

```
REQUIRED: Run the benchmark before committing:
  .venv/bin/python -m omlx.bench.hypercar_bench

For quick iteration (smoke + coherence only, ~30s):
  .venv/bin/python -m omlx.bench.hypercar_bench --quick

For full validation including HumanEval (~15min):
  .venv/bin/python -m omlx.bench.hypercar_bench --full

ALL gates must pass before committing. Fix any failures first.
Do not use --no-verify to skip hooks.

Run numbering: analyst cron uses "Run N", devloop uses "devloop-N" or "sample-N"
in commit messages. See bench/snapshots/README.md for details.
```

Memory limits are percentage-based (auto-detected from system RAM):
  - Metal peak: 80% of system memory
  - Swap delta: 25% of system memory
  - Metal at load: 70% of system memory (8-bit model = ~32GB on 48GB)
  - Breaches cause IMMEDIATE abort (fail-fast watchdog)

Gate summary:
  - Memory: no breach during any phase (Metal, swap)
  - Smoke: model loads, generates tokens
  - Coherence: 2/2 basic checks
  - Code intelligence: >= 3/5 problems
  - NIAH: 4K retrieval passes
  - HumanEval (--full): >= 35% pass@1

## Key Architecture

- Default model: Qwen3.6-35B-A3B-4bit (19.5 GB, 40 layers hybrid 30 SSM + 10 full-attn, head_dim=256, MoE 3B active, native 262K)
- Previous default (still supported via `--model`): Qwen3-Coder-30B-A3B-Instruct-8bit (17.2GB, 48 dense attention layers, MoE 3B active). All KV-mode tables and tuning notes below currently reflect Qwen3-Coder numbers; Qwen3.6 numbers are catalogued in `BENCHMARKS.md` and `research/qwen36_migration_ready.md`.
- Server: omlx/hypercar_server.py (OpenAI-compat, prompt caching, tool parse safety)

### KV Cache Modes (`--kv-mode`)

| Mode | Cache Type | Compression | Features | Use Case |
|------|-----------|-------------|----------|----------|
| **`duo` (default)** | DuoKVCache (fp16 retrieval + ring-buffer streaming) | ~2x at 16K+ | Best quality (MMLU-Pro 62%, HumanEval 95%), zero swap | **Best for ≤16K context** |
| `duo --duo-quantize` | DuoKVCache (3-bit retrieval + fp16 ring streaming) | ~5x | DuoAttention quality + 3-bit memory, NIAH PASS | **Goal 1 path to 1M** |
| `native` | MLX QuantizedKVCache(bits=3, group_size=64) | 5.3x | Battle-tested, long context | Long context (16K-1M) |
| `tq3` | TurboQuantKVCache (WHT rotation + codebook) | 5.3x | save/load, rewind, fork, best RULER quality | Agentic workflows |
| `fp16` | Standard KVCache | 1x | Baseline quality | Testing, ~75K max |

**Mode selection guide**: Use `duo` (default) for interactive coding — best quality and speed up to 16K.
Use `duo --duo-quantize` for long context (64K+) — DuoAttention quality with 3-bit memory efficiency.
Switch to `native` or `tq3` for maximum compression or agentic features.

TQ3 mode uses Walsh-Hadamard Transform (per arXiv:2504.19874) for full dimension decorrelation. This makes the Beta((d-1)/2, (d-1)/2) codebook valid, producing correct code output.

### Memory Budget

1M context at 39.7GB with 3-bit KV (native or tq3). Fits in 48GB with 8GB headroom.

| Context | KV Cache | Total |
|--------:|----------:|------:|
| 65K | 1.4 GB | 18.6 GB |
| 256K | 5.6 GB | 22.8 GB |
| 1M | 22.5 GB | 39.7 GB |

## Performance Targets (with duo mode + warmup)

- Decode: >50 tok/s at 2K context (**52.4 achieved**), >10 tok/s at 16K
- Prefill: >500 tok/s at 4K (**566 achieved**), drops to 340 at 16K
- Memory: <42% system RAM at load, <80% peak, **zero swap in duo mode**
- Quality: 5/5 coding, HumanEval 95%, MMLU-Pro 62%, RULER 100%
- TTFT (cached prompt): <1s (warmup eliminates Metal JIT cold-start)

## Server Usage

```bash
# Recommended (best quality, zero swap, 52 tok/s decode)
python -m omlx.hypercar_server --port 8080

# With agentic features (session persistence, rewind, fork)
python -m omlx.hypercar_server --kv-mode tq3 --fp16-layers 1 --port 8080

# Long context (64K+, lower quality but fits more tokens)
python -m omlx.hypercar_server --kv-mode native --port 8080

# OpenCode connection (default Qwen3.6 — pin Qwen3-Coder via --model if needed)
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=hypercar
opencode --model "hypercar/mlx-community/Qwen3.6-35B-A3B-4bit"
```

### Sparse prefill (MInference, Task 351 path identified, opt-in)

**Goal 4 lever for 32K+ prefill.** Task 351 measured that at T_kv=32K
prefill, attention SDPA is **82% of the budget** (vs 19% at decode).
At 50% sparsity, attention halves → predicted **575 tok/s at 32K**
(Goal 4 MET vs current 339 tok/s baseline). MInference per-head
sparse-attention pattern dispatch (Tasks 4+5, vertical_slash + sink
tokens + block-sparse) is **already implemented** and the calibration
pattern table for Qwen3-Coder-30B-A3B-Instruct-8bit is shipped at
`omlx/patches/minference_patterns/qwen3_coder_30b_a3b_instruct_8bit.json`.

> ⚠ **CAVEAT (Task 381, 2026-04-29)**: the shipped pattern table is
> currently a **SYNTHETIC PLACEHOLDER** — its `note` field reads
> "SYNTHETIC placeholder — run scripts/minference_calibrate.py to
> generate real patterns". The dispatch logic is correct, but per-head
> patterns are programmatic defaults, not measured from actual
> attention maps. The 575 tok/s prediction above is **unsubstantiated
> on this codebase** until calibration runs.
>
> Calibration is a one-time ~10 min user-invoked step:
> ```
> .venv/bin/python scripts/minference_calibrate.py \
>     --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit
> ```
> Run `.venv/bin/python scripts/check_1m_context_ready.py` to see the
> current calibration status surfaced in the Goal 1 dashboard.

```bash
python -m omlx.hypercar_server \
    --prefill-sparse minference \
    --port 8080
```

The flag activates `omlx/patches/minference_prefill.py` which monkey-
patches `scaled_dot_product_attention` to dispatch per (layer, head)
based on the calibration table. Composes with all KV modes (duo, tq3,
native, fp16) and with SnapKV eviction.

**When to use**:
- T_kv ≥ 16K prefill: predicted 1.4×–2.5× prefill speedup (probe-based)
- T_kv < 8K: marginal benefit; calibration overhead may exceed savings
- Quality regression risk: validated at 16K+ in original Task 4 calibration; longer contexts (64K, 128K, 256K) need re-validation if used for production

**Validation** (user-invoked, requires bench run):
```bash
.venv/bin/python -m omlx.bench.hypercar_bench --full --prefill-sparse minference
```
Compare 32K prefill tok/s vs baseline `--full` (no flag). Expected: ≥500 tok/s
at 32K (Goal 4 MET) with NIAH/RULER quality unchanged at the validated
context lengths.

**Why this matters for Goal 4**: at 32K prefill, attention dominates
80%+ of the budget. Sparse attention is the highest-leverage lever
because it directly attacks the dominant cost. Unlike Goal 3 (where
all architectural levers were falsified), Goal 4 has this lever
already implemented and predicted to close the gap.

### TTT-Linear head routing (Task 388 Phase 2, opt-in)

Per-(layer, head) attention dispatcher. Streaming-tagged heads with a
loaded TTT-Linear block route through an O(D²)-per-token recurrence
(replaces softmax attention's O(N·D) cost for those heads); retrieval-
tagged heads keep softmax. The structural Goal 4 lever — see Task 388.

Two modes:

```bash
# Bit-equivalence mode (no behavior change). Use this to confirm the
# patch installs cleanly before turning on TTT blocks.
python -m omlx.hypercar_server \
    --ttt-router-policy omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json \
    --port 8080

# Active mode: routes streaming heads with loaded TTT blocks.
# The blocks dir is produced by:
#   python scripts/ttt_distill_single_head.py --capture --save-block-dir phase0_ttt/
python -m omlx.hypercar_server \
    --ttt-router-policy omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json \
    --ttt-router-blocks-dir phase0_ttt/ \
    --port 8080
```

Server logs will print one of:
- `TTT head router INSTALLED in bit-equivalence mode` (no blocks loaded)
- `TTT head router INSTALLED with N TTT blocks loaded across L layers × H heads`

Phase 3 validation pattern: install in bit-equivalence first, run a
golden-output bench → confirm output unchanged. Then enable blocks
incrementally (e.g., copy one (layer, head) safetensors into the dir),
re-run, confirm degradation is bounded. The bit-equivalence path is
pinned by tests in `tests/test_ttt_head_router.py`.

The same flags work on `hypercar_bench`:

```bash
# A: bit-equivalence — gates should pass exactly the same as baseline
.venv/bin/python -m omlx.bench.hypercar_bench --quick \
    --ttt-router-policy omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json

# B: with TTT blocks loaded — gates measure the routing impact
.venv/bin/python -m omlx.bench.hypercar_bench --full \
    --ttt-router-policy omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json \
    --ttt-router-blocks-dir phase0_ttt/
```

Phase 3 close-out is "(B) gates within tolerance of (A)".

### Speculative decoding (Task 348 integration, opt-in)

After Task 347 falsified all fusion-based Goal 3 levers, speculative
decoding is the only remaining cheap path. Server scaffolding is in
place (Task 348); enable with `--draft-model`:

```bash
python -m omlx.hypercar_server \
    --draft-model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
    --num-draft-tokens 4 \
    --port 8080
```

Drafter MUST use the same Qwen tokenizer as the main model (server
checks vocab size per request; mismatch disables drafter for that
request and warns). Memory cost: ~1 GB for the 4-bit drafter on top
of the 17 GB main model = 18 GB total, well within M4 Pro 48 GB budget.

**Per-request telemetry** in server logs:
```
🏁 DONE: 200 tokens in 8.4s (24.0 tok/s decode) | spec α=52% (104/200 from draft, N=4)
```

The `α` (acceptance rate) tells you the lever's payoff on YOUR workload:
- α ≥ 0.5: spec-decoding is winning. Predicted ~3× decode speedup at N=8.
- 0.3 ≤ α < 0.5: partial win. Try smaller drafter (less overhead) or larger N.
- α < 0.3: not winning on this workload. Disable (`--draft-model` unset).

**Before enabling on production traffic**: run Task 341's standalone probe
with a representative prompt to measure α offline:

```bash
.venv/bin/python scripts/probe_speculative_decoding.py \
    --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit \
    --drafter mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
    --num-draft-tokens 4 \
    --max-tokens 256 \
    --output research/analyst_runs/$(date +%F)/spec_decode.json
```

Predicted speedup math (T_main=62.5 ms, T_draft=5 ms):

| α | N=4 | N=8 |
|---:|:---:|:---:|
| 0.5 | 2.27× | **3.05×** |
| 0.7 | 2.88× | 4.02× |

Goal 3 (50 tok/s) target requires ≥3.1× speedup from 16 tok/s baseline
at 16K context. Achievable at α≥0.5, N=8.

## Forensic Performance Tooling

When investigating perf regressions or comparing runs, use the
`omlx.observability` stack + `tools/analyst_kit/` (Tasks 357-364):

**Probe writing**: `from omlx.observability import median_of_n` —
drop-in for the inline median-of-N timer pattern. Defaults
(warmup=30, n=60) enforce the Task 340 floor; per-iteration timings
go into the named registry timer for queryable percentiles.

**Persisting registry data**: set `OMLX_REGISTRY_DUMP=path.json`
before invoking any probe. An atexit hook dumps the full registry
state on Python exit — zero probe-side code changes.

```bash
OMLX_REGISTRY_DUMP=/tmp/before.json .venv/bin/python scripts/probe_X.py
# ... edit code ...
OMLX_REGISTRY_DUMP=/tmp/after.json  .venv/bin/python scripts/probe_X.py
```

**Comparing runs**: `python -m tools.analyst_kit.registry_diff
A.json B.json` — reports per-timer p50/p95 deltas with
REGRESSION/IMPROVEMENT/mixed/noise flags. `--exit-nonzero-on-regression`
makes it usable as a CI gate.

**Microbenches** in `tools/analyst_kit/`: `duokv_microbench`,
`snapkv_microbench`, `tq3_microbench`, `duokv_leak_test`,
`mlx_invariant_claims`. Each is self-contained, runs in <10s without
a model load, and traces production code paths via `trace_class` /
`trace_module`. See `tools/analyst_kit/README.md` for the
symptom→tool decision table.

**Disable globally**: `OMLX_OBSERVABILITY=0` makes timer/counter
calls near-zero-cost no-ops, suitable for shipping instrumentation
in production hot paths.

## Code Style

- Python 3.13, MLX framework
- No unnecessary abstractions — minimize for current task
- Test-first: write test, verify failure, implement, verify pass
- Every memory allocation must be justified
- mx.eval() only when needed — don't force graph evaluation unnecessarily
- mx.clear_cache() after large operations to prevent Metal memory growth

## Critical Warnings

- NEVER commit without running the benchmark — `hypercar_bench` gates are non-negotiable
- prefill_last_logit_patch is REQUIRED for contexts > 8K (prevents 40GB logits tensor)
- vertical_eval patch is needed for TQ3 mode only (prevents graph hoarding), NOT for native mode
- TQ3 mode requires `--fp16-layers 1` minimum (layer 0 must be fp16 for quality)
- The TQ3 codebook uses WHT rotation (NOT Givens — Givens is broken, produces garbage)
