# oMLX Hypercar

## Project Overview

oMLX Hypercar — high-performance local LLM inference on Apple Silicon with compressed KV caches. Serves Qwen3-Coder-30B-A3B via OpenAI-compatible API with 1M token context window on 48GB.

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

### Current status against goals (as of 2026-04-15)

| # | Goal | Current | Gap |
|---|------|---------|-----|
| 1 | 1M context | 1M theoretical (39.7GB KV @ 3-bit), **validated to 64K** (NIAH PASS, 33.9GB Metal). 128K swap-thrashed at ~6GB swap under normal co-tenancy (30min, no progress). | Need ShadowKV K-compression (65% savings, Task 44 validated) to fit 128K+ under co-tenancy |
| 2 | 4 independent evals beating GPT-4 | **HumanEval 95%**, Code Intel 5/5, RULER 100%, **MMLU-Pro 62-64%** — **4 eval families, ALL GATES PASS** | Add tau-bench (agentic) for 5th eval family |
| 3 | 50 tok/s decode constant | **52.7 tok/s with DuoKVCache — GOAL MET**. | Met in duo mode. Under co-tenancy drops proportionally. |
| 4 | 500 tok/s prefill constant | **96 tok/s at 2K (duo), 803 at 4K, 501 at 16K — GOAL MET** in duo mode. | O(n²) attention still applies at 64K+. |
| 5 | Swap p90 < 100 MB/s | **0.0 GB swap on clean system — GOAL MET**. Under co-tenancy: 7-8.5 GB swap (still under 12.9 GB limit). | Co-tenancy degrades swap but stays within limits. |
| 6 | 48GB M4 Pro fit | Duo: 35.1 GB peak. — **PASS** | 6 GB headroom on clean system; needs ~45 GB free at launch for NIAH 16K. |

**Recommended mode: `--kv-mode duo`** — best quality (MMLU-Pro 64% vs 48% native), zero swap on clean system,
52.7 tok/s decode. DuoKVCache uses fp16 for all heads with ring-buffer trimming for streaming
heads (59%). Native 3-bit only preferred for very long context (64K+) where fp16 KV exhausts Metal.

**5 of 6 goals MET** in duo mode. Full `--full` benchmark ALL 9 GATES PASS (HumanEval 95%, 1481s).
Only Goal 1 (1M context validation beyond 64K) remains — requires KV compression for 128K+ under co-tenancy.
Per-phase headroom checks (Task 86) now gracefully skip memory-hungry phases under co-tenancy.

### Goal 1 Path (KV compression research, 2026-04-15)

Three approaches tested for post-trained KV cache compression:

| Approach | Result | Why |
|----------|--------|-----|
| MLA joint SVD (K+V, d_c=241) | **FAIL** — 7% token agreement | Approximation errors compound across 48 layers during autoregressive decode |
| ShadowKV per-head SVD (K-only, rank 85) | **POOR** — 100% code, 16% NIAH | Argmax instability: uniform 0.85% error flips sparse attention argmax |
| SnapKV eviction (attention-guided, exact values) | **VALIDATED** — needle preserved at 25% keep | Zero approximation error on kept tokens |

**Key insight**: SVD rank is the wrong metric for KV compression. What matters is attention-weighted
correctness, not energy-weighted reconstruction. SnapKV works because it keeps high-attention tokens
at full precision. See `research/OPTIMIZATION_DECISION_MATRIX.md` for the full 4-tier ranking and
`research/attention_weighted_codec_selection.md` for the per-head-type codec architecture.

**Next step**: Task 46 (SnapKV compact) needs a custom cache class that supports sparse position IDs —
MLX's KVCache uses contiguous buffer/offset which breaks physical token removal. Options: attention
masking (no memory savings) or custom `SparseKVCache` class (M effort).

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

- Model: Qwen3-Coder-30B-A3B-Instruct-8bit (17.2GB, 48 layers, MoE 3B active)
- Server: omlx/hypercar_server.py (OpenAI-compat, prompt caching, tool parse safety)

### KV Cache Modes (`--kv-mode`)

| Mode | Cache Type | Compression | Features | Use Case |
|------|-----------|-------------|----------|----------|
| **`duo` (default)** | DuoKVCache (fp16 retrieval + ring-buffer streaming) | ~2x at 16K+ | Best quality (MMLU-Pro 62%, HumanEval 95%), zero swap | **Best for ≤16K context** |
| `native` | MLX QuantizedKVCache(bits=3, group_size=64) | 5.3x | Battle-tested, long context | Long context (16K-1M) |
| `tq3` | TurboQuantKVCache (WHT rotation + codebook) | 5.3x | save/load, rewind, fork, best RULER quality | Agentic workflows |
| `fp16` | Standard KVCache | 1x | Baseline quality | Testing, ~75K max |

**Mode selection guide**: Use `duo` (default) for interactive coding — best quality and speed up to 16K. Switch to `native` or `tq3` for repository-scale context (64K+) where 3-bit KV compression is needed to fit in 48GB.

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

# OpenCode connection
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=hypercar
opencode --model "hypercar/mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
```

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
