# oMLX Hypercar

## Project Overview

oMLX Hypercar — high-performance local LLM inference on Apple Silicon with compressed KV caches. Serves Qwen3-Coder-30B-A3B via OpenAI-compatible API with 1M token context window on 48GB.

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
```

Memory limits are percentage-based (auto-detected from system RAM):
  - Metal peak: 80% of system memory
  - Swap delta: 17% of system memory
  - Metal at load: 42% of system memory
  - Breaches cause IMMEDIATE abort (fail-fast watchdog)

Gate summary:
  - Memory: no breach during any phase (Metal, swap)
  - Smoke: model loads, generates tokens
  - Coherence: 2/2 basic checks
  - Code intelligence: >= 3/5 problems
  - NIAH: 4K retrieval passes
  - HumanEval (--full): >= 50% pass@1

## Key Architecture

- Model: Qwen3-Coder-30B-A3B-Instruct-4bit (17.2GB, 48 layers, MoE 3B active)
- Server: omlx/hypercar_server.py (OpenAI-compat, prompt caching, tool parse safety)

### KV Cache Modes (`--kv-mode`)

| Mode | Cache Type | Compression | Features | Use Case |
|------|-----------|-------------|----------|----------|
| `native` (default) | MLX QuantizedKVCache(bits=3, group_size=64) | 5.3x | Fast decode, battle-tested | Production serving |
| `tq3` | TurboQuantKVCache (WHT rotation + codebook) | 5.3x | save/load, rewind, fork | Agentic workflows |
| `fp16` | Standard KVCache | 1x | Baseline quality | Testing, ~75K max |

TQ3 mode uses Walsh-Hadamard Transform (per arXiv:2504.19874) for full dimension decorrelation. This makes the Beta((d-1)/2, (d-1)/2) codebook valid, producing correct code output.

### Memory Budget

1M context at 39.7GB with 3-bit KV (native or tq3). Fits in 48GB with 8GB headroom.

| Context | KV Cache | Total |
|--------:|----------:|------:|
| 65K | 1.4 GB | 18.6 GB |
| 256K | 5.6 GB | 22.8 GB |
| 1M | 22.5 GB | 39.7 GB |

## Performance Targets

- Decode: >40 tok/s at 2K context, >10 tok/s at 16K
- Prefill: >300 tok/s
- Memory: <42% system RAM at load, <80% peak
- Quality: 3/5 coding problems must pass
- TTFT (cached prompt): <1s

## Server Usage

```bash
# Standard (fast, proven)
python -m omlx.hypercar_server --kv-mode native --port 8080

# With agentic features (session persistence, rewind, fork)
python -m omlx.hypercar_server --kv-mode tq3 --fp16-layers 1 --port 8080

# OpenCode connection
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=hypercar
opencode --model "hypercar/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
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
