# oMLX Hypercar

## Project Overview

oMLX Hypercar — high-performance local LLM inference on Apple Silicon with compressed KV caches. Serves Qwen3-Coder-30B-A3B via OpenAI-compatible API.

## Before Every Commit

REQUIRED: Run the benchmark before committing:
  .venv/bin/python -m omlx.bench.hypercar_bench

For quick iteration (smoke + coherence only, ~30s):
  .venv/bin/python -m omlx.bench.hypercar_bench --quick

For full validation including HumanEval (~15min):
  .venv/bin/python -m omlx.bench.hypercar_bench --full

ALL gates must pass before committing. Fix any failures first.

Memory limits are percentage-based (auto-detected from system RAM):
  Metal peak: 80% of system memory
  Swap delta: 17% of system memory
  Metal at load: 42% of system memory
  Breaches cause IMMEDIATE abort (fail-fast watchdog).

Gate summary:
  - Memory: no breach during any phase (Metal, swap)
  - Smoke: model loads, generates tokens
  - Coherence: 2/2 basic checks
  - Code intelligence: >= 3/5 problems
  - NIAH: 4K retrieval passes
  - HumanEval (--full): >= 50% pass@1

## Key Architecture

- Model: Qwen3-Coder-30B-A3B-Instruct-4bit (17.2GB, 48 layers, MoE 3B active)
- KV Cache: MLX native QuantizedKVCache(bits=3, group_size=64) — 5.3x compression
- Memory budget: 48GB M4 Pro, 38GB Metal limit, 8GB swap limit
- Server: omlx/hypercar_server.py (OpenAI-compat, prompt caching)
- TurboQuant KV: omlx/turboquant_kv.py (Walsh-Hadamard rotation + codebook, being fixed)

## Performance Targets

- Decode: >40 tok/s at 2K context, >10 tok/s at 16K
- Prefill: >300 tok/s
- Memory: <20GB at load, <28GB at 16K context
- Quality: 3/5 coding problems must pass

## Code Style

- Python 3.13, MLX framework
- No unnecessary abstractions — minimize for current task
- Test-first: write test, verify failure, implement, verify pass
- Every memory allocation must be justified
- mx.eval() only when needed — don't force graph evaluation unnecessarily
- mx.clear_cache() after large operations to prevent Metal memory growth

## Critical Warnings

- NEVER commit with swap > 8GB during benchmark — indicates memory leak
- The TurboQuantKVCache (custom codebook) is experimental — use QuantizedKVCache for production
- prefill_last_logit_patch is REQUIRED for contexts > 8K (prevents 40GB logits tensor)
- vertical_eval patch is only needed for TurboQuant streaming, NOT for native QuantizedKVCache
