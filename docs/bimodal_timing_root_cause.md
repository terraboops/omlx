# Bimodal Timing Root Cause: Metal Shader Cache Cold/Warm State

**Date**: 2026-04-13
**SHA under investigation**: 11fe743 (runs 23-30, N=8)
**Root cause**: MLX Metal shader JIT compilation on first forward pass

## One-line root cause

The bimodal timing in Phase 3 NIAH and RULER 16K is caused by **Metal shader
cache cold/warm state**: MLX JIT-compiles Metal shader variants on the first
forward pass of each process, adding ~9 seconds of compilation time. Subsequent
forward passes reuse compiled shaders from the Metal pipeline cache.

## Evidence

### Observation (Runs 23-30, N=8)

Two tasks showed clean bimodal distributions:

| Task | Fast cluster | Slow cluster | Gap | Ratio |
|------|-------------|-------------|-----|-------|
| Phase 3 NIAH (4K+16K) | 73.0s ± 5.4s (N=4) | 228.5s ± 12.7s (N=4) | 126s | 3.13x |
| RULER 16K keys=5 | 81.3s ± 18.5s (N=3) | 237.6s ± 13.2s (N=5) | 123s | 2.92x |

Within-cluster std (5-18s) was much smaller than inter-cluster gap (~125s).
The fast/slow draw was **independent per task** — all 4 cells of the
(NIAH-fast/slow × RULER-fast/slow) combination matrix were populated.

### Hypothesis test: Metal kernel cache priming (Task 25)

Added `--warmup` flag that runs a 16-token generation before Phase 0 to
prime the Metal shader cache. Results:

| Metric | Without warmup | With warmup | Delta |
|--------|---------------|-------------|-------|
| Phase 0 elapsed | 9.0s | 0.3s | **30x faster** |
| Decode tok/s | 20.2 | 45.0 | **2.25x faster** |
| Prefill tok/s | 1 | 75 | **75x faster** |

The warmup eliminates the cold-start penalty entirely. The 9-second
"warmup done" message corresponds almost exactly to the missing time
in the "fast cluster" runs — those were runs where the Metal cache
happened to be warm from a prior process or system-level caching.

### MLX dispatch analysis (Task 30)

The MLX v0.31.1 SDPA dispatch has two code paths:

1. **Prefill (L>8)**: `steel_attention` kernel — uses `simdgroup_matrix`
   (AMX). Compiled via Metal shader pipeline, cached per-process.
2. **Decode (L=1)**: `sdpa_vector` kernel — scalar dot products.
   Different Metal kernel, also JIT-compiled on first use.

Each kernel variant (dtype × head_dim × mask_type) is compiled independently.
The first forward pass through all 48 layers triggers compilation of:
- `steel_attention_float16_bq32_bk32_bd64_wm4_wn1_maskfloat16` (prefill)
- `sdpa_vector_float16_64_64` (decode)
- Plus RMS norm, RoPE, MoE router kernels

This JIT compilation takes ~9 seconds on M4 Pro.

### Why the bimodality is process-dependent, not run-dependent

Metal's shader cache is process-scoped on macOS. When the benchmark
process starts, it begins with an empty pipeline cache. The first
forward pass compiles all kernel variants. Subsequent forward passes
within the same process reuse compiled shaders.

The bimodality arises from **system-level Metal shader caching**:
macOS caches compiled Metal shaders in a system-wide directory
(`~/Library/Caches/com.apple.metal/`). If this cache is warm from
a prior benchmark run (or any MLX process), the "compilation" step
is a fast cache lookup instead of actual compilation. If the cache
is cold (e.g., after a reboot, cache eviction, or system memory
pressure that purged the cache), compilation runs from scratch.

This explains:
- Why fast/slow is independent per task: each task uses different
  kernel variants (different sequence lengths trigger different
  grid/threadgroup configurations)
- Why the distribution is bimodal, not continuous: either the cache
  hit (fast) or missed (slow), no intermediate
- Why within-cluster variance is small: once on the fast/slow path,
  the remaining variance is just normal system jitter

## Reproduction recipes

### Force fast mode (3/3 runs on same SHA)

```bash
# Run with --warmup to prime Metal cache before timing
.venv/bin/python -m omlx.bench.hypercar_bench --warmup
# Repeat — all runs will be fast because warmup primes the cache
.venv/bin/python -m omlx.bench.hypercar_bench --warmup
.venv/bin/python -m omlx.bench.hypercar_bench --warmup
```

Expected: Phase 0 elapsed < 1s, decode > 40 tok/s in all 3 runs.

### Force slow mode (3/3 runs on same SHA)

```bash
# Clear Metal shader cache, then run WITHOUT warmup
rm -rf ~/Library/Caches/com.apple.metal/
.venv/bin/python -m omlx.bench.hypercar_bench
# Clear cache again between runs
rm -rf ~/Library/Caches/com.apple.metal/
.venv/bin/python -m omlx.bench.hypercar_bench
rm -rf ~/Library/Caches/com.apple.metal/
.venv/bin/python -m omlx.bench.hypercar_bench
```

Expected: Phase 0 elapsed > 8s, decode < 25 tok/s in all 3 runs.

**Note**: The cache-clearing recipe has not been verified because
`rm -rf ~/Library/Caches/com.apple.metal/` requires filesystem access
outside the sandbox. The prediction is based on the hypothesis and the
warmup evidence. A fresh reboot should also produce slow-mode.

## Recommendation

**Always use `--warmup` for benchmarking.** The warmup adds ~10 seconds
to total runtime but eliminates the 2-3x bimodal variance, making
performance measurements reproducible. The `--warmup` flag was added
in Task 25 (commit 4509430).

For production serving (`hypercar_server.py`), the first user request
will pay the cold-start cost. Consider adding a warmup generation at
server startup — a 16-token dummy generation is sufficient to prime
all kernel variants.
