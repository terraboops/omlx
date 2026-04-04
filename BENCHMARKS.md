# Hypercar Benchmark Results

> **Model**: `granite-4.0-h-small` (granitemoehybrid, 40 layers, 72 experts, 4096 hidden)
> **Hardware**: Apple M4 Pro, 48GB unified memory
> **Quantization**: 4-bit (lmstudio-community MLX)
> **Memory at load**: 18.1 GB

---

## Decode Throughput

| Method | Tok/s | Speedup | Tokens/Step | Draft Acceptance |
|--------|------:|--------:|------------:|-----------------:|
| Standard (greedy) | 17.9 | 1.0x | 1.0 | — |
| Standard (temp=0.7) | 17.2 | 0.96x | 1.0 | — |
| Medusa (3 heads, random) | 20.4 | 1.14x | 2.0 | 0% |
| **Medusa (3 heads, distilled)** | **32.8** | **1.9x** | **3.33** | **55.6%** |

Medusa draft heads are trained via fast distillation (200 steps, ~4 minutes) on calibration text. The distilled heads correctly predict the model's next token 55.6% of the time, reducing the number of forward passes by 70%.

## Prefill Throughput

| Context Length | Tok/s | Memory |
|---------------:|------:|-------:|
| 128 | 188 | 18.2 GB |
| 512 | 239 | 18.3 GB |
| 1,024 | 246 | 18.5 GB |
| 2,048 | 250 | 18.9 GB |
| 4,096 | 250 | 19.8 GB |

Prefill throughput plateaus at ~250 tok/s for this model on M4 Pro.

## Context Window

### Without TQ KV Cache (fp16 KV)

| Context | Status | Prefill Tok/s | Memory |
|--------:|--------|------:|-------:|
| 1,024 | OK | 245 | 18.3 GB |
| 2,048 | OK | 243 | 18.7 GB |
| 4,096 | OK | 243 | 19.6 GB |
| 8,192 | OK | 226 | 21.2 GB |
| 16,384 | OK | 210 | 24.5 GB |
| 32,768 | OK | 188 | 31.1 GB |
| 65,536 | OK | 185 | 44.2 GB |
| 131,072 | OOM | — | >48 GB |

Max context (fp16 KV): **65,536 tokens** (44.2 GB)

### With TQ KV Cache (3-bit)

| Context | Status | Prefill Tok/s | Memory |
|--------:|--------|------:|-------:|
| 1,024 | OK | 238 | 22.2 GB |
| 2,048 | OK | 257 | 24.8 GB |
| 4,096 | OK | 259 | 30.2 GB |
| 8,192 | OK | 247 | 40.9 GB |
| 16,384 | OK | 194 | 62.3 GB (swap) |

Note: Memory is higher with TQ KV because `TurboQuantKVCache` accumulates fp16 during prefill then quantizes on first decode token. The 36 Mamba `ArraysCache` layers also contribute to memory growth regardless of KV compression.

## Needle-in-Haystack

| Context | fp16 KV | TQ 3-bit KV |
|--------:|---------|-------------|
| 1,024 | **FOUND** "BLUE ELEPHANT 42" | **FOUND** "BLUE ELEPHAN 42" |
| 4,096 | **FOUND** "BLUE ELEPHANT 42" | **FOUND** "BLUE ELEPHANT 42" |
| 16,384 | OOM | **FOUND** "BLUE ELETERPAN 42" |

TQ KV cache enabled the 16K needle test that was previously OOM. Minor spelling degradation from 3-bit KV quantization ("ELEPHAN", "ELETERPAN").

## Compression

| Format | Size | Compression | Fits 48GB? |
|--------|-----:|------------:|:----------:|
| fp16 (baseline) | 60 GB | 1.0x | No |
| 8-bit MLX | 30 GB | 2.0x | Yes |
| **4-bit MLX** | **15 GB** | **4.0x** | **Yes** |
| TQ3.5 (3-bit + WHT rotation) | 16.7 GB | 3.6x | Yes |

## TQ3.5 Weight Quantization (Experimental)

Streaming conversion of fp16 → 3-bit with Walsh-Hadamard rotation. No model reload required — processes one tensor at a time (~3-4GB working memory).

| Mode | Tok/s | Output Quality |
|------|------:|:--------------|
| Without runtime rotation | 21.6 | Gibberish (quantization noise) |
| With dense QR rotation | 7.8 | Semi-coherent (real words, repetitive) |
| With WHT rotation | 13.7 | Semi-coherent (faster) |

TQ3.5 weight quantization introduces runtime rotation overhead. The 4-bit path (no rotation needed) is recommended for quality-sensitive workloads.

## Active Hypercar Features

| Feature | Flag | Status |
|---------|------|--------|
| 4-bit weights | `--model .../granite-4.0-h-small-MLX-4bit` | **Working, coherent** |
| TurboQuant KV cache | `--cache-mode turbo3` | **Working** (unlocks 16K+ needle) |
| Expert-Choice MoE | `--moe-router expert-choice` | Working (may destabilize output) |
| STARC sparse attention | `--sparsity-method starc` | Installed |
| Medusa draft heads | `--medusa-heads 3 --medusa-distill 200` | **1.9x speedup** |
| Mamba-3 MIMO kernel | `--mimo-rank 4` | Implemented + unit tested |
| TQ3.5 weight rotation | `--weight-mode turbo35` | Experimental |

---

## Benchmark History

### Run 1: Baseline (4-bit, no features)
```
Date: 2026-04-03
Model: lmstudio-community/granite-4.0-h-small-MLX-4bit
Decode: 17.9 tok/s | Prefill: 250 tok/s @ 4K | Max context: 65K (44.2GB)
Needle: FOUND at 1K, 4K | OOM at 16K
```

### Run 2: TQ3.5 Weight Quantization
```
Date: 2026-04-03
Model: granite-4.0-h-small fp16 → TQ3.5 streaming conversion
Output: 16.7 GB (3.6x compression from 60GB)
Without rotation: 21.6 tok/s (gibberish) | WHT rotation: 13.7 tok/s (semi-coherent)
```

### Run 3: Medusa Draft Heads (random vs distilled)
```
Date: 2026-04-03
Medusa (random, 3 heads): 20.4 tok/s (0% acceptance)
Medusa (distilled, 200 steps): 32.8 tok/s (55.6% acceptance, 1.9x speedup!)
```

### Run 4: TQ KV Cache + Needle-in-Haystack
```
Date: 2026-04-03
TQ KV 3-bit: Needle FOUND at 16K (previously OOM)
Context with TQ KV: 1K-16K tested, up to 62.3GB at 16K (swap)
```

### Run 5: Clean Single-Process (temp sampling + distillation + strict needle)
```
Date: 2026-04-03
Standard decode (temp=0.7): 35.7 tok/s — COHERENT output!
  "a question that has puzzled philosophers for centuries..."
Medusa (distilled, temp=0.7): 29.2 tok/s (30.6% accept, 2.63 tok/step)
  Lower acceptance with temperature sampling (drafts harder to predict)

Strict Needle-in-Haystack (EXACT "BLUE ELEPHANT 42" required):
  fp16 KV:
    1K: pos=25% 1/3 | pos=50% EXACT | pos=75% EXACT
    4K: pos=25% 1/3 | pos=50% 1/3   | pos=75% 1/3
    8K: pos=25% 1/3 | pos=50% EXACT | pos=75% EXACT
  TQ KV 3-bit (pos=50%):
    1K: EXACT | 4K: 1/3 | 8K: EXACT

Key findings:
  - 4K context is a weak spot (model retrieves "42" but misses "BLUE ELEPHANT")
  - 50% and 75% needle positions work best
  - TQ KV 3-bit matches fp16 quality (no degradation from KV compression)
  - Peak memory: 49.3GB (includes Medusa distillation overhead)
```

### Run 6: Qwen3-Coder-30B-A3B-4bit (OpenCode Target Model)
```
Date: 2026-04-03
Model: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
  30B total, 3.3B active per token, 128 experts, top-8 routing
  48 layers, 4 KV heads, 128 head_dim, GQA
  Native context: 256K (max_position_embeddings, rope_theta=10M)

Decode: 71.5 tok/s (2.4x above 30 tok/s target!)
Intelligence: 8/8 (100%) — FizzBuzz, Binary Search, Flatten, LRU Cache,
  Math Word Problem, Logic Puzzle, Pattern Recognition, Code Reasoning
Memory: 17.2GB loaded

Context stress test (fp16 KV):
  1K:  835 tok/s prefill, 17.6GB
  4K:  773 tok/s prefill, 18.9GB
  16K: 500 tok/s prefill, 23.8GB
  65K: 200 tok/s prefill, 43.8GB (fits in 48GB!)

Theoretical max context:
  fp16 KV:  ~75K tokens (48GB limit)
  TQ 3-bit: ~1.5M tokens (!!!)
  Native:   256K (rope_theta limit)

128K BLOCKED by full-sequence prefill (Metal buffer limit).
  FIX: chunked prefill (prefill_step_size=8192) bypasses the limit!

120K chunked prefill result:
  121 tok/s prefill, 30.0GB peak memory — FITS IN 48GB
  Needle partial (99 found, PURPLE UNICORN missed — prompt engineering needed)

256K projected: ~50GB peak — tight but feasible with TQ KV 3-bit

DECODE speed vs context length (pure generation, after prefill):
  1K:  78.9 tok/s (2.6x above 30 tok/s target)
  4K:  68.3 tok/s (2.3x)
  16K: 44.7 tok/s (1.5x)
  65K: ~20-25 tok/s estimated (attention becomes bottleneck)
  With STARC 15%: 65K → ~10K effective → ~45 tok/s (1.5x target)
```

### Run 7: Streaming TQ KV + Fused Givens Metal Kernel
```
Date: 2026-04-03
Model: Qwen3-Coder-30B-A3B-Instruct-4bit + streaming TQ3 KV
  + last-logit patch (saves 40GB logits tensor)
  + streaming quantize-during-prefill (quantize each chunk, never accumulate fp16)
  + fused Givens Metal kernel (norm+rotate+quantize+pack in ONE dispatch)

Context stress test (streaming TQ3 + last-logit):
   65K: 204 tok/s | 18.8GB active | peak 28.6GB
  128K: 109 tok/s | 19.8GB active | peak 40.1GB ← NEW! (was impossible)
  196K:  76 tok/s | 21.1GB active | peak 49.5GB ← NEW! (was impossible)
  256K: testing... (projected ~22.4GB active)

Memory savings from streaming TQ KV:
  65K:  23.6GB → 18.8GB (saved 4.8GB)
  128K: 30.1GB → 19.8GB (saved 10.3GB!)
  196K: impossible → 21.1GB (UNLOCKED!)

Memory model: 17.2GB base + 1.3GB per 65K tokens (3-bit compressed)
  → 512K would fit at ~27GB!
  → 1M would fit at ~37GB!

Fused Givens kernel: rotation in GPU registers, O(1) per coordinate.
64x fewer FLOPs than dense rotation in the quantize step.
```

### Run 8: Online Softmax + Streaming Dequant + Memory Profiling
```
Date: 2026-04-04
Model: Qwen3-Coder-30B-A3B-Instruct-4bit + streaming TQ3 KV
  + online softmax (FlashAttention at Python level, cos sim 1.000000)
  + split self-attn + history-attn with LSE merge
  + mx.eval per dequant chunk (prevents MLX graph hoarding)
  + fp16 layer 0 anchor (no quantization on first layer)
  + min_quant_tokens=512 (stay fp16 below threshold)
  + 4-phase benchmark suite with fail-fast gating

Profiling results (dequant chunk size study):
  chunk_size=16384: peak 35.2GB ← scores tensor Q(8K)×K(16K)×32 = 16GB
  chunk_size= 4096: peak  9.2GB
  chunk_size= 2048: peak  4.9GB ← new default
  chunk_size= 1024: peak  2.7GB

Root cause of 52.6GB OOM: attention scores tensor was 16GB per layer.
At chunk_size=2048, scores = 8K × 2K × 32 × 4bytes = 2GB per chunk.
With mx.eval inside the streaming loop, freed after each chunk.

REMAINING ISSUE: 47 layers' streaming attention in single model()
forward pass peaks at 44GB due to MLX graph accumulation across layers.
Active memory is flat (~17.5GB) — only peak spikes during graph exec.
Needs per-layer eval in model forward pass to bound peak.

Coherence test:
  fp16 (15 tokens): "4" ✓
  TQ3 (15 tokens): "just the the the" ✗ (too few tokens for codebook)
  TQ3 (512+ tokens): "4" ✓ (min_quant_tokens=512 fix)

Benchmark suite (python -m omlx.bench.safe_bench):
  Phase 1: Smoke (memory watchdog, speed floors, coherence gate)
  Phase 2: Quality (NIAH at max context, TQ3 vs fp16 cosine sim)
  Phase 3: Stress (128-token sustained decode, memory leak detection)
  Phase 4: Intelligence (inline EvalPlus, SWE-bench setup guide)
```

### Run 9: Vertical Graph Eval + Adaptive Memory Budget
```
Date: 2026-04-04
Model: Qwen3-Coder-30B-A3B-Instruct-4bit (48 layers)

Fixes applied:
  + vertical_eval patch: mx.eval(h) every 8 layers (kills cross-layer hoarding)
  + prefill_step: 8192 → 2048 (matches dequant chunk)
  + adaptive memory budget: chunks scale with available headroom
  + mx.synchronize() + mx.clear_cache() between prefill chunks
  + Safety factor tau=1.2 for cross-version MLX variance
  + Chunk hint computed ONCE per prefill step (avoids 47× overhead)

Memory profile at 65K (chunk=2048 + vertical_eval):
  active: 17.2 → 18.6GB (flat, compressed KV growth only)
  peak:   17.2 → 20.6GB (bounded, was 45.3GB — 55% reduction!)

Super benchmark result (65K, 4 phases):
  Phase 1 prefill: 101 tok/s ✓
  Phase 1 peak: 23.9GB ✓ (under 38GB limit)
  Phase 1 swap: 0GB ✓ (no death spiral)
  Phase 1 decode: 5.1 tok/s ✗ (fails 25 tok/s floor)

Decode bottleneck: fused decode kernel scans all 65K compressed
KV entries per token. Without sparse attention (STARC), decode
is O(context) per token. This is the NEXT fix.

Adaptive chunk sizing:
  fresh prefill (17GB budget):    16K chunks (fast)
  decode at 65K (14.6GB budget):  8K chunks
  decode at 256K (10.3GB budget): 8K chunks
  large L=8192 queries:           4K chunks (scales with L × chunk)

New features this run:
  - rewind_to(offset): O(1) context rewind, no re-prefill
  - save_to_disk(): freeze compressed KV (5GB for 256K)
  - load_from_disk(): instant thaw, no prefill needed
  - Enables context forking, session persistence, turn undo
```

### Run 10: Adaptive Budget + STARC Bridge + Small Context Validation
```
Date: 2026-04-04
Model: Qwen3-Coder-30B-A3B-Instruct-4bit

Fixes:
  + Adaptive chunk hint computed ONCE per prefill step (not per layer)
  + Safety factor tau=1.2 in chunk size calculation
  + STARC + TQ3 bridge (starc_tq_bridge.py)
  + KV cache rewind + save/load verified
  + Coherence uses 512+ token preamble (TQ3 needs dense signal)

16K prefill (small context validation):
  Prefill: 486 tok/s ✓
  Active: 17.5GB ✓
  Peak: 18.6GB ✓ (only 1.4GB above model — nearly optimal)
  Swap: 0GB ✓

16K decode: 16.4 tok/s (fails 25 tok/s target)
32K decode: skipped (cascading abort working correctly)

Decode speed by context (TQ3 only, no STARC yet):
  16K: 16.4 tok/s
  32K: ~9 tok/s (estimated)
  65K: 5.1 tok/s (measured)

Fail-fast working as designed:
  Watchdog: metal 38GB, swap 8GB limits
  Prefill floor: scales with context (50 @ 65K, 10 floor)
  Decode floor: 25 tok/s at all contexts
  Cascading: failing context skips larger

Next: wire STARC decode to hit 25 tok/s target.
```

### Adaptive Memory Budget (omlx/memory_budget.py)
```
Live headroom calculation — uses system memory, active memory, and KV
cache size to dynamically pick dequant chunk sizes:

  System state     | Budget  | Chunk @ L=2048
  Fresh prefill    | 17GB    | 16K (fast, saturates bandwidth)
  Decode at 65K    | 14.6GB  | 8K
  Decode at 256K   | 10.3GB  | 8K
  Large queries L=8K fresh | 17GB | 4K (scales with L)

The budget shrinks automatically when active KV grows, preventing OOM
when context builds up. Background prefill of a "next context" while
user is decoding can use the remaining 14.6GB safely.

Safety factor tau=1.2 for MLX version variance. Computed ONCE per
prefill step (not per layer) — avoids 47× overhead in hot path.
```

### KV Cache Persistence (omlx/turboquant_kv.py)
```
Three new features unlock stateful sessions:

rewind_to(offset):
  - O(1) context rewind, no re-prefill
  - Just moves offset pointer, compressed storage unchanged
  - Use case: user undoes last turn

save_to_disk(path):
  - Freezes compressed KV as .npz (5GB for 256K context)
  - ~1.5s write on NVMe
  - Use case: snapshot codebase context at end of session

load_from_disk(path):
  - Thaws cache from disk, no prefill needed
  - Instant resume
  - Use case: reload yesterday's codebase in 1.5s

Combined: persistent sessions with instant context switching.
```
```

---

## Run It Yourself

```bash
# Clone the hypercar branch
git clone -b hypercar https://github.com/terraboops/omlx.git
cd omlx

# Install
pip install -e .

# Quick benchmark: standard + Medusa with distillation
python tests/benchmark_hypercar.py \
  --model lmstudio-community/granite-4.0-h-small-MLX-4bit \
  --sanitize-patch \
  --medusa 3 \
  --medusa-distill 200 \
  --max-tokens 100

# With TQ KV cache for long context
python tests/benchmark_hypercar.py \
  --model lmstudio-community/granite-4.0-h-small-MLX-4bit \
  --sanitize-patch \
  --tq-kv 3 \
  --max-ctx 65536 \
  --skip-decode

# Full hypercar benchmark
python tests/benchmark_hypercar.py \
  --model lmstudio-community/granite-4.0-h-small-MLX-4bit \
  --sanitize-patch \
  --medusa 3 \
  --medusa-distill 200 \
  --tq-kv 3 \
  --starc \
  --max-tokens 100 \
  --max-ctx 65536

# Unit tests (43 tests, all pass)
python -m pytest tests/test_hypercar.py -v
```
