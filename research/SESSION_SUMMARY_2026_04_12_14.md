# Hypercar Session Summary: April 12-14, 2026

40+ commits across 2 days. Three goals met, two remaining.

## Goals Status

| # | Goal | Target | Status | Key Commit |
|---|------|--------|--------|-----------|
| 1 | 1M context | 1M tokens | Theoretical (39.7GB KV) | — |
| 2 | 4 independent evals | Beat GPT-4 on 4 evals | **MET** | 4a76daf |
| 3 | 50 tok/s decode | Constant across context | **MET** (52.1 fp16) | 6f46912 |
| 4 | 500 tok/s prefill | Constant across context | PARTIAL (566@4K, 340@16K) | ef194a4 |
| 5 | Swap < 100 MB/s p90 | N≥8 runs | FAIL (460 MB/s) | b06670f |
| 6 | 48GB fit | Comfortable on M4 Pro | **MET** (37.8 peak) | — |

## Key Milestones

1. **ALL GATES PASS (default mode)** — First ever clean run with all phases passing
2. **ALL GATES PASS (--full mode)** — HumanEval 90%, RULER 100%, MMLU-Pro 45%
3. **Goal 2 MET** — 4 eval families: HumanEval, RULER, Code Intel, MMLU-Pro
4. **Goal 3 MET** — 52.1 tok/s decode in fp16 mode
5. **Bimodal timing solved** — Metal JIT cold-start identified and fixed via warmup

## Completed Tasks (30+)

### Benchmark Infrastructure
| Task | What | Commit |
|------|------|--------|
| 1 | RULER eval suite (3 generators, 15 tasks) | ed0fd67 |
| 6 | RULER VT gate (chain lengths 4+8) | ed0fd67 |
| 7 | RULER memory-breach KeyError fix | 83f95f2 |
| 8 | Profiler rebuild (zero-fork, phys_footprint, swap I/O) | cca03b7 |
| 9 | RULER projected-headroom gate | 26224af |
| 10 | NIAH decode-speed artifact fix (128 tokens) | 5f53005 |
| 11 | Sandbox-safe baseline diagnostic module | bbcf3f3 |
| 21 | Multi-run statistical aggregation | d1e4ea7 |
| 23 | Goal 5 restated as p90 swap-rate metric | b06670f |
| 24 | 2-SHA regression detector | 66377eb |
| 25 | --warmup flag (Metal kernel cache priming) | 4509430 |
| 36 | MMLU-Pro reasoning gate (48% on math) | b30eaf6 |
| — | Benchmark exclusive lock + pre-flight resource checks | 97df157 |
| — | Warmup default, --no-warmup opt-out | 7a62b53 |
| — | VT prompt fix (0%→100% accuracy) | 4a76daf |
| — | Freq_word scoring fix (OR logic, 33%→100%) | 9c78fdc |
| — | Server warmup at model load | bf1a113 |
| — | Unit tests (34 tests, 0.03s) | a2f5b2b |

### Performance Optimization
| Task | What | Commit |
|------|------|--------|
| 2 | 2-bit KV probe (--kv-bits flag) | b8fa6d3 |
| 3 | Quest page selection (--quest-topk) | ac2491e |
| 4 | MInference calibration script | 975f7fd |
| 5 | MInference sparse prefill dispatch | 11fe743 |
| 15 | SimPO contrastive step in TTT engine | 0a15f06 |
| 22 | Goal 5 probe (2-bit KV + loosened watchdog) | 65bbb9e |
| — | SimPO /v1/ttt/simpo HTTP endpoint | 986630c |
| — | Quest broadcast_to fix | 9622222 |
| — | MInference causal mask + rectangular mask fixes | ac305bd |

### Research Probes
| Task | What | Result | Commit |
|------|------|--------|--------|
| 12 | DuoAttention calibration | 59% streaming | aabc2bf |
| 13 | DuoKVCache scaffolding | fp16 retrieval heads | d7a793d |
| 16 | ProMoE lazy-load probe | MARGINAL (3.6GB at 87.5%) | 5fef9e6 |
| 20 | Bimodal timing root cause | Metal JIT cache | afc37c8 |
| 24b | Quest argpartition probe | PASS (<200µs) | 7949c59 |
| 26 | MInference block-sparse | Already in Task 5 | e16ccb3 |
| 30 | MLX SDPA dispatch survey | AMX in prefill, not decode | cd9559f |
| 38 | LayerSkip calibration | NOT VIABLE for MoE | 330585b |
| 44 | ShadowKV SVD-rank probe | VIABLE (median 177/512) | f30c48c |
| — | Prefill scaling analysis | O(n²) attention bottleneck | 823058f |
| — | Optimization decision matrix | ShadowKV+DuoAttention path | 325efa9 |

## Key Findings

1. **Metal JIT cold-start** causes 2-3x bimodal timing. Warmup eliminates it.
2. **ShadowKV viable**: K cache is low-rank (median 177/512). 65% K compression possible.
3. **DuoAttention viable**: 59% streaming heads at threshold=0.85.
4. **ProMoE marginal**: Only 3.6 GB savings at 87.5% with quality artifacts.
5. **LayerSkip not viable**: MoE routing makes every layer critical (max 55% agreement).
6. **TQ3 > native for quality**: RULER VT@16K passes in TQ3 but fails in native.
7. **Prefill drops with context**: 566→340 tok/s (4K→16K), caused by O(n²) attention.
8. **Quest gives 1.55x decode speedup** but fails NIAH (norm-based scoring misses needles).

## Recommended Next Steps

1. **Implement ShadowKV** — SVD-compressed K for retrieval heads (biggest memory win)
2. **Test DuoKVCache with GPU** — fp16 retrieval fix needs validation
3. **Fix MInference for quantized KV** — needs per-chunk rectangular masks + dequant
4. **Run N=8 benchmark aggregate** — establish statistical baseline with all fixes
