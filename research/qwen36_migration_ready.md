# Qwen3.6 migration: bench-validated, ready to flip

_2026-05-02 — Task 253 / Task 293 close-out._

## Headline

**`mlx-community/Qwen3.6-35B-A3B-4bit` is the new default model.** It matches Qwen3-Coder-30B-A3B-8bit on every gate the codebase tracks, beats it on the two gates we cared most about, runs lighter on Metal, and crosses the 256K-context barrier that Qwen3-Coder cannot.

## Bench results vs Qwen3-Coder-30B-A3B-Instruct-8bit baseline

| Gate | Qwen3.6-4bit (fp16 KV) | Qwen3-Coder-8bit | Δ |
|---|---|---|---|
| Phase 0 decode | **70 tok/s** | ~52 | +35% |
| Phase 1 coherence | PASS | PASS | = |
| Phase 2 code intel | 5/5 | 5/5 | = |
| Phase 3 NIAH 4K + 16K | PASS | PASS | = |
| Phase 3b RULER | 100% on all 6 subtasks | 100% | = |
| Phase 3c MMLU-Pro | **71%** | 62% | **+9 pp** |
| Phase 3d LiveCodeBench | **55%** | 30% | **+25 pp** |
| Phase 4 HumanEval Lite | 95% | 95% | = |
| Phase 5 Memory | 26.7 GB peak | ~35 GB | −8 GB |
| Wall time | 43.5 min | ~40 min | ~ = |

## Long-context (Goal 1)

| Context | Cache | Result | Metal peak |
|---|---|---|---|
| 4K | fp16 | PASS | 19.6 GB |
| 16K | fp16 | PASS | 19.6 GB |
| 64K | fp16 | PASS | 22.8 GB |
| 256K | fp16 | **FAIL** — disk thrash (KV 10.5 GB + transients > RAM) | swap death |
| **256K** | **`--kv-mode native --kv-bits 4`** | **PASS** | **26.3 GB** |

The 256K result on Qwen3.6-4bit + int4 KV is the major Goal 1 milestone. With 15 GB of Metal headroom remaining at 256K, 512K is plausible (projected ~110 min wall time, KV ~5.2 GB) and 1M is structurally feasible (~10.5 GB KV, ~7.5 hr wall time).

## Why Qwen3.6 has structural long-context advantages

Hybrid architecture: 40 layers split as **30 SSM/linear + 10 full-attention** (every 4th is attention; `full_attention_interval=4`). Only 25% of layers carry per-token KV state. Concretely at 256K with int4 KV:
- 10 attention layers × 4 KV heads × 256 head_dim × 256K × 2 (K+V) × 0.5 bytes (int4) ≈ **2.6 GB**
- vs Qwen3-Coder dense: 48 attn × 4 × 128 × 256K × 2 × 0.5 = ~6.3 GB (2.4× heavier)

The bench's old memory projector overstates 256K by ~5x because it assumes all layers carry KV; in reality Qwen3.6 fits 256K in 26 GB total Metal.

## Bench-side fixes that landed during the migration

All in `omlx/bench/hypercar_bench.py`. 10 lock-in unit tests in `tests/test_bench_helpers.py`.

| Fix | Symptom on Qwen3.6 | Location |
|---|---|---|
| Hybrid `_make_cache` | `'DuoKVCache' object is not subscriptable` on SSM layers | `_make_cache` rebuilt to call `model.make_cache()` first, swap only KVCache slots |
| Multi-EOS `_eos_token_ids` | HumanEval emitted literal `<|endoftext|>` text → SyntaxError on exec → 60% pass rate | New helper unifies `eos_token_id` (single int) and `eos_token_ids` (set); used in 4 sites |
| Thinking-aware `_format_chat_prompt` | `<think>` block consumed answer budget → coherence/NIAH/HumanEval failed | New helper, applied to all 6 chat-template call sites with per-phase `enable_thinking` choice |
| MMLU-Pro 1536-token cap | 22% accuracy (sub-random) at 512 — answer truncated mid-reasoning | `MMLU_PRO_MIN_MAX_TOKENS` 512 → 1536 |
| DuoKV hybrid auto-fallback | DuoAttention policy was Qwen3-Coder-calibrated; on Qwen3.6's 10 attn layers, eviction kicked the needle out | `_make_cache` warns and falls back to fp16 for KV layers when duo + hybrid |
| SnapKV hybrid skip | `'DecoderLayer' object has no attribute 'self_attn'` (Qwen3.6 SSM layers) | `phase3e_snapkv_quality` early-returns on hybrid (SnapKV is per-token KV eviction; SSM state is fixed-size, not applicable) |
| `native --kv-bits 3` shape mismatch | mlx_lm `QuantizedKVCache` pre-allocates `head_dim // (32 // bits)` slots but `mx.quantize` produces `head_dim * bits // 32` — formulas only agree when bits ∈ {2, 4, 8} | `_make_cache` warns and falls back to bits=4 when native + bits=3 + non-divisible head_dim |
| NIAH headline tok/s key | `KeyError: 4096` when `--niah-context 64K` skipped 4K | use `min(results.keys())` instead of hardcoded 4096 |

## How to ship

The flip is one line. `omlx/model_constants.py:31` currently reads:

```python
DEFAULT_MODEL_ID = MODEL_QWEN3_CODER_30B_8BIT
```

Changing to `MODEL_QWEN36_35B_4BIT` propagates through `SERVER_DEFAULT_MODEL_ID`, `BENCH_DEFAULT_MODEL_ID`, all 17 call sites that import these constants. `AGENTIC_DEFAULT_MODEL_ID` and `CALIBRATION_DEFAULT_MODEL_ID` could remain on Qwen3-Coder-4bit for fast-iteration scripts, or also flip — they're independent.

CLAUDE.md changes that follow:
- `Project Overview` — replace "Qwen3-Coder-30B-A3B" with "Qwen3.6-35B-A3B (Tier S2)"
- Goal-status table — update memory and decode numbers to the Qwen3.6 baseline
- "Key Architecture" — Model line, KV cache modes table (note hybrid layout)
- "Performance Targets" — bump the achieved numbers (decode 70, MMLU-Pro 71, LCB 55)
- "Server Usage" examples — update model path

Recommended sequence:
1. Flip `omlx/model_constants.py:31`
2. Run `pytest tests/test_model_constants.py` — should still pass (tests aren't pinned to Qwen3-Coder)
3. Run `pytest -k 'not slow'` for a full sanity sweep
4. Update CLAUDE.md per the bullets above
5. Commit as one or two atomic changes

## Backwards-compatibility checked (2026-05-02 17:04)

`hypercar_bench --quick --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` still **ALL GATES PASSED** with every Qwen3.6-targeted fix in place. The unified bench code handles both architectures:
- `_make_cache(model)` on Qwen3-Coder: model has no SSM layers, every cache slot is `KVCache`, so it gets the requested KV variant exactly like before.
- `_format_chat_prompt(enable_thinking=False)` on Qwen3-Coder: tokenizer ignores the unknown kwarg via the `TypeError` retry path (Qwen tokenizers happen to accept it silently).
- `_eos_token_ids` on Qwen3-Coder: just collects the singular `eos_token_id` — same behavior as the old singular check.
- The `MMLU_PRO_MIN_MAX_TOKENS` 512 → 1536 bump is purely additive headroom; Qwen3-Coder's terse responses still fit.
- The duo-hybrid-fallback only triggers when the cache mix contains non-`KVCache` slots, which Qwen3-Coder never produces.

This means the Qwen3.6 work has zero impact on existing Qwen3-Coder workflows — flipping the default model is a pure-additive change.

## What's not yet validated on Qwen3.6

- 512K and 1M NIAH (projected feasible from 256K result; 1M is wall-time-impractical for cycle work)
- TQ3 mode at long context (only `--quick` smoke validated)
- DuoKV with a Qwen3.6-specific calibration (fp16 fallback works for now; a regenerated policy could speed up decode at 16K+)
- Agentic flows / OpenCode — `AGENTIC_DEFAULT_MODEL_ID` is independent, not flipped here
