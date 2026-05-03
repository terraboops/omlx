# Analyst Kit

Standalone tools for measuring and probing inference performance. Each tool
loads (or doesn't load) the model directly — they don't go through the
hypercar bench pipeline. That keeps them quick to write and fast to iterate
when investigating a specific question.

Run with `.venv/bin/python -m tools.analyst_kit.<tool>`. Most accept
`--help` for the full flag list.

## Symptom → tool decision table

Use this when you don't know which tool fits the question. Tools below
the line have detailed sections later in the README.

| Symptom / question | Tool |
|---|---|
| Decode tok/s regressed; need component breakdown | `decode_op_ablation.py` |
| Decode template — copy-and-adapt for a new probe | `decode_template.py` |
| Validate int4 weight quantization saves what we think | `int4_decode_speedup.py` |
| Compare two observability registry dumps | `registry_diff.py` |
| Audit an MLX behavior assumption (broadcast, in-place, etc.) | `mlx_invariant_claims.py` |
| Microbench DuoKVCache update_and_fetch / trim hot path | `duokv_microbench.py` |
| Hunt a DuoKVCache memory leak across decode iterations | `duokv_leak_test.py` |
| Microbench SnapKV gather + scatter at long context | `snapkv_microbench.py` |
| Microbench TQ3 fused quantize + dequantize codec | `tq3_microbench.py` |
| O proj kernel — synthetic matmul at varied bit-widths | `oproj_kernel_microbench.py` |
| Real-model O proj quantization + decode delta | `oproj_int4_patch.py` |
| Real-model MoE quantization + decode delta | `moe_int4_patch.py` |
| Stack MoE + O proj quantization, three-stage validator | `stacked_int4_patch.py` |
| Per-layer expert routing frequency calibration | `expert_frequency_probe.py` |
| Pareto curves over expert coverage | `expert_coverage_analysis.py` |

## Decode-side decomposition

Where does decode wall-time go? These tools answer the question without
the cascade artifact that zero-ablation introduces.

| Tool | What it does |
|------|--------------|
| `decode_op_ablation.py` | Monkey-patch a target component (MoE / Attention / SDPA / QKV / O proj) to return zeros, measure decode tok/s delta vs baseline. **Includes the calibration finding from Task #20**: zero-ablation reports `component + downstream cascade`, NOT `component alone`. Use `qkv_only_zeros` / `oproj_only_zeros` for finer split, or quantize-and-measure (below) for kernel-only attribution. |
| `decode_template.py` | Per-context decode timing template: load model once, install tracers, dump heap diff per phase. Copy-and-adapt starting point. |
| `int4_decode_speedup.py` | Apply int4 MoE patch (with optional skip-first-N) and measure decode tok/s gain at multiple contexts. Used in Task #28: validated +27.5% at 4K, +13.5% at 16K, +28.1% at 64K. |

## Quantize-and-measure (the calibration-corrected path)

Per Task #20, zero-ablation overstates kernel-specific cost because it
zeros the downstream cascade too. To isolate kernel cost, quantize the
component (preserves output values, just changes how matmul runs) and
measure the decode delta.

| Tool | Component | Result captured |
|------|-----------|----------------|
| `oproj_kernel_microbench.py` | O proj kernel | Synthetic `(1, 4096) @ packed_weight` at varied bit widths. Validated 1.52× speedup q4_g64 vs q8_g64. |
| `oproj_int4_patch.py` | All `o_proj` modules | Real-model quantize-and-measure. **Result: +1.7-2.7%** decode (vs zero-ablation's +84-286% wall share — the cascade caveat in action). |
| `moe_int4_patch.py` | All MoE expert weights (gate/up/down) | **+20.8% / +13.0% decode @ 4K/16K**, NIAH PASS, 13.5 GB savings. The actually-shippable win. |
| `stacked_int4_patch.py` | MoE + O proj stacked | Three-stage validator. +28.1% at 4K combined; +14.6% at 16K (O contribution near zero at 16K — once MoE pressure relieved, O proj cache locality recovers). |

## Per-expert routing analysis

Foundation for mixed-precision MoE quantization (the only known recovery
path for the MMLU-Pro -9pp residual under uniform int4).

| Tool | What it does |
|------|--------------|
| `expert_frequency_probe.py` | Monkey-patch `Qwen3MoeSparseMoeBlock.__call__`, record per-layer routing decisions across calibration prompts. Outputs `expert_frequency.json` with raw freq vector + top-5/bottom-5 lists. **31% of expert slots unused** on the cycle's 6-prompt calibration. |
| `expert_coverage_analysis.py` | Read the freq JSON, compute per-layer cumulative coverage curves + elbow points (smallest K reaching 90%/95%/99%). Aggregate stats: top-32 captures 81.3%, top-48 captures 92.5% across layers. **Pareto curve for design decisions on per-expert mixed precision**. |

## Calibration rules captured (read these before adding new tools)

1. **Zero-ablation overstates kernel cost.** When component X "explains 70% of wall" via zero-ablation, that includes its downstream cascade. The `*_zeros` ablations short-circuit residual + RMSNorm + MoE input. To isolate kernel-only cost, **quantize-and-measure**: replace component with a different bit-width version that produces real (just less precise) output, then measure decode delta. Predicted +17% from kernel microbench → actual +1.7% from real-model patch is the canonical example.

2. **NIAH is necessary but not sufficient for quantization claims.** Knowledge tasks (MMLU-Pro), competitive coding (LiveCodeBench), and precision-sensitive KV eviction (SnapKV) reveal regressions that single-token retrieval can't. The full hypercar_bench gate is mandatory before declaring a quantization recipe production-ready. The int4 stack passed NIAH at every stage but failed MMLU-Pro / LCB / SnapKV in the full bench.

3. **Pareto curves on quantization configs are non-monotonic.** Threshold sweep (Task #35) showed protecting MORE layers (threshold=2.0%, 25 layers) gave WORSE quality than the optimum at threshold=3.0% (14 layers). Reasons may include implicit-regularization effects of int4 imprecision, or simply N=100 sample noise on MMLU-Pro. Either way, "more protection = more recovery" is falsified.

## Production recipes (validated 2026-05-02, full hypercar_bench passes)

### Memory-priority (Goal 1, 1M context):
```bash
python -m omlx.hypercar_server --quantize-moe-int4 --quantize-moe-skip-first 1
```
- 13.22 GB weight savings
- Decode +27.5% at 4K, +13.5% at 16K, +28.1% at 64K
- MMLU-Pro 53% (-9pp from baseline 62%)
- All other gates: NIAH/RULER/SnapKV/HumanEval/LCB PASS

### Quality-priority (Goal 2, intelligence):
```bash
python -m omlx.hypercar_server \
    --quantize-moe-int4 \
    --quantize-moe-skip-from-freq research/analyst_runs/2026-05-02/expert_frequency.json
```
- 9.56 GB weight savings (3.66 GB less than memory-priority)
- MMLU-Pro 55% (-7pp, best of any int4 config)
- All other gates equal to memory-priority recipe
- Tradeoff: ~1.83 GB memory per 1pp MMLU-Pro recovered

### ⚠️ Never combine with `--snapkv-keep`
The runtime guard rejects `--quantize-oproj-int4 --snapkv-keep` because
O proj int4 corrupts attention scores enough to break SnapKV's top-K
eviction (digit corruption, Task #25). The MoE recipes above don't
include O proj quantization for this reason.

## Comparing two registry dumps with `registry_diff.py`

For sub-millisecond timings (per-op decode microbenches), GPU pipeline
noise is a larger fraction of the measurement than the default 10% /
0.05 ms threshold. Use `--threshold 30 --floor 0.1` to suppress
false-positive REGRESSION/IMPROVEMENT verdicts on identical-code reruns.
See `registry_diff.py` for the full flag list.

## Capturing observability registries

Observability is opt-in via `OMLX_OBSERVABILITY=1`. To persist a snapshot
across a run, additionally set `OMLX_REGISTRY_DUMP=/path/to/run.json`
before invoking any of these tools — the registry of counters / timers /
heap entries is then written at process exit. `registry_diff.py`
compares two such dumps. Sources of truth: `omlx/observability/registry.py`
for `OMLX_OBSERVABILITY`, `omlx/observability/autodump.py:_ENV_VAR` for
`OMLX_REGISTRY_DUMP`. References must stay in lockstep across the docs
and tests that mention them.

## Adding a new tool

If your investigation needs more than 30 lines of one-off code, write
it here. Pattern:
1. Header docstring with: what question, which task # if applicable, usage example.
2. Output to `research/analyst_runs/<YYYY-MM-DD>/<filename>.json` for any data the next analyst should be able to find.
3. Update this README's table.
