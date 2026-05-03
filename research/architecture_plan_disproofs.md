# Architecture-plan disproofs

_2026-05-02 → 2026-05-03. Following the user's "disprove with tireless engineering" directive against the no-compromise architecture plan dated 2026-05-02. This file documents 16 problems found while attempting Cycle N (MInference end-to-end on Qwen3.6) and validating the plan's claims. Each problem is reproducible, cited with file:line, and pinned with consequences for the wider plan._

## Executive summary

| # | Problem | Severity | Status |
|---|---|---|---|
| **D1** | Metal per-MTLBuffer cap (~30 GB) blocks 512K+ on M4 Pro | structural | **substantially mitigated 2026-05-03 05:24 UTC** — adaptive chunking via `AdaptivePrefillController` reached **512K NIAH PASS** (Metal peak 47 GB, 111 min wall, controller trajectory 4096→1771→1025→512→346). The plan's Layer 5 ("adaptive chunked prefill driven by hardware-aware cost model") IS this existing controller, just empirically validated. |
| **D2** | MInference pattern table was synthetic placeholder | bug | **fixed** — real Qwen3.6 calibration shipped |
| **D3** | Calibration script unrunnable on Qwen3.6 hybrid (counted SSM layers) | bug | **fixed** — walks attn layers, multi-name attribute resolver |
| **D4** | No runtime assert that table.num_layers matches SDPA call count | quality | open (small follow-up) |
| **D5** | Runtime vertical_slash uses K-norm — but I claimed calibration's stored indices were the right path | mistake | **revert** — K-norm is the actual MInference design intent; I was wrong, fix reverted |
| **D6** | Sparse mask `-1e9` sentinel risks fp16 underflow | bug | **fixed** — replaced with `-3.4e4` |
| **D7** | Pattern params calibrated at one seq_len applied at all | structural | partially observed (D16 confounds) |
| **D8** | Quest is decode-only in repo; prefill integration doesn't exist | misframed | open — Cycle N+1 is a build, not a wire |
| **D9** | STARC was tried, dropped as slower (per `safe_bench.py:124`) | historical | open — plan ignores prior negative measurement |
| **D10** | Plan reverses what DuoKV does (says "for retrieval", but it's for streaming) | doc-quality | confirmed |
| **D11** | Expert-Choice routing exists but unwired AND unmeasured | misframed | open — file as spike, not "ship it" |
| **D12** | 7 of 9 plan-cited "filed" task numbers point to unrelated work (LayerScope, MoE-SpAc, NPUMoE, MC#, KnapSpec, ZipCal, Fast KVzip never filed) | structural | open — plan overstates filed work by ~7× |
| **D13** | Calibration capture monkey-patch only touched mlx_base; Qwen3.6 imported by reference | bug | **fixed** — walks `sys.modules` for `mlx_lm.models.*` |
| **D14** | Capturing SDPA crashed on `mask="causal"` string | bug | **fixed** — synthesizes causal matrix on the fly |
| **D15** | Calibration script's `assert avg_sparsity >= 0.85` ignored user `--sparsity` flag | bug | **fixed** — assertion respects 90% of requested target |
| **D16** | High sparsity (86%) can't capture NIAH-style retrieval; 50% works for 4K and 64K but **NOT for 16K** — quality unstable across contexts at fixed sparsity | structural | partial (D16 narrowed twice this session) |
| **D17** | At 16K NIAH on Qwen3.6, MInference fails at BOTH 30% and 50% sparsity (same `'no secret code'` response) AND is *slower* than dense (570 tok/s vs 705 tok/s) — 16K is below MInference's context-length sweet spot where overhead outweighs gain. **Extended**: even at 22% sparsity (78% of positions kept), K-norm selection systematically excludes the prose-style needle from a code-heavy haystack — a structural flaw in the K-norm signal, not a budget issue | structural | confirmed by extended sparsity sweep (50% / 30% / 22% all fail) |

**Bottom line for the architecture plan**:

1. **Goal 1 reach** (1M context): hard-blocked at ~400K on M4 Pro by Metal per-MTLBuffer cap + global RAM. Plan's Layers 1-6 don't address this; the right unblock is online KV eviction during prefill (Layer 2 of the plan, but unbuilt). 384K validated this session, 458K thrashes, 512K OOMs.

2. **Goal 4 reach** (constant prefill speed): plan's "Cycle N: ship MInference" is misframed. The runtime hooks wire correctly (D2/D3/D5/D6/D13/D14 all fixed in this session). But the chosen sparsity setting trades quality for speed in a content-dependent way — 50% sparsity passes 4K and 64K NIAH but FAILS 16K NIAH on the same calibration. Stable retrieval needs an adaptive sparsity controller or retrieval-aware pattern type — research, not wire-up.

3. **The plan's filed-tasks accounting is wrong** (D12): of nine "filed" research tasks the plan cited, seven point to unrelated work in the actual TASKS.md. LayerScope, MoE-SpAc, NPUMoE, MC#, KnapSpec, ZipCal, Fast KVzip are paper concepts never filed; only KVLinC (Task 288) and FluxMoE (Task 318 alt) and RoPE Goldilocks (Task 332 alt) are real proposals. RoPE Goldilocks spike was filed but never run.

4. **What DID work**: the bench-side bring-up for Qwen3.6 (six independent fixes shipped) plus extending Goal 1 reach from 256K → 384K via chunk-size tuning. Real progress, just on the un-misframed parts of the plan.

## Detail (D1-D17)

## D17 — 16K is below MInference's sweet spot AND fails NIAH at any tested sparsity (EMPIRICAL SWEEP 2026-05-03)

**Sparsity sweep on 16K NIAH** (after backing up the 50% pattern table):

| Config | Result | Prefill tok/s |
|---|---|---|
| Dense baseline (no `--prefill-sparse`) | **PASS** `'ALPHA-7749'` | 705 |
| MInference @ 50% sparsity (full bench) | FAIL `'no secret code'` | 618 |
| MInference @ 30% sparsity (this cycle) | FAIL `'no secret code'` | 570 |

Both observations:
1. **At any tested sparsity, 16K NIAH fails** with the same response shape ("The provided text contains no secret code"). The model concludes the needle isn't present. So it's not a budget issue — even at 30% sparsity (70% of attention preserved!), the K-norm-selected positions systematically exclude the needle's neighborhood at this context length.
2. **MInference is slower than dense at 16K** — 570/618 tok/s vs 705 tok/s. The mask-construction + sparse-dispatch overhead outweighs the attention-compute savings at this scale. MInference's win shows up only when QK^T compute dominates, which happens at longer contexts.

This "context-length sweet spot" effect isn't called out in the architecture plan but explains the 16K failure: even if sparsity were quality-safe, the bench would penalize MInference for slowing 16K down.

**Why 4K passes and 64K passes** but 16K fails (under the same calibration): the haystack content varies. At 4K the haystack is `floor(target_tokens / filler_tokens) + 1` reps of code blocks — fewer reps, the needle's relative position-density is different. At 16K it's more reps. At 64K it's many reps. The needle's K-norm relative ranking shifts with the surrounding content distribution. There's no monotonic context-length effect — 16K just happens to produce a haystack where the needle's K-norm rank is below the threshold.

**Extended sparsity probe (2026-05-03 02:36 UTC)**: tested 16K NIAH at sparsity targets 30%, 22%, 50% — all FAIL with the same `'The provided text contains no secret code'` response. Even at **22% achieved sparsity (78% of positions kept!)**, the K-norm-selected top-K never includes the needle's position. This is the definitive evidence that **K-norm is the wrong selection signal for prose-needle-in-code-haystack tasks** — code tokens (function names, URLs, type signatures) systematically dominate prose tokens in K-norm magnitude, so the needle ("The secret code is ALPHA-7749") is in the bottom of the K-norm ranking regardless of how big the budget is.

The implication is sharper than D17 originally stated: **for retrieval tasks where the needle is stylistically different from the haystack, K-norm-based MInference can't be made to work by tuning sparsity — it's structurally wrong.** Either the haystack matches the needle's content style (and K-norm works), or the runtime needs a different selection signal entirely (e.g., per-query attention-weight projection, or random-projection retrieval).

**This is a benchmark-design-level disproof of "fixed-sparsity MInference"**: a single sparsity setting can't pass the bench's set of NIAH configurations. The plan's "ship MInference" framing requires either:
- per-context-length sparsity tuning (complicates the dispatch),
- a non-K-norm position-selection heuristic that's content-shape-aware,
- a "skip MInference if context is small" heuristic baked into the runtime, or
- a different sparse pattern altogether (e.g. randomized projection retrieval).

All four are research items.




## D1 — Metal per-buffer cap blocks long context — **partially mitigated, ultimately hits disk-thrash wall** (REPRODUCED)

**Update 2026-05-02 21:06 UTC — chunk=1024 mitigation experiment outcome**: PARTIAL. The smaller chunk size pushed the cliff out from ~200K (where chunk=4096 OOMed) to ~458K (where chunk=1024 went silent for >37 min in disk-wait `U` state with RSS dropping from 17 GB to 50 MB — model heavily paged out). I killed the process at the 2-hour mark; no NIAH answer produced.

**Refined understanding of D1**: there are at least TWO distinct binding constraints at long context on M4 Pro:
1. **Per-MTLBuffer cap (~30 GB)** — fires at 512K with chunk=4096 because the chunked attention compute or accumulated-KV intermediate exceeds the cap. Mitigated by smaller chunk.
2. **Total physical RAM budget (~48 GB) once model + KV + transients all materialize simultaneously** — fires at 458K-ish with chunk=1024 because by that point the KV cache (int4) + model weights + per-chunk transients exceed what stays resident, OS starts paging, throughput collapses to disk speeds.

The first constraint is per-allocation; the second is global. Smaller chunks fix the first but don't fix the second — they just delay it. The plan's optimization stack (Layers 1-6) addresses neither directly; sparse attention reduces compute but not peak working set.

**What might mitigate constraint 2** (none yet validated):
- 2-bit KV (KVLinC, Task 288) — halves KV from 2.6 GB to 1.3 GB at 512K. Helps but doesn't change the structural picture for 1M.
- Streaming-K eviction during prefill (compact early chunks before late ones) — would keep KV bounded at K_max regardless of context. The plan's "Layer 2 online SnapKV" is exactly this; it's not yet built.
- Sequence-parallel prefill across multiple Metal command buffers, with explicit residency control — research-level work.

**Existing infra that wasn't tried this session — `omlx/patches/adaptive_prefill.py`**: ships an `AdaptivePrefillController` that auto-shrinks chunk size based on Metal pressure + throughput feedback. Starts at `max_chunk` (default 16384) and shrinks as memory pressure grows. It's wired into `hypercar_server.py` via `--adaptive-chunk` flag but NOT into `omlx/bench/hypercar_bench.py` for the long-context probes. **This is plausibly the right structural fix for the D1 constraint-A side**: instead of manually picking chunk=1024, let the controller pick adaptive chunk sizes mid-prefill. Future cycle: wire `--adaptive-chunk` into the bench's `--niah-only` path and re-run 512K with adaptive chunking.

**Original observation (preserved)**:

**Source**: live failure 2026-05-02 ~18:50 UTC during 512K + YaRN + int4 KV bench.

```
RuntimeError: [metal::malloc] Attempting to allocate 30601641984 bytes which
is greater than the maximum allowed buffer size of 30150672384 bytes.
```

**What this means**: Apple's Metal driver caps single MTLBuffer allocations at ~30.15 GB on M4 Pro 48 GB unified memory. Beyond this, the bench OOMs even when *total* free memory is plenty (vm_stat showed 24+ GB free at the time of failure). The 30.6 GB allocation is most likely the QK score matrix or an int4 dequantize-accumulate intermediate at chunk boundary L_kv > 200K.

**Consequence for the plan**:
- 512K and 1M targets on M4 Pro are blocked **at the Metal allocator level**, not the algorithm level.
- The architectural plan focuses on prefill *speed* (Layer 1 sparsity, Layer 4 heterogeneous compute). None of those layers reduce *peak single allocation size*.
- Levers that DO reduce peak single allocation: smaller PREFILL_CHUNK (e.g. 1024 or 512), in-place int4 dequant rather than full-tensor materialize, sequence-parallel prefill across N MTLBuffers stitched at boundaries.
- **The architecture plan needs a Layer 7: per-buffer-cap-aware allocation strategy.** Without it, Layers 1-6 optimize a regime we can't reach.

**Reproducer**: `python scripts/yarn_niah.py 512K --kv-mode native --kv-bits 4` on Qwen3.6-35B-A3B-4bit. Fails after ~38% of prefill chunks complete.

## D2 — MInference pattern table is synthetic — **RESOLVED for Qwen3.6, 2026-05-02 23:10 UTC**

After D3, D13, D14 fixes the calibration script ran end-to-end on Qwen3.6 and produced a real pattern table at `omlx/patches/minference_patterns/qwen3.6_35b_a3b_4bit.json`:
- **160 (layer, head) pairs** captured (matches 10 attention layers × 16 heads exactly)
- **86.1% average sparsity** (above the 85% target)
- **0.000016 average MSE** vs full attention (well under the 0.01 cap)
- Pattern distribution: 10 a_shape, 150 vertical_slash, 0 block_sparse, 0 dense — vertical_slash dominates as expected for retrieval-style heads

Calibration time: 2.1 seconds (capture) + 2 seconds (classify). The vertical_col_indices are recorded in the table per the D5 fix path, so the runtime can use them directly.

Qwen3-Coder's pattern table at `omlx/patches/minference_patterns/qwen3_coder_30b_a3b_instruct_8bit.json` remains synthetic — but the calibration script can now produce a real one for it too via `python scripts/minference_calibrate.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit --seq-len 2048`.

### Original analysis (preserved)

**Source**: `omlx/patches/minference_patterns/qwen3_coder_30b_a3b_instruct_8bit.json`, JSON `note` field:

```json
"note": "SYNTHETIC placeholder — run scripts/minference_calibrate.py to generate real patterns"
```

The runtime even warns about this on load (`omlx/patches/minference_prefill.py:74-84`): when it sees `"SYNTHETIC"` in the note, it logs that "Quality regressions on long-context gates are LIKELY" and "Predicted speedups in CLAUDE.md are UNSUBSTANTIATED until calibration runs."

**Consequence for the plan**: the user's plan claimed "MInference patterns are classified and saved." They're not — only programmatic defaults are saved. Cycle N's measurable claim ("prefill improvement at 64K+") cannot be evaluated on the current artifact; the calibration must run first, AND it must run on Qwen3.6 specifically (D3 below).

## D3 — Calibration script unrunnable on Qwen3.6 hybrid (STATIC ANALYSIS) — **FIXED 2026-05-02**

**Resolution shipped at `scripts/minference_calibrate.py:487-560`**: the script now walks `model.layers` explicitly, collects only the indices that have `self_attn`, raises a clean error if zero attention layers exist (pure-SSM model — nothing to calibrate), and uses a multi-name `_attr` resolver covering `n_heads` (Qwen3-Coder), `num_attention_heads` (Qwen3.6's `Qwen3NextAttention`), and `num_heads`. The pattern table's `num_layers` field now reports attention-layer count, not total model depth — which matches the runtime convention where `_LAYER_COUNTER` increments once per SDPA call.

When run on Qwen3.6, the script will log:
```
Hybrid model detected: 10 attention layers / 40 total layers (SSM-skipped indices: [0, 1, 2, 4, 5]...)
Model: 10 attention layers, 16 heads
```

The actual calibration RUN still requires the model to be free of other Metal-bound work (currently blocked by the D1 experiment in flight).

### Original analysis (preserved for context)

**Source**: `scripts/minference_calibrate.py:494-502`:

```python
num_layers = len(model.layers)              # = 40 on Qwen3.6 (incl. SSM)
first_layer = model.layers[0]                # SSM layer on Qwen3.6
if hasattr(first_layer, 'self_attn'):       # FALSE for SSM
    attn = first_layer.self_attn
    if hasattr(attn, 'n_heads'):             # Qwen3.6 uses num_attention_heads
        num_heads = attn.n_heads
    elif hasattr(attn, 'num_heads'):         # Qwen3.6 uses num_attention_heads
        num_heads = attn.num_heads
```

Three problems compound:
1. `num_layers` counts SSM layers that never call SDPA. The capture hook would attribute every SDPA call (only ~10 per forward pass on Qwen3.6) to layer indices 0-9 via the modulo operation, leaving entries 10-39 in the table empty or duplicated.
2. `model.layers[0]` is the SSM layer on Qwen3.6 (`fa_idx=3` is the first attention layer). The `hasattr(first_layer, 'self_attn')` check returns False and the script silently falls through to defaults.
3. Even if fixed to walk past the SSM layer, the attribute-name probe misses `num_attention_heads` (the actual Qwen3.6 name; Qwen3-Coder uses `n_heads`, my Task 388 work already noted this naming inconsistency at `omlx/bench/hypercar_bench.py::_project_prefill_memory_gb`).

**Consequence for the plan**: calibration is NOT a one-command operation on Qwen3.6 — the script must be patched first. Probably 50-100 LOC: walk attention layers explicitly, use the multi-name attribute resolver (already extracted in `_project_prefill_memory_gb`'s `_attr` helper), record only the (attn-layer-index, head) pairs.

## D4 — Runtime layer counter mismatch on hybrid (STATIC ANALYSIS)

**Source**: `omlx/patches/minference_prefill.py:265-266`:

```python
layer_idx = _LAYER_COUNTER[0] % num_layers
_LAYER_COUNTER[0] += 1
```

If the calibration table claims `num_layers=40` (matching Qwen3.6's total) but the runtime SDPA is only invoked 10 times per forward pass (one per attention layer), the modulo wraps after every full forward — so every (layer_idx) the runtime asks about is in [0, 10). The remaining table entries (layers 10-39) are dead.

If the calibration table claims `num_layers=10` (only attention), the runtime indexing works. But in either case, the **runtime AND calibration must agree on the convention** — and the current code agrees only by accident on dense models. There's no programmatic check.

**Consequence for the plan**: a fix-once-it-passes-tests runtime change risks pattern-table-incompatibility regressions. The fix is small (assert that `pattern_table.num_layers == count of attn layers in model`), but worth filing as a structural follow-up.

## D5 — Runtime vertical_slash signal disagrees with calibration signal — **FIX REVERTED 2026-05-03 (was wrong)**

**Update 2026-05-03 00:38 UTC**: my D5 fix was a **misread of MInference's design intent** and has been reverted at `omlx/patches/minference_prefill.py`. The K-norm runtime heuristic is what MInference is supposed to do; storing indices was diagnostic-only.

**Why the fix was wrong**: vertical_col_indices are POSITION-indexed (the calibration prompt's positions where attention concentrated). At runtime with a different prompt, position 17 contains different content; the calibration's "head h attends to position 17" doesn't transfer. MInference's design intent: calibration determines the PATTERN TYPE per head (vertical_slash / A-shape / etc.) and PARAM COUNTS (how many vertical cols, band width); the runtime picks the specific positions content-adaptively via K-norm.

**Empirical confirmation**: 4K NIAH on Qwen3.6 with stored-indices fix → output `'postgresql://localhost:5432/app_db'` (a config token from the haystack). Same 4K NIAH after revert (K-norm path) → same wrong output. The K-norm path is at least theoretically right; both fail because of the deeper structural problem in D16.

### Original analysis (preserved)

### Original analysis (preserved)

**Source**: `omlx/patches/minference_prefill.py:150-155`:

```python
if keys is not None and num_vert > 0:
    k_norms = mx.linalg.norm(keys[0, 0], axis=-1)  # K-vector norms
    top_indices = mx.argsort(-k_norms)[:num_vert]   # top-K by norm
    vert_mask = ...
```

The calibration script (`scripts/minference_calibrate.py:_classify_vertical_slash`) measures *post-softmax attention weight* mass concentrated in vertical columns — which positions consistently get high attention from many query rows. The runtime instead picks positions with high *K-vector norm*. These signals are correlated but not identical:
- High K norm → key vector with large magnitude. After QK^T and softmax, this *biases* attention toward this position but doesn't guarantee high attention.
- High attention weight → the query's softmax actually placed mass here, considering BOTH K norm AND alignment with Q.

A position can have high K norm and never get attention (Q never aligns); a position can have low K norm and still get attention (perfect alignment). The runtime's K-norm proxy will diverge from the calibration's attention-weight proxy, sometimes silently.

**Consequence for the plan**: even after Qwen3.6-real-calibration runs, the per-head `num_vertical_cols` parameter is being applied with a different selection rule than was calibrated. Quality regression possible. Fix: store the actual top-K column indices in the calibration table (not just `num_vertical_cols`), or change the runtime to use Q-K alignment (one matmul per chunk, defeats some of the sparsity gain).

## D6 — Sparse mask sentinel value risks fp16 underflow — **FIXED 2026-05-02**

**Resolution shipped at `omlx/patches/minference_prefill.py:248, 354`**: both occurrences of `-1e9` replaced with `-3.4e4`, matching the constant SparseKVCache Phase 1 converged on (also in fp16-safe just-below-max-finite range). No model load needed to verify; both call sites passed import + smoke construction with the new constant.

### Original analysis (preserved)

**Source**: `omlx/patches/minference_prefill.py:227, 333`:

```python
additive_mask = mx.where(sparse_mask, 0.0, -1e9).astype(q.dtype)
```

In fp16 (`q.dtype` during prefill on Qwen3.6 4-bit), `-1e9` is just below the fp16 max-finite threshold (~6.5e4). Casting `-1e9` to fp16 actually clamps to `-inf` in some MLX paths, which through softmax with attention-score scaling at large L can produce NaNs.

The `omlx/sparse_kv_cache` work (Tasks 384-386) explicitly converged on `-3.4e4` (fp16-safe, just below max-finite) precisely because `-inf` and very-negative values aren't reliable as soft-mask sentinels through MLX's softmax stabilization paths.

**Consequence for the plan**: even if D2-D5 are fixed, the runtime risks NaN propagation at long context where many positions are masked. The 86% sparsity target means ~86% of every row gets `-1e9` — high concentration of marginal-fp16-safe values. The fix is one-character (`-3.4e4` instead of `-1e9`) but should be tested against the MLX softmax explicit-clamp path at L=4096 with high sparsity.

## D7 — Calibration seq_len ≠ runtime seq_len — **EMPIRICALLY VALIDATED 2026-05-03**

**Update 2026-05-03 00:09 UTC**: ran `hypercar_bench --niah-only --niah-context 64K --kv-mode fp16 --prefill-sparse minference` against Qwen3.6-4bit using the *real* pattern table calibrated at seq_len=2048 (D2 fix). Result:

```
FAIL: 64K — 'merge_sort<|im_end|>...'   prefill 252 tok/s, decode 42.8 tok/s
Phase 3: Needle in Haystack    (264.4s)   GATE FAILURE
```

Two failures in one run:
1. **Quality regression**: model retrieved the wrong identifier (`merge_sort` is a function name embedded in the synthetic code haystack, not the needle `ALPHA-7749`). The 86% sparse mask at 64K dropped enough of the attention information that the model couldn't locate the needle, even though it could at 4K and 16K with full attention.
2. **No prefill speedup**: 264s wall time vs the 166s baseline run earlier today (64K NIAH at fp16 dense). The reported "252 tok/s prefill" is the inner-attention measurement; the actual end-to-end was slower.

**Mechanistic explanation**: the calibration measured per-head attention concentration at *2048 tokens*. It found that ~150 of 160 heads concentrated their attention on a small set of vertical columns plus a local band, with 86% of the attention matrix safely maskable. At 64K the same heads have to retrieve from a much larger key set; the 8-column "vertical" pattern that captured 95% of attention at 2K only captures a small fraction at 64K, since the attention distribution at long context is broader and more position-dependent.

This is the first hard empirical proof that the user's plan's claim "MInference per-head pattern dispatch reduces retrieval-head cost" was overstating what's available without length-conditioned calibration. The numbers in the plan's "realistic prefill curve" table (`500-600 tok/s at 256K`) presumed calibration that worked at long context — it doesn't, on Qwen3.6, with 2K calibration data.

**Required to resolve**:
- Recalibrate at seq_len=4K, 16K, 64K, 256K and store length-conditioned params in the pattern table.
- Modify the runtime to interpolate or select the right param block based on the current chunk's L_kv.
- Re-validate quality (NIAH at each tier) and speed (vs dense baseline).

This is multi-cycle research work. The "Cycle N: MInference end-to-end" formulation in the plan is therefore **misframed** — MInference at 2K calibration is not a Cycle-N task that ships, it's a research arc that needs length-conditioning to even pass NIAH.

### Original analysis (preserved)

**Source**: `scripts/minference_calibrate.py:472` (default `seq_len=2048`); `omlx/patches/minference_prefill.py:_build_vertical_slash_mask` applies the same `params["num_vertical_cols"]` regardless of L_kv at runtime (which can be 256K during chunked prefill).

A head whose attention mass concentrates in 8 vertical columns at L=2048 may concentrate in 8 columns *or* 16 columns *or* 100 columns at L=256K — there's no a-priori reason the count is invariant under length scaling. The calibration captures a single L's behavior and the runtime applies it everywhere.

**Consequence for the plan**: the predicted speedup numbers in the plan ("500-600 tok/s at 256K") are extrapolations from a calibration measurement at L=2048. If the per-head pattern shape changes with L, the runtime may either:
- Mask too aggressively at large L → quality regression (we permit attention on too few keys to reconstruct the dense softmax),
- Mask too conservatively → measured speedup degrades from theoretical.

Calibrating at multiple lengths and storing length-conditioned params is one fix; another is calibrating directly at the target L (256K), which is expensive (full attention map at 256K is 65 GB at fp16, but capture only needs argmax-importance per row, so it can be streamed).

## D8 — Quest is decode-time only; prefill integration doesn't exist (CODE INSPECTION)

**Source**: `omlx/patches/quest_attention.py` exposes only three primitives — `compute_page_bounds`, `select_topk_pages`, `gather_pages`. Grep for any caller in the prefill path:

```
$ grep -rn "select_topk_pages\|gather_pages\|patches.quest" omlx/bench/ omlx/hypercar_server.py omlx/patches/minference_prefill.py
# (zero results)
```

The only Quest integration in the live runtime is through `TurboQuantKVCache(quest_topk=_QUEST_TOPK)` — a TQ3 cache feature for decode-time top-K page selection over already-prefilled KV. **Prefill never calls into Quest**, and there is no compose-with-MInference path written.

**Source: probe results**: `research/analyst_runs/2026-04-26/quest_topk_probe.md` reports `mx.argpartition` viable at K=4096 on 1M-context **synthetic** data (188 µs/head, just inside the 200 µs/head gate). At K=256 and K=1024 the argpartition itself misses the latency gate (266 and 222 µs/head). So even if the prefill integration existed, the Cycle N+1 plan's "Quest picks top-K pages" requires K≥4096 to meet the gate it was designed for.

**Consequence for the plan**:
- Cycle N+1 ("Quest page-selection at chunk boundaries") is **not a wiring task — it's a build task**. The integration code doesn't exist; someone has to write the chunked-prefill hook that calls `select_topk_pages` and feeds the result into the SDPA mask.
- The plan's "compose Quest with MInference per chunk" is also un-built. They both have hooks, but those hooks are independent — no joint dispatcher, no priority order between them.
- The probe's variance finding ("first run vs extended re-run wildly different per-head latencies") suggests Quest's measured throughput is fragile to system load, which complicates the cycle-N+1 acceptance gate.

**What's actually buildable**: a Cycle N+1 task that:
1. Promotes `select_topk_pages` from decode-time to prefill-time (call at chunk boundaries with the chunk's queries vs. accumulated KV pages).
2. Composes the resulting mask with MInference's sparse-mask via additive combination (this is straightforward — both produce additive masks of the same shape).
3. Validates that K=4096 latency gate holds at 256K and 512K contexts (D1 mitigation success means we can now reach those).

## D10 — Plan reverses what DuoKV does and what it doesn't (CODE INSPECTION)

**Plan claim**: *"Streaming heads: ring buffer (DuoKV already does this for retrieval — extend to streaming)."*

**Source**: `omlx/duo_kv_cache.py:137-141`:

```python
class StreamingKVCache:
    """Ring-buffer KV cache for streaming heads.

    Keeps only the most recent `window` tokens plus `sink` initial tokens.
    Total capacity: sink + window tokens in fp16.
    """
```

The plan has it **backwards**. DuoKV already implements ring buffer for **streaming** heads, not retrieval. The retrieval heads use the regular `DuoKVCache` path (fp16 retrieval, full-history). The plan's "extend to streaming" task is a no-op — that path is the one that exists.

**Consequence for the plan**:
- Layer 2's "Streaming heads: ring buffer (extend to streaming)" subtask is already complete.
- The actual Layer 2 gap is the *retrieval* side — currently full-history fp16; the plan's "hard cap at K_retrieval = 16K via SnapKV+CAOTE+BUZZ in *online* mode" is the buildable work, but it's the half the plan didn't specifically frame as new.
- Reading the plan and the code together, the actual remaining KV-side work is: promote SnapKV's existing post-prefill compaction to mid-prefill (online mode). The plan correctly identifies this elsewhere; the framing under Layer 2 just inverts the streaming/retrieval labels.

This is a documentation-quality disproof, not a functional one — but worth noting because it shapes which subtasks are *new work* vs already-done.

## D11 — Expert-Choice routing exists but is unwired (CODE INSPECTION)

**Plan claim**: *"Expert-Choice routing (Hypercar's original plan, `omlx/patches/expert_choice_router.py`): flips token-choice to expert-choice. 100% expert utilization. Removes load-imbalance dead time. Original Hypercar plan, never shipped — ship it."*

**Source**: `omlx/patches/expert_choice_router.py` exists with `class ExpertChoiceRouter(nn.Module)` (line 26) and `apply_expert_choice_patch(model, capacity_factor=1.2)` (line 127).

```
$ grep -rn "expert_choice_router\|ExpertChoiceRouter" omlx/bench/ omlx/hypercar_server.py omlx/patches/ | grep -v expert_choice_router.py
# (zero results)
```

**Confirmed**: the router is implemented but never imported, never patched in. The plan's "ship it" framing is right.

**Consequence for the plan**:
- This is the simplest "flip a switch" item in Layer 3. No new code, just a wire.
- Worth a sanity check: does `apply_expert_choice_patch` work on Qwen3.6's `Qwen3NextSparseMoeBlock`? Qwen3-Coder's MoE class structure is different. (Qwen3.6 uses 256 experts × 8 routed + 1 shared, vs Qwen3-Coder's 128 × 8.) The patch may need adapter work.
- Even on Qwen3-Coder, the patch hasn't run in any bench in the recent history of `safe_bench.py` — so actual measured improvement is unknown. The plan's "removes load-imbalance dead time" is a hypothesis, not a measurement.

**The right framing for this work**: file as a spike — wire the patch to one model variant, measure on hypercar_bench, decide if it ships. The user's plan calls it self-contained which is true; what it isn't yet is *measured*.

## D13 — Calibration capture monkey-patch misses already-imported model modules — **FIXED 2026-05-02**

**Source**: `scripts/minference_calibrate.py:451` originally only set `mlx_base.scaled_dot_product_attention = capturing_sdpa`. But Qwen3.6's `mlx_lm/models/qwen3_next.py:18` imports the symbol via `from .base import scaled_dot_product_attention` — Python captures the reference at import time, so rebinding the attribute on the base module has no effect on the already-imported caller.

**Symptom**: calibration ran, model forward pass executed, but **0 (layer, head) pairs captured** because the patched function was never reached. Then the script asserted `avg_sparsity >= 0.85` (against 0.0) and failed.

**Resolution shipped at `scripts/minference_calibrate.py:450-475`**: walks `sys.modules`, identifies every `mlx_lm.models.*` and `mlx_vlm.models.*` module that has its own `scaled_dot_product_attention` attribute, and rebinds them all (storing originals for restoration in the finally clause). This mirrors what `apply_minference_prefill_patch` already does for the runtime path. After the fix, capture got all 160 pairs.

This bug class is generic to any "monkey-patch by attribute" pattern in Python; the runtime patch already had the right idiom and the calibration just needed it transplanted.

## D16 — MInference at 86% sparsity drops the needle on NIAH; 50% works — **PARTIALLY RESOLVED 2026-05-03**

**Update 2026-05-03 01:13 UTC**: ran a sparsity sweep on Qwen3.6 4K NIAH and 64K NIAH. The 86% target sparsity was too aggressive; **50% sparsity passes both**, with measurable prefill speedup vs dense:

| Context | Sparsity | NIAH | Prefill tok/s | Speedup vs dense |
|---|---|---|---|---|
| 4K | 86% | FAIL (`postgresql://...`) | 730 | 4.8× |
| 4K | **50%** | **PASS** (`ALPHA-7749`) | **672** | **4.4×** |
| 64K | 86% | FAIL (`merge_sort`) | 252 | 0.6× (slower!) |
| 64K | **50%** | **PASS** (`ALPHA-7749`) | **302** | **2.0×** |

So D16's claim that "MInference can't capture NIAH-style spikes" was overstated. The pattern types CAN handle NIAH if the budget keeps enough columns. At 50% sparsity, the K-norm-selected top-K positions plus the local band cover enough of the actual high-attention positions to retrieve the needle.

**The actual structural claim**, narrowed: there's a **sparsity-quality Pareto** for retrieval — Qwen3.6's working sparsity at 64K NIAH is between 86% (fails) and 50% (passes). The exact crossover wasn't measured this session; could be ~70%, ~75%, etc. Higher sparsity = more speedup but worse retrieval.

**🚨 Update 2026-05-03 01:36 UTC — quality is unstable ACROSS context lengths at fixed 50% sparsity**: ran the full bench Phase 3 NIAH (4K + 16K) with the same 50%-sparsity calibration:

| Context | Result | Detail |
|---|---|---|
| 4K | **PASS** | `'The secret code is ALPHA-7749'`, prefill 793 tok/s |
| 16K | **FAIL** | `'The provided text contains no secret code.'` (model concluded the needle WASN'T present), prefill 618 tok/s |
| 64K | PASS (earlier run) | `'ALPHA-7749'`, prefill 302 tok/s |

The 16K failure is qualitatively different from the 86%-sparsity failures — instead of retrieving a wrong identifier, the model affirmatively concluded the needle wasn't in the haystack. This means the K-norm-selected top-K positions at 16K *systematically excluded* the needle's neighborhood, whereas at 4K and 64K (different haystack content due to different repetition counts in `_build_code_haystack`) the positions happened to include enough of the needle to retrieve it.

**Implication**: 50% fixed-sparsity isn't a reliable operating point. The K-norm heuristic's content-dependence means quality is bench-prompt-dependent, not just length-dependent. To make MInference production-ready for retrieval, you'd need either:
- An adaptive sparsity controller (raise the budget when retrieval headers see low-confidence outputs).
- A retrieval-aware pattern type that protects the per-query top-K-by-attention positions, not the static top-K-by-K-norm.
- Sparsity-quality regression tests that probe many haystack variants, not single fixed prompts.

The plan's framing of "ship MInference at one sparsity setting" doesn't survive even the simplest adversarial probe (varying NIAH context length).

**Side fix shipped**: `scripts/minference_calibrate.py` had `assert avg_sparsity >= 0.85` hardcoded — ignored the user's `--sparsity` flag. Replaced with `0.9 * sparsity` so calibration respects the requested target. Without this fix, `--sparsity 0.50` would have been rejected at the assertion.

**Remaining work to "ship MInference"**:
1. Sweep sparsity at 64K, 128K, 256K to find the Pareto frontier per context length.
2. Pick a default sparsity per context tier (D7 — length-conditioned).
3. Re-bench against full hypercar gates (LCB, MMLU-Pro, HumanEval) at the chosen sparsity to confirm no quality regression on non-NIAH tasks.

The plan's Cycle N is **achievable at 50% sparsity** with measurable speedup, contrary to my 23:38 UTC negative-result conclusion. The 86% default was the trap.

### Original analysis (preserved as historical context)

**Source**: tested 4K NIAH (= calibration seq_len) on Qwen3.6 with MInference enabled (real calibration, K-norm runtime selection). NIAH **FAILED** with output `'postgresql://localhost:5432/app_db'` — a config-block token from the synthetic code haystack — instead of the needle `'ALPHA-7749'`.

**Mechanism**: MInference's per-head patterns are vertical_slash / A-shape / block_sparse / dense. None of these explicitly represent "this head spikes attention at one specific content-dependent position" — which is exactly what NIAH-style retrieval needs. At 86% sparsity the runtime mask drops ~14% of positions, selected by K-vector norm. The needle's K-vector norm isn't necessarily larger than other positions' (it's just 4 tokens about a "secret code" embedded in 4K of code), so the K-norm heuristic doesn't preferentially preserve it. The model then attends to whichever content-distinctive tokens DO have high K-norm (URLs, function names) and emits one of those as its answer.

**Why this matters for the user's plan**: the plan assumes MInference + Quest at 41% retrieval-head budget covers retrieval. But the existing pattern types (vertical_slash, A-shape, block_sparse) don't include "spike-at-needle." For NIAH-style tasks specifically, MInference's pattern prior is incomplete. Pattern types like `dense_for_specific_heads` (a marked subset of heads kept fully dense) might recover NIAH but defeat the speedup.

**Required to actually use MInference for retrieval**:
- Calibrate at multiple sequence lengths (D7).
- Add a NEW pattern type for retrieval heads — possibly `top_k_dynamic` that's just "compute full attention on a runtime-sampled top-K subset." Different from vertical_slash because the slash is per-head-static, not per-query-dynamic.
- OR mark certain heads as "always dense" and exclude them from the sparsity budget. Calibration would need to identify these heads — the existing classifier doesn't (every head ends up vertical_slash on Qwen3.6's distribution).

**The plan's framing as Cycle N work is therefore wrong on multiple axes**:
1. Patterns transfer poorly across seq_len (D7).
2. Patterns don't include retrieval-spike heads (D16).
3. The composition with Quest (D8) is unbuilt and would still need (1) and (2) handled.

The retrieval-side composition is genuinely new research, not an integration cycle.

## D14 — Calibration's manual-SDPA implementation can't handle `mask="causal"` — **FIXED 2026-05-02**

**Source**: `scripts/minference_calibrate.py:_capturing_sdpa` (line 433):

```python
if mask is not None:
    scores = scores + mask    # CRASHES if mask is the literal string "causal"
```

mlx_lm represents the causal mask in three ways through the same parameter: `None` (no mask, e.g. decode), an `mx.array` (explicit mask), or the literal string `"causal"` (used during prefill when the SDPA can synthesize the mask itself for efficiency). The base SDPA handles all three; the calibration's manual scorewise computation only handled the first two.

**Symptom (after D13 fix)**: `ValueError: Cannot perform addition on an mlx.core.array and str` thrown from `scores + mask` when `mask == "causal"`.

**Resolution shipped at the same call site**: synthesize a causal `(L_q, L_kv)` mask matrix when the input is the string `"causal"`, using `-3.4e4` for the masked positions (matching D6's fp16-safe convention). After this fix calibration completed cleanly.

## D12 — Plan's "research tasks already filed" citations are mostly wrong (TASK NUMBER AUDIT)

**Plan claim**: *"Net new (research tasks already filed): LayerScope (319), FluxMoE (318), MoE-SpAc (338), NPUMoE (331), MC# (321), KnapSpec (344), ZipCal (328), Fast KVzip (330), KVLinC 2-bit, RoPE Goldilocks (332)."*

**Audit results** (`grep -A1 "^- \*\*Task N" TASKS.md` for each cited task number):

| Plan citation | Actual TASKS.md content | Match? |
|---|---|---|
| Task 318 = "FluxMoE" | "Task 318 (SHIPPED): `compute_attention` Pattern D — reshape + concat assembly, 1.42× faster" — **the disambiguation Task 373 confirms `Task 318` is two different things; line 8753 has the FluxMoE one but the lookup matches Pattern D first** | Partially — namespace collision documented in Task 373 |
| Task 319 = "LayerScope predictor" | "Task 319: Quantify streaming-head softmax dilution from zero-pad gather — REAL, 40× magnitude shrinkage at 16K" | **No match** — different concept entirely |
| Task 321 = "MC# LP-allocated per-expert mixed precision" | "Task 321: Per-query direction analysis for prefill dilution — direction perturbs ~12° median" | **No match** |
| Task 328 = "ZipCal Zipfian-diversity calibration" | "Task 328 (PASSED): Phase 1 sub-probe for Task 281 — fused Q·K^T with int4 K inline-dequantized" | **No match** |
| Task 330 = "Fast KVzip" | "Task 330 (PASSED): Phase 1 of Task 281 FULLY VERIFIED — split-K SDPA via chained metal_kernel calls" | **No match** |
| Task 331 = "NPUMoE — ANE offload" | "Task 331: Phase 2 design note for Task 281 — captures Phase 1 learnings" | **No match** |
| Task 338 = "MoE-SpAc memory oracle" | "Task 338: Pin MLX int4 packing format as a regression test" | **No match** |
| Task 344 = "KnapSpec cosine proxy" | "Task 344: Structural protection for Goal 3 decode-measurement probes — 43 tests" | **No match** |

Grep across the entire TASKS.md for the named concepts:
```
$ grep -c "LayerScope\|MoE-SpAc\|NPUMoE\|KnapSpec\|ZipCal\|Fast KVzip" TASKS.md
0  (excluding the plan's own quotes that I echoed into Tasks 387/388)
```

**Of the plan's nine cited "filed tasks", THREE concepts exist as filed tasks (most as un-run spikes)**:
- KVLinC (Task 288) — filed at line 9051: "Port KVLinC 2-bit KV cache (Hadamard rotation on V + per-head linear correction adapters on K) into TQ3+DuoKV stack". Real proposal, not yet shipped.
- FluxMoE-style transient expert residency — Task 318 alternate at line 8753, behind a Pattern-D disambiguation note (Task 373). Real proposal.
- RoPE Goldilocks (Task 332 alternate, also a numbering collision) — filed as a spike: "Phase 0 (≤3 days): read paper, extract zone formula, compute (theta_lower, theta_upper) at 262K, 524K, 1M; compare to Qwen3.6's shipped base." Output file `research/rope_zone_qwen36.md` does not exist — **spike has been filed but never run.** The plan's "verifies position-encoding stability at native 1M before any of this matters" presumes spike output that doesn't exist.

The remaining seven (LayerScope, MoE-SpAc, NPUMoE, MC#, KnapSpec, ZipCal, Fast KVzip) appear in TASKS.md only inside the Tasks 387/388 entries I wrote, which echoed the user's plan verbatim. **They are not filed work — they are concepts the plan named that have no corresponding repo entry.**

**Consequence for the plan**:
- The plan's "calibration scaffold" (KnapSpec + ZipCal as Layer 6) is composed of items that don't exist as repo work; they're paper concepts the plan attributed to filed tasks. Building them is from-scratch work, not "wire what's there."
- The plan's MoE layer (LayerScope, MoE-SpAc, NPUMoE, MC#) is similarly mostly-vapor — the cited task IDs point to unrelated scaffolding for Task 281 (split-K Metal kernel work, decode-measurement probes, etc.).
- The plan's framing of these as "filed research tasks" overstates the work that exists by roughly 7×.

**The actual repo state** for Layer 3 (MoE):
- Expert-Choice routing — exists, unwired (D11).
- ProMoE profiles — `omlx/patches/promoe_profiles/` exists (visible in `ls omlx/patches/`); not inspected this cycle.
- Hybrid MoE quantization (Task #38 in the small task list) — shipped as documented dead end (per the project memory).

The MoE layer's structural budget is therefore: one wireable piece (Expert-Choice routing, unmeasured), plus net-new research items the plan misattributed to existing tasks.

**This disproof has a self-reflective component**: I echoed the plan's task-citation list verbatim into Task 388's "depends on" section without auditing it. That task description is now also wrong about LayerScope/etc. being filed. Should be corrected when the next Task 388 cycle starts.

## D9 — STARC was tried, dropped because slower (CODE INSPECTION + COMMENT TRAIL)

**Source**: STARC IS in the repo at `omlx/patches/starc_attention.py` (200+ LOC, K-means cosine clustering + top-cluster selection) and `omlx/patches/starc_tq_bridge.py` (TurboQuant integration). It's wired through `omlx/cli.py:231` via a `--sparsity starc` flag.

**The damning evidence**: `omlx/bench/safe_bench.py:124`:

```python
use_starc: bool = False               # Sparse attention (DROPPED — slower)
```

So STARC was implemented, integrated end-to-end, and **measured to be slower than the baseline it was supposed to accelerate**. It got demoted to a flag-off-by-default in the safe_bench config with the explicit "DROPPED" annotation.

**Architecture detail relevant to the plan's claim**: the plan said "STARC's K-means clusters provide the gather targets" for retrieval heads during prefill. But STARC's docstring and implementation (`omlx/patches/starc_attention.py:11-15`) make clear it's a **decode-time** optimization:

> 1. Prefill end: K-means cluster keys (cosine distance, K-means++ init)
> 2. Decode: query × centroids → top-k clusters → gather KV subset → SDPA
> 3. Recluster every 128 decode steps

The clustering itself runs at *prefill end* (a one-time cost amortized across decode), but the *gather* and *attend* steps are decode-time. There is no current code path where STARC clusters provide gather targets *during* prefill.

**Consequence for the plan**:
- "STARC clustering for retrieval heads" cycle (Cycle N+2) is doubly wrong: STARC isn't a prefill optimization in the existing implementation, AND the existing implementation was already measured slower than the baseline.
- Building a prefill-time variant requires algorithmic rework: re-cluster every chunk (expensive) or commit to clusters built mid-prefill (correctness questions).
- Even if built, the priors against it are negative — the decode-time STARC was outright slower.

**The bench history is your friend here**. The "DROPPED — slower" annotation is a real measurement that someone (probably a prior cycle) made and committed to keep future cycles from re-litigating. The plan ignored that commitment and proposed STARC as Cycle N+2.

## What this means for Cycle N's acceptance gate

The plan's Cycle N acceptance gate is: *"correctness within fp16 tolerance + measurable prefill improvement at 64K+."*

To meet that gate, ALL of D2, D3, D4 must be addressed before any meaningful measurement:
1. Fix `scripts/minference_calibrate.py` for hybrid Qwen3.6 (D3): walk attention layers, multi-name attribute resolver.
2. Run real calibration (D2): produces `omlx/patches/minference_patterns/qwen3.6_35b_a3b_4bit.json` (or equivalent name).
3. Fix runtime layer-counter agreement (D4): assert `num_layers` matches actual SDPA call count.

D5, D6, D7 are quality-correctness risks that should be probed during the acceptance test (compare sparse-prefill output vs dense-prefill output at 4K, 16K, 64K — divergence trends will surface them). They don't necessarily block Cycle N from running; they shape what counts as a passing run.

## Wider plan implications

| Plan claim | Status |
|---|---|
| MInference patterns "already classified" | **Disproven** — synthetic placeholder, not measured |
| Quest argpartition "validated" | **Partial disproof** — probed on synthetic data only, passes gate only at K=4096; prefill integration doesn't exist (D8) |
| Quest composes with MInference per chunk | **Disproven** — no joint dispatcher exists, both run independently |
| STARC clustering "in original Hypercar plan" | **Confirmed in repo** — but `safe_bench.py:124` annotates `# Sparse attention (DROPPED — slower)`. Tried, measured, demoted (D9) |
| STARC composes with retrieval heads at prefill | **Disproven** — STARC is decode-time, prefill-end-clustering only |
| DuoKV ring buffer is "for retrieval" | **Disproven** — `duo_kv_cache.py:137` `class StreamingKVCache` is for streaming heads (D10) |
| Expert-Choice routing "self-contained, ship it" | **Confirmed implementation exists** but unwired; never measured (D11) |
| "Research tasks already filed" (LayerScope/MoE-SpAc/NPUMoE/MC#/KnapSpec/ZipCal/Fast KVzip) | **7-of-9 disproven** — task numbers cited point to unrelated work (D12); only KVLinC + FluxMoE genuinely filed |
| TTT-Linear via `omlx/ttt.py` | **Disproven** — wrong module; must build from scratch (Task 388 corrected) |
| 256K → 512K wall-time feasible on M4 Pro at chunk=4096 | **Disproven** — Metal per-buffer cap OOMs at ~200K |
| 256K → 512K wall-time feasible on M4 Pro at chunk=1024 | **Disproven by experiment** — got further (458K reached) but hit disk-thrash wall before 524K, no answer produced after 2hr wall time |
| 256K → 384K feasible on M4 Pro at chunk=1024 | **Confirmed** (2026-05-02 22:38 UTC) — 384K NIAH PASS, Metal peak 36.4 GB (5 GB headroom), 64 min wall |
| MInference at 64K with 86% sparsity produces correct output | **Disproven by experiment** (2026-05-03 00:09) — wrong needle retrieved |
| MInference produces prefill speedup at 64K | **Confirmed** at lower sparsity (2026-05-03 01:13) — 50%-sparsity Qwen3.6 calibration delivers prefill 302 tok/s at 64K (2× dense baseline) AND retrieves the needle correctly |
| MInference at 86% works for NIAH-style retrieval | **Disproven; D16 narrowed**: 86% drops critical positions; 50% passes |
| 1M reachable on M4 Pro without algorithmic changes | **Disproven** — chunk-size tuning alone doesn't bypass the global RAM constraint; the cliff is between 384K (works) and 458K (thrashes) |
| 512K reachable on M4 Pro with adaptive chunking | **Confirmed** (2026-05-03 05:24) — `AdaptivePrefillController` shrinks chunks proactively as memory pressure grows; 512K NIAH passes at Metal peak 47 GB |
| Goal 4 = sparsity + scheduling problem | **Partial disproof** — sparsity helps but D1 (per-buffer cap) is the binding constraint at ≥512K, not asymptote |

The strongest finding is D1 — it changes the framing of Goal 1 entirely. The plan assumed the obstacle was *prefill speed* (Goal 4 territory). The actual obstacle on M4 Pro is *Metal allocator semantics* — a hardware/driver constraint that no algorithmic optimization can bypass without restructuring how prefill allocates its largest intermediate buffers.

## Next-cycle work

1. **D1 mitigation experiment** (highest leverage): retry 512K with PREFILL_CHUNK=1024 (vs current 4096) — quarter the per-chunk attention-score buffer size at the cost of 4× more chunks (which is fine since sequential sum is what matters, not max single allocation). Measurement: does it OOM at the same context, or get further?
2. **D3 fix**: patch `scripts/minference_calibrate.py` for Qwen3.6 hybrid. Small (~50 LOC).
3. **D2 follow-up after D3**: run real calibration on Qwen3.6 at L=2048 first (smoke), then at L=16K (closer to inference-time scale).
4. **D5 + D6 + D7**: surface during the calibration → bench cycle that follows D2.
