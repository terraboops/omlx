# Qwen3.6 long-context NIAH ladder — measured ceiling on M4 Pro 48 GB

_2026-05-02. Single-shot prefill+retrieve experiments in support of Goal 1 (1M context). All runs use `mlx-community/Qwen3.6-35B-A3B-4bit` + `--kv-mode native --kv-bits 4` + YaRN factor=2 (`scripts/yarn_niah.py`). Hardware: M4 Pro, 48 GB unified memory._

## TL;DR

The single-shot ceiling on this hardware is **between 384K and 458K** — roughly **~400K**. Going past it requires algorithmic change (online KV eviction during prefill, or sequence-parallel allocation). Chunk-size tuning got us from ~200K to ~400K but has hit its limit.

| Context | Result | Metal peak | Wall time | Notes |
|---|---|---|---|---|
| 4K | ✅ smoke | 24.8 GB | 5s | Within model+overhead |
| 16K | ✅ | small | small | (full bench Phase 3 — earlier session) |
| 64K | ✅ | small | small | (full bench Phase 3) |
| **256K** | **✅** | **26.3 GB** | **28 min** | 15 GB headroom; comfortable |
| **384K** | **✅** | **36.4 GB** | **64 min** | 5 GB headroom; tight, prefill 102 tok/s avg |
| 458K (in 512K run) | ⚠️ reached but stuck | (paged out) | killed at 2 hr | Disk-thrash regime |
| **512K (chunk=4096)** | **❌** | OOM at ~200K | n/a | Per-MTLBuffer cap (30.6 GB > 30.15 GB) |
| **512K (chunk=1024)** | **❌** | reaches 458K, stalls | killed at 2 hr | Disk-thrash, no answer |

## Wall-time scaling (chunk=1024, int4 KV)

Quadratic-attention dominates:

| ctx | wall time | tok/s avg | implied prefill cost |
|---|---|---|---|
| 256K | 28 min | ~155 | (1×) |
| 384K | 64 min | ~102 | (~2.3× of 256K — matches N²: (384/256)² = 2.25×) |
| 512K projected (if it ran) | ~120 min | ~70 | (4× of 256K) |

The projection vs reality at 512K matches expected N² scaling, but the projection ignores the per-MTLBuffer and global-RAM constraints.

## Memory peak scaling

Empirically observed Metal peak vs context (chunk=1024, native int4 KV):

| ctx | Metal peak | Δ from 256K |
|---|---|---|
| 256K | 26.3 GB | 0 |
| 384K | 36.4 GB | +10.1 GB |
| 512K | (OOM before completion) | — |

The 10 GB jump from 256K→384K is consistent with:
- KV (int4): 256K=2.6 GB → 384K=3.9 GB (+1.3 GB)
- Per-chunk attention transient (chunk × accumulated_kv × n_heads × fp16 bytes): 256K=8.4 GB → 384K=12.6 GB (+4.2 GB)
- Other transients (residuals, MoE router, normalization): +4-5 GB

Linear extrapolation suggests 512K Metal peak ≈ 36.4 + 10 ≈ 46.4 GB on disk-thrash-free path — which would breach the 41 GB Metal-peak cap and the 48 GB physical RAM ceiling almost simultaneously.

## Two binding constraints (refined understanding)

### Constraint A: Per-MTLBuffer cap (~30 GB on M4 Pro)
Single allocation > 30.15 GB → `RuntimeError: [metal::malloc]`. Tuneable by reducing PREFILL_CHUNK (smaller per-chunk QK score buffers, smaller per-chunk dequantize buffers).

### Constraint B: Global RAM budget
Once model + accumulated KV + transients all materialize simultaneously, OS starts paging the model out. Throughput collapses to disk speeds; process appears "stuck" in `U` state with low RSS. Not chunk-tuneable — smaller chunks delay constraint B by reducing per-chunk transient size, but accumulated KV grows monotonically with context. At ~458K with int4 KV, total working set exceeds physical RAM.

## What unblocks past 400K

In order of plausibility / engineering investment:

1. **2-bit KV (KVLinC, Task 288)**: KV at 1M shrinks from 10.5 GB (int4) to 5.2 GB. Halves constraint B's KV term but doesn't change the per-chunk transient. **Buys maybe 1.5× context reach if integrated.**

2. **Online SnapKV during prefill (Layer 2 of 2026-05-02 plan)**: cap accumulated KV at K_max (e.g. 16K) by mid-prefill compaction. Removes the monotonic growth from constraint B. **The structurally correct fix; multi-cycle build.**

3. **TTT-Linear streaming heads (Task 388)**: replace 59% of attention compute with O(1)-per-token state-space. Reduces per-chunk attention transient on streaming heads. **Helps constraint A but not B directly; complementary to online SnapKV.**

4. **Sparse-attention prefill (MInference)**: per-head pattern dispatch reduces per-chunk QK cost. **Helps both constraints; gated on real Qwen3.6 calibration (D2 in disproof file).**

5. **Sequence-parallel prefill across multiple Metal command buffers**: split chunks across separate MTLBuffers. Research-level Apple Silicon work. **Bypasses constraint A; not plausibly hand-rolled in MLX.**

The fastest near-term unlock is #1 — KVLinC integration (Task 288). The structurally correct one is #2. Both are buildable across loop cycles.

## Caveat: YaRN extrapolation quality not measured

All runs above 256K use YaRN factor=2 (RoPE extension 256K → 512K). The fact that 384K *retrieves* a needle is encouraging — RoPE generalization holds at least that far. But:
- The needle is in the middle of a synthetic code haystack. Not a pathological position-encoding test.
- Multi-needle and reasoning-over-context tasks at 384K were not run.
- The plan's Layer 6 RoPE Goldilocks zone analysis (Task 332 spike, never run) would tell us where the precision wall is on Qwen3.6 — we're operating without that information.

## What's next

For continuing Goal 1 from here, ordered by acceptance gate:
1. **Run RoPE Goldilocks spike (Task 332)** — paper-only analysis, no model load. Tells us if the YaRN factor=2 we're using is in-zone at 384K. If out of zone, the "PASS" we observed may be lucky on this prompt and brittle on others.
2. **Calibrate MInference for Qwen3.6** (calibration script now hybrid-aware, D3 fix shipped). One-time cost ~5 min. Produces per-head sparse pattern table.
3. **Bench MInference at 64K-256K** — establishes whether sparse prefill produces output within fp16 tolerance and whether prefill speed improves.
4. **Spike online SnapKV during prefill** — Layer 2 of the plan. The structural unblock for >400K context.
