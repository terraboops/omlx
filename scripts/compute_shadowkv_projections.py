#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compute per-head K-only SVD projections for ShadowKV (Step 1).

Unlike the MLA joint projections (which failed quality validation),
ShadowKV compresses K cache only, per-head. V cache stays exact.

For each (layer, head), computes:
  U_k: (D, r) — top-r right singular vectors of K
  Used to project K to low-rank: K_compressed = K @ U_k @ U_k.T

The sparse residual (high-norm tokens kept at full precision) is
handled at runtime, not in the projection computation.

Output:
  omlx/patches/shadowkv_projections/qwen3_coder_30b_a3b/meta.json
  omlx/patches/shadowkv_projections/qwen3_coder_30b_a3b/layer_N.npz

Usage:
    python scripts/compute_shadowkv_projections.py [--rank 128] [--context-len 4096]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
OUTPUT_DIR = Path("omlx/patches/shadowkv_projections/qwen3_coder_30b_a3b")


def main():
    parser = argparse.ArgumentParser(
        description="Compute ShadowKV per-head K projections (Step 1)")
    parser.add_argument("--rank", type=int, default=0,
                        help="Fixed rank per head (0 = adaptive from 99%% energy)")
    parser.add_argument("--context-len", type=int, default=4096,
                        help="Calibration context length (default: 4096)")
    parser.add_argument("--energy-threshold", type=float, default=0.99,
                        help="Energy threshold for adaptive rank (default: 0.99)")
    args = parser.parse_args()

    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    model, tokenizer = load(MODEL_ID)

    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)

    n_layers = len(model.layers)

    # Calibration prefill
    code = '''def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self): self.data = []
    def add(self, item): self.data.append(item)
    def process(self): return sorted(set(self.data))

def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1

'''
    tokens = tokenizer.encode(code)
    reps = (args.context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:args.context_len]
    logger.info(f"Calibration: {len(full_tokens)} tokens, energy threshold: {args.energy_threshold}")

    cache = [KVCache() for _ in range(n_layers)]
    chunk_size = 2048
    for start in range(0, len(full_tokens), chunk_size):
        end = min(start + chunk_size, len(full_tokens))
        x = mx.array([full_tokens[start:end]])
        logits = model(x, cache=cache)
        mx.eval(logits)

    logger.info(f"Metal after prefill: {mx.get_active_memory()/1e9:.1f} GB")

    # Compute per-head SVD
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "context_len": args.context_len,
        "energy_threshold": args.energy_threshold,
        "fixed_rank": args.rank,
        "per_layer": [],
    }

    logger.info(f"\n{'Layer':>5} {'Head':>5} {'D':>4} {'rank':>5} {'energy':>8} {'rel_err':>10}")
    logger.info("-" * 50)

    for layer_idx in range(n_layers):
        c = cache[layer_idx]
        keys = c.state[0][0].astype(mx.float32)  # (H_kv, T, D)
        mx.eval(keys)
        H_kv, T, D = keys.shape

        layer_ranks = []
        layer_errors = []
        projections = {}

        for head_idx in range(H_kv):
            K_head = np.array(keys[head_idx])  # (T, D)

            # SVD
            U, S, Vt = np.linalg.svd(K_head, full_matrices=False)
            energy = np.cumsum(S ** 2)
            total_energy = energy[-1]

            if args.rank > 0:
                r = min(args.rank, len(S))
            else:
                # Adaptive: find rank for energy_threshold
                if total_energy > 0:
                    frac = energy / total_energy
                    r = int(np.searchsorted(frac, args.energy_threshold)) + 1
                    r = min(r, len(S))
                else:
                    r = 1

            # Projection matrix: V_r = Vt[:r].T, shape (D, r)
            # To compress: K_low = K @ V_r @ V_r.T = K @ (V_r @ V_r.T)
            # Store V_r as the projection basis
            V_r = Vt[:r].T.astype(np.float16)  # (D, r)

            # Reconstruction error
            K_low = K_head @ V_r.astype(np.float32) @ V_r.T.astype(np.float32)
            mse = np.mean((K_head - K_low) ** 2)
            rel_err = mse / (np.mean(K_head ** 2) + 1e-10)

            captured = energy[r - 1] / total_energy if total_energy > 0 else 0

            layer_ranks.append(r)
            layer_errors.append(float(rel_err))
            projections[f"head_{head_idx}"] = mx.array(V_r)

        # Save per-layer
        mx.savez(str(OUTPUT_DIR / f"layer_{layer_idx}.npz"), **projections)

        avg_rank = int(np.mean(layer_ranks))
        avg_err = float(np.mean(layer_errors))
        meta["per_layer"].append({
            "layer": layer_idx,
            "ranks": layer_ranks,
            "avg_rank": avg_rank,
            "avg_rel_error": round(avg_err, 6),
            "H_kv": int(H_kv),
            "D": int(D),
        })

        if layer_idx % 8 == 0 or layer_idx == n_layers - 1:
            logger.info(f"  {layer_idx:>3}   {H_kv:>3}×  {D:>3}  {avg_rank:>4}   "
                        f"{args.energy_threshold:>6.1%}  {avg_err:>9.6f}")

        del keys
        gc.collect()
        mx.clear_cache()

    # Aggregate
    all_ranks = [r for ld in meta["per_layer"] for r in ld["ranks"]]
    all_errors = [ld["avg_rel_error"] for ld in meta["per_layer"]]

    meta["summary"] = {
        "total_heads": len(all_ranks),
        "median_rank": int(np.median(all_ranks)),
        "min_rank": int(min(all_ranks)),
        "max_rank": int(max(all_ranks)),
        "mean_rel_error": round(float(np.mean(all_errors)), 6),
        "max_rel_error": round(float(np.max(all_errors)), 6),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }

    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    s = meta["summary"]
    logger.info(f"\n{'='*50}")
    logger.info(f"DONE: {n_layers} layers × {H_kv} heads = {s['total_heads']} projections")
    logger.info(f"  Rank: median={s['median_rank']}, range=[{s['min_rank']}, {s['max_rank']}]")
    logger.info(f"  Error: mean={s['mean_rel_error']:.6f}, max={s['max_rel_error']:.6f}")
    logger.info(f"  Elapsed: {s['elapsed_s']:.1f}s")
    logger.info(f"  Output: {OUTPUT_DIR}")

    # Memory projection
    median_r = s["median_rank"]
    D_full = meta["per_layer"][0]["D"]
    compression = median_r / D_full
    logger.info(f"\n  K compression: {median_r}/{D_full} = {compression:.0%} of original")
    logger.info(f"  At 128K: K cache {2.7 * compression:.1f} GB (was 2.7 GB)")
    logger.info(f"  At 1M:   K cache {11.25 * compression:.1f} GB (was 11.25 GB)")


if __name__ == "__main__":
    main()
