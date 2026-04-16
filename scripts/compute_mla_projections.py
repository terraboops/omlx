#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compute MLA projection matrices for LatentKVCache.

Runs a calibration prefill, extracts K and V caches, computes SVD,
and saves the projection matrices that LatentKVCache needs at runtime.

This is a one-time offline computation. The output is:
  omlx/patches/mla_projections/qwen3_coder_30b_a3b/

Usage:
    python scripts/compute_mla_projections.py [--d-c 241] [--context-len 4096]
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
OUTPUT_DIR = Path("omlx/patches/mla_projections/qwen3_coder_30b_a3b")


def main():
    parser = argparse.ArgumentParser(
        description="Compute MLA projection matrices")
    parser.add_argument("--d-c", type=int, default=241,
                        help="Latent dimension (default: 241 from Task 53)")
    parser.add_argument("--context-len", type=int, default=4096,
                        help="Calibration context length (default: 4096)")
    args = parser.parse_args()

    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    model, tokenizer = load(MODEL_ID)

    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)

    n_layers = len(model.layers)
    logger.info(f"Model: {n_layers} layers, d_c={args.d_c}")

    # Build calibration context — diverse code
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

import json, os, sys
from pathlib import Path
from typing import List, Dict, Optional

'''
    tokens = tokenizer.encode(code)
    reps = (args.context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:args.context_len]
    logger.info(f"Calibration context: {len(full_tokens)} tokens")

    # Prefill
    cache = [KVCache() for _ in range(n_layers)]
    chunk_size = 2048
    logger.info("Prefilling...")
    for start in range(0, len(full_tokens), chunk_size):
        end = min(start + chunk_size, len(full_tokens))
        x = mx.array([full_tokens[start:end]])
        logits = model(x, cache=cache)
        mx.eval(logits)
    logger.info(f"Metal after prefill: {mx.get_active_memory()/1e9:.1f} GB")

    # Compute SVD projections per layer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "d_c": args.d_c,
        "context_len": args.context_len,
        "kv_dim": None,
        "per_layer_error": [],
    }

    logger.info(f"\nComputing SVD projections (d_c={args.d_c})...")
    logger.info(f"{'Layer':>5} {'kv_dim':>8} {'d_c':>5} {'rel_err':>10} {'time':>6}")
    logger.info("-" * 40)

    for layer_idx in range(n_layers):
        lt0 = time.perf_counter()
        c = cache[layer_idx]
        keys = c.state[0][0].astype(mx.float32)    # (H_kv, T, D)
        values = c.state[1][0].astype(mx.float32)   # (H_kv, T, D)
        mx.eval(keys, values)

        H_kv, T, D = keys.shape

        # Concatenate K and V: (T, H_kv*2D)
        K_flat = np.array(keys.transpose(1, 0, 2).reshape(T, H_kv * D))
        V_flat = np.array(values.transpose(1, 0, 2).reshape(T, H_kv * D))
        KV_flat = np.concatenate([K_flat, V_flat], axis=1)

        kv_dim = KV_flat.shape[1]
        if meta["kv_dim"] is None:
            meta["kv_dim"] = kv_dim

        # SVD
        U, S, Vt = np.linalg.svd(KV_flat, full_matrices=False)
        d_c = min(args.d_c, len(S))

        W_down = Vt[:d_c].T.astype(np.float16)  # (kv_dim, d_c)
        W_up = Vt[:d_c].astype(np.float16)       # (d_c, kv_dim)

        # Reconstruction error
        latent = KV_flat @ W_down.astype(np.float32)
        reconstructed = latent @ W_up.astype(np.float32)
        mse = np.mean((KV_flat - reconstructed) ** 2)
        rel_err = mse / (np.mean(KV_flat ** 2) + 1e-10)

        # Save
        mx.savez(str(OUTPUT_DIR / f"layer_{layer_idx}.npz"),
                 W_down=mx.array(W_down),
                 W_up=mx.array(W_up))

        elapsed = time.perf_counter() - lt0
        meta["per_layer_error"].append(round(float(rel_err), 6))

        if layer_idx % 8 == 0 or layer_idx == n_layers - 1:
            logger.info(f"  {layer_idx:>3}   {kv_dim:>6}  {d_c:>4}  {rel_err:>9.6f}  {elapsed:>5.1f}s")

        del keys, values, K_flat, V_flat, KV_flat, U, S, Vt, W_down, W_up
        gc.collect()
        mx.clear_cache()

    # Save metadata
    total_time = time.perf_counter() - t0
    meta["elapsed_s"] = round(total_time, 1)
    meta["mean_rel_error"] = round(float(np.mean(meta["per_layer_error"])), 6)
    meta["max_rel_error"] = round(float(np.max(meta["per_layer_error"])), 6)

    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"\n{'='*40}")
    logger.info(f"DONE: {n_layers} projections saved to {OUTPUT_DIR}")
    logger.info(f"  d_c={args.d_c}, kv_dim={meta['kv_dim']}")
    logger.info(f"  Mean rel error: {meta['mean_rel_error']:.6f}")
    logger.info(f"  Max rel error:  {meta['max_rel_error']:.6f}")
    logger.info(f"  Elapsed: {total_time:.1f}s")

    # Memory projection
    from omlx.latent_kv_cache import memory_estimate_gb
    mem_full = 22.5  # Current 3-bit GQA at 1M
    mem_latent_fp16 = memory_estimate_gb(n_layers, 1_000_000, args.d_c, 2.0)
    mem_latent_3bit = memory_estimate_gb(n_layers, 1_000_000, args.d_c, 0.375)
    logger.info(f"\n  Memory at 1M context:")
    logger.info(f"    Current 3-bit GQA: {mem_full:.1f} GB")
    logger.info(f"    Latent fp16:       {mem_latent_fp16:.1f} GB ({(1-mem_latent_fp16/mem_full)*100:.0f}% savings)")
    logger.info(f"    Latent 3-bit:      {mem_latent_3bit:.1f} GB ({(1-mem_latent_3bit/mem_full)*100:.0f}% savings)")


if __name__ == "__main__":
    main()
