#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MLA rank probe on Qwen3-Coder KV cache (Task 53).

Extends the ShadowKV K-only probe (Task 44) to capture BOTH K and V
cache rank data. This informs whether Multi-head Latent Attention
(DeepSeek-V2 style joint KV compression) is viable for Qwen3-Coder.

For each layer, runs SVD on:
  - K cache: (T, H_kv * D) — same as Task 44
  - V cache: (T, H_kv * D) — new measurement
  - Joint KV: (T, H_kv * 2D) — concatenated K+V for MLA analysis

Reports the rank needed to capture 99% and 99.9% spectral energy.

Usage:
    python -m omlx.bench.mla_rank_probe [--context-len 8192]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np


def probe_kv_rank(
    model_id: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    context_len: int = 8192,
):
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)

    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)

    n_layers = len(model.layers)
    print(f"Model: {n_layers} layers")
    print(f"Metal at load: {mx.get_active_memory()/1e9:.1f} GB")

    # Diverse code context for calibration
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
    reps = (context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:context_len]

    print(f"Context: {len(full_tokens)} tokens")

    # Chunked prefill with fp16 KV cache
    cache = [KVCache() for _ in range(n_layers)]
    chunk_size = 2048

    print("Prefilling...")
    t0 = time.perf_counter()
    for start in range(0, len(full_tokens), chunk_size):
        end = min(start + chunk_size, len(full_tokens))
        x = mx.array([full_tokens[start:end]])
        logits = model(x, cache=cache)
        mx.eval(logits)
    prefill_time = time.perf_counter() - t0
    print(f"Prefill done in {prefill_time:.1f}s")
    print(f"Metal after prefill: {mx.get_active_memory()/1e9:.1f} GB")

    # SVD analysis per layer
    results = []
    print(f"\n{'Layer':>5} {'K rank@99%':>12} {'V rank@99%':>12} {'KV rank@99%':>13} {'K@99.9':>8} {'V@99.9':>8} {'KV@99.9':>8}")
    print("-" * 75)

    for layer_idx in range(n_layers):
        c = cache[layer_idx]
        keys = c.state[0][0].astype(mx.float32)   # (H_kv, T, D)
        values = c.state[1][0].astype(mx.float32)  # (H_kv, T, D)
        mx.eval(keys, values)

        H_kv, T, D = keys.shape

        # Reshape: (H_kv, T, D) → (T, H_kv*D)
        K_flat = np.array(keys.transpose(1, 0, 2).reshape(T, H_kv * D))
        V_flat = np.array(values.transpose(1, 0, 2).reshape(T, H_kv * D))
        # Joint KV: (T, H_kv*2D) for MLA analysis
        KV_flat = np.concatenate([K_flat, V_flat], axis=1)

        layer_result = {"layer": layer_idx, "H_kv": int(H_kv), "T": int(T), "D": int(D)}

        for name, mat in [("K", K_flat), ("V", V_flat), ("KV", KV_flat)]:
            try:
                S = np.linalg.svd(mat, compute_uv=False)
                energy = np.cumsum(S ** 2)
                total = energy[-1]
                if total == 0:
                    layer_result[f"{name}_rank_99"] = 0
                    layer_result[f"{name}_rank_999"] = 0
                    continue
                frac = energy / total
                r99 = int(np.searchsorted(frac, 0.99)) + 1
                r999 = int(np.searchsorted(frac, 0.999)) + 1
                layer_result[f"{name}_rank_99"] = r99
                layer_result[f"{name}_rank_999"] = r999
                layer_result[f"{name}_max_rank"] = len(S)
            except Exception as e:
                layer_result[f"{name}_error"] = str(e)

        results.append(layer_result)

        k99 = layer_result.get("K_rank_99", "?")
        v99 = layer_result.get("V_rank_99", "?")
        kv99 = layer_result.get("KV_rank_99", "?")
        k999 = layer_result.get("K_rank_999", "?")
        v999 = layer_result.get("V_rank_999", "?")
        kv999 = layer_result.get("KV_rank_999", "?")
        max_k = layer_result.get("K_max_rank", "?")
        max_kv = layer_result.get("KV_max_rank", "?")

        if layer_idx % 8 == 0 or layer_idx == n_layers - 1:
            print(f"  {layer_idx:>3}   {k99:>5}/{max_k}   {v99:>5}/{max_k}   {kv99:>6}/{max_kv}  {k999:>6}  {v999:>6}  {kv999:>6}")

        del keys, values, K_flat, V_flat, KV_flat
        gc.collect()
        mx.clear_cache()

    # Aggregate
    valid = [r for r in results if "K_rank_99" in r]
    k_ranks = [r["K_rank_99"] for r in valid]
    v_ranks = [r["V_rank_99"] for r in valid]
    kv_ranks = [r["KV_rank_99"] for r in valid]

    med_k = int(np.median(k_ranks))
    med_v = int(np.median(v_ranks))
    med_kv = int(np.median(kv_ranks))
    max_k_rank = valid[0].get("K_max_rank", 0)
    max_kv_rank = valid[0].get("KV_max_rank", 0)

    print(f"\n{'='*75}")
    print(f"SUMMARY (context={context_len}, {n_layers} layers, H_kv={valid[0]['H_kv']}, D={valid[0]['D']})")
    print(f"  K rank@99%:  median {med_k}/{max_k_rank} ({med_k/max_k_rank*100:.0f}%)")
    print(f"  V rank@99%:  median {med_v}/{max_k_rank} ({med_v/max_k_rank*100:.0f}%)")
    print(f"  KV rank@99%: median {med_kv}/{max_kv_rank} ({med_kv/max_kv_rank*100:.0f}%)")

    # Memory projection at 1M context
    # Current: 48 layers × 2 (K+V) × 4 KV-heads × 64 dim × 1M tokens × 3-bit = 22.5 GB
    kv_1m_gb = 22.5
    k_compression = med_k / max_k_rank
    v_compression = med_v / max_k_rank
    kv_joint_compression = med_kv / max_kv_rank

    projected_separate = kv_1m_gb * (k_compression + v_compression) / 2
    projected_joint = kv_1m_gb * kv_joint_compression

    print(f"\n  Memory projection at 1M context (current: {kv_1m_gb} GB):")
    print(f"    Separate SVD (K+V):   ~{projected_separate:.1f} GB ({(1-projected_separate/kv_1m_gb)*100:.0f}% savings)")
    print(f"    Joint MLA (KV):       ~{projected_joint:.1f} GB ({(1-projected_joint/kv_1m_gb)*100:.0f}% savings)")

    # Verdict
    print(f"\n  VERDICT:")
    if med_kv <= max_kv_rank * 0.5:
        print(f"    MLA joint compression VIABLE — KV rank is {med_kv/max_kv_rank*100:.0f}% of max")
        if projected_joint < projected_separate * 0.9:
            print(f"    Joint MLA preferred over separate SVD ({projected_joint:.1f} vs {projected_separate:.1f} GB)")
        else:
            print(f"    Separate SVD equally good ({projected_separate:.1f} vs {projected_joint:.1f} GB)")
    else:
        print(f"    MLA joint compression NOT VIABLE — KV rank is {med_kv/max_kv_rank*100:.0f}% of max")
        if k_compression < 0.5:
            print(f"    ShadowKV (K-only) is the better path (K rank {med_k/max_k_rank*100:.0f}%)")

    # Save results
    output = {
        "model": model_id,
        "context_len": context_len,
        "n_layers": n_layers,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "summary": {
            "K_median_rank_99": med_k,
            "V_median_rank_99": med_v,
            "KV_median_rank_99": med_kv,
            "K_max_rank": max_k_rank,
            "KV_max_rank": max_kv_rank,
            "K_compression_ratio": round(k_compression, 3),
            "V_compression_ratio": round(v_compression, 3),
            "KV_compression_ratio": round(kv_joint_compression, 3),
            "projected_1m_separate_gb": round(projected_separate, 1),
            "projected_1m_joint_gb": round(projected_joint, 1),
        },
        "per_layer": results,
    }

    out_path = Path(f"research/mla_rank_{datetime.now().strftime('%Y%m%d')}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="MLA rank probe (Task 53)")
    parser.add_argument("--context-len", type=int, default=8192,
                        help="Context length for calibration (default: 8192)")
    parser.add_argument("--model", type=str,
                        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    args = parser.parse_args()
    probe_kv_rank(args.model, args.context_len)


if __name__ == "__main__":
    main()
