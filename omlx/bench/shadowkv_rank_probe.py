#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ShadowKV SVD-rank probe on Qwen3-Coder K cache.

Prefills a context, extracts the fp16 K cache per layer, runs SVD,
and reports the rank needed to capture 99/99.5/99.9% of the
Frobenius norm. Determines whether ShadowKV (low-rank K approximation)
or InfLLM (block-based tiering) is the right approach.

Decision rule:
  median rank <= 256 for 99% norm → ShadowKV viable
  median rank > 256 → InfLLM is the better bet

Usage:
    .venv/bin/python omlx/bench/shadowkv_rank_probe.py
    .venv/bin/python omlx/bench/shadowkv_rank_probe.py --context-len 16384
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
    context_len: int = 4096,
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

    # Build a synthetic code context
    code = '''def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self): self.data = []
    def add(self, item): self.data.append(item)
    def process(self): return [x*2 for x in self.data]

'''
    tokens = tokenizer.encode(code)
    # Repeat to fill context
    reps = (context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:context_len]

    print(f"Context: {len(full_tokens)} tokens")

    # Prefill with fp16 KV cache to capture raw K vectors
    cache = [KVCache() for _ in range(n_layers)]
    x = mx.array([full_tokens])

    print("Prefilling...")
    t0 = time.perf_counter()
    logits = model(x, cache=cache)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0
    print(f"Prefill done in {prefill_time:.1f}s")
    print(f"Metal after prefill: {mx.get_active_memory()/1e9:.1f} GB")

    # Extract K cache and run SVD per layer
    results = []
    print(f"\nSVD analysis per layer (context={context_len}):")
    print(f"{'Layer':>6} {'K shape':>20} {'rank@99%':>10} {'rank@99.5%':>12} {'rank@99.9%':>12}")
    print("-" * 65)

    for layer_idx in range(n_layers):
        c = cache[layer_idx]
        # KVCache stores keys as (B, H, T, D)
        keys = c.state[0]  # (1, H_kv, T, D)
        mx.eval(keys)

        # Reshape to (H_kv, T, D) and process per head
        K = keys[0].astype(mx.float32)  # (H_kv, T, D)
        H_kv, T, D = K.shape

        # SVD on concatenated heads: (T, H_kv*D)
        K_flat = K.transpose(1, 0, 2).reshape(T, H_kv * D)  # (T, H_kv*D)
        mx.eval(K_flat)

        # MLX SVD
        try:
            # MLX SVD is GPU-only limitation — use numpy SVD on CPU
            K_np = np.array(K_flat)
            S_np = np.linalg.svd(K_np, compute_uv=False)
        except Exception as e:
            print(f"  Layer {layer_idx}: SVD failed ({e})")
            results.append({"layer": layer_idx, "error": str(e)})
            continue

        # Compute cumulative energy (Frobenius norm = sqrt(sum(s^2)))
        energy = np.cumsum(S_np ** 2)
        total_energy = energy[-1]

        if total_energy == 0:
            results.append({"layer": layer_idx, "rank_99": 0, "rank_995": 0, "rank_999": 0})
            continue

        frac = energy / total_energy

        rank_99 = int(np.searchsorted(frac, 0.99)) + 1
        rank_995 = int(np.searchsorted(frac, 0.995)) + 1
        rank_999 = int(np.searchsorted(frac, 0.999)) + 1
        max_rank = len(S_np)

        print(f"  {layer_idx:>4}   ({H_kv}, {T}, {D}) → ({T}, {H_kv*D})"
              f"  {rank_99:>8}/{max_rank}"
              f"  {rank_995:>10}/{max_rank}"
              f"  {rank_999:>10}/{max_rank}")

        results.append({
            "layer": layer_idx,
            "K_shape": [int(H_kv), int(T), int(D)],
            "max_rank": int(max_rank),
            "rank_99": int(rank_99),
            "rank_995": int(rank_995),
            "rank_999": int(rank_999),
            "top5_singular": [round(float(s), 4) for s in S_np[:5]],
        })

        # Free memory between layers
        del K, K_flat
        gc.collect()
        mx.clear_cache()

    # Aggregate
    valid = [r for r in results if "rank_99" in r]
    if not valid:
        print("\nNo valid SVD results!")
        return

    ranks_99 = [r["rank_99"] for r in valid]
    median_rank_99 = int(np.median(ranks_99))
    max_rank = valid[0]["max_rank"]

    print(f"\n{'=' * 65}")
    print(f"SUMMARY (context={context_len}, {n_layers} layers)")
    print(f"  rank@99% — median: {median_rank_99}, min: {min(ranks_99)}, max: {max(ranks_99)}")
    print(f"  max possible rank: {max_rank}")

    if median_rank_99 <= 256:
        verdict = "ShadowKV VIABLE"
        recommendation = ("K cache is low-rank. ShadowKV's SVD-based "
                          "compression will work. Recommend replacing "
                          "Task 43 (InfLLM) with a ShadowKV implementation.")
    else:
        verdict = "InfLLM is the better bet"
        recommendation = ("K cache is NOT low-rank (median rank@99% > 256). "
                          "ShadowKV would need too many components. "
                          "Proceed with Task 43 (InfLLM block-based tiering).")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {recommendation}")
    print(f"{'=' * 65}")

    # Save results
    output = {
        "model": model_id,
        "context_len": context_len,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "median_rank_99": median_rank_99,
        "verdict": verdict,
        "layers": results,
    }

    out_dir = Path("research")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"shadowkv_rank_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    # Cleanup
    del cache, logits
    gc.collect()
    mx.clear_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description="ShadowKV SVD-rank probe")
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    parser.add_argument("--context-len", type=int, default=4096,
                        help="Context length for prefill (default: 4096)")
    args = parser.parse_args()
    probe_kv_rank(args.model, args.context_len)


if __name__ == "__main__":
    main()
