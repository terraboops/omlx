#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe: attention-weighted vs uniform SVD reconstruction error.

Hypothesis: SVD compression fails on retrieval (NIAH) because it
preferentially discards low-energy K directions that happen to be
high-importance for attention. The "needle" token's K vector contributes
little to the Frobenius norm but is critical for the attention spike.

This probe measures:
  1. Uniform MSE: mean((K - K_approx)^2) — what SVD optimizes
  2. Attention-weighted MSE: mean(attn_weight * (K - K_approx)^2)
     where attn_weight comes from the observation window

If the hypothesis is correct:
  - For NIAH prompts: attn-weighted error >> uniform error (10-100x)
  - For code prompts: attn-weighted error ≈ uniform error

This would explain why rank 85 gets 100% code agreement but 16% NIAH.

Usage:
    python scripts/probe_attn_weighted_svd_error.py
"""

from __future__ import annotations

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


def compute_errors(K_np, K_approx_np, attn_weights_np):
    """Compute uniform and attention-weighted reconstruction errors.

    Args:
        K_np: (T, D) original K vectors
        K_approx_np: (T, D) SVD-reconstructed K vectors
        attn_weights_np: (T,) per-token attention importance

    Returns:
        (uniform_mse, weighted_mse, ratio)
    """
    sq_err = np.sum((K_np - K_approx_np) ** 2, axis=-1)  # (T,) per-token error
    uniform_mse = np.mean(sq_err)

    # Normalize attention weights to sum to 1
    w = attn_weights_np / (attn_weights_np.sum() + 1e-10)
    weighted_mse = np.sum(w * sq_err)

    ratio = weighted_mse / (uniform_mse + 1e-10)
    return float(uniform_mse), float(weighted_mse), float(ratio)


def main():
    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    # Two prompts: one retrieval (NIAH), one code
    prompts = {
        "niah": ("The secret code is NEEDLE-42. " + "Filler text. " * 60
                 + "What is the secret code?"),
        "code": ("Write a Python function to sort a list using quicksort:\n\n"
                 "```python\ndef quicksort(arr):\n"
                 + "    # implementation\n" * 20),
    }

    results = {}

    for prompt_name, prompt_text in prompts.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt: {prompt_name}")

        messages = [{"role": "user", "content": prompt_text}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_prompt = prompt_text + "\n"
        input_ids = tokenizer.encode(chat_prompt)
        T = len(input_ids)
        logger.info(f"  Tokens: {T}")

        # Prefill
        cache = [KVCache() for _ in range(n_layers)]
        x = mx.array([input_ids])
        logits = model(x, cache=cache)
        mx.eval(logits)

        # Analyze a sample of layers
        test_layers = [0, 12, 24, 36, 47]
        ranks = [64, 85, 96, 108, 116, 124]
        prompt_results = {}

        for layer_idx in test_layers:
            c = cache[layer_idx]
            keys = c.state[0][0]  # (H_kv, T, D)
            mx.eval(keys)
            H_kv, T_cache, D = keys.shape

            # Get attention config for this layer
            attn = model.layers[layer_idx].self_attn
            H_q = attn.n_heads if hasattr(attn, 'n_heads') else 32
            scale = D ** -0.5

            # Compute attention weights from last 64 tokens (observation window)
            obs_window = min(64, T_cache)
            obs_start = T_cache - obs_window

            # Use Q from the KV cache (K as proxy for Q in GQA)
            Q_proxy = keys[:, obs_start:, :]  # (H_kv, obs_len, D)

            # Attention scores: Q_obs @ K.T
            scores_np = np.array(
                (Q_proxy @ keys.swapaxes(-1, -2) * scale).astype(mx.float32))
            # (H_kv, obs_len, T)

            # Causal mask
            for h in range(H_kv):
                for q in range(obs_window):
                    q_pos = obs_start + q
                    scores_np[h, q, q_pos+1:] = -1e9

            # Softmax
            scores_np -= scores_np.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores_np)
            attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-10)
            # (H_kv, obs_len, T)

            # Pool: max over query positions, mean over heads
            importance = attn_weights.max(axis=1).mean(axis=0)  # (T,)

            # Now compute SVD errors at various ranks
            layer_results = {}
            for rank in ranks:
                head_uniform = []
                head_weighted = []
                head_ratio = []

                for h in range(H_kv):
                    K_np = np.array(keys[h].astype(mx.float32))  # (T, D)
                    _, S, Vt = np.linalg.svd(K_np, full_matrices=False)
                    r = min(rank, len(S))
                    V_r = Vt[:r].T  # (D, r)
                    K_approx = K_np @ V_r @ V_r.T

                    u_mse, w_mse, ratio = compute_errors(K_np, K_approx, importance)
                    head_uniform.append(u_mse)
                    head_weighted.append(w_mse)
                    head_ratio.append(ratio)

                layer_results[rank] = {
                    "uniform_mse": round(float(np.mean(head_uniform)), 6),
                    "weighted_mse": round(float(np.mean(head_weighted)), 6),
                    "ratio": round(float(np.mean(head_ratio)), 2),
                }

            prompt_results[layer_idx] = layer_results

        results[prompt_name] = prompt_results
        del cache; gc.collect(); mx.clear_cache()

    # Print comparison
    elapsed = time.perf_counter() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"ATTENTION-WEIGHTED vs UNIFORM SVD ERROR")
    logger.info(f"{'='*60}")
    logger.info(f"{'Prompt':>8} {'Layer':>6} {'Rank':>5} {'Uniform':>10} {'Weighted':>10} {'Ratio':>7}")
    logger.info("-" * 55)

    for prompt_name in ["niah", "code"]:
        for layer_idx in [0, 24, 47]:
            for rank in [85, 116, 124]:
                r = results[prompt_name].get(layer_idx, {}).get(rank, {})
                if r:
                    logger.info(f"  {prompt_name:>6}  L{layer_idx:>3}  {rank:>4}  "
                                f"{r['uniform_mse']:>9.6f}  {r['weighted_mse']:>9.6f}  "
                                f"{r['ratio']:>5.1f}x")
        logger.info("")

    # Verdict
    niah_ratios = []
    code_ratios = []
    for layer_idx in results.get("niah", {}):
        for rank in results["niah"][layer_idx]:
            niah_ratios.append(results["niah"][layer_idx][rank]["ratio"])
    for layer_idx in results.get("code", {}):
        for rank in results["code"][layer_idx]:
            code_ratios.append(results["code"][layer_idx][rank]["ratio"])

    avg_niah = np.mean(niah_ratios) if niah_ratios else 0
    avg_code = np.mean(code_ratios) if code_ratios else 0

    logger.info(f"Average ratio (weighted/uniform):")
    logger.info(f"  NIAH: {avg_niah:.1f}x")
    logger.info(f"  Code: {avg_code:.1f}x")

    if avg_niah > avg_code * 2:
        logger.info(f"\nHYPOTHESIS CONFIRMED: Retrieval errors are {avg_niah/avg_code:.1f}x "
                    f"more attention-concentrated than code errors.")
        logger.info(f"SVD discards low-energy directions that are high-attention for retrieval.")
    else:
        logger.info(f"\nHYPOTHESIS NOT CONFIRMED: Ratio difference is only "
                    f"{avg_niah/avg_code:.1f}x (expected >2x).")

    logger.info(f"\nElapsed: {elapsed:.1f}s")

    Path("research/attn_weighted_svd_error.json").write_text(
        json.dumps({"results": {k: {str(lk): lv for lk, lv in v.items()}
                     for k, v in results.items()},
                    "avg_niah_ratio": round(avg_niah, 2),
                    "avg_code_ratio": round(avg_code, 2),
                    "elapsed_s": round(elapsed, 1)}, indent=2))
    logger.info("Results: research/attn_weighted_svd_error.json")


if __name__ == "__main__":
    main()
