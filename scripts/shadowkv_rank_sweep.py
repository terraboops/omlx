#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ShadowKV rank sweep — find optimal rank for quality vs compression.

Tests multiple rank values to find the sweet spot where token agreement
is >80% while still getting meaningful K compression. Uses the same
compress_k_with_projections infrastructure but recomputes projections
at each rank from the cached SVD data.

Usage:
    python scripts/shadowkv_rank_sweep.py
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


def generate_tokens(model, cache, logits, n_tokens=32):
    tokens = []
    for _ in range(n_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(token.item())
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
    return tokens


def compress_k_at_rank(cache, rank, n_layers, calib_cache):
    """Compress K cache using SVD truncated to specific rank.

    Instead of loading pre-computed projections at a fixed rank,
    this recomputes from the calibration cache's K vectors directly.
    """
    for layer_idx in range(n_layers):
        c = cache[layer_idx]
        keys = c.state[0]  # (B, H_kv, T, D)
        B, H_kv, T, D = keys.shape

        new_heads = []
        for head_idx in range(H_kv):
            K_head = keys[:, head_idx, :, :]  # (B, T, D)

            # Get calibration K to compute SVD basis
            calib_K = calib_cache[layer_idx][head_idx]  # (D, full_rank) Vt basis
            r = min(rank, calib_K.shape[1])
            V_r = calib_K[:, :r]  # (D, r) — truncated basis

            # Project: K_approx = K @ V_r @ V_r.T
            K_low = K_head @ V_r
            K_approx = K_low @ V_r.T
            new_heads.append(K_approx)

        K_compressed = mx.stack(new_heads, axis=1)
        c.state = (K_compressed, c.state[1])

    mx.eval(*[c.state[0] for c in cache])


def main():
    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    # First, build calibration SVD basis from a prefill
    logger.info("Computing calibration SVD basis...")
    code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n\n"
    tokens = tokenizer.encode(code)
    calib_tokens = (tokens * 20)[:1024]

    calib_kv = [KVCache() for _ in range(n_layers)]
    x = mx.array([calib_tokens])
    logits = model(x, cache=calib_kv)
    mx.eval(logits)

    # Extract full SVD basis per head
    calib_basis = []  # [layer][head] = Vt.T (D, D) full basis
    for layer_idx in range(n_layers):
        keys = calib_kv[layer_idx].state[0][0]  # (H_kv, T, D)
        mx.eval(keys)
        H_kv = keys.shape[0]
        layer_heads = []
        for head_idx in range(H_kv):
            K_np = np.array(keys[head_idx].astype(mx.float32))
            _, _, Vt = np.linalg.svd(K_np, full_matrices=False)
            # Store full Vt.T as basis: (D, D)
            layer_heads.append(mx.array(Vt.T.astype(np.float16)))
        calib_basis.append(layer_heads)
    del calib_kv; gc.collect(); mx.clear_cache()

    # Test prompts
    test_prompts = [
        ("Math", "What is 2+2? Answer with just the number:"),
        ("Code", "Write a Python function to check if a number is prime:\n\n```python\ndef is_prime(n):"),
        ("NIAH", "The secret code is SHADOWKV-99. " + "Filler. " * 80 + "What is the secret code?"),
    ]

    # Get baseline outputs
    logger.info("\nGenerating baseline outputs...")
    baselines = {}
    for name, prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_prompt = prompt + "\n"
        input_ids = tokenizer.encode(chat_prompt)

        cache = [KVCache() for _ in range(n_layers)]
        x = mx.array([input_ids])
        logits = model(x, cache=cache)
        mx.eval(logits)
        base_tokens = generate_tokens(model, cache, logits, 32)
        baselines[name] = {"input_ids": input_ids, "tokens": base_tokens,
                           "text": tokenizer.decode(base_tokens)}
        del cache; gc.collect(); mx.clear_cache()

    # Sweep ranks
    ranks = [64, 85, 96, 108, 116, 124, 128]
    results = []

    logger.info(f"\n{'Rank':>5} {'Compress':>10} {'Math':>6} {'Code':>6} {'NIAH':>6} {'Avg':>6}")
    logger.info("-" * 50)

    for rank in ranks:
        compression = round((1 - rank / 128) * 100, 0)
        agreements = {}

        for name, prompt in test_prompts:
            bl = baselines[name]
            cache = [KVCache() for _ in range(n_layers)]
            x = mx.array([bl["input_ids"]])
            logits = model(x, cache=cache)
            mx.eval(logits)

            compress_k_at_rank(cache, rank, n_layers, calib_basis)
            skv_tokens = generate_tokens(model, cache, logits, 32)

            agree = sum(1 for a, b in zip(bl["tokens"], skv_tokens) if a == b)
            agreements[name] = round(agree / len(bl["tokens"]) * 100, 0)
            del cache; gc.collect(); mx.clear_cache()

        avg = round(sum(agreements.values()) / len(agreements), 0)
        results.append({
            "rank": rank,
            "compression_pct": compression,
            "agreements": agreements,
            "avg_agreement": avg,
        })

        logger.info(f"  {rank:>3}   {compression:>7.0f}%  {agreements['Math']:>5.0f}%"
                    f" {agreements['Code']:>5.0f}% {agreements['NIAH']:>5.0f}% {avg:>5.0f}%")

    # Find sweet spot
    elapsed = time.perf_counter() - t0
    logger.info(f"\nElapsed: {elapsed:.1f}s")

    # Best rank that achieves >80% avg agreement
    viable = [r for r in results if r["avg_agreement"] >= 80]
    if viable:
        best = min(viable, key=lambda r: r["rank"])
        logger.info(f"\nRECOMMENDATION: rank={best['rank']} "
                    f"({best['compression_pct']:.0f}% K compression, "
                    f"{best['avg_agreement']:.0f}% agreement)")
    else:
        # Fallback: best agreement regardless
        best = max(results, key=lambda r: r["avg_agreement"])
        logger.info(f"\nNo rank achieves >80% agreement. Best: rank={best['rank']} "
                    f"({best['avg_agreement']:.0f}%)")

    Path("research/shadowkv_rank_sweep.json").write_text(json.dumps(results, indent=2))
    logger.info("Results: research/shadowkv_rank_sweep.json")


if __name__ == "__main__":
    main()
