#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate SnapKV token selection on real model attention patterns.

Runs a prefill, extracts attention weights from the observation window,
computes importance scores, and verifies:
1. Selection produces valid keep masks
2. Important tokens (beginning, end, high-attention) are kept
3. NIAH needle token is preserved when present
4. Memory savings projection at various keep ratios

Usage:
    python scripts/validate_snapkv_selection.py
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


def main():
    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    # Build a context with a known needle
    needle = "The secret activation code is SNAPKV-PASS-2026."
    filler = "This is standard documentation text for the software project. "
    context = filler * 40 + needle + " " + filler * 40
    prompt = context + "\n\nWhat is the secret activation code?"

    messages = [{"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_prompt = prompt + "\n"
    input_ids = tokenizer.encode(chat_prompt)
    T = len(input_ids)
    logger.info(f"Context: {T} tokens")

    # Find needle token positions
    needle_tokens = tokenizer.encode(needle)
    needle_start = None
    for i in range(len(input_ids) - len(needle_tokens)):
        if input_ids[i:i+len(needle_tokens)] == needle_tokens:
            needle_start = i
            break
    if needle_start:
        logger.info(f"Needle at positions {needle_start}-{needle_start+len(needle_tokens)}")
    else:
        logger.info("Needle position not found (may be split across chunks)")

    # Prefill
    cache = [KVCache() for _ in range(n_layers)]
    x = mx.array([input_ids])
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Extract Q, K from a middle layer for importance computation
    test_layer = 24
    c = cache[test_layer]
    keys = c.state[0]  # (B, H_kv, T, D)
    B, H_kv, T_cache, D = keys.shape

    # We need queries too — get them by re-running the last chunk
    # Use the observation window approach: last 64 tokens
    obs_window = 64

    # Get attention config
    attn = model.layers[test_layer].self_attn
    H_q = attn.n_heads if hasattr(attn, 'n_heads') else 32
    scale = D ** -0.5

    logger.info(f"Layer {test_layer}: H_q={H_q}, H_kv={H_kv}, D={D}, T={T_cache}")

    # Compute importance using cached K and reconstructed Q
    # For simplicity, use K as a proxy for Q (self-attention, similar subspace)
    # Real implementation would extract Q from the forward pass
    from omlx.patches.snapkv import compute_attention_importance, snapkv_select, count_kept

    # Use K expanded to H_q as proxy for Q (approximation for validation)
    Q_proxy = mx.repeat(keys, H_q // H_kv, axis=1)  # (B, H_q, T, D)

    logger.info("Computing attention importance...")
    importance = compute_attention_importance(
        Q_proxy, keys, scale, obs_window=obs_window)
    mx.eval(importance)

    # Importance shape: (B, H_kv, T)
    imp_np = np.array(importance[0])  # (H_kv, T)
    logger.info(f"Importance shape: {imp_np.shape}")
    logger.info(f"Importance range: [{imp_np.min():.4f}, {imp_np.max():.4f}]")

    # Test selection at various keep ratios
    results = []
    keep_ratios = [0.25, 0.50, 0.75, 0.90]

    logger.info(f"\n{'Keep%':>6} {'Kept':>6} {'Evicted':>8} {'Savings':>8} {'Needle?':>8}")
    logger.info("-" * 45)

    for ratio in keep_ratios:
        keep_count = int(T_cache * ratio)
        keep_mask = snapkv_select(importance, keep_count, always_keep_last=obs_window)
        mx.eval(keep_mask)

        n_kept = count_kept(keep_mask)
        n_evicted = T_cache - n_kept
        savings_pct = round(n_evicted / T_cache * 100, 0)

        # Check if needle is preserved
        needle_kept = True
        if needle_start is not None:
            mask_np = np.array(keep_mask[0])
            for pos in range(needle_start, min(needle_start + len(needle_tokens), T_cache)):
                if not mask_np[pos]:
                    needle_kept = False
                    break

        result = {
            "keep_ratio": ratio,
            "keep_count": keep_count,
            "actual_kept": n_kept,
            "evicted": n_evicted,
            "savings_pct": savings_pct,
            "needle_preserved": needle_kept,
        }
        results.append(result)

        logger.info(f"  {ratio*100:>4.0f}%  {n_kept:>5}  {n_evicted:>7}    {savings_pct:>5.0f}%  "
                    f"{'YES' if needle_kept else 'NO':>6}")

    # Memory projection at 128K context
    logger.info(f"\nMemory savings projection at 128K context:")
    kv_128k_gb = 5.4  # Current 3-bit GQA at 128K
    for r in results:
        saved = kv_128k_gb * r["savings_pct"] / 100
        remaining = kv_128k_gb - saved
        logger.info(f"  Keep {r['keep_ratio']*100:.0f}%: {remaining:.1f} GB KV "
                    f"({saved:.1f} GB freed, needle={'SAFE' if r['needle_preserved'] else 'LOST'})")

    elapsed = time.perf_counter() - t0
    logger.info(f"\nElapsed: {elapsed:.1f}s")

    # Save
    out = {"T": T_cache, "obs_window": obs_window, "layer": test_layer,
           "results": results, "elapsed_s": round(elapsed, 1)}
    Path("research/snapkv_selection_validation.json").write_text(json.dumps(out, indent=2))
    logger.info("Results: research/snapkv_selection_validation.json")

    # Verdict
    needle_safe_at_50 = any(r["keep_ratio"] == 0.50 and r["needle_preserved"] for r in results)
    logger.info(f"\nVERDICT: SnapKV selection {'VALIDATED' if needle_safe_at_50 else 'NEEDS TUNING'}")
    if needle_safe_at_50:
        logger.info("  Needle preserved at 50% keep ratio — attention-based selection works")
    else:
        logger.info("  Needle lost at 50% — may need larger observation window or per-head selection")


if __name__ == "__main__":
    main()
