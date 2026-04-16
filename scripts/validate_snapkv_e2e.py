#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SnapKV validation: prefill → select → compact → decode.

Tests the full pipeline: compute attention importance, select tokens,
compact cache in-place, then decode and compare against baseline.
This is the integration test that proves SnapKV works for real inference.

Usage:
    python scripts/validate_snapkv_e2e.py
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


def main():
    t0 = time.perf_counter()

    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    from omlx.patches.snapkv import (
        capture_attention_weights,
        snapkv_select, get_keep_indices, compact_cache, count_kept,
    )

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    tests = [
        ("NIAH", "The activation code is SNAPKV-E2E-PASS. " + "Filler text for padding the context to a reasonable length. " * 200
         + "What is the activation code? Reply with just the code:"),
        ("Math", "What is 17 * 23? Show your work:"),
        ("Code", "def reverse_string(s: str) -> str:\n    '''Reverse a string.'''\n    return"),
    ]

    results = []
    for name, prompt in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Test: {name}")

        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_prompt = prompt + "\n"
        input_ids = tokenizer.encode(chat_prompt)
        T = len(input_ids)

        # === Baseline ===
        cache_base = [KVCache() for _ in range(n_layers)]
        logits = model(mx.array([input_ids]), cache=cache_base)
        mx.eval(logits)
        base_tokens = generate_tokens(model, cache_base, logits)
        base_text = tokenizer.decode(base_tokens)
        del cache_base; gc.collect(); mx.clear_cache()

        # === SnapKV: prefill → select → compact → decode ===
        cache_skv = [KVCache() for _ in range(n_layers)]
        logits = model(mx.array([input_ids]), cache=cache_skv)
        mx.eval(logits)

        # Compute importance from last 4 layers using per-KV-head attention
        importance = capture_attention_weights(
            model, cache_skv, obs_window=64,
            layers=list(range(n_layers - 4, n_layers)))
        H_kv = cache_skv[0].state[0].shape[1]

        # Select tokens to keep (50%)
        keep_count = T // 2
        keep_mask = snapkv_select(importance, keep_count, always_keep_last=64)
        mx.eval(keep_mask)
        keep_idx = get_keep_indices(keep_mask)
        n_kept = len(keep_idx)

        # Compact cache
        compact_cache(cache_skv, keep_idx)

        # Decode from compacted cache
        skv_tokens = generate_tokens(model, cache_skv, logits)
        skv_text = tokenizer.decode(skv_tokens)
        del cache_skv; gc.collect(); mx.clear_cache()

        # Compare
        agree = sum(1 for a, b in zip(base_tokens, skv_tokens) if a == b)
        agreement = round(agree / len(base_tokens) * 100, 0)

        result = {
            "name": name,
            "T": T,
            "kept": n_kept,
            "evicted": T - n_kept,
            "keep_pct": round(n_kept / T * 100, 0),
            "agreement_pct": agreement,
            "base_text": base_text[:120],
            "skv_text": skv_text[:120],
        }
        results.append(result)

        logger.info(f"  Tokens: {T} → {n_kept} kept ({result['keep_pct']:.0f}%)")
        logger.info(f"  Agreement: {agreement:.0f}%")
        logger.info(f"  Base: {base_text[:80]!r}")
        logger.info(f"  SKV:  {skv_text[:80]!r}")

    # Summary
    elapsed = time.perf_counter() - t0
    avg_agree = np.mean([r["agreement_pct"] for r in results])

    logger.info(f"\n{'='*50}")
    logger.info(f"SNAPKV END-TO-END VALIDATION")
    logger.info(f"{'='*50}")
    for r in results:
        s = "PASS" if r["agreement_pct"] >= 30 else "FAIL"
        logger.info(f"  {r['name']:>6}: {r['agreement_pct']:>3.0f}% agree, "
                    f"{r['keep_pct']:.0f}% kept  {s}")

    logger.info(f"\n  Average agreement: {avg_agree:.0f}%")
    if avg_agree >= 50:
        logger.info("  VERDICT: SnapKV E2E PASS — pipeline works")
    else:
        logger.info("  VERDICT: SnapKV E2E NEEDS WORK")
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    Path("research/snapkv_e2e_validation.json").write_text(
        json.dumps({"results": results, "avg_agreement": round(avg_agree, 1),
                    "elapsed_s": round(elapsed, 1)}, indent=2))


if __name__ == "__main__":
    main()
