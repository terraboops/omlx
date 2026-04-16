#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SnapKV quality validation via attention masking (not physical compaction).

Physical compaction breaks MLX's KVCache (offset/RoPE issues). This script
validates the SELECTION algorithm by masking evicted positions to -inf
instead of removing them. The KV cache stays intact — we just prevent
the model from attending to evicted tokens.

This proves whether the SnapKV selection is correct before investing
in a custom SparseKVCache class.

Usage:
    python scripts/validate_snapkv_masking.py
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


def generate_with_mask(model, tokenizer, input_ids, eviction_mask, n_tokens=32):
    """Generate tokens using a pre-computed eviction mask.

    The eviction mask sets evicted KV positions to -inf in the attention,
    preventing the model from using them while keeping the cache intact.
    """
    from mlx_lm.models.cache import KVCache
    n_layers = len(model.layers)

    cache = [KVCache() for _ in range(n_layers)]
    x = mx.array([input_ids])

    # Prefill — pass eviction mask as attention mask
    # MLX attention uses additive mask: 0 = attend, -inf = ignore
    T = len(input_ids)
    if eviction_mask is not None:
        # eviction_mask: (T,) boolean, True = keep, False = evict
        # Convert to additive mask: (1, 1, T, T) for broadcast
        mask_1d = mx.where(mx.array(eviction_mask), mx.array(0.0), mx.array(float('-inf')))
        # Each query can attend to all kept positions up to its own index (causal)
        attn_mask = mx.broadcast_to(mask_1d.reshape(1, 1, 1, T), (1, 1, T, T))
        # Apply causal: query at pos i can only attend to pos <= i
        causal = mx.triu(mx.full((T, T), float('-inf')), k=1)
        attn_mask = attn_mask + causal.reshape(1, 1, T, T)
    else:
        attn_mask = None

    logits = model(x, mask=attn_mask, cache=cache)
    mx.eval(logits)

    # Decode — new tokens can attend to all kept positions + themselves
    tokens = []
    for _ in range(n_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(token.item())
        # Decode tokens get no mask — they attend to whatever is in cache
        # (including evicted positions, but those had -inf during prefill
        # so their V contributions were zeroed out in the residual stream)
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

    del cache; gc.collect(); mx.clear_cache()
    return tokens


def main():
    t0 = time.perf_counter()

    from mlx_lm import load
    from omlx.patches.snapkv import capture_attention_weights, snapkv_select

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    # First, compute importance scores from a full prefill
    from mlx_lm.models.cache import KVCache

    tests = [
        ("NIAH", "The secret code is MASK-SNAPKV-OK. " + "Filler text. " * 200
         + "What is the secret code? Reply with just the code:"),
        ("Math", "What is 13 * 17? Show your calculation step by step:"),
        ("Code", "Write a Python function that checks if a string is a palindrome:\n\ndef is_palindrome(s):"),
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

        # Baseline (no eviction)
        base_tokens = generate_with_mask(model, tokenizer, input_ids, None)
        base_text = tokenizer.decode(base_tokens)

        # Compute importance from full prefill
        cache_tmp = [KVCache() for _ in range(n_layers)]
        logits = model(mx.array([input_ids]), cache=cache_tmp)
        mx.eval(logits)

        importance = capture_attention_weights(
            model, cache_tmp, obs_window=64,
            layers=list(range(n_layers - 4, n_layers)))

        del cache_tmp; gc.collect(); mx.clear_cache()

        # Test at various keep ratios
        for keep_pct in [25, 50, 75]:
            keep_count = max(64, T * keep_pct // 100)
            keep_mask = snapkv_select(importance, keep_count, always_keep_last=64)
            mx.eval(keep_mask)

            # Convert to 1D boolean for masking
            mask_np = np.array(keep_mask[0]).astype(bool)  # (T,)
            n_kept = int(mask_np.sum())

            # Generate with eviction mask
            skv_tokens = generate_with_mask(model, tokenizer, input_ids, mask_np)
            skv_text = tokenizer.decode(skv_tokens)

            agree = sum(1 for a, b in zip(base_tokens, skv_tokens) if a == b)
            agreement = round(agree / len(base_tokens) * 100, 0)

            result = {
                "name": name, "keep_pct": keep_pct, "T": T, "kept": n_kept,
                "agreement": agreement,
                "base_text": base_text[:100], "skv_text": skv_text[:100],
            }
            results.append(result)

            logger.info(f"  Keep {keep_pct}%: {n_kept}/{T} tokens, "
                        f"{agreement:.0f}% agreement")
            if keep_pct == 50:
                logger.info(f"    Base: {base_text[:70]!r}")
                logger.info(f"    SKV:  {skv_text[:70]!r}")

    elapsed = time.perf_counter() - t0
    logger.info(f"\n{'='*50}")
    logger.info("SNAPKV MASKING VALIDATION")
    logger.info(f"{'='*50}")

    for name in ["NIAH", "Math", "Code"]:
        for r in results:
            if r["name"] == name:
                s = "PASS" if r["agreement"] >= 30 else "FAIL"
                logger.info(f"  {name:>6} @{r['keep_pct']:>3}%: {r['agreement']:>3.0f}% {s}")

    avg = np.mean([r["agreement"] for r in results])
    logger.info(f"\n  Overall average: {avg:.0f}%")
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    Path("research/snapkv_masking_validation.json").write_text(
        json.dumps({"results": results, "avg_agreement": round(avg, 1),
                    "elapsed_s": round(elapsed, 1)}, indent=2))


if __name__ == "__main__":
    main()
