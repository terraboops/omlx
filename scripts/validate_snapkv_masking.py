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


def patch_model_for_eviction_mask(model, eviction_mask_1d):
    """Monkey-patch model to inject eviction mask into attention.

    Patches create_attention_mask in the model's module to merge
    the eviction mask with the standard causal mask.

    Args:
        eviction_mask_1d: (T,) boolean array — True=keep, False=evict.
            Set to None to remove the patch.
    """
    import importlib
    qwen_mod = importlib.import_module("mlx_lm.models.qwen3_moe")
    original_create_mask = qwen_mod.create_attention_mask

    if eviction_mask_1d is None:
        # Remove patch
        if hasattr(qwen_mod, '_original_create_attention_mask'):
            qwen_mod.create_attention_mask = qwen_mod._original_create_attention_mask
        return

    # Save original
    qwen_mod._original_create_attention_mask = original_create_mask

    # Eviction additive mask: 0 for kept, -inf for evicted (bfloat16 for SDPA)
    evict_additive = mx.where(
        mx.array(eviction_mask_1d),
        mx.array(0.0, dtype=mx.bfloat16),
        mx.array(float('-inf'), dtype=mx.bfloat16))

    def patched_create_attention_mask(h, cache=None, **kwargs):
        N = h.shape[1]  # current chunk length

        if N == 1:
            # Decode: single token query attends to cached KV + itself.
            # cache.offset is BEFORE update_and_fetch adds the new token,
            # but SDPA gets keys AFTER update_and_fetch (offset+1 entries).
            # Return mask for offset+1 positions.
            if cache is not None:
                T_after = cache.offset + 1  # will be this many KV entries after update
                T_evict = len(eviction_mask_1d)
                if T_after > T_evict:
                    extra = mx.zeros(T_after - T_evict, dtype=mx.bfloat16)
                    full_mask = mx.concatenate([evict_additive[:T_evict], extra])
                else:
                    full_mask = evict_additive[:T_after]
                return full_mask.reshape(1, 1, 1, T_after)
            return None

        # Prefill: N tokens attending to N tokens (causal + eviction)
        T = N
        if T <= len(eviction_mask_1d):
            # Causal mask (bfloat16 for SDPA compatibility)
            causal = mx.triu(mx.full((T, T), float('-inf'), dtype=mx.bfloat16), k=1)
            # Eviction: broadcast (1, T) to (T, T) — each query position
            # sees the same eviction pattern for key positions
            evict_2d = evict_additive[:T].reshape(1, T)
            combined = causal + evict_2d
            return combined
        else:
            return original_create_mask(h, cache=cache, **kwargs)

    qwen_mod.create_attention_mask = patched_create_attention_mask


def generate_with_mask(model, tokenizer, input_ids, eviction_mask, n_tokens=32):
    """Generate tokens using a pre-computed eviction mask.

    Monkey-patches create_attention_mask to inject eviction mask into
    the model's standard attention pipeline. Cache stays intact.
    """
    from mlx_lm.models.cache import KVCache
    n_layers = len(model.layers)

    # Apply eviction mask patch
    patch_model_for_eviction_mask(model, eviction_mask)

    try:
        cache = [KVCache() for _ in range(n_layers)]
        x = mx.array([input_ids])
        logits = model(x, cache=cache)
        mx.eval(logits)

        # Keep eviction mask active during decode — new tokens must also
        # not attend to evicted positions (their V values are corrupted
        # from the masked prefill pass)
        tokens = []
        for _ in range(n_tokens):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tokens.append(token.item())
            logits = model(token.reshape(1, 1), cache=cache)
            mx.eval(logits)

        del cache; gc.collect(); mx.clear_cache()
        return tokens
    finally:
        # Always clean up patch
        patch_model_for_eviction_mask(model, None)


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
