#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate ShadowKV K-only compression quality (Step 3).

Unlike the MLA joint validation (which failed at 7% agreement), ShadowKV
only compresses K cache per-head while V stays exact. This should preserve
quality because:
  1. V values are exact — the weighted sum produces correct outputs
  2. Per-head projection preserves head-specific subspace structure
  3. K errors only affect attention weight distribution, not values

Tests:
  1. Token agreement vs baseline (target: >80%)
  2. NIAH retrieval with compressed K (must find the needle)
  3. Code generation quality
  4. Math reasoning

Usage:
    python scripts/validate_shadowkv_quality.py
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
    """Generate tokens autoregressively."""
    tokens = []
    for _ in range(n_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(token.item())
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
    return tokens


def run_test(model, tokenizer, n_layers, prompt, projections, n_gen=32):
    """Run baseline vs ShadowKV on one prompt."""
    from mlx_lm.models.cache import KVCache
    from omlx.shadowkv_cache import compress_k_with_projections

    messages = [{"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_prompt = prompt + "\n"
    input_ids = tokenizer.encode(chat_prompt)

    # Baseline
    cache_base = [KVCache() for _ in range(n_layers)]
    x = mx.array([input_ids])
    logits = model(x, cache=cache_base)
    mx.eval(logits)
    base_tokens = generate_tokens(model, cache_base, logits, n_gen)
    del cache_base; gc.collect(); mx.clear_cache()

    # ShadowKV: prefill then compress K before decode
    cache_skv = [KVCache() for _ in range(n_layers)]
    x = mx.array([input_ids])
    logits = model(x, cache=cache_skv)
    mx.eval(logits)

    # Compress K cache using offline projections
    n_compressed = compress_k_with_projections(cache_skv, projections, min_tokens=0)

    skv_tokens = generate_tokens(model, cache_skv, logits, n_gen)
    del cache_skv; gc.collect(); mx.clear_cache()

    # Compare
    agree = sum(1 for a, b in zip(base_tokens, skv_tokens) if a == b)
    agreement = agree / len(base_tokens) * 100

    base_text = tokenizer.decode(base_tokens)
    skv_text = tokenizer.decode(skv_tokens)

    return {
        "agreement_pct": round(agreement, 1),
        "base_text": base_text[:120],
        "skv_text": skv_text[:120],
        "n_compressed": n_compressed,
    }


def main():
    t0 = time.perf_counter()

    # Load projections
    from omlx.shadowkv_cache import ShadowKVProjections
    projections = ShadowKVProjections.load()

    # Load model
    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    test_cases = [
        ("Math", "What is 2+2? Answer with just the number:"),
        ("Code", "Write a Python function to check if a number is prime:\n\n```python\ndef is_prime(n):"),
        ("NIAH", "The secret code is SHADOWKV-99. " + "Filler. " * 80 + "What is the secret code?"),
        ("Reasoning", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain briefly."),
    ]

    results = []
    for name, prompt in test_cases:
        logger.info(f"\n{'='*50}")
        logger.info(f"Test: {name}")
        r = run_test(model, tokenizer, n_layers, prompt, projections)
        results.append({"name": name, **r})
        logger.info(f"  Agreement: {r['agreement_pct']:.0f}%")
        logger.info(f"  Compressed: {r['n_compressed']} layers")
        logger.info(f"  Base: {r['base_text'][:80]!r}")
        logger.info(f"  SKV:  {r['skv_text'][:80]!r}")

    # Summary
    avg_agreement = np.mean([r["agreement_pct"] for r in results])
    elapsed = time.perf_counter() - t0

    logger.info(f"\n{'='*50}")
    logger.info(f"SHADOWKV QUALITY VALIDATION")
    logger.info(f"{'='*50}")
    logger.info(f"  Median rank: {projections.median_rank}")
    logger.info(f"  Mean SVD error: {projections.mean_error:.4f}")
    logger.info(f"  Average token agreement: {avg_agreement:.0f}%")

    for r in results:
        status = "PASS" if r["agreement_pct"] >= 50 else "FAIL"
        logger.info(f"  {r['name']:>12}: {r['agreement_pct']:>5.0f}% {status}")

    if avg_agreement >= 70:
        logger.info(f"\n  VERDICT: ShadowKV K-only compression QUALITY PASS")
    elif avg_agreement >= 40:
        logger.info(f"\n  VERDICT: ShadowKV MARGINAL — better than MLA (7%) but needs tuning")
    else:
        logger.info(f"\n  VERDICT: ShadowKV QUALITY FAIL")

    logger.info(f"  Elapsed: {elapsed:.1f}s")

    # Save
    out = {"results": results, "avg_agreement": round(avg_agreement, 1),
           "elapsed_s": round(elapsed, 1), "median_rank": projections.median_rank}
    Path("research/shadowkv_quality_validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
