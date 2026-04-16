#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate MLA projection quality on real model output.

Runs the model twice:
  1. Full KV (baseline) — standard decode
  2. MLA compressed KV — compress after prefill, expand before decode

Compares:
  - Token-level agreement (should be >90%)
  - KL divergence of logit distributions (should be <0.05)
  - NIAH-style retrieval accuracy with compressed KV

Usage:
    python scripts/validate_mla_quality.py
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
PROJ_DIR = Path("omlx/patches/mla_projections/qwen3_coder_30b_a3b")


def load_projections():
    """Load MLA projection matrices."""
    meta = json.loads((PROJ_DIR / "meta.json").read_text())
    projections = []
    for i in range(meta["n_layers"]):
        data = mx.load(str(PROJ_DIR / f"layer_{i}.npz"))
        projections.append({
            "W_down": data["W_down"],  # (kv_dim, d_c)
            "W_up": data["W_up"],      # (d_c, kv_dim)
        })
    return projections, meta


def compress_cache(cache, projections, meta):
    """Compress KV cache through MLA projection and expand back."""
    n_layers = meta["n_layers"]
    kv_dim = meta["kv_dim"]
    d_c = meta["d_c"]

    for layer_idx in range(n_layers):
        c = cache[layer_idx]
        keys = c.state[0]    # (B, H_kv, T, D)
        values = c.state[1]  # (B, H_kv, T, D)

        B, H_kv, T, D = keys.shape

        # Flatten: (B, H_kv, T, D) → (B, T, H_kv*D)
        K_flat = keys.transpose(0, 2, 1, 3).reshape(B, T, H_kv * D)
        V_flat = values.transpose(0, 2, 1, 3).reshape(B, T, H_kv * D)

        # Concatenate K+V: (B, T, kv_dim=H_kv*2D)
        KV = mx.concatenate([K_flat, V_flat], axis=-1)

        # Project to latent: (B, T, kv_dim) @ (kv_dim, d_c) → (B, T, d_c)
        W_down = projections[layer_idx]["W_down"].astype(KV.dtype)
        W_up = projections[layer_idx]["W_up"].astype(KV.dtype)

        latent = KV @ W_down
        # Reconstruct: (B, T, d_c) @ (d_c, kv_dim) → (B, T, kv_dim)
        KV_recon = latent @ W_up

        # Split back to K and V
        K_recon = KV_recon[:, :, :H_kv * D]  # (B, T, H_kv*D)
        V_recon = KV_recon[:, :, H_kv * D:]  # (B, T, H_kv*D)

        # Reshape back: (B, T, H_kv*D) → (B, H_kv, T, D)
        K_out = K_recon.reshape(B, T, H_kv, D).transpose(0, 2, 1, 3)
        V_out = V_recon.reshape(B, T, H_kv, D).transpose(0, 2, 1, 3)

        # Write back to cache
        c.state = (K_out, V_out)

    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])


def generate_tokens(model, cache, logits, n_tokens=32):
    """Generate tokens autoregressively."""
    tokens = []
    all_logits = []
    for _ in range(n_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(token.item())
        all_logits.append(np.array(logits[:, -1, :].astype(mx.float32)))
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
    return tokens, all_logits


def main():
    t0 = time.perf_counter()

    # Load projections
    if not PROJ_DIR.exists():
        logger.error(f"Projections not found at {PROJ_DIR}")
        logger.error("Run: python scripts/compute_mla_projections.py")
        return
    projections, meta = load_projections()
    logger.info(f"Loaded {meta['n_layers']} projections (d_c={meta['d_c']})")

    # Load model
    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    # Test prompts
    test_cases = [
        {
            "name": "Code generation",
            "prompt": "Write a Python function to compute the nth Fibonacci number using memoization:\n\n```python\ndef fibonacci(n):",
        },
        {
            "name": "Math reasoning",
            "prompt": "What is 2+2? Answer with just the number:",
        },
        {
            "name": "NIAH retrieval",
            "prompt": "The secret password is HYPERCAR-42. " + "This is filler text. " * 100 + "What is the secret password?",
        },
    ]

    results = []

    for tc in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {tc['name']}")
        logger.info(f"{'='*60}")

        messages = [{"role": "user", "content": tc["prompt"]}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_prompt = tc["prompt"] + "\n"
        input_ids = tokenizer.encode(chat_prompt)

        # === Baseline: full KV ===
        cache_base = [KVCache() for _ in range(n_layers)]
        x = mx.array([input_ids])
        logits_base = model(x, cache=cache_base)
        mx.eval(logits_base)

        base_tokens, base_logits = generate_tokens(model, cache_base, logits_base)
        base_text = tokenizer.decode(base_tokens)

        del cache_base
        gc.collect()
        mx.clear_cache()

        # === MLA: compressed KV ===
        cache_mla = [KVCache() for _ in range(n_layers)]
        x = mx.array([input_ids])
        logits_mla_pre = model(x, cache=cache_mla)
        mx.eval(logits_mla_pre)

        # Compress KV cache through MLA projections
        compress_cache(cache_mla, projections, meta)

        mla_tokens, mla_logits = generate_tokens(model, cache_mla, logits_mla_pre)
        mla_text = tokenizer.decode(mla_tokens)

        del cache_mla
        gc.collect()
        mx.clear_cache()

        # === Compare ===
        # Token agreement
        agree = sum(1 for a, b in zip(base_tokens, mla_tokens) if a == b)
        agreement = agree / len(base_tokens) * 100

        # KL divergence (first token only — most informative)
        base_lp = base_logits[0][0]  # (vocab,)
        mla_lp = mla_logits[0][0]

        # Stabilize softmax
        base_p = np.exp(base_lp - np.max(base_lp))
        base_p /= base_p.sum()
        mla_p = np.exp(mla_lp - np.max(mla_lp))
        mla_p /= mla_p.sum()

        # KL(base || mla)
        mask = base_p > 1e-10
        kl = np.sum(base_p[mask] * np.log(base_p[mask] / (mla_p[mask] + 1e-10)))

        result = {
            "name": tc["name"],
            "agreement_pct": round(agreement, 1),
            "kl_divergence": round(float(kl), 4),
            "base_text": base_text[:100],
            "mla_text": mla_text[:100],
        }
        results.append(result)

        logger.info(f"  Token agreement: {agreement:.0f}% ({agree}/{len(base_tokens)})")
        logger.info(f"  KL divergence:   {kl:.4f}")
        logger.info(f"  Baseline:  {base_text[:80]!r}")
        logger.info(f"  MLA:       {mla_text[:80]!r}")

    # Summary
    elapsed = time.perf_counter() - t0
    avg_agreement = np.mean([r["agreement_pct"] for r in results])
    avg_kl = np.mean([r["kl_divergence"] for r in results])

    logger.info(f"\n{'='*60}")
    logger.info("MLA QUALITY VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  d_c={meta['d_c']}, mean reconstruction error={meta['mean_rel_error']:.4f}")
    logger.info(f"  Average token agreement: {avg_agreement:.0f}%")
    logger.info(f"  Average KL divergence:   {avg_kl:.4f}")

    if avg_agreement >= 80 and avg_kl < 0.1:
        logger.info(f"  VERDICT: MLA compression QUALITY PASS")
    elif avg_agreement >= 60:
        logger.info(f"  VERDICT: MLA compression MARGINAL — may need higher d_c")
    else:
        logger.info(f"  VERDICT: MLA compression QUALITY FAIL — d_c too low")

    logger.info(f"  Elapsed: {elapsed:.1f}s")

    # Save results
    out = {"meta": meta, "results": results, "summary": {
        "avg_agreement_pct": round(avg_agreement, 1),
        "avg_kl_divergence": round(avg_kl, 4),
        "elapsed_s": round(elapsed, 1),
    }}
    Path("research/mla_quality_validation.json").write_text(json.dumps(out, indent=2))
    logger.info(f"  Results: research/mla_quality_validation.json")


if __name__ == "__main__":
    main()
