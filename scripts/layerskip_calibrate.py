#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LayerSkip calibration: per-layer early-exit confidence profiling.

For each layer depth K in [8, 12, 16, 20, 24], measures what fraction
of decode tokens would produce the same argmax if we ran the LM head
on the hidden state at layer K instead of layer 47 (the full model).

This tells us the optimal draft depth for self-speculative decoding:
the depth where the early-exit agreement rate is highest while still
skipping enough layers to be worthwhile.

Usage:
    .venv/bin/python scripts/layerskip_calibrate.py
    .venv/bin/python scripts/layerskip_calibrate.py --draft-depths 8,16,24,32
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


def calibrate(
    model_id: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    draft_depths: list[int] | None = None,
    n_tokens: int = 128,
):
    from mlx_lm import load

    if draft_depths is None:
        draft_depths = [8, 12, 16, 20, 24, 32]

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)

    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)

    n_layers = len(model.layers)
    print(f"Model: {n_layers} layers")
    print(f"Draft depths to test: {draft_depths}")
    print(f"Tokens to generate: {n_tokens}")

    # Get the LM head for early-exit logit computation
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    else:
        lm_head = None
        print("WARNING: No lm_head found — cannot compute early-exit logits")
        return

    # Get the final RMS norm (applied before lm_head)
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    else:
        final_norm = None
        print("WARNING: No final norm found")

    # Prompt for calibration
    prompt = (
        "Write a Python function that implements a binary search tree "
        "with insert, search, and delete operations. Include type hints "
        "and docstrings."
    )
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    tokens = tokenizer.encode(chat_prompt)

    # Prefill
    print(f"\nPrefilling {len(tokens)} tokens...")
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in range(n_layers)]
    x = mx.array([tokens])
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Generate tokens and measure per-depth agreement
    # For each token, we need intermediate hidden states at each draft depth
    # We can't easily get these through the standard forward pass, so we'll
    # use a layer-by-layer forward approach

    results_per_depth = {d: {"agree": 0, "total": 0, "tokens": []} for d in draft_depths}

    # Hook: capture hidden states at draft depths during normal forward
    # by patching the model's layer list to store intermediates
    draft_depth_set = set(draft_depths)
    captured_hiddens = {}  # depth → hidden state

    original_model_call = model.model.__class__.__call__

    def capturing_model_call(self, inputs, cache=None, **kwargs):
        h = self.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = mx.nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            h = layer(h, mask=mask, cache=c)

            if (i + 1) in draft_depth_set:
                captured_hiddens[i + 1] = h

        return self.norm(h)

    model.model.__class__.__call__ = capturing_model_call

    print(f"Generating {n_tokens} tokens with per-layer exit analysis...")

    for tok_idx in range(n_tokens):
        # Pick next token from current logits
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        # Forward next_token — capturing_model_call stores intermediate hiddens
        captured_hiddens.clear()
        x_next = next_token.reshape(1, 1)
        logits = model(x_next, cache=cache)
        mx.eval(logits)

        # The full model's prediction for the NEXT position
        full_next = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(full_next)
        full_next_id = full_next.item()

        # Check: would early-exit at each depth produce the same next token?
        for depth in draft_depths:
            if depth not in captured_hiddens:
                continue
            h_early = captured_hiddens[depth]
            h_normed = final_norm(h_early) if final_norm else h_early
            early_logits = lm_head(h_normed)
            early_token = mx.argmax(early_logits[:, -1, :], axis=-1)
            mx.eval(early_token)
            early_token_id = early_token.item()

            agrees = early_token_id == full_next_id
            results_per_depth[depth]["agree"] += int(agrees)
            results_per_depth[depth]["total"] += 1

        if (tok_idx + 1) % 32 == 0:
            rates = {d: f"{r['agree']}/{r['total']} ({r['agree']/max(r['total'],1)*100:.0f}%)"
                     for d, r in results_per_depth.items() if r['total'] > 0}
            print(f"  [{tok_idx+1}/{n_tokens}] agreement: {rates}")

    # Summary
    print(f"\n{'=' * 65}")
    print(f"LAYERSKIP CALIBRATION RESULTS (context={len(tokens)}, gen={n_tokens})")
    print(f"{'=' * 65}")
    print(f"{'Depth':>6} {'Layers skipped':>15} {'Agreement':>12} {'Speedup est':>12}")
    print("-" * 50)

    best_depth = None
    best_score = 0

    for depth in sorted(draft_depths):
        r = results_per_depth[depth]
        if r["total"] == 0:
            continue
        rate = r["agree"] / r["total"]
        layers_skipped = n_layers - depth
        # Speedup estimate: agreement_rate * (layers_skipped / total_layers)
        # If agreement is high, we skip layers_skipped layers on accepted tokens
        speedup = 1.0 / (1.0 - rate * layers_skipped / n_layers) if rate > 0 else 1.0
        speedup = min(speedup, n_layers / depth)  # Can't exceed depth ratio

        print(f"  {depth:>4}   {layers_skipped:>13}   {rate:>10.0%}   {speedup:>10.2f}x")

        score = rate * layers_skipped  # Higher = better (high agreement + many skipped)
        if score > best_score:
            best_score = score
            best_depth = depth

    if best_depth is not None:
        print(f"\nBest draft depth: {best_depth} "
              f"(agreement {results_per_depth[best_depth]['agree']}/{results_per_depth[best_depth]['total']}, "
              f"skips {n_layers - best_depth} layers)")
    else:
        print("\nNo viable draft depth found (0% agreement at all depths)")

    # Save results
    output = {
        "model": model_id,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "context_len": len(tokens),
        "timestamp": datetime.now().isoformat(),
        "best_draft_depth": best_depth,
        "results": {
            str(d): {
                "depth": d,
                "agree": r["agree"],
                "total": r["total"],
                "rate": round(r["agree"] / max(r["total"], 1), 4),
            }
            for d, r in results_per_depth.items()
        },
    }

    out_dir = Path("omlx/patches/layerskip_thresholds")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_id.split("/")[-1].lower().replace("-", "_")
    out_path = out_dir / f"{model_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    # Restore original model forward
    model.model.__class__.__call__ = original_model_call

    del cache, logits
    gc.collect()
    mx.clear_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description="LayerSkip calibration")
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    parser.add_argument("--draft-depths", type=str, default="8,12,16,20,24,32",
                        help="Comma-separated draft depths to test")
    parser.add_argument("--n-tokens", type=int, default=128)
    args = parser.parse_args()

    depths = [int(d) for d in args.draft_depths.split(",")]
    calibrate(args.model, depths, args.n_tokens)


if __name__ == "__main__":
    main()
