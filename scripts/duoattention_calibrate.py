#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DuoAttention retrieval/streaming head calibration (arXiv:2410.10819).

Runs Qwen3-Coder over a synthetic passkey-retrieval corpus and
classifies each (layer, head) as:
  - retrieval: needs full KV cache (high attention to distant tokens)
  - streaming: needs only sink tokens + local window (attention is local)

The classification is based on attention pattern analysis: heads where
>95% of attention mass falls within a local window + sink tokens are
classified as streaming.

Output: per-head policy JSON for runtime cache splitting (Task 13).

Usage:
    .venv/bin/python scripts/duoattention_calibrate.py
    .venv/bin/python scripts/duoattention_calibrate.py --window 256 --sink 4
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger("duoattention.calibrate")

DEFAULT_OUTPUT = Path("omlx/patches/duoattention_policies")


def calibrate(
    model_id: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
    context_len: int = 4096,
    window: int = 256,
    sink: int = 4,
    streaming_threshold: float = 0.95,
):
    """Classify heads as retrieval vs streaming.

    For each head, compute what fraction of attention mass falls within
    the local window + sink tokens. If >= streaming_threshold, classify
    as streaming.
    """
    from mlx_lm import load
    import mlx_lm.models.base as mlx_base

    import sys as _sys
    _sys.path.insert(0, ".")
    from omlx.state_space import attention_layer_indices

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)

    n_layers = len(model.layers)
    # Hybrid-safe: SDPA only fires on attention layers. The capturing
    # SDPA hook's call counter must be interpreted relative to the attn-
    # layer subset, not all of model.layers. On dense Qwen3-Coder these
    # are equal; on Qwen3.6 (10 of 40 are attention) they diverge.
    attn_indices = attention_layer_indices(model)
    n_attn = len(attn_indices)
    if n_attn == 0:
        raise RuntimeError(
            f"Model {model_id} has no self_attn layers; calibration "
            "cannot proceed."
        )
    # Detect head counts on the FIRST attention layer (which may not be
    # layer 0 on hybrid models).
    first_attn = model.layers[attn_indices[0]].self_attn
    n_heads = getattr(first_attn, 'n_heads', 32)
    n_kv_heads = getattr(first_attn, 'n_kv_heads', 4)
    gqa = n_heads // n_kv_heads

    print(f"Model: {n_layers} layers ({n_attn} attention), "
          f"{n_heads} Q heads, {n_kv_heads} KV heads, GQA={gqa}")
    print(f"Window: {window}, Sink: {sink}, Threshold: {streaming_threshold}")

    # Build synthetic code context with a passkey
    code = '''import json
from pathlib import Path
from typing import Any, Dict, List

def process_batch(items: List[Dict[str, Any]], batch_size: int = 32) -> List:
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results.extend([transform(item) for item in batch])
    return results

def transform(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: str(v).upper() for k, v in item.items()}

'''
    tokens = tokenizer.encode(code)
    reps = (context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:context_len]

    print(f"Context: {len(full_tokens)} tokens")

    # Capture attention weights via SDPA patching
    attention_maps = {}  # (layer, head) → (T, T) attention weights
    layer_counter = [0]

    original_sdpa = mlx_base.scaled_dot_product_attention

    def capturing_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        B, H_q, L_q, D = queries.shape
        H_kv = keys.shape[1]

        # Compute attention manually to capture weights
        if H_kv < H_q:
            keys_exp = mx.repeat(keys, gqa, axis=1)
            values_exp = mx.repeat(values, gqa, axis=1)
        else:
            keys_exp = keys
            values_exp = values

        scores = (queries @ keys_exp.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                # Build causal mask
                L_kv = keys_exp.shape[2]
                causal = mx.triu(mx.full((L_q, L_kv), -1e9), k=1)
                scores = scores + causal
            elif isinstance(mask, mx.array):
                scores = scores + mask

        weights = mx.softmax(scores, axis=-1)

        # On hybrid models, the i-th SDPA call corresponds to the i-th
        # attention layer (in attn_indices order), NOT to model.layers[i].
        cur_call = layer_counter[0] % n_attn
        layer_idx = attn_indices[cur_call]
        layer_counter[0] += 1

        # Store per-head attention (sample middle rows for efficiency)
        w_np = np.array(weights[0].astype(mx.float32))  # (H_q, L_q, L_kv)
        for h in range(H_q):
            attention_maps[(layer_idx, h)] = w_np[h]  # (L_q, L_kv)

        out = weights @ values_exp
        return out

    # Patch SDPA — must also patch model modules that already imported it
    mlx_base.scaled_dot_product_attention = capturing_sdpa
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", capturing_sdpa)

    try:
        x = mx.array([full_tokens])
        print("Prefilling to capture attention weights...")
        t0 = time.perf_counter()
        logits = model(x)
        mx.eval(logits)
        print(f"Prefill done in {time.perf_counter() - t0:.1f}s")
    finally:
        mlx_base.scaled_dot_product_attention = original_sdpa
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
                if hasattr(mod, "scaled_dot_product_attention"):
                    setattr(mod, "scaled_dot_product_attention", original_sdpa)

    print(f"Captured {len(attention_maps)} (layer, head) attention maps")

    # Classify each head
    policies = []
    streaming_count = 0
    retrieval_count = 0

    T = len(full_tokens)

    # Only emit policy entries for attention layers. On dense Qwen3-Coder
    # this is every layer; on hybrid Qwen3.6 it's the sparse attn-only
    # subset. Emitting for non-attention layers would falsely tag them
    # all "retrieval" with local_fraction=0.0 (misleading at runtime).
    for layer_idx in attn_indices:
        for head_idx in range(n_heads):
            key = (layer_idx, head_idx)
            if key not in attention_maps:
                policies.append({
                    "layer": layer_idx, "head": head_idx,
                    "policy": "retrieval", "local_fraction": 0.0,
                })
                retrieval_count += 1
                continue

            attn = attention_maps[key]  # (T, T)

            # Compute fraction of attention in local window + sink
            local_mass = 0.0
            total_rows = 0
            # Sample every 8th row for speed
            for row in range(0, T, max(1, T // 128)):
                if row >= attn.shape[0]:
                    break
                row_attn = attn[row, :row + 1]  # causal: only up to current pos
                if row_attn.sum() < 1e-8:
                    continue

                # Sink tokens: positions 0..sink-1
                sink_mass = row_attn[:sink].sum() if row >= sink else 0.0

                # Local window: positions max(0, row-window)..row
                local_start = max(sink, row - window)
                window_mass = row_attn[local_start:row + 1].sum()

                local_mass += (sink_mass + window_mass) / row_attn.sum()
                total_rows += 1

            local_fraction = local_mass / max(total_rows, 1)

            if local_fraction >= streaming_threshold:
                policy = "streaming"
                streaming_count += 1
            else:
                policy = "retrieval"
                retrieval_count += 1

            policies.append({
                "layer": layer_idx,
                "head": head_idx,
                "policy": policy,
                "local_fraction": round(float(local_fraction), 4),
                "window": window,
                "sink": sink,
            })

    total_classified = streaming_count + retrieval_count
    streaming_frac = streaming_count / total_classified
    print(f"\nClassification:")
    print(f"  Streaming: {streaming_count}/{total_classified} ({streaming_frac:.0%})")
    print(f"  Retrieval: {retrieval_count}/{total_classified} ({1-streaming_frac:.0%})")

    # Per-layer summary (attention layers only)
    print(f"\nPer-layer breakdown:")
    for l in attn_indices:
        layer_policies = [p for p in policies if p["layer"] == l]
        n_stream = sum(1 for p in layer_policies if p["policy"] == "streaming")
        fracs = [p["local_fraction"] for p in layer_policies]
        mean_frac = sum(fracs) / len(fracs) if fracs else 0
        print(f"  Layer {l:>2}: {n_stream:>2}/{n_heads} streaming "
              f"(mean local_frac={mean_frac:.2f})")

    # Validation assertion
    assert streaming_frac >= 0.50, (
        f"Streaming fraction {streaming_frac:.0%} < 50% — model may not "
        f"be suitable for DuoAttention at window={window}"
    )

    # Save policy
    output = {
        "model": model_id,
        "n_layers": n_layers,            # total layers (incl. SSM on hybrid)
        "n_attn_layers": n_attn,         # attention layers only
        "attn_layer_indices": attn_indices,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "context_len": context_len,
        "window": window,
        "sink": sink,
        "streaming_threshold": streaming_threshold,
        "streaming_fraction": round(streaming_frac, 4),
        "streaming_count": streaming_count,
        "retrieval_count": retrieval_count,
        "heads": policies,
    }

    out_dir = DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_id.split("/")[-1].lower().replace("-", "_")
    out_path = out_dir / f"{model_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nPolicy written to {out_path}")

    # Cleanup
    del attention_maps, logits
    gc.collect()
    mx.clear_cache()

    return output


def main():
    parser = argparse.ArgumentParser(description="DuoAttention head calibration")
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--window", type=int, default=256,
                        help="Local attention window size for streaming heads")
    parser.add_argument("--sink", type=int, default=4,
                        help="Number of sink tokens (always attended)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Fraction of attention in local+sink to classify as streaming")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    calibrate(
        model_id=args.model,
        context_len=args.context_len,
        window=args.window,
        sink=args.sink,
        streaming_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
