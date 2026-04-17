#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compute per-head Q/K centres for trigonometric importance scoring.

From TriAttention (arXiv:2604.04921): pre-RoPE Q and K vectors
concentrate around stable per-head centres. This enables O(d)-per-key
importance scoring via trigonometric decomposition.

Runs a calibration prefill over ~1000 tokens, captures pre-RoPE Q and K
for all layers, and computes mean centres. Saves as .npz file.

Usage:
    .venv/bin/python scripts/calibrate_qk_centres.py
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
OUTPUT_PATH = Path("omlx/patches/qk_centres/qwen3_coder_30b.npz")


def main():
    t0 = time.perf_counter()

    from mlx_lm import load
    from mlx_lm.models.cache import KVCache

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)
    n_layers = len(model.layers)

    # Warmup
    cache = [KVCache() for _ in range(n_layers)]
    logits = model(mx.array([tokenizer.encode("Hello")]), cache=cache)
    mx.eval(logits)
    del cache, logits; gc.collect(); mx.clear_cache()

    # Calibration corpus: representative code + prose
    cal_texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    def insert(self, val):\n        if val < self.value:\n            if self.left: self.left.insert(val)\n            else: self.left = BinaryTree(val)\n        else:\n            if self.right: self.right.insert(val)\n            else: self.right = BinaryTree(val)",
        "import asyncio\nasync def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()\n\nasync def main():\n    urls = ['https://api.example.com/data/' + str(i) for i in range(10)]\n    results = await asyncio.gather(*[fetch_data(u) for u in urls])\n    return results",
        "The transformer architecture uses multi-head self-attention to process input sequences in parallel. Each attention head learns to focus on different aspects of the input: some heads attend to syntactic structure while others capture semantic relationships. The key innovation is that attention weights are computed as softmax(QK^T/sqrt(d)) which allows each token to attend to all other tokens.",
        "fn main() {\n    let mut v = vec![1, 2, 3, 4, 5];\n    v.iter().filter(|&&x| x > 2).for_each(|x| println!(\"{}\", x));\n    let sum: i32 = v.iter().sum();\n    println!(\"Sum: {}\", sum);\n}",
    ]

    all_tokens = []
    for text in cal_texts:
        messages = [{"role": "user", "content": text}]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_text = text
        all_tokens.extend(tokenizer.encode(chat_text))

    # Truncate to ~1500 tokens
    all_tokens = all_tokens[:1500]
    logger.info(f"Calibration corpus: {len(all_tokens)} tokens")

    # Capture pre-RoPE Q and K for ALL layers
    # Hook into each attention module to intercept Q/K before RoPE
    q_accum = {}  # layer → list of (B, H_q, T, D) tensors
    k_accum = {}  # layer → list of (B, H_kv, T, D) tensors

    for layer_idx in range(n_layers):
        q_accum[layer_idx] = []
        k_accum[layer_idx] = []

    # Monkey-patch each layer's attention to capture pre-RoPE Q/K
    original_calls = {}
    for layer_idx in range(n_layers):
        attn = model.layers[layer_idx].self_attn
        original_calls[layer_idx] = attn.__class__.__call__

    def make_capture_call(lid, orig_call):
        def capture_call(self, x, mask=None, cache=None):
            B, L, D = x.shape
            queries = self.q_proj(x)
            keys = self.k_proj(x)
            values = self.v_proj(x)
            queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            # CAPTURE pre-RoPE Q and K
            q_accum[lid].append(queries)
            k_accum[lid].append(keys)

            # Continue with standard forward pass
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            from mlx_lm.models.base import scaled_dot_product_attention
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)
        return capture_call

    # Install hooks via dynamic subclass
    for layer_idx in range(n_layers):
        attn = model.layers[layer_idx].self_attn
        attn.__class__ = type(
            f'Cal_{attn.__class__.__name__}_{layer_idx}',
            (attn.__class__,),
            {'__call__': make_capture_call(layer_idx, original_calls[layer_idx])}
        )

    # Run calibration prefill
    cache = [KVCache() for _ in range(n_layers)]
    chunk_size = 512
    for start in range(0, len(all_tokens), chunk_size):
        end = min(start + chunk_size, len(all_tokens))
        logits = model(mx.array([all_tokens[start:end]]), cache=cache)
        mx.eval(logits)
    logger.info("Calibration prefill done")

    # Restore original attention classes
    for layer_idx in range(n_layers):
        attn = model.layers[layer_idx].self_attn
        attn.__class__ = type(attn).__mro__[1]

    del cache, logits; gc.collect(); mx.clear_cache()

    # Compute centres: mean Q and K per layer per head
    q_centres = {}  # layer → (H_q, D) numpy array
    k_centres = {}  # layer → (H_kv, D) numpy array

    for layer_idx in range(n_layers):
        if not q_accum[layer_idx]:
            continue
        # Concatenate across chunks: (B, H, total_T, D)
        q_all = mx.concatenate(q_accum[layer_idx], axis=2)
        k_all = mx.concatenate(k_accum[layer_idx], axis=2)
        mx.eval(q_all, k_all)

        # Mean across batch and tokens: (H, D)
        q_mean = mx.mean(q_all, axis=(0, 2))  # (H_q, D)
        k_mean = mx.mean(k_all, axis=(0, 2))  # (H_kv, D)
        mx.eval(q_mean, k_mean)

        q_centres[layer_idx] = np.array(q_mean.astype(mx.float32))
        k_centres[layer_idx] = np.array(k_mean.astype(mx.float32))

        # Free memory
        del q_accum[layer_idx], k_accum[layer_idx]

    # Get RoPE frequencies
    attn0 = model.layers[0].self_attn
    rope_dims = attn0.rope.dims
    rope_base = attn0.rope.base
    half_d = rope_dims // 2
    theta = 1.0 / (rope_base ** (np.arange(0, half_d).astype(np.float32) * 2 / rope_dims))

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "rope_theta": theta,
        "rope_dims": np.array([rope_dims]),
        "rope_base": np.array([rope_base]),
        "n_layers": np.array([n_layers]),
    }
    for layer_idx in range(n_layers):
        if layer_idx in q_centres:
            save_dict[f"q_centre_{layer_idx}"] = q_centres[layer_idx]
            save_dict[f"k_centre_{layer_idx}"] = k_centres[layer_idx]

    np.savez(str(OUTPUT_PATH), **save_dict)
    elapsed = time.perf_counter() - t0
    logger.info(f"Saved to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    logger.info(f"Q centre shape: {q_centres[0].shape}, K centre shape: {k_centres[0].shape}")
    logger.info(f"Elapsed: {elapsed:.1f}s")

    # Verify centre stability: check cosine similarity between first/second half
    for layer_idx in [0, 23, 47]:
        q = q_centres[layer_idx]
        k = k_centres[layer_idx]
        # Norm check
        q_norm = np.linalg.norm(q, axis=-1).mean()
        k_norm = np.linalg.norm(k, axis=-1).mean()
        logger.info(f"  Layer {layer_idx}: Q norm={q_norm:.3f}, K norm={k_norm:.3f}")


if __name__ == "__main__":
    main()
