# SPDX-License-Identifier: Apache-2.0
"""Hybrid SSM+attention layer introspection helpers.

Tools that monkey-patch ``scaled_dot_product_attention`` for per-layer
dispatch (DuoAttention calibration, MInference calibration, TTT-Linear
capture) need to know which slots in ``model.layers`` actually fire SDPA.
On dense Qwen3-Coder this is every layer; on hybrid Qwen3.6 only ~25%
of layers are attention.

Pre-migration code did ``layer_counter[0] % len(model.layers)`` which
silently misses on hybrid models — only the attention calls fire SDPA,
so the counter never reaches ``len(model.layers)`` and the modulo never
wraps around to address the higher-numbered slots correctly. The fix is
to thread the **attention-layer position** (0..n_attn-1) instead of
the model-layer index (0..n_layers-1).
"""

from __future__ import annotations


def attention_layer_indices(model) -> list[int]:
    """Return the sorted list of indices in ``model.layers`` whose layer
    has a non-None ``self_attn`` module.

    Examples
    --------
    Dense Qwen3-Coder (every layer is attention)::

        attention_layer_indices(qwen3_coder)
        # [0, 1, 2, ..., 47]

    Hybrid Qwen3.6 (every-4th-layer attention, full_attention_interval=4)::

        attention_layer_indices(qwen3_6)
        # [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    Use this to derive the modulo used in capturing-SDPA hooks:

        attn_indices = attention_layer_indices(model)
        target_call_idx = attn_indices.index(policy_layer_idx)
        # In the hook:
        #     cur_call = layer_counter[0] % len(attn_indices)
        #     if cur_call == target_call_idx: capture()
    """
    indices = []
    for i, layer in enumerate(model.layers):
        if getattr(layer, "self_attn", None) is not None:
            indices.append(i)
    return indices
