# SPDX-License-Identifier: Apache-2.0
"""Monkeypatch granitemoehybrid sanitize() to handle pre-quantized MoE weights.

The default sanitize() in granitemoehybrid.py expects raw input_linear/output_linear
weights (3D fp16) and splits them into gate_proj/up_proj/down_proj for SwitchGLU.

When loading a TQ3.5-converted model, the weights are already split and quantized
into switch_mlp.{gate,up,down}_proj.{weight,scales,biases}. This patch makes
sanitize() skip the MoE transform when the pre-split keys are already present.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_sanitize_patch() -> bool:
    """Monkeypatch granitemoehybrid's sanitize to handle pre-split quantized MoE weights."""
    global _patch_applied
    if _patch_applied:
        return True

    try:
        from mlx_lm.models import granitemoehybrid
        ModelCls = granitemoehybrid.Model
    except ImportError:
        logger.debug("granitemoehybrid not available, skipping sanitize patch")
        return False

    original_sanitize = ModelCls.sanitize

    def _patched_sanitize(self, weights):
        # Check if ANY layer has pre-split SwitchGLU format (TQ3.5 converter output)
        has_switch_mlp = any("switch_mlp.gate_proj" in k for k in weights)

        if has_switch_mlp:
            # TQ3.5 model: MoE weights already split into switch_mlp format.
            # Handle conv1d fixup
            for k, v in weights.items():
                if "conv1d.weight" in k and v.shape[-1] != 1:
                    weights[k] = v.moveaxis(2, 1)

            # Handle any remaining input_linear/output_linear (e.g., fp16 layer 0)
            # by applying the original split logic only for those specific layers
            num_layers = getattr(self.args, 'num_hidden_layers', 0)
            for l in range(num_layers):
                prefix = f"model.layers.{l}.block_sparse_moe"
                input_key = f"{prefix}.input_linear.weight"
                output_key = f"{prefix}.output_linear.weight"

                if input_key in weights:
                    # This layer still has unsplit weights (e.g., fp16 layer)
                    input_weight = weights.pop(input_key)
                    _, expert_hidden, _ = input_weight.shape
                    gate_proj = input_weight[:, :expert_hidden // 2, :]
                    up_proj = input_weight[:, expert_hidden // 2:, :]
                    weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
                    weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_proj

                if output_key in weights:
                    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(output_key)

            # shared_mlp uses input_linear/output_linear directly (no split needed)
            # — the split happens inside __call__ via mx.split()

            logger.info("granitemoehybrid sanitize: TQ3.5 hybrid (pre-split MoE + raw fp16 layers)")
            return weights

        # Fall back to original sanitize for non-TQ3.5 models
        return original_sanitize(self, weights)

    ModelCls.sanitize = _patched_sanitize
    _patch_applied = True
    logger.info("granitemoehybrid sanitize patch applied (TQ3.5 MoE support)")
    return True
