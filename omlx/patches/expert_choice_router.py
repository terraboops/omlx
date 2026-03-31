# SPDX-License-Identifier: Apache-2.0
"""Expert-Choice MoE routing: experts pick tokens instead of tokens picking experts.

Standard MoE uses token-choice routing (each token picks top-K experts), which
causes load imbalance — some experts overloaded, others idle.

Expert-Choice flips this: each expert selects a fixed capacity of tokens,
guaranteeing balanced utilization across all Metal GPU threads.

Output format is compatible with SwitchGLU (same index shape as GraniteMoeTopKGating).
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_patch_applied = False


class ExpertChoiceRouter(nn.Module):
    """Expert-Choice routing: experts pick their preferred tokens.

    Produces output in the same format as GraniteMoeTopKGating:
      - top_k_idx: (B, L, top_k) expert indices per token
      - top_k_gates: (B, L, top_k) routing weights per token

    Args:
        input_size: Hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token (for output shape compatibility).
        capacity_factor: Overprovisioning factor (1.0 = exact, 1.2 = 20% extra).
    """

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array):
        """Expert-choice routing with token-indexed output.

        Args:
            hidden_states: (B, L, D) or (B*L, D)

        Returns:
            top_k_idx: Expert indices per token, same shape as token-choice.
            top_k_gates: Routing weights per token, same shape as token-choice.
        """
        orig_shape = hidden_states.shape
        if hidden_states.ndim == 3:
            B, L, D = hidden_states.shape
            flat = hidden_states.reshape(B * L, D)
        else:
            flat = hidden_states
            B, L = 1, flat.shape[0]

        num_tokens = flat.shape[0]

        # Compute routing scores: (num_tokens, num_experts)
        scores = self.layer(flat)

        # Expert-choice: each expert picks top-C tokens
        capacity = max(1, int((num_tokens * self.top_k / self.num_experts) * self.capacity_factor))

        # Transpose to (num_experts, num_tokens) so each expert is a row
        expert_scores = scores.T  # (E, T)

        # Each expert picks its top-capacity tokens
        if capacity >= num_tokens:
            # All tokens selected by all experts — fall back to token-choice
            top_k_idx = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k:]
            top_k_logits = mx.take_along_axis(scores, top_k_idx, axis=-1)
            top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1)
            if hidden_states.ndim == 3:
                top_k_idx = top_k_idx.reshape(B, L, self.top_k)
                top_k_gates = top_k_gates.reshape(B, L, self.top_k)
            return top_k_idx, top_k_gates

        # Expert picks top tokens: (E, capacity)
        top_token_idx = mx.argpartition(-expert_scores, kth=capacity, axis=-1)[..., :capacity]
        top_token_scores = mx.take_along_axis(expert_scores, top_token_idx, axis=-1)

        # Build per-token expert assignments from expert selections
        # For each token: find which experts selected it, pick top_k highest-scoring
        # Initialize with -1 (no expert) and 0.0 (no weight)
        token_expert_scores = mx.full((num_tokens, self.num_experts), -1e9)

        # Scatter expert scores back to token-indexed format
        for e in range(self.num_experts):
            # For expert e, the tokens it selected are top_token_idx[e]
            # with scores top_token_scores[e]
            indices = top_token_idx[e]  # (capacity,)
            e_scores = top_token_scores[e]  # (capacity,)
            # Use scatter to place scores
            token_expert_scores = token_expert_scores.at[indices, e].maximum(e_scores)

        # Now each token has scores for experts that selected it
        # Pick top_k experts per token (same as token-choice output format)
        top_k_idx = mx.argpartition(token_expert_scores, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        top_k_logits = mx.take_along_axis(token_expert_scores, top_k_idx, axis=-1)

        # Replace -1e9 (unselected) with 0 before softmax
        top_k_logits = mx.where(top_k_logits > -1e8, top_k_logits, mx.zeros_like(top_k_logits))
        top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1)

        if hidden_states.ndim == 3:
            top_k_idx = top_k_idx.reshape(B, L, self.top_k)
            top_k_gates = top_k_gates.reshape(B, L, self.top_k)

        return top_k_idx, top_k_gates


def apply_expert_choice_patch(model: Any, capacity_factor: float = 1.2) -> int:
    """Replace GraniteMoeTopKGating with ExpertChoiceRouter in all MoE layers.

    Preserves the gate Linear weights (same shape: [input_size, num_experts]).

    Args:
        model: The loaded MLX model.
        capacity_factor: Expert overprovisioning factor.

    Returns:
        Number of routers replaced.
    """
    global _patch_applied
    if _patch_applied:
        logger.info("Expert-choice patch already applied")
        return 0

    replaced = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name not in ("GraniteMoeTopKGating", "GraniteMoeHybridTopKGating"):
            continue

        # Extract parameters from original router
        gate_weight = module.layer.weight  # (num_experts, input_size)
        num_experts = gate_weight.shape[0]
        input_size = gate_weight.shape[1]
        top_k = getattr(module, "top_k", getattr(module, "num_experts_per_tok", 2))

        # Create replacement router with correct dimensions from weight
        new_router = ExpertChoiceRouter(
            input_size=input_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )

        # Transfer gate weights — must match shape exactly
        new_router.layer = module.layer  # Reuse the original Linear directly
        new_router.freeze()

        # Replace in parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                parent = parent[int(part)]

        setattr(parent, parts[-1], new_router)
        replaced += 1

    if replaced > 0:
        logger.info(
            f"Expert-choice: replaced {replaced} routers (capacity_factor={capacity_factor})"
        )

    _patch_applied = True
    return replaced
