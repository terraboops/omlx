# SPDX-License-Identifier: Apache-2.0
"""Medusa Draft Heads: multi-token lookahead for speculative decoding.

Adds lightweight residual draft heads alongside the main lm_head that predict
tokens t+1, t+2, ..., t+K simultaneously. On memory-bound hardware (M4 Pro),
the extra linear projections are essentially free since they don't increase
peak memory bandwidth.

A tree-verification step checks drafted tokens against the main model's
predictions, accepting the longest correct prefix for ~2x decode throughput.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class MedusaResidualBlock(nn.Module):
    """Single residual block: Linear + SiLU + skip connection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.linear(x)) + x


class MedusaDraftHeads(nn.Module):
    """Multi-token lookahead draft heads.

    Each head is a small residual block followed by a vocabulary projection.
    Heads are chained: head_k receives the refined state from head_{k-1}.
    This lets each head specialize in progressively more distant predictions.

    Args:
        d_model: Hidden dimension of the base model.
        vocab_size: Vocabulary size.
        num_heads: Number of draft heads (predicts t+1 through t+num_heads).
    """

    def __init__(self, d_model: int, vocab_size: int, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.res_blocks = [MedusaResidualBlock(d_model) for _ in range(num_heads)]
        self.lm_heads = [
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(num_heads)
        ]

    def __call__(self, hidden_states: mx.array) -> List[mx.array]:
        """Generate draft logits for future tokens.

        Args:
            hidden_states: (B, L, D) or (B, D) last hidden state from base model.

        Returns:
            List of logits tensors, one per head:
            [logits_t+1, logits_t+2, ..., logits_t+num_heads]
            Each has shape matching hidden_states batch/seq dims + (vocab_size,).
        """
        draft_logits = []
        current_state = hidden_states

        for i in range(self.num_heads):
            current_state = self.res_blocks[i](current_state)
            logits = self.lm_heads[i](current_state)
            draft_logits.append(logits)

        return draft_logits


def verify_draft_tokens(
    model_logits: mx.array,
    draft_tokens: mx.array,
) -> int:
    """Tree-verification: find longest correct prefix in drafted tokens.

    Args:
        model_logits: (num_candidates, vocab_size) — model's logits for each
            candidate position (obtained by feeding all candidates in one pass).
        draft_tokens: (num_candidates,) — tokens predicted by draft heads.

    Returns:
        Number of accepted tokens (0 means none accepted, all need resampling).
    """
    # Greedy: check if the model's argmax matches the draft at each position
    model_predictions = mx.argmax(model_logits, axis=-1)  # (num_candidates,)
    mx.eval(model_predictions)

    # Find longest matching prefix
    accepted = 0
    model_np = model_predictions.tolist()
    draft_np = draft_tokens.tolist()

    for i in range(len(draft_np)):
        if model_np[i] == draft_np[i]:
            accepted += 1
        else:
            break

    return accepted
