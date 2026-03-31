# SPDX-License-Identifier: Apache-2.0
"""Medusa patch: attach draft heads to model and enable speculative decoding.

Monkey-patches the model's forward pass to also produce draft logits
stored on the model instance for the scheduler to read.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_medusa_patch(model: Any, num_heads: int = 3) -> bool:
    """Attach MedusaDraftHeads to model and patch forward pass.

    1. Initialize MedusaDraftHeads with random weights.
    2. Store as model._medusa_heads.
    3. Monkey-patch model.__call__ to store draft logits after each forward.

    The draft logits are stored on model._last_draft_logits for the
    scheduler/generation loop to read. The main return value is unchanged
    to preserve compatibility.

    Args:
        model: The loaded MLX model.
        num_heads: Number of draft heads.

    Returns:
        True if patch was applied.
    """
    global _patch_applied
    if _patch_applied:
        return True

    from ..medusa_heads import MedusaDraftHeads

    # Extract model dimensions
    args = getattr(model, "args", None)
    if args is None:
        logger.warning("Medusa: model has no args attribute, cannot determine dimensions")
        return False

    d_model = getattr(args, "hidden_size", None)
    vocab_size = getattr(args, "vocab_size", None)
    if d_model is None or vocab_size is None:
        logger.warning("Medusa: cannot determine hidden_size or vocab_size")
        return False

    # Create draft heads
    heads = MedusaDraftHeads(d_model, vocab_size, num_heads=num_heads)
    model._medusa_heads = heads
    model._last_draft_logits = None

    # Monkey-patch __call__ to also run draft heads
    original_call = type(model).__call__

    def _medusa_call(self, inputs, cache=None, **kwargs):
        # Run original forward pass
        logits = original_call(self, inputs, cache=cache, **kwargs)

        # Only compute drafts during decode (seq_len=1) when cache exists
        if cache is not None and inputs.shape[-1] == 1:
            # Get hidden states: re-run the backbone (without lm_head)
            # This is the internal model (before lm_head projection)
            if hasattr(self, "model"):
                hidden = self.model(inputs, cache=cache)
                self._last_draft_logits = self._medusa_heads(hidden)
            else:
                self._last_draft_logits = None
        else:
            self._last_draft_logits = None

        return logits

    type(model).__call__ = _medusa_call
    _patch_applied = True
    logger.info(f"Medusa: attached {num_heads} draft heads (d_model={d_model}, vocab={vocab_size})")
    return True
