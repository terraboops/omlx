# SPDX-License-Identifier: Apache-2.0
"""Prefill optimization: only compute logits for the last token during prefill.

During prefill, the model projects ALL hidden states through lm_head, creating
a (batch, seq_len, vocab_size) tensor. For 131K tokens × 151936 vocab × 2 bytes
= 40 GB — this exceeds the Metal buffer limit.

But we only need logits for the LAST token (to start generation). This patch
intercepts the model's forward pass to project only the final hidden state
through lm_head, reducing the logits tensor from 40GB to 0.3MB.

This unlocks 256K+ context without any quality impact.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_prefill_last_logit_patch(model: Any) -> bool:
    """Monkey-patch model.__call__ to only compute last-token logits during prefill.

    During prefill (seq_len > 1), only project the last hidden state through
    lm_head. During decode (seq_len == 1), behavior is unchanged.

    Args:
        model: The loaded MLX model.

    Returns:
        True if patch was applied.
    """
    global _patch_applied
    if _patch_applied:
        return True

    ModelClass = type(model)
    original_call = ModelClass.__call__

    def _last_logit_call(self, inputs, cache=None, **kwargs):
        # Get hidden states from the backbone
        if hasattr(self, 'model'):
            hidden = self.model(inputs, cache=cache, **kwargs)
        else:
            return original_call(self, inputs, cache=cache, **kwargs)

        # During prefill (seq_len > 1): only project the last token
        if hidden.shape[1] > 1:
            hidden = hidden[:, -1:, :]

        # Apply lm_head
        if hasattr(self.args, 'tie_word_embeddings') and self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = self.lm_head(hidden)

        # Apply logits scaling if present
        if hasattr(self, 'logits_scaling'):
            out = out / self.logits_scaling

        return out

    ModelClass.__call__ = _last_logit_call
    _patch_applied = True
    logger.info("Prefill last-logit patch applied (saves ~40GB at 128K context)")
    return True
