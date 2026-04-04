# SPDX-License-Identifier: Apache-2.0
"""Vertical graph eval patch — kills cross-layer graph hoarding.

During large prefills, MLX builds a single computation graph across
all 47+ transformer layers. This graph holds all intermediate tensors
(attention scores, MLP activations, residuals) for every layer
simultaneously, causing 44GB+ peak on a 17GB-active workload.

Fix: force mx.eval(h) every N layers during prefill. This collapses
the graph and frees intermediates before the next batch of layers.

Only activates during prefill (seq_len > 1). Decode (seq_len == 1)
is untouched — its intermediates are tiny.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
EVAL_EVERY = 8  # Collapse graph every N layers


def apply_vertical_eval_patch(model, eval_every: int = EVAL_EVERY) -> bool:
    """Monkey-patch the model's forward to eval every N layers during prefill."""
    global _PATCHED
    if _PATCHED:
        return False

    inner = model.model if hasattr(model, "model") else model

    if not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
        logger.warning("vertical_eval: model structure not recognized, skipping")
        return False

    original_call = inner.__call__

    def patched_call(x, mask=None, cache=None):
        h = inner.embed_tokens(x)
        is_prefill = x.shape[1] > 1

        for i, layer in enumerate(inner.layers):
            c = cache[i] if cache is not None else None
            h = layer(h, mask, c)

            # Collapse graph every N layers during prefill only.
            # Decode (L=1) intermediates are tiny — no need to eval.
            if is_prefill and (i + 1) % eval_every == 0:
                mx.eval(h)
                logger.debug("vertical_eval: collapsed graph after layer %d", i)

        h = inner.norm(h)
        return h

    inner.__call__ = patched_call
    _PATCHED = True
    logger.info("Vertical graph eval patch applied (every %d layers)", eval_every)
    return True
