# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear block (Sun et al., arXiv:2407.04620).

A linear-time attention replacement whose hidden state is itself a linear
layer ``W ∈ R^{D×D}``, updated per token via inner-loop gradient descent
on a reconstruction objective. Per-token compute is O(D²) regardless of
context length — the structural property that makes prefill cost flat
beyond the chunk-amortized constant.

Math reminder (paper Section 3.1, the core update rule)::

    Q_t = W_Q x_t,  K_t = W_K x_t,  V_t = W_V x_t        (token projections)
    l(W; x_t)  = || W K_t  -  V_t ||²                     (reconstruction loss)
    ∇_W l      = 2 (W K_t - V_t) K_t^T                    (closed-form gradient)
    W_t        = W_{t-1}  -  η ∇_W l(W_{t-1}; x_t)        (inner SGD step)
    o_t        = W_t Q_t                                  (token output)

Mini-batch TTT (paper Section 3.3) processes ``b`` tokens at once with a
single gradient step against the batch — this is what makes the inner loop
fast enough to be practical, since each gradient update on B tokens costs
only marginally more than on 1 token but advances the hidden state ``b``
positions. For Hypercar's 4096-token chunked prefill, the natural choice
is ``mini_batch_size = 4096`` matching ``PREFILL_CHUNK``.

This module is intentionally bench-machinery-free — it's a pure layer.
The integration story (replacing the streaming-tagged heads in Qwen3.6's
``self_attn`` with a TTT-Linear instance per (layer, head)) lives in a
patches module, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TTTLinearConfig:
    """Configuration for a single (layer, head) TTT-Linear block.

    Attributes:
        head_dim: ``D`` — dimension of the per-head representation. For
            Qwen3.6 with ``head_dim=256`` this is 256.
        eta: Inner-loop learning rate for the per-token SGD step. Paper
            uses ``0.1`` for TTT-Linear at start of training; for
            distillation we may need to learn this per (layer, head).
        mini_batch_size: How many tokens are processed per inner gradient
            step. Defaults to 1 (true online); set to 4096 to match
            ``PREFILL_CHUNK`` for prefill efficiency.
        use_layer_norm: Apply LayerNorm to ``Q_t`` before output computation
            (paper found this stabilizes training).
    """

    head_dim: int
    eta: float = 0.1
    mini_batch_size: int = 1
    use_layer_norm: bool = True


class TTTLinear(nn.Module):
    """A single (layer, head) TTT-Linear block.

    NOT YET IMPLEMENTED — this is a scaffolding stub from Task 388 Phase 0
    step 1. The forward pass below documents the intended shape contract.
    Each method is annotated with the paper section it implements.
    """

    def __init__(self, config: TTTLinearConfig):
        super().__init__()
        self.config = config
        D = config.head_dim
        # Token projections (Q, K, V) — same shape and role as in standard
        # attention; the difference is purely in how outputs are computed
        # from them.
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        # Optional output norm (paper uses LN before output; some variants
        # use it on Q as well).
        self.ln = nn.LayerNorm(D) if config.use_layer_norm else None
        # The hidden state W is created/managed at forward time. It has
        # shape (B, D, D) — one matrix per batch item. Stored on the
        # cache-like state object passed in.

    def init_state(self, batch_size: int, dtype=mx.float32) -> mx.array:
        """Return the initial hidden state ``W_0`` for ``batch_size`` items.

        Paper convention: identity-like initialization so that the first
        few tokens' outputs are roughly ``Q``, before any K/V history
        accumulates.
        """
        D = self.config.head_dim
        # Identity init scaled by 1/sqrt(D) — small but non-zero. Subject
        # to revision based on distillation results in Phase 0 step 3.
        W = mx.broadcast_to(
            mx.eye(D, dtype=dtype) / (D ** 0.5),
            (batch_size, D, D),
        )
        return W

    def __call__(
        self,
        x: mx.array,                       # (B, L, D)
        state: Optional[mx.array] = None,  # (B, D, D)
    ) -> tuple[mx.array, mx.array]:
        """Forward pass over ``L`` tokens, threading hidden state.

        Returns
        -------
        outputs : (B, L, D)
            Per-token output ``o_t = W_t Q_t``.
        new_state : (B, D, D)
            Final hidden state after consuming all ``L`` tokens. Returned
            so the caller can resume on the next chunk.

        Phase 0 implementation TODO (this stub raises NotImplementedError):
            1. Compute Q, K, V for the whole sequence in one matmul.
            2. Iterate over mini-batches of size ``mini_batch_size``:
               - For each mini-batch, compute the closed-form gradient
                 averaged over the batch.
               - Apply the inner SGD step to W.
               - Compute outputs ``W Q`` for the batch (vectorized).
            3. Return concatenated outputs + final W.

        The vectorized mini-batch update is the load-bearing optimization
        — without it, this is a slow per-token Python loop. With it, the
        inner loop is two matmuls per mini-batch and the throughput is
        comparable to attention's QK^T.
        """
        raise NotImplementedError(
            "TTT-Linear forward pass not yet implemented. See Task 388 "
            "Phase 0 step 1 for the scope. The scaffolding (config, "
            "shape contract, init_state) is in place; the inner-loop "
            "update + mini-batch vectorization is the remaining work."
        )
