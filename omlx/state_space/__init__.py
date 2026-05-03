# SPDX-License-Identifier: Apache-2.0
"""State-space replacements for streaming attention heads.

This package houses experimental linear-time alternatives to softmax
attention, intended to be drop-in for the *streaming-tagged* heads
identified by DuoAttention calibration. The retrieval-tagged heads keep
softmax attention (with sparse pattern dispatch); the streaming heads get
replaced with a recurrence whose per-token compute is independent of
context length — that's the move that rotates the O(N²) prefill curve to
O(N) for ~59% of Qwen3.6's attention budget.

Currently scaffolded:
- ``ttt_linear`` — TTT-Linear from Sun et al. (arXiv:2407.04620). Hidden
  state is a single linear layer ``W ∈ R^{D×D}`` updated via inner-loop
  gradient descent on a reconstruction loss per token.

Planned:
- ``ttt_mlp`` — TTT-MLP variant for richer hidden state (when TTT-Linear's
  cos-sim against the original streaming head falls short).
- ``mamba2_ssm`` — selective SSM as a third option if neither TTT variant
  distills cleanly.
"""

from .ttt_linear import TTTLinear, TTTLinearConfig  # noqa: F401
