# SPDX-License-Identifier: Apache-2.0
"""Phase 0 step 1 lock-in tests for ``omlx/state_space/ttt_linear.py``.

These tests verify the scaffolding contract — what's in place today.
The forward pass is deferred to Phase 0 step 2 (next loop cycle); a
``test_forward_raises_pending_implementation`` test pins that contract so
we notice when step 2 lands.
"""

import pytest
import mlx.core as mx

from omlx.state_space import TTTLinear, TTTLinearConfig


def test_config_defaults():
    cfg = TTTLinearConfig(head_dim=256)
    assert cfg.head_dim == 256
    assert cfg.eta == 0.1
    assert cfg.mini_batch_size == 1
    assert cfg.use_layer_norm is True


def test_config_overrides():
    cfg = TTTLinearConfig(
        head_dim=128, eta=0.5, mini_batch_size=4096, use_layer_norm=False)
    assert cfg.eta == 0.5
    assert cfg.mini_batch_size == 4096
    assert cfg.use_layer_norm is False


def test_module_constructs_with_qwen36_head_dim():
    """Qwen3.6 has head_dim=256 — the most common target shape."""
    cfg = TTTLinearConfig(head_dim=256)
    ttt = TTTLinear(cfg)
    # Three projections + optional LayerNorm
    assert ttt.W_Q.weight.shape == (256, 256)
    assert ttt.W_K.weight.shape == (256, 256)
    assert ttt.W_V.weight.shape == (256, 256)
    assert ttt.ln is not None


def test_module_omits_ln_when_disabled():
    cfg = TTTLinearConfig(head_dim=64, use_layer_norm=False)
    ttt = TTTLinear(cfg)
    assert ttt.ln is None


def test_init_state_shape_and_identity_scaling():
    """init_state should return (B, D, D) identity-scaled by 1/sqrt(D)."""
    cfg = TTTLinearConfig(head_dim=4)
    ttt = TTTLinear(cfg)
    state = ttt.init_state(batch_size=3)
    assert state.shape == (3, 4, 4)
    # Diagonal should be 1/sqrt(D); off-diagonal 0.
    expected_diag = 1.0 / (4 ** 0.5)
    s_np = state.tolist()
    for batch in s_np:
        for i in range(4):
            for j in range(4):
                expected = expected_diag if i == j else 0.0
                assert abs(batch[i][j] - expected) < 1e-6


def test_init_state_supports_dtype_override():
    cfg = TTTLinearConfig(head_dim=8)
    ttt = TTTLinear(cfg)
    state_fp16 = ttt.init_state(batch_size=1, dtype=mx.float16)
    assert state_fp16.dtype == mx.float16


def test_forward_raises_pending_implementation():
    """Phase 0 step 1 is scaffolding only — forward pass is step 2.

    This test pins the contract: we expect NotImplementedError today.
    When step 2 lands, this test should be deleted (and replaced with the
    actual forward-pass correctness tests against a hand-computed 3-token
    reference per the design note).
    """
    cfg = TTTLinearConfig(head_dim=8)
    ttt = TTTLinear(cfg)
    x = mx.zeros((1, 4, 8))
    with pytest.raises(NotImplementedError, match="Phase 0 step 1"):
        ttt(x)
