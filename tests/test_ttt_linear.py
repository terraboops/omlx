# SPDX-License-Identifier: Apache-2.0
"""Phase 0 step 2 lock-in tests for ``omlx/state_space/ttt_linear.py``.

The forward pass implements the closed-form mini-batch update from
Sun et al. (arXiv:2407.04620) Section 3.3. These tests verify:

- Scaffolding contract (config, init_state, module shapes).
- Output shape and dtype invariants.
- A hand-computed 3-token reference at D=2 with identity projections,
  η=0.5, no LayerNorm — pins the SGD update direction and rate against
  arithmetic that fits on a single page.
- State-threading equivalence: running ``[A; B]`` in one call must equal
  running ``A`` then ``B`` with the state passed through, for the online
  (mini_batch_size=1) regime.
- Mini-batch behavior: with ``mini_batch_size=L`` (one update over the
  full chunk), the implementation degenerates to a single SGD step.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from omlx.state_space import TTTLinear, TTTLinearConfig


# ---------------------------------------------------------------------------
# Scaffolding contract (carried over from Phase 0 step 1)
# ---------------------------------------------------------------------------


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
    assert ttt.W_Q.weight.shape == (256, 256)
    assert ttt.W_K.weight.shape == (256, 256)
    assert ttt.W_V.weight.shape == (256, 256)
    assert ttt.ln is not None


def test_module_omits_ln_when_disabled():
    cfg = TTTLinearConfig(head_dim=64, use_layer_norm=False)
    ttt = TTTLinear(cfg)
    assert ttt.ln is None


def test_init_state_shape_and_identity_scaling():
    cfg = TTTLinearConfig(head_dim=4)
    ttt = TTTLinear(cfg)
    state = ttt.init_state(batch_size=3)
    assert state.shape == (3, 4, 4)
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


# ---------------------------------------------------------------------------
# Forward-pass shape / dtype invariants
# ---------------------------------------------------------------------------


def test_forward_returns_outputs_and_state_shapes():
    cfg = TTTLinearConfig(head_dim=8)
    ttt = TTTLinear(cfg)
    x = mx.random.normal((2, 16, 8))
    out, state = ttt(x)
    assert out.shape == (2, 16, 8)
    assert state.shape == (2, 8, 8)


def test_forward_preserves_input_dtype():
    """When the module is cast to fp16 (the deployment dtype), forward
    should keep the working tensors in fp16 throughout."""
    cfg = TTTLinearConfig(head_dim=8, use_layer_norm=False)
    ttt = TTTLinear(cfg)
    ttt.set_dtype(mx.float16)
    x = mx.random.normal((1, 4, 8)).astype(mx.float16)
    out, state = ttt(x)
    assert out.dtype == mx.float16
    assert state.dtype == mx.float16


def test_forward_rejects_wrong_head_dim():
    cfg = TTTLinearConfig(head_dim=8)
    ttt = TTTLinear(cfg)
    x = mx.zeros((1, 4, 16))   # D=16 instead of 8
    with pytest.raises(ValueError, match="head_dim"):
        ttt(x)


# ---------------------------------------------------------------------------
# Hand-computed 3-token reference (Phase 0 step 2 gate from the design note)
# ---------------------------------------------------------------------------


def _identity_ttt(D: int, eta: float, mini_batch_size: int = 1):
    """Build a TTTLinear with identity Q/K/V projections, no LN.

    Used by the reference test below — with identity projections,
    Q = K = V = x, so the SGD update rule simplifies and can be
    hand-computed.
    """
    cfg = TTTLinearConfig(
        head_dim=D, eta=eta, mini_batch_size=mini_batch_size,
        use_layer_norm=False,
    )
    ttt = TTTLinear(cfg)
    eye = mx.eye(D, dtype=mx.float32)
    ttt.W_Q.weight = eye
    ttt.W_K.weight = eye
    ttt.W_V.weight = eye
    return ttt


def test_forward_three_token_hand_computed_reference():
    """Hand-computed 3-token reference at D=2, η=0.5, identity proj, no LN.

    Setup:
      W_0 = (1/√2) I = [[0.7071, 0], [0, 0.7071]]
      x_1 = [1, 0],  x_2 = [0, 1],  x_3 = [1, 1]

    Step 1 (token x_1 = [1,0]):
      Q=K=V=[1,0]
      W_0 K = [0.7071, 0];  r = WK - V = [-0.2929, 0]
      grad = 2 r K^T = [[-0.5858, 0], [0, 0]]
      W_1 = W_0 - 0.5·grad = [[1.0, 0], [0, 0.7071]]
      o_1 = W_1 Q = [1.0, 0]

    Step 2 (token x_2 = [0,1]):
      Q=K=V=[0,1]
      W_1 K = [0, 0.7071];  r = [0, -0.2929]
      grad = 2 r K^T = [[0, 0], [0, -0.5858]]
      W_2 = W_1 - 0.5·grad = [[1, 0], [0, 1]]   = identity
      o_2 = W_2 Q = [0, 1]

    Step 3 (token x_3 = [1,1]):
      Q=K=V=[1,1]
      W_2 K = [1, 1];  r = [0, 0]   ← already perfectly fits
      grad = 0;  W_3 = W_2 = identity
      o_3 = [1, 1]

    Final state = identity, outputs = [[1,0], [0,1], [1,1]].
    """
    D = 2
    ttt = _identity_ttt(D, eta=0.5, mini_batch_size=1)
    x = mx.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=mx.float32)
    out, state = ttt(x)

    expected_out = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    expected_state = [[[1.0, 0.0], [0.0, 1.0]]]

    out_l = out.tolist()
    state_l = state.tolist()
    for b in range(1):
        for t in range(3):
            for d in range(D):
                assert abs(out_l[b][t][d] - expected_out[b][t][d]) < 1e-5, (
                    f"out[{b}][{t}][{d}] = {out_l[b][t][d]} "
                    f"vs expected {expected_out[b][t][d]}"
                )
        for i in range(D):
            for j in range(D):
                assert abs(state_l[b][i][j] - expected_state[b][i][j]) < 1e-5


# ---------------------------------------------------------------------------
# State-threading equivalence
# ---------------------------------------------------------------------------


def test_forward_state_threading_matches_single_call():
    """Online (b=1): running [A;B] equals running A then B with state passed
    through. Catches off-by-one errors in the mini-batch loop bookkeeping."""
    D = 4
    ttt = _identity_ttt(D, eta=0.1, mini_batch_size=1)
    x = mx.random.normal((1, 6, D))

    out_full, state_full = ttt(x)

    # Split: 2 then 4
    out_a, state_a = ttt(x[:, :2, :])
    out_b, state_b = ttt(x[:, 2:, :], state=state_a)

    out_combined = mx.concatenate([out_a, out_b], axis=1)

    diff_out = mx.max(mx.abs(out_combined - out_full)).item()
    diff_state = mx.max(mx.abs(state_b - state_full)).item()
    assert diff_out < 1e-5, f"output diff {diff_out}"
    assert diff_state < 1e-5, f"state diff {diff_state}"


# ---------------------------------------------------------------------------
# Mini-batch behavior (paper Section 3.3)
# ---------------------------------------------------------------------------


def test_minibatch_size_L_is_single_update():
    """With mini_batch_size = L, the whole chunk gets one SGD step.

    Verify by construction: the state after the call should differ from
    the initial state by exactly -η/L · 2 Σ (W_0 K_i - V_i) K_i^T
    (i.e., the closed-form gradient against the batch).
    """
    D = 3
    L = 4
    eta = 0.5
    ttt = _identity_ttt(D, eta=eta, mini_batch_size=L)
    x = mx.random.normal((1, L, D))

    W0 = ttt.init_state(1)            # (1, D, D)
    out, W_final = ttt(x)

    # Hand-computed expected update (identity projections → Q=K=V=x):
    K = x                              # (1, L, D)
    V = x                              # (1, L, D)
    K_T = mx.transpose(K, (0, 2, 1))   # (1, D, L)
    V_T = mx.transpose(V, (0, 2, 1))   # (1, D, L)
    R = mx.matmul(W0, K_T) - V_T       # (1, D, L)
    grad = 2.0 * mx.matmul(R, K)       # (1, D, D)
    expected_W = W0 - (eta / L) * grad

    diff = mx.max(mx.abs(W_final - expected_W)).item()
    assert diff < 1e-5, f"final-state diff {diff}"


def test_minibatch_b_equals_1_takes_L_updates():
    """Online (b=1) runs L updates; b=L runs 1 update; results differ."""
    D = 3
    L = 4
    eta = 0.5
    x = mx.random.normal((1, L, D))

    ttt_online = _identity_ttt(D, eta=eta, mini_batch_size=1)
    ttt_chunk = _identity_ttt(D, eta=eta, mini_batch_size=L)

    _, W_online = ttt_online(x)
    _, W_chunk = ttt_chunk(x)

    diff = mx.max(mx.abs(W_online - W_chunk)).item()
    assert diff > 1e-3, (
        "online vs chunked TTT should give different states — they apply "
        f"different numbers of SGD steps. got diff={diff}"
    )


# ---------------------------------------------------------------------------
# Per-block save/load — round-trip correctness
# ---------------------------------------------------------------------------


def test_save_load_block_roundtrip(tmp_path):
    """Saving a TTTLinear and reloading must return a module that
    produces bit-identical outputs on the same input."""
    from omlx.state_space.ttt_linear import save_block, load_block

    cfg = TTTLinearConfig(head_dim=8, eta=0.05, mini_batch_size=4,
                          use_layer_norm=True)
    ttt = TTTLinear(cfg)
    x = mx.random.normal((1, 4, 8), key=mx.random.key(123))
    out_orig, _ = ttt(x)

    stem = tmp_path / "block"
    save_block(ttt, stem)
    assert (tmp_path / "block.safetensors").exists()
    assert (tmp_path / "block.json").exists()

    ttt2 = load_block(stem)
    out_loaded, _ = ttt2(x)
    diff = mx.max(mx.abs(out_orig - out_loaded)).item()
    assert diff == 0.0, f"roundtrip not bit-identical: {diff}"


def test_save_load_block_preserves_config(tmp_path):
    """Config sidecar JSON must round-trip every field of
    ``TTTLinearConfig`` so reconstruction picks up eta / mini_batch
    / use_layer_norm correctly."""
    from omlx.state_space.ttt_linear import save_block, load_block

    cfg = TTTLinearConfig(head_dim=12, eta=0.123, mini_batch_size=128,
                          use_layer_norm=False)
    ttt = TTTLinear(cfg)
    save_block(ttt, tmp_path / "blk")
    ttt2 = load_block(tmp_path / "blk")
    assert ttt2.config.head_dim == 12
    assert abs(ttt2.config.eta - 0.123) < 1e-9
    assert ttt2.config.mini_batch_size == 128
    assert ttt2.config.use_layer_norm is False
    assert ttt2.ln is None  # use_layer_norm=False propagated


def test_layer_norm_applied_when_enabled():
    """LN-enabled outputs must be unit-variance per token (within tol)."""
    D = 16
    cfg = TTTLinearConfig(head_dim=D, eta=0.1, mini_batch_size=1,
                          use_layer_norm=True)
    ttt = TTTLinear(cfg)
    eye = mx.eye(D, dtype=mx.float32)
    ttt.W_Q.weight = eye
    ttt.W_K.weight = eye
    ttt.W_V.weight = eye
    x = mx.random.normal((1, 8, D))
    out, _ = ttt(x)

    # Per-token mean ≈ 0, var ≈ 1 (LayerNorm normalizes the last dim).
    means = mx.mean(out, axis=-1)
    variances = mx.var(out, axis=-1)
    assert mx.max(mx.abs(means)).item() < 1e-3
    # LN's default eps is small but nonzero, and the affine γ defaults to 1
    # so variance ≈ 1.
    assert mx.max(mx.abs(variances - 1.0)).item() < 1e-2
