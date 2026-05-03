# SPDX-License-Identifier: Apache-2.0
"""Lock-in tests for ``scripts/ttt_distill_single_head.py``.

These tests cover the parts of the distillation harness that don't
require a model load — policy parsing, loss + metric correctness, and
synthetic-data generation. The actual training-loop convergence is
exercised in ``test_synthetic_training_recovers_ground_truth`` against
a known TTT-Linear target; if this test passes, the script can be
trusted for capture-mode runs.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ttt_distill_single_head import (
    HeadSelection,
    cos_sim_per_token,
    make_synthetic_pairs,
    mse_loss,
    pick_streaming_head,
    run_synthetic,
)


# ---------------------------------------------------------------------------
# Loss + metric correctness
# ---------------------------------------------------------------------------


def test_mse_loss_zero_when_pred_equals_target():
    a = mx.array([[1.0, 2.0, 3.0]])
    b = a + 0.0
    assert mse_loss(a, b).item() < 1e-10


def test_mse_loss_simple_offset():
    a = mx.zeros((1, 4))
    b = mx.ones((1, 4))
    # MSE = mean((0-1)²) = 1
    assert abs(mse_loss(a, b).item() - 1.0) < 1e-6


def test_cos_sim_orthogonal_is_zero():
    a = mx.array([[[1.0, 0.0]]])
    b = mx.array([[[0.0, 1.0]]])
    assert abs(cos_sim_per_token(a, b).item()) < 1e-6


def test_cos_sim_parallel_is_one():
    a = mx.array([[[1.0, 2.0, 3.0]]])
    b = a * 5.0
    assert abs(cos_sim_per_token(a, b).item() - 1.0) < 1e-5


def test_cos_sim_antiparallel_is_minus_one():
    a = mx.array([[[1.0, 2.0]]])
    b = a * -1.0
    assert abs(cos_sim_per_token(a, b).item() - (-1.0)) < 1e-5


def test_cos_sim_returns_per_token_shape():
    a = mx.random.normal((2, 5, 8))
    b = mx.random.normal((2, 5, 8))
    sims = cos_sim_per_token(a, b)
    assert sims.shape == (2, 5)


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------


def test_pick_streaming_head_chooses_highest_local_fraction(tmp_path):
    policy = {
        "model": "test", "n_layers": 2, "n_heads": 4,
        "n_kv_heads": 4, "context_len": 1024, "window": 256, "sink": 4,
        "streaming_threshold": 0.85, "streaming_fraction": 0.5,
        "streaming_count": 4, "retrieval_count": 4,
        "heads": [
            {"layer": 0, "head": 0, "policy": "retrieval",
             "local_fraction": 0.7},
            {"layer": 0, "head": 1, "policy": "streaming",
             "local_fraction": 0.86},   # streaming, lower
            {"layer": 1, "head": 2, "policy": "streaming",
             "local_fraction": 0.95},   # streaming, HIGHEST
            {"layer": 1, "head": 3, "policy": "streaming",
             "local_fraction": 0.88},
        ],
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))

    sel = pick_streaming_head(p)
    assert isinstance(sel, HeadSelection)
    assert sel.layer == 1
    assert sel.head == 2
    assert abs(sel.local_fraction - 0.95) < 1e-9
    assert sel.n_layers == 2
    assert sel.n_heads == 4


def test_pick_streaming_head_raises_on_no_streaming(tmp_path):
    policy = {
        "model": "test", "n_layers": 1, "n_heads": 1,
        "n_kv_heads": 1, "context_len": 1024, "window": 256, "sink": 4,
        "streaming_threshold": 0.85, "streaming_fraction": 0.0,
        "streaming_count": 0, "retrieval_count": 1,
        "heads": [{"layer": 0, "head": 0, "policy": "retrieval",
                   "local_fraction": 0.3}],
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="No streaming"):
        pick_streaming_head(p)


def test_pick_streaming_head_on_real_qwen3_coder_policy():
    """The shipped Qwen3-Coder policy must have at least one streaming
    head with local_fraction ≥ 0.85 (the calibration threshold)."""
    p = Path("omlx/patches/duoattention_policies/"
             "qwen3_coder_30b_a3b_instruct_8bit.json")
    if not p.exists():
        pytest.skip("Qwen3-Coder policy file not present")
    sel = pick_streaming_head(p)
    assert sel.local_fraction >= 0.85
    assert 0 <= sel.layer < sel.n_layers
    assert 0 <= sel.head < sel.n_heads


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def test_make_synthetic_pairs_shapes():
    x, o = make_synthetic_pairs(L=64, D=8, eta_truth=0.1)
    assert x.shape == (1, 64, 8)
    assert o.shape == (1, 64, 8)


def test_make_synthetic_pairs_deterministic_per_seed():
    x1, o1 = make_synthetic_pairs(L=16, D=4, seed=42)
    x2, o2 = make_synthetic_pairs(L=16, D=4, seed=42)
    assert mx.max(mx.abs(x1 - x2)).item() < 1e-12
    assert mx.max(mx.abs(o1 - o2)).item() < 1e-12


def test_make_synthetic_pairs_distinct_seeds():
    x1, _ = make_synthetic_pairs(L=8, D=4, seed=0)
    x2, _ = make_synthetic_pairs(L=8, D=4, seed=1)
    assert mx.max(mx.abs(x1 - x2)).item() > 0.1


# ---------------------------------------------------------------------------
# Convergence sanity check (the load-bearing math test)
# ---------------------------------------------------------------------------


def test_synthetic_training_recovers_ground_truth():
    """The script's training loop must recover a TTT-Linear ground truth.

    If we generate (x, o) by forward-running a TRUE TTT-Linear with known
    random projections, then training another TTT-Linear from random init
    on (x, o) should drive cos-sim to a high value on held-out data — the
    target IS exactly expressible as a TTT-Linear, so the only obstacle
    is the optimizer landscape, not architectural mismatch.

    This is a very fast (~5s) convergence test; we don't aim for the full
    0.95 spike gate (which would need more steps + tuning), but for a
    clear improvement above the random-init baseline.
    """
    result = run_synthetic(
        D=16, n_steps=400, lr=3e-3, eta=0.05, mini_batch_size=64,
    )
    summary = result["summary"]
    # After 400 Adam steps on a 1K-token synthetic, the held-out cos-sim
    # mean must clearly clear random init (~0). 0.5 is a soft sanity gate.
    assert summary["mean_cos"] > 0.5, (
        f"training did not move cos-sim above 0.5 — optimizer "
        f"landscape may be broken. summary={summary}"
    )
    # The val MSE must improve over the course of training.
    val_curve = result["val_curve"]
    first, last = val_curve[0][1], val_curve[-1][1]
    assert last < first, (
        f"val MSE did not decrease: first={first:.4e} last={last:.4e}"
    )
