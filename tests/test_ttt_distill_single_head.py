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
    make_synthetic_pairs_multi,
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
    """Multi-sequence + early-stop must clear the 0.95 spike gate on the
    synthetic-recovery EASY case (target exactly expressible as TTT-Linear).

    With single-sequence training (the previous setup), the same harness
    overfit at step ~200 to peak val cos-sim 0.69 → degraded. Multi-seq
    averaging across N=8 sequences sharing one ground-truth gives the
    optimizer enough signal to learn the dynamics rather than memorize
    one sequence's specifics.

    Compute budget is intentionally tight (~5s) — uses D=16, fewer steps
    than the script's default — so this catches regressions in the loop
    without slowing the suite.
    """
    result = run_synthetic(
        D=16, n_steps=400, lr=3e-3, eta=0.05, mini_batch_size=64,
        n_train_seqs=4, n_val_seqs=2, L=512,
    )
    summary = result["summary"]
    assert summary["mean_cos"] > 0.95, (
        f"synthetic recovery missed 0.95 gate at small budget: "
        f"summary={summary}"
    )
    # Best-step bookkeeping must be populated by early stopping.
    assert summary["best_step"] >= 0
    assert summary["best_val_cos_mean"] > 0.95


def test_make_synthetic_pairs_multi_shapes():
    x, o = make_synthetic_pairs_multi(n_seqs=4, L=32, D=8)
    assert x.shape == (4, 32, 8)
    assert o.shape == (4, 32, 8)


def test_make_synthetic_pairs_multi_distinct_data_seeds():
    """Same truth_seed + different data_seed → same dynamics, different
    inputs. Sequences from data_seed=0 must differ from data_seed=1, and
    both run through the same ground-truth dynamics produce structurally
    related (but not identical) outputs."""
    x0, o0 = make_synthetic_pairs_multi(
        n_seqs=2, L=16, D=8, truth_seed=0, data_seed=0)
    x1, o1 = make_synthetic_pairs_multi(
        n_seqs=2, L=16, D=8, truth_seed=0, data_seed=1)
    # Inputs should differ
    assert mx.max(mx.abs(x0 - x1)).item() > 0.1
    # Outputs differ too (different inputs → different dynamics trace)
    assert mx.max(mx.abs(o0 - o1)).item() > 0.1


def test_make_synthetic_pairs_multi_truth_seed_pins_dynamics():
    """Same data_seed, different truth_seed → same inputs run through
    different ground-truth projections, so outputs differ."""
    x0, o0 = make_synthetic_pairs_multi(
        n_seqs=2, L=16, D=8, truth_seed=0, data_seed=42)
    x1, o1 = make_synthetic_pairs_multi(
        n_seqs=2, L=16, D=8, truth_seed=99, data_seed=42)
    # Inputs match
    assert mx.max(mx.abs(x0 - x1)).item() < 1e-10
    # Outputs differ (different projections)
    assert mx.max(mx.abs(o0 - o1)).item() > 0.05


def test_early_stop_restores_best_params():
    """Early stopping is the load-bearing post-fix from the methodology
    note. Verify the bookkeeping: best_step / best_val_cos_mean are in
    the summary, and the params at end equal the best snapshot.

    Sanity, not perf: we run a tiny 200-step training. The implementation
    is correct iff the best_val_cos_mean in the summary equals the
    eval-on-restored-params cos-sim (the script's `summary["mean_cos"]`
    is computed AFTER restoration). They should match within tolerance.
    """
    result = run_synthetic(
        D=8, n_steps=200, lr=3e-3, eta=0.05, mini_batch_size=64,
        n_train_seqs=2, n_val_seqs=1, L=256,
    )
    s = result["summary"]
    # After restoration, the post-restore eval cos-sim should equal the
    # tracked best_val_cos_mean (within fp tolerance for re-eval noise).
    assert abs(s["mean_cos"] - s["best_val_cos_mean"]) < 1e-3, (
        f"early-stop bookkeeping mismatch: post-restore mean_cos="
        f"{s['mean_cos']:.6f} vs best_val_cos_mean={s['best_val_cos_mean']:.6f}"
    )
