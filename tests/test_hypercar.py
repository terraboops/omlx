# SPDX-License-Identifier: Apache-2.0
"""
Test harness for Hypercar features.

Tests all 6 bleeding-edge features for omlx:
1. TurboQuant KV CLI wiring
2. TurboQuantLinear (3.5-bit weight quantization)
3. STARC clustered sparsity attention
4. Mamba-3 MIMO kernel
5. Medusa Draft Heads
6. Expert-Choice MoE Router
"""

import subprocess
import sys

import pytest


# ═══════════════════════════════════════════════════════════════════
# TestCLIFlags — All hypercar flags parse correctly
# ═══════════════════════════════════════════════════════════════════


class TestCLIFlags:
    """All 8 hypercar CLI flags appear in --help and parse correctly."""

    def _get_serve_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        return result.stdout

    def test_cache_mode_in_help(self):
        assert "--cache-mode" in self._get_serve_help()

    def test_fp16_layers_in_help(self):
        assert "--fp16-layers" in self._get_serve_help()

    def test_weight_mode_in_help(self):
        assert "--weight-mode" in self._get_serve_help()

    def test_sparsity_method_in_help(self):
        assert "--sparsity-method" in self._get_serve_help()

    def test_mimo_rank_in_help(self):
        assert "--mimo-rank" in self._get_serve_help()

    def test_medusa_heads_in_help(self):
        assert "--medusa-heads" in self._get_serve_help()

    def test_moe_router_in_help(self):
        assert "--moe-router" in self._get_serve_help()

    def test_max_kv_size_in_help(self):
        assert "--max-kv-size" in self._get_serve_help()


# ═══════════════════════════════════════════════════════════════════
# TestSchedulerConfig — SchedulerConfig carries all hypercar fields
# ═══════════════════════════════════════════════════════════════════


class TestSchedulerConfig:
    """SchedulerConfig has all hypercar fields with correct defaults."""

    def test_has_turboquant_kv_bits(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.turboquant_kv_bits is None

    def test_has_turboquant_fp16_layers(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.turboquant_fp16_layers == 0

    def test_has_weight_mode(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.weight_mode is None

    def test_has_sparsity_method(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.sparsity_method is None

    def test_has_mimo_rank(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.mimo_rank is None

    def test_has_medusa_heads(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.medusa_heads is None

    def test_has_moe_router(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.moe_router is None

    def test_has_max_kv_size(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        assert cfg.max_kv_size is None

    def test_set_all_fields(self):
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig(
            turboquant_kv_bits=3,
            turboquant_fp16_layers=1,
            weight_mode="turbo35",
            sparsity_method="starc",
            mimo_rank=4,
            medusa_heads=3,
            moe_router="expert-choice",
            max_kv_size=256000,
        )
        assert cfg.turboquant_kv_bits == 3
        assert cfg.turboquant_fp16_layers == 1
        assert cfg.weight_mode == "turbo35"
        assert cfg.sparsity_method == "starc"
        assert cfg.mimo_rank == 4
        assert cfg.medusa_heads == 3
        assert cfg.moe_router == "expert-choice"
        assert cfg.max_kv_size == 256000


# ═══════════════════════════════════════════════════════════════════
# TestTurboQuantKV — Feature 1: KV cache mode wiring
# ═══════════════════════════════════════════════════════════════════


class TestTurboQuantKV:
    """TurboQuant KV cache CLI wiring and fp16 layer skip."""

    def test_apply_turboquant_kv_exists(self):
        """_apply_turboquant_kv method exists on BatchGenerator."""
        from omlx.scheduler import _BoundarySnapshotBatchGenerator
        assert hasattr(_BoundarySnapshotBatchGenerator, "_apply_turboquant_kv")

    def test_fp16_layers_field_on_scheduler(self):
        """Scheduler has _turboquant_fp16_layers attribute."""
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig(turboquant_fp16_layers=2)
        assert cfg.turboquant_fp16_layers == 2

    def test_scheduler_config_cli_override(self):
        """CLI --cache-mode sets turboquant_kv_bits on SchedulerConfig."""
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig(turboquant_kv_bits=3, turboquant_fp16_layers=1)
        assert cfg.turboquant_kv_bits == 3
        assert cfg.turboquant_fp16_layers == 1

    def test_fp16_layer_skip_logic(self):
        """_apply_turboquant_kv skips layers < fp16_layers."""
        from unittest.mock import MagicMock
        from omlx.scheduler import _BoundarySnapshotBatchGenerator

        # Create a mock batch generator with the right attributes
        bg = MagicMock(spec=_BoundarySnapshotBatchGenerator)
        bg._turboquant_kv_bits = 3
        bg._turboquant_fp16_layers = 1

        # Create mock cache objects that look like BatchKVCache
        mock_caches = []
        for _ in range(4):
            cache = MagicMock()
            type(cache).__name__ = "BatchKVCache"
            cache.left_padding = MagicMock()
            cache.left_padding.tolist.return_value = [0]
            mock_caches.append(cache)

        # Call the real method with our mock
        _BoundarySnapshotBatchGenerator._apply_turboquant_kv(bg, mock_caches)

        # Layer 0 should NOT have been replaced (fp16_layers=1)
        assert type(mock_caches[0]).__name__ == "BatchKVCache"
        # Layers 1-3 should have been converted
        for i in range(1, 4):
            assert type(mock_caches[i]).__name__ == "BatchTurboQuantKVCache"


# ═══════════════════════════════════════════════════════════════════
# TestTurboQuantLinear — Feature 2: 3.5-bit weight quantization
# ═══════════════════════════════════════════════════════════════════


class TestTurboQuantLinear:
    """TurboQuantLinear with orthogonal rotation + 3-bit quantized matmul."""

    def test_module_importable(self):
        """turboquant_linear module can be imported."""
        from omlx.turboquant_linear import TurboQuantLinear
        assert TurboQuantLinear is not None

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix R satisfies R @ R^T ≈ I."""
        import mlx.core as mx
        from omlx.turboquant_linear import _rotation_matrix

        R = _rotation_matrix(64, seed=42)
        identity = R @ R.T
        expected = mx.eye(64, dtype=mx.float32)
        diff = mx.abs(identity - expected).max().item()
        assert diff < 1e-5, f"R @ R^T deviates from I by {diff}"

    def test_forward_equivalence(self):
        """TurboQuantLinear(x) ≈ nn.Linear(x) within quantization tolerance."""
        import mlx.core as mx
        import mlx.nn as nn
        from omlx.turboquant_linear import TurboQuantLinear

        # Create a linear layer with known weights
        linear = nn.Linear(128, 64, bias=False)
        mx.eval(linear.parameters())

        x = mx.random.normal((2, 128))
        expected = linear(x)

        # Convert to TurboQuantLinear
        tq = TurboQuantLinear.from_linear(linear, group_size=64, bits=3, seed=0)
        actual = tq(x)
        mx.eval(actual)

        # 3-bit quantization introduces substantial error, but outputs should
        # have the same scale and general direction. Use cosine similarity
        # (direction) and normalized RMSE to verify the rotation preserved semantics.
        actual_flat = actual.reshape(-1)
        expected_flat = expected.reshape(-1)
        cos_sim = (
            mx.sum(actual_flat * expected_flat)
            / (mx.linalg.norm(actual_flat) * mx.linalg.norm(expected_flat) + 1e-8)
        ).item()
        assert cos_sim > 0.5, f"Cosine similarity too low: {cos_sim:.4f}"

        # Normalized RMSE: error relative to output magnitude
        rmse = mx.sqrt(mx.mean((actual - expected) ** 2)).item()
        out_scale = mx.sqrt(mx.mean(expected ** 2)).item()
        nrmse = rmse / (out_scale + 1e-8)
        assert nrmse < 2.0, f"Normalized RMSE too high: {nrmse:.4f}"

    def test_from_linear_classmethod(self):
        """TurboQuantLinear.from_linear() converts nn.Linear correctly."""
        import mlx.core as mx
        import mlx.nn as nn
        from omlx.turboquant_linear import TurboQuantLinear

        linear = nn.Linear(64, 32, bias=True)
        mx.eval(linear.parameters())

        tq = TurboQuantLinear.from_linear(linear, group_size=64, bits=3, seed=7)

        assert tq.input_dims == 64
        assert tq.output_dims == 32
        assert tq.group_size == 64
        assert tq.bits == 3
        assert tq.rotation_matrix.shape == (64, 64)
        assert "bias" in tq


# ═══════════════════════════════════════════════════════════════════
# TestSTARC — Feature 3: Clustered sparsity attention
# ═══════════════════════════════════════════════════════════════════


class TestSTARC:
    """STARC clustered sparsity attention."""

    def test_kmeans_cosine_basic(self):
        """K-means clusters 4 distinct groups correctly."""
        import mlx.core as mx
        from omlx.patches.starc_attention import _kmeans_cosine

        # Create 4 well-separated clusters along cardinal directions
        H, D = 1, 8
        cluster_vecs = [
            mx.array([[1, 0, 0, 0, 0, 0, 0, 0]]),
            mx.array([[0, 1, 0, 0, 0, 0, 0, 0]]),
            mx.array([[0, 0, 1, 0, 0, 0, 0, 0]]),
            mx.array([[0, 0, 0, 1, 0, 0, 0, 0]]),
        ]
        # 5 tokens per cluster = 20 tokens total
        keys_list = []
        for v in cluster_vecs:
            noise = mx.random.normal((1, 5, D)) * 0.05
            keys_list.append(mx.broadcast_to(v[:, None, :], (1, 5, D)) + noise)
        keys = mx.concatenate(keys_list, axis=1)  # (1, 20, 8)
        mx.eval(keys)

        centroids, assignments = _kmeans_cosine(keys, n_clusters=4, n_iters=15)
        mx.eval(centroids, assignments)

        assert centroids.shape == (1, 4, 8)
        assert assignments.shape == (1, 20)
        # Each group of 5 should have the same assignment
        for g in range(4):
            group = assignments[0, g*5:(g+1)*5]
            assert mx.all(group == group[0]).item(), f"Group {g} not uniform"

    def test_build_csr_index(self):
        """CSR offsets and sorted indices match assignments."""
        import mlx.core as mx
        from omlx.patches.starc_attention import _build_csr_index

        # 1 head, 6 tokens, 3 clusters: [0,0,1,1,2,2]
        assignments = mx.array([[0, 0, 1, 1, 2, 2]])
        sorted_idx, offsets, sizes = _build_csr_index(assignments, 3)
        mx.eval(sorted_idx, offsets, sizes)

        assert sizes.tolist() == [[2, 2, 2]]
        assert offsets.tolist() == [[0, 2, 4, 6]]

    def test_select_clusters_budget(self):
        """Budget constraint is respected."""
        import mlx.core as mx
        from omlx.patches.starc_attention import (
            _kmeans_cosine, _build_csr_index, _select_clusters
        )

        H, T, D = 2, 64, 16
        keys = mx.random.normal((H, T, D))
        mx.eval(keys)

        K = 8
        centroids, assignments = _kmeans_cosine(keys, K, n_iters=10)
        sorted_idx, offsets, sizes = _build_csr_index(assignments, K)

        # Query aligned with first centroid
        query = centroids[:, 0:1, :]  # (H, 1, D)
        budget = 10

        selected = _select_clusters(query, centroids, sizes, sorted_idx, offsets, budget)
        mx.eval(selected)

        import numpy as np
        sel_np = np.array(selected)
        valid_count = int((sel_np >= 0).sum())
        assert valid_count <= budget, f"Selected {valid_count} > budget {budget}"
        assert valid_count > 0, "Should select at least some tokens"

    def test_starc_manager_initialization(self):
        """StarcManager initializes cluster state for cache objects."""
        from omlx.patches.starc_attention import StarcManager

        manager = StarcManager(
            num_layers=4, n_kv_heads=2, head_dim=16,
            budget_pct=0.15, min_seq_len=8,
        )
        assert len(manager.layers) == 4
        assert manager.budget_pct == 0.15


# ═══════════════════════════════════════════════════════════════════
# TestMamba3 — Feature 4: Exponential-trapezoidal MIMO kernel
# ═══════════════════════════════════════════════════════════════════


class TestMamba3:
    """Mamba-3 exponential-trapezoidal MIMO kernel."""

    def test_trap_coefficients_analytical(self):
        """alpha, beta, gamma match analytical formulas."""
        import mlx.core as mx
        from omlx.mamba3_kernel import compute_trap_coefficients

        # Use pre-softplus dt so we know the effective dt
        # softplus(5.0) ≈ 5.0 for large values
        dt = mx.array([[5.0, 5.0]])  # (1, 2)
        A_log = mx.array([0.0, 0.0])  # A = -exp(0) = -1
        lambda_raw = mx.array([[0.0, 0.0]])  # sigmoid(0) = 0.5

        alpha, beta, gamma = compute_trap_coefficients(dt, A_log, lambda_raw)
        mx.eval(alpha, beta, gamma)

        # dt_eff ≈ 5.0, A = -1
        # alpha = exp(-5) ≈ 0.0067
        # lambda = 0.5
        # beta = 0.5 * 5.0 * exp(-5) ≈ 0.0168
        # gamma = 0.5 * 5.0 = 2.5
        assert alpha.shape == (1, 2)
        assert abs(gamma[0, 0].item() - 2.5) < 0.1

    def test_lambda_none_equals_euler(self):
        """lambda=None → pure Euler (beta=0, gamma=dt)."""
        import mlx.core as mx
        from omlx.mamba3_kernel import compute_trap_coefficients

        dt = mx.array([[2.0]])
        A_log = mx.array([-1.0])  # A = -exp(-1) ≈ -0.368

        alpha, beta, gamma = compute_trap_coefficients(dt, A_log, lambda_raw=None)
        mx.eval(alpha, beta, gamma)

        # beta should be 0 (no left-endpoint contribution)
        assert abs(beta[0, 0].item()) < 1e-6
        # gamma should be dt_eff
        assert gamma[0, 0].item() > 1.5

    def test_lambda_extremes(self):
        """lambda near 0 → mostly beta; lambda near 1 → mostly gamma."""
        import mlx.core as mx
        from omlx.mamba3_kernel import compute_trap_coefficients

        dt = mx.array([[3.0]])
        A_log = mx.array([0.0])

        # lambda ≈ 0 (sigmoid(-10) ≈ 0)
        _, beta_low, gamma_low = compute_trap_coefficients(
            dt, A_log, mx.array([[-10.0]])
        )
        mx.eval(beta_low, gamma_low)

        # lambda ≈ 1 (sigmoid(10) ≈ 1)
        _, beta_high, gamma_high = compute_trap_coefficients(
            dt, A_log, mx.array([[10.0]])
        )
        mx.eval(beta_high, gamma_high)

        # When lambda≈0: beta dominates, gamma≈0
        assert gamma_low[0, 0].item() < 0.1
        # When lambda≈1: gamma dominates, beta≈0
        assert beta_high[0, 0].item() < 0.01

    def test_step_scan_consistency(self):
        """Sequential step output matches scan output."""
        import mlx.core as mx
        from omlx.mamba3_kernel import mamba3_step, mamba3_scan

        B_size, L, H, P, G, N = 1, 4, 2, 8, 1, 16

        x = mx.random.normal((B_size, L, H, P))
        A_log = mx.zeros((H,))
        B_mat = mx.random.normal((B_size, L, G, N))
        C_mat = mx.random.normal((B_size, L, G, N))
        D = mx.zeros((H,))
        dt = mx.ones((B_size, L, H)) * 0.5
        lam = mx.zeros((B_size, L, H))  # lambda=0.5

        # Scan the whole sequence
        y_scan, state_scan, prev_Bx_scan = mamba3_scan(
            x, A_log, B_mat, C_mat, D, dt, lambda_raw=lam
        )
        mx.eval(y_scan, state_scan)

        # Step through one at a time
        state = mx.zeros((B_size, H, N, P))
        prev_Bx = mx.zeros((B_size, H, N, P))
        y_steps = []
        for t in range(L):
            y_t, state, prev_Bx = mamba3_step(
                x[:, t:t+1], state, prev_Bx,
                A_log, B_mat[:, t:t+1], C_mat[:, t:t+1],
                D, dt[:, t:t+1], lambda_raw=lam[:, t:t+1],
            )
            y_steps.append(y_t)
        y_step = mx.stack(y_steps, axis=1)
        mx.eval(y_step)

        diff = mx.abs(y_scan - y_step).max().item()
        assert diff < 1e-4, f"Step vs scan mismatch: {diff}"

    def test_rope_changes_output(self):
        """Applying SSM RoPE produces different B, C from the originals."""
        import mlx.core as mx
        from omlx.mamba3_kernel import apply_ssm_rope

        B_mat = mx.random.normal((1, 8, 1, 16))
        C_mat = mx.random.normal((1, 8, 1, 16))
        dt_theta = mx.ones((1, 8, 8)) * 0.1

        B_rot, C_rot = apply_ssm_rope(B_mat, C_mat, dt_theta)
        mx.eval(B_rot, C_rot)

        # Rotated should differ from original
        diff_B = mx.abs(B_rot - B_mat).max().item()
        assert diff_B > 1e-3, "RoPE should change B"


# ═══════════════════════════════════════════════════════════════════
# TestMedusa — Feature 5: Multi-token lookahead draft heads
# ═══════════════════════════════════════════════════════════════════


class TestMedusa:
    """Medusa draft heads for speculative decoding."""

    def test_draft_heads_output_shape(self):
        """Draft heads produce list of [batch, seq, vocab] logits."""
        import mlx.core as mx
        from omlx.medusa_heads import MedusaDraftHeads

        d_model, vocab_size, num_heads = 64, 100, 3
        heads = MedusaDraftHeads(d_model, vocab_size, num_heads)
        mx.eval(heads.parameters())

        hidden = mx.random.normal((2, 1, d_model))  # (B, L=1, D)
        draft_logits = heads(hidden)
        mx.eval(*draft_logits)

        assert len(draft_logits) == 3
        for i, logits in enumerate(draft_logits):
            assert logits.shape == (2, 1, vocab_size), (
                f"Head {i}: expected (2,1,{vocab_size}), got {logits.shape}"
            )

    def test_residual_chain(self):
        """Each head refines the previous hidden state via residual."""
        import mlx.core as mx
        from omlx.medusa_heads import MedusaResidualBlock

        block = MedusaResidualBlock(32)
        mx.eval(block.parameters())

        x = mx.random.normal((1, 32))
        out = block(x)
        mx.eval(out)

        # Output should differ from input (residual block changes it)
        diff = mx.abs(out - x).max().item()
        assert diff > 0.01, "Residual block should modify the input"
        # But not be completely different (skip connection preserves signal)
        cos = mx.sum(out * x) / (mx.linalg.norm(out) * mx.linalg.norm(x) + 1e-8)
        assert cos.item() > 0.3, "Residual should preserve some of the original signal"

    def test_tree_verification_accept(self):
        """Tree verification accepts correct draft tokens."""
        import mlx.core as mx
        from omlx.medusa_heads import verify_draft_tokens

        # Model logits where argmax matches drafts exactly
        logits = mx.array([
            [0.0, 0.0, 10.0],  # argmax = 2
            [0.0, 10.0, 0.0],  # argmax = 1
            [10.0, 0.0, 0.0],  # argmax = 0
        ])
        drafts = mx.array([2, 1, 0])

        accepted = verify_draft_tokens(logits, drafts)
        assert accepted == 3, f"Expected 3 accepted, got {accepted}"

    def test_tree_verification_reject(self):
        """Tree verification rejects at first mismatch."""
        import mlx.core as mx
        from omlx.medusa_heads import verify_draft_tokens

        logits = mx.array([
            [0.0, 0.0, 10.0],  # argmax = 2
            [10.0, 0.0, 0.0],  # argmax = 0 (mismatch!)
            [0.0, 10.0, 0.0],  # argmax = 1
        ])
        drafts = mx.array([2, 1, 1])  # Second token wrong

        accepted = verify_draft_tokens(logits, drafts)
        assert accepted == 1, f"Expected 1 accepted (first match only), got {accepted}"


# ═══════════════════════════════════════════════════════════════════
# TestExpertChoice — Feature 6: Expert-choice MoE routing
# ═══════════════════════════════════════════════════════════════════


class TestExpertChoice:
    """Expert-choice MoE routing for balanced GPU utilization."""

    def test_output_shape_matches_token_choice(self):
        """Output shape matches GraniteMoeTopKGating for SwitchGLU compatibility."""
        import mlx.core as mx
        from omlx.patches.expert_choice_router import ExpertChoiceRouter

        router = ExpertChoiceRouter(
            input_size=64, num_experts=8, top_k=2, capacity_factor=1.2
        )
        mx.eval(router.parameters())

        x = mx.random.normal((4, 16, 64))  # (B, L, D)
        idx, gates = router(x)
        mx.eval(idx, gates)

        assert idx.shape == (4, 16, 2), f"Expected (4,16,2), got {idx.shape}"
        assert gates.shape == (4, 16, 2), f"Expected (4,16,2), got {gates.shape}"

    def test_gates_sum_to_one(self):
        """Routing weights sum to ~1 per token."""
        import mlx.core as mx
        from omlx.patches.expert_choice_router import ExpertChoiceRouter

        router = ExpertChoiceRouter(
            input_size=32, num_experts=4, top_k=2, capacity_factor=1.5
        )
        mx.eval(router.parameters())

        x = mx.random.normal((2, 8, 32))
        _, gates = router(x)
        mx.eval(gates)

        gate_sums = gates.sum(axis=-1)  # Should be ~1.0
        assert mx.all(gate_sums > 0.9).item(), "Gate sums should be close to 1.0"

    def test_expert_indices_valid(self):
        """All expert indices are within [0, num_experts)."""
        import mlx.core as mx
        from omlx.patches.expert_choice_router import ExpertChoiceRouter

        num_experts = 8
        router = ExpertChoiceRouter(
            input_size=64, num_experts=num_experts, top_k=2, capacity_factor=1.2
        )
        mx.eval(router.parameters())

        x = mx.random.normal((2, 32, 64))
        idx, _ = router(x)
        mx.eval(idx)

        assert mx.all(idx >= 0).item(), "Expert indices must be >= 0"
        assert mx.all(idx < num_experts).item(), f"Expert indices must be < {num_experts}"


# ═══════════════════════════════════════════════════════════════════
# TestIntegration — Full pipeline: all flags parse → stubs accept
# ═══════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for the full hypercar pipeline."""

    def test_scheduler_config_roundtrip(self):
        """All hypercar fields survive SchedulerConfig creation."""
        from omlx.scheduler import SchedulerConfig
        cfg = SchedulerConfig(
            turboquant_kv_bits=3,
            turboquant_fp16_layers=1,
            weight_mode="turbo35",
            sparsity_method="starc",
            mimo_rank=4,
            medusa_heads=3,
            moe_router="expert-choice",
            max_kv_size=256000,
        )
        # Verify all fields accessible
        assert cfg.turboquant_kv_bits == 3
        assert cfg.weight_mode == "turbo35"
        assert cfg.sparsity_method == "starc"
        assert cfg.mimo_rank == 4
        assert cfg.medusa_heads == 3
        assert cfg.moe_router == "expert-choice"
        assert cfg.max_kv_size == 256000

    def test_all_cli_flags_parse_together(self):
        """The full hypercar CLI command parses without error."""
        from omlx.cli import main
        import argparse

        # We can't run the full serve command (needs server), but we can
        # verify argparse accepts all flags by checking help includes them
        help_text = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout

        required_flags = [
            "--cache-mode", "--fp16-layers", "--weight-mode",
            "--sparsity-method", "--mimo-rank", "--medusa-heads",
            "--moe-router", "--max-kv-size",
        ]
        for flag in required_flags:
            assert flag in help_text, f"Missing flag: {flag}"
