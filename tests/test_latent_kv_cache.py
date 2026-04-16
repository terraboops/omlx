# SPDX-License-Identifier: Apache-2.0
"""Tests for LatentKVCache — SVD-projected low-rank KV cache.

Pure Python/numpy tests — no MLX or GPU required.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Load module without triggering omlx package imports
_spec = importlib.util.spec_from_file_location(
    "omlx.latent_kv_cache", "omlx/latent_kv_cache.py")
lkv_mod = importlib.util.module_from_spec(_spec)
sys.modules["omlx.latent_kv_cache"] = lkv_mod
_spec.loader.exec_module(lkv_mod)

LatentProjection = lkv_mod.LatentProjection
memory_estimate_gb = lkv_mod.memory_estimate_gb


class TestLatentProjection:
    """Test the LatentProjection compress/expand round-trip."""

    def _make_projection(self, kv_dim=1024, d_c=241):
        """Build a projection from random orthogonal matrices."""
        # Use SVD of random matrix to get valid projection
        rng = np.random.default_rng(42)
        A = rng.standard_normal((kv_dim, kv_dim))
        U, _, Vt = np.linalg.svd(A, full_matrices=True)
        W_down = Vt[:d_c].T  # (kv_dim, d_c)
        W_up = Vt[:d_c]       # (d_c, kv_dim)
        return LatentProjection(
            W_down=W_down,
            W_up=W_up,
            kv_dim=kv_dim,
            d_c=d_c,
            layer_idx=0,
        )

    def test_compress_shape(self):
        proj = self._make_projection(kv_dim=1024, d_c=241)
        kv = np.random.randn(1, 100, 1024)  # (B, T, kv_dim)
        latent = proj.compress(kv)
        assert latent.shape == (1, 100, 241)

    def test_expand_shape(self):
        proj = self._make_projection(kv_dim=1024, d_c=241)
        latent = np.random.randn(1, 100, 241)  # (B, T, d_c)
        kv = proj.expand(latent)
        assert kv.shape == (1, 100, 1024)

    def test_round_trip_preserves_projection(self):
        """Compress then expand should approximately reconstruct within d_c subspace."""
        proj = self._make_projection(kv_dim=512, d_c=100)
        # Generate data that lives in the d_c subspace
        latent_orig = np.random.randn(1, 50, 100)
        kv_in_subspace = proj.expand(latent_orig)
        # Round trip
        latent_recovered = proj.compress(kv_in_subspace)
        kv_recovered = proj.expand(latent_recovered)
        # Should be nearly exact since data was in the subspace
        np.testing.assert_allclose(kv_recovered, kv_in_subspace, atol=1e-10)

    def test_compression_ratio(self):
        """d_c < kv_dim means we're actually compressing."""
        proj = self._make_projection(kv_dim=1024, d_c=241)
        assert proj.d_c < proj.kv_dim
        ratio = proj.d_c / proj.kv_dim
        assert 0.2 < ratio < 0.3  # 24% compression

    def test_reconstruction_error_on_random_data(self):
        """Random data should have bounded reconstruction error."""
        proj = self._make_projection(kv_dim=512, d_c=256)
        kv = np.random.randn(1, 100, 512)
        latent = proj.compress(kv)
        reconstructed = proj.expand(latent)
        # With d_c=256 out of 512, we capture ~50% of random variance
        rel_error = np.mean((kv - reconstructed) ** 2) / np.mean(kv ** 2)
        assert rel_error < 0.6  # Less than 60% relative error

    def test_higher_d_c_means_lower_error(self):
        """More latent dimensions should mean lower reconstruction error."""
        kv = np.random.randn(1, 100, 512)

        proj_low = self._make_projection(kv_dim=512, d_c=64)
        proj_high = self._make_projection(kv_dim=512, d_c=256)

        err_low = np.mean((kv - proj_low.expand(proj_low.compress(kv))) ** 2)
        err_high = np.mean((kv - proj_high.expand(proj_high.compress(kv))) ** 2)

        assert err_high < err_low

    def test_layer_idx_stored(self):
        proj = self._make_projection()
        assert proj.layer_idx == 0


class TestMemoryEstimate:
    """Test memory projection calculations."""

    def test_qwen3_coder_1m_fp16(self):
        """48 layers, 1M tokens, d_c=241, fp16."""
        mem = memory_estimate_gb(48, 1_000_000, 241, bytes_per_elem=2.0)
        # 48 × 1M × 241 × 2 = 23.1 GB
        assert 22.0 < mem < 24.0

    def test_qwen3_coder_1m_3bit(self):
        """48 layers, 1M tokens, d_c=241, 3-bit quantized."""
        mem = memory_estimate_gb(48, 1_000_000, 241, bytes_per_elem=0.375)
        # 48 × 1M × 241 × 0.375 = 4.3 GB
        assert 3.5 < mem < 5.0

    def test_scales_linearly_with_context(self):
        mem_16k = memory_estimate_gb(48, 16384, 241)
        mem_64k = memory_estimate_gb(48, 65536, 241)
        ratio = mem_64k / mem_16k
        assert 3.9 < ratio < 4.1  # Should be 4x

    def test_zero_context_is_zero(self):
        assert memory_estimate_gb(48, 0, 241) == 0.0

    def test_higher_d_c_means_more_memory(self):
        mem_low = memory_estimate_gb(48, 100000, 128)
        mem_high = memory_estimate_gb(48, 100000, 256)
        assert mem_high > mem_low


class TestProjectionDimensions:
    """Test that projection dimensions are consistent."""

    def test_w_down_shape(self):
        proj = LatentProjection(
            W_down=np.zeros((1024, 241)),
            W_up=np.zeros((241, 1024)),
            kv_dim=1024, d_c=241, layer_idx=5,
        )
        assert proj.W_down.shape == (1024, 241)
        assert proj.W_up.shape == (241, 1024)

    def test_kv_dim_matches_qwen3_coder(self):
        """Qwen3-Coder: H_kv=4, D=128, joint K+V → kv_dim = 4*128*2 = 1024."""
        kv_dim = 4 * 128 * 2
        assert kv_dim == 1024
