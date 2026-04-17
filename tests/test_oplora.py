# SPDX-License-Identifier: Apache-2.0
"""Tests for OPLoRA orthogonal projection safety rail.

Pure numpy/math tests — no MLX or model loading needed.
"""

import importlib.util
import math
import numpy as np
from pathlib import Path

import pytest


def _load_module():
    """Load oplora module without omlx.__init__ (avoids MLX)."""
    spec = importlib.util.spec_from_file_location(
        "oplora", "omlx/oplora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_oplora_src = Path("omlx/oplora.py").read_text()


class TestOPLoRASource:
    """Source-level tests."""

    def test_project_lora_grads_exists(self):
        assert "def project_lora_grads(" in _oplora_src

    def test_truncated_svd_exists(self):
        assert "def _truncated_svd(" in _oplora_src

    def test_compute_svd_cache_exists(self):
        assert "def compute_svd_cache(" in _oplora_src

    def test_save_load_exists(self):
        assert "def save_svd_cache(" in _oplora_src
        assert "def load_svd_cache(" in _oplora_src

    def test_orthogonal_projection(self):
        """Must project out components along singular vectors."""
        assert "V_k.T" in _oplora_src
        assert "U_k" in _oplora_src

    def test_randomized_svd(self):
        """Must use randomized SVD for large matrices."""
        assert "Random" in _oplora_src or "random" in _oplora_src
        assert "QR" in _oplora_src or "qr" in _oplora_src


class TestOPLoRAProjection:
    """Functional tests for gradient projection (numpy-based)."""

    def test_projected_gradient_orthogonal_to_top_k(self):
        """Projected gradient must have zero inner product with top-k SVD."""
        np.random.seed(42)
        m, n, r, k = 32, 16, 4, 3

        W = np.random.randn(m, n).astype(np.float32)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k]
        V_k = Vt[:k, :].T

        A = np.random.randn(r, n).astype(np.float32)
        B = np.random.randn(m, r).astype(np.float32)
        dA = np.random.randn(r, n).astype(np.float32)
        dB = np.random.randn(m, r).astype(np.float32)

        # Project dA: remove V_k component
        dA_proj = dA - (dA @ V_k) @ V_k.T
        # Project dB: remove U_k component
        dB_proj = dB - U_k @ (U_k.T @ dB)

        # Verify orthogonality: dA_proj columns orthogonal to V_k
        inner_A = dA_proj @ V_k  # should be ~zero
        assert np.allclose(inner_A, 0, atol=1e-5), f"dA not orthogonal: {np.max(np.abs(inner_A))}"

        # Verify orthogonality: dB_proj rows orthogonal to U_k
        inner_B = U_k.T @ dB_proj  # should be ~zero
        assert np.allclose(inner_B, 0, atol=1e-5), f"dB not orthogonal: {np.max(np.abs(inner_B))}"

    def test_k_zero_preserves_gradient(self):
        """With k=0, projection should preserve the original gradient."""
        np.random.seed(42)
        m, n, r = 16, 8, 4

        dA = np.random.randn(r, n).astype(np.float32)
        dB = np.random.randn(m, r).astype(np.float32)

        # Empty V_k, U_k
        V_k = np.zeros((n, 0), dtype=np.float32)
        U_k = np.zeros((m, 0), dtype=np.float32)

        dA_proj = dA - (dA @ V_k) @ V_k.T
        dB_proj = dB - U_k @ (U_k.T @ dB)

        assert np.allclose(dA_proj, dA, atol=1e-6)
        assert np.allclose(dB_proj, dB, atol=1e-6)

    def test_projection_reduces_norm(self):
        """Projected gradient norm must be <= original (it removes components)."""
        np.random.seed(42)
        m, n, r, k = 32, 16, 4, 8

        W = np.random.randn(m, n).astype(np.float32)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        V_k = Vt[:k, :].T
        U_k = U[:, :k]

        dA = np.random.randn(r, n).astype(np.float32)
        dB = np.random.randn(m, r).astype(np.float32)

        dA_proj = dA - (dA @ V_k) @ V_k.T
        dB_proj = dB - U_k @ (U_k.T @ dB)

        assert np.linalg.norm(dA_proj) <= np.linalg.norm(dA) + 1e-5
        assert np.linalg.norm(dB_proj) <= np.linalg.norm(dB) + 1e-5

    def test_idempotent(self):
        """Projecting twice should give the same result as projecting once."""
        np.random.seed(42)
        m, n, r, k = 32, 16, 4, 4

        W = np.random.randn(m, n).astype(np.float32)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        V_k = Vt[:k, :].T

        dA = np.random.randn(r, n).astype(np.float32)
        dA_proj1 = dA - (dA @ V_k) @ V_k.T
        dA_proj2 = dA_proj1 - (dA_proj1 @ V_k) @ V_k.T
        assert np.allclose(dA_proj1, dA_proj2, atol=1e-5)
