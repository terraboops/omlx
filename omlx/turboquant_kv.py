# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache: codebook-quantized KV with fused Flash Attention.

Key design:
  - MSE codec: rotation → codebook quantization (per-coordinate optimal)
  - 2-pass fused SDPA kernel: score + softmax + weighted_sum in GPU
  - 8K+ context: faster than fp16 SDPA (bandwidth reduction)
  - Memory: ~70% reduction vs fp16 KV cache
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import _BaseCache


# ---------------------------------------------------------------------------
# Codebook generation (TurboQuant paper: arXiv:2504.19874)
#
# The correct distribution for coordinates of a randomly rotated unit vector
# in R^d has density: (1-x^2)^((d-3)/2) on [-1, 1].
# This corresponds to Beta((d-1)/2, (d-1)/2) after rescaling from [0,1]→[-1,1].
# The codebook is DATA-INDEPENDENT — depends only on dim and bits.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _codebook(dim: int, bits: int) -> mx.array:
    """Optimal scalar codebook for TurboQuant's coordinate distribution.

    Per the paper (arXiv:2504.19874), after WHT rotation each coordinate of
    a unit vector follows density f(x) = C * (1-x^2)^((d-3)/2) on [-1,1].
    This is Beta((d-1)/2, (d-1)/2) mapped from [0,1] to [-1,1].
    """
    n_levels = 1 << bits
    # Correct parameter: (d-1)/2, NOT d/2
    alpha = (dim - 1) / 2.0
    rng = np.random.default_rng(seed=0)
    samples = 2.0 * rng.beta(alpha, alpha, size=200_000) - 1.0
    # Lloyd-Max optimal quantizer
    centroids = np.linspace(samples.min(), samples.max(), n_levels)
    for _ in range(200):  # More iterations for better convergence
        dists = np.abs(samples[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)
        for j in range(n_levels):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = samples[mask].mean()
    return mx.array(sorted(centroids), dtype=mx.float32)


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (replaces broken Givens rotation)
#
# WHT spreads energy across ALL dimensions via butterfly operations.
# Combined with random sign flips, it's equivalent to a random orthogonal
# rotation but O(d log d) instead of O(d^2). This is what makes the Beta
# distribution codebook valid — Givens only mixed pairs and didn't decorrelate.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _random_signs(dim: int, seed: int = 0) -> mx.array:
    """Random ±1 sign vector for randomized WHT."""
    key = mx.random.key(seed)
    uniform = mx.random.uniform(shape=(dim,), key=key)
    signs = mx.where(uniform > 0.5, mx.ones(dim), -mx.ones(dim)).astype(mx.float32)
    mx.eval(signs)
    return signs


def _wht(x: mx.array) -> mx.array:
    """Walsh-Hadamard Transform via in-place butterfly.

    Input: (..., D) where D must be a power of 2.
    Output: (..., D) transformed, normalized by 1/sqrt(D).
    """
    shape = x.shape
    D = shape[-1]
    # Reshape to 2D for processing
    flat = x.reshape(-1, D).astype(mx.float32)

    # Butterfly passes
    h = 1
    while h < D:
        # Split into blocks of 2h, combine pairs separated by h
        flat_r = flat.reshape(-1, D // (2 * h), 2, h)
        a = flat_r[:, :, 0, :]  # first half
        b = flat_r[:, :, 1, :]  # second half
        flat_r = mx.stack([a + b, a - b], axis=2)
        flat = flat_r.reshape(-1, D)
        h *= 2

    # Normalize
    flat = flat / math.sqrt(D)
    return flat.reshape(shape)


def _apply_wht_rotation(x: mx.array, signs: mx.array) -> mx.array:
    """Randomized WHT: sign flip → WHT. This decorrelates all dimensions."""
    return _wht(x * signs)


def _apply_wht_rotation_inverse(x: mx.array, signs: mx.array) -> mx.array:
    """Inverse randomized WHT: WHT → sign flip (WHT is its own inverse)."""
    return _wht(x) * signs


# Legacy Givens functions (kept for backward compatibility with fused kernels)
@lru_cache(maxsize=16)
def _rotation_matrix(dim: int, seed: int) -> mx.array:
    """Random orthogonal rotation via QR decomposition (legacy)."""
    key = mx.random.key(seed)
    Q, R = mx.linalg.qr(mx.random.normal(shape=(dim, dim), key=key), stream=mx.cpu)
    signs = mx.sign(mx.diag(R))
    signs = mx.where(signs == 0, mx.ones_like(signs), signs)
    Q = (Q * signs[None, :]).astype(mx.float32)
    mx.eval(Q)
    return Q


@lru_cache(maxsize=16)
def _givens_angles(dim: int, seed: int = 0):
    """PlanarQuant/RotorQuant: random Givens rotation angles (LEGACY — use WHT instead)."""
    key = mx.random.key(seed)
    n_pairs = dim // 2
    angles = mx.random.uniform(shape=(n_pairs,), key=key) * 2.0 * 3.14159265
    cos_a = mx.cos(angles).astype(mx.float32)
    sin_a = mx.sin(angles).astype(mx.float32)
    mx.eval(cos_a, sin_a)
    return cos_a, sin_a


def _apply_givens(x: mx.array, cos_a: mx.array, sin_a: mx.array) -> mx.array:
    """Apply D/2 Givens rotations: O(D) per vector (LEGACY)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = cos_a * x_even - sin_a * x_odd
    y_odd = sin_a * x_even + cos_a * x_odd
    return mx.stack([y_even, y_odd], axis=-1).reshape(*x.shape)


def _apply_givens_inverse(x: mx.array, cos_a: mx.array, sin_a: mx.array) -> mx.array:
    """Apply inverse Givens rotation (LEGACY)."""
    return _apply_givens(x, cos_a, -sin_a)


# ---------------------------------------------------------------------------
# Contiguous bit packing (compatible with mlx-vlm / Flash Attention kernel)
# ---------------------------------------------------------------------------

def _packed_width(dim: int, bits: int) -> int:
    return (dim * bits + 31) // 32


@lru_cache(maxsize=None)
def _pack_lowbit_kernel():
    """Metal kernel for contiguous bit packing."""
    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) return;

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) packed_word |= value << offset;
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) packed_word |= value >> (Bits - spill);
            }
        }
        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="tq_pack_lowbit",
        input_names=["values"],
        output_names=["out"],
        source=source,
    )


def _pack_contiguous(indices: mx.array, bits: int, dim: int) -> mx.array:
    """Pack indices using contiguous bit packing via Metal kernel."""
    batch_shape = indices.shape[:-1]
    flat = indices.reshape(-1, dim).astype(mx.uint32)
    pw = _packed_width(dim, bits)
    rows = flat.shape[0]

    kernel = _pack_lowbit_kernel()
    packed = kernel(
        inputs=[flat],
        output_shapes=[(rows, pw)],
        output_dtypes=[mx.uint32],
        grid=(pw, rows, 1),
        threadgroup=(min(pw, 32), 1, 1),
        template=[("Bits", bits), ("Length", dim), ("PackedWidth", pw)],
        init_value=0,
    )[0]
    return packed.reshape(*batch_shape, pw)


@lru_cache(maxsize=None)
def _fused_quantize_kernel():
    """Fused: norm + normalize + rotate + boundary + pack in ONE kernel (dense rotation)."""
    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= vectors_shape[0] || word >= PackedWidth) return;

        auto vec = vectors + row * Dim;

        // Step 1: Compute norm
        float norm_sq = 0.0f;
        for (int d = 0; d < Dim; d++) {
            float v = static_cast<float>(vec[d]);
            norm_sq += v * v;
        }
        float norm = sqrt(norm_sq);
        float inv_norm = norm > 1e-10f ? 1.0f / norm : 1.0f;

        if (word == 0) {
            out_norms[row] = norm;
        }

        // Step 2+3+4: rotate (dense) + boundary + pack
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Dim, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            // Dense rotation: O(Dim) per output coordinate
            float rotated_val = 0.0f;
            for (int j = 0; j < Dim; j++) {
                rotated_val += static_cast<float>(vec[j]) * inv_norm * rotation[j * Dim + idx];
            }

            uint quant_idx = 0u;
            for (int b = 0; b < NLevels - 1; b++) {
                if (rotated_val > boundaries[b]) quant_idx++;
            }

            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            if (word_idx == word) {
                packed_word |= (quant_idx & ((1u << Bits) - 1u)) << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= (quant_idx & ((1u << Bits) - 1u)) >> (Bits - spill);
                }
            }
        }

        out_packed[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="tq_fused_quantize",
        input_names=["vectors", "rotation", "boundaries"],
        output_names=["out_packed", "out_norms"],
        source=source,
    )


@lru_cache(maxsize=None)
def _fused_quantize_givens_kernel():
    """Fused: norm + Givens rotate + boundary + pack in ONE kernel.

    RotorQuant/PlanarQuant variant: uses D/2 Givens pair rotations instead
    of dense D×D matrix. Each coordinate only depends on its paired coordinate.
    O(1) per output (2 multiplies + 1 add) instead of O(D).
    """
    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= vectors_shape[0] || word >= PackedWidth) return;

        auto vec = vectors + row * Dim;

        // Step 1: Compute norm
        float norm_sq = 0.0f;
        for (int d = 0; d < Dim; d++) {
            float v = static_cast<float>(vec[d]);
            norm_sq += v * v;
        }
        float norm = sqrt(norm_sq);
        float inv_norm = norm > 1e-10f ? 1.0f / norm : 1.0f;

        if (word == 0) {
            out_norms[row] = norm;
        }

        // Step 2+3+4: Givens rotate + boundary + pack
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Dim, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            // Givens rotation: O(1) per coordinate — just 2 muls + 1 add
            // Pair (2i, 2i+1) rotated by angle[i]:
            //   rotated[2i]   = cos[i] * norm_a - sin[i] * norm_b
            //   rotated[2i+1] = sin[i] * norm_a + cos[i] * norm_b
            int pair = idx / 2;
            float a = static_cast<float>(vec[2 * pair]) * inv_norm;
            float b = static_cast<float>(vec[2 * pair + 1]) * inv_norm;
            float c = cos_angles[pair];
            float s = sin_angles[pair];
            float rotated_val = (idx % 2 == 0)
                ? (c * a - s * b)
                : (s * a + c * b);

            // Boundary quantize
            uint quant_idx = 0u;
            for (int bb = 0; bb < NLevels - 1; bb++) {
                if (rotated_val > boundaries[bb]) quant_idx++;
            }

            // Pack into word
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            if (word_idx == word) {
                packed_word |= (quant_idx & ((1u << Bits) - 1u)) << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= (quant_idx & ((1u << Bits) - 1u)) >> (Bits - spill);
                }
            }
        }

        out_packed[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="tq_fused_quantize_givens",
        input_names=["vectors", "cos_angles", "sin_angles", "boundaries"],
        output_names=["out_packed", "out_norms"],
        source=source,
    )


def _fused_quantize(vectors: mx.array, rotation: mx.array, boundaries: mx.array,
                     bits: int, dim: int) -> tuple:
    """Fused quantize: norm + dense rotate + boundary + pack in one Metal dispatch."""
    batch_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, dim)
    rows = flat.shape[0]
    pw = _packed_width(dim, bits)
    n_levels = 1 << bits

    kernel = _fused_quantize_kernel()
    packed, norms = kernel(
        inputs=[flat.astype(mx.float16), rotation.astype(mx.float32), boundaries.astype(mx.float32)],
        output_shapes=[(rows, pw), (rows,)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(pw, rows, 1),
        threadgroup=(min(pw, 32), 1, 1),
        template=[("Bits", bits), ("Dim", dim), ("PackedWidth", pw), ("NLevels", n_levels)],
        init_value=0,
    )
    return norms.reshape(*batch_shape), packed.reshape(*batch_shape, pw)


def _fused_quantize_givens(vectors: mx.array, cos_a: mx.array, sin_a: mx.array,
                           boundaries: mx.array, bits: int, dim: int) -> tuple:
    """Fused quantize with Givens rotation: norm + Givens rotate + boundary + pack.

    64x fewer FLOPs in the rotation step vs dense matrix. The Givens rotation
    happens entirely in GPU registers — zero memory traffic for the rotation.
    """
    batch_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, dim)
    rows = flat.shape[0]
    pw = _packed_width(dim, bits)
    n_levels = 1 << bits

    kernel = _fused_quantize_givens_kernel()
    packed, norms = kernel(
        inputs=[flat.astype(mx.float16), cos_a.astype(mx.float32),
                sin_a.astype(mx.float32), boundaries.astype(mx.float32)],
        output_shapes=[(rows, pw), (rows,)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(pw, rows, 1),
        threadgroup=(min(pw, 32), 1, 1),
        template=[("Bits", bits), ("Dim", dim), ("PackedWidth", pw), ("NLevels", n_levels)],
        init_value=0,
    )
    return norms.reshape(*batch_shape), packed.reshape(*batch_shape, pw)


@lru_cache(maxsize=None)
def _unpack_lowbit_kernel():
    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;
        if (row >= packed_shape[0] || idx >= Length) return;

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) value |= packed_ptr[word_idx + 1] << (Bits - spill);
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """
    return mx.fast.metal_kernel(
        name="tq_unpack_lowbit",
        input_names=["packed"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _fused_dequant_wht_kernel():
    """Fused dequant: unpack 3-bit → codebook → inverse WHT → scale by norm.

    One Metal dispatch replaces 4 separate operations:
      1. Bit unpacking from uint32 words
      2. Codebook lookup (8 entries in thread-local registers)
      3. Inverse WHT via dense matrix multiply (R^T @ coords)
      4. Multiply by per-vector norm

    Each thread handles QK_PER_THREAD dimensions of one vector.
    SIMD group (32 threads) covers the full D=128 dimensions.
    Output is fp16, ready for AMX matmul.
    """
    source = r"""
        auto simd_lid = thread_index_in_simdgroup;
        auto row = threadgroup_position_in_grid.x;

        if (row >= packed_shape[0]) return;

        auto packed_ptr = packed + row * PackedWidth;
        float norm = static_cast<float>(norms[row]);

        // Step 1+2: Unpack and codebook lookup for QK_PER_THREAD coordinates
        float coords[QK_PER_THREAD];
        for (int j = 0; j < QK_PER_THREAD; j++) {
            int d = simd_lid * QK_PER_THREAD + j;
            int bit_off = d * Bits;
            int word = bit_off / 32;
            int off = bit_off % 32;
            uint val = packed_ptr[word] >> off;
            int spill = off + Bits - 32;
            if (spill > 0) val |= packed_ptr[word + 1] << (Bits - spill);
            val &= ((1u << Bits) - 1u);
            coords[j] = codebook[val];
        }

        // Step 3: Inverse WHT via dense matrix R^T
        // out[d] = sum_j(coords[j] * R^T[j, d]) = sum_j(coords[j] * R[d, j])
        // Each thread computes QK_PER_THREAD output dimensions
        float restored[QK_PER_THREAD];
        for (int j = 0; j < QK_PER_THREAD; j++) {
            int out_d = simd_lid * QK_PER_THREAD + j;
            float val = 0.0f;
            // Accumulate across all input dimensions via SIMD shuffle
            // Each thread has QK_PER_THREAD coords; broadcast to all threads
            for (int src_lane = 0; src_lane < 32; src_lane++) {
                for (int k = 0; k < QK_PER_THREAD; k++) {
                    int src_d = src_lane * QK_PER_THREAD + k;
                    float src_coord = simd_shuffle(coords[k], src_lane);
                    val += src_coord * rotation_t[src_d * Dim + out_d];
                }
            }
            restored[j] = val * norm;
        }

        // Step 4: Write fp16 output
        for (int j = 0; j < QK_PER_THREAD; j++)
            out[row * Dim + simd_lid * QK_PER_THREAD + j] = static_cast<half>(restored[j]);
    """
    return mx.fast.metal_kernel(
        name="tq_fused_dequant_wht",
        input_names=["packed", "norms", "codebook", "rotation_t"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _fused_dequant_wht(packed: mx.array, norms: mx.array,
                        codebook: mx.array, rotation_t: mx.array,
                        bits: int, dim: int) -> mx.array:
    """Fused dequant: packed 3-bit → fp16 vectors in one Metal dispatch.

    Replaces: _unpack_contiguous + codebook[indices] + WHT inverse + norm scale
    """
    batch_shape = packed.shape[:-1]
    flat_packed = packed.reshape(-1, packed.shape[-1])
    flat_norms = norms.reshape(-1)
    rows = flat_packed.shape[0]
    pw = flat_packed.shape[-1]
    qpt = dim // 32

    kernel = _fused_dequant_wht_kernel()
    out = kernel(
        inputs=[flat_packed.astype(mx.uint32), flat_norms.astype(mx.float32),
                codebook.astype(mx.float32), rotation_t.astype(mx.float32)],
        output_shapes=[(rows, dim)],
        output_dtypes=[mx.float16],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        template=[("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
                  ("QK_PER_THREAD", qpt)],
        init_value=0.0,
    )
    return out[0].reshape(*batch_shape, dim)


def _unpack_contiguous(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack contiguous bit-packed indices via Metal kernel."""
    batch_shape = packed.shape[:-1]
    flat = packed.reshape(-1, packed.shape[-1])
    rows = flat.shape[0]
    pw = flat.shape[-1]

    kernel = _unpack_lowbit_kernel()
    indices = kernel(
        inputs=[flat.astype(mx.uint32)],
        output_shapes=[(rows, dim)],
        output_dtypes=[mx.uint32],
        grid=(dim, rows, 1),
        threadgroup=(min(dim, 32), 1, 1),
        template=[("Bits", bits), ("Length", dim), ("PackedWidth", pw)],
        init_value=0,
    )[0]
    return indices.reshape(*batch_shape, dim)


# ---------------------------------------------------------------------------
# MSE Codec: rotation → nearest codebook → pack
# ---------------------------------------------------------------------------

class TurboQuantMSECodec:
    """TurboQuant MSE-optimal vector quantization codec (arXiv:2504.19874).

    Correct pipeline per the paper:
      1. x → norm * unit_vec  (decompose)
      2. unit_vec → WHT(signs * unit_vec)  (randomized Walsh-Hadamard)
      3. Each coordinate → codebook index  (scalar quantization)
      4. Codebook is data-independent (Beta((d-1)/2, (d-1)/2) Lloyd-Max)

    The key insight: WHT decorrelates ALL dimensions (unlike Givens which
    only mixes pairs), making the Beta distribution codebook valid.
    """

    def __init__(self, dim: int, bits: int, seed: int = 0,
                 use_givens: bool = False, use_wht: bool = True):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = _codebook(dim, bits)
        self.use_wht = use_wht
        self.use_givens = use_givens and not use_wht and (dim % 2 == 0)

        if self.use_wht:
            self._signs = _random_signs(dim, seed)
            # Pre-compute WHT as dense matrix for fused decode kernel compatibility.
            # Forward rotation: y = H @ diag(signs) @ x
            # In the fused kernel, query rotation is: q_rot = q @ R (row-vector convention)
            # q @ R = (R^T @ q^T)^T, so we need R^T = H @ diag(signs), i.e. R = diag(signs) @ H
            # Equivalently: R[i, j] = signs[i] * H[i, j]
            I = mx.eye(dim, dtype=mx.float32)
            H = _wht(I)  # Each row of H is WHT of a basis vector = columns of Hadamard
            self.rotation = (self._signs[:, None] * H)  # diag(signs) @ H
            mx.eval(self.rotation)
        elif self.use_givens:
            self._givens_cos, self._givens_sin = _givens_angles(dim, seed)
        else:
            self.rotation = _rotation_matrix(dim, seed)

        self._pw = _packed_width(dim, bits)
        cb = self.codebook
        self._boundaries = (cb[:-1] + cb[1:]) / 2

    def quantize(self, vectors: mx.array):
        """Quantize vectors: (..., D) → (norms, packed_indices).

        WHT path uses MLX ops (no fused kernel yet — can be added later).
        Givens/dense paths use fused Metal kernels.
        """
        if self.use_wht:
            return self._quantize_wht(vectors)
        elif self.use_givens:
            return _fused_quantize_givens(
                vectors, self._givens_cos, self._givens_sin,
                self._boundaries, self.bits, self.dim,
            )
        else:
            return _fused_quantize(
                vectors, self.rotation, self._boundaries,
                self.bits, self.dim,
            )

    def _quantize_wht(self, vectors: mx.array):
        """Quantize using Walsh-Hadamard Transform (correct per paper)."""
        shape = vectors.shape
        flat = vectors.reshape(-1, self.dim).astype(mx.float32)

        # Step 1: decompose into norm + unit vector
        norms = mx.linalg.norm(flat, axis=-1)
        safe_norms = mx.maximum(norms, 1e-10)
        unit = flat / safe_norms[..., None]

        # Step 2: randomized WHT
        rotated = _apply_wht_rotation(unit, self._signs)

        # Step 3: scalar quantize each coordinate
        # Use searchsorted on boundaries for fast quantization
        indices = mx.zeros(rotated.shape, dtype=mx.uint32)
        for b in range(len(self._boundaries)):
            indices = indices + (rotated > self._boundaries[b]).astype(mx.uint32)

        # Step 4: pack indices
        packed = _pack_contiguous(indices, self.bits, self.dim)

        return norms.reshape(shape[:-1]), packed.reshape(*shape[:-1], self._pw)

    def dequantize(self, norms: mx.array, packed: mx.array) -> mx.array:
        """Dequantize: (norms, packed) → vectors."""
        indices = _unpack_contiguous(packed, self.bits, self.dim)
        coords = self.codebook[indices]

        # Inverse rotate
        if self.use_wht:
            restored = _apply_wht_rotation_inverse(
                coords.astype(mx.float32), self._signs
            )
        elif self.use_givens:
            restored = _apply_givens_inverse(coords, self._givens_cos, self._givens_sin)
        else:
            shape = coords.shape
            grouped = coords.reshape(*shape[:-1], shape[-1] // self.dim, self.dim)
            restored = (grouped @ self.rotation.T).reshape(shape)

        return (restored * norms[..., None]).astype(coords.dtype)

    def dequantize_fused(self, norms: mx.array, packed: mx.array) -> mx.array:
        """Fused dequant: one Metal dispatch for unpack+codebook+WHT+norm.

        Only works with WHT rotation. Falls back to standard dequantize for others.
        Output is fp16, ready for AMX matmul.
        """
        if not self.use_wht:
            return self.dequantize(norms, packed)
        return _fused_dequant_wht(
            packed, norms, self.codebook, self.rotation.T,
            self.bits, self.dim,
        )


# ---------------------------------------------------------------------------
# 2-pass Fused Flash Attention kernels
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _tq_sdpa_2pass_1_kernel():
    source = r"""
        auto simd_lid = thread_index_in_simdgroup;
        auto kv_head = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto gqa_factor = threads_per_threadgroup.y;
        auto q_head = gqa_factor * kv_head + thread_position_in_threadgroup.y;
        auto num_kv_heads = threadgroups_per_grid.x;
        auto q_batch_head = batch_idx * num_kv_heads * gqa_factor + q_head;
        auto total_tokens = k_norms_shape[2];

        auto q_ptr = queries + q_batch_head * Dim;
        float q[QK_PER_THREAD];
        for (int i = 0; i < QK_PER_THREAD; i++)
            q[i] = static_cast<float>(q_ptr[simd_lid * QK_PER_THREAD + i]) * scale[0];

        float o[QK_PER_THREAD] = {0};
        float max_score = -INFINITY;
        float sum_exp = 0.0f;

        auto kv_bh = batch_idx * num_kv_heads + kv_head;
        auto k_base = k_packed + kv_bh * total_tokens * KPackedWidth;
        auto v_base = v_packed + kv_bh * total_tokens * VPackedWidth;
        auto kn_base = k_norms + kv_bh * total_tokens;
        auto vn_base = v_norms + kv_bh * total_tokens;

        for (int t = block_idx; t < total_tokens; t += Blocks) {
            auto k_ptr = k_base + t * KPackedWidth;
            float score = 0.0f;
            for (int j = 0; j < QK_PER_THREAD; j++) {
                int d = simd_lid * QK_PER_THREAD + j;
                int bit_off = d * KBits;
                int word = bit_off / 32;
                int off = bit_off % 32;
                uint val = k_ptr[word] >> off;
                int spill = off + KBits - 32;
                if (spill > 0) val |= k_ptr[word + 1] << (KBits - spill);
                val &= ((1u << KBits) - 1u);
                score += q[j] * k_codebook[val];
            }
            score = simd_sum(score) * static_cast<float>(kn_base[t]);

            float new_max = max(max_score, score);
            float factor = exp(max_score - new_max);
            float exp_score = exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;

            auto v_ptr = v_base + t * VPackedWidth;
            float v_norm = static_cast<float>(vn_base[t]);
            for (int j = 0; j < QK_PER_THREAD; j++) {
                int d = simd_lid * QK_PER_THREAD + j;
                int bit_off = d * VBits;
                int word = bit_off / 32;
                int off = bit_off % 32;
                uint val = v_ptr[word] >> off;
                int spill = off + VBits - 32;
                if (spill > 0) val |= v_ptr[word + 1] << (VBits - spill);
                val &= ((1u << VBits) - 1u);
                o[j] = o[j] * factor + exp_score * v_codebook[val] * v_norm;
            }
        }

        auto out_idx = q_batch_head * Blocks * Dim + block_idx * Dim;
        if (simd_lid == 0) {
            sums[q_batch_head * Blocks + block_idx] = sum_exp;
            maxs[q_batch_head * Blocks + block_idx] = max_score;
        }
        for (int j = 0; j < QK_PER_THREAD; j++)
            partial_out[out_idx + simd_lid * QK_PER_THREAD + j] = static_cast<half>(o[j]);
    """
    return mx.fast.metal_kernel(
        name="tq_fused_sdpa_pass1",
        input_names=["queries", "k_packed", "k_norms", "k_codebook",
                     "v_packed", "v_norms", "v_codebook", "scale"],
        output_names=["partial_out", "sums", "maxs"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _tq_sdpa_2pass_2_kernel():
    source = r"""
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;
        auto head_idx = threadgroup_position_in_grid.x;

        auto s_base = sums + head_idx * Blocks;
        auto m_base = maxs + head_idx * Blocks;
        auto o_ptr = out + head_idx * Dim + simd_gid * QK_PER_THREAD;

        threadgroup float tg_outputs[BN * BD];

        float max_score = -INFINITY;
        for (int b = 0; b < Blocks / BN; b++)
            max_score = max(max_score, m_base[simd_lid + BN * b]);
        if (Blocks % BN != 0) {
            int b_idx = (Blocks / BN) * BN + simd_lid;
            if (b_idx < Blocks) max_score = max(max_score, m_base[b_idx]);
        }
        max_score = simd_max(max_score);

        float sum_exp = 0.0f;
        for (int b = 0; b < Blocks / BN; b++) {
            float factor = exp(m_base[simd_lid + BN * b] - max_score);
            sum_exp += factor * s_base[simd_lid + BN * b];
        }
        if (Blocks % BN != 0) {
            int b_idx = (Blocks / BN) * BN + simd_lid;
            if (b_idx < Blocks) {
                sum_exp += exp(m_base[b_idx] - max_score) * s_base[b_idx];
            }
        }
        sum_exp = simd_sum(sum_exp);

        float o[QK_PER_THREAD] = {0};
        for (int b = 0; b < Blocks / BN; b++) {
            float factor = exp(m_base[simd_gid + BN * b] - max_score);
            auto p_ptr = partials + head_idx * Blocks * Dim
                         + (simd_gid + BN * b) * Dim + simd_lid * QK_PER_THREAD;
            for (int i = 0; i < QK_PER_THREAD; i++)
                o[i] += factor * static_cast<float>(p_ptr[i]);
        }

        for (int i = 0; i < QK_PER_THREAD; i++) {
            tg_outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp > 0 ? o[i] / sum_exp : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < QK_PER_THREAD; i++)
                o_ptr[i] = static_cast<half>(o[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="tq_fused_sdpa_pass2",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _tq_sdpa_prefill_kernel():
    """Fused TQ SDPA for L>1 queries (prefill streaming attention).

    Key change from decode kernel: grid.y now encodes (batch × L × GQA).
    All L queries for each head/batch are processed in a single dispatch.
    """
    source = r"""
        auto simd_lid = thread_index_in_simdgroup;
        auto kv_head = threadgroup_position_in_grid.x;
        auto combined_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto gqa_factor = threads_per_threadgroup.y;
        auto q_head = gqa_factor * kv_head + thread_position_in_threadgroup.y;
        auto num_kv_heads = threadgroups_per_grid.x;

        // Decode combined_idx = (l * B + batch_idx)
        // Templates: NumQueries = L, BatchSize = B
        auto l_idx = combined_idx / BatchSize;
        auto batch_idx = combined_idx % BatchSize;

        // Flat query-batch-head index for output
        auto q_bhl = batch_idx * NumQueries * num_kv_heads * gqa_factor
                   + l_idx * num_kv_heads * gqa_factor
                   + q_head;
        auto total_tokens = k_norms_shape[2];

        // Load query from (B, L, H_q, D) — interleaved layout
        // queries shape: B * L * H_q * D flat, indexed by q_bhl
        auto q_ptr = queries + q_bhl * Dim;
        float q[QK_PER_THREAD];
        for (int i = 0; i < QK_PER_THREAD; i++)
            q[i] = static_cast<float>(q_ptr[simd_lid * QK_PER_THREAD + i]) * scale[0];

        float o[QK_PER_THREAD] = {0};
        float max_score = -INFINITY;
        float sum_exp = 0.0f;

        // K/V use only (batch_idx, kv_head) — shared across all L queries
        auto kv_bh = batch_idx * num_kv_heads + kv_head;
        auto k_base = k_packed + kv_bh * total_tokens * KPackedWidth;
        auto v_base = v_packed + kv_bh * total_tokens * VPackedWidth;
        auto kn_base = k_norms + kv_bh * total_tokens;
        auto vn_base = v_norms + kv_bh * total_tokens;

        for (int t = block_idx; t < total_tokens; t += Blocks) {
            auto k_ptr = k_base + t * KPackedWidth;
            float score = 0.0f;
            for (int j = 0; j < QK_PER_THREAD; j++) {
                int d = simd_lid * QK_PER_THREAD + j;
                int bit_off = d * KBits;
                int word = bit_off / 32;
                int off = bit_off % 32;
                uint val = k_ptr[word] >> off;
                int spill = off + KBits - 32;
                if (spill > 0) val |= k_ptr[word + 1] << (KBits - spill);
                val &= ((1u << KBits) - 1u);
                score += q[j] * k_codebook[val];
            }
            score = simd_sum(score) * static_cast<float>(kn_base[t]);

            float new_max = max(max_score, score);
            float factor = exp(max_score - new_max);
            float exp_score = exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;

            auto v_ptr = v_base + t * VPackedWidth;
            float v_norm = static_cast<float>(vn_base[t]);
            for (int j = 0; j < QK_PER_THREAD; j++) {
                int d = simd_lid * QK_PER_THREAD + j;
                int bit_off = d * VBits;
                int word = bit_off / 32;
                int off = bit_off % 32;
                uint val = v_ptr[word] >> off;
                int spill = off + VBits - 32;
                if (spill > 0) val |= v_ptr[word + 1] << (VBits - spill);
                val &= ((1u << VBits) - 1u);
                o[j] = o[j] * factor + exp_score * v_codebook[val] * v_norm;
            }
        }

        // Output layout matches partials shape: (total_heads * Blocks, Dim)
        // where total_heads = B * L * H_q
        auto out_idx = q_bhl * Blocks * Dim + block_idx * Dim;
        if (simd_lid == 0) {
            sums[q_bhl * Blocks + block_idx] = sum_exp;
            maxs[q_bhl * Blocks + block_idx] = max_score;
        }
        for (int j = 0; j < QK_PER_THREAD; j++)
            partial_out[out_idx + simd_lid * QK_PER_THREAD + j] = static_cast<half>(o[j]);
    """
    return mx.fast.metal_kernel(
        name="tq_fused_sdpa_prefill_pass1",
        input_names=["queries", "k_packed", "k_norms", "k_codebook",
                     "v_packed", "v_norms", "v_codebook", "scale"],
        output_names=["partial_out", "sums", "maxs"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _tq_fused_prefill_kernel():
    """Fused TQ3 FlashAttention for prefill: dequant + scores + softmax + V accumulate.

    Single-pass over all history tokens for a tile of TILE_Q queries.
    No intermediate partials buffer — online softmax in registers.

    Grid: (H_kv, B, num_q_tiles)
    Threadgroup: (32, GQA, 1) — 32 threads per SIMD, GQA query heads per KV head

    For each threadgroup:
      - Iterates over ALL history tokens (no block splitting)
      - Each SIMD thread handles QK_PER_THREAD = D/32 dimensions
      - Codebook (8 entries) loaded into thread-local registers
      - Online softmax: running max + exp rescale per query per head

    Memory layout per threadgroup (SRAM):
      - Q tile: TILE_Q * Dim * 4 bytes = 16 * 128 * 4 = 8KB
      - Online softmax state: TILE_Q * (max + sum + D output) in registers
      - Codebook: 8 floats in registers (32 bytes)
      Total SRAM: ~8KB (well under 32KB limit)
    """
    source = r"""
        auto simd_lid = thread_index_in_simdgroup;       // 0..31 within SIMD
        auto kv_head = threadgroup_position_in_grid.x;    // which KV head
        auto batch_idx = threadgroup_position_in_grid.y;  // batch index
        auto q_tile_idx = threadgroup_position_in_grid.z; // which Q tile
        auto gqa_factor = threads_per_threadgroup.y;      // GQA ratio
        auto q_head_in_group = thread_position_in_threadgroup.y;  // 0..GQA-1
        auto q_head = gqa_factor * kv_head + q_head_in_group;
        auto num_kv_heads = threadgroups_per_grid.x;
        auto total_tokens = k_norms_shape[2];

        // Flat output index: batch * (num_q_tiles * TILE_Q) * H_q * Dim
        // But we use a simpler layout: output[(batch, q_head, q_tile_idx * TILE_Q + qi), d]

        // KV base pointers (shared across all queries in this tile)
        auto kv_bh = batch_idx * num_kv_heads + kv_head;
        auto k_base = k_packed + kv_bh * total_tokens * KPackedWidth;
        auto v_base = v_packed + kv_bh * total_tokens * VPackedWidth;
        auto kn_base = k_norms + kv_bh * total_tokens;
        auto vn_base = v_norms + kv_bh * total_tokens;

        // Process TILE_Q queries sequentially (pinned KV, looped Q)
        for (int qi = 0; qi < TileQ; qi++) {
            int global_qi = q_tile_idx * TileQ + qi;
            if (global_qi >= NumQueries) break;

            // Load query from input
            auto q_bhl = (batch_idx * NumQueries * num_kv_heads * gqa_factor)
                       + (global_qi * num_kv_heads * gqa_factor)
                       + q_head;
            auto q_ptr = queries + q_bhl * Dim;

            float q[QK_PER_THREAD];
            for (int i = 0; i < QK_PER_THREAD; i++)
                q[i] = static_cast<float>(q_ptr[simd_lid * QK_PER_THREAD + i]) * scale[0];

            // Online softmax state for this query
            float o[QK_PER_THREAD] = {0};
            float max_score = -INFINITY;
            float sum_exp = 0.0f;

            // Loop over ALL history tokens (single pass, no block splitting)
            for (int t = 0; t < total_tokens; t++) {
                // Unpack K and compute dot product
                auto k_ptr = k_base + t * KPackedWidth;
                float score = 0.0f;
                for (int j = 0; j < QK_PER_THREAD; j++) {
                    int d = simd_lid * QK_PER_THREAD + j;
                    int bit_off = d * KBits;
                    int word = bit_off / 32;
                    int off = bit_off % 32;
                    uint val = k_ptr[word] >> off;
                    int spill = off + KBits - 32;
                    if (spill > 0) val |= k_ptr[word + 1] << (KBits - spill);
                    val &= ((1u << KBits) - 1u);
                    score += q[j] * k_codebook[val];
                }
                score = simd_sum(score) * static_cast<float>(kn_base[t]);

                // Online softmax update
                float new_max = max(max_score, score);
                float factor = exp(max_score - new_max);
                float exp_score = exp(score - new_max);
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;

                // Unpack V and accumulate
                auto v_ptr = v_base + t * VPackedWidth;
                float v_norm = static_cast<float>(vn_base[t]);
                for (int j = 0; j < QK_PER_THREAD; j++) {
                    int d = simd_lid * QK_PER_THREAD + j;
                    int bit_off = d * VBits;
                    int word = bit_off / 32;
                    int off = bit_off % 32;
                    uint val = v_ptr[word] >> off;
                    int spill = off + VBits - 32;
                    if (spill > 0) val |= v_ptr[word + 1] << (VBits - spill);
                    val &= ((1u << VBits) - 1u);
                    o[j] = o[j] * factor + exp_score * v_codebook[val] * v_norm;
                }
            }

            // Normalize output
            float inv_sum = sum_exp > 0 ? 1.0f / sum_exp : 0.0f;
            for (int j = 0; j < QK_PER_THREAD; j++)
                o[j] *= inv_sum;

            // Write output
            auto out_idx = q_bhl * Dim;
            for (int j = 0; j < QK_PER_THREAD; j++)
                out[out_idx + simd_lid * QK_PER_THREAD + j] = static_cast<half>(o[j]);
        }
    """
    return mx.fast.metal_kernel(
        name="tq_fused_prefill",
        input_names=["queries", "k_packed", "k_norms", "k_codebook",
                     "v_packed", "v_norms", "v_codebook", "scale"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _fused_tq_prefill(
    queries: mx.array,       # (B*L*H_q, D) — pre-rotated, flattened
    k_packed: mx.array,      # (B, H_kv, T, packed_width)
    k_norms: mx.array,       # (B, H_kv, T)
    k_codebook: mx.array,    # (n_levels,)
    v_packed: mx.array,      # (B, H_kv, T, packed_width)
    v_norms: mx.array,       # (B, H_kv, T)
    v_codebook: mx.array,    # (n_levels,)
    scale: float,
    B: int, L: int, H_q: int, H_kv: int, D: int, bits: int,
    tile_q: int = 16,
) -> mx.array:
    """Fused TQ3 prefill: single-pass FlashAttention for L queries × T history.

    One kernel dispatch per call. Handles all L queries by tiling into
    groups of tile_q, each processed sequentially within a threadgroup.
    """
    GQA = H_q // H_kv
    pw = _packed_width(D, bits)
    qpt = D // 32
    T = k_norms.shape[2]
    total_heads = B * L * H_q

    num_q_tiles = (L + tile_q - 1) // tile_q
    scale_arr = mx.array([scale], dtype=mx.float32)

    out = _tq_fused_prefill_kernel()(
        inputs=[queries, k_packed, k_norms, k_codebook,
                v_packed, v_norms, v_codebook, scale_arr],
        output_shapes=[(total_heads, D)],
        output_dtypes=[mx.float16],
        grid=(H_kv * 32, B * GQA, num_q_tiles),
        threadgroup=(32, GQA, 1),
        template=[
            ("Dim", D), ("QK_PER_THREAD", qpt),
            ("KBits", bits), ("VBits", bits),
            ("KPackedWidth", pw), ("VPackedWidth", pw),
            ("TileQ", tile_q), ("NumQueries", L),
        ],
        init_value=0.0,
    )

    # Reshape: (B*L*H_q, D) → (B, H_q, L, D)
    return out[0].reshape(B, L, H_q, D).transpose(0, 2, 1, 3)


def _fused_tq_sdpa_prefill(
    queries: mx.array,       # (B, L, H_q, D) — interleaved layout
    k_packed: mx.array,      # (B, H_kv, T, packed_width)
    k_norms: mx.array,       # (B, H_kv, T)
    k_codebook: mx.array,    # (n_levels,)
    v_packed: mx.array,      # (B, H_kv, T, packed_width)
    v_norms: mx.array,       # (B, H_kv, T)
    v_codebook: mx.array,    # (n_levels,)
    scale: float,
    B: int, L: int, H_q: int, H_kv: int, D: int, bits: int,
) -> mx.array:
    """Fused TQ SDPA for L queries × T history. Single kernel dispatch per layer."""
    GQA = H_q // H_kv
    pw = _packed_width(D, bits)
    qpt = D // 32
    BN = BD = 32
    T = k_norms.shape[2]
    total_heads = B * L * H_q  # Expanded heads count

    # Adaptive blocks
    num_blocks = min(1024, ((max(32, T // 32) + 31) // 32) * 32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    # Pass 1: parallel block attention across all L queries
    partials, sums, maxs = _tq_sdpa_prefill_kernel()(
        inputs=[queries, k_packed, k_norms, k_codebook,
                v_packed, v_norms, v_codebook, scale_arr],
        output_shapes=[
            (total_heads * num_blocks, D),
            (total_heads, num_blocks),
            (total_heads, num_blocks),
        ],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
        grid=(H_kv * 32, B * L * GQA, num_blocks),
        threadgroup=(32, GQA, 1),
        template=[
            ("Dim", D), ("Blocks", num_blocks), ("QK_PER_THREAD", qpt),
            ("KBits", bits), ("VBits", bits),
            ("KPackedWidth", pw), ("VPackedWidth", pw),
            ("NumQueries", L), ("BatchSize", B),
        ],
        init_value=0.0,
    )

    maxs = mx.where(sums == 0, mx.full(maxs.shape, float("-inf"), dtype=maxs.dtype), maxs)

    # Pass 2: reduce blocks — reuse existing pass2 kernel
    out = _tq_sdpa_2pass_2_kernel()(
        inputs=[partials, sums, maxs],
        output_shapes=[(total_heads, D)],
        output_dtypes=[mx.float16],
        grid=(total_heads * 32, BN, 1),
        threadgroup=(32, BN, 1),
        template=[
            ("Dim", D), ("Blocks", num_blocks), ("QK_PER_THREAD", qpt),
            ("BN", BN), ("BD", BD),
        ],
        init_value=0.0,
    )
    # Reshape (total_heads, D) → (B, L, H_q, D) → (B, H_q, L, D)
    out = out[0].reshape(B, L, H_q, D).transpose(0, 2, 1, 3)
    return out


def _fused_tq_sdpa(
    queries: mx.array,       # (B*H_q, D)
    k_packed: mx.array,      # (B, H_kv, T, packed_width)
    k_norms: mx.array,       # (B, H_kv, T)
    k_codebook: mx.array,    # (n_levels,)
    v_packed: mx.array,      # (B, H_kv, T, packed_width)
    v_norms: mx.array,       # (B, H_kv, T)
    v_codebook: mx.array,    # (n_levels,)
    scale: float,
    B: int, H_q: int, H_kv: int, D: int, bits: int,
) -> mx.array:
    """Fused 2-pass TurboQuant Flash Attention (supports B>=1)."""
    GQA = H_q // H_kv
    pw = _packed_width(D, bits)
    qpt = D // 32
    BN = BD = 32
    T = k_norms.shape[2]
    total_heads = B * H_q

    # Adaptive blocks: cap at 1024, each handles ~32 tokens
    num_blocks = min(1024, ((max(32, T // 32) + 31) // 32) * 32)

    scale_arr = mx.array([scale], dtype=mx.float32)

    # Pass 1: parallel block-wise attention
    # grid.y = B * GQA → threadgroup_position_in_grid.y = batch_idx (0..B-1)
    partials, sums, maxs = _tq_sdpa_2pass_1_kernel()(
        inputs=[queries, k_packed, k_norms, k_codebook,
                v_packed, v_norms, v_codebook, scale_arr],
        output_shapes=[
            (total_heads * num_blocks, D),
            (total_heads, num_blocks),
            (total_heads, num_blocks),
        ],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
        grid=(H_kv * 32, B * GQA, num_blocks),
        threadgroup=(32, GQA, 1),
        template=[
            ("Dim", D), ("Blocks", num_blocks), ("QK_PER_THREAD", qpt),
            ("KBits", bits), ("VBits", bits),
            ("KPackedWidth", pw), ("VPackedWidth", pw),
        ],
        init_value=0.0,
    )

    # Fix: empty blocks have maxs=0 from init_value, should be -inf
    maxs = mx.where(sums == 0, mx.full(maxs.shape, float("-inf"), dtype=maxs.dtype), maxs)

    # Pass 2: SIMD-parallel reduction
    out = _tq_sdpa_2pass_2_kernel()(
        inputs=[partials, sums, maxs],
        output_shapes=[(total_heads, D)],
        output_dtypes=[mx.float16],
        grid=(total_heads * 32, BN, 1),
        threadgroup=(32, BN, 1),
        template=[
            ("Dim", D), ("Blocks", num_blocks), ("QK_PER_THREAD", qpt),
            ("BN", BN), ("BD", BD),
        ],
        init_value=0.0,
    )
    return out[0]


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------

class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant codebook quantization.

    Stores keys and values as packed codebook indices + norms.
    Decode attention uses fused 2-pass Flash Attention kernel.
    Prefill uses dequantize + standard mx.fast.scaled_dot_product_attention.
    """

    def __init__(self, bits: int = 4, seed: int = 0, dequant_chunk_size: int = 2048,
                 min_quant_tokens: int = 512):
        self.bits = bits
        self.seed = seed
        self._dequant_chunk_size = dequant_chunk_size  # Tokens per dequant chunk
        self._min_quant_tokens = min_quant_tokens  # Stay fp16 below this threshold
        # Safety: mlx-lm's base.py SDPA checks hasattr(cache, "bits") and then
        # accesses cache.group_size for affine quantized caches.  Prevents
        # AttributeError if our attention patch doesn't intercept.
        self.group_size = 0
        self.offset = 0
        self._k_norms = None
        self._k_packed = None
        self._v_norms = None
        self._v_packed = None
        self._fp16_keys = None
        self._fp16_values = None
        self._quantized = False
        self._codec: Optional[TurboQuantMSECodec] = None
        self._step = 256
        self._streaming_active = False
        self._new_chunk_start = 0

    def _ensure_codec(self, dim: int):
        if self._codec is None:
            self._codec = TurboQuantMSECodec(dim, self.bits, self.seed)

    def _quantize_fp16_buffer(self):
        """Convert accumulated fp16 KV to quantized format."""
        if self._fp16_keys is None or self._quantized:
            return
        B, H, T, D = self._fp16_keys.shape
        logger.info(f"TurboQuant: quantizing {T} tokens ({B}×{H} heads, dim={D}) to {self.bits}-bit")
        self._ensure_codec(D)
        k_norms, k_packed = self._codec.quantize(self._fp16_keys)
        v_norms, v_packed = self._codec.quantize(self._fp16_values)
        pw = _packed_width(D, self.bits)
        alloc = ((T + self._step - 1) // self._step) * self._step
        self._k_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
        self._k_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
        self._v_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
        self._v_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
        self._k_norms[:, :, :T] = k_norms
        self._k_packed[:, :, :T] = k_packed
        self._v_norms[:, :, :T] = v_norms
        self._v_packed[:, :, :T] = v_packed
        self._quantized = True
        self._fp16_keys = None
        self._fp16_values = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new K,V with streaming quantization during prefill.

        Each prefill chunk is quantized immediately. For attention, the
        compressed history is dequantized into a temporary buffer.
        Memory: model + (compressed_history × 3-bit) + (2 × current_chunk × fp16).
        """
        B, H, T_new, D = keys.shape
        self._ensure_codec(D)

        if T_new > 1:
            logger.debug(
                "update_and_fetch: offset=%d T_new=%d quantized=%s",
                self.offset, T_new, self._quantized,
            )

            # Below threshold: accumulate fp16, no quantization yet.
            # TQ3 codebook needs enough tokens for meaningful compression.
            # 512 tokens in fp16 is ~0.001GB per layer — negligible.
            if self.offset + T_new < self._min_quant_tokens and not self._quantized:
                if self._fp16_keys is None:
                    self._fp16_keys = keys
                    self._fp16_values = values
                else:
                    self._fp16_keys = mx.concatenate([self._fp16_keys, keys], axis=2)
                    self._fp16_values = mx.concatenate([self._fp16_values, values], axis=2)
                self.offset += T_new
                self._streaming_active = False
                return self._fp16_keys, self._fp16_values

            # Cross the threshold — quantize accumulated fp16 buffer first
            if not self._quantized and self._fp16_keys is not None:
                self._quantize_fp16_buffer()

            # Streaming prefill: quantize this chunk and store compressed
            pw = _packed_width(D, self.bits)

            # Quantize the new chunk
            k_norms, k_packed = self._codec.quantize(keys)
            v_norms, v_packed = self._codec.quantize(values)

            new_end = self.offset + T_new

            # Initialize or extend compressed storage
            if self._k_norms is None:
                alloc = ((new_end + self._step - 1) // self._step) * self._step
                self._k_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
                self._k_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
                self._v_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
                self._v_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
            elif new_end > self._k_norms.shape[2]:
                alloc = ((new_end + self._step - 1) // self._step) * self._step
                pad = alloc - self._k_norms.shape[2]
                self._k_norms = mx.concatenate([self._k_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._k_packed = mx.concatenate([self._k_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)
                self._v_norms = mx.concatenate([self._v_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._v_packed = mx.concatenate([self._v_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)

            # Store compressed
            self._k_norms[:, :, self.offset:new_end] = k_norms
            self._k_packed[:, :, self.offset:new_end] = k_packed
            self._v_norms[:, :, self.offset:new_end] = v_norms
            self._v_packed[:, :, self.offset:new_end] = v_packed
            self.offset = new_end
            self._quantized = True

            # Track where the NEW chunk starts in compressed storage
            # (streaming attention needs to know what's "history" vs "current")
            self._new_chunk_start = self.offset - T_new

            # For short history: dequantize everything (fast path)
            history_tokens = self._new_chunk_start
            logger.debug(
                "  history=%d dequant_chunk=%d → %s",
                history_tokens, self._dequant_chunk_size,
                "short_dequant" if history_tokens <= self._dequant_chunk_size else "STREAMING",
            )
            if history_tokens <= self._dequant_chunk_size:
                all_k = self._codec.dequantize(
                    self._k_norms[:, :, :self.offset],
                    self._k_packed[:, :, :self.offset],
                )
                all_v = self._codec.dequantize(
                    self._v_norms[:, :, :self.offset],
                    self._v_packed[:, :, :self.offset],
                )
                self._streaming_active = False
                return all_k, all_v
            else:
                # STREAMING MODE: return ONLY the new chunk's fp16 K,V
                # The attention patch will handle history via streaming_tq_attention
                # NO ghost concatenation — no full dequant — constant memory
                self._streaming_active = True
                return keys, values
        else:
            # Decode (T_new == 1)

            # Still in fp16 warmup phase — append and return fp16
            if not self._quantized and self._fp16_keys is not None:
                self._fp16_keys = mx.concatenate([self._fp16_keys, keys], axis=2)
                self._fp16_values = mx.concatenate([self._fp16_values, values], axis=2)
                self.offset += 1
                return self._fp16_keys, self._fp16_values

            # Quantize the new token and append to compressed storage
            k_norms, k_packed = self._codec.quantize(keys)
            v_norms, v_packed = self._codec.quantize(values)

            new_end = self.offset + 1
            pw = _packed_width(D, self.bits)
            if self._k_norms is None:
                alloc = self._step
                self._k_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
                self._k_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
                self._v_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
                self._v_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
            elif new_end > self._k_norms.shape[2]:
                alloc = ((new_end + self._step - 1) // self._step) * self._step
                pad = alloc - self._k_norms.shape[2]
                self._k_norms = mx.concatenate([self._k_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._k_packed = mx.concatenate([self._k_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)
                self._v_norms = mx.concatenate([self._v_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._v_packed = mx.concatenate([self._v_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)

            self._k_norms[:, :, self.offset:new_end] = k_norms
            self._k_packed[:, :, self.offset:new_end] = k_packed
            self._v_norms[:, :, self.offset:new_end] = v_norms
            self._v_packed[:, :, self.offset:new_end] = v_packed
            self.offset = new_end
            return self._quantized_state

    @property
    def _quantized_state(self):
        return (
            (self._k_norms[:, :, :self.offset], self._k_packed[:, :, :self.offset]),
            (self._v_norms[:, :, :self.offset], self._v_packed[:, :, :self.offset]),
        )

    @property
    def state(self):
        if self._fp16_keys is not None and not self._quantized:
            return self._fp16_keys[:, :, :self.offset], self._fp16_values[:, :, :self.offset]
        if self._k_norms is None:
            return None, None
        return self._quantized_state

    @state.setter
    def state(self, v):
        if v[0] is None:
            self._k_norms = self._k_packed = self._v_norms = self._v_packed = None
            self.offset = 0
        else:
            (self._k_norms, self._k_packed), (self._v_norms, self._v_packed) = v
            self.offset = self._k_norms.shape[2]

    @property
    def meta_state(self):
        return (self.offset, self.bits, self.seed)

    @meta_state.setter
    def meta_state(self, v):
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            self.offset = int(v[0])
            self.bits = int(v[1])
            self.seed = int(v[2])
        else:
            self.offset = int(v) if not isinstance(v, (list, tuple)) else int(v[0])

    def dequantize(self, keys_state=None, values_state=None):
        """Full dequantize for prefill fallback."""
        if keys_state is None:
            keys_state, values_state = self.state
        k_norms, k_packed = keys_state
        v_norms, v_packed = values_state
        # Lazy codec init from packed tensor shape
        if self._codec is None:
            pw = k_packed.shape[-1]
            dim = pw * 32 // self.bits
            self._ensure_codec(dim)
        keys = self._codec.dequantize(k_norms, k_packed)
        values = self._codec.dequantize(v_norms, v_packed)
        return keys, values

    def decode_attention(
        self,
        queries: mx.array,      # (B, H_q, 1, D)
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        """Fused 2-pass Flash Attention from quantized KV. No dequantize."""
        if keys_state is None:
            keys_state, values_state = self.state
        k_norms, k_packed = keys_state
        v_norms, v_packed = values_state

        B, H_q, L, D = queries.shape
        H_kv = k_norms.shape[1]

        # Prepare queries: scale and flatten
        q_flat = (queries.squeeze(2) * scale).reshape(B * H_q, D).astype(mx.float16)

        # Rotate queries (same rotation as codec)
        if self._codec.use_givens:
            q_rot = _apply_givens(
                q_flat.astype(mx.float32),
                self._codec._givens_cos,
                self._codec._givens_sin,
            ).astype(mx.float16)
        else:
            R = self._codec.rotation
            q_grouped = q_flat.reshape(B * H_q, D // self._codec.dim, self._codec.dim)
            q_rot = (q_grouped.astype(mx.float32) @ R).reshape(B * H_q, D).astype(mx.float16)

        # Fused 2-pass SDPA
        out = _fused_tq_sdpa(
            q_rot, k_packed, k_norms, self._codec.codebook,
            v_packed, v_norms, self._codec.codebook,
            scale=1.0,  # already applied to queries
            B=B, H_q=H_q, H_kv=H_kv, D=D, bits=self.bits,
        )

        # Inverse rotate output (values were in rotated space)
        if self._codec.use_givens:
            out_restored = _apply_givens_inverse(
                out.reshape(B * H_q, D).astype(mx.float32),
                self._codec._givens_cos,
                self._codec._givens_sin,
            ).astype(queries.dtype)
        else:
            out_grouped = out.reshape(B * H_q, D // self._codec.dim, self._codec.dim).astype(mx.float32)
            out_restored = (out_grouped @ R.T).reshape(B * H_q, D).astype(queries.dtype)

        return out_restored.reshape(B, H_q, 1, D)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self._k_norms is None or self.offset == 0

    @property
    def nbytes(self) -> int:
        if self._k_norms is None:
            return 0
        T = self.offset
        return (
            self._k_norms[:, :, :T].nbytes + self._k_packed[:, :, :T].nbytes +
            self._v_norms[:, :, :T].nbytes + self._v_packed[:, :, :T].nbytes
        )

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def rewind_to(self, target_offset: int) -> int:
        """O(1) context rewind: drop tokens after target_offset.

        Compressed storage is unchanged — we just move the offset pointer.
        Future writes will overwrite the discarded range. No re-prefill needed.

        Args:
            target_offset: keep only the first N tokens

        Returns:
            Number of tokens actually kept (may be less if cache is shorter)
        """
        target_offset = max(0, min(target_offset, self.offset))
        dropped = self.offset - target_offset
        self.offset = target_offset
        # Reset streaming flag — next write needs to recompute
        self._streaming_active = False
        self._new_chunk_start = target_offset
        logger.info("TQ KV: rewound from %d to %d (dropped %d tokens)",
                    self.offset + dropped, target_offset, dropped)
        return target_offset

    def save_to_disk(self, filepath: str) -> dict:
        """Freeze compressed KV cache to disk as .npz file.

        At 256K context with TQ3, the cache is ~5GB — writes in ~1.5s on NVMe.
        Load with load_from_disk() to resume without re-prefill.

        Args:
            filepath: path to .npz file (extension added if missing)

        Returns:
            dict with metadata about what was saved
        """
        import numpy as np

        if self._k_norms is None or self.offset == 0:
            raise ValueError("Cannot save empty cache")

        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        T = self.offset
        # Slice to actual used range (don't save padding)
        data = {
            "k_norms": np.array(self._k_norms[:, :, :T]),
            "k_packed": np.array(self._k_packed[:, :, :T]),
            "v_norms": np.array(self._v_norms[:, :, :T]),
            "v_packed": np.array(self._v_packed[:, :, :T]),
            "offset": T,
            "bits": self.bits,
            "seed": self.seed,
            "min_quant_tokens": self._min_quant_tokens,
            "dequant_chunk_size": self._dequant_chunk_size,
        }

        # Save fp16 warmup buffer if present (for caches below threshold)
        if self._fp16_keys is not None:
            data["fp16_keys"] = np.array(self._fp16_keys)
            data["fp16_values"] = np.array(self._fp16_values)
            data["quantized"] = self._quantized
        else:
            data["quantized"] = True

        np.savez_compressed(filepath, **data)

        size_gb = sum(v.nbytes if hasattr(v, 'nbytes') else 0 for v in data.values()) / 1e9
        logger.info("TQ KV: saved %d tokens to %s (%.2fGB)", T, filepath, size_gb)
        return {"tokens": T, "path": filepath, "size_gb": size_gb, "bits": self.bits}

    def load_from_disk(self, filepath: str) -> int:
        """Thaw a frozen KV cache from disk. Model prefill not needed.

        Args:
            filepath: path to .npz file

        Returns:
            Number of tokens loaded
        """
        import numpy as np

        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        data = np.load(filepath)
        self.bits = int(data["bits"])
        self.seed = int(data["seed"])
        self._min_quant_tokens = int(data["min_quant_tokens"])
        self._dequant_chunk_size = int(data["dequant_chunk_size"])

        self._k_norms = mx.array(data["k_norms"])
        self._k_packed = mx.array(data["k_packed"])
        self._v_norms = mx.array(data["v_norms"])
        self._v_packed = mx.array(data["v_packed"])
        self.offset = int(data["offset"])
        self._quantized = bool(data["quantized"])
        self._streaming_active = False
        self._new_chunk_start = self.offset

        # Rebuild codec for dequant/decode
        D = self._k_norms.shape[-1] * 32 // self.bits  # Reverse _packed_width
        # Actually D should be inferred from packed width
        pw = self._k_packed.shape[-1]
        D = pw * 32 // self.bits
        self._ensure_codec(D)

        # Restore fp16 warmup if present
        if "fp16_keys" in data.files:
            self._fp16_keys = mx.array(data["fp16_keys"])
            self._fp16_values = mx.array(data["fp16_values"])

        logger.info("TQ KV: loaded %d tokens from %s", self.offset, filepath)
        return self.offset

    @classmethod
    def from_cache(cls, cache, bits: int = 4, seed: int = 0) -> "TurboQuantKVCache":
        """Convert an existing KVCache to TurboQuantKVCache."""
        tq = cls(bits=bits, seed=seed)
        keys, values = cache.state
        if keys is not None:
            tq.update_and_fetch(keys, values)
        return tq


class BatchTurboQuantKVCache(_BaseCache):
    """Batched TurboQuant KV cache for continuous batching.

    Prefill phase: stores fp16 (like BatchKVCache) for full-quality attention.
    Decode phase: quantizes to TurboQuant for memory-efficient generation.
    Implements all BatchKVCache methods for BatchGenerator compatibility.
    """
    step = 256

    def __init__(self, left_padding, bits: int = 4, seed: int = 0):
        self.bits = bits
        self.seed = seed
        # Safety: mlx-lm's base.py SDPA checks hasattr(cache, "bits") and then
        # accesses cache.group_size for affine quantized caches.  If our attention
        # patch doesn't intercept (e.g. VLM proxy), this prevents AttributeError.
        self.group_size = 0
        # fp16 storage (prefill phase)
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        # Quantized storage (decode phase)
        self._k_norms = None
        self._k_packed = None
        self._v_norms = None
        self._v_packed = None
        self._quantized = False
        self._codec = None

    def _ensure_codec(self, dim):
        if self._codec is None:
            self._codec = TurboQuantMSECodec(dim, self.bits, self.seed)

    def _quantize_buffer(self):
        """Convert fp16 KV to quantized. Called at decode start."""
        if self._quantized or self.keys is None:
            return
        B, H, T, D = self.keys.shape
        logger.info(f"TurboQuant batch: quantizing {self._idx} tokens ({B}×{H} heads, dim={D}) to {self.bits}-bit")
        self._ensure_codec(D)
        # Quantize full buffer
        k = self.keys[..., :self._idx, :]
        v = self.values[..., :self._idx, :]
        k_norms, k_packed = self._codec.quantize(k)
        v_norms, v_packed = self._codec.quantize(v)
        pw = _packed_width(D, self.bits)
        self._k_norms = mx.zeros((B, H, self._idx, ), dtype=mx.float32)
        self._k_packed = mx.zeros((B, H, self._idx, pw), dtype=mx.uint32)
        self._v_norms = mx.zeros((B, H, self._idx, ), dtype=mx.float32)
        self._v_packed = mx.zeros((B, H, self._idx, pw), dtype=mx.uint32)
        self._k_norms[:] = k_norms
        self._k_packed[:] = k_packed
        self._v_norms[:] = v_norms
        self._v_packed[:] = v_packed
        self._quantized = True
        # Free fp16
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        B, H, T_new, D = keys.shape

        if T_new > 1:
            # Prefill: fp16 (same logic as BatchKVCache)
            prev = self._idx
            if self.keys is None or (prev + T_new) > self.keys.shape[2]:
                n_steps = (self.step + T_new - 1) // self.step
                k_shape = (B, H, n_steps * self.step, D)
                v_shape = (B, H, n_steps * self.step, values.shape[3])
                new_k = mx.zeros(k_shape, keys.dtype)
                new_v = mx.zeros(v_shape, values.dtype)
                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys = self.keys[..., :prev, :]
                        self.values = self.values[..., :prev, :]
                    self.keys = mx.concatenate([self.keys, new_k], axis=2)
                    self.values = mx.concatenate([self.values, new_v], axis=2)
                else:
                    self.keys, self.values = new_k, new_v
            self.offset += T_new
            self._idx += T_new
            self.keys[..., prev:self._idx, :] = keys
            self.values[..., prev:self._idx, :] = values
            return self.keys[..., :self._idx, :], self.values[..., :self._idx, :]
        else:
            # Decode: quantize on first token
            if not self._quantized:
                self._quantize_buffer()

            self._ensure_codec(D)
            k_norms, k_packed = self._codec.quantize(keys)
            v_norms, v_packed = self._codec.quantize(values)
            pw = _packed_width(D, self.bits)

            # Grow quantized storage
            new_idx = self._idx + 1
            if new_idx > self._k_norms.shape[2]:
                alloc = ((new_idx + self.step - 1) // self.step) * self.step
                pad = alloc - self._k_norms.shape[2]
                self._k_norms = mx.concatenate([self._k_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._k_packed = mx.concatenate([self._k_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)
                self._v_norms = mx.concatenate([self._v_norms, mx.zeros((B, H, pad), dtype=mx.float32)], axis=2)
                self._v_packed = mx.concatenate([self._v_packed, mx.zeros((B, H, pad, pw), dtype=mx.uint32)], axis=2)

            self._k_norms[:, :, self._idx:new_idx] = k_norms
            self._k_packed[:, :, self._idx:new_idx] = k_packed
            self._v_norms[:, :, self._idx:new_idx] = v_norms
            self._v_packed[:, :, self._idx:new_idx] = v_packed
            self.offset += 1
            self._idx = new_idx
            return self._quantized_state

    @property
    def _quantized_state(self):
        return (
            (self._k_norms[:, :, :self._idx], self._k_packed[:, :, :self._idx]),
            (self._v_norms[:, :, :self._idx], self._v_packed[:, :, :self._idx]),
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("Left padding can only be added to empty cache")
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None and not self._quantized:
            from mlx_lm.models.cache import dynamic_roll
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        if self._quantized:
            return self._quantized_state
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., :self._idx, :]
            v = v[..., :self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        if len(v) == 4:
            self.keys, self.values, self.offset, self.left_padding = v
            self._idx = self.keys.shape[2] if self.keys is not None else 0
        else:
            # Quantized state
            (self._k_norms, self._k_packed), (self._v_norms, self._v_packed) = v[0], v[1]
            self._idx = self._k_norms.shape[2]
            self._quantized = True

    def filter(self, batch_indices):
        if self._quantized:
            self._k_norms = self._k_norms[batch_indices]
            self._k_packed = self._k_packed[batch_indices]
            self._v_norms = self._v_norms[batch_indices]
            self._v_packed = self._v_packed[batch_indices]
        else:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0 and not self._quantized:
            self.keys = self.keys[..., min_left_pad:, :]
            self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self._quantized != other._quantized:
            # Force both to quantized
            if not self._quantized:
                self._quantize_buffer()
            if not other._quantized:
                other._quantize_buffer()

        if self._quantized:
            max_idx = max(self._idx, other._idx)
            # Pad quantized tensors (trim to _idx first, then left-pad)
            def pad_q(c):
                # Trim to actual used length
                kn = c._k_norms[:, :, :c._idx]
                kp = c._k_packed[:, :, :c._idx]
                vn = c._v_norms[:, :, :c._idx]
                vp = c._v_packed[:, :, :c._idx]
                left = max_idx - c._idx
                if left > 0:
                    B, H = kn.shape[:2]
                    pw = kp.shape[-1]
                    kn = mx.concatenate([mx.zeros((B, H, left), dtype=mx.float32), kn], axis=2)
                    kp = mx.concatenate([mx.zeros((B, H, left, pw), dtype=mx.uint32), kp], axis=2)
                    vn = mx.concatenate([mx.zeros((B, H, left), dtype=mx.float32), vn], axis=2)
                    vp = mx.concatenate([mx.zeros((B, H, left, pw), dtype=mx.uint32), vp], axis=2)
                return kn, kp, vn, vp, c.offset, c.left_padding + left
            s_kn, s_kp, s_vn, s_vp, s_off, s_lp = pad_q(self)
            o_kn, o_kp, o_vn, o_vp, o_off, o_lp = pad_q(other)
            self._k_norms = mx.concatenate([s_kn, o_kn], axis=0)
            self._k_packed = mx.concatenate([s_kp, o_kp], axis=0)
            self._v_norms = mx.concatenate([s_vn, o_vn], axis=0)
            self._v_packed = mx.concatenate([s_vp, o_vp], axis=0)
            self.offset = mx.concatenate([s_off, o_off])
            self.left_padding = mx.concatenate([s_lp, o_lp])
            self._idx = max_idx
        else:
            # fp16 extend (same as BatchKVCache)
            max_idx = max(self._idx, other._idx)
            max_size = max(self.keys.shape[2], other.keys.shape[2])
            def pad(c):
                left = max_idx - c._idx
                right = max_size - c.keys.shape[2] - left
                k, v = c.keys, c.values
                if right < 0:
                    k = k[..., :right, :]; v = v[..., :right, :]
                    right = 0
                if left != 0 or right != 0:
                    p = [(0,0),(0,0),(left,right),(0,0)]
                    k = mx.pad(k, p); v = mx.pad(v, p)
                return k, v, c.offset, c.left_padding + left
            self.keys, self.values, self.offset, self.left_padding = map(
                mx.concatenate, zip(*(pad(self), pad(other)))
            )
            self._idx = max_idx

    def extract(self, idx):
        """Extract single request as TurboQuantKVCache."""
        if not self._quantized:
            self._quantize_buffer()
        padding = self.left_padding[idx].item()
        tq = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tq._ensure_codec(self._k_packed.shape[-1] * 32 // self.bits)  # infer dim from packed_width
        # Copy quantized data for this request
        end = self._idx
        tq._k_norms = mx.contiguous(self._k_norms[idx:idx+1, :, padding:end])
        tq._k_packed = mx.contiguous(self._k_packed[idx:idx+1, :, padding:end])
        tq._v_norms = mx.contiguous(self._v_norms[idx:idx+1, :, padding:end])
        tq._v_packed = mx.contiguous(self._v_packed[idx:idx+1, :, padding:end])
        tq.offset = end - padding
        tq._quantized = True
        tq._codec = self._codec
        return tq

    @classmethod
    def merge(cls, caches):
        """Merge TurboQuantKVCache instances into BatchTurboQuantKVCache."""
        bits = caches[0].bits
        seed = caches[0].seed
        lengths = [c.offset for c in caches]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        B = len(caches)

        batch = cls(padding, bits=bits, seed=seed)
        batch._codec = caches[0]._codec

        # All caches should be quantized
        if not all(c._quantized for c in caches):
            # Force quantize
            for c in caches:
                if not c._quantized and c._fp16_keys is not None:
                    c._quantize_fp16_buffer()

        H = caches[0]._k_norms.shape[1]
        pw = caches[0]._k_packed.shape[-1]

        k_norms = mx.zeros((B, H, max_length), dtype=mx.float32)
        k_packed = mx.zeros((B, H, max_length, pw), dtype=mx.uint32)
        v_norms = mx.zeros((B, H, max_length), dtype=mx.float32)
        v_packed = mx.zeros((B, H, max_length, pw), dtype=mx.uint32)

        for i, (p, c) in enumerate(zip(padding, caches)):
            T = c.offset
            if c._k_norms is not None:
                k_norms[i:i+1, :, p:p+T] = c._k_norms[:, :, :T]
                k_packed[i:i+1, :, p:p+T] = c._k_packed[:, :, :T]
                v_norms[i:i+1, :, p:p+T] = c._v_norms[:, :, :T]
                v_packed[i:i+1, :, p:p+T] = c._v_packed[:, :, :T]

        batch._k_norms = k_norms
        batch._k_packed = k_packed
        batch._v_norms = v_norms
        batch._v_packed = v_packed
        batch.offset += max_length
        batch._idx = max_length
        batch._quantized = True
        return batch

    def decode_attention(
        self,
        queries: mx.array,      # (B, H_q, 1, D)
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        """Fused 2-pass Flash Attention for batch decode. No dequantize."""
        if keys_state is None:
            keys_state, values_state = self._quantized_state
        k_norms, k_packed = keys_state
        v_norms, v_packed = values_state

        B, H_q, L, D = queries.shape
        H_kv = k_norms.shape[1]

        # Prepare queries: scale and flatten to (B*H_q, D)
        q_flat = (queries.squeeze(2) * scale).reshape(B * H_q, D).astype(mx.float16)

        # Rotate queries (same rotation as codec)
        R = self._codec.rotation
        q_grouped = q_flat.reshape(B * H_q, D // self._codec.dim, self._codec.dim)
        q_rot = (q_grouped.astype(mx.float32) @ R).reshape(B * H_q, D).astype(mx.float16)

        # Fused 2-pass SDPA
        out = _fused_tq_sdpa(
            q_rot, k_packed, k_norms, self._codec.codebook,
            v_packed, v_norms, self._codec.codebook,
            scale=1.0,  # already applied to queries
            B=B, H_q=H_q, H_kv=H_kv, D=D, bits=self.bits,
        )

        # Inverse rotate output
        out_grouped = out.reshape(B * H_q, D // self._codec.dim, self._codec.dim).astype(mx.float32)
        out_restored = (out_grouped @ R.T).reshape(B * H_q, D).astype(queries.dtype)

        return out_restored.reshape(B, H_q, 1, D)

    def dequantize(self, keys_state=None, values_state=None):
        """Dequantize batch quantized state for SDPA fallback."""
        if keys_state is None:
            keys_state, values_state = self._quantized_state
        k_norms, k_packed = keys_state
        v_norms, v_packed = values_state
        keys = self._codec.dequantize(k_norms, k_packed)
        values = self._codec.dequantize(v_norms, v_packed)
        return keys, values

    def make_mask(self, N, return_array=False, **kwargs):
        from mlx_lm.models.cache import create_causal_mask
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def empty(self):
        return self.keys is None and self._k_norms is None

    @property
    def nbytes(self):
        if self._quantized:
            return (self._k_norms.nbytes + self._k_packed.nbytes +
                    self._v_norms.nbytes + self._v_packed.nbytes)
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def size(self):
        return max(0, max(self.offset.tolist())) if isinstance(self.offset, mx.array) else self.offset

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass


def turboquant_enabled(bits, scheme=None):
    """Check if TurboQuant should be used for given bits/scheme."""
    if scheme == "turboquant":
        return True
    if bits is not None and not float(bits).is_integer():
        return True
    return False
