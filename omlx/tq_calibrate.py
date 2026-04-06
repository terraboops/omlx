# SPDX-License-Identifier: Apache-2.0
"""TQ3 codebook calibration on real KV data.

Runs calibration prompts through the model, collects KV vectors from
all layers, then learns:
  1. Optimal Givens rotation angles (minimize component correlation)
  2. Data-adaptive codebook boundaries (Lloyd's algorithm on real KV)

This replaces the random rotation + Beta-distribution codebook with
data-driven parameters that match the model's actual KV distribution.

Usage:
    python -m omlx.tq_calibrate \\
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \\
        --output tq3_calibrated.npz \\
        --num-tokens 4096

Then use with hypercar_server:
    python -m omlx.hypercar_server \\
        --model ... --tq-calibration tq3_calibrated.npz
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger("tq_calibrate")


# Calibration text — diverse Python code to exercise the model
CALIBRATION_CODE = '''
"""Utility functions for data processing and API handling."""

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

@dataclass
class Config:
    """Application configuration with validation."""
    name: str
    port: int = 8080
    workers: int = 4
    debug: bool = False
    database_url: str = "sqlite:///app.db"

    def validate(self) -> List[str]:
        errors = []
        if self.port < 1 or self.port > 65535:
            errors.append(f"Invalid port: {self.port}")
        if self.workers < 1:
            errors.append("Workers must be >= 1")
        return errors

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self._cache: Dict[str, Any] = {}
        self._stats = defaultdict(int)

    def process_batch(self, items: List[Dict]) -> List[Dict]:
        results = []
        for item in items:
            try:
                processed = self._transform(item)
                results.append(processed)
                self._stats["processed"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                results.append({"error": str(e), "item": item})
        return results

    def _transform(self, item: Dict) -> Dict:
        return {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}

def fibonacci_generator(n: int):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def binary_search(arr: List[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

async def fetch_data(urls: List[str]) -> List[Dict]:
    async def fetch_one(url: str) -> Dict:
        await asyncio.sleep(0.1)
        return {"url": url, "status": 200}
    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)

class Graph:
    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        self.edges[src].append((dst, weight))

    def bfs(self, start: str) -> List[str]:
        from collections import deque
        visited = {start}
        queue = deque([start])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor, _ in self.edges[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result
'''


def collect_kv_vectors(model, tokenizer, calibration_text: str,
                       target_tokens: int = 4096) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Run model on calibration text and collect KV vectors from all layers.

    Returns: dict[layer_idx] → (keys_array, values_array)
    Each array has shape (num_tokens, head_dim) flattened across heads.
    """
    # Tokenize and truncate
    tokens = tokenizer.encode(calibration_text)[:target_tokens]
    logger.info(f"Collecting KV from {len(tokens)} tokens across {model.args.num_hidden_layers} layers")

    # Create standard fp16 cache to collect real KV data
    from mlx_lm.models.cache import KVCache
    n_layers = model.args.num_hidden_layers
    cache = [KVCache() for _ in range(n_layers)]

    # Run model
    x = mx.array([tokens])
    _ = model(x, cache=cache)
    mx.eval(_)

    # Extract KV vectors
    kv_data = {}
    for i, c in enumerate(cache):
        if c.keys is not None:
            # keys/values shape: (B, H_kv, T, D) — take first batch, flatten heads
            k = np.array(c.keys[0].astype(mx.float32))  # (H_kv, T, D)
            v = np.array(c.values[0].astype(mx.float32))
            H, T, D = k.shape
            # Flatten heads and tokens → (H*T, D)
            k_flat = k.reshape(-1, D)
            v_flat = v.reshape(-1, D)
            kv_data[i] = (k_flat, v_flat)

    del cache
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    logger.info(f"Collected KV from {len(kv_data)} layers, shape per layer: {kv_data[0][0].shape}")
    return kv_data


def calibrate_givens_angles(vectors: np.ndarray, n_pairs: int = None,
                            n_angles_per_pair: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Learn Givens rotation angles that minimize component correlation.

    For each pair of dimensions (i, 2i+1), find the angle that makes the
    rotated components as independent as possible (minimizes |correlation|).

    Returns: (cos_angles, sin_angles) each of shape (D//2,)
    """
    N, D = vectors.shape
    if n_pairs is None:
        n_pairs = D // 2

    cos_angles = np.ones(n_pairs, dtype=np.float32)
    sin_angles = np.zeros(n_pairs, dtype=np.float32)

    for p in range(n_pairs):
        i, j = 2 * p, 2 * p + 1
        if j >= D:
            break

        x, y = vectors[:, i], vectors[:, j]
        best_mse = float('inf')
        best_angle = 0.0

        # Search over angles
        for angle in np.linspace(0, np.pi, n_angles_per_pair):
            c, s = np.cos(angle), np.sin(angle)
            rx = c * x - s * y
            ry = s * x + c * y

            # After rotation, quantize and reconstruct
            # MSE = ||original - reconstructed||^2
            # For now, use correlation as proxy (lower = better decorrelation)
            corr = np.abs(np.corrcoef(rx, ry)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            if corr < best_mse:
                best_mse = corr
                best_angle = angle

        cos_angles[p] = np.cos(best_angle)
        sin_angles[p] = np.sin(best_angle)

    return cos_angles, sin_angles


def calibrate_codebook(vectors: np.ndarray, cos_a: np.ndarray, sin_a: np.ndarray,
                       bits: int = 3) -> np.ndarray:
    """Learn codebook from rotated real KV data using Lloyd's algorithm.

    1. Apply Givens rotation to vectors
    2. Normalize to unit vectors
    3. Run Lloyd's on the rotated+normalized coordinate values
    """
    N, D = vectors.shape
    n_levels = 1 << bits

    # Apply Givens rotation
    rotated = vectors.copy()
    for p in range(len(cos_a)):
        i, j = 2 * p, 2 * p + 1
        if j >= D:
            break
        c, s = cos_a[p], sin_a[p]
        x, y = rotated[:, i].copy(), rotated[:, j].copy()
        rotated[:, i] = c * x - s * y
        rotated[:, j] = s * x + c * y

    # Normalize
    norms = np.linalg.norm(rotated, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = rotated / norms

    # Collect all coordinate values for Lloyd's
    coords = normalized.flatten()

    # Lloyd's algorithm
    centroids = np.linspace(coords.min(), coords.max(), n_levels)
    for _ in range(100):
        dists = np.abs(coords[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)
        for j in range(n_levels):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = coords[mask].mean()

    return np.sort(centroids).astype(np.float32)


def calibrate(model, tokenizer, bits: int = 3,
              target_tokens: int = 4096) -> Dict[str, np.ndarray]:
    """Full calibration pipeline. Returns calibration data dict."""

    # Collect real KV data
    calib_text = CALIBRATION_CODE * ((target_tokens // len(CALIBRATION_CODE.split())) + 1)
    kv_data = collect_kv_vectors(model, tokenizer, calib_text, target_tokens)

    # Calibrate per-layer (or use global stats)
    # For efficiency, use a representative subset of layers
    sample_layers = [0, 8, 16, 24, 32, 40, 47]
    all_keys = []
    all_values = []
    for layer_idx in sample_layers:
        if layer_idx in kv_data:
            k, v = kv_data[layer_idx]
            all_keys.append(k)
            all_values.append(v)

    keys_combined = np.concatenate(all_keys, axis=0)
    values_combined = np.concatenate(all_values, axis=0)
    logger.info(f"Calibrating on {keys_combined.shape[0]} key vectors + {values_combined.shape[0]} value vectors")

    # Subsample if too large
    if keys_combined.shape[0] > 100_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(keys_combined.shape[0], 100_000, replace=False)
        keys_combined = keys_combined[idx]
        values_combined = values_combined[idx]

    # Calibrate Givens angles on keys
    t0 = time.perf_counter()
    k_cos, k_sin = calibrate_givens_angles(keys_combined)
    logger.info(f"Key Givens calibration: {time.perf_counter()-t0:.1f}s")

    # Calibrate codebook on keys
    t0 = time.perf_counter()
    codebook = calibrate_codebook(keys_combined, k_cos, k_sin, bits=bits)
    logger.info(f"Codebook calibration: {time.perf_counter()-t0:.1f}s")

    # Measure reconstruction error
    D = keys_combined.shape[1]
    from omlx.turboquant_kv import TurboQuantMSECodec
    # Build calibrated codec
    calib_codec = TurboQuantMSECodec.__new__(TurboQuantMSECodec)
    calib_codec.dim = D
    calib_codec.bits = bits
    calib_codec.seed = 0
    calib_codec.codebook = mx.array(codebook)
    calib_codec.use_givens = True
    calib_codec._givens_cos = mx.array(k_cos)
    calib_codec._givens_sin = mx.array(k_sin)
    from omlx.turboquant_kv import _packed_width
    calib_codec._pw = _packed_width(D, bits)
    cb = calib_codec.codebook
    calib_codec._boundaries = (cb[:-1] + cb[1:]) / 2

    # Test reconstruction
    test_keys = mx.array(keys_combined[:1000])
    k_norms, k_packed = calib_codec.quantize(test_keys.reshape(1, 1, -1, D))
    k_recon = calib_codec.dequantize(k_norms, k_packed).reshape(-1, D)
    mx.eval(k_recon)

    mse = float(mx.mean((test_keys - k_recon) ** 2).item())
    cos_sim = float(mx.mean(
        mx.sum(test_keys * k_recon, axis=-1) /
        (mx.linalg.norm(test_keys, axis=-1) * mx.linalg.norm(k_recon, axis=-1) + 1e-8)
    ).item())

    logger.info(f"Reconstruction: MSE={mse:.6f} cosine={cos_sim:.6f}")

    return {
        "givens_cos": k_cos,
        "givens_sin": k_sin,
        "codebook": codebook,
        "bits": np.array([bits]),
        "dim": np.array([D]),
        "mse": np.array([mse]),
        "cosine": np.array([cos_sim]),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate TQ3 codebook on real KV data")
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
    parser.add_argument("--output", default="tq3_calibrated.npz")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--num-tokens", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from mlx_lm import load
    logger.info(f"Loading {args.model}...")
    model, tokenizer = load(args.model)
    logger.info(f"Model loaded: {mx.get_active_memory()/1e9:.1f}GB")

    result = calibrate(model, tokenizer, bits=args.bits, target_tokens=args.num_tokens)

    np.savez(args.output, **result)
    logger.info(f"Saved calibration to {args.output}")
    logger.info(f"  bits={int(result['bits'][0])} dim={int(result['dim'][0])}")
    logger.info(f"  MSE={float(result['mse'][0]):.6f} cosine={float(result['cosine'][0]):.6f}")


if __name__ == "__main__":
    main()
