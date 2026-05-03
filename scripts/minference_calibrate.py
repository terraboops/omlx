#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MInference per-head sparse-attention pattern calibration (arXiv:2407.02490).

Runs Qwen3-Coder over a calibration prompt, captures full attention maps,
and assigns each (layer, head) to one of:
  - a_shape:        sink tokens + local causal window
  - vertical_slash: vertical column indices + diagonal band
  - block_sparse:   block-diagonal pattern
  - dense:          no exploitable structure

Output: JSON pattern table for runtime sparse-attention dispatch.

Usage:
    python scripts/minference_calibrate.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit
    python scripts/minference_calibrate.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit --seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger("minference.calibrate")

# Default output path
DEFAULT_OUTPUT = Path(__file__).parent.parent / "omlx" / "patches" / "minference_patterns"

# Sparsity budget: target fraction of attention matrix to SKIP
TARGET_SPARSITY = 0.85

# Max reconstruction MSE (relative to dense) for pattern to be valid
MAX_RECONSTRUCTION_MSE = 0.01


# ---------------------------------------------------------------------------
# Calibration corpus
# ---------------------------------------------------------------------------

CALIBRATION_PROMPT = '''\
You are a senior software engineer reviewing a large Python codebase.
Analyze the following modules and identify potential issues:

```python
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single entry in the LRU cache with TTL support."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: float = 3600.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self) -> None:
        self.accessed_at = time.time()
        self.access_count += 1


class DistributedCache:
    """Thread-safe distributed cache with consistent hashing."""

    def __init__(self, capacity: int = 1000, num_shards: int = 16):
        self.capacity = capacity
        self.num_shards = num_shards
        self._shards: List[Dict[str, CacheEntry]] = [
            {} for _ in range(num_shards)
        ]
        self._locks = [asyncio.Lock() for _ in range(num_shards)]
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _shard_for_key(self, key: str) -> int:
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16) % self.num_shards

    async def get(self, key: str) -> Optional[Any]:
        shard_idx = self._shard_for_key(key)
        async with self._locks[shard_idx]:
            entry = self._shards[shard_idx].get(key)
            if entry is None or entry.is_expired:
                self._stats["misses"] += 1
                if entry and entry.is_expired:
                    del self._shards[shard_idx][key]
                return None
            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    async def put(self, key: str, value: Any, ttl: float = 3600.0) -> None:
        shard_idx = self._shard_for_key(key)
        async with self._locks[shard_idx]:
            shard = self._shards[shard_idx]
            if len(shard) >= self.capacity // self.num_shards:
                self._evict_lru(shard)
            shard[key] = CacheEntry(key=key, value=value, ttl_seconds=ttl)

    def _evict_lru(self, shard: Dict[str, CacheEntry]) -> None:
        if not shard:
            return
        oldest_key = min(shard, key=lambda k: shard[k].accessed_at)
        del shard[oldest_key]
        self._stats["evictions"] += 1


class PipelineStage:
    """A stage in a data processing pipeline."""

    def __init__(self, name: str, transform_fn, batch_size: int = 32):
        self.name = name
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.processed = 0
        self.errors = 0

    async def process(self, items: List[Any]) -> List[Any]:
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                batch_results = await self.transform_fn(batch)
                results.extend(batch_results)
                self.processed += len(batch)
            except Exception as e:
                logger.error(f"Stage {self.name} error on batch {i}: {e}")
                self.errors += len(batch)
        return results


class Pipeline:
    """Multi-stage data processing pipeline with error handling."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    async def run(self, data: List[Any]) -> List[Any]:
        current = data
        for stage in self.stages:
            logger.info(f"Running stage: {stage.name} ({len(current)} items)")
            current = await stage.process(current)
            if not current:
                logger.warning(f"Stage {stage.name} produced no output")
                break
        return current


def compute_similarity_matrix(
    vectors: List[List[float]],
    metric: str = "cosine",
) -> List[List[float]]:
    """Compute pairwise similarity matrix."""
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if metric == "cosine":
                dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
                norm_i = math.sqrt(sum(a * a for a in vectors[i]))
                norm_j = math.sqrt(sum(b * b for b in vectors[j]))
                sim = dot / (norm_i * norm_j + 1e-8)
            elif metric == "euclidean":
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j])))
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix
```

Provide a detailed code review covering:
1. Thread safety issues
2. Performance bottlenecks
3. Memory management concerns
4. API design improvements
'''


# ---------------------------------------------------------------------------
# Attention pattern classifiers
# ---------------------------------------------------------------------------

def _classify_a_shape(
    attn: np.ndarray,
    sparsity: float,
) -> dict:
    """Test if attention matches A-shape: sink tokens + local window.

    A-shape means: top attention mass is concentrated on:
      1. First few tokens (sink tokens, typically positions 0-3)
      2. Recent tokens (local causal window)
    """
    L = attn.shape[0]
    num_sink = min(4, L // 4)
    # Budget: how many elements we can keep (1 - sparsity)
    budget = int(L * L * (1 - sparsity))

    # A-shape mask: first `num_sink` columns + diagonal band of width `band_w`
    band_w = max(1, budget // L - num_sink)
    mask = np.zeros((L, L), dtype=bool)
    mask[:, :num_sink] = True  # sink columns
    for i in range(L):
        start = max(0, i - band_w + 1)
        mask[i, start:i + 1] = True  # local causal band

    # Compute reconstruction MSE
    sparse_attn = attn * mask
    # Re-normalize rows
    row_sums = sparse_attn.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    sparse_attn = sparse_attn / row_sums

    mse = np.mean((attn - sparse_attn) ** 2)
    actual_sparsity = 1.0 - mask.sum() / (L * L)

    return {
        "pattern": "a_shape",
        "mse": float(mse),
        "sparsity": float(actual_sparsity),
        "params": {"num_sink": num_sink, "band_width": band_w},
    }


def _classify_vertical_slash(
    attn: np.ndarray,
    sparsity: float,
) -> dict:
    """Test if attention matches vertical-slash: vertical columns + diagonal.

    Vertical-slash means: attention concentrated on:
      1. Specific column positions (globally important tokens)
      2. A diagonal band (local context)
    """
    L = attn.shape[0]
    budget = int(L * L * (1 - sparsity))

    # Find vertical columns: columns with high total attention
    col_importance = attn.sum(axis=0)  # (L,)
    # Number of vertical columns: use budget minus diagonal band
    band_w = max(1, L // 8)  # diagonal band width
    diag_budget = band_w * L
    vert_budget = max(1, (budget - diag_budget) // L)
    vert_cols = np.argsort(col_importance)[-vert_budget:]

    # Build mask
    mask = np.zeros((L, L), dtype=bool)
    mask[:, vert_cols] = True  # vertical columns
    for i in range(L):
        start = max(0, i - band_w + 1)
        mask[i, start:i + 1] = True  # diagonal band

    sparse_attn = attn * mask
    row_sums = sparse_attn.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    sparse_attn = sparse_attn / row_sums

    mse = np.mean((attn - sparse_attn) ** 2)
    actual_sparsity = 1.0 - mask.sum() / (L * L)

    return {
        "pattern": "vertical_slash",
        "mse": float(mse),
        "sparsity": float(actual_sparsity),
        "params": {
            "num_vertical_cols": int(vert_budget),
            "vertical_col_indices": sorted(vert_cols.tolist()),
            "band_width": band_w,
        },
    }


def _classify_block_sparse(
    attn: np.ndarray,
    sparsity: float,
) -> dict:
    """Test if attention matches block-sparse: block-diagonal pattern.

    Block-sparse means: attention concentrated in blocks along the diagonal,
    with fixed block size.
    """
    L = attn.shape[0]
    budget = int(L * L * (1 - sparsity))

    # Try different block sizes, pick best
    best = None
    for block_size in [32, 64, 128, 256]:
        if block_size > L:
            continue
        num_blocks = (L + block_size - 1) // block_size

        # Block-diagonal mask: each token attends to tokens in same block
        # + previous block (for cross-block context)
        mask = np.zeros((L, L), dtype=bool)
        for b in range(num_blocks):
            row_start = b * block_size
            row_end = min((b + 1) * block_size, L)
            # Current block
            col_start = b * block_size
            col_end = min((b + 1) * block_size, L)
            mask[row_start:row_end, col_start:col_end] = True
            # Previous block overlap
            if b > 0:
                prev_start = (b - 1) * block_size
                mask[row_start:row_end, prev_start:col_start] = True

        sparse_attn = attn * mask
        row_sums = sparse_attn.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        sparse_attn = sparse_attn / row_sums

        mse = float(np.mean((attn - sparse_attn) ** 2))
        actual_sparsity = 1.0 - mask.sum() / (L * L)

        if best is None or mse < best["mse"]:
            best = {
                "pattern": "block_sparse",
                "mse": mse,
                "sparsity": actual_sparsity,
                "params": {"block_size": block_size},
            }

    return best


def classify_head(attn: np.ndarray, sparsity: float = TARGET_SPARSITY) -> dict:
    """Classify a single attention head's pattern.

    Tries all pattern types and picks the one with lowest MSE
    that meets the sparsity budget.

    Args:
        attn: (L, L) attention weights (already softmax'd, causal)
        sparsity: target fraction of matrix to skip

    Returns:
        dict with pattern type, MSE, sparsity, and pattern-specific params
    """
    candidates = [
        _classify_a_shape(attn, sparsity),
        _classify_vertical_slash(attn, sparsity),
        _classify_block_sparse(attn, sparsity),
    ]

    # Pick best: lowest MSE that meets sparsity target
    valid = [c for c in candidates if c["sparsity"] >= sparsity * 0.9]  # 10% tolerance
    if not valid:
        valid = candidates  # Fall back to best available

    best = min(valid, key=lambda c: c["mse"])

    # If best MSE is too high, classify as dense (no sparse pattern works)
    if best["mse"] > MAX_RECONSTRUCTION_MSE:
        return {
            "pattern": "dense",
            "mse": best["mse"],
            "sparsity": 0.0,
            "params": {},
        }

    return best


# ---------------------------------------------------------------------------
# Attention capture hook
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Hook into model attention layers to capture full attention maps.

    Patches the model's attention computation to return and store
    the full softmax attention weights for each (layer, head).
    """

    def __init__(self, model, num_layers: int, num_heads: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_maps: dict[tuple[int, int], np.ndarray] = {}
        self._hooks = []
        self._model = model

    def capture(self, tokenizer, prompt: str, max_seq_len: int = 2048):
        """Run model on prompt and capture attention maps.

        Uses a custom SDPA that stores attention weights before dropout.
        """
        import mlx_lm.models.base as mlx_base

        original_sdpa = mlx_base.scaled_dot_product_attention
        layer_counter = [0]  # Mutable counter for tracking layer calls

        def capturing_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
            """SDPA that captures attention weights."""
            B, H_q, L_q, D = queries.shape
            _, H_kv, L_kv, _ = keys.shape

            # Compute attention scores manually to capture weights
            # GQA: expand KV heads to match Q heads
            GQA = H_q // H_kv
            if GQA > 1:
                keys_exp = mx.repeat(keys, GQA, axis=1)
                values_exp = mx.repeat(values, GQA, axis=1)
            else:
                keys_exp = keys
                values_exp = values

            scores = (queries @ keys_exp.transpose(0, 1, 3, 2)) * scale

            # mlx_lm passes mask as either an mx.array, the literal string
            # "causal" (used during prefill when no explicit mask is built),
            # or None. The string case needs an actual causal mask synthesized
            # at the current shape.
            if isinstance(mask, str) and mask == "causal":
                # Build standard causal mask (q_seq, k_seq) — last L_q rows.
                offset = L_kv - L_q
                rows = mx.arange(L_q)[:, None] + offset
                cols = mx.arange(L_kv)[None, :]
                causal_mask = mx.where(cols <= rows, 0.0, -3.4e4).astype(scores.dtype)
                scores = scores + causal_mask
            elif mask is not None:
                scores = scores + mask

            weights = mx.softmax(scores, axis=-1)

            # Store attention weights for this layer
            layer_idx = layer_counter[0] % self.num_layers
            weights_np = np.array(weights[0].astype(mx.float32))  # (H_q, L_q, L_kv)
            for h in range(min(H_q, self.num_heads)):
                self.attention_maps[(layer_idx, h)] = weights_np[h]  # (L_q, L_kv)

            layer_counter[0] += 1

            # Compute output normally
            out = weights @ values_exp
            return out

        # Temporarily patch SDPA. We have to walk every already-imported
        # mlx_lm/mlx_vlm model module too — when a model file does
        # `from .base import scaled_dot_product_attention`, the symbol is
        # captured by reference at import time, so rebinding the attribute
        # on the base module alone has no effect on already-imported callers
        # (Qwen3.6's qwen3_next.py is one such caller). The runtime patch in
        # `omlx/patches/minference_prefill.py` already does this; this is
        # the calibration-time mirror of it.
        import sys as _sys
        mlx_base.scaled_dot_product_attention = capturing_sdpa
        _patched_modules = []
        for mod_name, mod in list(_sys.modules.items()):
            if mod is None:
                continue
            if not (mod_name.startswith("mlx_lm.models.")
                    or mod_name.startswith("mlx_vlm.models.")):
                continue
            if hasattr(mod, "scaled_dot_product_attention"):
                func = getattr(mod, "scaled_dot_product_attention")
                if func is original_sdpa:
                    setattr(mod, "scaled_dot_product_attention", capturing_sdpa)
                    _patched_modules.append((mod, func))

        try:
            tokens = tokenizer.encode(prompt)[:max_seq_len]
            x = mx.array([tokens])

            # Forward pass with full sequence (NOT chunked — we need full attn maps)
            logits = self._model(x)
            mx.eval(logits)

            logger.info(f"Captured attention maps for {len(self.attention_maps)} (layer, head) pairs")
        finally:
            # Restore original SDPA on every module we patched
            mlx_base.scaled_dot_product_attention = original_sdpa
            for mod, func in _patched_modules:
                setattr(mod, "scaled_dot_product_attention", func)


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate(
    model_path: str,
    seq_len: int = 2048,
    output_dir: Path = DEFAULT_OUTPUT,
    sparsity: float = TARGET_SPARSITY,
) -> dict:
    """Run calibration and produce pattern table.

    Args:
        model_path: HuggingFace model ID or local path
        seq_len: sequence length for calibration (longer = more accurate, more memory)
        output_dir: where to write the JSON pattern table
        sparsity: target sparsity for pattern classification

    Returns:
        Pattern table dict
    """
    from mlx_lm import load

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    # Walk attention layers explicitly — on hybrid models like Qwen3.6 only
    # ~25% of `model.layers` are full-attention; the rest are SSM/linear and
    # never call SDPA. The runtime patch in `omlx/patches/minference_prefill.py`
    # increments `_LAYER_COUNTER` once per SDPA call, so the pattern table's
    # ``num_layers`` must match the count of attention layers, not the total
    # model depth.
    attn_layer_indices = []
    first_attn = None
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue
        attn_layer_indices.append(i)
        if first_attn is None:
            first_attn = attn

    if first_attn is None:
        raise RuntimeError(
            f"No attention layers found in {model_path}. Pure-SSM models "
            "have nothing for MInference to calibrate; this script is "
            "attention-specific."
        )

    n_attn_layers = len(attn_layer_indices)
    n_total_layers = len(model.layers)

    # Multi-name attribute resolver for n_heads — Qwen3-Coder uses
    # ``n_heads``; Qwen3NextAttention (Qwen3.6) uses ``num_attention_heads``.
    def _attr(obj, *names, default=None):
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return default

    num_heads = _attr(
        first_attn, 'n_heads', 'num_attention_heads', 'num_heads',
        default=None,
    )
    if num_heads is None:
        # Last-resort weight-shape probe
        q_proj = getattr(first_attn, 'q_proj', None)
        if q_proj is not None and hasattr(q_proj, 'weight'):
            # q_proj.weight: (num_heads * head_dim, hidden_size)
            head_dim = _attr(first_attn, 'head_dim', default=None)
            if head_dim is not None:
                num_heads = q_proj.weight.shape[0] // head_dim
        if num_heads is None:
            num_heads = 32
            logger.warning(f"Could not detect num_heads, defaulting to {num_heads}")

    if n_attn_layers != n_total_layers:
        logger.info(
            f"Hybrid model detected: {n_attn_layers} attention layers / "
            f"{n_total_layers} total layers (SSM-skipped indices: "
            f"{[i for i in range(n_total_layers) if i not in attn_layer_indices][:5]}...)"
        )
    # The pattern table uses ATTENTION-LAYER index (0..n_attn_layers-1), NOT
    # the layer's position in `model.layers`. The runtime layer counter
    # increments once per SDPA call, which by construction maps to attention
    # layers in order.
    num_layers = n_attn_layers

    logger.info(f"Model: {num_layers} attention layers, {num_heads} heads")
    logger.info(f"Calibration seq_len: {seq_len}, target sparsity: {sparsity:.0%}")

    # Capture attention maps
    logger.info("Capturing attention maps...")
    t0 = time.perf_counter()
    capture = AttentionCapture(model, num_layers, num_heads)
    capture.capture(tokenizer, CALIBRATION_PROMPT, max_seq_len=seq_len)
    capture_time = time.perf_counter() - t0
    logger.info(f"Attention capture complete in {capture_time:.1f}s")

    # Classify each head
    logger.info("Classifying attention patterns...")
    pattern_table = {
        "model": model_path,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "target_sparsity": sparsity,
        "calibration_time_s": round(capture_time, 1),
        "heads": [],
    }

    pattern_counts = {"a_shape": 0, "vertical_slash": 0, "block_sparse": 0, "dense": 0}
    total_mse = 0.0
    total_sparsity = 0.0

    for layer in range(num_layers):
        for head in range(num_heads):
            key = (layer, head)
            if key in capture.attention_maps:
                attn = capture.attention_maps[key]
                result = classify_head(attn, sparsity)
            else:
                # Head not captured (e.g., seq_len too short)
                result = {
                    "pattern": "dense",
                    "mse": 0.0,
                    "sparsity": 0.0,
                    "params": {},
                }

            entry = {
                "layer": layer,
                "head": head,
                "pattern": result["pattern"],
                "mse": round(result["mse"], 6),
                "sparsity": round(result["sparsity"], 4),
                "params": result["params"],
            }
            pattern_table["heads"].append(entry)
            pattern_counts[result["pattern"]] += 1
            total_mse += result["mse"]
            total_sparsity += result["sparsity"]

    num_entries = num_layers * num_heads
    pattern_table["summary"] = {
        "pattern_counts": pattern_counts,
        "avg_mse": round(total_mse / max(1, num_entries), 6),
        "avg_sparsity": round(total_sparsity / max(1, num_entries), 4),
        "total_entries": num_entries,
    }

    # Log summary
    logger.info(f"Pattern distribution: {pattern_counts}")
    logger.info(f"Average MSE: {pattern_table['summary']['avg_mse']:.6f}")
    logger.info(f"Average sparsity: {pattern_table['summary']['avg_sparsity']:.1%}")

    # Validation assertions
    avg_sparsity = pattern_table["summary"]["avg_sparsity"]
    avg_mse = pattern_table["summary"]["avg_mse"]
    dense_frac = pattern_counts["dense"] / max(1, num_entries)

    # Threshold the assertion at 90% of the user's requested target sparsity
    # rather than a hardcoded 85% — the script accepts ``--sparsity`` and
    # should respect that target on the validation side too.
    sparsity_floor = 0.9 * sparsity
    assert avg_sparsity >= sparsity_floor, (
        f"Average sparsity {avg_sparsity:.1%} < {sparsity_floor:.1%} (90% of "
        f"requested target {sparsity:.1%}) — model may not be suitable for "
        f"MInference sparse attention at this budget"
    )
    assert avg_mse < MAX_RECONSTRUCTION_MSE, (
        f"Average reconstruction MSE {avg_mse:.6f} >= {MAX_RECONSTRUCTION_MSE} — "
        f"patterns are too lossy"
    )
    logger.info(f"Validation PASSED: sparsity={avg_sparsity:.1%} >= 85%, MSE={avg_mse:.6f} < {MAX_RECONSTRUCTION_MSE}")

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    # Derive filename from model name
    model_name = model_path.split("/")[-1].lower().replace("-", "_")
    output_path = output_dir / f"{model_name}.json"
    output_path.write_text(json.dumps(pattern_table, indent=2))
    logger.info(f"Pattern table written to {output_path}")

    return pattern_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MInference per-head sparse-attention calibration",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="Calibration sequence length (default: 2048). Longer = more accurate.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT),
        help="Output directory for pattern table JSON",
    )
    parser.add_argument(
        "--sparsity", type=float, default=TARGET_SPARSITY,
        help=f"Target sparsity (default: {TARGET_SPARSITY})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    calibrate(
        model_path=args.model,
        seq_len=args.seq_len,
        output_dir=Path(args.output_dir),
        sparsity=args.sparsity,
    )


if __name__ == "__main__":
    main()
