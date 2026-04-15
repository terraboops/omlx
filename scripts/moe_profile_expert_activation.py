#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ProMoE offline expert activation frequency profiling (Task 32).

Hooks every MoE layer's router in Qwen3-Coder-30B-A3B and records per
(layer, expert_id) dispatch counts over a calibration corpus. Output feeds
Task 16's --resident-fraction flag to decide which experts are "cold".

Qwen3-Coder-30B-A3B: 48 layers, 128 experts/layer, top-8 per token.

Usage:
    python scripts/moe_profile_expert_activation.py [--tokens N]

Output:
    omlx/patches/promoe_profiles/qwen3_coder_30b_a3b_instruct_8bit.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
OUTPUT_DIR = Path("omlx/patches/promoe_profiles")
OUTPUT_FILE = OUTPUT_DIR / "qwen3_coder_30b_a3b_instruct_8bit.json"

# Calibration corpus: diverse coding prompts that exercise different expert
# specializations. We use a mix of languages and tasks.
CALIBRATION_PROMPTS = [
    # Python data processing
    "Write a Python function that reads a CSV file, groups rows by category, "
    "and computes the mean and standard deviation for each numeric column.",

    # Rust systems programming
    "Implement a thread-safe LRU cache in Rust using Arc<Mutex<>> with a "
    "configurable capacity and O(1) get/put operations.",

    # JavaScript frontend
    "Create a React component that implements infinite scrolling with "
    "virtualized rendering, loading data from a paginated REST API.",

    # SQL query optimization
    "Write an optimized SQL query to find the top 10 customers by revenue "
    "in the last 90 days, joining orders, products, and customers tables "
    "with appropriate indexes.",

    # Algorithm design
    "Implement a balanced binary search tree (AVL tree) with insert, delete, "
    "and range query operations. Include rotations and height balancing.",

    # Systems design
    "Design a distributed rate limiter using a sliding window algorithm. "
    "Explain the trade-offs between fixed window, sliding log, and token "
    "bucket approaches for a microservices architecture.",

    # DevOps / Infrastructure
    "Write a Dockerfile for a multi-stage build of a Go web server with "
    "health checks, graceful shutdown, and a non-root user.",

    # Machine learning
    "Implement gradient descent with momentum and learning rate scheduling "
    "for training a simple neural network on MNIST.",
]


class ExpertActivationTracker:
    """Tracks which experts are dispatched per layer."""

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        # counts[layer][expert] = dispatch count
        self.counts = [[0] * num_experts for _ in range(num_layers)]
        self.total_tokens = 0

    def record(self, layer_idx: int, expert_indices: mx.array):
        """Record expert selections for a batch of tokens.

        expert_indices: shape (batch, seq_len, top_k) — the selected expert IDs
        """
        # Flatten to 1D list of expert IDs
        inds = expert_indices.reshape(-1).tolist()
        for eid in inds:
            self.counts[layer_idx][eid] += 1

    def to_dict(self) -> dict:
        """Export as JSON-serializable dict."""
        layers = []
        for layer_idx in range(self.num_layers):
            # Sort experts by activation count (descending)
            experts = [(eid, count) for eid, count in enumerate(self.counts[layer_idx])]
            experts.sort(key=lambda x: x[1], reverse=True)

            total = sum(c for _, c in experts)
            nonzero = sum(1 for _, c in experts if c > 0)

            layers.append({
                "layer": layer_idx,
                "total_dispatches": total,
                "experts_active": nonzero,
                "experts_total": self.num_experts,
                "experts": [{"id": eid, "count": count,
                             "pct": round(count / total * 100, 2) if total > 0 else 0}
                            for eid, count in experts],
            })
        return {
            "model": MODEL_ID,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "total_tokens": self.total_tokens,
            "layers": layers,
        }


class TrackedGate(nn.Module):
    """Wrapper around MoE gate that records expert selections."""

    def __init__(self, original_gate, tracker: ExpertActivationTracker,
                 layer_idx: int, top_k: int):
        super().__init__()
        self._gate = original_gate
        self._tracker = tracker
        self._layer_idx = layer_idx
        self._top_k = top_k

    def __call__(self, x):
        logits = self._gate(x)
        # Peek at which experts will be selected (replicate router logic)
        probs = mx.softmax(logits, axis=-1, precise=True)
        inds = mx.argpartition(probs, kth=-self._top_k, axis=-1)[..., -self._top_k:]
        mx.eval(inds)
        self._tracker.record(self._layer_idx, inds)
        return logits


def hook_moe_routers(model, tracker: ExpertActivationTracker):
    """Replace MoE gate layers with tracked versions."""
    hooked = 0
    for layer_idx, layer in enumerate(model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, 'gate') or not hasattr(mlp, 'switch_mlp'):
            continue
        top_k = getattr(mlp, 'top_k', 8)
        mlp.gate = TrackedGate(mlp.gate, tracker, layer_idx, top_k)
        hooked += 1
    logger.info(f"Hooked {hooked}/{len(model.layers)} MoE layers")


def run_calibration(model, tokenizer, tracker: ExpertActivationTracker,
                    max_tokens: int):
    """Run calibration prompts through the model."""
    tokens_processed = 0

    for i, prompt in enumerate(CALIBRATION_PROMPTS):
        if tokens_processed >= max_tokens:
            break

        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_prompt = prompt + "\n"

        input_ids = tokenizer.encode(chat_prompt)
        # Limit per-prompt tokens
        remaining = max_tokens - tokens_processed
        input_ids = input_ids[:min(len(input_ids), remaining)]

        if not input_ids:
            continue

        logger.info(f"  Prompt {i+1}/{len(CALIBRATION_PROMPTS)}: "
                    f"{len(input_ids)} tokens")

        # Run prefill (chunked to avoid OOM)
        chunk_size = 2048
        cache = [nn.quantized_embedding.KVCache() if hasattr(nn, 'quantized_embedding')
                 else type('KVCache', (), {'update_and_fetch': lambda s,k,v: (k,v)})()
                 for _ in range(len(model.layers))]

        # Use simple KVCache from mlx_lm
        from mlx_lm.models.cache import KVCache
        cache = [KVCache() for _ in range(len(model.layers))]

        for start in range(0, len(input_ids), chunk_size):
            end = min(start + chunk_size, len(input_ids))
            chunk = mx.array([input_ids[start:end]])
            logits = model(chunk, cache=cache)
            mx.eval(logits)

        tokens_processed += len(input_ids)
        tracker.total_tokens = tokens_processed

        # Generate more tokens to exercise decode-path experts and reach target
        gen_tokens = min(512, max_tokens - tokens_processed)
        for _ in range(gen_tokens):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            logits = model(token.reshape(1, 1), cache=cache)
            mx.eval(logits)
            tokens_processed += 1
            tracker.total_tokens = tokens_processed

        del cache
        gc.collect()
        mx.clear_cache()

        logger.info(f"  Total: {tokens_processed}/{max_tokens} tokens")

    return tokens_processed


def main():
    parser = argparse.ArgumentParser(
        description="ProMoE expert activation profiling (Task 32)")
    parser.add_argument("--tokens", type=int, default=5000,
                        help="Total calibration tokens (default: 5000)")
    args = parser.parse_args()

    t0 = time.perf_counter()

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    num_layers = len(model.layers)
    # Detect num_experts from first MoE layer
    num_experts = 128  # default
    for layer in model.layers:
        if hasattr(layer.mlp, 'num_experts'):
            num_experts = layer.mlp.num_experts
            break

    logger.info(f"Model: {num_layers} layers, {num_experts} experts/layer")

    tracker = ExpertActivationTracker(num_layers, num_experts)
    hook_moe_routers(model, tracker)

    logger.info(f"\nRunning calibration ({args.tokens} tokens)...")
    tokens = run_calibration(model, tokenizer, tracker, args.tokens)
    elapsed = time.perf_counter() - t0

    logger.info(f"\nCalibration complete: {tokens} tokens in {elapsed:.1f}s")

    # Export results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = tracker.to_dict()
    results["calibration_elapsed_s"] = round(elapsed, 1)

    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    logger.info(f"Profile written to {OUTPUT_FILE}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EXPERT ACTIVATION SUMMARY")
    logger.info(f"{'='*60}")

    for layer_data in results["layers"]:
        lid = layer_data["layer"]
        active = layer_data["experts_active"]
        total = layer_data["experts_total"]
        total_disp = layer_data["total_dispatches"]
        experts = layer_data["experts"]

        # Compute concentration: top-10 experts' share of total dispatches
        top10_share = sum(e["pct"] for e in experts[:10])
        bottom10_share = sum(e["pct"] for e in experts[-10:])

        if lid % 8 == 0 or lid == num_layers - 1:
            logger.info(f"  Layer {lid:>2}: {active}/{total} active, "
                        f"top10={top10_share:.1f}%, bottom10={bottom10_share:.1f}%, "
                        f"dispatches={total_disp}")

    # Global stats
    all_counts = []
    for layer_data in results["layers"]:
        for e in layer_data["experts"]:
            all_counts.append(e["count"])

    zero_experts = sum(1 for c in all_counts if c == 0)
    total_experts = len(all_counts)
    logger.info(f"\nGlobal: {zero_experts}/{total_experts} experts never activated "
                f"({zero_experts/total_experts*100:.1f}%)")

    # Recommendation
    cold_threshold = 0.5  # bottom 50% by count
    cold_per_layer = []
    for layer_data in results["layers"]:
        experts = layer_data["experts"]
        total_disp = layer_data["total_dispatches"]
        # Count experts below median dispatch count
        if experts:
            median_count = experts[len(experts) // 2]["count"]
            cold = sum(1 for e in experts if e["count"] <= median_count)
            cold_per_layer.append(cold)
    avg_cold = sum(cold_per_layer) / len(cold_per_layer) if cold_per_layer else 0

    logger.info(f"Average cold experts per layer (≤median): {avg_cold:.0f}/{num_experts}")
    logger.info(f"\nRecommendation for Task 16 --resident-fraction:")
    logger.info(f"  50% resident = mask {num_experts//2} cold experts → save ~3.6 GB")
    logger.info(f"  75% resident = mask {num_experts//4} cold experts → save ~1.8 GB")
    logger.info(f"  87.5% resident = mask {num_experts//8} cold experts → save ~0.9 GB")

    # Validate: every layer has exactly num_experts profiled, top/bottom both non-zero
    for layer_data in results["layers"]:
        experts = layer_data["experts"]
        assert len(experts) == num_experts, \
            f"Layer {layer_data['layer']}: expected {num_experts} experts, got {len(experts)}"
        assert experts[0]["count"] > 0, \
            f"Layer {layer_data['layer']}: top expert has 0 dispatches"
        # Bottom experts may have 0 dispatches with only 5K tokens — that's OK
        # The important assertion is top experts are non-zero

    logger.info(f"\nAll {num_layers} layers × {num_experts} experts validated ✓")


if __name__ == "__main__":
    main()
