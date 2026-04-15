# SPDX-License-Identifier: Apache-2.0
"""Calibrate speculative decoding cost-model constants (Task 56).

Measures per-token KV-load vs model-compute latency at various context
lengths to fit the SpecDecGate cost model. Produces calibrated constants
that replace the defaults.

Usage:
    python -m omlx.bench.specdec_calibrate [--contexts 2K,16K,64K]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"


def measure_decode_latency(model, tokenizer, context_len: int,
                            n_decode: int = 32) -> dict:
    """Measure per-token decode latency at a given context length."""
    n_layers = len(model.layers)

    # Build context
    prompt = "Explain " + "the " * (context_len // 2)
    tokens = tokenizer.encode(prompt)[:context_len]

    cache = [KVCache() for _ in range(n_layers)]

    # Prefill
    chunk_size = 2048
    for start in range(0, len(tokens), chunk_size):
        end = min(start + chunk_size, len(tokens))
        x = mx.array([tokens[start:end]])
        logits = model(x, cache=cache)
        mx.eval(logits)

    # Decode: measure per-token latency
    decode_times = []
    for i in range(n_decode):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)

        t0 = time.perf_counter()
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        decode_times.append(time.perf_counter() - t0)

    # Skip first 2 (warmup within context)
    stable_times = decode_times[2:]
    median_us = sorted(stable_times)[len(stable_times) // 2] * 1e6

    del cache
    gc.collect()
    mx.clear_cache()

    return {
        "context_len": context_len,
        "n_decode": n_decode,
        "median_us": round(median_us, 1),
        "min_us": round(min(stable_times) * 1e6, 1),
        "max_us": round(max(stable_times) * 1e6, 1),
        "tok_per_s": round(1e6 / median_us, 1) if median_us > 0 else 0,
    }


def fit_constants(measurements: list[dict]) -> dict:
    """Fit cost-model constants from measurements."""
    from omlx.specdec_gate import DEFAULT_CONSTANTS

    constants = DEFAULT_CONSTANTS.copy()

    if len(measurements) < 2:
        # Not enough data for regression, use single measurement
        if measurements:
            constants["compute_us"] = measurements[0]["median_us"]
        constants["calibrated"] = True
        constants["model"] = MODEL_ID
        return constants

    # Linear regression: latency = kv_base + kv_per_token * context_len + compute
    # Since compute dominates, we approximate:
    #   compute_us ≈ latency at shortest context
    #   kv_per_token_us ≈ (latency_long - latency_short) / (ctx_long - ctx_short)
    sorted_m = sorted(measurements, key=lambda m: m["context_len"])
    shortest = sorted_m[0]
    longest = sorted_m[-1]

    delta_lat = longest["median_us"] - shortest["median_us"]
    delta_ctx = longest["context_len"] - shortest["context_len"]

    kv_per_token = delta_lat / delta_ctx if delta_ctx > 0 else 0.005
    compute_us = shortest["median_us"] - kv_per_token * shortest["context_len"]

    # compute_us should be positive and reasonable
    if compute_us < 1000:
        compute_us = shortest["median_us"] * 0.95  # fallback: 95% is compute

    constants["compute_us"] = round(compute_us, 1)
    constants["kv_per_token_us"] = round(max(kv_per_token, 0.001), 6)
    constants["kv_base_us"] = round(shortest["median_us"] - compute_us, 1)
    constants["calibrated"] = True
    constants["model"] = MODEL_ID

    return constants


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate speculative decoding cost model (Task 56)")
    parser.add_argument("--contexts", type=str, default="2048,8192,32768",
                        help="Comma-separated context lengths (default: 2048,8192,32768)")
    parser.add_argument("--decode-tokens", type=int, default=32,
                        help="Decode tokens per measurement (default: 32)")
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",")]

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    # Apply prefill patch
    try:
        from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
        apply_prefill_last_logit_patch(model)
    except ImportError:
        pass

    # Warmup
    logger.info("Warming up...")
    cache = [KVCache() for _ in range(len(model.layers))]
    logits = model(mx.array([[1, 2, 3, 4]]), cache=cache)
    mx.eval(logits)
    del cache, logits
    gc.collect()
    mx.clear_cache()

    # Measure at each context length
    measurements = []
    for ctx in contexts:
        logger.info(f"\nMeasuring at {ctx} tokens ({ctx//1024}K)...")
        m = measure_decode_latency(model, tokenizer, ctx, args.decode_tokens)
        measurements.append(m)
        logger.info(f"  Median: {m['median_us']:.0f}µs/tok ({m['tok_per_s']:.1f} tok/s)")
        logger.info(f"  Range:  [{m['min_us']:.0f} - {m['max_us']:.0f}]µs")

    # Fit constants
    constants = fit_constants(measurements)

    # Save
    from omlx.specdec_gate import save_constants, SpecDecGate
    save_constants(constants)

    # Print gate decisions at various context lengths
    gate = SpecDecGate(constants)
    logger.info(f"\n{'='*60}")
    logger.info("CALIBRATED GATE DECISIONS")
    logger.info(f"{'='*60}")
    logger.info(f"  compute_us: {constants['compute_us']:.0f}")
    logger.info(f"  kv_per_token_us: {constants['kv_per_token_us']:.6f}")
    logger.info(f"  kv_base_us: {constants['kv_base_us']:.0f}")

    test_contexts = [2048, 4096, 16384, 65536, 262144]
    logger.info(f"\n{'Context':>10} {'Speculate':>10} {'Speedup':>10} {'Reason'}")
    logger.info("-" * 70)
    for ctx in test_contexts:
        should, dec = gate.should_speculate(ctx)
        logger.info(f"  {ctx:>8} {'YES' if should else 'NO':>10} "
                    f"{dec.projected_speedup:>8.2f}x  {dec.reason}")


if __name__ == "__main__":
    main()
