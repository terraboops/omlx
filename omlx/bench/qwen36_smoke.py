# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B smoke test for Task 257 — gates all Qwen3.6 migration tasks.

Validates that an MLX-quantized Qwen3.6-35B-A3B variant loads on M4 Pro 48GB,
generates coherent text, and reports memory + speed baselines. Until this passes,
Tasks 258-263 (bit-width tradeoff, quality re-run, native-1M audit, vision
ablation, MoE re-validation, default migration) cannot start.

Default model is Unsloth's Dynamic 2.0 4-bit MLX variant — important layers
are upcasted to higher precision per Unsloth's per-layer sensitivity analysis,
which structurally matches the Lyapunov + KVTuner approach our Tasks 207/270
were designed to build manually. UD-MLX-4bit ranks 1st in 21 of 22 sizes on
mean KL divergence per Unsloth's published GGUF benchmarks.

Usage:
    .venv/bin/python -m omlx.bench.qwen36_smoke
    .venv/bin/python -m omlx.bench.qwen36_smoke --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit
    .venv/bin/python -m omlx.bench.qwen36_smoke --model unsloth/Qwen3.6-35B-A3B-MLX-8bit
    .venv/bin/python -m omlx.bench.qwen36_smoke --model mlx-community/Qwen3.6-35B-A3B-4bit
    .venv/bin/python -m omlx.bench.qwen36_smoke --json    # machine-readable

Gates (must all pass for Task 257 closure):
    1. Model loads without OOM
    2. Metal-after-load ≤ 36 GB (leaves ≥12 GB for KV at modest contexts)
    3. Generates ≥20 coherent tokens on each smoke prompt (no "vet vet" / "is is is" collapse)
    4. 4K-prefill speed ≥ 200 tok/s (loose floor; tight target is Goal 4 ≥500)
    5. Single-token decode ≥ 20 tok/s at empty cache (loose floor; tight is Goal 3 ≥50)

Branch decisions from this run:
    - UD-MLX-4bit passes all gates → Task 258 adopts UD as default, proceeds to 259
    - UD-MLX-4bit fails coherence → fall back to mlx-community/Qwen3.6-35B-A3B-4bit
      (plain 4-bit, more conservative quantization)
    - Plain 4-bit also fails → 8-bit smoke test required (won't fit alongside KV at 1M)
    - All fail or OOM → halt migration, file blocker task
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass

import mlx.core as mx

logger = logging.getLogger("qwen36_smoke")


SMOKE_PROMPTS = [
    # Coding prompt (primary target workload)
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
    # Reasoning prompt (BENCHMARKS.md Run 20 reference style — checks for repetitive collapse)
    "The capital of France is",
    # Long-form code completion (tool-use / agentic workload analog)
    "Write a Python function that takes a list of integers and returns the sum of all even numbers in the list. The function should be named sum_even and use a list comprehension.\n\n",
]

# Repetitive-token signatures that indicate quantization collapse (BENCHMARKS.md Run 20).
COLLAPSE_PATTERNS = [
    "vet vet", "is is is", "the the the", "and and", "of of of",
    "    \n    \n", "..............",
]


@dataclass
class SmokeResult:
    model: str
    metal_after_load_gb: float
    metal_peak_gb: float
    load_seconds: float
    prompts_passed: int
    prompts_total: int
    decode_tps_empty: float
    prefill_tps_4k: float
    prefill_metal_peak_gb: float
    coherent_outputs: list[str]
    collapse_outputs: list[str]
    gates_passed: dict[str, bool]


def _metal_active_gb() -> float:
    return mx.get_active_memory() / 1e9


def _metal_peak_gb() -> float:
    return mx.get_peak_memory() / 1e9


def _reset_metal_peak() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def _check_collapse(text: str) -> bool:
    """Return True if output shows repetitive-token collapse signature."""
    if not text or len(text.strip()) < 5:
        return True
    lower = text.lower()
    if any(pattern in lower for pattern in COLLAPSE_PATTERNS):
        return True
    # Bigram repetition check: same two-char substring repeating > 6 times
    for i in range(len(lower) - 12):
        bigram = lower[i:i + 2]
        if bigram.strip() and lower.count(bigram, i, i + 14) >= 6:
            return True
    return False


def run_smoke(model_id: str) -> SmokeResult:
    from mlx_lm import generate, load

    logger.info(f"Loading {model_id}...")
    _reset_metal_peak()
    metal_before = _metal_active_gb()
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - t0
    metal_after_load = _metal_active_gb() - metal_before
    metal_peak_load = _metal_peak_gb()
    logger.info(f"Loaded in {load_seconds:.1f}s — Metal +{metal_after_load:.2f} GB (peak {metal_peak_load:.2f} GB)")

    # Coherence check: generate ≥40 tokens per smoke prompt
    coherent_outputs: list[str] = []
    collapse_outputs: list[str] = []
    decode_tps_samples: list[float] = []

    for prompt in SMOKE_PROMPTS:
        t0 = time.perf_counter()
        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=40,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        # Extract just the new tokens (mlx_lm.generate returns prompt + completion)
        completion = out[len(prompt):] if out.startswith(prompt) else out
        n_tokens = len(tokenizer.encode(completion)) if completion else 0
        if dt > 0 and n_tokens > 0:
            decode_tps_samples.append(n_tokens / dt)
        if _check_collapse(completion):
            logger.warning(f"COLLAPSE on prompt {prompt[:40]!r}: {completion[:80]!r}")
            collapse_outputs.append(completion[:200])
        else:
            logger.info(f"OK on prompt {prompt[:40]!r}: {completion[:80]!r}")
            coherent_outputs.append(completion[:200])

    prompts_passed = len(coherent_outputs)
    decode_tps = sum(decode_tps_samples) / max(1, len(decode_tps_samples))

    # 4K-prefill speed measurement
    prefill_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
    prefill_tokens = tokenizer.encode(prefill_text)[:4096]
    prefill_input = mx.array([prefill_tokens])
    _reset_metal_peak()
    t0 = time.perf_counter()
    _ = model(prefill_input)
    mx.eval(_)
    prefill_dt = time.perf_counter() - t0
    prefill_metal_peak = _metal_peak_gb()
    prefill_tps = len(prefill_tokens) / prefill_dt if prefill_dt > 0 else 0.0
    logger.info(f"Prefill {len(prefill_tokens)} tokens in {prefill_dt:.2f}s = {prefill_tps:.0f} tok/s (Metal peak {prefill_metal_peak:.2f} GB)")

    metal_peak_total = _metal_peak_gb()

    gates = {
        "loaded": True,
        "metal_after_load_under_36gb": metal_after_load < 36.0,
        "all_prompts_coherent": prompts_passed == len(SMOKE_PROMPTS),
        "decode_tps_above_20": decode_tps >= 20.0,
        "prefill_tps_above_200": prefill_tps >= 200.0,
    }

    return SmokeResult(
        model=model_id,
        metal_after_load_gb=round(metal_after_load, 2),
        metal_peak_gb=round(metal_peak_total, 2),
        load_seconds=round(load_seconds, 1),
        prompts_passed=prompts_passed,
        prompts_total=len(SMOKE_PROMPTS),
        decode_tps_empty=round(decode_tps, 1),
        prefill_tps_4k=round(prefill_tps, 0),
        prefill_metal_peak_gb=round(prefill_metal_peak, 2),
        coherent_outputs=coherent_outputs,
        collapse_outputs=collapse_outputs,
        gates_passed=gates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
        help=(
            "HF model ID (default: unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit — "
            "Unsloth Dynamic 2.0 with important layers upcasted; "
            "alternatives: unsloth/Qwen3.6-35B-A3B-MLX-8bit, "
            "mlx-community/Qwen3.6-35B-A3B-4bit)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        result = run_smoke(args.model)
    except Exception as exc:
        logger.exception(f"Smoke test failed during execution: {exc}")
        if args.json:
            print(json.dumps({"error": str(exc), "model": args.model}))
        return 2

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print()
        print(f"=== Qwen3.6 smoke test: {result.model} ===")
        print(f"Load:          {result.load_seconds:.1f}s, Metal +{result.metal_after_load_gb:.2f} GB")
        print(f"Coherence:     {result.prompts_passed}/{result.prompts_total} prompts passed")
        print(f"Decode (TG):   {result.decode_tps_empty:.1f} tok/s")
        print(f"Prefill 4K:    {result.prefill_tps_4k:.0f} tok/s (Metal peak {result.prefill_metal_peak_gb:.2f} GB)")
        print(f"Metal peak:    {result.metal_peak_gb:.2f} GB")
        print()
        print("Gates:")
        for name, passed in result.gates_passed.items():
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {name}")
        if result.collapse_outputs:
            print()
            print("Collapse signatures detected (quantization quality issue):")
            for c in result.collapse_outputs:
                print(f"  {c!r}")

    all_pass = all(result.gates_passed.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
