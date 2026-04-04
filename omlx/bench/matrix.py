# SPDX-License-Identifier: Apache-2.0
"""Feature matrix runner — A/B test hypercar feature combinations.

Runs safe_bench.py as a subprocess for each flag combination, collects
profile data, scores results, outputs a leaderboard.

Running combinations in subprocesses ensures:
  - Clean Metal state per run
  - No cross-contamination of patches
  - Crash in one combo doesn't kill the matrix
  - Each run gets its own process memory profile

Usage:
    # Run predefined matrix
    python -m omlx.bench.matrix --preset core --context 8192

    # Run specific combinations
    python -m omlx.bench.matrix --preset decoder --context 16384

    # Custom context lengths
    python -m omlx.bench.matrix --preset core --contexts 4096 8192 16384
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .scoring import RunMetrics, describe_run, score_run

logger = logging.getLogger("matrix")


# ---------------------------------------------------------------------------
# Predefined feature matrices
# ---------------------------------------------------------------------------

# Each entry is a dict of CLI flags to set (True = set flag, int/str = value)
PRESETS: Dict[str, List[Dict[str, object]]] = {
    # Core memory features — isolate impact of each
    "core": [
        {"name": "baseline-all-off",
         "no_fp16_layer0": True, "no_vertical_eval": True, "no_adaptive_budget": True},
        {"name": "vertical-only",
         "no_fp16_layer0": True, "no_adaptive_budget": True},
        {"name": "fp16-only",
         "no_vertical_eval": True, "no_adaptive_budget": True},
        {"name": "adaptive-only",
         "no_fp16_layer0": True, "no_vertical_eval": True},
        {"name": "fp16+vertical",
         "no_adaptive_budget": True},
        {"name": "fp16+adaptive",
         "no_vertical_eval": True},
        {"name": "vertical+adaptive",
         "no_fp16_layer0": True},
        {"name": "all-memory-on"},  # all defaults
    ],

    # Chunk size sweep
    "chunks": [
        {"name": "chunk-1024", "dequant_chunk": 1024},
        {"name": "chunk-2048", "dequant_chunk": 2048},
        {"name": "chunk-4096", "dequant_chunk": 4096},
        {"name": "chunk-8192", "dequant_chunk": 8192},
    ],

    # Decoder features
    "decoder": [
        {"name": "baseline"},
        {"name": "medusa-random", "use_medusa": True, "medusa_distill": 0},
        {"name": "medusa-distilled", "use_medusa": True, "medusa_distill": 200},
        {"name": "prompt-lookup", "use_prompt_lookup": True},
        {"name": "medusa+lookup",
         "use_medusa": True, "medusa_distill": 200, "use_prompt_lookup": True},
    ],

    # Prefill chunk sweep
    "prefill": [
        {"name": "prefill-512",  "prefill_chunk": 512},
        {"name": "prefill-1024", "prefill_chunk": 1024},
        {"name": "prefill-2048", "prefill_chunk": 2048},
        {"name": "prefill-4096", "prefill_chunk": 4096},
        {"name": "prefill-8192", "prefill_chunk": 8192},
    ],

    # Quick smoke (for validating matrix runner itself)
    "smoke": [
        {"name": "default"},
    ],
}


def build_cli_args(combo: Dict[str, object], context: int, extra_args: List[str]) -> List[str]:
    """Translate a combo dict into argparse CLI args."""
    cli = [
        sys.executable, "-m", "omlx.bench.safe_bench",
        "--contexts", str(context),
        "--phases", "1",  # Smoke only for matrix (fast)
        "--no-coherence",  # Skip per-run coherence (we test once separately)
        "--max-metal-gb", "38",
        "--max-swap-gb", "8",
    ]
    cli.extend(extra_args)

    for key, value in combo.items():
        if key == "name":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
        else:
            cli.extend([flag, str(value)])

    return cli


def run_combo(combo: Dict[str, object], context: int, json_out: Path,
              extra_args: List[str], timeout: int = 3600) -> RunMetrics:
    """Run one combo as a subprocess. Returns RunMetrics."""
    name = combo.get("name", "unnamed")
    logger.info("=" * 80)
    logger.info(f"MATRIX RUN: {name} @ {context:,} tokens")

    # Features for logging (copy without 'name')
    features = {k: v for k, v in combo.items() if k != "name"}

    metrics = RunMetrics(run_id=name, features=features)

    cli = build_cli_args(combo, context, extra_args + ["--json", str(json_out)])
    logger.info("CMD: %s", " ".join(cli))

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cli, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.perf_counter() - t0

        # Log output snippets for debugging
        last_lines = "\n".join(result.stdout.splitlines()[-15:])
        logger.info("Last 15 lines of output:\n%s", last_lines)

        if result.returncode != 0:
            # Check if it was a speed/memory fail (benchmark exits nonzero on any fail)
            if json_out.exists():
                data = json.loads(json_out.read_text())
                if data and data[0].get("status"):
                    _populate_metrics(metrics, data[0])
                    return metrics
            metrics.status = "error"
            metrics.error_msg = (result.stderr or result.stdout).splitlines()[-1][:200] if result.stderr or result.stdout else "nonzero exit"
            return metrics

        # Parse JSON result
        if json_out.exists():
            data = json.loads(json_out.read_text())
            if data:
                _populate_metrics(metrics, data[0])
        else:
            metrics.status = "error"
            metrics.error_msg = "no JSON output"

    except subprocess.TimeoutExpired:
        metrics.status = "error"
        metrics.error_msg = f"timeout after {timeout}s"
    except Exception as e:
        metrics.status = "error"
        metrics.error_msg = f"{type(e).__name__}: {str(e)[:200]}"

    return metrics


def _populate_metrics(metrics: RunMetrics, data: dict):
    """Fill RunMetrics from safe_bench JSON output."""
    metrics.status = data.get("status", "unknown")
    metrics.prefill_toks = data.get("prefill_toks", 0.0)
    metrics.decode_toks = data.get("decode_toks", 0.0)
    metrics.metal_peak_gb = data.get("peak_gb", 0.0)
    metrics.metal_avg_gb = data.get("active_gb", 0.0)
    metrics.swap_peak_gb = data.get("swap_gb", 0.0)
    metrics.error_msg = data.get("error_msg", "")


def run_matrix(preset: str, contexts: List[int], extra_args: List[str],
               output_dir: Path) -> List[RunMetrics]:
    """Run a full matrix of combos × contexts."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Options: {list(PRESETS.keys())}")

    combos = PRESETS[preset]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[RunMetrics] = []
    total_runs = len(combos) * len(contexts)
    run_idx = 0

    for context in contexts:
        for combo in combos:
            run_idx += 1
            logger.info(f"\n### MATRIX PROGRESS: run {run_idx}/{total_runs} ###")

            json_path = output_dir / f"{combo['name']}_ctx{context}.json"
            metrics = run_combo(combo, context, json_path, extra_args)
            metrics.run_id = f"{combo['name']}_ctx{context}"
            metrics.score = score_run(metrics)
            all_metrics.append(metrics)

            logger.info(describe_run(metrics))

    return all_metrics


def print_leaderboard(metrics: List[RunMetrics]):
    """Print ranked leaderboard sorted by score."""
    sorted_runs = sorted(metrics, key=lambda m: -m.score)

    logger.info("\n" + "=" * 100)
    logger.info("LEADERBOARD (higher = better)")
    logger.info("=" * 100)
    logger.info(
        f"{'rank':>4}  {'score':>6}  {'run_id':>28}  {'status':>12}  "
        f"{'dec t/s':>8}  {'pre t/s':>8}  {'peak GB':>8}  {'swap GB':>8}"
    )
    logger.info("-" * 100)
    for i, m in enumerate(sorted_runs, 1):
        logger.info(
            f"{i:>4}  {m.score:>6.1f}  {m.run_id:>28}  {m.status:>12}  "
            f"{m.decode_toks:>8.1f}  {m.prefill_toks:>8.0f}  "
            f"{m.metal_peak_gb:>8.1f}  {m.swap_peak_gb:>8.1f}"
        )
    logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Hypercar feature matrix runner")
    parser.add_argument("--preset", default="core",
                        help=f"Matrix preset: {list(PRESETS.keys())}")
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192],
                        help="Context lengths to test")
    parser.add_argument("--output-dir", default="/tmp/omlx_matrix",
                        help="Where to write per-run JSON files")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout per run (seconds)")
    parser.add_argument("--min-decode", type=float, default=5.0,
                        help="Min decode tok/s for each subrun")
    parser.add_argument("--min-prefill", type=float, default=30.0,
                        help="Min prefill tok/s for each subrun")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    extra_args = [
        "--min-decode-toks", str(args.min_decode),
        "--min-prefill-toks", str(args.min_prefill),
    ]
    metrics = run_matrix(args.preset, args.contexts, extra_args, output_dir)

    # Save aggregated results
    results_path = output_dir / f"matrix_{args.preset}.json"
    with open(results_path, "w") as f:
        json.dump([asdict(m) for m in metrics], f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    print_leaderboard(metrics)


if __name__ == "__main__":
    main()
