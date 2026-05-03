#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 387 Phase 1 spike: AMX vs Metal matmul microbench.

The Hypercar architecture plan calls for heterogeneous compute pipelining
across the M4 Pro's six compute blocks (Metal GPU, ANE, AMX, P/E cores,
Memory Compression). Today the entire inference pipeline runs on Metal,
leaving the AMX matrix unit idle. This probe measures whether routing
specific matmuls to the CPU stream (which Apple Accelerate dispatches
through AMX where applicable) is faster than Metal at the shapes
representative of Qwen3.6's decode-time hot paths.

Decision criterion (Task 387 Phase 2 gate): AMX ≥1.3× faster than
Metal for at least one matmul shape in Qwen3.6's hot path. Pass →
proceed to Phase 2 (dispatch primitive). Fail → file a falsification
note.

Note on terminology: MLX exposes ``stream=mx.cpu`` and ``stream=mx.gpu``
to dispatch ops to specific devices. The CPU path goes through Apple's
Accelerate BLAS, which transparently uses AMX for matmuls where the
matrix sizes match its kernel templates. There's no explicit "AMX"
flag in user-mode MLX; the proxy is the CPU stream.

Usage:
    .venv/bin/python scripts/probe_amx_microbench.py
    .venv/bin/python scripts/probe_amx_microbench.py --output research/amx_microbench.json

Heavy by Hypercar standards (~10 s of compute per shape) — user-invoked,
not a cron task.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import mlx.core as mx

sys.path.insert(0, ".")
from omlx.observability import median_of_n

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("amx_probe")


# ---------------------------------------------------------------------------
# Shape catalog: Qwen3.6 hot paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatmulShape:
    """One matmul shape to bench. Per-token decode shape is ``(B, M) @ (M, N)``;
    prefill scales L tokens at once via ``(B, L, M) @ (M, N)``.

    Attributes
    ----------
    name : str
        Human-readable label (e.g., ``"qwen36_kv_proj"``).
    M : int
        Input feature dim (rows of B / cols of A, the contraction axis).
    N : int
        Output feature dim (cols of B / cols of result).
    L : int
        Token count. ``L=1`` for decode, ``L=4096`` for prefill chunk.
    note : str
        Where this shape comes from in the model.
    """

    name: str
    M: int
    N: int
    L: int
    note: str


# Qwen3.6-35B-A3B per-token decode shapes (B=1):
#   hidden_dim = 2048, head_dim = 256, n_heads = 16, n_kv_heads = 2
#   K/V proj:   2048 × (2 × 256) =   2048 × 512
#   Q proj:     2048 × (16 × 256) = 2048 × 4096
#   Expert FFN gate/up: 2048 × 4096 (per-expert, MoE picks 8 of 256)
#   Expert FFN down:    4096 × 2048
#   lm_head:    4096 × 152064 (vocab — typically GPU-only)
SHAPES: list[MatmulShape] = [
    # Decode (L=1) shapes — small per-step matmuls, AMX hypothesis is
    # that these are sub-Metal-launch-overhead and CPU wins.
    MatmulShape("qwen36_kv_proj_decode",   2048,  512, 1,
                "Qwen3.6 K/V proj at decode (per-token)"),
    MatmulShape("qwen36_q_proj_decode",    2048, 4096, 1,
                "Qwen3.6 Q proj at decode (per-token)"),
    MatmulShape("qwen36_expert_gate_decode", 2048, 4096, 1,
                "Qwen3.6 MoE expert gate/up at decode (per-token)"),
    MatmulShape("qwen36_expert_down_decode", 4096, 2048, 1,
                "Qwen3.6 MoE expert down at decode (per-token)"),
    MatmulShape("qwen36_o_proj_decode",    4096, 2048, 1,
                "Qwen3.6 attention O proj at decode (per-token)"),
    # Prefill chunk shapes — multiple tokens per matmul.
    MatmulShape("qwen36_kv_proj_prefill",   2048,  512, 4096,
                "Qwen3.6 K/V proj at prefill (4K-chunk)"),
    MatmulShape("qwen36_q_proj_prefill",    2048, 4096, 4096,
                "Qwen3.6 Q proj at prefill (4K-chunk)"),
]


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def bench_one_device(shape: MatmulShape, device, *, dtype=mx.float16,
                     warmup: int = 10, n: int = 30) -> float:
    """Median ms for a single (shape, device) cell. ``device`` is
    ``mx.cpu`` or ``mx.gpu`` (per MLX's stream API)."""
    A_shape = (shape.L, shape.M) if shape.L > 1 else (1, shape.M)
    B_shape = (shape.M, shape.N)
    A = mx.random.normal(A_shape).astype(dtype)
    B = mx.random.normal(B_shape).astype(dtype)
    mx.eval(A, B)   # ensure both materialized before timing

    def call():
        C = mx.matmul(A, B, stream=device)
        mx.eval(C)

    return median_of_n(
        call, name=f"amx_probe.{shape.name}.{device.name}",
        warmup=warmup, n=n,
    )


def bench_shape(shape: MatmulShape, *, dtype=mx.float16) -> dict:
    """Bench one shape on both devices, return a result dict."""
    flop = 2 * shape.L * shape.M * shape.N
    gpu_ms = bench_one_device(shape, mx.gpu, dtype=dtype)
    cpu_ms = bench_one_device(shape, mx.cpu, dtype=dtype)
    speedup_cpu_over_gpu = gpu_ms / max(cpu_ms, 1e-9)
    decision = (
        "CPU/AMX FASTER (≥1.3×)" if speedup_cpu_over_gpu >= 1.3
        else "GPU/Metal faster"
    )
    return {
        "name": shape.name,
        "note": shape.note,
        "M": shape.M,
        "N": shape.N,
        "L": shape.L,
        "flop": flop,
        "gpu_ms": gpu_ms,
        "cpu_ms": cpu_ms,
        "speedup_cpu_over_gpu": speedup_cpu_over_gpu,
        "decision": decision,
    }


def run_bench(dtype=mx.float16) -> list[dict]:
    results = []
    for shape in SHAPES:
        logger.info(f"Bench: {shape.name} (M={shape.M} N={shape.N} L={shape.L}) ...")
        try:
            row = bench_shape(shape, dtype=dtype)
        except Exception as e:
            logger.warning(f"  failed: {e}")
            row = {
                "name": shape.name,
                "note": shape.note,
                "error": str(e),
            }
        results.append(row)
        if "error" not in row:
            logger.info(
                f"  GPU {row['gpu_ms']:.3f} ms | "
                f"CPU {row['cpu_ms']:.3f} ms | "
                f"CPU/GPU {row['speedup_cpu_over_gpu']:.2f}x | "
                f"{row['decision']}"
            )
    return results


def summarize(results: list[dict]) -> dict:
    cpu_wins = [r for r in results if r.get("speedup_cpu_over_gpu", 0) >= 1.3]
    return {
        "n_shapes": len(results),
        "n_cpu_wins_at_1_3x": len(cpu_wins),
        "cpu_winning_shapes": [r["name"] for r in cpu_wins],
        "phase2_gate": (
            "PASS — proceed to Phase 2 (dispatch primitive)"
            if cpu_wins
            else "FAIL — file falsification note; AMX not faster than "
                 "Metal at any tested shape"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AMX vs Metal matmul microbench (Task 387 Phase 1)",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"],
                        default="float16",
                        help="Matmul dtype. fp16 matches Hypercar's runtime; "
                             "fp32 helps isolate AMX from Metal MPSGraph "
                             "fp16 fast paths.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to this path.")
    args = parser.parse_args()

    dtype_map = {"float16": mx.float16, "float32": mx.float32}
    dtype = dtype_map[args.dtype]
    logger.info(f"AMX microbench: dtype={args.dtype}")
    logger.info(f"Default device: {mx.default_device()}")

    t0 = time.perf_counter()
    results = run_bench(dtype=dtype)
    summary = summarize(results)
    elapsed = time.perf_counter() - t0

    output = {
        "schema_version": 1,
        "task": "387 Phase 1: AMX dispatch microbench",
        "dtype": args.dtype,
        "default_device": str(mx.default_device()),
        "elapsed_s": round(elapsed, 1),
        "shapes": results,
        "summary": summary,
    }

    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info(f"  shapes:                       {summary['n_shapes']}")
    logger.info(f"  CPU/AMX wins at ≥1.3×:        {summary['n_cpu_wins_at_1_3x']}")
    if summary["cpu_winning_shapes"]:
        logger.info(f"  winning shapes: {summary['cpu_winning_shapes']}")
    logger.info(f"  Phase 2 gate: {summary['phase2_gate']}")
    logger.info(f"  total wall time: {elapsed:.1f}s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        logger.info(f"  results written to {args.output}")


if __name__ == "__main__":
    main()
