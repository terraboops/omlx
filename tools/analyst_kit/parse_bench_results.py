# SPDX-License-Identifier: Apache-2.0
"""Parse hypercar_bench --full output and copy artifacts to analyst dir.

Extracts decode tok/s by phase, prefill tok/s, gate verdicts, memory
peaks. Writes a compact summary used by post-bench tickets.

Usage:
    python -m tools.analyst_kit.parse_bench_results \\
        --results /tmp/hypercar_bench_results.json \\
        --profile /tmp/hypercar_profile.json \\
        --log research/analyst_runs/2026-04-26/baseline_full.log \\
        --output research/analyst_runs/2026-04-26/baseline_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def parse_log_for_decode_speeds(log_path: Path) -> dict:
    """Pull decode tok/s and prefill tok/s out of the bench log."""
    speeds = {}
    if not log_path.exists():
        return speeds
    text = log_path.read_text()

    # Match: "Prefill: 80 tok/s  Decode: 53.9 tok/s"
    m = re.search(r"Prefill:\s+([\d.]+)\s+tok/s\s+Decode:\s+([\d.]+)\s+tok/s", text)
    if m:
        speeds["smoke_prefill_tok_s"] = float(m.group(1))
        speeds["smoke_decode_tok_s"] = float(m.group(2))

    # Match per-context decode: "@4K decode: ... tok/s" patterns
    for ctx_pattern in [r"@(\d+)K", r"context (\d+)K"]:
        for m in re.finditer(
            ctx_pattern + r".*?(?:decode|tok/s)[^0-9]*?([\d.]+)\s+tok/s",
            text, re.IGNORECASE,
        ):
            ctx = m.group(1)
            tok_s = float(m.group(2))
            speeds[f"decode_{ctx}K_tok_s"] = tok_s

    return speeds


def parse_log_for_phase_times(log_path: Path) -> dict:
    """Extract per-phase wall-clock from the bench summary."""
    phases = {}
    if not log_path.exists():
        return phases
    text = log_path.read_text()
    # Match summary lines like "[PASS] Phase 0: Smoke (0.3s)"
    for m in re.finditer(
        r"\[(PASS|FAIL|SKIP)\]\s+Phase\s+([\w\d:]+)\s+([^\(]*)\(([\d.]+)s\)",
        text,
    ):
        verdict, phase_id, phase_name, secs = m.groups()
        phases[f"phase_{phase_id}_{phase_name.strip().replace(' ', '_')}"] = {
            "verdict": verdict,
            "seconds": float(secs),
        }
    return phases


def parse_log_for_evals(log_path: Path) -> dict:
    """Extract eval scores from the log."""
    evals = {}
    if not log_path.exists():
        return evals
    text = log_path.read_text()

    # HumanEval pass@1
    m = re.search(r"HumanEval[^0-9]*([\d.]+)/([\d.]+)\s*\(([\d.]+)%\)", text)
    if m:
        evals["humaneval"] = {
            "pass": int(float(m.group(1))),
            "total": int(float(m.group(2))),
            "pct": float(m.group(3)),
        }

    # MMLU-Pro
    m = re.search(r"MMLU-Pro:?\s*([\d.]+)/([\d.]+)\s*\(([\d.]+)%\)", text)
    if m:
        evals["mmlu_pro"] = {
            "correct": int(float(m.group(1))),
            "total": int(float(m.group(2))),
            "pct": float(m.group(3)),
        }

    # LiveCodeBench
    m = re.search(r"LiveCodeBench:?\s*([\d.]+)/([\d.]+)", text)
    if m:
        evals["livecodebench"] = {
            "pass": int(float(m.group(1))),
            "total": int(float(m.group(2))),
        }

    # Code intelligence
    m = re.search(r"Code intelligence:?\s*([\d.]+)/([\d.]+)", text)
    if m:
        evals["code_intel"] = {
            "pass": int(float(m.group(1))),
            "total": int(float(m.group(2))),
        }

    # NIAH
    for m in re.finditer(r"NIAH @ (\d+)K[^P]*?(PASS|FAIL)", text):
        evals[f"niah_{m.group(1)}K"] = m.group(2)

    return evals


def parse_log_for_memory(log_path: Path) -> dict:
    """Pull Metal/swap peaks from the log summary."""
    mem = {}
    if not log_path.exists():
        return mem
    text = log_path.read_text()
    m = re.search(r"Metal peak:\s+([\d.]+)\s+GB.*?\(limit\s+([\d.]+)GB\)", text)
    if m:
        mem["metal_peak_gb"] = float(m.group(1))
        mem["metal_limit_gb"] = float(m.group(2))
    m = re.search(r"Swap peak:\s+([\d.]+)\s+GB.*?\(limit\s+([\d.]+)GB\)", text)
    if m:
        mem["swap_peak_gb"] = float(m.group(1))
        mem["swap_limit_gb"] = float(m.group(2))
    return mem


def parse_log_for_timing(log_path: Path) -> dict:
    """Extract total bench time and start/end timestamps."""
    timing = {}
    if not log_path.exists():
        return timing
    text = log_path.read_text()
    m = re.search(r"\[ALL GATES PASSED\]\s+Total:\s+([\d.]+)s", text)
    if m:
        timing["total_seconds"] = float(m.group(1))
    return timing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="/tmp/hypercar_bench_results.json")
    ap.add_argument("--profile", default="/tmp/hypercar_profile.json")
    ap.add_argument("--log", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "decode_speeds": parse_log_for_decode_speeds(log_path),
        "phase_times": parse_log_for_phase_times(log_path),
        "evals": parse_log_for_evals(log_path),
        "memory": parse_log_for_memory(log_path),
        "timing": parse_log_for_timing(log_path),
    }

    # Copy artifacts into the analyst dir for permanence
    for src in (args.results, args.profile):
        src_path = Path(src)
        if src_path.exists():
            dst = out_dir / src_path.name
            shutil.copy2(src_path, dst)
            print(f"copied {src} -> {dst}")
            try:
                summary[src_path.stem] = json.loads(src_path.read_text())
            except Exception as e:
                summary[src_path.stem + "_load_error"] = str(e)
        else:
            print(f"missing: {src}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")

    # Pretty-print
    print("\n=== bench summary ===")
    if summary["timing"]:
        print(f"total: {summary['timing'].get('total_seconds', 'n/a')}s")
    print(f"\ndecode speeds:")
    for k, v in summary["decode_speeds"].items():
        print(f"  {k}: {v}")
    print(f"\nphase times:")
    for k, v in summary["phase_times"].items():
        print(f"  {k}: {v.get('seconds', 'n/a')}s [{v.get('verdict', '')}]")
    print(f"\nevals:")
    for k, v in summary["evals"].items():
        print(f"  {k}: {v}")
    print(f"\nmemory:")
    for k, v in summary["memory"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
