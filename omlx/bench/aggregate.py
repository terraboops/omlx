# SPDX-License-Identifier: Apache-2.0
"""Multi-run statistical aggregation for hypercar benchmarks.

Walks bench/snapshots/run*/ directories, groups by git SHA + KV mode,
and computes per-phase statistics (mean, std, median, percentiles).

Usage (run as script to avoid omlx root package MLX import):
    .venv/bin/python omlx/bench/aggregate.py                    # build aggregates
    .venv/bin/python omlx/bench/aggregate.py --report HEAD      # report for latest SHA
    .venv/bin/python omlx/bench/aggregate.py --report abc1234   # report for specific SHA
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


SNAPSHOTS_DIR = Path("bench/snapshots")
AGGREGATE_DIR = SNAPSHOTS_DIR / "aggregate"


# ---------------------------------------------------------------------------
# Statistics (stdlib only, no numpy)
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def _stats(vals: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not vals:
        return {"N": 0}
    return {
        "N": len(vals),
        "mean": round(_mean(vals), 2),
        "std": round(_std(vals), 2),
        "median": round(_median(vals), 2),
        "p10": round(_percentile(vals, 10), 2),
        "p50": round(_percentile(vals, 50), 2),
        "p90": round(_percentile(vals, 90), 2),
        "p99": round(_percentile(vals, 99), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

def load_snapshots(snapshots_dir: Path = SNAPSHOTS_DIR) -> list[dict]:
    """Load all results.json files from bench/snapshots/run*/."""
    results = []
    for run_dir in sorted(snapshots_dir.glob("run*")):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        try:
            data = json.loads(results_path.read_text())
            data["_run_dir"] = run_dir.name

            # Also load env.json for swap_peak_gb if available
            env_path = run_dir / "env.json"
            if env_path.exists():
                env = json.loads(env_path.read_text())
                data["_env"] = env

            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {run_dir.name}: {e}", file=sys.stderr)
    return results


def group_by_sha(snapshots: list[dict]) -> dict[str, list[dict]]:
    """Group snapshots by git commit SHA."""
    groups: dict[str, list[dict]] = {}
    for snap in snapshots:
        sha = snap.get("git_commit", "unknown")
        groups.setdefault(sha, []).append(snap)
    return groups


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_group(snapshots: list[dict]) -> dict:
    """Compute per-phase statistics for a group of runs."""
    if not snapshots:
        return {}

    sha = snapshots[0].get("git_commit", "unknown")
    N = len(snapshots)

    # Collect per-phase timing
    phase_timings: dict[str, list[float]] = {}
    phase_decode: dict[str, list[float]] = {}
    phase_prefill: dict[str, list[float]] = {}

    total_times: list[float] = []
    swap_peaks: list[float] = []
    metal_peaks: list[float] = []

    for snap in snapshots:
        total_times.append(snap.get("total_elapsed_s", 0.0))

        mem = snap.get("memory", {})
        metal_peaks.append(mem.get("metal_peak_gb", 0.0))
        swap_peaks.append(mem.get("swap_peak_gb", 0.0))

        # Also check env.json for swap_peak_gb (more accurate)
        env = snap.get("_env", {})
        if "swap_peak_gb" in env:
            # Use env swap if available (overrides)
            swap_peaks[-1] = env["swap_peak_gb"]

        for phase_name, phase_data in snap.get("phases", {}).items():
            if phase_name == "CRASH":
                continue
            elapsed = phase_data.get("elapsed_s", 0.0)
            phase_timings.setdefault(phase_name, []).append(elapsed)

            decode = phase_data.get("decode_toks", 0.0)
            if decode > 0:
                phase_decode.setdefault(phase_name, []).append(decode)

            prefill = phase_data.get("prefill_toks", 0.0)
            if prefill > 0:
                phase_prefill.setdefault(phase_name, []).append(prefill)

    result = {
        "git_commit": sha,
        "N": N,
        "runs": [s.get("_run_dir", "?") for s in snapshots],
        "total_elapsed": _stats(total_times),
        "metal_peak": _stats(metal_peaks),
        "swap_peak": _stats(swap_peaks),
        "phases": {},
    }

    for phase_name in sorted(phase_timings.keys()):
        phase_stats: dict[str, Any] = {
            "elapsed": _stats(phase_timings[phase_name]),
        }
        if phase_name in phase_decode:
            phase_stats["decode_toks"] = _stats(phase_decode[phase_name])
        if phase_name in phase_prefill:
            phase_stats["prefill_toks"] = _stats(phase_prefill[phase_name])
        result["phases"][phase_name] = phase_stats

    return result


# ---------------------------------------------------------------------------
# Report: compare two SHAs
# ---------------------------------------------------------------------------

def _resolve_sha(sha_spec: str, groups: dict[str, list[dict]]) -> str | None:
    """Resolve HEAD or partial SHA to full key in groups."""
    if sha_spec == "HEAD":
        # Use the SHA with the most recent run directory name
        latest = None
        for sha, snaps in groups.items():
            for s in snaps:
                run_dir = s.get("_run_dir", "")
                if latest is None or run_dir > latest[1]:
                    latest = (sha, run_dir)
        return latest[0] if latest else None

    # Partial match
    matches = [k for k in groups if k.startswith(sha_spec)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous SHA '{sha_spec}': {matches}", file=sys.stderr)
    return sha_spec if sha_spec in groups else None


def print_report(sha_spec: str, groups: dict[str, list[dict]]):
    """Print per-phase comparison report for a SHA."""
    sha = _resolve_sha(sha_spec, groups)
    if sha is None:
        print(f"SHA '{sha_spec}' not found in snapshots", file=sys.stderr)
        return

    agg = aggregate_group(groups[sha])
    N = agg["N"]

    print(f"\n{'=' * 72}")
    print(f"AGGREGATE REPORT: {sha} (N={N})")
    print(f"{'=' * 72}")
    print(f"Runs: {', '.join(agg['runs'])}")

    # Overall
    te = agg["total_elapsed"]
    print(f"\nTotal elapsed:  {te['median']:.1f}s median  "
          f"[{te['min']:.1f} - {te['max']:.1f}]  "
          f"std={te['std']:.1f}s  CV={te['std']/te['mean']*100:.0f}%" if te['mean'] > 0 else "")

    mp = agg["metal_peak"]
    print(f"Metal peak:     {mp['median']:.2f} GB median  [{mp['min']:.2f} - {mp['max']:.2f}]")

    sp = agg["swap_peak"]
    print(f"Swap peak:      {sp['median']:.2f} GB median  [{sp['min']:.2f} - {sp['max']:.2f}]")
    if sp["N"] > 0 and sp["max"] > 8.0:
        violate_count = sum(1 for s in groups[sha]
                           for v in [s.get("memory", {}).get("swap_peak_gb", 0)]
                           if v > 8.0)
        print(f"  ⚠ Goal 5 violations: {violate_count}/{N} runs exceed 8 GB")

    # Per-phase table
    print(f"\n{'Phase':<45} {'Median':>8} {'Mean':>8} {'Std':>6} {'Min':>8} {'Max':>8} {'N':>3}")
    print("-" * 86)

    for phase_name, phase_data in sorted(agg["phases"].items()):
        e = phase_data["elapsed"]
        if e["N"] == 0:
            continue
        cv = f"{e['std']/e['mean']*100:.0f}%" if e['mean'] > 0 else "—"
        print(f"  {phase_name:<43} {e['median']:>7.1f}s {e['mean']:>7.1f}s "
              f"{e['std']:>5.1f}s {e['min']:>7.1f}s {e['max']:>7.1f}s {e['N']:>3}")

        if "decode_toks" in phase_data:
            d = phase_data["decode_toks"]
            print(f"    decode tok/s:{'':>27} {d['median']:>7.1f}  "
                  f"[{d['min']:.1f} - {d['max']:.1f}]")
        if "prefill_toks" in phase_data:
            p = phase_data["prefill_toks"]
            print(f"    prefill tok/s:{'':>26} {p['median']:>7.1f}  "
                  f"[{p['min']:.1f} - {p['max']:.1f}]")

    print(f"\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-run statistical aggregation for hypercar benchmarks",
    )
    parser.add_argument("--report", type=str, default=None,
                        help="Print report for SHA (or HEAD)")
    parser.add_argument("--snapshots-dir", type=str, default=str(SNAPSHOTS_DIR),
                        help="Snapshots directory")
    args = parser.parse_args()

    snapshots_dir = Path(args.snapshots_dir)
    snapshots = load_snapshots(snapshots_dir)

    if not snapshots:
        print(f"No snapshots found in {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    groups = group_by_sha(snapshots)
    print(f"Loaded {len(snapshots)} runs across {len(groups)} SHA(s)")

    if args.report:
        print_report(args.report, groups)
        return

    # Build aggregate files
    agg_dir = snapshots_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    for sha, snaps in groups.items():
        agg = aggregate_group(snaps)
        out_path = agg_dir / f"{sha}.json"
        out_path.write_text(json.dumps(agg, indent=2))
        print(f"  {sha}: N={len(snaps)} → {out_path}")

    print(f"Wrote {len(groups)} aggregate file(s) to {agg_dir}")


if __name__ == "__main__":
    main()
