# SPDX-License-Identifier: Apache-2.0
"""Entry point for `python -m omlx.bench`.

Shows available bench commands and current system status.
"""

import sys


def main():
    print("oMLX Hypercar Benchmark Suite")
    print("=" * 50)
    print()
    print("Commands:")
    print("  python -m omlx.bench.hypercar_bench          # Default benchmark (duo mode)")
    print("  python -m omlx.bench.hypercar_bench --quick   # Smoke + coherence (~15s)")
    print("  python -m omlx.bench.hypercar_bench --full    # Full with HumanEval (~25min)")
    print()
    print("Analysis:")
    print("  python omlx/bench/aggregate.py                # Build aggregates")
    print("  python omlx/bench/aggregate.py --report HEAD  # Report for latest SHA")
    print("  python omlx/bench/baseline.py --pretty        # System health check")
    print()
    print("Probes (run as scripts to avoid MLX import):")
    print("  python omlx/bench/shadowkv_rank_probe.py      # SVD rank analysis")
    print("  python scripts/duoattention_calibrate.py      # Head classification")
    print("  python scripts/layerskip_calibrate.py         # Early-exit profiling")
    print("  python scripts/moe_lazyload_probe.py          # Expert masking test")
    print()
    print("Results: bench/snapshots/run*/results.json")
    print("Docs:    bench/snapshots/README.md")


if __name__ == "__main__":
    main()
