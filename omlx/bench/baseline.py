# SPDX-License-Identifier: Apache-2.0
"""System baseline capture for benchmark environment diagnostics.

Gathers system state via native macOS APIs (ctypes + psutil) — no
subprocess calls, so it works under the Claude Code sandbox that
blocks sysctl/iostat/pgrep.

Usage (run as script to avoid omlx root package MLX import):
    .venv/bin/python omlx/bench/baseline.py           # JSON to stdout
    .venv/bin/python omlx/bench/baseline.py --pretty   # human-readable
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import sys
import time


def _find_libc():
    try:
        return ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libc.dylib")
    except Exception:
        return None


_LIBC = _find_libc()


def _sysctlbyname_uint64(name: bytes) -> int | None:
    """Read a uint64 sysctl value via ctypes (no subprocess)."""
    if _LIBC is None:
        return None
    try:
        buf = ctypes.c_uint64(0)
        size = ctypes.c_size_t(8)
        ret = _LIBC.sysctlbyname(name, ctypes.byref(buf), ctypes.byref(size), None, 0)
        return buf.value if ret == 0 else None
    except Exception:
        return None


def get_system_memory_gb() -> float:
    """Total system memory in GB via hw.memsize."""
    val = _sysctlbyname_uint64(b"hw.memsize")
    if val is not None:
        return val / 1e9
    # Fallback: psutil
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def get_swap_used_gb() -> float:
    """Swap used in GB via vm.swapusage or psutil fallback."""
    if _LIBC is not None:
        try:
            # struct xsw_usage { uint64_t total, avail, used; }
            buf = (ctypes.c_uint64 * 3)()
            size = ctypes.c_size_t(ctypes.sizeof(buf))
            ret = _LIBC.sysctlbyname(
                b"vm.swapusage", ctypes.byref(buf), ctypes.byref(size), None, 0,
            )
            if ret == 0:
                return buf[2] / 1e9
        except Exception:
            pass
    try:
        import psutil
        return psutil.swap_memory().used / 1e9
    except Exception:
        return 0.0


def get_load_avg() -> tuple[float, float, float]:
    """1/5/15 minute load averages via os.getloadavg()."""
    try:
        return os.getloadavg()
    except Exception:
        return (0.0, 0.0, 0.0)


def get_mlx_processes() -> list[dict]:
    """Find running processes that look like MLX/omlx workloads."""
    results = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                info = proc.info
                cmdline = info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if any(kw in cmd_str for kw in ["omlx", "mlx", "hypercar"]):
                    results.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cmdline": cmd_str[:200],
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return results


def get_disk_io() -> dict:
    """Disk I/O counters via psutil."""
    try:
        import psutil
        counters = psutil.disk_io_counters()
        if counters:
            return {
                "read_bytes": counters.read_bytes,
                "write_bytes": counters.write_bytes,
                "read_count": counters.read_count,
                "write_count": counters.write_count,
            }
    except Exception:
        pass
    return {}


def get_virtual_memory() -> dict:
    """Virtual memory stats via psutil."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "used_gb": round(vm.used / 1e9, 2),
            "percent": vm.percent,
            "active_gb": round(vm.active / 1e9, 2),
            "inactive_gb": round(vm.inactive / 1e9, 2),
            "wired_gb": round(vm.wired / 1e9, 2),
        }
    except Exception:
        return {}


def capture_baseline() -> dict:
    """Capture full system baseline — all data via native APIs, no subprocess."""
    load1, load5, load15 = get_load_avg()

    baseline = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "system_memory_gb": round(get_system_memory_gb(), 2),
        "swap_used_gb": round(get_swap_used_gb(), 2),
        "load_avg_1m": round(load1, 2),
        "load_avg_5m": round(load5, 2),
        "load_avg_15m": round(load15, 2),
        "virtual_memory": get_virtual_memory(),
        "mlx_processes": get_mlx_processes(),
        "disk_io": get_disk_io(),
    }

    return baseline


def main():
    parser = argparse.ArgumentParser(
        description="System baseline capture (sandbox-safe, no subprocess)",
    )
    parser.add_argument("--pretty", action="store_true",
                        help="Human-readable output")
    args = parser.parse_args()

    baseline = capture_baseline()

    if args.pretty:
        print(f"System Memory:  {baseline['system_memory_gb']} GB")
        print(f"Swap Used:      {baseline['swap_used_gb']} GB")
        print(f"Load Average:   {baseline['load_avg_1m']} / {baseline['load_avg_5m']} / {baseline['load_avg_15m']}")
        vm = baseline.get("virtual_memory", {})
        if vm:
            print(f"VM Used:        {vm.get('used_gb', '?')} GB ({vm.get('percent', '?')}%)")
            print(f"VM Available:   {vm.get('available_gb', '?')} GB")
            print(f"VM Wired:       {vm.get('wired_gb', '?')} GB")
        procs = baseline.get("mlx_processes", [])
        print(f"MLX Processes:  {len(procs)}")
        for p in procs:
            print(f"  PID {p['pid']}: {p['cmdline'][:80]}")
    else:
        json.dump(baseline, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
