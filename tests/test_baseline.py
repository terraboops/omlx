# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/bench/baseline.py — system baseline capture.

Pure Python tests, no MLX required. Tests verify function signatures,
return types, and basic sanity checks against live system state.
"""

import importlib.util
import json
import sys

import pytest

# Load baseline module without triggering omlx package imports
_spec = importlib.util.spec_from_file_location(
    "omlx.bench.baseline", "omlx/bench/baseline.py")
baseline_mod = importlib.util.module_from_spec(_spec)
sys.modules["omlx.bench.baseline"] = baseline_mod
_spec.loader.exec_module(baseline_mod)


class TestSystemMemory:
    def test_returns_positive_float(self):
        mem = baseline_mod.get_system_memory_gb()
        assert isinstance(mem, float)
        assert mem > 0

    def test_reasonable_range(self):
        """System memory should be between 1 GB and 1 TB."""
        mem = baseline_mod.get_system_memory_gb()
        assert 1.0 < mem < 1024.0


class TestSwapUsed:
    def test_returns_float(self):
        swap = baseline_mod.get_swap_used_gb()
        assert isinstance(swap, float)

    def test_non_negative(self):
        swap = baseline_mod.get_swap_used_gb()
        assert swap >= 0.0


class TestLoadAverage:
    def test_returns_three_floats(self):
        load = baseline_mod.get_load_avg()
        assert len(load) == 3
        assert all(isinstance(x, float) for x in load)

    def test_non_negative(self):
        load = baseline_mod.get_load_avg()
        assert all(x >= 0 for x in load)


class TestMLXProcesses:
    def test_returns_list(self):
        procs = baseline_mod.get_mlx_processes()
        assert isinstance(procs, list)

    def test_process_has_expected_fields(self):
        procs = baseline_mod.get_mlx_processes()
        if procs:  # May be empty if no MLX processes running
            p = procs[0]
            assert "pid" in p
            assert "cmdline" in p


class TestDiskIO:
    def test_returns_dict(self):
        io = baseline_mod.get_disk_io()
        assert isinstance(io, dict)

    def test_has_read_write_fields(self):
        io = baseline_mod.get_disk_io()
        if io:  # May be empty on some systems
            assert "read_bytes" in io
            assert "write_bytes" in io


class TestVirtualMemory:
    def test_returns_dict(self):
        vm = baseline_mod.get_virtual_memory()
        assert isinstance(vm, dict)

    def test_has_key_fields(self):
        vm = baseline_mod.get_virtual_memory()
        assert "total_gb" in vm
        assert "available_gb" in vm
        assert "used_gb" in vm
        assert "percent" in vm

    def test_total_matches_system_memory(self):
        """Virtual memory total should match system memory."""
        vm = baseline_mod.get_virtual_memory()
        sys_mem = baseline_mod.get_system_memory_gb()
        # Allow 10% tolerance for rounding
        assert abs(vm["total_gb"] - sys_mem) / sys_mem < 0.1


class TestCaptureBaseline:
    def test_returns_dict(self):
        bl = baseline_mod.capture_baseline()
        assert isinstance(bl, dict)

    def test_has_required_fields(self):
        bl = baseline_mod.capture_baseline()
        required = [
            "timestamp", "platform", "system_memory_gb",
            "swap_used_gb", "load_avg_1m", "virtual_memory",
            "mlx_processes", "disk_io",
        ]
        for field in required:
            assert field in bl, f"Missing field: {field}"

    def test_json_serializable(self):
        bl = baseline_mod.capture_baseline()
        # Must be JSON-serializable (used by cron loop)
        serialized = json.dumps(bl)
        assert len(serialized) > 50

    def test_platform_info(self):
        bl = baseline_mod.capture_baseline()
        assert "system" in bl["platform"]
        assert "python" in bl["platform"]
        assert bl["platform"]["system"] == "Darwin"


class TestSysctlByName:
    def test_hw_memsize(self):
        val = baseline_mod._sysctlbyname_uint64(b"hw.memsize")
        assert val is not None
        assert val > 1e9  # > 1 GB

    def test_invalid_sysctl_returns_none(self):
        val = baseline_mod._sysctlbyname_uint64(b"nonexistent.sysctl.name")
        assert val is None
