# SPDX-License-Identifier: Apache-2.0
"""Lock-in tests for ``scripts/probe_amx_microbench.py`` — Task 387 Phase 1.

These tests exercise the probe's STRUCTURE without running the full
matmul bench (which is several seconds of compute). The actual perf
numbers live in the JSON output of a user-invokable run.

What's pinned:
- The shape catalog covers Qwen3.6's hot decode paths (kv_proj,
  q_proj, expert FFN gate/up/down, o_proj) — drift would silently
  miss the lever the task is meant to validate.
- ``summarize()`` correctly identifies ≥1.3× CPU wins per the Phase 2
  gate criterion.
- The output schema includes the fields downstream Phase 2 work needs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.probe_amx_microbench import (
    SHAPES,
    MatmulShape,
    summarize,
)


# ---------------------------------------------------------------------------
# Shape catalog
# ---------------------------------------------------------------------------


def test_shape_catalog_covers_qwen36_hot_paths():
    """Each Qwen3.6 decode-time hot path must be in the catalog.
    Drift would silently miss the lever Task 387 is meant to validate."""
    names = {s.name for s in SHAPES}
    required = {
        "qwen36_kv_proj_decode",
        "qwen36_q_proj_decode",
        "qwen36_expert_gate_decode",
        "qwen36_expert_down_decode",
        "qwen36_o_proj_decode",
    }
    missing = required - names
    assert not missing, f"shape catalog missing: {missing}"


def test_shape_catalog_includes_prefill_shapes():
    """Phase 2 dispatch decisions need both decode (L=1) and prefill
    (L=4096) numbers — they may have different crossover points."""
    decode_count = sum(1 for s in SHAPES if s.L == 1)
    prefill_count = sum(1 for s in SHAPES if s.L > 1)
    assert decode_count >= 5, "need ≥5 decode shapes for hot-path coverage"
    assert prefill_count >= 1, "need ≥1 prefill shape for chunk-size analysis"


def test_shape_catalog_has_distinct_names():
    """Duplicates would silently overwrite registry timer entries."""
    names = [s.name for s in SHAPES]
    assert len(names) == len(set(names)), (
        f"duplicate shape names: {[n for n in names if names.count(n) > 1]}"
    )


def test_shape_catalog_has_meaningful_dimensions():
    """Each shape must have plausible Qwen3.6-scale dimensions —
    catches accidentally-zero or negative entries."""
    for s in SHAPES:
        assert s.M > 0, f"{s.name}: non-positive M={s.M}"
        assert s.N > 0, f"{s.name}: non-positive N={s.N}"
        assert s.L > 0, f"{s.name}: non-positive L={s.L}"


# ---------------------------------------------------------------------------
# summarize() — Phase 2 gate criterion
# ---------------------------------------------------------------------------


def test_summarize_passes_gate_when_cpu_wins_at_1_3x():
    results = [
        {"name": "shape_a", "speedup_cpu_over_gpu": 0.8, "decision": "GPU/Metal faster"},
        {"name": "shape_b", "speedup_cpu_over_gpu": 1.5, "decision": "CPU/AMX FASTER (≥1.3×)"},
        {"name": "shape_c", "speedup_cpu_over_gpu": 1.0, "decision": "GPU/Metal faster"},
    ]
    s = summarize(results)
    assert s["n_shapes"] == 3
    assert s["n_cpu_wins_at_1_3x"] == 1
    assert s["cpu_winning_shapes"] == ["shape_b"]
    assert "PASS" in s["phase2_gate"]


def test_summarize_fails_gate_when_no_cpu_wins():
    results = [
        {"name": "shape_a", "speedup_cpu_over_gpu": 0.5},
        {"name": "shape_b", "speedup_cpu_over_gpu": 1.2},  # under 1.3 threshold
    ]
    s = summarize(results)
    assert s["n_cpu_wins_at_1_3x"] == 0
    assert s["cpu_winning_shapes"] == []
    assert "FAIL" in s["phase2_gate"]


def test_summarize_threshold_is_strict_1_3x():
    """1.299× must NOT pass; 1.3× exactly must pass. Pin so a future
    refactor doesn't silently move the gate."""
    just_under = [{"name": "x", "speedup_cpu_over_gpu": 1.299}]
    just_at = [{"name": "x", "speedup_cpu_over_gpu": 1.3}]
    assert summarize(just_under)["n_cpu_wins_at_1_3x"] == 0
    assert summarize(just_at)["n_cpu_wins_at_1_3x"] == 1


def test_summarize_handles_failed_bench_rows():
    """If bench_shape raised, the row has an 'error' key and no
    speedup. summarize() should not crash on these."""
    results = [
        {"name": "shape_a", "speedup_cpu_over_gpu": 1.5},
        {"name": "shape_b", "error": "MLX kernel crashed"},
    ]
    s = summarize(results)
    assert s["n_shapes"] == 2
    assert s["n_cpu_wins_at_1_3x"] == 1
    # The errored row is not in the winning list (no speedup field).
    assert "shape_b" not in s["cpu_winning_shapes"]


# ---------------------------------------------------------------------------
# MatmulShape dataclass invariants
# ---------------------------------------------------------------------------


def test_matmul_shape_is_frozen():
    """Shapes are dataclass(frozen=True) so they can be used as
    set/dict keys without surprise mutation."""
    s = MatmulShape("test", 64, 128, 1, "smoke")
    with pytest.raises(Exception):
        s.M = 999  # type: ignore[misc]
