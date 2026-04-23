# SPDX-License-Identifier: Apache-2.0
"""Tests for the observability scaffold."""

from __future__ import annotations

import json
import time

import pytest

from omlx.observability import counter, registry, reset, timer
from omlx.observability.heap import snapshot, snapshot_diff
from omlx.observability.leak_loop import leak_loop
from omlx.observability.registry import TimerStats


@pytest.fixture(autouse=True)
def _clean_registry():
    reset()
    yield
    reset()


def test_timer_records_duration():
    with timer("t.a"):
        time.sleep(0.01)
    with timer("t.a"):
        time.sleep(0.02)
    stats = registry.timer("t.a").summary()
    assert stats["count"] == 2
    assert stats["total_s"] >= 0.03
    assert stats["max_ms"] >= stats["p50_ms"] >= stats["min_ms"]


def test_timer_nested_regions_do_not_interfere():
    with timer("outer"):
        with timer("inner"):
            time.sleep(0.005)
        time.sleep(0.005)
    o = registry.timer("outer").summary()
    i = registry.timer("inner").summary()
    assert o["total_s"] >= i["total_s"] > 0


def test_counter_add_and_set():
    counter("c.evictions").add(3)
    counter("c.evictions").add(2)
    counter("c.current").set(42.0)
    assert registry.counter("c.evictions").value == 5
    assert registry.counter("c.current").value == 42.0


def test_timer_disabled_is_cheap_and_records_nothing():
    registry.enabled = False
    try:
        with timer("disabled"):
            pass
        counter("disabled").add(100)
    finally:
        registry.enabled = True
    snap = registry.snapshot()
    assert not any(t["name"] == "disabled" for t in snap["timers"])
    assert not any(c["name"] == "disabled" for c in snap["counters"])


def test_percentile_on_distribution():
    stats = TimerStats(name="p")
    # 100 samples of known distribution: 0.0..0.99 seconds
    for i in range(100):
        stats.record(i / 100.0)
    assert stats.percentile(0.0) == pytest.approx(0.0, abs=0.01)
    assert stats.percentile(0.5) == pytest.approx(0.5, abs=0.02)
    assert stats.percentile(0.95) == pytest.approx(0.95, abs=0.02)


def test_reservoir_sampling_bounds_memory():
    stats = TimerStats(name="p", sample_cap=64)
    for i in range(10_000):
        stats.record(i / 10_000)
    assert len(stats.samples) == 64
    assert stats.count == 10_000


def test_registry_dump_writes_json(tmp_path):
    with timer("a"):
        pass
    counter("b").add(5)
    path = registry.dump(tmp_path / "out.json")
    data = json.loads(path.read_text())
    assert "timers" in data
    assert "counters" in data
    assert data["elapsed_s"] >= 0


def test_snapshot_captures_metal_and_rss():
    snap = snapshot(label="test")
    assert snap.label == "test"
    assert snap.rss_gb > 0
    # Metal keys may or may not be present depending on MLX build
    for k, v in snap.metal.items():
        assert v >= 0


def test_snapshot_diff_structure():
    before = snapshot(label="b")
    after = snapshot(label="a")
    diff = snapshot_diff(before, after)
    d = diff.to_dict()
    assert "delta_metal_active_gb" in d
    assert "delta_phys_footprint_gb" in d
    text = diff.format()
    assert "heap diff" in text


def test_leak_loop_clean_on_noop_body(tmp_path):
    """A no-op body must not be flagged as leaking."""
    csv_path = tmp_path / "leak.csv"
    result = leak_loop(
        body=lambda i: None,
        iterations=20,
        warmup=5,
        csv_path=str(csv_path),
        log_every=0,
    )
    assert csv_path.exists()
    # No body = no allocations; slope must be ~0
    assert abs(result.slope_metal_peak_gb_per_iter()) < 0.01
    assert result.verdict(threshold_mb_per_iter=1.0).startswith("CLEAN")


def test_leak_loop_body_receives_iteration_index():
    seen = []
    leak_loop(body=lambda i: seen.append(i), iterations=5, warmup=1, log_every=0)
    assert seen == [0, 1, 2, 3, 4]
