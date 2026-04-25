# SPDX-License-Identifier: Apache-2.0
"""Tests for the tracer (monkey-patch instrumentation) and claims harness."""

from __future__ import annotations

import time

import pytest

from omlx.observability import registry, reset
from omlx.observability.claims import Claim, format_summary, run_claim, verify_all
from omlx.observability.tracer import trace_class, trace_module, untrace


@pytest.fixture(autouse=True)
def _clean_registry():
    reset()
    yield
    reset()


# --- Tracer ----------------------------------------------------------------


class _Cache:
    """Stand-in for an instrumented class."""

    def __init__(self) -> None:
        self.state = []

    def add(self, x: int) -> int:
        time.sleep(0.001)
        self.state.append(x)
        return len(self.state)

    def reset(self) -> None:
        self.state = []

    @classmethod
    def make(cls) -> "_Cache":
        return cls()

    @staticmethod
    def helper(x: int) -> int:
        return x * 2

    @property
    def size(self) -> int:
        return len(self.state)


def test_trace_class_records_method_calls():
    handle = trace_class(_Cache, prefix="cache", mlx=False)
    try:
        c = _Cache()
        c.add(1)
        c.add(2)
        c.reset()
        c.add(3)
    finally:
        untrace(handle)

    # add called 3 times, reset once.
    assert registry.timer("cache.add").count == 3
    assert registry.timer("cache.reset").count == 1
    assert registry.counter("cache.add.calls").value == 3
    assert registry.counter("cache.reset.calls").value == 1


def test_trace_class_handles_classmethods_and_staticmethods():
    handle = trace_class(_Cache, prefix="cache", mlx=False)
    try:
        _Cache.make()
        _Cache.helper(5)
    finally:
        untrace(handle)
    assert registry.timer("cache.make").count == 1
    assert registry.timer("cache.helper").count == 1


def test_trace_class_handles_properties():
    handle = trace_class(_Cache, prefix="cache", mlx=False)
    try:
        c = _Cache()
        c.add(1)
        c.add(2)
        _ = c.size
        _ = c.size
    finally:
        untrace(handle)
    assert registry.timer("cache.size.get").count == 2


def test_trace_class_skips_dunder_by_default():
    handle = trace_class(_Cache, prefix="cache", mlx=False)
    try:
        c = _Cache()
        repr(c)
    finally:
        untrace(handle)
    snap = registry.snapshot()
    assert not any(t["name"].endswith("__init__") for t in snap["timers"])
    assert not any(t["name"].endswith("__repr__") for t in snap["timers"])


def test_untrace_restores_originals():
    original_add = _Cache.add
    handle = trace_class(_Cache, prefix="cache", mlx=False)
    assert _Cache.add is not original_add
    untrace(handle)
    assert _Cache.add is original_add


def test_trace_class_idempotent_on_already_traced():
    h1 = trace_class(_Cache, prefix="cache", mlx=False)
    method_after_first = _Cache.add
    h2 = trace_class(_Cache, prefix="cache", mlx=False)
    # The second pass should not double-wrap (idempotent).
    assert _Cache.add is method_after_first
    untrace(h2)
    untrace(h1)


def test_trace_module_records_module_functions(tmp_path):
    # Make a tiny module on the fly.
    import types
    mod = types.ModuleType("ephemeral_test_mod")
    mod.__name__ = "ephemeral_test_mod"

    def f(x: int) -> int:
        return x + 1

    f.__module__ = "ephemeral_test_mod"
    mod.f = f

    handle = trace_module(mod, prefix="emod", mlx=False)
    try:
        mod.f(1)
        mod.f(2)
        mod.f(3)
    finally:
        untrace(handle)
    assert registry.timer("emod.f").count == 3


def test_trace_skips_imports_from_other_modules():
    """Functions imported from another module must not be wrapped."""
    import types
    mod = types.ModuleType("ephemeral_test_mod_b")
    mod.__name__ = "ephemeral_test_mod_b"

    def native(x: int) -> int:
        return x + 1
    native.__module__ = "ephemeral_test_mod_b"

    def imported(x: int) -> int:
        return x + 1
    imported.__module__ = "some.other.module"

    mod.native = native
    mod.imported = imported

    handle = trace_module(mod, prefix="emodb", mlx=False)
    try:
        mod.native(1)
        mod.imported(1)
    finally:
        untrace(handle)
    assert registry.timer("emodb.native").count == 1
    # Imported functions are not wrapped, so there's no timer for them.
    snap = registry.snapshot()
    assert not any(t["name"] == "emodb.imported" for t in snap["timers"])


# --- Claims -----------------------------------------------------------------


def test_claim_passes_when_ratio_matches():
    def baseline():
        time.sleep(0.005)

    def optimized():
        time.sleep(0.001)

    c = Claim(
        name="5x_speedup",
        description="optimized is 5x faster",
        baseline=baseline,
        optimized=optimized,
        expected_ratio=5.0,
        tolerance=0.5,
        repeats=3,
        warmup=1,
    )
    result = run_claim(c)
    assert result.verdict() in ("PASS", "EXCEEDS")
    assert result.observed_ratio > 1.5


def test_claim_fails_on_dead_claim():
    def baseline():
        time.sleep(0.001)

    def optimized():
        time.sleep(0.001)

    c = Claim(
        name="fake_100x",
        description="claims 100x but does nothing",
        baseline=baseline,
        optimized=optimized,
        expected_ratio=100.0,
        repeats=3,
        warmup=1,
    )
    result = run_claim(c)
    assert result.verdict() == "FAIL"


def test_claim_captures_errors():
    def boom():
        raise RuntimeError("kaboom")

    c = Claim(
        name="errors",
        description="raises",
        baseline=boom,
        optimized=lambda: None,
        expected_ratio=2.0,
        repeats=3,
        warmup=1,
    )
    result = run_claim(c)
    assert result.verdict() == "ERROR"
    assert "kaboom" in (result.error or "")


def test_verify_all_writes_json(tmp_path):
    claims = [
        Claim(
            name="noop",
            description="noop",
            baseline=lambda: None,
            optimized=lambda: None,
            expected_ratio=1.0,
            tolerance=0.5,
            repeats=2,
            warmup=1,
        )
    ]
    out = tmp_path / "claims.json"
    results = verify_all(claims, dump_path=out)
    assert out.exists()
    assert len(results) == 1
    text = format_summary(results)
    assert "noop" in text
