# SPDX-License-Identifier: Apache-2.0
"""Tests for MagicDec-style speculative decoding cost-model gate.

Pure Python tests — no MLX or GPU required.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Import directly — no MLX dependency
import importlib.util
import sys
_spec = importlib.util.spec_from_file_location("omlx.specdec_gate", "omlx/specdec_gate.py")
gate_mod = importlib.util.module_from_spec(_spec)
sys.modules["omlx.specdec_gate"] = gate_mod
_spec.loader.exec_module(gate_mod)

SpecDecGate = gate_mod.SpecDecGate
SpecDecDecision = gate_mod.SpecDecDecision
DEFAULT_CONSTANTS = gate_mod.DEFAULT_CONSTANTS
load_constants = gate_mod.load_constants
save_constants = gate_mod.save_constants


class TestSpecDecGate:
    """Test the speculative decoding gate logic."""

    def test_below_min_context_never_speculates(self):
        gate = SpecDecGate()
        should, decision = gate.should_speculate(context_len=2048)
        assert not should
        assert decision.projected_speedup == 1.0
        assert "min" in decision.reason

    def test_at_min_context_makes_decision(self):
        gate = SpecDecGate()
        should, decision = gate.should_speculate(context_len=4096)
        # At min context, the gate should make a real decision (not auto-reject)
        assert "min" not in decision.reason
        assert decision.projected_speedup > 0

    def test_high_accept_rate_favors_speculation(self):
        gate = SpecDecGate()
        _, dec_high = gate.should_speculate(context_len=16384, accept_rate=0.9)
        _, dec_low = gate.should_speculate(context_len=16384, accept_rate=0.3)
        # Higher accept rate should give better speedup
        assert dec_high.projected_speedup > dec_low.projected_speedup

    def test_more_draft_tokens_increases_throughput(self):
        gate = SpecDecGate()
        _, dec_few = gate.should_speculate(context_len=16384, draft_tokens=2)
        _, dec_many = gate.should_speculate(context_len=16384, draft_tokens=8)
        # More draft tokens should produce more throughput (with good accept rate)
        assert dec_many.projected_speedup > dec_few.projected_speedup

    def test_speedup_is_reasonable(self):
        """Speedup should be between 0.5x and 5x for realistic parameters."""
        gate = SpecDecGate()
        _, decision = gate.should_speculate(
            context_len=16384, accept_rate=0.7, draft_tokens=5)
        assert 0.5 < decision.projected_speedup < 5.0

    def test_decision_fields_populated(self):
        gate = SpecDecGate()
        _, decision = gate.should_speculate(context_len=8192)
        assert decision.context_len == 8192
        assert decision.accept_rate == 0.7  # default
        assert decision.draft_tokens == 5  # default
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_zero_accept_rate_never_profitable(self):
        gate = SpecDecGate()
        should, decision = gate.should_speculate(
            context_len=65536, accept_rate=0.0)
        # With 0% acceptance, only 1 token per cycle but draft cost is paid
        assert decision.projected_speedup < 1.0

    def test_custom_constants(self):
        """Custom constants should change the gate behavior."""
        constants = DEFAULT_CONSTANTS.copy()
        constants["compute_us"] = 50000  # Very slow model → spec-decode more helpful
        gate = SpecDecGate(constants)
        should, decision = gate.should_speculate(context_len=16384)
        # With a very slow model, draft overhead is relatively cheaper
        assert decision.projected_speedup > 1.0


class TestAutoRegressiveLatency:
    """Test the autoregressive latency model."""

    def test_latency_increases_with_context(self):
        gate = SpecDecGate()
        lat_short = gate._autoregressive_latency_us(1024)
        lat_long = gate._autoregressive_latency_us(65536)
        assert lat_long > lat_short

    def test_latency_is_positive(self):
        gate = SpecDecGate()
        lat = gate._autoregressive_latency_us(0)
        assert lat > 0  # base + compute even at 0 context

    def test_compute_dominates_at_short_context(self):
        """At short context, compute should dominate over KV load."""
        gate = SpecDecGate()
        lat = gate._autoregressive_latency_us(1024)
        compute = DEFAULT_CONSTANTS["compute_us"]
        # KV load at 1024 tokens is tiny compared to compute
        assert lat < compute * 1.1


class TestConstantsPersistence:
    """Test saving and loading calibration constants."""

    def test_save_and_load(self):
        constants = DEFAULT_CONSTANTS.copy()
        constants["calibrated"] = True
        constants["model"] = "test-model"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_constants(constants, path)
            loaded = load_constants(path)
            assert loaded["calibrated"] is True
            assert loaded["model"] == "test-model"
            assert loaded["compute_us"] == constants["compute_us"]
        finally:
            path.unlink(missing_ok=True)

    def test_load_missing_file_returns_defaults(self):
        loaded = load_constants("/nonexistent/path.json")
        assert loaded["calibrated"] is False
        assert loaded["compute_us"] == DEFAULT_CONSTANTS["compute_us"]

    def test_load_corrupt_file_returns_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w",
                                          delete=False) as f:
            f.write("not valid json{{{")
            path = Path(f.name)

        try:
            loaded = load_constants(path)
            assert loaded == DEFAULT_CONSTANTS
        finally:
            path.unlink(missing_ok=True)
