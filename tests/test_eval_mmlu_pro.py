# SPDX-License-Identifier: Apache-2.0
"""Tests for MMLU-Pro evaluation guardrails.

These tests run without MLX or model loading — pure Python logic tests.
"""

import importlib.util

import pytest


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load hypercar_bench constants without triggering full omlx import
# We can't import the module directly (MLX dependency), so we parse
# the constant from source.
def _read_mmlu_pro_min_max_tokens():
    """Read MMLU_PRO_MIN_MAX_TOKENS from hypercar_bench source."""
    import re
    from pathlib import Path
    src = Path("omlx/bench/hypercar_bench.py").read_text()
    m = re.search(r"^MMLU_PRO_MIN_MAX_TOKENS\s*=\s*(\d+)", src, re.MULTILINE)
    assert m, "MMLU_PRO_MIN_MAX_TOKENS constant not found in hypercar_bench.py"
    return int(m.group(1))


class TestMMLUProMaxTokensFloor:
    """Guardrail: MMLU-Pro max_tokens must not drop below 512.

    Cutting max_tokens from 512 to 256 caused a 64→24% silent regression
    because CoT reasoning was truncated before the model emitted its
    final answer letter. See commits 1e803b6→20df582.
    """

    def test_constant_is_at_least_512(self):
        val = _read_mmlu_pro_min_max_tokens()
        assert val >= 512, (
            f"MMLU_PRO_MIN_MAX_TOKENS={val} is below 512 — this will "
            f"silently regress MMLU-Pro accuracy (64→24% at 256 tokens). "
            f"See commits 1e803b6→20df582."
        )

    def test_constant_is_used_in_generate_call(self):
        """Verify the constant is actually wired into the generate call."""
        from pathlib import Path
        src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "max_tokens=MMLU_PRO_MIN_MAX_TOKENS" in src, (
            "phase3c_mmlu_pro must use MMLU_PRO_MIN_MAX_TOKENS constant, "
            "not a hardcoded value"
        )

    def test_assertion_exists_in_phase3c(self):
        """Verify there's a runtime assertion guarding the constant."""
        from pathlib import Path
        src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "assert MMLU_PRO_MIN_MAX_TOKENS >= 512" in src, (
            "phase3c_mmlu_pro must assert MMLU_PRO_MIN_MAX_TOKENS >= 512 "
            "to prevent silent quality regression"
        )


# Also test the MMLU-Pro answer extraction (already tested in
# test_hypercar_tools.py but having a dedicated file makes the
# guardrail discoverable via `pytest -k mmlu_pro`)
mmlu_tasks = _load_module("mmlu_tasks", "omlx/eval/mmlu_pro/tasks.py")


class TestMMLUProAnswerExtraction:
    def test_standard_format(self):
        assert mmlu_tasks.extract_answer("The answer is (A)") == "A"

    def test_no_parens(self):
        assert mmlu_tasks.extract_answer("The answer is B") == "B"

    def test_answer_colon(self):
        assert mmlu_tasks.extract_answer("After analysis, Answer: C") == "C"

    def test_truncated_reasoning_no_answer(self):
        """Simulate what happens with max_tokens=256: reasoning is cut off
        before the model emits its answer letter."""
        truncated = (
            "Let me think step by step. First, we need to consider "
            "the relationship between the variables. The function f(x) "
            "represents a polynomial of degree 3, which means..."
        )
        # No A-J letter at all in this truncated reasoning
        result = mmlu_tasks.extract_answer(truncated)
        assert result == "", (
            "Truncated reasoning without an answer letter should return empty"
        )
