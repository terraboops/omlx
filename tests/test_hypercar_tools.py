# SPDX-License-Identifier: Apache-2.0
"""Tests for Hypercar tooling: RULER generators, aggregate stats, MMLU-Pro.

These tests run without MLX or model loading — pure Python logic tests.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers: load modules without triggering omlx root import (avoids MLX)
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load ruler tasks directly
ruler_tasks = _load_module("ruler_tasks", "omlx/eval/ruler/tasks.py")


class FakeTokenizer:
    """Minimal tokenizer mock for RULER generators."""
    def encode(self, s):
        return list(range(max(1, len(s) // 4)))
    def decode(self, tokens):
        return " ".join(f"w{t}" for t in tokens)


# ---------------------------------------------------------------------------
# RULER generator tests
# ---------------------------------------------------------------------------

class TestMultiKeyNIAH:
    def test_basic_generation(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_multi_key_niah(tok, target_tokens=500, num_keys=3, seed=42)
        assert result["task_type"] == "multi_key_niah"
        assert len(result["expected"]) == 3
        assert result["question"]
        assert result["context"]

    def test_different_seeds_produce_different_keys(self):
        tok = FakeTokenizer()
        r1 = ruler_tasks.generate_multi_key_niah(tok, target_tokens=500, num_keys=2, seed=1)
        r2 = ruler_tasks.generate_multi_key_niah(tok, target_tokens=500, num_keys=2, seed=2)
        assert r1["expected"] != r2["expected"]

    def test_params_contain_keys(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_multi_key_niah(tok, target_tokens=500, num_keys=2, seed=42)
        # With a real tokenizer, needles would be in context. With FakeTokenizer,
        # verify the params record the keys correctly.
        assert len(result["params"]["keys"]) == 2
        for key, value in result["params"]["keys"]:
            assert key.startswith("key_")
            assert "-" in value  # XXXX-YYYY format

    def test_num_keys_matches(self):
        tok = FakeTokenizer()
        for n in [1, 2, 5]:
            result = ruler_tasks.generate_multi_key_niah(tok, target_tokens=500, num_keys=n, seed=42)
            assert len(result["expected"]) == n


class TestVariableTracking:
    def test_basic_generation(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_variable_tracking(tok, target_tokens=500, chain_length=4, seed=42)
        assert result["task_type"] == "variable_tracking"
        assert len(result["expected"]) == 1
        assert len(result["params"]["var_names"]) == 4

    def test_chain_length_creates_correct_vars(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_variable_tracking(tok, target_tokens=500, chain_length=6, seed=42)
        assert result["params"]["var_names"] == ["var_a", "var_b", "var_c", "var_d", "var_e", "var_f"]

    def test_value_is_random_code_format(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_variable_tracking(tok, target_tokens=500, chain_length=3, seed=42)
        value = result["expected"][0]
        # Format: XXXX-YYYY
        assert "-" in value
        assert len(value) == 9


class TestFrequentWord:
    def test_basic_generation(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_frequent_word(tok, target_tokens=500, num_target_words=5, seed=42)
        assert result["task_type"] == "frequent_word"
        assert len(result["expected"]) == 3  # word + lower + title

    def test_winner_is_in_expected(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_frequent_word(tok, target_tokens=500, num_target_words=5, seed=42)
        winner = result["params"]["winner"]
        assert winner in result["expected"]
        assert winner.lower() in result["expected"]

    def test_winner_has_highest_frequency(self):
        tok = FakeTokenizer()
        result = ruler_tasks.generate_frequent_word(tok, target_tokens=500, num_target_words=5, seed=42)
        markers = result["params"]["markers"]
        winner_freq = markers[0][1]  # (word, freq) pairs, winner is first
        for _, freq in markers[1:]:
            assert winner_freq >= freq


class TestSuites:
    def test_quick_suite_has_entries(self):
        assert len(ruler_tasks.RULER_QUICK_SUITE) >= 4

    def test_full_suite_has_entries(self):
        assert len(ruler_tasks.RULER_FULL_SUITE) >= 13

    def test_suite_entries_have_required_keys(self):
        for entry in ruler_tasks.RULER_QUICK_SUITE + ruler_tasks.RULER_FULL_SUITE:
            assert "generator" in entry
            assert "target_tokens" in entry
            assert "seed" in entry

    def test_vt_chain_lengths_4_and_8_present(self):
        chains = set()
        for entry in ruler_tasks.RULER_FULL_SUITE:
            if entry["generator"] == "variable_tracking":
                chains.add(entry["chain_length"])
        assert 4 in chains
        assert 8 in chains


# ---------------------------------------------------------------------------
# Aggregate stats tests
# ---------------------------------------------------------------------------

aggregate = _load_module("aggregate", "omlx/bench/aggregate.py")


class TestAggregateStats:
    def test_median_odd(self):
        assert aggregate._median([1, 3, 5]) == 3

    def test_median_even(self):
        assert aggregate._median([1, 2, 3, 4]) == 2.5

    def test_median_single(self):
        assert aggregate._median([42]) == 42

    def test_mean(self):
        assert aggregate._mean([2, 4, 6]) == 4.0

    def test_std_zero(self):
        assert aggregate._std([5, 5, 5]) == 0.0

    def test_std_nonzero(self):
        s = aggregate._std([2, 4, 6])
        assert 1.9 < s < 2.1  # sqrt(4) = 2

    def test_percentile_50(self):
        assert aggregate._percentile([1, 2, 3, 4, 5], 50) == 3

    def test_stats_full(self):
        vals = [10, 20, 30, 40, 50]
        s = aggregate._stats(vals)
        assert s["N"] == 5
        assert s["mean"] == 30.0
        assert s["median"] == 30.0
        assert s["min"] == 10.0
        assert s["max"] == 50.0

    def test_stats_empty(self):
        s = aggregate._stats([])
        assert s["N"] == 0


class TestAggregateGrouping:
    def test_group_by_sha(self):
        snaps = [
            {"git_commit": "abc", "_run_dir": "run1"},
            {"git_commit": "abc", "_run_dir": "run2"},
            {"git_commit": "def", "_run_dir": "run3"},
        ]
        groups = aggregate.group_by_sha(snaps)
        assert len(groups) == 2
        assert len(groups["abc"]) == 2
        assert len(groups["def"]) == 1


# ---------------------------------------------------------------------------
# MMLU-Pro answer extraction tests
# ---------------------------------------------------------------------------

mmlu_tasks = _load_module("mmlu_tasks", "omlx/eval/mmlu_pro/tasks.py")


class TestMMLUProAnswerExtraction:
    def test_standard_format(self):
        assert mmlu_tasks.extract_answer("The answer is (A)") == "A"

    def test_no_parens(self):
        assert mmlu_tasks.extract_answer("The answer is B") == "B"

    def test_answer_colon(self):
        assert mmlu_tasks.extract_answer("After analysis, Answer: C") == "C"

    def test_last_letter_fallback(self):
        assert mmlu_tasks.extract_answer("I think it's D based on the formula") == "D"

    def test_no_valid_letter(self):
        # No A-J letter in response
        assert mmlu_tasks.extract_answer("i don't know about that") == ""

    def test_multi_letter_takes_last(self):
        assert mmlu_tasks.extract_answer("Between A and C, I choose C") == "C"

    def test_format_prompt_has_options(self):
        q = {
            "question": "What is 2+2?",
            "options": ["3", "4", "5"],
            "category": "math",
        }
        prompt = mmlu_tasks.format_prompt(q)
        assert "A." in prompt
        assert "B." in prompt


# ---------------------------------------------------------------------------
# DuoAttention policy JSON tests (pure JSON, no MLX)
# ---------------------------------------------------------------------------


class TestDuoPolicy:
    def _load_policy(self):
        import json
        path = Path("omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json")
        if not path.exists():
            pytest.skip("DuoAttention policy not calibrated")
        return json.loads(path.read_text())

    def test_load_policy(self):
        policy = self._load_policy()
        assert "heads" in policy
        assert policy["streaming_fraction"] > 0

    def test_policy_has_all_layers(self):
        policy = self._load_policy()
        layers_seen = set(h["layer"] for h in policy["heads"])
        for layer in range(policy["n_layers"]):
            assert layer in layers_seen, f"Layer {layer} missing"

    def test_streaming_fraction_matches(self):
        policy = self._load_policy()
        total = len(policy["heads"])
        streaming = sum(1 for h in policy["heads"] if h["policy"] == "streaming")
        frac = streaming / total
        assert abs(frac - policy["streaming_fraction"]) < 0.01

    def test_all_heads_have_required_fields(self):
        policy = self._load_policy()
        for h in policy["heads"]:
            assert "layer" in h
            assert "head" in h
            assert "policy" in h
            assert h["policy"] in ("streaming", "retrieval")
            assert "local_fraction" in h

    def test_streaming_heads_have_high_local_fraction(self):
        policy = self._load_policy()
        for h in policy["heads"]:
            if h["policy"] == "streaming":
                assert h["local_fraction"] >= 0.80, (
                    f"Layer {h['layer']} head {h['head']}: "
                    f"streaming but local_frac={h['local_fraction']}"
                )

    def test_per_layer_distribution_reasonable(self):
        policy = self._load_policy()
        n_layers = policy["n_layers"]
        n_heads = policy["n_heads"]
        # Each layer should have exactly n_heads entries
        layer_counts = {}
        for h in policy["heads"]:
            layer_counts[h["layer"]] = layer_counts.get(h["layer"], 0) + 1
        for layer in range(n_layers):
            assert layer_counts.get(layer, 0) == n_heads, (
                f"Layer {layer}: {layer_counts.get(layer, 0)} heads, expected {n_heads}"
            )
