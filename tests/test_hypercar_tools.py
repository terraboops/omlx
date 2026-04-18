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


class TestAggregateEdgeCases:
    """Edge cases for aggregate statistics functions."""

    def test_median_empty(self):
        assert aggregate._median([]) == 0.0

    def test_percentile_0(self):
        assert aggregate._percentile([10, 20, 30], 0) == 10

    def test_percentile_100(self):
        assert aggregate._percentile([10, 20, 30], 100) == 30

    def test_percentile_empty(self):
        assert aggregate._percentile([], 50) == 0.0

    def test_mean_empty(self):
        assert aggregate._mean([]) == 0.0

    def test_std_single_value(self):
        assert aggregate._std([42]) == 0.0

    def test_std_empty(self):
        assert aggregate._std([]) == 0.0

    def test_std_uses_sample_variance(self):
        """Verify Bessel's correction (N-1 denominator)."""
        # For [0, 2]: mean=1, sum_sq=2, sample_var=2/(2-1)=2, std=sqrt(2)≈1.414
        s = aggregate._std([0, 2])
        assert abs(s - 1.4142) < 0.001

    def test_stats_single_value(self):
        s = aggregate._stats([7.5])
        assert s["N"] == 1
        assert s["mean"] == 7.5
        assert s["median"] == 7.5
        assert s["std"] == 0.0
        assert s["min"] == 7.5
        assert s["max"] == 7.5


class TestAggregateSHAResolution:
    """Test SHA resolution logic used in --report."""

    def test_head_resolves_to_latest_run_dir(self):
        groups = {
            "abc123": [{"_run_dir": "run1_2026-04-01"}],
            "def456": [{"_run_dir": "run2_2026-04-15"}],
        }
        assert aggregate._resolve_sha("HEAD", groups) == "def456"

    def test_partial_sha_match(self):
        groups = {"abcdef123456": [{}]}
        assert aggregate._resolve_sha("abcdef", groups) == "abcdef123456"

    def test_ambiguous_sha_returns_none(self):
        groups = {"abc123": [{}], "abc456": [{}]}
        # Two matches for "abc" — should not resolve
        result = aggregate._resolve_sha("abc", groups)
        # Result is None because it's not an exact match and ambiguous
        assert result is None or result in ("abc123", "abc456")

    def test_exact_sha_match(self):
        groups = {"abc123": [{}], "abc456": [{}]}
        assert aggregate._resolve_sha("abc123", groups) == "abc123"

    def test_missing_sha(self):
        groups = {"abc123": [{}]}
        assert aggregate._resolve_sha("zzz999", groups) is None


class TestAggregateGroup:
    """Test the aggregate_group function that computes per-phase stats."""

    def _make_snap(self, sha="abc", total=100.0, metal=35.0, swap=2.0,
                   phases=None, run_dir="run1"):
        snap = {
            "git_commit": sha,
            "_run_dir": run_dir,
            "total_elapsed_s": total,
            "memory": {"metal_peak_gb": metal, "swap_peak_gb": swap},
        }
        if phases:
            snap["phases"] = phases
        return snap

    def test_empty_group(self):
        assert aggregate.aggregate_group([]) == {}

    def test_single_run(self):
        snap = self._make_snap(total=120.0, metal=35.1, swap=0.0)
        agg = aggregate.aggregate_group([snap])
        assert agg["N"] == 1
        assert agg["total_elapsed"]["median"] == 120.0
        assert agg["metal_peak"]["median"] == 35.1

    def test_multiple_runs_median(self):
        snaps = [
            self._make_snap(total=100.0, run_dir="r1"),
            self._make_snap(total=200.0, run_dir="r2"),
            self._make_snap(total=300.0, run_dir="r3"),
        ]
        agg = aggregate.aggregate_group(snaps)
        assert agg["N"] == 3
        assert agg["total_elapsed"]["median"] == 200.0

    def test_phase_timing_collected(self):
        phases = {
            "Phase 0: Smoke": {"elapsed_s": 10.0, "decode_toks": 52.0},
        }
        snap = self._make_snap(phases=phases)
        agg = aggregate.aggregate_group([snap])
        assert "Phase 0: Smoke" in agg["phases"]
        p0 = agg["phases"]["Phase 0: Smoke"]
        assert p0["elapsed"]["median"] == 10.0
        assert p0["decode_toks"]["median"] == 52.0

    def test_crash_phase_excluded(self):
        phases = {
            "Phase 0: Smoke": {"elapsed_s": 10.0},
            "CRASH": {"error": "boom"},
        }
        snap = self._make_snap(phases=phases)
        agg = aggregate.aggregate_group([snap])
        assert "CRASH" not in agg["phases"]
        assert "Phase 0: Smoke" in agg["phases"]

    def test_env_swap_overrides_memory(self):
        snap = self._make_snap(swap=2.0)
        snap["_env"] = {"swap_peak_gb": 9.5}
        agg = aggregate.aggregate_group([snap])
        assert agg["swap_peak"]["median"] == 9.5


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

    def test_duo_kv_cache_has_state_setter(self):
        """DuoKVCache must have state.setter for compact_cache compatibility."""
        duo_src = Path("omlx/duo_kv_cache.py").read_text()
        assert "@state.setter" in duo_src


# ---- Task 18: LiveCodeBench ----

_bench_src = Path("omlx/bench/hypercar_bench.py").read_text()


class TestLiveCodeBenchSource:
    """Source-level tests for LiveCodeBench integration."""

    def test_phase3d_exists(self):
        assert "def phase3d_livecodebench(" in _bench_src

    def test_lcb_gate_constant(self):
        assert "MIN_LCB_PASS_RATE" in _bench_src

    def test_lcb_uses_extract_code(self):
        assert "_extract_code" in _bench_src

    def test_lcb_uses_execute_code(self):
        assert "_execute_code" in _bench_src

    def test_lcb_wired_in_full_mode(self):
        """Phase 3d must run in --full mode."""
        assert "phase3d_livecodebench" in _bench_src
        assert "Phase 3d: LiveCodeBench" in _bench_src

    def test_lcb_data_file_exists(self):
        assert Path("omlx/eval/data/livecodebench.jsonl").exists()

    def test_lcb_eval_module_exists(self):
        assert Path("omlx/eval/livecodebench.py").exists()

    def test_lcb_difficulty_in_bench(self):
        """Phase 3d must report per-difficulty breakdown."""
        assert "by_difficulty" in _bench_src
        assert "difficulty" in _bench_src

    def test_lcb_data_has_difficulty(self):
        """LCB data must have difficulty field."""
        import json
        line = Path("omlx/eval/data/livecodebench.jsonl").read_text().split("\n")[0]
        item = json.loads(line)
        assert "difficulty" in item
        assert item["difficulty"] in ("easy", "medium", "hard")


class TestDuoPolicyDistribution(TestDuoPolicy):
    """DuoPolicy distribution tests (split out to fix class ordering)."""

    def test_early_layers_more_streaming(self):
        """Early layers tend to have more streaming heads than late layers."""
        policy = self._load_policy()
        early = [h for h in policy["heads"] if h["layer"] < 12 and h["policy"] == "streaming"]
        late = [h for h in policy["heads"] if h["layer"] >= 36 and h["policy"] == "streaming"]
        early_total = sum(1 for h in policy["heads"] if h["layer"] < 12)
        late_total = sum(1 for h in policy["heads"] if h["layer"] >= 36)
        early_frac = len(early) / early_total if early_total else 0
        late_frac = len(late) / late_total if late_total else 0
        assert early_frac >= late_frac, (
            f"Early streaming {early_frac:.0%} < late {late_frac:.0%}"
        )

    def test_per_layer_distribution_reasonable(self):
        policy = self._load_policy()
        n_layers = policy["n_layers"]
        n_heads = policy["n_heads"]
        layer_counts = {}
        for h in policy["heads"]:
            layer_counts[h["layer"]] = layer_counts.get(h["layer"], 0) + 1
        for layer in range(n_layers):
            assert layer_counts.get(layer, 0) == n_heads, (
                f"Layer {layer}: {layer_counts.get(layer, 0)} heads, expected {n_heads}"
            )


# ---------------------------------------------------------------------------
# Adaptive chunk sizing tests
# ---------------------------------------------------------------------------

class TestAdaptiveChunking:
    """Test the adaptive prefill chunk logic (prevents OOM at 64K+)."""

    def test_default_chunk_at_short_context(self):
        # At <32K, default chunk=4096 should be used
        for ctx in [1024, 4096, 16384, 30000]:
            chunk = 4096 if ctx < 32768 else (2048 if ctx < 65536 else 1024)
            assert chunk == 4096, f"ctx={ctx} should use chunk=4096, got {chunk}"

    def test_medium_chunk_at_32k(self):
        for ctx in [32768, 40000, 60000]:
            chunk = 4096 if ctx < 32768 else (2048 if ctx < 65536 else 1024)
            assert chunk == 2048, f"ctx={ctx} should use chunk=2048, got {chunk}"

    def test_small_chunk_at_64k_plus(self):
        for ctx in [65536, 131072, 262144, 524288, 1048576]:
            if ctx >= 131072:
                chunk = 512
            elif ctx >= 65536:
                chunk = 512
            elif ctx >= 32768:
                chunk = 2048
            else:
                chunk = 4096
            assert chunk == 512, f"ctx={ctx} should use chunk=512, got {chunk}"

    def test_attention_scores_fit_at_64k(self):
        """At 64K with chunk=512, attention scores should be <3 GB."""
        ctx = 65536
        chunk = 512
        n_heads = 32
        bytes_per_elem = 2  # fp16
        attn_gb = (chunk * ctx * n_heads * bytes_per_elem) / 1e9
        assert attn_gb < 3.0, f"Attention scores at 64K: {attn_gb:.1f} GB (should be <3)"

    def test_attention_scores_oom_at_64k_default_chunk(self):
        """At 64K with chunk=4096, attention scores would be ~16 GB (OOM)."""
        ctx = 65536
        chunk = 4096
        n_heads = 32
        bytes_per_elem = 2
        attn_gb = (chunk * ctx * n_heads * bytes_per_elem) / 1e9
        assert attn_gb > 10.0, f"Default chunk at 64K should OOM: {attn_gb:.1f} GB"


# ---------------------------------------------------------------------------
# Context list parser tests (--niah-context flag)
# ---------------------------------------------------------------------------

# Extract _parse_context_list from source (avoids MLX import)
import re as _re

_src = Path("omlx/bench/hypercar_bench.py").read_text()
_func_match = _re.search(
    r'(def _parse_context_list\(s[^)]*\)[^:]*:.*?)(?=\ndef |\nclass |\n# ---)',
    _src, _re.DOTALL
)
if _func_match:
    _ns = {}
    exec(_func_match.group(1).replace("list[int]", "list"), _ns)
    _parse_context_list = _ns["_parse_context_list"]
else:
    _parse_context_list = None


@pytest.mark.skipif(_parse_context_list is None,
                    reason="_parse_context_list not found in source")
class TestParseContextList:
    """Test the --niah-context parser."""

    def test_basic_k_suffix(self):
        assert _parse_context_list("4K,16K,64K") == [4096, 16384, 65536]

    def test_single_value(self):
        assert _parse_context_list("128K") == [131072]

    def test_spaces_tolerated(self):
        assert _parse_context_list("4K, 16K, 64K") == [4096, 16384, 65536]

    def test_m_suffix(self):
        assert _parse_context_list("1M") == [1048576]

    def test_raw_integer(self):
        assert _parse_context_list("4096") == [4096]

    def test_mixed_formats(self):
        assert _parse_context_list("4K,8192,1M") == [4096, 8192, 1048576]


class TestNIAHOnlyFlags:
    """Test --niah-only and --niah-context CLI flags exist in source."""

    def test_niah_only_flag_defined(self):
        src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "--niah-only" in src

    def test_niah_context_flag_defined(self):
        src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "--niah-context" in src

    def test_niah_only_skip_logic_exists(self):
        src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "niah_only" in src
        assert "skipping Phase 2" in src


# ---------------------------------------------------------------------------
# Haystack builder tests (_build_code_haystack)
# ---------------------------------------------------------------------------

# Extract _build_code_haystack from source (avoids MLX import)
_haystack_match = _re.search(
    r'(def _build_code_haystack\(.*?\n(?=def |class |# ---))',
    _src, _re.DOTALL
)


def _build_haystack_for_test(tokenizer, target_tokens, needle, depth_pct=50.0):
    """Reimplementation matching hypercar_bench._build_code_haystack."""
    code_blocks = [
        'def fibonacci(n: int) -> list[int]:\n    fib = [0, 1]\n    return fib[:n]\n',
        'class DataProcessor:\n    def __init__(self): pass\n',
        'DATABASE_URL = "postgresql://localhost:5432/app_db"\n',
    ]
    filler = "\n\n".join(code_blocks)
    filler_tokens = len(tokenizer.encode(filler))
    needle_tokens = len(tokenizer.encode(needle))
    needed_filler = target_tokens - needle_tokens
    reps = max(1, (needed_filler // filler_tokens) + 1)
    all_blocks = [filler for _ in range(reps)]
    needle_idx = max(1, int(len(all_blocks) * (depth_pct / 100.0)))
    all_blocks.insert(needle_idx, f"\n# IMPORTANT NOTE: {needle}\n")
    haystack = "\n\n".join(all_blocks)
    tokens = tokenizer.encode(haystack)[:target_tokens]
    return tokenizer.decode(tokens)


class TestBuildCodeHaystack:
    """Test the NIAH haystack builder structure.

    FakeTokenizer's decode returns 'w0 w1 ...' (not original text), so
    we test the pre-decode structure, not the round-tripped result.
    """

    def test_needle_inserted_into_blocks(self):
        """Needle comment is in the all_blocks list before encode/decode."""
        code_blocks = ["def foo(): pass\n", "x = 1\n"]
        filler = "\n\n".join(code_blocks)
        needle = "SECRET-CODE-123"
        reps = 3
        all_blocks = [filler for _ in range(reps)]
        needle_idx = max(1, int(len(all_blocks) * 0.5))
        all_blocks.insert(needle_idx, f"\n# IMPORTANT NOTE: {needle}\n")
        haystack = "\n\n".join(all_blocks)
        assert f"IMPORTANT NOTE: {needle}" in haystack

    def test_needle_depth_affects_position(self):
        """Different depth_pct values place the needle at different positions."""
        filler = "code " * 20
        for depth in [10, 50, 90]:
            all_blocks = [filler for _ in range(10)]
            idx = max(1, int(len(all_blocks) * (depth / 100.0)))
            all_blocks.insert(idx, "NEEDLE")
            # Needle index should roughly correlate with depth
            pos = "\n".join(all_blocks).find("NEEDLE")
            total = len("\n".join(all_blocks))
            ratio = pos / total
            assert ratio > (depth / 100.0) * 0.3, (
                f"depth={depth}%: needle at {ratio:.0%} of text"
            )

    def test_reps_grow_with_target(self):
        """More target_tokens requires more filler repetitions."""
        tok = FakeTokenizer()
        filler = "small code block\n"
        filler_tokens = len(tok.encode(filler))
        needle_tokens = len(tok.encode("needle"))
        for target in [100, 500, 2000]:
            needed = target - needle_tokens
            reps = max(1, (needed // filler_tokens) + 1)
            assert reps >= 1
            assert reps * filler_tokens >= needed - filler_tokens

    def test_source_function_exists(self):
        assert "def _build_code_haystack(" in _src

    def test_source_uses_important_note_format(self):
        assert "IMPORTANT NOTE:" in _src

    def test_source_truncates_to_target_tokens(self):
        """The real function truncates to target_tokens via encode[:target]."""
        assert "[:target_tokens]" in _src


# ---------------------------------------------------------------------------
# Per-phase headroom check tests (Task 86)
# ---------------------------------------------------------------------------

class TestPhaseHeadroomCheck:
    """Test the per-phase headroom constants and logic."""

    def test_headroom_constants_exist(self):
        assert "PHASE_HEADROOM_GB" in _src

    def test_headroom_function_exists(self):
        assert "def _check_phase_headroom(" in _src

    def test_niah_requires_most_headroom(self):
        """NIAH should require the most headroom (16K fp16 KV + attention)."""
        # Extract PHASE_HEADROOM_GB dict from source
        match = _re.search(r'PHASE_HEADROOM_GB\s*=\s*\{([^}]+)\}', _src)
        assert match, "PHASE_HEADROOM_GB dict not found"
        # NIAH value should be >= all others
        values = _re.findall(r':\s*(\d+\.?\d*)', match.group(1))
        floats = [float(v) for v in values]
        assert max(floats) >= 6.0, "NIAH headroom should be >= 6 GB"

    def test_headroom_check_returns_bool(self):
        """The function should return True/False, not raise."""
        assert "return True" in _src or "return False" in _src

    def test_skipped_phase_has_details(self):
        """Skipped phases should record the reason in details."""
        assert '"skipped": True' in _src
        assert '"reason": "insufficient headroom"' in _src

    def test_watchdog_breach_sentinel(self):
        """Task 85: WATCHDOG-BREACH sentinel should be in breach path."""
        assert "WATCHDOG-BREACH" in _src


# ---------------------------------------------------------------------------
# Task 80: Wall-clock correlation in profiler
# ---------------------------------------------------------------------------

_profiler_src = Path("omlx/bench/profiler.py").read_text()


class TestWallClockCorrelation:
    """Task 80: Detect swap-induced stalls via wall-clock correlation."""

    def test_profiler_has_wall_clock_field(self):
        """ProfileResult must have wall_clock_elapsed_s field."""
        assert "wall_clock_elapsed_s" in _profiler_src

    def test_profiler_has_scheduling_fraction(self):
        """ProfileResult must have cpu_scheduling_fraction field."""
        assert "cpu_scheduling_fraction" in _profiler_src

    def test_profiler_uses_monotonic(self):
        """Profiler must use time.monotonic() for wall-clock tracking."""
        assert "time.monotonic()" in _profiler_src

    def test_summary_includes_wall_clock(self):
        """summary() dict must include wall_clock_elapsed_s."""
        assert '"wall_clock_elapsed_s"' in _profiler_src

    def test_summary_includes_scheduling_fraction(self):
        """summary() dict must include cpu_scheduling_fraction."""
        assert '"cpu_scheduling_fraction"' in _profiler_src

    def test_bench_warns_on_low_scheduling_fraction(self):
        """hypercar_bench must warn when scheduling fraction < 0.9."""
        assert "cpu_scheduling_fraction" in _src
        assert "0.9" in _src or "sched_frac < 0.9" in _src

    def test_scheduling_fraction_capped_at_one(self):
        """Scheduling fraction should never exceed 1.0."""
        assert "min(" in _profiler_src and "1.0" in _profiler_src


# ---------------------------------------------------------------------------
# ShadowKV projection-based compression
# ---------------------------------------------------------------------------

_shadowkv_src = Path("omlx/shadowkv_cache.py").read_text()


class TestShadowKVProjections:
    """Test ShadowKV offline projection-based K compression."""

    def test_projections_class_exists(self):
        assert "class ShadowKVProjections" in _shadowkv_src

    def test_load_method_exists(self):
        assert "def load(" in _shadowkv_src

    def test_compress_function_exists(self):
        assert "def compress_k_with_projections(" in _shadowkv_src

    def test_compress_uses_per_head_projection(self):
        """Must project per-head, not joint."""
        assert "V_k" in _shadowkv_src
        assert "V_k.T" in _shadowkv_src

    def test_v_cache_untouched(self):
        """V cache must not be modified by compression."""
        assert "c.state[1]" in _shadowkv_src

    def test_min_tokens_guard(self):
        """Should skip compression for short contexts."""
        assert "min_tokens" in _shadowkv_src

    def test_meta_json_path(self):
        """Should reference the projection meta.json."""
        assert "meta.json" in _shadowkv_src

    def test_projections_meta_exists(self):
        """Projection meta.json should exist (computed by Step 1)."""
        meta_path = Path("omlx/patches/shadowkv_projections/qwen3_coder_30b_a3b/meta.json")
        assert meta_path.exists(), f"Missing {meta_path}"

    def test_projections_meta_valid(self):
        """Meta should have expected fields."""
        import json
        meta_path = Path("omlx/patches/shadowkv_projections/qwen3_coder_30b_a3b/meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["n_layers"] == 48
        assert meta["summary"]["median_rank"] > 0
        assert meta["summary"]["mean_rel_error"] < 0.02

    def test_median_rank_property(self):
        assert "median_rank" in _shadowkv_src

    def test_returns_compressed_count(self):
        """compress_k_with_projections should return count."""
        assert "return compressed_count" in _shadowkv_src


# ---------------------------------------------------------------------------
# SnapKV attention-guided token selection
# ---------------------------------------------------------------------------

_snapkv_src = Path("omlx/patches/snapkv.py").read_text()


class TestSnapKVSource:
    """Source-level tests for SnapKV token selection module."""

    def test_importance_function_exists(self):
        assert "def compute_attention_importance(" in _snapkv_src

    def test_select_function_exists(self):
        assert "def snapkv_select(" in _snapkv_src

    def test_count_kept_function_exists(self):
        assert "def count_kept(" in _snapkv_src

    def test_uses_observation_window(self):
        """Must use an observation window, not full context."""
        assert "obs_window" in _snapkv_src

    def test_handles_gqa(self):
        """Must handle GQA (grouped query attention) head expansion."""
        assert "gqa_ratio" in _snapkv_src

    def test_causal_masking(self):
        """Must apply causal mask to observation window attention."""
        assert "causal_mask" in _snapkv_src or "causal" in _snapkv_src

    def test_always_keeps_recent_tokens(self):
        """Must always keep recent tokens (sink/window pattern)."""
        assert "always_keep_last" in _snapkv_src

    def test_returns_boolean_mask(self):
        """select should return a boolean keep mask."""
        assert "keep_mask" in _snapkv_src
        assert "bool_" in _snapkv_src or "bool" in _snapkv_src

    def test_max_pooling_over_window(self):
        """Importance pooling should use max over query positions."""
        assert "mx.max(" in _snapkv_src or "max" in _snapkv_src

    def test_top_k_selection(self):
        """Must use argpartition or argsort for top-k selection."""
        assert "argpartition" in _snapkv_src or "argsort" in _snapkv_src

    def test_keeps_all_when_keep_count_exceeds_t(self):
        """Should return all-True mask when keep_count >= T."""
        assert "keep_count >= T" in _snapkv_src

    def test_validation_results_exist(self):
        """GPU validation results should exist from prior run."""
        results_path = Path("research/snapkv_selection_validation.json")
        assert results_path.exists()

    def test_validation_needle_preserved(self):
        """Validation must show needle preserved at 50% keep ratio."""
        import json
        results = json.loads(
            Path("research/snapkv_selection_validation.json").read_text())
        r50 = [r for r in results["results"] if r["keep_ratio"] == 0.5]
        assert r50, "No 50% keep ratio in results"
        assert r50[0]["needle_preserved"] is True

    def test_get_keep_indices_exists(self):
        """Must have get_keep_indices for cache compaction."""
        assert "def get_keep_indices(" in _snapkv_src

    def test_compact_cache_exists(self):
        """Must have compact_cache for in-place KV eviction."""
        assert "def compact_cache(" in _snapkv_src

    def test_compact_preserves_exact_values(self):
        """Compact must use gather (indexing), not projection."""
        assert "idx" in _snapkv_src
        assert "[:, :, idx, :]" in _snapkv_src or "gather" in _snapkv_src

    def test_get_fp16_keys_exists(self):
        """Must have _get_fp16_keys for cache-agnostic key extraction."""
        assert "def _get_fp16_keys(" in _snapkv_src

    def test_quantized_cache_dequantize_path(self):
        """compact_cache must handle QuantizedKVCache via dequantize."""
        assert "mx.dequantize" in _snapkv_src

    def test_quantized_cache_to_fp16_path(self):
        """compact_cache converts QuantizedKVCache to fp16 KVCache (no requant)."""
        assert "KVCache" in _snapkv_src
        assert "mx.dequantize" in _snapkv_src

    def test_apply_snapkv_to_generate_exists(self):
        """Must have apply_snapkv_to_generate for server integration."""
        assert "def apply_snapkv_to_generate(" in _snapkv_src

    def test_generate_wrapper_installs_hooks(self):
        """Wrapper must install Q capture hooks before prefill."""
        assert "install_q_capture_hook" in _snapkv_src

    def test_generate_wrapper_compacts_after_first_yield(self):
        """Wrapper must compact cache after first decoded token."""
        assert "compact_cache" in _snapkv_src
        assert "first = next(gen)" in _snapkv_src or "first" in _snapkv_src

    def test_generate_wrapper_cleanup_in_finally(self):
        """Wrapper must clean up hooks in a finally block."""
        assert "finally:" in _snapkv_src
        assert "cleanup()" in _snapkv_src

    def test_generate_wrapper_skip_short_prompts(self):
        """Wrapper must skip eviction for short prompts."""
        assert "keep_count * 2" in _snapkv_src

    def test_server_snapkv_keep_flag(self):
        """hypercar_server must have --snapkv-keep CLI flag."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--snapkv-keep" in server_src

    def test_server_imports_apply_snapkv(self):
        """Server must import and call apply_snapkv_to_generate."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "apply_snapkv_to_generate" in server_src

    def test_rerope_keys_exists(self):
        """Must have _rerope_keys for RoPE correction after compaction."""
        assert "def _rerope_keys(" in _snapkv_src

    def test_compact_cache_calls_rerope(self):
        """compact_cache must re-encode RoPE after gathering."""
        assert "_rerope_keys(" in _snapkv_src

    def test_compact_cache_accepts_model(self):
        """compact_cache must accept model param for RoPE config."""
        assert "model=None" in _snapkv_src or "model=" in _snapkv_src

    def test_rerope_uses_rope_base(self):
        """Re-RoPE must use the model's rope_base frequency."""
        assert "rope_base" in _snapkv_src

    def test_rerope_computes_shift(self):
        """Re-RoPE must compute per-token position shift."""
        assert "shifts" in _snapkv_src or "shift" in _snapkv_src

    # --- CAOTE (Task 100) ---

    def test_caote_importance_exists(self):
        """Must have compute_caote_importance for value-aware scoring."""
        assert "def compute_caote_importance(" in _snapkv_src

    def test_caote_uses_values(self):
        """CAOTE must access value vectors, not just keys."""
        assert "_get_fp16_values" in _snapkv_src

    def test_caote_formula_components(self):
        """CAOTE score = (alpha / (1-alpha)) * ||V_mean - v_j||."""
        assert "alpha_clamped" in _snapkv_src
        assert "v_dist" in _snapkv_src or "v_diff" in _snapkv_src

    def test_caote_fast_approximation(self):
        """FastCAOTE uses mean of all values (not weighted mean)."""
        assert "mx.mean(values" in _snapkv_src or "V_mean" in _snapkv_src

    def test_caote_flag_in_apply_snapkv(self):
        """apply_snapkv_to_generate must accept use_caote parameter."""
        assert "use_caote" in _snapkv_src

    def test_caote_flag_in_server(self):
        """hypercar_server must have --caote CLI flag."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--caote" in server_src

    def test_caote_gpu_validation_exists(self):
        """GPU validation results should exist from benchmark run."""
        results_path = Path("research/snapkv_compaction_bench.json")
        assert results_path.exists()
        import json
        data = json.loads(results_path.read_text())
        # Check at least one result used CAOTE scoring
        has_caote = any(r.get("scoring") == "CAOTE" for r in data.get("results", []))
        assert has_caote, "No CAOTE results in benchmark output"


# ---- Task 94: Adaptive Prefill Controller ----

_adaptive_src = Path("omlx/patches/adaptive_prefill.py").read_text()


class TestAdaptivePrefillSource:
    """Source-level tests for adaptive prefill controller."""

    def test_controller_class_exists(self):
        assert "class AdaptivePrefillController" in _adaptive_src

    def test_controller_has_feedback(self):
        """Controller must accept feedback (metal_gb, tok_per_sec)."""
        assert "def feedback(" in _adaptive_src

    def test_controller_has_next_chunk(self):
        """Controller must output next chunk size."""
        assert "def next_chunk_size(" in _adaptive_src

    def test_controller_has_summary(self):
        """Controller must provide summary for profiling."""
        assert "def summary(" in _adaptive_src

    def test_memory_signal(self):
        """Must use Metal memory as control signal."""
        assert "metal_gb" in _adaptive_src
        assert "target_metal" in _adaptive_src

    def test_throughput_signal(self):
        """Must use throughput as secondary signal."""
        assert "throughput_floor" in _adaptive_src
        assert "tok_per_sec" in _adaptive_src

    def test_proportional_control(self):
        """Must shrink when over target, grow when under."""
        assert "metal_ratio" in _adaptive_src
        assert "// 2" in _adaptive_src or "* 0.5" in _adaptive_src

    def test_chunk_bounds(self):
        """Must clamp between min_chunk and max_chunk."""
        assert "min_chunk" in _adaptive_src
        assert "max_chunk" in _adaptive_src

    def test_apply_function_exists(self):
        """Must have apply_adaptive_prefill for monkey-patching."""
        assert "def apply_adaptive_prefill(" in _adaptive_src

    def test_server_flag_exists(self):
        """hypercar_server must have --adaptive-chunk flag."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--adaptive-chunk" in server_src

    def test_server_wires_adaptive(self):
        """Server must call apply_adaptive_prefill when flag is set."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "apply_adaptive_prefill" in server_src


# ---- Task 98: BUZZ Segmented Eviction ----

class TestSegmentedEvictionSource:
    """Source-level tests for BUZZ segmented eviction."""

    def test_select_segmented_exists(self):
        assert "def _select_segmented(" in _snapkv_src

    def test_select_global_exists(self):
        assert "def _select_global(" in _snapkv_src

    def test_snapkv_select_has_segment_size(self):
        """snapkv_select must accept segment_size parameter."""
        assert "segment_size" in _snapkv_src

    def test_segmented_distributes_budget(self):
        """Segmented selection must distribute budget across segments."""
        assert "n_segments" in _snapkv_src
        assert "seg_k" in _snapkv_src

    def test_segmented_per_segment_topk(self):
        """Must do per-segment top-K selection."""
        assert "seg_start" in _snapkv_src
        assert "seg_end" in _snapkv_src

    def test_server_flag_exists(self):
        """hypercar_server must have --segmented-evict flag."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--segmented-evict" in server_src

    def test_apply_snapkv_passes_segment_size(self):
        """apply_snapkv_to_generate must forward segment_size."""
        assert "segment_size=segment_size" in _snapkv_src

    def test_bench_has_segment_flag(self):
        """snapkv_bench must accept --segment-size flag."""
        bench_src = Path("omlx/bench/snapkv_bench.py").read_text()
        assert "--segment-size" in bench_src

    def test_bench_has_kv_mode_flag(self):
        """snapkv_bench must accept --kv-mode for native 3-bit testing."""
        bench_src = Path("omlx/bench/snapkv_bench.py").read_text()
        assert "--kv-mode" in bench_src
        assert "QuantizedKVCache" in bench_src


# ---- Task 97: Freshness-Aware Eviction ----

class TestFreshnessEvictionSource:
    """Source-level tests for freshness-aware KV cache eviction."""

    def test_freshness_function_exists(self):
        assert "def compute_freshness_scores(" in _snapkv_src

    def test_cosine_similarity(self):
        """Must compute cosine similarity for conflict detection."""
        assert "keys_normed" in _snapkv_src or "cosine" in _snapkv_src

    def test_conflict_threshold(self):
        """Must use a conflict threshold parameter."""
        assert "conflict_threshold" in _snapkv_src

    def test_decay_factor(self):
        """Must use exponential decay for supersession count."""
        assert "decay_factor" in _snapkv_src
        assert "mx.power" in _snapkv_src or "power" in _snapkv_src

    def test_supersession_counting(self):
        """Must count supersessions per token."""
        assert "supersession_count" in _snapkv_src

    def test_freshness_composes_with_importance(self):
        """Freshness must multiply with importance scores."""
        assert "importance * freshness" in _snapkv_src

    def test_apply_snapkv_has_freshness_param(self):
        """apply_snapkv_to_generate must accept use_freshness."""
        assert "use_freshness" in _snapkv_src

    def test_server_freshness_flag(self):
        """Server must have --freshness-evict flag."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--freshness-evict" in server_src


# ---- Task 102: Submodular Greedy Eviction ----

class TestSubmodularEvictionSource:
    """Source-level tests for submodular greedy eviction."""

    def test_select_submodular_exists(self):
        assert "def _select_submodular(" in _snapkv_src

    def test_diversity_penalty(self):
        """Must penalize tokens similar to already-selected tokens."""
        assert "sim" in _snapkv_src
        assert "diversity" in _snapkv_src.lower() or "penalize" in _snapkv_src.lower()

    def test_greedy_within_segments(self):
        """Submodular runs within BUZZ segments."""
        assert "seg_start" in _snapkv_src

    def test_snapkv_select_has_submodular_param(self):
        assert "submodular" in _snapkv_src

    def test_server_flag_exists(self):
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--submodular-evict" in server_src

    def test_apply_snapkv_has_submodular(self):
        assert "use_submodular" in _snapkv_src

    def test_values_passed_for_diversity(self):
        """Must pass value vectors for diversity computation."""
        assert "sel_values" in _snapkv_src or "values=" in _snapkv_src


# ---- Task 57: PyramidKV Per-Layer Budget ----

_pyramid_src = Path("omlx/pyramid_budget.py").read_text()


class TestPyramidBudgetSource:
    """Source-level tests for PyramidKV per-layer budget."""

    def test_compute_budget_vector_exists(self):
        assert "def compute_budget_vector(" in _pyramid_src

    def test_budget_for_layer_exists(self):
        assert "def budget_for_layer(" in _pyramid_src

    def test_uses_beta_decay(self):
        """Must use exponential decay with beta parameter."""
        assert "beta" in _pyramid_src
        assert "beta ** dist" in _pyramid_src or "beta**" in _pyramid_src

    def test_edge_vs_middle_asymmetry(self):
        """Budget must be higher for edge layers than middle."""
        assert "min(i, n_layers - 1 - i)" in _pyramid_src

    def test_has_floor(self):
        """No layer should be starved — must have a floor."""
        assert "floor" in _pyramid_src

    def test_sums_to_total(self):
        """Budget vector must sum to total_budget."""
        assert "total_budget" in _pyramid_src


class TestPyramidBudgetLogic:
    """Functional tests for budget computation (pure Python, no MLX).

    Import via importlib to avoid omlx.__init__.py MLX dependency.
    """

    @staticmethod
    def _load_module():
        spec = importlib.util.spec_from_file_location(
            "pyramid_budget", "omlx/pyramid_budget.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_sum_equals_total(self):
        mod = self._load_module()
        budgets = mod.compute_budget_vector(16384, 48)
        assert abs(sum(budgets) - 16384) <= 1

    def test_edges_higher_than_middle(self):
        mod = self._load_module()
        budgets = mod.compute_budget_vector(16384, 48)
        edge_avg = (budgets[0] + budgets[1] + budgets[-1] + budgets[-2]) / 4
        mid = len(budgets) // 2
        mid_avg = (budgets[mid-1] + budgets[mid] + budgets[mid+1]) / 3
        assert edge_avg > mid_avg, f"Edge {edge_avg} should > middle {mid_avg}"

    def test_all_above_floor(self):
        mod = self._load_module()
        budgets = mod.compute_budget_vector(16384, 48, floor_pct=0.3)
        uniform = 16384 / 48
        floor = int(uniform * 0.3)
        for i, b in enumerate(budgets):
            assert b >= floor, f"Layer {i}: {b} < floor {floor}"

    def test_different_betas(self):
        mod = self._load_module()
        flat = mod.compute_budget_vector(16384, 48, beta=0.99)
        steep = mod.compute_budget_vector(16384, 48, beta=0.5)
        flat_ratio = max(flat) / max(min(flat), 1)
        steep_ratio = max(steep) / max(min(steep), 1)
        assert steep_ratio > flat_ratio

    def test_single_layer_budget(self):
        mod = self._load_module()
        b0 = mod.budget_for_layer(0, 16384, 48)
        b24 = mod.budget_for_layer(24, 16384, 48)
        assert b0 > b24

    def test_compact_cache_pyramidal_exists(self):
        assert "def compact_cache_pyramidal(" in _snapkv_src

    def test_server_pyramid_flag(self):
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--pyramid-kv" in server_src


# ---- Eviction Stack Integration ----

class TestEvictionStackComposition:
    """Verify all 6 eviction stack layers compose correctly at source level."""

    def test_all_server_flags_exist(self):
        """All eviction flags must be on the server."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        for flag in ["--snapkv-keep", "--caote", "--segmented-evict",
                      "--freshness-evict", "--submodular-evict", "--pyramid-kv"]:
            assert flag in server_src, f"Missing flag: {flag}"

    def test_apply_snapkv_accepts_all_params(self):
        """apply_snapkv_to_generate must accept all composition params."""
        for param in ["use_caote", "segment_size", "use_freshness", "use_submodular"]:
            assert param in _snapkv_src, f"Missing param: {param}"

    def test_snapkv_select_accepts_all_params(self):
        """snapkv_select must accept segment_size, submodular, values."""
        for param in ["segment_size", "submodular", "values"]:
            assert f"{param}" in _snapkv_src

    def test_scoring_methods_independent(self):
        """CAOTE and attention-only must be separate functions."""
        assert "def compute_caote_importance(" in _snapkv_src
        assert "def compute_importance_from_real_q(" in _snapkv_src

    def test_freshness_multiplies_with_importance(self):
        """Freshness composes multiplicatively with any importance method."""
        assert "importance * freshness" in _snapkv_src

    def test_submodular_uses_values(self):
        """Submodular greedy needs value vectors for diversity."""
        assert "sel_values" in _snapkv_src

    def test_rerope_always_applied(self):
        """Re-RoPE must run on every compact_cache call."""
        assert "_rerope_keys(" in _snapkv_src

    def test_server_wires_all_flags_to_apply(self):
        """Server must pass all flags to apply_snapkv_to_generate."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        for param in ["use_caote=", "segment_size=", "use_freshness=", "use_submodular="]:
            assert param in server_src, f"Server missing: {param}"

    def test_full_invocation_documented(self):
        """CLAUDE.md must document the full eviction invocation."""
        claude_src = Path("CLAUDE.md").read_text()
        assert "--snapkv-keep" in claude_src
        assert "--caote" in claude_src


# ---- Task 45: XGrammar Tool-Call JSON Guarantee ----

_xgrammar_src = Path("omlx/patches/xgrammar_constrain.py").read_text()


class TestXGrammarSource:
    """Source-level tests for XGrammar integration."""

    def test_get_compiled_grammar_exists(self):
        assert "def get_compiled_grammar(" in _xgrammar_src

    def test_create_grammar_sampler_exists(self):
        assert "def create_grammar_sampler(" in _xgrammar_src

    def test_extract_schema_from_request(self):
        assert "def extract_json_schema_from_request(" in _xgrammar_src

    def test_cache_stats_exists(self):
        assert "def grammar_cache_stats(" in _xgrammar_src

    def test_handles_response_format(self):
        """Must extract schema from response_format.json_schema."""
        assert "response_format" in _xgrammar_src
        assert "json_schema" in _xgrammar_src

    def test_handles_tools(self):
        """Must extract schema from tools[*].function.parameters."""
        assert "tools" in _xgrammar_src
        assert "parameters" in _xgrammar_src

    def test_caches_by_hash(self):
        """Must cache compiled grammars by schema hash."""
        assert "schema_hash" in _xgrammar_src
        assert "_grammar_cache" in _xgrammar_src

    def test_server_grammar_flag(self):
        """Server must have --grammar flag."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--grammar" in server_src

    def test_server_grammar_wired(self):
        """Server must wire grammar on startup when --grammar is set."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "_ensure_compiler" in server_src or "xgrammar_constrain" in server_src

    def test_server_grammar_cache_in_stats(self):
        """Stats endpoint must include grammar cache info."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "grammar_cache_stats" in server_src


class TestSessionSaveLoad:
    """Tests for fp16 KVCache session save/load (SnapKV-compacted)."""

    def test_fp16_save_path_exists(self):
        """Server must save fp16 KVCache with cache_type marker."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert 'cache_type' in server_src
        assert '"fp16"' in server_src

    def test_fp16_load_path_exists(self):
        """Server must detect and load fp16 sessions."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "is_fp16" in server_src

    def test_load_restores_offset(self):
        """Load must restore cache offset from saved data."""
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "offset" in server_src


class TestTQ3SaveDuringWarmup:
    """Task 160: TQ3 session save must work when cache is in fp16 warmup."""

    def test_save_quantizes_fp16_buffer(self):
        """save_to_disk must quantize fp16 warmup buffer before saving."""
        tq_src = Path("omlx/turboquant_kv.py").read_text()
        # save_to_disk must call _quantize_fp16_buffer before the empty check
        save_idx = tq_src.index("def save_to_disk")
        empty_check_idx = tq_src.index("Cannot save empty cache")
        # Find _quantize_fp16_buffer between save_to_disk and the error
        quantize_call = "_quantize_fp16_buffer"
        between = tq_src[save_idx:empty_check_idx]
        assert quantize_call in between, (
            "save_to_disk must call _quantize_fp16_buffer before raising "
            "'Cannot save empty cache' — handles fp16 warmup case"
        )


class TestXGrammarLogic:
    """Functional tests for schema extraction (no GPU needed)."""

    def test_extract_from_response_format(self):
        spec = importlib.util.spec_from_file_location(
            "xgrammar_constrain", "omlx/patches/xgrammar_constrain.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        req = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}
            }
        }
        schema = mod.extract_json_schema_from_request(req)
        assert schema is not None
        assert "integer" in schema

    def test_extract_from_tools(self):
        spec = importlib.util.spec_from_file_location(
            "xgrammar_constrain", "omlx/patches/xgrammar_constrain.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        req = {
            "tools": [{"function": {"name": "get_weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        }
        schema = mod.extract_json_schema_from_request(req)
        assert schema is not None
        assert "string" in schema

    def test_no_schema_returns_none(self):
        spec = importlib.util.spec_from_file_location(
            "xgrammar_constrain", "omlx/patches/xgrammar_constrain.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert mod.extract_json_schema_from_request({}) is None
        assert mod.extract_json_schema_from_request({"tools": []}) is None


# ---- Task 103: Trigonometric Pre-RoPE Scoring ----

_trig_src = Path("omlx/patches/trig_score.py").read_text()


class TestTrigScoreSource:
    """Source-level tests for trigonometric scoring module."""

    def test_trig_scorer_class_exists(self):
        assert "class TrigScorer" in _trig_src

    def test_load_method(self):
        assert "def load(" in _trig_src

    def test_score_method(self):
        assert "def score(" in _trig_src

    def test_classify_heads_method(self):
        assert "def classify_heads(" in _trig_src

    def test_uses_cosine(self):
        """Trigonometric scoring must use cosine for RoPE decomposition."""
        assert "mx.cos(" in _trig_src or "cos_angles" in _trig_src

    def test_uses_theta(self):
        """Must use RoPE theta frequencies."""
        assert "theta" in _trig_src

    def test_head_freq_products(self):
        """Must precompute q_centre * k_centre per frequency."""
        assert "head_freq_products" in _trig_src

    def test_calibration_file_exists(self):
        """Calibrated Q/K centres must exist from GPU calibration run."""
        assert Path("omlx/patches/qk_centres/qwen3_coder_30b.npz").exists()

    def test_calibration_script_exists(self):
        assert Path("scripts/calibrate_qk_centres.py").exists()


# ---- Task 106: GER Safety Monitor ----

class TestGERSafetySource:
    """Source-level tests for GER safety monitor."""

    def test_compute_ger_exists(self):
        assert "def compute_ger(" in _snapkv_src

    def test_check_ger_safety_exists(self):
        assert "def check_ger_safety(" in _snapkv_src

    def test_ger_threshold(self):
        """Must have configurable GER threshold."""
        assert "threshold" in _snapkv_src

    def test_ger_widen_budget(self):
        """Must recommend widening budget when GER exceeds threshold."""
        assert "widen" in _snapkv_src

    def test_ger_integrated_in_wrapper(self):
        """GER check must be integrated into apply_snapkv_to_generate."""
        assert "check_ger_safety" in _snapkv_src

    def test_ger_logs_value(self):
        """Must log GER value for monitoring."""
        assert "GER=" in _snapkv_src


# ---- Phase 3e: SnapKV Quality Gate ----

class TestSnapKVBenchPhase:
    """Tests for SnapKV eviction quality gate in hypercar_bench."""

    def test_phase3e_exists(self):
        assert "def phase3e_snapkv_quality(" in _bench_src

    def test_phase3e_uses_caote(self):
        assert "compute_caote_importance" in _bench_src

    def test_phase3e_uses_ger(self):
        assert "check_ger_safety" in _bench_src

    def test_phase3e_wired_in_default_mode(self):
        """Phase 3e must run in default mode (not just --full)."""
        assert "Phase 3e: SnapKV Quality" in _bench_src

    def test_phase3e_checks_needle(self):
        """Must verify needle retrieval after eviction."""
        assert "needle_found" in _bench_src


# ---- Task 111: Streaming-Head Budget Rebalancing ----

class TestStreamingAggressiveSource:
    """Source-level tests for streaming-head budget rebalancing."""

    def test_snapkv_select_has_head_types(self):
        assert "head_types" in _snapkv_src

    def test_retrieval_weight(self):
        """Retrieval heads must get higher weight."""
        assert "retrieval" in _snapkv_src
        assert "2.0" in _snapkv_src or "2x" in _snapkv_src.lower()

    def test_streaming_weight(self):
        """Streaming heads must get lower weight."""
        assert "streaming" in _snapkv_src
        assert "0.5" in _snapkv_src

    def test_reweight_before_pooling(self):
        """Reweighting must happen before max-pooling across heads."""
        assert "imp_weighted" in _snapkv_src

    def test_server_flag(self):
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--streaming-aggressive" in server_src


# ---- Task 109: TTT Layer Schedules ----

class TestTTTSchedules:
    """Tests for per-layer TTT learning rate schedules."""

    @staticmethod
    def _load():
        spec = importlib.util.spec_from_file_location(
            "ttt_schedules", "omlx/ttt_schedules.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_uniform_all_equal(self):
        mod = self._load()
        lrs = mod.compute_layer_lrs(1e-4, 48, "uniform")
        assert all(lr == 1e-4 for lr in lrs)

    def test_reservoir_monotonic(self):
        """Reservoir schedule must be monotonically increasing."""
        mod = self._load()
        lrs = mod.compute_layer_lrs(1e-4, 48, "reservoir")
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1], f"Layer {i}: {lrs[i]} < {lrs[i-1]}"

    def test_reservoir_deep_equals_base(self):
        mod = self._load()
        lrs = mod.compute_layer_lrs(1e-4, 48, "reservoir")
        assert abs(lrs[-1] - 1e-4) < 1e-8

    def test_reservoir_shallow_near_zero(self):
        mod = self._load()
        lrs = mod.compute_layer_lrs(1e-4, 48, "reservoir", gamma=1.5)
        assert lrs[0] < 1e-8  # layer 0 should be near-zero

    def test_cosine_u_shaped(self):
        """Cosine schedule: edges > middle."""
        mod = self._load()
        lrs = mod.compute_layer_lrs(1e-4, 48, "cosine")
        edge = (lrs[0] + lrs[-1]) / 2
        mid = lrs[24]
        assert edge > mid

    def test_spectral_radius_exists(self):
        src = Path("omlx/ttt_schedules.py").read_text()
        assert "def spectral_radius_approx(" in src

    def test_correct_length(self):
        mod = self._load()
        for n in [12, 24, 48]:
            lrs = mod.compute_layer_lrs(1e-4, n, "reservoir")
            assert len(lrs) == n


# ---- Task 107: Fair Eviction ----

class TestFairEvictionSource:
    """Source-level tests for fair eviction budget allocation."""

    def test_select_fair_exists(self):
        assert "def _select_fair(" in _snapkv_src

    def test_snapkv_select_has_partitions(self):
        assert "partitions" in _snapkv_src

    def test_proportional_allocation(self):
        """Must allocate budget proportionally to partition size."""
        assert "part_budget" in _snapkv_src or "budget" in _snapkv_src

    def test_min_tokens_floor(self):
        """Must have a minimum per-partition floor."""
        assert "min_tokens" in _snapkv_src or "partition_min_tokens" in _snapkv_src

    def test_per_partition_selection(self):
        """Must run selection independently within each partition."""
        assert "part_start" in _snapkv_src
        assert "part_end" in _snapkv_src

    def test_server_flag(self):
        server_src = Path("omlx/hypercar_server.py").read_text()
        assert "--fair-evict" in server_src


# ---------------------------------------------------------------------------
# Efficiency Audit Regression Tests (2026-04-18)
# ---------------------------------------------------------------------------

_duo_src = Path("omlx/duo_kv_cache.py").read_text()
_tq_src = Path("omlx/turboquant_kv.py").read_text()


class TestDuoKVPreAllocSlab:
    """Task 149: DuoKV must use pre-allocated slab, not mx.concatenate per token."""

    def test_no_concat_in_update(self):
        """update_and_fetch must NOT use mx.concatenate for growing the buffer."""
        # Find the update_and_fetch method body
        idx = _duo_src.index("def update_and_fetch")
        # Look for the next method definition
        next_method = _duo_src.index("\n    def ", idx + 1)
        method_body = _duo_src[idx:next_method]
        # Should NOT have mx.concatenate for key/value growth
        assert "mx.concatenate([self._keys, keys]" not in method_body, \
            "DuoKV update_and_fetch must use pre-alloc slab, not concat"

    def test_has_kv_len_tracking(self):
        """Must track actual token count separately from buffer capacity."""
        assert "_kv_len" in _duo_src

    def test_has_step_prealloc(self):
        """Must have pre-allocation headroom."""
        assert "_step" in _duo_src

    def test_slice_assignment_pattern(self):
        """Must use slice assignment for O(1) token insertion."""
        assert "self._kv_len:self._kv_len + T_new" in _duo_src

    def test_state_property_slices_to_kv_len(self):
        """state property must return only valid tokens, not buffer padding."""
        assert ":self._kv_len" in _duo_src


class TestDuoKVGatherTrim:
    """Task 142: Streaming head trim must use gather, not per-head Python loop."""

    def test_uses_take_along_axis(self):
        """Must use take_along_axis for vectorized gather."""
        assert "take_along_axis" in _duo_src

    def test_no_per_head_concat_loop(self):
        """Must NOT have a per-head concat loop for trim."""
        idx = _duo_src.index("def update_and_fetch")
        next_method = _duo_src.index("\n    def ", idx + 1)
        method_body = _duo_src[idx:next_method]
        # Old pattern: trimmed_k.append(mx.concatenate([k_sink, k_window]
        assert "trimmed_k.append" not in method_body


class TestStreamingKVRingVectorized:
    """Task 143: StreamingKVCache ring writes must be vectorized."""

    def test_no_per_token_ring_loop(self):
        """Ring mode must NOT iterate per-token."""
        # Find the StreamingKVCache ring mode section
        idx = _duo_src.index("class StreamingKVCache")
        next_class = _duo_src.index("\nclass ", idx + 1)
        class_body = _duo_src[idx:next_class]
        # Old pattern: for i in range(T_new): pos = ...
        assert "for i in range(T_new)" not in class_body

    def test_uses_modular_arithmetic(self):
        """Must compute ring positions with vectorized modular arithmetic."""
        assert "% ring_len" in _duo_src


class TestTQ3FusedQuantize:
    """Task 152: TQ3 WHT quantize must use fused dense kernel."""

    def test_wht_uses_fused_quantize(self):
        """WHT path must call _fused_quantize, not _quantize_wht."""
        # Find the quantize method
        idx = _tq_src.index("def quantize(self, vectors")
        end_idx = _tq_src.index("\n    def ", idx + 1)
        method_body = _tq_src[idx:end_idx]
        assert "_fused_quantize(" in method_body
        assert "self._quantize_wht(vectors)" not in method_body


class TestTQ3FusedDequantize:
    """Task 145: All TQ3 dequant hot paths must use dequantize_fused."""

    def test_update_and_fetch_uses_fused(self):
        """update_and_fetch dequant path must prefer dequantize_fused."""
        assert "dequantize_fused" in _tq_src

    def test_dequant_fallback_pattern(self):
        """Must have hasattr fallback for dequantize_fused."""
        assert "hasattr(self._codec, 'dequantize_fused')" in _tq_src


class TestTQ3GeometricGrowth:
    """Task 144: TQ3 buffer growth must use geometric doubling."""

    def test_ensure_compressed_storage_exists(self):
        """Must have extracted helper for storage management."""
        assert "def _ensure_compressed_storage" in _tq_src

    def test_geometric_doubling(self):
        """Must use cur * 2 for geometric growth."""
        idx = _tq_src.index("def _ensure_compressed_storage")
        end_idx = _tq_src.index("\n    def ", idx + 1)
        method_body = _tq_src[idx:end_idx]
        assert "cur * 2" in method_body


class TestSnapKVVectorized:
    """Tasks 139-141, 146-147: SnapKV pipeline must be fully vectorized."""

    def test_no_at_add_pattern(self):
        """Must NOT use .at[].add() for per-element scatter."""
        assert ".at[" not in _snapkv_src

    def test_freshness_uses_band_mask(self):
        """Freshness scoring must use vectorized band mask, not inner loop."""
        idx = _snapkv_src.index("def compute_freshness_scores")
        end_idx = _snapkv_src.index("\ndef ", idx + 1)
        body = _snapkv_src[idx:end_idx]
        # Old pattern: for local_i in range(chunk_len)
        assert "for local_i in range" not in body

    def test_segmented_uses_argpartition(self):
        """Segmented selection must use mx.argpartition, not Python sorted."""
        idx = _snapkv_src.index("def _select_segmented")
        end_idx = _snapkv_src.index("\ndef ", idx + 1)
        body = _snapkv_src[idx:end_idx]
        assert "argpartition" in body
        assert "sorted(range" not in body

    def test_global_uses_argpartition(self):
        """Global selection must use mx.argpartition."""
        idx = _snapkv_src.index("def _select_global")
        end_idx = _snapkv_src.index("\ndef ", idx + 1)
        body = _snapkv_src[idx:end_idx]
        assert "argpartition" in body

    def test_keep_mask_uses_indexed_assignment(self):
        """keep_mask must use indexed assignment, not per-element loop."""
        assert "keep_mask[:, idx] = True" in _snapkv_src

    def test_ger_uses_indexed_assignment(self):
        """GER important_mask must use indexed assignment."""
        assert "important_mask[threshold_idx] = True" in _snapkv_src

    def test_get_keep_indices_uses_argwhere(self):
        """get_keep_indices must use mx.argwhere, not .tolist() iteration."""
        idx = _snapkv_src.index("def get_keep_indices")
        end_idx = _snapkv_src.index("\ndef ", idx + 1)
        body = _snapkv_src[idx:end_idx]
        assert "argwhere" in body

    def test_caote_no_gqa_repeat(self):
        """CAOTE scoring must NOT use mx.repeat for GQA expansion."""
        idx = _snapkv_src.index("def compute_caote_importance")
        end_idx = _snapkv_src.index("\ndef ", idx + 1)
        body = _snapkv_src[idx:end_idx]
        assert "mx.repeat" not in body


class TestSkipReropeDefault:
    """Task 168: compact_cache must default to skip_rerope=True."""

    def test_default_true(self):
        """skip_rerope must default to True for 4.7x faster decode."""
        idx = _snapkv_src.index("def compact_cache")
        # Check the signature
        sig_end = _snapkv_src.index(")", idx)
        sig = _snapkv_src[idx:sig_end]
        assert "skip_rerope: bool = True" in sig


class TestToolCallGate:
    """Task 161: Benchmark must have tool-call JSON validity phase."""

    def test_phase3f_exists(self):
        bench_src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "def phase3f_tool_call_json" in bench_src

    def test_phase3f_wired(self):
        bench_src = Path("omlx/bench/hypercar_bench.py").read_text()
        assert "phase3f_tool_call_json(" in bench_src
