# SPDX-License-Identifier: Apache-2.0
"""Lock-in tests for `omlx/bench/hypercar_bench.py` helper functions added
during the 2026-05-02 Qwen3.6 bench bring-up.

Two helpers, three behaviors each:
  - ``_format_chat_prompt`` with ``enable_thinking=False`` on a tokenizer
    that supports the kwarg (Qwen3.6-style).
  - The same helper falling back to plain ``apply_chat_template`` when the
    tokenizer raises ``TypeError`` on the kwarg (older non-thinking models).
  - The same helper falling back to ``user_message + fallback_suffix`` when
    the tokenizer has no chat template at all.
"""

import pytest


class _ChatTemplateTokenizer:
    """Mock tokenizer that supports ``enable_thinking`` like Qwen3.6."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=True):
        assert tokenize is False
        assert add_generation_prompt is True
        body = messages[0]["content"]
        if enable_thinking:
            return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
        return (f"<|im_start|>user\n{body}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")


class _LegacyChatTokenizer:
    """Mock tokenizer that has chat template but rejects thinking kwarg."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"USER: {messages[0]['content']}\nASSISTANT: "


class _NoTemplateTokenizer:
    """Mock tokenizer with no working chat template at all."""

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("no chat template configured")


def _import_helper():
    # Import here so the import error from a missing module fails the test
    # rather than the collection.
    from omlx.bench.hypercar_bench import _format_chat_prompt
    return _format_chat_prompt


def test_format_chat_prompt_thinking_off_qwen36():
    fmt = _import_helper()
    tok = _ChatTemplateTokenizer()
    out = fmt(tok, "What is 2+2?", enable_thinking=False)
    assert "<think>\n\n</think>" in out
    assert "What is 2+2?" in out


def test_format_chat_prompt_thinking_on_qwen36():
    fmt = _import_helper()
    tok = _ChatTemplateTokenizer()
    out = fmt(tok, "What is 2+2?", enable_thinking=True)
    assert "<think>" not in out  # template did not pre-fill
    assert "What is 2+2?" in out


def test_format_chat_prompt_legacy_template_falls_back_silently():
    fmt = _import_helper()
    tok = _LegacyChatTokenizer()
    # Should not raise even with enable_thinking=False — falls back to
    # bare apply_chat_template.
    out = fmt(tok, "hello", enable_thinking=False)
    assert out == "USER: hello\nASSISTANT: "


def test_format_chat_prompt_no_template_returns_user_msg_plus_suffix():
    fmt = _import_helper()
    tok = _NoTemplateTokenizer()
    out = fmt(tok, "hi", enable_thinking=False, fallback_suffix="\n---\n")
    assert out == "hi\n---\n"


def test_format_chat_prompt_default_fallback_suffix_is_newline():
    fmt = _import_helper()
    tok = _NoTemplateTokenizer()
    out = fmt(tok, "hi", enable_thinking=False)
    assert out == "hi\n"


def test_format_short_answer_alias_routes_through_main_helper():
    from omlx.bench.hypercar_bench import _format_short_answer_prompt
    tok = _ChatTemplateTokenizer()
    out = _format_short_answer_prompt(tok, "Q?")
    # The alias must use enable_thinking=False — verified by the pre-fill.
    assert "<think>\n\n</think>" in out


# ---------------------------------------------------------------------------
# _eos_token_ids: bench-wide stop-token collection
# ---------------------------------------------------------------------------

class _SingleEosTokenizer:
    """Old-style tokenizer with one int EOS (Qwen3-Coder-style)."""
    eos_token_id = 151645


class _MultiEosTokenizer:
    """Thinking-model tokenizer (Qwen3.6) with both attrs populated."""
    eos_token_id = 248046           # <|im_end|>
    eos_token_ids = {248044, 248046}  # <|endoftext|>, <|im_end|>


class _NoEosTokenizer:
    pass


def test_eos_token_ids_single():
    from omlx.bench.hypercar_bench import _eos_token_ids
    assert _eos_token_ids(_SingleEosTokenizer()) == {151645}


def test_eos_token_ids_multi_unifies_both_attrs():
    from omlx.bench.hypercar_bench import _eos_token_ids
    # Set form must be flattened, and singular attr is also included
    # (idempotent overlap).
    assert _eos_token_ids(_MultiEosTokenizer()) == {248044, 248046}


def test_eos_token_ids_missing_returns_empty_set():
    from omlx.bench.hypercar_bench import _eos_token_ids
    out = _eos_token_ids(_NoEosTokenizer())
    assert out == set()


def test_eos_token_ids_accepts_list_or_tuple():
    from omlx.bench.hypercar_bench import _eos_token_ids

    class _T:
        eos_token_ids = [1, 2, 3]

    assert _eos_token_ids(_T()) == {1, 2, 3}


# ---------------------------------------------------------------------------
# _project_prefill_memory_gb: hybrid-aware memory headroom projection
# ---------------------------------------------------------------------------

class _DenseAttn:
    """Old-style attention module (Qwen3-Coder) with n_heads/n_kv_heads."""
    n_heads = 32
    n_kv_heads = 4
    head_dim = 128


class _HybridAttn:
    """Thinking-model attention (Qwen3.6) with num_attention_heads/num_key_value_heads."""
    num_attention_heads = 16
    num_key_value_heads = 2
    head_dim = 256


class _SSMLayer:
    """Linear/SSM layer — no self_attn."""
    is_linear = True


class _AttnLayer:
    def __init__(self, attn):
        self.self_attn = attn


class _DenseModel:
    """All 48 layers attention (Qwen3-Coder shape)."""
    def __init__(self):
        self.layers = [_AttnLayer(_DenseAttn()) for _ in range(48)]


class _HybridModel:
    """40 layers, every 4th is attention (Qwen3.6 shape: 30 SSM + 10 attn)."""
    def __init__(self):
        self.layers = []
        for i in range(40):
            if (i + 1) % 4 == 0:
                self.layers.append(_AttnLayer(_HybridAttn()))
            else:
                self.layers.append(_SSMLayer())


def test_project_dense_uses_all_layers():
    """Dense model: projector must count every layer as KV-bearing."""
    import omlx.bench.hypercar_bench as hb
    hb._KV_MODE = "fp16"
    proj = hb._project_prefill_memory_gb(64 * 1024, _DenseModel())
    # 48 attn × 4 kv × 128 head × 64K × 2(K+V) × 2(fp16) = ~6.3 GB
    # Plus attention transient term and 1.5x safety. Should be in the 10-20 GB range.
    assert 5 < proj < 30, f"dense projection out of range: {proj} GB"


def test_project_hybrid_skips_ssm_layers():
    """Hybrid model: only 25% of layers carry KV (10 of 40)."""
    import omlx.bench.hypercar_bench as hb
    hb._KV_MODE = "fp16"
    proj_hybrid = hb._project_prefill_memory_gb(64 * 1024, _HybridModel())
    proj_dense = hb._project_prefill_memory_gb(64 * 1024, _DenseModel())
    # The KV term should be smaller on the hybrid model — both because it
    # has 10 attn layers vs 48 AND because the per-layer config differs. The
    # attention-transient term is shared across both, so the ratio isn't 4.8x
    # but the hybrid must not exceed the dense projection.
    assert proj_hybrid <= proj_dense, (
        f"hybrid {proj_hybrid:.1f} should not exceed dense {proj_dense:.1f}")


def test_project_hybrid_reads_qwen36_attribute_names():
    """Hybrid attn uses ``num_attention_heads`` / ``num_key_value_heads``;
    projector must not silently fall back to defaults (which would be ~2x
    too high for Qwen3.6's 16/2 vs the 32/4 default fallback)."""
    import omlx.bench.hypercar_bench as hb
    hb._KV_MODE = "native"
    hb.KV_BITS = 4

    # Build two equivalent hybrid models, one with the Qwen3.6 attr names
    # and one with the Qwen3-Coder names, sized to the same KV footprint.
    class _Qwen36Style:
        num_attention_heads = 16
        num_key_value_heads = 2
        head_dim = 256

    class _Qwen3CoderStyle:
        n_heads = 16
        n_kv_heads = 2
        head_dim = 256

    class _M:
        def __init__(self, attn):
            self.layers = [_AttnLayer(attn) for _ in range(10)]

    proj_36 = hb._project_prefill_memory_gb(64 * 1024, _M(_Qwen36Style()))
    proj_coder = hb._project_prefill_memory_gb(64 * 1024, _M(_Qwen3CoderStyle()))
    assert abs(proj_36 - proj_coder) < 0.1, (
        f"Qwen3.6 ({proj_36:.2f}) and Qwen3-Coder ({proj_coder:.2f}) "
        "attr conventions should yield same projection")


def test_project_kv_mode_compresses_kv():
    """native int4 must project less than fp16 (smaller bytes_per_kv_elem)."""
    import omlx.bench.hypercar_bench as hb
    model = _HybridModel()

    hb._KV_MODE = "fp16"
    proj_fp16 = hb._project_prefill_memory_gb(256 * 1024, model)

    hb._KV_MODE = "native"
    hb.KV_BITS = 4
    proj_int4 = hb._project_prefill_memory_gb(256 * 1024, model)

    assert proj_int4 < proj_fp16, (
        f"int4 ({proj_int4:.1f}) should be smaller than fp16 ({proj_fp16:.1f})")
