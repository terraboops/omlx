# SPDX-License-Identifier: Apache-2.0
"""Lock-in tests for ``omlx/state_space/hybrid_layers.py``.

The ``attention_layer_indices`` helper is the seam used by every tool
that monkey-patches SDPA for per-layer dispatch (DuoAttention calibration,
MInference calibration, TTT-Linear capture). It must correctly identify
attention-bearing layers across both dense (Qwen3-Coder) and hybrid
(Qwen3.6) layouts, otherwise modulo-counter dispatch silently misses on
the hybrid path.
"""

from omlx.state_space.hybrid_layers import attention_layer_indices


class _StubLayer:
    def __init__(self, has_attn: bool):
        if has_attn:
            self.self_attn = object()


class _StubDenseModel:
    """Mimics Qwen3-Coder's all-attention layout."""

    def __init__(self, n_layers: int):
        self.layers = [_StubLayer(has_attn=True) for _ in range(n_layers)]


class _StubHybridModel:
    """Mimics Qwen3.6's hybrid layout (every-Nth-layer is attention)."""

    def __init__(self, n_layers: int, full_attention_interval: int):
        self.layers = [
            _StubLayer(has_attn=((i + 1) % full_attention_interval == 0))
            for i in range(n_layers)
        ]


def test_dense_model_returns_full_range():
    """Dense Qwen3-Coder: every index is an attention layer."""
    model = _StubDenseModel(n_layers=48)
    assert attention_layer_indices(model) == list(range(48))


def test_hybrid_qwen36_pattern():
    """Qwen3.6 hybrid: 40 layers / interval=4 → attention at
    [3, 7, 11, ..., 39]. This is the load-bearing case that pre-fix
    code silently mis-handled."""
    model = _StubHybridModel(n_layers=40, full_attention_interval=4)
    indices = attention_layer_indices(model)
    assert indices == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    assert len(indices) == 10


def test_layers_without_self_attn_attribute_excluded():
    """Layers entirely missing the ``self_attn`` attribute (not just
    ``self_attn = None``) must be treated as non-attention."""
    class _BareLayer:
        pass

    class _MixedModel:
        layers = [_StubLayer(has_attn=True), _BareLayer(),
                  _StubLayer(has_attn=True)]

    assert attention_layer_indices(_MixedModel()) == [0, 2]


def test_self_attn_set_to_none_excluded():
    """A layer with ``self_attn = None`` (which the SSM-layer scaffolding
    might leave) is not an attention layer."""
    layer = _StubLayer(has_attn=False)
    layer.self_attn = None  # type: ignore[attr-defined]

    class _Model:
        layers = [layer, _StubLayer(has_attn=True)]

    assert attention_layer_indices(_Model()) == [1]


def test_empty_model_returns_empty_list():
    class _Empty:
        layers = []
    assert attention_layer_indices(_Empty()) == []


def test_returns_list_of_python_ints():
    """Type pin: callers depend on `.index()` lookup and JSON-serialization
    of the result, both of which require Python ints (not numpy/mlx)."""
    model = _StubHybridModel(n_layers=8, full_attention_interval=2)
    indices = attention_layer_indices(model)
    assert all(type(i) is int for i in indices)
