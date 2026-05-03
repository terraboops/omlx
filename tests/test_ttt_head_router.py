# SPDX-License-Identifier: Apache-2.0
"""Lock-in tests for ``omlx/patches/ttt_head_router.py`` (Task 388
Phase 2 cycle 1).

The load-bearing property pinned here is **bit-equivalence in
passthrough mode**: if a router is constructed with no TTT blocks
loaded, every routed SDPA call must produce the same output as
the original SDPA.

Tests in this file don't load any model — they exercise the router
against synthetic Q/K/V tensors and a stub ``original_sdpa``.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from omlx.patches.ttt_head_router import TTTHeadRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """Minimal SDPA implementation for tests — returns a deterministic
    transformation of inputs we can identity-check."""
    # weights @ values, with simple causal-aware scoring.
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
    if mask is not None and isinstance(mask, mx.array):
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ values


def _toy_policy_dict(n_layers=2, n_heads=4, streaming_layer=0,
                     streaming_heads=(1, 3)):
    heads = []
    for L in range(n_layers):
        for H in range(n_heads):
            policy = ("streaming" if (L == streaming_layer and H in streaming_heads)
                      else "retrieval")
            heads.append({
                "layer": L, "head": H, "policy": policy,
                "local_fraction": 0.99 if policy == "streaming" else 0.5,
                "window": 256, "sink": 4,
            })
    return {
        "model": "test", "n_layers": n_layers, "n_heads": n_heads,
        "n_kv_heads": n_heads, "context_len": 1024, "window": 256,
        "sink": 4, "streaming_threshold": 0.85,
        "streaming_fraction": len(streaming_heads) / (n_layers * n_heads),
        "streaming_count": len(streaming_heads),
        "retrieval_count": n_layers * n_heads - len(streaming_heads),
        "heads": heads,
    }


def _make_q_k_v(B=1, H=4, L=8, D=16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, D))
    k = mx.random.normal((B, H, L, D))
    v = mx.random.normal((B, H, L, D))
    return q, k, v


# ---------------------------------------------------------------------------
# Construction + policy parsing
# ---------------------------------------------------------------------------


def test_router_construction_with_explicit_classification():
    classification = {(0, 1): "streaming", (0, 3): "streaming"}
    router = TTTHeadRouter(
        n_layers=2, n_heads=4, head_classification=classification)
    assert router.n_layers == 2
    assert router.n_heads == 4
    assert router.head_classification[(0, 1)] == "streaming"
    assert router.layer_counter == 0
    assert router.ttt_blocks == {}


def test_router_from_policy(tmp_path):
    policy = _toy_policy_dict()
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    router = TTTHeadRouter.from_policy(p)
    assert router.n_layers == 2
    assert router.n_heads == 4
    assert router.head_classification[(0, 1)] == "streaming"
    assert router.head_classification[(0, 0)] == "retrieval"
    assert router.head_classification[(1, 0)] == "retrieval"
    assert router.ttt_blocks == {}    # default to bit-equivalence mode


def test_router_from_policy_accepts_real_qwen3_coder_policy():
    p = Path("omlx/patches/duoattention_policies/"
             "qwen3_coder_30b_a3b_instruct_8bit.json")
    if not p.exists():
        pytest.skip("Qwen3-Coder policy file not present")
    router = TTTHeadRouter.from_policy(p)
    assert router.n_layers == 48
    assert router.n_heads == 32
    # Policy file has "streaming" entries — at least one must show up.
    streaming = [k for k, v in router.head_classification.items()
                 if v == "streaming"]
    assert len(streaming) > 0
    # And ttt_blocks must default to empty (bit-equivalence mode).
    assert router.ttt_blocks == {}


# ---------------------------------------------------------------------------
# streaming_heads_with_ttt
# ---------------------------------------------------------------------------


def test_streaming_heads_with_ttt_empty_when_no_blocks_loaded():
    classification = {(0, 1): "streaming", (0, 3): "streaming"}
    router = TTTHeadRouter(n_layers=2, n_heads=4,
                           head_classification=classification)
    # No TTT blocks → no head triggers TTT routing.
    assert router.streaming_heads_with_ttt(layer_idx=0) == []
    assert router.streaming_heads_with_ttt(layer_idx=1) == []


def test_streaming_heads_with_ttt_filters_to_streaming_AND_loaded():
    """Loaded TTT for a retrieval head must NOT trigger TTT routing —
    that would silently change retrieval semantics."""
    classification = {
        (0, 0): "retrieval",   # retrieval head with TTT loaded → ignored
        (0, 1): "streaming",   # streaming, has TTT     → routes to TTT
        (0, 2): "streaming",   # streaming, no TTT      → falls through
        (0, 3): "retrieval",
    }
    blocks = {(0, 0): "block_a", (0, 1): "block_b"}
    router = TTTHeadRouter(n_layers=1, n_heads=4,
                           head_classification=classification,
                           ttt_blocks=blocks)
    assert router.streaming_heads_with_ttt(0) == [1]


# ---------------------------------------------------------------------------
# route_sdpa: the bit-equivalence load-bearing property
# ---------------------------------------------------------------------------


def test_route_sdpa_bit_equivalence_with_no_ttt_blocks():
    """Empty ttt_blocks → output bit-matches original SDPA on every call."""
    classification = {(0, 1): "streaming", (0, 3): "streaming"}
    router = TTTHeadRouter(n_layers=2, n_heads=4,
                           head_classification=classification)
    q, k, v = _make_q_k_v()
    expected = _stub_sdpa(q, k, v, cache=None, scale=0.25, mask=None)
    actual = router.route_sdpa(
        q, k, v, cache=None, scale=0.25, mask=None,
        original_sdpa=_stub_sdpa,
    )
    diff = mx.max(mx.abs(actual - expected)).item()
    assert diff == 0.0, f"bit-equivalence broken at diff={diff}"


def test_route_sdpa_bit_equivalence_when_streaming_classified_but_no_ttt():
    """Streaming-classified heads with NO TTT block loaded → still
    bit-equivalent. This is the Phase 3 starting point: install the
    router with the policy file, but with zero TTT weights → no
    behavior change."""
    classification = {(0, 1): "streaming", (1, 2): "streaming"}
    router = TTTHeadRouter(n_layers=2, n_heads=4,
                           head_classification=classification)
    q, k, v = _make_q_k_v(seed=7)
    expected = _stub_sdpa(q, k, v, None, 0.5, None)
    # Call twice (one per layer).
    actual_l0 = router.route_sdpa(q, k, v, None, 0.5, None,
                                  original_sdpa=_stub_sdpa)
    actual_l1 = router.route_sdpa(q, k, v, None, 0.5, None,
                                  original_sdpa=_stub_sdpa)
    assert mx.max(mx.abs(actual_l0 - expected)).item() == 0.0
    assert mx.max(mx.abs(actual_l1 - expected)).item() == 0.0


def test_route_sdpa_increments_layer_counter():
    router = TTTHeadRouter(n_layers=3, n_heads=2)
    q, k, v = _make_q_k_v(H=2)
    assert router.layer_counter == 0
    router.route_sdpa(q, k, v, None, 1.0, None, original_sdpa=_stub_sdpa)
    assert router.layer_counter == 1
    router.route_sdpa(q, k, v, None, 1.0, None, original_sdpa=_stub_sdpa)
    assert router.layer_counter == 2
    router.route_sdpa(q, k, v, None, 1.0, None, original_sdpa=_stub_sdpa)
    assert router.layer_counter == 3
    # Doesn't auto-wrap; modulo is applied at dispatch time.
    router.route_sdpa(q, k, v, None, 1.0, None, original_sdpa=_stub_sdpa)
    assert router.layer_counter == 4


def test_layer_counter_modulo_handles_wraparound():
    """When more SDPA calls happen than n_layers, the modulo decides
    which (layer, head) classifications apply. This pin guards against
    a future regression where someone changes the dispatch path to use
    raw layer_counter without the modulo."""
    router = TTTHeadRouter(n_layers=2, n_heads=4,
                           head_classification={(0, 1): "streaming"})
    q, k, v = _make_q_k_v()
    expected = _stub_sdpa(q, k, v, None, 1.0, None)
    # 5 calls: layer indices 0, 1, 0, 1, 0 (mod 2).
    for _ in range(5):
        out = router.route_sdpa(q, k, v, None, 1.0, None,
                                original_sdpa=_stub_sdpa)
        assert mx.max(mx.abs(out - expected)).item() == 0.0
    assert router.layer_counter == 5


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------


def test_reset_state_clears_layer_counter_and_states():
    router = TTTHeadRouter(n_layers=2, n_heads=4)
    router.layer_counter = 7
    router.ttt_states[(0, 1)] = "stale"
    router.reset_state()
    assert router.layer_counter == 0
    assert router.ttt_states == {}


# ---------------------------------------------------------------------------
# TTT-routed path (cycle 2 — split-and-reassemble using a TTTLinear instance)
# ---------------------------------------------------------------------------


def _make_ttt_block(D=16, eta=0.05, mini_batch_size=64):
    """Build an untrained TTT-Linear instance for routing tests."""
    from omlx.state_space import TTTLinear, TTTLinearConfig
    cfg = TTTLinearConfig(head_dim=D, eta=eta,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=True)
    return TTTLinear(cfg)


def test_ttt_routed_output_differs_from_original_sdpa_at_routed_head():
    """Loading a TTT block on head 1 must change head 1's output (vs
    bit-equivalence-to-SDPA), confirming the dispatcher actually
    executes the TTT path. Other heads must remain bit-equivalent."""
    D = 16
    H = 4
    classification = {(0, 1): "streaming"}
    blocks = {(0, 1): _make_ttt_block(D=D)}
    router = TTTHeadRouter(n_layers=1, n_heads=H,
                           head_classification=classification,
                           ttt_blocks=blocks)
    q, k, v = _make_q_k_v(H=H, L=8, D=D, seed=42)

    plain = _stub_sdpa(q, k, v, None, 1.0, None)
    routed = router.route_sdpa(q, k, v, None, 1.0, None,
                               original_sdpa=_stub_sdpa)

    # Head 1: TTT-routed → must differ.
    diff_routed = mx.max(mx.abs(routed[:, 1, :, :] - plain[:, 1, :, :])).item()
    assert diff_routed > 1e-3, (
        f"head 1 should be TTT-routed (different from SDPA). diff={diff_routed}"
    )

    # Heads 0, 2, 3: not routed → must bit-match SDPA.
    for h in (0, 2, 3):
        diff_h = mx.max(mx.abs(routed[:, h, :, :] - plain[:, h, :, :])).item()
        assert diff_h == 0.0, (
            f"head {h} not TTT-routed but diverged from SDPA. diff={diff_h}"
        )


def test_ttt_routed_head_matches_direct_ttt_call_on_query_slice():
    """The TTT-routed head's output slot must equal the TTTLinear block
    called directly on Q_h (with no prior state). Pins the dispatcher's
    input-passing contract."""
    D = 16
    H = 4
    ttt = _make_ttt_block(D=D)
    classification = {(0, 1): "streaming"}
    blocks = {(0, 1): ttt}
    router = TTTHeadRouter(n_layers=1, n_heads=H,
                           head_classification=classification,
                           ttt_blocks=blocks)
    q, k, v = _make_q_k_v(H=H, L=8, D=D, seed=11)

    out = router.route_sdpa(q, k, v, None, 1.0, None,
                            original_sdpa=_stub_sdpa)

    # Reference: call the SAME ttt block directly on Q for head 1, fresh state.
    o_ref, _ = ttt(q[:, 1, :, :])
    diff = mx.max(mx.abs(out[:, 1, :, :] - o_ref)).item()
    assert diff < 1e-5, f"routed slot diverges from direct TTT call: {diff}"


def test_ttt_state_threads_across_calls():
    """Two route calls with state threading must equal one route call on
    the concatenated input — confirms ``self.ttt_states`` correctly
    persists the TTT recurrence across SDPA invocations."""
    D = 8
    H = 2
    L_chunk = 4
    classification = {(0, 0): "streaming"}
    ttt = _make_ttt_block(D=D, mini_batch_size=L_chunk)
    blocks = {(0, 0): ttt}
    router = TTTHeadRouter(n_layers=1, n_heads=H,
                           head_classification=classification,
                           ttt_blocks=blocks)

    q_full, _, _ = _make_q_k_v(H=H, L=2 * L_chunk, D=D, seed=99)
    k_full = mx.zeros_like(q_full)
    v_full = mx.zeros_like(q_full)
    q_a = q_full[:, :, :L_chunk, :]
    q_b = q_full[:, :, L_chunk:, :]
    k_a, k_b = k_full[:, :, :L_chunk, :], k_full[:, :, L_chunk:, :]
    v_a, v_b = v_full[:, :, :L_chunk, :], v_full[:, :, L_chunk:, :]

    router.reset_state()
    out_a = router.route_sdpa(q_a, k_a, v_a, None, 1.0, None,
                              original_sdpa=_stub_sdpa)
    out_b = router.route_sdpa(q_b, k_b, v_b, None, 1.0, None,
                              original_sdpa=_stub_sdpa)
    routed_combined = mx.concatenate([out_a, out_b], axis=2)

    # Reference: the same ttt block called once on the full sequence.
    o_ref, _ = ttt(q_full[:, 0, :, :])
    diff = mx.max(mx.abs(routed_combined[:, 0, :, :] - o_ref)).item()
    assert diff < 1e-4, f"state-threading diverges from single call: {diff}"


def test_reset_state_actually_resets_ttt_recurrence():
    """After ``reset_state()``, replaying the same input must produce
    the same output as the first call — i.e., the state cache really
    cleared, leaving TTT to re-init from scratch."""
    D = 8
    H = 2
    classification = {(0, 0): "streaming"}
    ttt = _make_ttt_block(D=D, mini_batch_size=4)
    blocks = {(0, 0): ttt}
    router = TTTHeadRouter(n_layers=1, n_heads=H,
                           head_classification=classification,
                           ttt_blocks=blocks)
    q, k, v = _make_q_k_v(H=H, L=4, D=D, seed=7)

    out_first = router.route_sdpa(q, k, v, None, 1.0, None,
                                  original_sdpa=_stub_sdpa)
    router.reset_state()
    out_after_reset = router.route_sdpa(q, k, v, None, 1.0, None,
                                        original_sdpa=_stub_sdpa)

    diff = mx.max(mx.abs(out_first - out_after_reset)).item()
    assert diff == 0.0, (
        f"reset_state should restore equivalent input to same output. "
        f"diff={diff}"
    )


# ---------------------------------------------------------------------------
# Persistence — save_blocks / load_blocks round-trip
# ---------------------------------------------------------------------------


def test_save_load_blocks_roundtrip_preserves_outputs(tmp_path):
    """Saving a dict of TTT blocks and reloading must produce blocks
    that, when called on the same input, return bit-identical outputs."""
    from omlx.patches.ttt_head_router import save_blocks, load_blocks

    D = 8
    block_a = _make_ttt_block(D=D, mini_batch_size=4)
    block_b = _make_ttt_block(D=D, mini_batch_size=4)
    blocks = {(0, 1): block_a, (5, 12): block_b}

    save_blocks(blocks, tmp_path)
    # Files: manifest + 2 safetensors + 2 sidecar JSONs.
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "manifest.json" in files
    assert "L0_H1.safetensors" in files
    assert "L0_H1.json" in files
    assert "L5_H12.safetensors" in files
    assert "L5_H12.json" in files

    loaded = load_blocks(tmp_path)
    assert set(loaded.keys()) == set(blocks.keys())

    x = mx.random.normal((1, 4, D), key=mx.random.key(0))
    for key in blocks:
        out_orig, _ = blocks[key](x)
        out_loaded, _ = loaded[key](x)
        diff = mx.max(mx.abs(out_orig - out_loaded)).item()
        assert diff == 0.0, f"key {key} not bit-identical: diff={diff}"


def test_save_blocks_manifest_format(tmp_path):
    """Manifest schema is the on-disk contract Phase 1 will load
    against; pin it explicitly so we notice a breaking change."""
    from omlx.patches.ttt_head_router import save_blocks

    D = 4
    blocks = {(2, 7): _make_ttt_block(D=D, mini_batch_size=2)}
    save_blocks(blocks, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["n_blocks"] == 1
    assert manifest["blocks"] == [
        {"layer": 2, "head": 7, "stem": "L2_H7"}
    ]


def test_load_blocks_sorted_keys_independent_of_save_order(tmp_path):
    """Manifest is written sorted by (layer, head) so two save_blocks
    calls with different dict iteration orders produce identical files
    (helps reproducibility + git diffs)."""
    from omlx.patches.ttt_head_router import save_blocks

    D = 4
    block_a = _make_ttt_block(D=D, mini_batch_size=2)
    block_b = _make_ttt_block(D=D, mini_batch_size=2)
    block_c = _make_ttt_block(D=D, mini_batch_size=2)
    # Insert in reverse "expected" order
    blocks = {(5, 1): block_a, (0, 0): block_b, (3, 8): block_c}
    save_blocks(blocks, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    layers_heads = [(b["layer"], b["head"]) for b in manifest["blocks"]]
    assert layers_heads == [(0, 0), (3, 8), (5, 1)]


def test_from_policy_loads_ttt_blocks_from_directory(tmp_path):
    """`from_policy(ttt_dir=...)` is the production path — loads blocks
    from a Phase 1 distillation directory and routes through them."""
    from omlx.patches.ttt_head_router import save_blocks

    # Save one block keyed (0, 1).
    D = 8
    block = _make_ttt_block(D=D, mini_batch_size=2)
    save_blocks({(0, 1): block}, tmp_path / "blocks")

    # Build a policy file marking (0, 1) as streaming.
    policy = _toy_policy_dict(streaming_layer=0, streaming_heads=(1,))
    (tmp_path / "policy.json").write_text(json.dumps(policy))

    router = TTTHeadRouter.from_policy(
        tmp_path / "policy.json", ttt_dir=tmp_path / "blocks",
    )
    assert (0, 1) in router.ttt_blocks
    assert router.head_classification[(0, 1)] == "streaming"

    # Bit-identity to a router built with the in-memory block.
    q, k, v = _make_q_k_v(H=4, L=4, D=D, seed=5)
    out_loaded = router.route_sdpa(q, k, v, None, 1.0, None,
                                   original_sdpa=_stub_sdpa)
    expected_router = TTTHeadRouter.from_policy(
        tmp_path / "policy.json", ttt_blocks={(0, 1): block},
    )
    out_inmem = expected_router.route_sdpa(q, k, v, None, 1.0, None,
                                           original_sdpa=_stub_sdpa)
    diff = mx.max(mx.abs(out_loaded - out_inmem)).item()
    assert diff == 0.0, f"loaded vs in-memory differ at diff={diff}"


def test_from_policy_rejects_both_blocks_and_dir(tmp_path):
    """Specifying both ttt_blocks and ttt_dir is an error — ambiguous."""
    policy = _toy_policy_dict()
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="not both"):
        TTTHeadRouter.from_policy(
            tmp_path / "policy.json",
            ttt_blocks={(0, 1): "stub"},
            ttt_dir=tmp_path / "blocks",
        )


def test_ttt_state_persists_across_layer_dispatch_within_one_chunk():
    """Different (layer, head) keys must NOT interfere — head (0,0) and
    head (1,0) maintain independent ``self.ttt_states`` entries."""
    D = 8
    H = 2
    classification = {(0, 0): "streaming", (1, 0): "streaming"}
    ttt0 = _make_ttt_block(D=D, mini_batch_size=4)
    ttt1 = _make_ttt_block(D=D, mini_batch_size=4)
    blocks = {(0, 0): ttt0, (1, 0): ttt1}
    router = TTTHeadRouter(n_layers=2, n_heads=H,
                           head_classification=classification,
                           ttt_blocks=blocks)
    q, k, v = _make_q_k_v(H=H, L=4, D=D, seed=3)

    router.reset_state()
    router.route_sdpa(q, k, v, None, 1.0, None, original_sdpa=_stub_sdpa)
    router.route_sdpa(q, k, v, None, 1.0, None, original_sdpa=_stub_sdpa)

    # Both keys should now have populated state.
    assert (0, 0) in router.ttt_states
    assert (1, 0) in router.ttt_states
    # And they must differ — different TTT blocks, different states.
    s0 = router.ttt_states[(0, 0)]
    s1 = router.ttt_states[(1, 0)]
    diff = mx.max(mx.abs(s0 - s1)).item()
    assert diff > 1e-6, (
        f"per-(layer,head) state collision: states identical at diff={diff}"
    )
