# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the speculative-decoding integration in
`omlx/hypercar_server.py` (Task 348 — server-side scaffolding for
Task 341's spec-decoding lever).

These tests validate the CLI surface, helper signature changes, and
the runtime invariants of the scaffolding — without loading the 17 GB
production model OR a drafter. The actual decode-time behavior is
validated end-to-end when the user invokes Task 341's
`probe_speculative_decoding.py` against real models.

What this file guards:
- `--draft-model` and `--num-draft-tokens` CLI flags exist
- `apply_progress_logging` accepts `draft_model` and `num_draft_tokens`
  keyword args
- The server defaults to spec-decode-OFF (drafter must be opt-in)
- Recommended drafter is documented in the CLI help text (so a future
  cycle can't drift the recommendation silently)
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parent.parent.resolve()
_SERVER_PATH = _REPO_ROOT / "omlx" / "hypercar_server.py"


# ---- CLI argparse contract ------------------------------------------------

@pytest.fixture(scope="module")
def server_help() -> str:
    """`hypercar_server.py --help` output.

    Loads the server module via subprocess so we can inspect the
    rendered argparse without executing any post-args code that would
    require the model to load.
    """
    rc = subprocess.run(
        [sys.executable, "-m", "omlx.hypercar_server", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert rc.returncode == 0, f"--help failed: {rc.stderr}"
    return rc.stdout


def test_draft_model_flag_present(server_help):
    """The `--draft-model` flag must appear in --help output. If a future
    refactor removes it, the spec-decode lever loses its server entry
    point."""
    assert "--draft-model" in server_help


def test_num_draft_tokens_flag_present(server_help):
    """`--num-draft-tokens` controls the depth of speculation per main-
    model verification. Without this flag, the user is locked to mlx_lm's
    default (2), which doesn't match Task 341's recommended N=4 for
    α≥0.5."""
    assert "--num-draft-tokens" in server_help


def test_default_num_draft_tokens_is_4(server_help):
    """Task 341's predicted-speedup math identifies N=4 as a reasonable
    default at α≈0.5 (predicted ~2.27× speedup). Drift away from 4
    without re-running the probe would be a silent change to the
    advertised performance contract."""
    assert "default 4" in server_help.lower() or "default: 4" in server_help


def test_recommended_drafter_in_help(server_help):
    """The --draft-model help text should reference the
    UAG-MLX-LM-recommended drafter (`Qwen2.5-Coder-1.5B-Instruct-4bit`).
    If this drifts, the user's first-pass drafter selection has no
    in-tree pointer."""
    # Match flexibly — the help uses kebab-case path
    assert "Qwen2.5-Coder-1.5B" in server_help


def test_default_draft_model_is_none(server_help):
    """spec-decode must default to OFF — the lever is opt-in until the
    user runs Task 341's probe and chooses to enable based on measured α."""
    # Task 355 Cycle 3: argparse declarations moved to omlx/server/cli_args.py.
    cli_args_path = _REPO_ROOT / "omlx" / "server" / "cli_args.py"
    src = cli_args_path.read_text() if cli_args_path.exists() else _SERVER_PATH.read_text()
    # Find the `--draft-model` argparse line
    assert 'parser.add_argument("--draft-model"' in src
    # Default None means opt-in
    assert "default=None" in src.split('parser.add_argument("--draft-model"')[1].split("parser.add_argument")[0]


# ---- Helper signature contract --------------------------------------------

@pytest.fixture(scope="module")
def server_module():
    """Import omlx.hypercar_server WITHOUT running its main()."""
    # The module-level imports are lightweight (argparse, logging); main()
    # is only run via `if __name__ == "__main__"` which doesn't trigger
    # under importlib.
    import importlib
    return importlib.import_module("omlx.hypercar_server")


def test_progress_logging_module_extracted(server_module):
    """Task 355 Cycle 1: apply_progress_logging + validate_drafter_for_request
    were extracted to `omlx/server/progress_logging.py`. Verify:
    1. The new module exists and is importable
    2. Both functions live there now
    3. hypercar_server.py re-exports them for backward compat
    """
    import importlib
    pl = importlib.import_module("omlx.server.progress_logging")
    assert hasattr(pl, "apply_progress_logging")
    assert hasattr(pl, "validate_drafter_for_request")
    # Re-export from hypercar_server.py points at the same objects
    assert server_module.apply_progress_logging is pl.apply_progress_logging
    assert server_module.validate_drafter_for_request is pl.validate_drafter_for_request


def test_patches_module_extracted(server_module):
    """Task 355 Cycle 2: apply_hypercar_patches was extracted to
    `omlx/server/patches.py`. Verify:
    1. The new module exists and is importable
    2. The function lives there now
    3. hypercar_server.py re-exports it for backward compat (existing
       call sites doing `from omlx.hypercar_server import
       apply_hypercar_patches` continue to work).
    """
    import importlib
    pm = importlib.import_module("omlx.server.patches")
    assert hasattr(pm, "apply_hypercar_patches")
    assert callable(pm.apply_hypercar_patches)
    # Re-export from hypercar_server.py points at the same object
    assert server_module.apply_hypercar_patches is pm.apply_hypercar_patches


def test_cli_args_module_extracted():
    """Task 355 Cycle 3: argparse declarations were extracted to
    `omlx/server/cli_args.py`. Verify:
    1. The new module exists and is importable
    2. `build_parser` is the public API
    3. The parser includes all the production flags from main()
       (--model, --kv-mode, --draft-model, --prefill-sparse, etc.)
    """
    import argparse
    import importlib
    cli_mod = importlib.import_module("omlx.server.cli_args")
    assert hasattr(cli_mod, "build_parser")
    assert callable(cli_mod.build_parser)
    parser = cli_mod.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Spot-check a representative flag from each major axis:
    # model loading, KV strategy, decode, prefill sparse, spec-decode.
    flag_strings = parser.format_help()
    for required_flag in (
        "--model",       # model loading
        "--kv-mode",     # KV strategy
        "--bits",        # KV bit-width
        "--snapkv-keep", # eviction
        "--prefill-sparse",  # Goal 4 lever
        "--draft-model", # Goal 3 lever (Task 348)
        "--num-draft-tokens",
        "--ttt-router-policy",      # Task 388 Phase 2 production wiring
        "--ttt-router-blocks-dir",
    ):
        assert required_flag in flag_strings, (
            f"build_parser() missing flag {required_flag!r} — Task 355 "
            f"Cycle 3 extraction must preserve all production flags."
        )


def test_apply_progress_logging_accepts_draft_model(server_module):
    """`apply_progress_logging(draft_model=, num_draft_tokens=)` is the
    integration point. The wrapper around `mlx_lm.stream_generate`
    closes over these parameters to inject `draft_model=...` into every
    request."""
    sig = inspect.signature(server_module.apply_progress_logging)
    assert "draft_model" in sig.parameters
    assert "num_draft_tokens" in sig.parameters


def test_apply_progress_logging_default_is_off(server_module):
    """Default values must keep spec-decode disabled — opt-in pattern."""
    sig = inspect.signature(server_module.apply_progress_logging)
    assert sig.parameters["draft_model"].default is None
    assert sig.parameters["num_draft_tokens"].default == 4


# ---- Runtime invariants on the wrapper -----------------------------------

def test_wrapper_handles_no_drafter(server_module):
    """When `draft_model=None`, the wrapper must be a clean pass-through
    — no `draft_model` injected into kwargs, no spec-decode telemetry.
    The wrapper delegates to `validate_drafter_for_request()` which
    returns `(None, None)` when draft_model is None."""
    # Helper exists and is the integration point
    assert hasattr(server_module, "validate_drafter_for_request")
    assert callable(server_module.validate_drafter_for_request)
    # Pass-through: None drafter → None active_drafter, no warning
    active, warn = server_module.validate_drafter_for_request(
        tokenizer=object(),  # tokenizer never inspected when draft_model is None
        draft_model=None,
    )
    assert active is None
    assert warn is None


def test_wrapper_does_tokenizer_identity_check(server_module):
    """Per-request tokenizer-identity check is the SAFETY net against
    silent wrong-output bugs (mlx_lm doesn't translate token IDs
    between models)."""
    # Task 355 Cycle 1: the wrapper + helper moved to omlx/server/progress_logging.py.
    # Check both that file AND hypercar_server.py (which re-exports the names).
    pl_path = _REPO_ROOT / "omlx" / "server" / "progress_logging.py"
    src = pl_path.read_text() if pl_path.exists() else _SERVER_PATH.read_text()
    assert "Drafter vocab mismatch" in src, (
        "Per-request tokenizer-identity check is missing. Without it, a "
        "drafter with a different vocab size silently produces wrong "
        "outputs — the integration must guard against this."
    )


def test_wrapper_tracks_acceptance_rate(server_module):
    """Acceptance-rate telemetry (`spec α=N%`) is what lets the user
    decide whether to keep spec-decode enabled. Without it, the user
    has no data to validate the lever in production."""
    pl_path = _REPO_ROOT / "omlx" / "server" / "progress_logging.py"
    src = pl_path.read_text() if pl_path.exists() else _SERVER_PATH.read_text()
    # The accumulator
    assert "n_from_draft" in src
    # The reporting
    assert "spec α=" in src or "from_draft" in src


# ---- Cross-task contract (Task 341 probe ↔ Task 348 server) -------------

def test_default_num_draft_tokens_matches_probe_default(server_module):
    """Task 341's probe defaults to --num-draft-tokens=4. The server
    default must match so users can replicate the probe's measured α
    against the server's behavior."""
    sig = inspect.signature(server_module.apply_progress_logging)
    server_default = sig.parameters["num_draft_tokens"].default
    # Read the probe's default to compare
    probe_path = _REPO_ROOT / "scripts" / "probe_speculative_decoding.py"
    probe_src = probe_path.read_text()
    assert "default=4" in probe_src, (
        "Task 341 probe must default to --num-draft-tokens=4 to match "
        "the server's default. If the probe drifts, this test catches it."
    )
    assert server_default == 4


# ---- validate_drafter_for_request unit tests (Task 353) ---------------

class _FakeTokenizer:
    """Minimal tokenizer stub that supports `len()` for vocab-size check."""
    def __init__(self, vocab_size: int):
        self._n = vocab_size

    def __len__(self):
        return self._n


class _FakeDrafterModel:
    """Minimal drafter stub matching the Qwen3MoeForCausalLM shape that
    `validate_drafter_for_request` introspects (`.model.embed_tokens.weight`)."""
    def __init__(self, vocab_size: int):
        # Build a fake `model.embed_tokens.weight` with shape[0] = vocab_size.
        class _EmbedTokens:
            class _W:
                shape = (0, 0)
            def __init__(self, n):
                self.weight = self._W()
                self.weight.shape = (n, 64)
        class _Inner:
            def __init__(self, n):
                self.embed_tokens = _EmbedTokens(n)
        self.model = _Inner(vocab_size)


def test_validate_drafter_returns_none_when_drafter_none(server_module):
    """No drafter configured → no validation needed → returns (None, None)."""
    active, warn = server_module.validate_drafter_for_request(
        tokenizer=_FakeTokenizer(151936),
        draft_model=None,
    )
    assert active is None
    assert warn is None


def test_validate_drafter_returns_drafter_on_vocab_match(server_module):
    """Matching vocab sizes → return drafter with no warning."""
    drafter = _FakeDrafterModel(vocab_size=151936)
    active, warn = server_module.validate_drafter_for_request(
        tokenizer=_FakeTokenizer(151936),
        draft_model=drafter,
    )
    assert active is drafter
    assert warn is None


def test_validate_drafter_drops_on_vocab_mismatch(server_module):
    """Mismatched vocab sizes → return None + warning explaining why."""
    drafter = _FakeDrafterModel(vocab_size=32000)  # different vocab
    active, warn = server_module.validate_drafter_for_request(
        tokenizer=_FakeTokenizer(151936),
        draft_model=drafter,
    )
    assert active is None
    assert warn is not None
    assert "mismatch" in warn.lower()
    assert "151936" in warn
    assert "32000" in warn


def test_validate_drafter_caches_vocab_on_drafter(server_module):
    """First call introspects drafter.model.embed_tokens.weight.shape;
    subsequent calls hit the `_vocab_size_for_check` attr cache (O(1))."""
    drafter = _FakeDrafterModel(vocab_size=151936)
    assert not hasattr(drafter, "_vocab_size_for_check")
    server_module.validate_drafter_for_request(
        tokenizer=_FakeTokenizer(151936), draft_model=drafter,
    )
    assert getattr(drafter, "_vocab_size_for_check", None) == 151936


def test_validate_drafter_uses_cache_on_subsequent_calls(server_module):
    """If `_vocab_size_for_check` is already set, the helper reads it
    directly without inspecting `.model.embed_tokens.weight`. Test by
    setting cache to a different value than the actual embed and
    verifying the cached value is what's returned."""
    drafter = _FakeDrafterModel(vocab_size=151936)
    drafter._vocab_size_for_check = 32000  # pretend cache says different
    active, warn = server_module.validate_drafter_for_request(
        tokenizer=_FakeTokenizer(151936),
        draft_model=drafter,
    )
    # Cached value (32000) is used, not the embed shape (151936)
    # → mismatch detected → drafter dropped
    assert active is None
    assert warn is not None and "32000" in warn


def test_validate_drafter_handles_exception_gracefully(server_module):
    """If something throws during the check (e.g., drafter has no
    `.model` attribute and isn't shape-introspectable), the helper
    returns (None, warning) instead of crashing the request."""
    class _BadDrafter:
        # No `.model`, len(tokenizer) call fails on something weird
        pass

    class _BadTokenizer:
        def __len__(self):
            raise RuntimeError("tokenizer doesn't support len()")

    active, warn = server_module.validate_drafter_for_request(
        tokenizer=_BadTokenizer(),
        draft_model=_BadDrafter(),
    )
    assert active is None
    assert warn is not None
    assert "failed" in warn.lower() or "exception" in warn.lower() or "len" in warn.lower()


# ---- CLAUDE.md user-facing documentation (Task 349) ---------------------

def test_claude_md_documents_speculative_decoding_usage():
    """CLAUDE.md (loaded into every Claude session) must surface the
    spec-decoding usage block. Without this, a cold-start session would
    not know the integration exists and might re-implement it from
    scratch."""
    claude_md = _REPO_ROOT / "CLAUDE.md"
    text = claude_md.read_text()
    # Section header
    assert "Speculative decoding" in text, (
        "CLAUDE.md must include a 'Speculative decoding' section "
        "documenting the Task 348 integration."
    )
    # The actual CLI invocation pattern users would copy-paste
    assert "--draft-model" in text and "Qwen2.5-Coder-1.5B" in text, (
        "CLAUDE.md must include the recommended --draft-model invocation."
    )
    # The α decision tree
    assert "α" in text or "acceptance rate" in text, (
        "CLAUDE.md must explain how to interpret the per-request α "
        "telemetry — without this, users have no decision framework."
    )


# ---------------------------------------------------------------------------
# Task 388 Phase 2 production wiring: --ttt-router-policy / --ttt-router-blocks-dir
# ---------------------------------------------------------------------------


def test_ttt_router_flags_default_to_none():
    """Both flags default to None — server runs without TTT routing
    unless the user opts in. Pinning this so we don't accidentally make
    them positional or required in a future refactor."""
    import importlib
    cli_mod = importlib.import_module("omlx.server.cli_args")
    parser = cli_mod.build_parser()
    args = parser.parse_args(["--model", "test/model"])
    assert args.ttt_router_policy is None
    assert args.ttt_router_blocks_dir is None


def test_ttt_router_flags_parse_string_values():
    """The flags carry string paths; the wiring in hypercar_server
    converts to pathlib.Path."""
    import importlib
    cli_mod = importlib.import_module("omlx.server.cli_args")
    parser = cli_mod.build_parser()
    args = parser.parse_args([
        "--model", "test/model",
        "--ttt-router-policy", "policies/qwen36.json",
        "--ttt-router-blocks-dir", "phase0_ttt/",
    ])
    assert args.ttt_router_policy == "policies/qwen36.json"
    assert args.ttt_router_blocks_dir == "phase0_ttt/"


def test_ttt_router_help_documents_bit_equivalence_mode():
    """Help text must surface the bit-equivalence (no-blocks) mode
    explicitly — Phase 3 validation starts there before turning on TTT
    blocks one (layer, head) at a time. Discoverability matters."""
    import importlib
    cli_mod = importlib.import_module("omlx.server.cli_args")
    parser = cli_mod.build_parser()
    help_text = parser.format_help().lower()
    assert "bit-equivalence" in help_text or "bit-equivalent" in help_text
    assert "ttt" in help_text
