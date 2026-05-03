# SPDX-License-Identifier: Apache-2.0
"""Pin the conftest's MLX-safety filter (AST-based ``omlx`` import
detection).

The filter exists to prevent SIGABRT on Metal-less environments. Three
test files were silently dropped from the default run before the AST
fix landed; this test catches similar regressions in either direction
(false-positive on docstring, false-negative on lazy-import-in-function).
"""

from pathlib import Path

import pytest

# Import directly from conftest — pytest's conftest is a module like any other.
import sys
sys.path.insert(0, str(Path(__file__).parent))
import conftest  # noqa: E402


def _write(tmp_path, content: str) -> Path:
    p = tmp_path / "fake_test.py"
    p.write_text(content)
    return p


def test_top_level_import_omlx_detected(tmp_path):
    src = "import omlx\n\ndef test_x(): pass\n"
    assert conftest._has_omlx_import(_write(tmp_path, src)) is True


def test_top_level_import_omlx_submodule_detected(tmp_path):
    src = "import omlx.hypercar_server\n"
    assert conftest._has_omlx_import(_write(tmp_path, src)) is True


def test_top_level_from_omlx_import_detected(tmp_path):
    src = "from omlx.state_space import TTTLinear\n"
    assert conftest._has_omlx_import(_write(tmp_path, src)) is True


def test_lazy_import_inside_function_body_detected(tmp_path):
    """An import inside a function body still executes at test-run time
    and would trigger MLX init. The substring filter caught these; the
    AST walk must still catch them via ``ast.walk`` (vs walking only
    top-level)."""
    src = (
        "def test_x():\n"
        "    from omlx.server import verify_api_key\n"
        "    assert verify_api_key is not None\n"
    )
    assert conftest._has_omlx_import(_write(tmp_path, src)) is True


def test_omlx_in_docstring_not_detected(tmp_path):
    """The bug that triggered the AST switch: substring filter
    false-positived on `from omlx.hypercar_server import` inside a
    docstring, silently filtering ``test_hypercar_server_spec_decode.py``
    from the default run for ~1 day."""
    src = (
        '"""Some docstring that mentions `from omlx.hypercar_server '
        'import apply_hypercar_patches` as an example."""\n'
        "def test_x(): pass\n"
    )
    assert conftest._has_omlx_import(_write(tmp_path, src)) is False


def test_omlx_in_string_literal_not_detected(tmp_path):
    src = (
        "MODULES_TO_DOCUMENT = ['omlx.bench', 'omlx.eval']\n"
        "def test_x(): assert MODULES_TO_DOCUMENT\n"
    )
    assert conftest._has_omlx_import(_write(tmp_path, src)) is False


def test_omlx_in_comment_not_detected(tmp_path):
    src = (
        "# This module wraps `from omlx.hypercar_server import ...`\n"
        "# but doesn't actually import omlx.\n"
        "def test_x(): pass\n"
    )
    assert conftest._has_omlx_import(_write(tmp_path, src)) is False


def test_unrelated_imports_not_detected(tmp_path):
    src = (
        "import os\n"
        "import json\n"
        "from pathlib import Path\n"
        "def test_x(): pass\n"
    )
    assert conftest._has_omlx_import(_write(tmp_path, src)) is False


def test_module_named_omlx_prefix_only_no_dot_not_detected(tmp_path):
    """Edge case: a fictional module named ``omlxify`` shouldn't trip
    the filter. The import-name match is exact (``omlx``) or
    dot-prefixed (``omlx.something``), not substring."""
    src = "import omlxify\n"
    assert conftest._has_omlx_import(_write(tmp_path, src)) is False


def test_syntax_error_returns_false(tmp_path):
    """Malformed Python returns False (don't crash the conftest)."""
    src = "def test_x("  # unterminated
    assert conftest._has_omlx_import(_write(tmp_path, src)) is False
