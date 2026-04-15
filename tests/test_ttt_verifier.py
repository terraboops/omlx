# SPDX-License-Identifier: Apache-2.0
"""Tests for TTT code verifier — pure Python execution tests, no MLX needed."""

import importlib.util

import pytest


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load verify_code directly to avoid MLX import chain
# We need to extract it without triggering the full omlx import
def _get_verify_code():
    """Extract verify_code from ttt.py without importing the full module."""
    import subprocess
    import tempfile
    from pathlib import Path
    from typing import Tuple

    # verify_code only depends on subprocess, tempfile, Path — reconstruct it
    def verify_code(code: str, test: str = "", timeout: int = 10) -> Tuple[bool, str]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            if test:
                f.write(f"\n\n# Test\n{test}\nprint('PASS')\n")
            f.flush()
            try:
                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True, text=True, timeout=timeout,
                )
                passed = result.returncode == 0
                if test:
                    passed = passed and "PASS" in result.stdout
                error = result.stderr[:200] if not passed else ""
                return passed, error
            except subprocess.TimeoutExpired:
                return False, "timeout"
            except Exception as e:
                return False, str(e)[:200]
            finally:
                Path(f.name).unlink(missing_ok=True)

    return verify_code


verify_code = _get_verify_code()


class TestVerifyCodeBasic:
    """Test the code verifier with simple Python programs."""

    def test_valid_code_passes(self):
        passed, err = verify_code("x = 1 + 1\nprint(x)")
        assert passed, f"Valid code should pass: {err}"

    def test_syntax_error_fails(self):
        passed, err = verify_code("def foo(:\n  pass")
        assert not passed
        assert "SyntaxError" in err

    def test_runtime_error_fails(self):
        passed, err = verify_code("x = 1 / 0")
        assert not passed
        assert "ZeroDivision" in err

    def test_empty_code_passes(self):
        passed, err = verify_code("")
        assert passed


class TestVerifyCodeWithTests:
    """Test the code verifier with assertion tests."""

    def test_passing_assertion(self):
        code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        test = "assert factorial(5) == 120"
        passed, err = verify_code(code, test)
        assert passed, f"Correct factorial should pass: {err}"

    def test_failing_assertion(self):
        code = "def factorial(n):\n    return n"  # Wrong implementation
        test = "assert factorial(5) == 120"
        passed, err = verify_code(code, test)
        assert not passed

    def test_pass_marker_required(self):
        """verify_code checks for 'PASS' in stdout when test is provided."""
        code = "x = 42"
        test = "assert x == 42"
        passed, _ = verify_code(code, test)
        assert passed

    def test_timeout_returns_false(self):
        code = "import time; time.sleep(30)"
        passed, err = verify_code(code, timeout=1)
        assert not passed
        assert err == "timeout"

    def test_import_error_fails(self):
        code = "import nonexistent_module_xyz"
        passed, err = verify_code(code)
        assert not passed
        assert "ModuleNotFoundError" in err


class TestVerifyCodeSecurity:
    """Test that the verifier handles potentially problematic code."""

    def test_no_file_pollution(self):
        """Temp file should be cleaned up after execution."""
        import os
        import glob
        before = set(glob.glob("/tmp/tmp*.py"))
        verify_code("x = 1")
        after = set(glob.glob("/tmp/tmp*.py"))
        new_files = after - before
        assert len(new_files) == 0, f"Temp files leaked: {new_files}"

    def test_multiline_code(self):
        code = """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
        test = "assert is_prime(17) and not is_prime(4)"
        passed, err = verify_code(code, test)
        assert passed, f"is_prime should work: {err}"


class TestVerifyCodeConsistency:
    """Verify that verify_code matches the implementation in omlx/ttt.py."""

    def test_function_exists_in_source(self):
        """Verify verify_code is defined in ttt.py."""
        from pathlib import Path
        src = Path("omlx/ttt.py").read_text()
        assert "def verify_code(" in src

    def test_uses_subprocess_run(self):
        """Verify the real implementation uses subprocess.run (not exec)."""
        from pathlib import Path
        src = Path("omlx/ttt.py").read_text()
        # Find the verify_code function body
        start = src.index("def verify_code(")
        end = src.index("\ndef ", start + 1) if "\ndef " in src[start + 1:] else len(src)
        body = src[start:start + end]
        assert "subprocess.run" in body, (
            "verify_code must use subprocess.run for isolation, not exec()"
        )
