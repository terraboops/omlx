# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration and fixtures for oMLX tests.

This module provides common fixtures used across test files.
MLX-dependent imports are deferred to fixture bodies so that tests
which don't need MLX (tool tests, eval tests) can run without GPU.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# --- MLX-safe collection filtering ---
# Most test files in this directory import omlx.* which triggers MLX Metal
# device init (SIGABRT if Metal unavailable). Only collect known-safe files
# unless MLX_AVAILABLE=1 is set in the environment.
_SAFE_WITHOUT_MLX = {
    "test_hypercar_tools.py",
    "test_eval_mmlu_pro.py",
    "test_ttt_verifier.py",
    "test_specdec_gate.py",
    "test_baseline.py",
    "test_chat_image_upload.py",
    "test_find_matching_dmg.py",
    "test_grammar_live.py",
    "test_latent_kv_cache.py",
}

collect_ignore = []
if not os.environ.get("MLX_AVAILABLE"):
    # Build ignore list: any test_*.py that imports omlx (chains to MLX SIGABRT)
    _tests_dir = Path(__file__).parent
    for _f in _tests_dir.rglob("test_*.py"):
        if _f.name in _SAFE_WITHOUT_MLX:
            continue
        try:
            _content = _f.read_text(encoding="utf-8")
            if "from omlx" in _content or "import omlx" in _content:
                collect_ignore.append(str(_f))
        except Exception:
            pass

import pytest


class MockTokenizer:
    """Mock tokenizer for testing without loading real models."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids (simple simulation)."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for i, word in enumerate(text.split()):
            token_id = (hash(word) % (self.vocab_size - 10)) + 10
            tokens.append(token_id)
        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text (simple simulation)."""
        if skip_special_tokens:
            token_ids = [
                t
                for t in token_ids
                if t not in (self.eos_token_id, self.pad_token_id, self.bos_token_id)
            ]
        return f"<decoded:{len(token_ids)} tokens>"

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Tokenize text and return dict with input_ids."""
        input_ids = self.encode(text)
        return {"input_ids": input_ids}


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        vocab_size: int = 32000,
        model_type: str = "llama",
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.model_type = model_type


class MockModel:
    """Mock model for testing without loading real models."""

    def __init__(self, config: Optional[MockModelConfig] = None):
        self.config = config or MockModelConfig()
        self._parameters: Dict[str, Any] = {}

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        """Forward pass (returns mock logits)."""
        mock_output = MagicMock()
        mock_output.shape = (1, len(input_ids) if hasattr(input_ids, "__len__") else 1, self.config.vocab_size)
        return mock_output

    def parameters(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self._parameters


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """Provide a mock tokenizer for tests."""
    return MockTokenizer()


@pytest.fixture
def mock_model() -> MockModel:
    """Provide a mock model for tests."""
    return MockModel()


@pytest.fixture
def mock_model_config() -> MockModelConfig:
    """Provide a mock model configuration for tests."""
    return MockModelConfig()


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_request():
    """Factory fixture for creating sample Request objects.

    Requires MLX/Metal — skips if unavailable.
    """
    from omlx.request import Request, SamplingParams
    return Request(
        request_id="test-request-001",
        prompt="Hello, world!",
        sampling_params=SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        ),
    )


@pytest.fixture
def sample_request_factory():
    """Factory fixture for creating multiple Request objects.

    Requires MLX/Metal — skips if unavailable.
    """
    from omlx.request import Request, SamplingParams

    def _create_request(
        request_id: str = "test-request-001",
        prompt: str = "Hello, world!",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Request:
        return Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )

    return _create_request


@pytest.fixture
def real_model_dir() -> Path:
    """Return the path to real models directory.

    Note: Tests using this fixture may require actual model files
    and should be marked with @pytest.mark.slow.
    """
    return Path.home() / "Workspace" / "models"
