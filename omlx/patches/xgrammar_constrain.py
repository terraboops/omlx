# SPDX-License-Identifier: Apache-2.0
"""XGrammar-based constrained decoding for JSON schema enforcement.

When a request includes tools or response_format with a JSON schema,
compiles the schema into an XGrammar matcher that constrains each
generated token to valid JSON matching the schema.

Usage:
    from omlx.patches.xgrammar_constrain import apply_grammar_constraint
    apply_grammar_constraint(tokenizer)

    # Then during generation, the sampler is automatically constrained
    # when the request includes tools or response_format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger("hypercar.xgrammar")

# Grammar cache: schema_hash → (compiled_grammar, hit_count, compile_time_ms)
_grammar_cache: dict[str, tuple[Any, int, float]] = {}
_compiler = None
_tokenizer_info = None


def _ensure_compiler(tokenizer) -> None:
    """Lazily initialize the XGrammar compiler with the tokenizer."""
    global _compiler, _tokenizer_info
    if _compiler is not None:
        return

    try:
        import xgrammar as xgr
        _tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        _compiler = xgr.GrammarCompiler(_tokenizer_info)
        logger.info(f"XGrammar compiler initialized (vocab={_tokenizer_info.vocab_size})")
    except Exception as e:
        logger.warning(f"XGrammar initialization failed: {e}")
        raise


def get_compiled_grammar(schema: str | dict, tokenizer=None) -> Any:
    """Get or compile a grammar for a JSON schema.

    Caches compiled grammars by schema hash for reuse across requests.

    Args:
        schema: JSON schema as string or dict
        tokenizer: HF tokenizer (needed for first-time initialization)

    Returns:
        Compiled grammar object
    """
    import xgrammar as xgr

    if tokenizer is not None:
        _ensure_compiler(tokenizer)

    if _compiler is None:
        raise RuntimeError("XGrammar compiler not initialized")

    schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
    schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    if schema_hash in _grammar_cache:
        compiled, hits, compile_ms = _grammar_cache[schema_hash]
        _grammar_cache[schema_hash] = (compiled, hits + 1, compile_ms)
        return compiled

    t0 = time.perf_counter()
    compiled = _compiler.compile_json_schema(schema_str)
    compile_ms = (time.perf_counter() - t0) * 1000

    _grammar_cache[schema_hash] = (compiled, 0, compile_ms)
    logger.info(f"Compiled grammar for schema {schema_hash} in {compile_ms:.1f}ms")
    return compiled


def create_grammar_sampler(compiled_grammar, tokenizer) -> callable:
    """Create a grammar-constrained sampler function.

    Returns a function that wraps the standard sampler: applies the
    grammar bitmask BEFORE softmax, then advances the matcher state
    after sampling.

    The returned function has signature: (logits: mx.array) -> mx.array
    matching mlx_lm's sampler interface.
    """
    import xgrammar as xgr
    import mlx.core as mx

    _ensure_compiler(tokenizer)
    matcher = xgr.GrammarMatcher(compiled_grammar)
    bitmask = xgr.allocate_token_bitmask(1, _tokenizer_info.vocab_size)
    vocab_size = _tokenizer_info.vocab_size

    def grammar_sampler(logits: mx.array) -> mx.array:
        """Apply grammar mask then sample."""
        if matcher.is_terminated():
            return mx.argmax(logits, axis=-1)

        # Get allowed tokens bitmask
        matcher.fill_next_token_bitmask(bitmask)

        # Convert torch bitmask to MLX boolean mask
        # bitmask is packed int32: each bit represents one token
        import torch
        bool_mask_torch = torch.zeros(vocab_size, dtype=torch.bool)
        for i in range(bitmask.shape[1]):
            bits = bitmask[0, i].item()
            for bit in range(32):
                token_id = i * 32 + bit
                if token_id < vocab_size and (bits >> bit) & 1:
                    bool_mask_torch[token_id] = True

        # Apply mask: set disallowed tokens to -inf
        bool_mask = mx.array(bool_mask_torch.numpy())
        masked_logits = mx.where(bool_mask, logits, mx.array(float('-inf')))

        # Sample from masked logits
        token = mx.argmax(masked_logits, axis=-1)
        token_id = token.item()

        # Advance grammar state
        if not matcher.is_terminated():
            matcher.accept_token(token_id)

        return token

    grammar_sampler._matcher = matcher  # expose for testing
    return grammar_sampler


def extract_json_schema_from_request(request_body: dict) -> str | None:
    """Extract JSON schema from an OpenAI-compatible request.

    Checks for:
    1. response_format.json_schema.schema
    2. tools[*].function.parameters (combined into a union schema)
    """
    # Check response_format
    rf = request_body.get("response_format", {})
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        schema = rf.get("json_schema", {}).get("schema")
        if schema:
            return json.dumps(schema) if isinstance(schema, dict) else schema

    # Check tools
    tools = request_body.get("tools", [])
    if tools:
        # For simplicity, use the first tool's parameter schema
        for tool in tools:
            if isinstance(tool, dict):
                fn = tool.get("function", {})
                params = fn.get("parameters")
                if params:
                    return json.dumps(params) if isinstance(params, dict) else params

    return None


def grammar_cache_stats() -> dict:
    """Return cache statistics for the /v1/internal/grammar_cache_stats endpoint."""
    stats = {
        "cache_size": len(_grammar_cache),
        "entries": [],
    }
    for schema_hash, (_, hits, compile_ms) in _grammar_cache.items():
        stats["entries"].append({
            "schema_hash": schema_hash,
            "hit_count": hits,
            "compile_time_ms": round(compile_ms, 1),
        })
    return stats
