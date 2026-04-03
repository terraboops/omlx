# SPDX-License-Identifier: Apache-2.0
"""Medusa speculative decoding: multi-token lookahead with tree verification.

Accelerates autoregressive generation by predicting multiple future tokens
in parallel using lightweight draft heads, then verifying them in a single
forward pass. On memory-bound hardware (M4 Pro), the draft heads add
negligible overhead while potentially accepting 2-4 tokens per step.

Usage:
    from omlx.medusa_decode import medusa_generate
    tokens, stats = medusa_generate(model, tokenizer, prompt, max_tokens=100)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx

from .medusa_heads import MedusaDraftHeads

logger = logging.getLogger(__name__)


@dataclass
class MedusaStats:
    """Statistics from a Medusa generation run."""
    total_tokens: int = 0
    total_steps: int = 0
    accepted_draft_tokens: int = 0
    total_draft_candidates: int = 0
    elapsed_seconds: float = 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.total_tokens / max(1, self.total_steps)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / max(1e-6, self.elapsed_seconds)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / max(1, self.total_draft_candidates)


def _get_hidden_states(model: Any, inputs: mx.array, cache: Any) -> Tuple[mx.array, mx.array]:
    """Run model backbone and lm_head separately to get both hidden states and logits.

    Args:
        model: The MLX model with model.model (backbone) and model.lm_head.
        inputs: (B, L) input token ids.
        cache: KV cache.

    Returns:
        (hidden_states, logits) — hidden before lm_head, logits after.
    """
    hidden = model.model(inputs, cache=cache)

    if hasattr(model.args, 'tie_word_embeddings') and model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)

    logits = logits / model.logits_scaling
    return hidden, logits


def _sample_token(logits: mx.array, temperature: float = 0.7, top_p: float = 0.9) -> int:
    """Sample a single token with temperature and top-p."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).item()
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    cutoff = (cumulative - sorted_probs) >= top_p
    sorted_probs = mx.where(cutoff, 0.0, sorted_probs)
    sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
    token_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))
    return mx.take_along_axis(sorted_indices, token_idx[..., None], axis=-1).squeeze(-1).item()


def medusa_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100,
    draft_heads: Optional[MedusaDraftHeads] = None,
    num_heads: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[int], MedusaStats]:
    """Generate tokens using Medusa speculative decoding.

    Args:
        model: The loaded MLX model.
        tokenizer: The tokenizer.
        prompt: Input prompt string.
        max_tokens: Maximum tokens to generate.
        draft_heads: Pre-initialized draft heads (creates random ones if None).
        num_heads: Number of draft heads (if creating new ones).

    Returns:
        (generated_token_ids, stats)
    """
    stats = MedusaStats()

    # Initialize draft heads if needed
    if draft_heads is None:
        d_model = model.args.hidden_size
        vocab_size = model.args.vocab_size
        draft_heads = MedusaDraftHeads(d_model, vocab_size, num_heads=num_heads)
        mx.eval(draft_heads.parameters())
        logger.info(f"Medusa: initialized {num_heads} draft heads (random weights)")

    K = draft_heads.num_heads

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    cache = model.make_cache()

    # Prefill
    x = mx.array([tokens])
    hidden, logits = _get_hidden_states(model, x, cache)
    mx.eval(hidden, logits)

    generated = []
    start = time.perf_counter()

    while len(generated) < max_tokens:
        stats.total_steps += 1

        # Sample the main token
        main_token = _sample_token(logits[0, -1, :], temperature, top_p)
        generated.append(main_token)
        stats.total_tokens += 1

        if len(generated) >= max_tokens:
            break

        # Get draft predictions from the last hidden state
        last_hidden = hidden[:, -1:, :]  # (1, 1, D)
        draft_logits_list = draft_heads(last_hidden)  # list of (1, 1, V)

        # Greedily sample from each draft head
        draft_tokens = []
        for dl in draft_logits_list:
            dt = mx.argmax(dl[:, -1, :], axis=-1).item()
            draft_tokens.append(dt)

        stats.total_draft_candidates += len(draft_tokens)

        # Build verification batch: [main_token, draft_1, draft_2, ..., draft_K]
        verify_tokens = [main_token] + draft_tokens
        verify_input = mx.array([verify_tokens]).reshape(1, -1)

        # Single forward pass to verify all candidates
        hidden, logits = _get_hidden_states(model, verify_input, cache)
        mx.eval(hidden, logits)

        # Verify: at position i, the model's prediction should match verify_tokens[i+1]
        # logits shape: (1, K+1, V)
        # logits[:, 0, :] is the model's prediction after seeing main_token → should predict draft_1
        # logits[:, 1, :] is after seeing draft_1 → should predict draft_2
        # etc.
        accepted = 0
        for i in range(len(draft_tokens)):
            model_pred = mx.argmax(logits[:, i, :], axis=-1).item()
            if model_pred == draft_tokens[i]:
                generated.append(draft_tokens[i])
                stats.total_tokens += 1
                stats.accepted_draft_tokens += 1
                accepted += 1
            else:
                # Mismatch: use the model's prediction instead and stop
                if len(generated) < max_tokens:
                    generated.append(model_pred)
                    stats.total_tokens += 1
                break

        if len(generated) >= max_tokens:
            break

        # For next iteration: the last position's logits are our starting point
        # If we accepted all K drafts, we need the logits after the last draft token
        # If we rejected at position i, we already have the correct next logits
        # In both cases, logits[:, accepted, :] is where we continue from
        # But we need to adjust: take the hidden state at the right position
        hidden = hidden[:, accepted:accepted+1, :]
        logits = logits[:, accepted:accepted+1, :]

    stats.elapsed_seconds = time.perf_counter() - start
    return generated, stats
