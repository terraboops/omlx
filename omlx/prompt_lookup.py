# SPDX-License-Identifier: Apache-2.0
"""Prompt Lookup Decoding: n-gram matching for free speculative drafts.

Instead of a draft model, matches n-grams from the prompt/context against
the most recently generated tokens. When a match is found, the continuation
in the prompt becomes draft tokens — verified in a single forward pass.

This is especially powerful for:
  - Code completion (identifiers, imports, patterns repeat)
  - Summarization (output phrases appear verbatim in input)
  - Multi-turn chat (repeating context from previous turns)
  - RAG (answer tokens often come from retrieved documents)

Zero memory cost. Zero model computation for drafting.

Based on: Saxena (2023) "Prompt Lookup Decoding"
Extended with: N-gram trie for efficient matching at large contexts.

Usage:
    from omlx.prompt_lookup import prompt_lookup_generate
    tokens, stats = prompt_lookup_generate(model, tokenizer, prompt, max_tokens=200)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class PromptLookupStats:
    """Statistics from a prompt lookup generation run."""
    total_tokens: int = 0
    total_steps: int = 0
    lookup_hits: int = 0
    lookup_attempts: int = 0
    lookup_tokens_accepted: int = 0
    elapsed_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / max(1e-6, self.elapsed_seconds)

    @property
    def hit_rate(self) -> float:
        return self.lookup_hits / max(1, self.lookup_attempts)

    @property
    def tokens_per_step(self) -> float:
        return self.total_tokens / max(1, self.total_steps)


class NGramIndex:
    """Efficient n-gram index for prompt lookup.

    Builds a hash map from n-grams to their positions in the token sequence,
    enabling O(1) lookup for draft candidates.
    """

    def __init__(self, tokens: List[int], n: int = 3):
        """Build n-gram index from token sequence.

        Args:
            tokens: The full token sequence (prompt + generated so far).
            n: N-gram size for matching (default 3).
        """
        self.n = n
        self.index: Dict[tuple, List[int]] = {}

        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            if gram not in self.index:
                self.index[gram] = []
            self.index[gram].append(i)

    def lookup(self, suffix: List[int], max_draft: int = 5) -> List[int]:
        """Find the longest continuation after matching the suffix.

        Args:
            suffix: Recent tokens to match (last n tokens of generated output).
            max_draft: Maximum draft tokens to return.

        Returns:
            List of draft token IDs (may be empty if no match).
        """
        if len(suffix) < self.n:
            return []

        gram = tuple(suffix[-self.n:])
        positions = self.index.get(gram, [])

        if not positions:
            return []

        # Find the longest matching continuation from any position
        best_draft = []
        for pos in positions:
            # The continuation starts at pos + n
            start = pos + self.n
            draft = []
            for j in range(max_draft):
                if start + j >= len(self.all_tokens):
                    break
                draft.append(self.all_tokens[start + j])
            if len(draft) > len(best_draft):
                best_draft = draft

        return best_draft

    def update(self, all_tokens: List[int]):
        """Update the index with the full token sequence."""
        self.all_tokens = all_tokens
        self.index.clear()
        for i in range(len(all_tokens) - self.n + 1):
            gram = tuple(all_tokens[i:i + self.n])
            if gram not in self.index:
                self.index[gram] = []
            self.index[gram].append(i)


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


def prompt_lookup_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 200,
    ngram_size: int = 3,
    max_draft: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[int], PromptLookupStats]:
    """Generate tokens with prompt lookup speculative decoding.

    Args:
        model: The loaded MLX model.
        tokenizer: The tokenizer.
        prompt: Input prompt string.
        max_tokens: Maximum tokens to generate.
        ngram_size: N-gram size for matching (3 = trigram).
        max_draft: Maximum draft tokens per lookup.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        (generated_token_ids, stats)
    """
    stats = PromptLookupStats()

    prompt_tokens = tokenizer.encode(prompt)
    all_tokens = list(prompt_tokens)
    cache = model.make_cache()

    # Prefill
    x = mx.array([prompt_tokens])
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Build n-gram index from prompt
    ngram_idx = NGramIndex(prompt_tokens, n=ngram_size)
    ngram_idx.all_tokens = list(prompt_tokens)

    generated = []
    start = time.perf_counter()

    while len(generated) < max_tokens:
        stats.total_steps += 1

        # Sample main token
        main_token = _sample_token(logits[0, -1, :], temperature, top_p)
        generated.append(main_token)
        all_tokens.append(main_token)
        stats.total_tokens += 1

        if len(generated) >= max_tokens:
            break

        # Try n-gram lookup for draft tokens
        stats.lookup_attempts += 1
        recent = all_tokens[-(ngram_size + 1):]  # include the just-generated token
        ngram_idx.update(all_tokens)
        draft_tokens = ngram_idx.lookup(recent, max_draft=max_draft)

        if draft_tokens:
            stats.lookup_hits += 1

            # Feed main_token + draft_tokens in one forward pass for verification
            verify_input = mx.array([[main_token] + draft_tokens])
            logits = model(verify_input, cache=cache)
            mx.eval(logits)

            # Verify: at position i, model's prediction should match draft_tokens[i]
            accepted = 0
            for i in range(len(draft_tokens)):
                # Position i in logits predicts what comes after verify_input[0, i]
                model_pred = mx.argmax(logits[0, i, :], axis=-1).item()
                if temperature <= 0:
                    matches = model_pred == draft_tokens[i]
                else:
                    # With temperature sampling, accept if draft token has reasonable probability
                    draft_logit = logits[0, i, draft_tokens[i]].item()
                    top_logit = logits[0, i, model_pred].item()
                    matches = (draft_logit > top_logit - 2.0)  # within top ~7x probability

                if matches:
                    generated.append(draft_tokens[i])
                    all_tokens.append(draft_tokens[i])
                    stats.total_tokens += 1
                    stats.lookup_tokens_accepted += 1
                    accepted += 1
                else:
                    # Accept model's correction instead
                    correction = _sample_token(logits[0, i, :], temperature, top_p)
                    generated.append(correction)
                    all_tokens.append(correction)
                    stats.total_tokens += 1
                    break

            if len(generated) >= max_tokens:
                break

            # Continue from the last accepted position's logits
            logits = logits[:, accepted:accepted + 1, :]

        else:
            # No lookup match — standard autoregressive step
            x = mx.array([[main_token]])
            logits = model(x, cache=cache)
            mx.eval(logits)

    stats.elapsed_seconds = time.perf_counter() - start
    return generated, stats
