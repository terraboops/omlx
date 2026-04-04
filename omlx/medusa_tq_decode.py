# SPDX-License-Identifier: Apache-2.0
"""Medusa speculative decoding over TQ3 caches — GPU-resident hot loop.

The existing medusa_decode.py creates its own KVCache. This variant:
  - Takes a pre-built TQ3 cache (streaming prefill already done)
  - Minimizes .item() calls in the verification hot loop
  - Uses GPU-resident comparison for accepted count
  - Works with the TQ3 decode_attention fused kernel

Speedup target: 1.9x over baseline TQ3 decode (verified at granite-4.0).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import mlx.core as mx

from .medusa_heads import MedusaDraftHeads

logger = logging.getLogger(__name__)


@dataclass
class MedusaTQStats:
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


def _forward_with_hidden(model, inputs: mx.array, cache: list) -> Tuple[mx.array, mx.array]:
    """Run model backbone + lm_head, returning both hidden states and logits."""
    hidden = model.model(inputs, cache=cache)
    if hasattr(model.args, 'tie_word_embeddings') and model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)
    if hasattr(model, 'logits_scaling'):
        logits = logits / model.logits_scaling
    return hidden, logits


def medusa_tq_generate(
    model: Any,
    cache: list,                  # Pre-prefilled TQ3 cache
    start_logits: mx.array,       # (B, 1, V) last logits from prefill
    start_hidden: mx.array,       # (B, 1, D) last hidden from prefill
    draft_heads: MedusaDraftHeads,
    max_tokens: int = 64,
    eos_token: Optional[int] = None,
) -> Tuple[List[int], MedusaTQStats]:
    """Generate tokens using Medusa over TQ3 cache.

    Assumes `cache` has been prefilled and `start_logits`/`start_hidden`
    contain the last position's outputs. The cache advances as tokens
    are accepted.
    """
    stats = MedusaTQStats()
    K = draft_heads.num_heads
    generated: List[int] = []

    logits = start_logits
    hidden = start_hidden
    start = time.perf_counter()

    while len(generated) < max_tokens:
        stats.total_steps += 1

        # 1. Sample main token from current logits (one sync to get scalar)
        main_token_arr = mx.argmax(logits[:, -1, :], axis=-1)  # (B,)
        mx.eval(main_token_arr)
        main_token = int(main_token_arr.item())
        generated.append(main_token)
        stats.total_tokens += 1

        if len(generated) >= max_tokens or (eos_token is not None and main_token == eos_token):
            break

        # 2. Draft K future tokens from the last hidden state
        last_hidden = hidden[:, -1:, :]  # (1, 1, D)
        draft_logits_list = draft_heads(last_hidden)  # list of (1, 1, V)

        # Stack draft predictions and get argmax in one go (single sync)
        draft_stack = mx.stack(
            [mx.argmax(dl[:, -1, :], axis=-1).squeeze(0) for dl in draft_logits_list]
        )  # (K,)
        mx.eval(draft_stack)
        # One .tolist() call instead of K .item() calls
        draft_tokens = draft_stack.tolist()
        stats.total_draft_candidates += K

        # 3. Verify: feed [main_token, draft_1, ..., draft_K] in one forward pass
        verify_input = mx.array([[main_token] + draft_tokens])  # (1, K+1)
        hidden, logits = _forward_with_hidden(model, verify_input, cache)
        mx.eval(hidden, logits)

        # 4. GPU-resident acceptance check:
        # logits[:, i, :] predicts token at position i+1
        # Check if argmax(logits[:, i, :]) == draft_tokens[i] for i in 0..K-1
        model_preds = mx.argmax(logits[0, :K, :], axis=-1)  # (K,)
        draft_arr = mx.array(draft_tokens, dtype=model_preds.dtype)
        matches = (model_preds == draft_arr).astype(mx.int32)  # (K,)
        # Find first mismatch via cumulative product
        # cum_matches[i] = 1 iff all 0..i matched
        cum_matches = mx.cumprod(matches, axis=0)
        accepted_arr = mx.sum(cum_matches, axis=0)  # scalar
        mx.eval(accepted_arr)
        accepted = int(accepted_arr.item())

        # 5. Append accepted draft tokens
        for i in range(accepted):
            generated.append(draft_tokens[i])
            stats.total_tokens += 1
            stats.accepted_draft_tokens += 1

        # 6. If a draft was rejected, take the model's prediction at that position
        if accepted < K and len(generated) < max_tokens:
            # Position `accepted` is where the rejection happened
            # model_preds[accepted] is the correct token model wanted to predict
            mp = int(model_preds[accepted].item())
            generated.append(mp)
            stats.total_tokens += 1

        # 7. Set up next iteration: use the last accepted position's outputs
        # Position to read from:
        #   - If all K accepted: read position K (the K-th draft)
        #   - If accepted=a<K: read position a (where model's prediction replaces draft)
        read_pos = accepted
        hidden = hidden[:, read_pos:read_pos+1, :]
        logits = logits[:, read_pos:read_pos+1, :]

        if eos_token is not None and any(t == eos_token for t in generated[-1-accepted:]):
            break

    stats.elapsed_seconds = time.perf_counter() - start
    return generated, stats
