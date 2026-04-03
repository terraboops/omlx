# SPDX-License-Identifier: Apache-2.0
"""Medusa draft head distillation: train draft heads to predict model outputs.

Runs the base model on calibration text, collects (hidden_state, next_token)
pairs, then trains the lightweight draft heads via cross-entropy loss.

This is a fast, one-pass distillation — no backward through the base model.
Only the draft head parameters are updated (residual block + lm_head per head).

Usage:
    from omlx.medusa_distill import distill_medusa_heads
    heads = distill_medusa_heads(model, tokenizer, num_heads=3, num_steps=200)
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .medusa_heads import MedusaDraftHeads

logger = logging.getLogger(__name__)


def _get_calibration_text() -> str:
    """Return calibration text for distillation."""
    # Mix of diverse content to cover model's distribution
    return """The theory of general relativity, published by Albert Einstein in 1915, describes gravity not as a force between masses, but as a curvature of spacetime caused by mass and energy. This revolutionary framework replaced Newton's law of universal gravitation and has been confirmed by numerous experiments, including the observation of gravitational waves by LIGO in 2015.

In computer science, the concept of algorithmic complexity helps us understand how the running time or space requirements of an algorithm grow as the input size increases. Big-O notation provides an upper bound on growth rate. For example, a binary search algorithm has O(log n) time complexity, while a naive sorting algorithm like bubble sort has O(n²).

The human genome contains approximately 3 billion base pairs of DNA, organized into 23 pairs of chromosomes. The Human Genome Project, completed in 2003, mapped all of the genes in the human genome. This achievement has led to advances in personalized medicine, genetic testing, and our understanding of hereditary diseases.

Machine learning models learn patterns from data without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with many layers to learn hierarchical representations. Transformer architectures, introduced in the "Attention Is All You Need" paper, have become the foundation for modern large language models.

Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities since the Industrial Revolution have been the primary driver of recent warming, mainly through burning fossil fuels. The Paris Agreement aims to limit global warming to 1.5 degrees Celsius above pre-industrial levels.

Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in superposition of both states simultaneously. This enables quantum computers to solve certain problems exponentially faster than classical computers.

The Renaissance was a cultural movement that began in Italy in the 14th century and spread across Europe. It marked a period of renewed interest in classical learning and values. Key figures include Leonardo da Vinci, Michelangelo, and Galileo Galilei. The invention of the printing press by Johannes Gutenberg around 1440 helped disseminate ideas rapidly.

Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions. It emphasizes immutability, pure functions, and declarative code. Languages like Haskell, Erlang, and Clojure are primarily functional, while languages like Python and JavaScript support functional programming concepts."""


def distill_medusa_heads(
    model: Any,
    tokenizer: Any,
    num_heads: int = 3,
    num_steps: int = 200,
    learning_rate: float = 1e-3,
    batch_chunk: int = 256,
    progress_callback=None,
) -> MedusaDraftHeads:
    """Train Medusa draft heads via fast distillation.

    Runs the base model on calibration text to collect hidden states,
    then trains draft heads to predict next tokens from those states.

    Args:
        model: The loaded MLX model.
        tokenizer: The tokenizer.
        num_heads: Number of draft heads to train.
        num_steps: Training steps (gradient updates).
        learning_rate: AdamW learning rate.
        batch_chunk: Tokens per training chunk.
        progress_callback: Optional fn(step, loss) called each step.

    Returns:
        Trained MedusaDraftHeads.
    """
    d_model = model.args.hidden_size
    vocab_size = model.args.vocab_size
    cb = progress_callback or (lambda step, loss: None)

    # Initialize draft heads (must be unfrozen for training)
    heads = MedusaDraftHeads(d_model, vocab_size, num_heads=num_heads)
    heads.train()  # Ensure training mode
    mx.eval(heads.parameters())

    # Collect calibration hidden states from base model
    print(f"  Collecting hidden states from calibration text...", flush=True)
    cal_text = _get_calibration_text()
    tokens = tokenizer.encode(cal_text)
    # Limit to reasonable size
    tokens = tokens[:2048]

    x = mx.array([tokens])
    # Get hidden states (before lm_head)
    hidden = model.model(x)
    mx.eval(hidden)
    # hidden shape: (1, seq_len, d_model)
    hidden = hidden[0]  # (seq_len, d_model)

    # Target tokens: for head k, target at position t is token at t+k+1
    # (head 0 predicts t+1, head 1 predicts t+2, etc.)
    token_ids = mx.array(tokens)  # (seq_len,)
    seq_len = len(tokens)

    print(f"  Collected {seq_len} hidden states ({hidden.shape})", flush=True)
    print(f"  Training {num_heads} heads for {num_steps} steps...", flush=True)

    # Set up optimizer — only train draft head parameters
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Use the standard MLX pattern: model as first arg to value_and_grad
    def loss_fn(model, hidden_chunk, target_chunks):
        """Cross-entropy loss for all heads."""
        draft_logits = model(hidden_chunk)  # list of (chunk_len, vocab_size)

        total_loss = mx.array(0.0)
        for k, dl in enumerate(draft_logits):
            # Cross-entropy: -log softmax(logits)[target]
            log_probs = dl - mx.logsumexp(dl, axis=-1, keepdims=True)
            targets = target_chunks[k]
            ce = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
            total_loss = total_loss + ce.mean()

        return total_loss / num_heads

    loss_and_grad = nn.value_and_grad(heads, loss_fn)

    start = time.perf_counter()

    for step in range(num_steps):
        # Random chunk from the sequence
        max_start = seq_len - batch_chunk - num_heads - 1
        if max_start <= 0:
            chunk_start = 0
            chunk_end = min(seq_len - num_heads - 1, batch_chunk)
        else:
            chunk_start = mx.random.randint(0, max_start, shape=()).item()
            chunk_end = chunk_start + batch_chunk

        hidden_chunk = hidden[chunk_start:chunk_end]  # (chunk_len, d_model)

        # Build targets for each head
        target_chunks = []
        for k in range(num_heads):
            offset = k + 1
            targets = token_ids[chunk_start + offset:chunk_end + offset]
            target_chunks.append(targets)

        # Forward + backward
        loss, grads = loss_and_grad(heads, hidden_chunk, target_chunks)
        optimizer.update(heads, grads)
        mx.eval(heads.parameters(), optimizer.state)

        loss_val = loss.item()
        cb(step, loss_val)

        if step % 50 == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - start
            print(f"    Step {step:>4}/{num_steps}: loss={loss_val:.4f} ({elapsed:.1f}s)", flush=True)

    elapsed = time.perf_counter() - start
    print(f"  Distillation complete in {elapsed:.1f}s", flush=True)

    return heads
