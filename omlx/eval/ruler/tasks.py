# SPDX-License-Identifier: Apache-2.0
"""Synthetic task generators for RULER long-context evaluation.

Each generator returns a dict with:
  - context: str — the full context with distractors
  - question: str — what to ask the model
  - expected: list[str] — acceptable answers (any match = correct)
  - task_type: str — identifier for logging/gating
  - params: dict — generation parameters for reproducibility

Based on RULER (arXiv:2404.06654) with adaptations for local bench use.
"""

from __future__ import annotations

import random
import string


# ---------------------------------------------------------------------------
# Filler text for haystacks
# ---------------------------------------------------------------------------

_CODE_BLOCKS = [
    '''def binary_search(arr: list, target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    '''class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._store: dict = {}
        self._order: list = []

    def get(self, key: str):
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, key: str, value):
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self.capacity:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = value
        self._order.append(key)
''',
    '''import hashlib
from pathlib import Path

def checksum(path: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
''',
    '''# Graph utilities
def bfs(graph: dict[str, list[str]], start: str) -> list[str]:
    visited = set()
    queue = [start]
    order = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            order.append(node)
            queue.extend(graph.get(node, []))
    return order
''',
    '''from typing import TypeVar, Generic
T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self):
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)
''',
    '''def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    assert cols_a == rows_b
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result
''',
]


def _build_filler(tokenizer, target_tokens: int, seed: int = 42) -> str:
    """Build filler text from shuffled code blocks to fill target_tokens."""
    rng = random.Random(seed)
    filler = "\n\n".join(_CODE_BLOCKS)
    filler_tokens = len(tokenizer.encode(filler))

    reps = max(1, (target_tokens // filler_tokens) + 1)
    blocks = _CODE_BLOCKS * reps
    rng.shuffle(blocks)
    text = "\n\n".join(blocks)

    # Trim to target length
    tokens = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(tokens)


def _random_code(length: int = 8, seed: int | None = None) -> str:
    """Generate a random alphanumeric code like 'XKCD-4829'."""
    rng = random.Random(seed)
    prefix = "".join(rng.choices(string.ascii_uppercase, k=4))
    suffix = "".join(rng.choices(string.digits, k=4))
    return f"{prefix}-{suffix}"


# ---------------------------------------------------------------------------
# Task 1: Multi-Key NIAH
# ---------------------------------------------------------------------------

def generate_multi_key_niah(
    tokenizer,
    target_tokens: int,
    num_keys: int = 3,
    seed: int = 42,
) -> dict:
    """Generate a multi-key needle-in-a-haystack task.

    Hides `num_keys` key-value pairs at random depths in filler text.
    The model must retrieve all values.

    Returns dict with context, question, expected answers.
    """
    rng = random.Random(seed)

    # Generate unique key-value pairs
    keys_and_values = []
    for i in range(num_keys):
        code = _random_code(seed=seed + i)
        keys_and_values.append((f"key_{i+1}", code))

    # Build filler, leaving room for needles + question overhead
    overhead_tokens = num_keys * 30 + 100  # ~30 tokens per needle + question
    filler = _build_filler(tokenizer, target_tokens - overhead_tokens, seed=seed)

    # Split filler into chunks and insert needles at random positions
    lines = filler.split("\n")
    chunk_size = max(1, len(lines) // (num_keys + 1))

    for i, (key, value) in enumerate(keys_and_values):
        needle = f"# IMPORTANT: The value for {key} is {value}"
        # Place each needle in its own chunk region (spread evenly)
        insert_pos = min(
            (i + 1) * chunk_size + rng.randint(0, max(1, chunk_size // 2)),
            len(lines) - 1,
        )
        lines.insert(insert_pos, needle)

    context = "\n".join(lines)

    # Trim to target tokens
    tokens = tokenizer.encode(context)[:target_tokens]
    context = tokenizer.decode(tokens)

    # Build question asking for all values
    key_list = ", ".join(k for k, _ in keys_and_values)
    question = (
        f"Based on the text above, what are the values for the following keys: "
        f"{key_list}? List each value on its own line in the format 'key: value'."
    )

    expected = [v for _, v in keys_and_values]

    return {
        "context": context,
        "question": question,
        "expected": expected,
        "task_type": "multi_key_niah",
        "params": {
            "target_tokens": target_tokens,
            "num_keys": num_keys,
            "seed": seed,
            "keys": keys_and_values,
        },
    }


# ---------------------------------------------------------------------------
# Task 2: Variable Tracking
# ---------------------------------------------------------------------------

def generate_variable_tracking(
    tokenizer,
    target_tokens: int,
    chain_length: int = 4,
    seed: int = 42,
) -> dict:
    """Generate a variable tracking task.

    Creates a chain of variable assignments spread across the context:
      x1 = <value>
      x2 = x1
      x3 = x2
      ...
    The model must determine the final value of the last variable.

    This tests multi-hop reasoning — the model can't just find one needle,
    it must follow the chain.
    """
    rng = random.Random(seed)

    # The actual value at the start of the chain
    value = _random_code(seed=seed)
    var_names = [f"var_{chr(ord('a') + i)}" for i in range(chain_length)]

    # Build assignment statements
    assignments = []
    assignments.append(f"# Assignment: {var_names[0]} = '{value}'")
    for i in range(1, chain_length):
        assignments.append(f"# Assignment: {var_names[i]} = {var_names[i-1]}")

    # Build filler
    overhead_tokens = chain_length * 20 + 100
    filler = _build_filler(tokenizer, target_tokens - overhead_tokens, seed=seed)
    lines = filler.split("\n")

    # Spread assignments evenly through the context
    chunk_size = max(1, len(lines) // (chain_length + 1))
    for i, assignment in enumerate(assignments):
        insert_pos = min(
            (i + 1) * chunk_size + rng.randint(0, max(1, chunk_size // 3)),
            len(lines) - 1,
        )
        lines.insert(insert_pos, assignment)

    context = "\n".join(lines)
    tokens = tokenizer.encode(context)[:target_tokens]
    context = tokenizer.decode(tokens)

    question = (
        f"Based on the variable assignments in the text above, "
        f"trace the chain of assignments to find what {var_names[-1]} "
        f"ultimately resolves to. The first variable in the chain was "
        f"assigned a literal string value like 'XXXX-1234'. "
        f"What is that string value? Respond with ONLY the string value "
        f"(e.g. ABCD-5678), nothing else."
    )

    return {
        "context": context,
        "question": question,
        "expected": [value],
        "task_type": "variable_tracking",
        "params": {
            "target_tokens": target_tokens,
            "chain_length": chain_length,
            "seed": seed,
            "var_names": var_names,
            "value": value,
        },
    }


# ---------------------------------------------------------------------------
# Task 3: Frequent-Word Aggregation
# ---------------------------------------------------------------------------

def generate_frequent_word(
    tokenizer,
    target_tokens: int,
    num_target_words: int = 5,
    seed: int = 42,
) -> dict:
    """Generate a frequent-word aggregation task.

    Inserts specific marker words at known frequencies throughout the context.
    One word appears significantly more often than the others. The model must
    identify which word is most frequent.

    This tests distributed attention — the model must scan the entire context
    and count occurrences rather than just finding a single location.
    """
    rng = random.Random(seed)

    # Generate distinct marker words (unusual enough to be countable)
    marker_pool = [
        "ZEPHYR", "QUARTZ", "VORTEX", "NEBULA", "PRISM",
        "COBALT", "HELIX", "ZENITH", "AXIOM", "CIPHER",
    ]
    rng.shuffle(marker_pool)
    markers = marker_pool[:num_target_words]

    # Assign frequencies — one word is clearly the winner
    # Winner gets ~2x the frequency of runner-up
    frequencies = []
    winner_freq = 10 + rng.randint(0, 5)  # 10-15 occurrences
    frequencies.append(winner_freq)
    for i in range(1, num_target_words):
        frequencies.append(max(2, winner_freq // 2 - rng.randint(0, 2)))

    winner_word = markers[0]

    # Build filler
    overhead_tokens = sum(frequencies) * 15 + 100
    filler = _build_filler(tokenizer, target_tokens - overhead_tokens, seed=seed)
    lines = filler.split("\n")

    # Insert marker words as comments at random positions
    insertions = []
    for marker, freq in zip(markers, frequencies):
        for _ in range(freq):
            insertions.append(f"# marker: {marker}")

    rng.shuffle(insertions)

    # Spread insertions evenly
    if lines:
        step = max(1, len(lines) // (len(insertions) + 1))
        for i, insertion in enumerate(insertions):
            pos = min((i + 1) * step, len(lines))
            lines.insert(pos + i, insertion)  # +i to account for previous inserts

    context = "\n".join(lines)
    tokens = tokenizer.encode(context)[:target_tokens]
    context = tokenizer.decode(tokens)

    marker_list = ", ".join(markers)
    question = (
        f"In the text above, the following marker words appear as comments: "
        f"{marker_list}. Which marker word appears MOST frequently? "
        f"Respond with ONLY the word, nothing else."
    )

    return {
        "context": context,
        "question": question,
        "expected": [winner_word, winner_word.lower(), winner_word.title()],
        "task_type": "frequent_word",
        "params": {
            "target_tokens": target_tokens,
            "num_target_words": num_target_words,
            "seed": seed,
            "markers": list(zip(markers, frequencies)),
            "winner": winner_word,
        },
    }


# ---------------------------------------------------------------------------
# Task suites
# ---------------------------------------------------------------------------

# Quick suite: 3 task types at 4K/16K = 6 runs
# Used in default mode (not --quick, which skips everything after Phase 1)
RULER_QUICK_SUITE = [
    {"generator": "multi_key_niah", "target_tokens": 4096, "num_keys": 2, "seed": 100},
    {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 3, "seed": 101},
    {"generator": "variable_tracking", "target_tokens": 4096, "chain_length": 4, "seed": 304},
    {"generator": "variable_tracking", "target_tokens": 16384, "chain_length": 4, "seed": 305},
    {"generator": "frequent_word", "target_tokens": 4096, "num_target_words": 4, "seed": 200},
    {"generator": "frequent_word", "target_tokens": 16384, "num_target_words": 5, "seed": 201},
]

# Full suite: 3 task types × multiple configs × 3 context lengths = ~13 runs
# Used in --full mode
RULER_FULL_SUITE = [
    # Multi-key NIAH at 3 context lengths, varying key counts
    {"generator": "multi_key_niah", "target_tokens": 4096, "num_keys": 2, "seed": 100},
    {"generator": "multi_key_niah", "target_tokens": 4096, "num_keys": 4, "seed": 102},
    {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 3, "seed": 101},
    {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 5, "seed": 103},
    {"generator": "multi_key_niah", "target_tokens": 65536, "num_keys": 3, "seed": 104},
    # Variable tracking at 3 context lengths, chain lengths 3-8
    {"generator": "variable_tracking", "target_tokens": 4096, "chain_length": 3, "seed": 300},
    {"generator": "variable_tracking", "target_tokens": 4096, "chain_length": 4, "seed": 304},
    {"generator": "variable_tracking", "target_tokens": 16384, "chain_length": 4, "seed": 301},
    {"generator": "variable_tracking", "target_tokens": 16384, "chain_length": 8, "seed": 306},
    {"generator": "variable_tracking", "target_tokens": 65536, "chain_length": 4, "seed": 303},
    {"generator": "variable_tracking", "target_tokens": 65536, "chain_length": 8, "seed": 307},
    # Frequent-word aggregation at 3 context lengths
    {"generator": "frequent_word", "target_tokens": 4096, "num_target_words": 4, "seed": 200},
    {"generator": "frequent_word", "target_tokens": 16384, "num_target_words": 5, "seed": 201},
    {"generator": "frequent_word", "target_tokens": 16384, "num_target_words": 7, "seed": 202},
    {"generator": "frequent_word", "target_tokens": 65536, "num_target_words": 5, "seed": 203},
]
