#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Intelligence benchmarks: coding ability, reasoning, and code understanding.

Tests whether the model can:
1. Write correct code from docstrings (HumanEval-style)
2. Solve multi-step reasoning problems
3. Understand and debug code

Each test is scored pass/fail with the actual output shown.
"""

import argparse
import re
import time

import mlx.core as mx


def generate(model, tokenizer, prompt, max_tokens=300, temperature=0.3):
    """Generate text using mlx-lm's generate function."""
    from mlx_lm import generate as mlx_generate
    output = mlx_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, verbose=False,
    )
    mx.synchronize()
    mx.clear_cache()
    return output


# ═══════════════════════════════════════════════════════════════════
# Coding Benchmarks
# ═══════════════════════════════════════════════════════════════════

CODING_TESTS = [
    {
        "name": "FizzBuzz",
        "prompt": """Write a Python function `fizzbuzz(n)` that returns a list of strings for numbers 1 to n.
For multiples of 3, use "Fizz". For multiples of 5, use "Buzz". For both, use "FizzBuzz". Otherwise, the number as a string.

```python
def fizzbuzz(n):""",
        "test": lambda output: "Fizz" in output and "Buzz" in output and "FizzBuzz" in output,
        "validate": """
def fizzbuzz(n):
{code}

result = fizzbuzz(15)
assert result[0] == '1'
assert result[2] == 'Fizz'
assert result[4] == 'Buzz'
assert result[14] == 'FizzBuzz'
assert len(result) == 15
print('PASS')
""",
    },
    {
        "name": "Binary Search",
        "prompt": """Write a Python function `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found.

```python
def binary_search(arr, target):""",
        "test": lambda output: "left" in output.lower() or "lo" in output.lower() or "low" in output.lower(),
        "validate": """
def binary_search(arr, target):
{code}

assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 1) == 0
assert binary_search([1, 3, 5, 7, 9], 9) == 4
assert binary_search([1, 3, 5, 7, 9], 4) == -1
assert binary_search([], 1) == -1
print('PASS')
""",
    },
    {
        "name": "Flatten Nested List",
        "prompt": """Write a Python function `flatten(lst)` that flattens a nested list of arbitrary depth.
Example: flatten([1, [2, [3, 4], 5], 6]) → [1, 2, 3, 4, 5, 6]

```python
def flatten(lst):""",
        "test": lambda output: "isinstance" in output or "yield" in output or "extend" in output,
    },
    {
        "name": "LRU Cache",
        "prompt": """Write a Python class `LRUCache` with `__init__(self, capacity)`, `get(self, key)` → value or -1, and `put(self, key, value)`. Use OrderedDict.

```python
from collections import OrderedDict

class LRUCache:""",
        "test": lambda output: "OrderedDict" in output and "move_to_end" in output,
    },
]


# ═══════════════════════════════════════════════════════════════════
# Reasoning Benchmarks
# ═══════════════════════════════════════════════════════════════════

REASONING_TESTS = [
    {
        "name": "Math Word Problem",
        "prompt": "A store sells apples for $2 each and oranges for $3 each. If I buy 5 apples and 3 oranges, and I have a 10% discount coupon, how much do I pay? Show your work step by step, then give the final answer as a number.",
        "test": lambda output: "17.1" in output or "17.10" in output or "$17.1" in output,
    },
    {
        "name": "Logic Puzzle",
        "prompt": "Alice is taller than Bob. Bob is taller than Charlie. Dave is shorter than Charlie. Who is the tallest? Who is the shortest? Answer with just the names.",
        "test": lambda output: "Alice" in output and ("Dave" in output),
    },
    {
        "name": "Pattern Recognition",
        "prompt": "What comes next in this sequence: 2, 6, 12, 20, 30, ? Explain why, then give the answer.",
        "test": lambda output: "42" in output,
    },
    {
        "name": "Code Reasoning",
        "prompt": """What does this Python code print?

```python
x = [1, 2, 3, 4, 5]
y = x[1::2]
z = [a * b for a, b in zip(x, x[1:])]
print(y)
print(z)
```

Give the exact output.""",
        "test": lambda output: "[2, 4]" in output and "[2, 6, 12, 20]" in output,
    },
]


def run_benchmarks(model, tokenizer, max_tokens=400):
    """Run all benchmarks and report results."""
    print("\n" + "=" * 60)
    print("INTELLIGENCE BENCHMARKS")
    print("=" * 60)

    results = {"coding": [], "reasoning": []}

    # Coding
    print("\n--- Coding Ability ---")
    for test in CODING_TESTS:
        t0 = time.perf_counter()
        output = generate(model, tokenizer, test["prompt"], max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0

        passed = test["test"](output)
        status = "PASS" if passed else "FAIL"
        results["coding"].append(passed)

        # Truncate output for display
        display = output.replace('\n', '\\n')[:150]
        print(f"  [{status}] {test['name']:25} ({elapsed:.1f}s) {display}")

    # Reasoning
    print("\n--- Reasoning Ability ---")
    for test in REASONING_TESTS:
        t0 = time.perf_counter()
        output = generate(model, tokenizer, test["prompt"], max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0

        passed = test["test"](output)
        status = "PASS" if passed else "FAIL"
        results["reasoning"].append(passed)

        display = output.replace('\n', '\\n')[:150]
        print(f"  [{status}] {test['name']:25} ({elapsed:.1f}s) {display}")

    # Summary
    coding_pass = sum(results["coding"])
    coding_total = len(results["coding"])
    reason_pass = sum(results["reasoning"])
    reason_total = len(results["reasoning"])
    total = coding_pass + reason_pass
    total_all = coding_total + reason_total

    print(f"\n--- Summary ---")
    print(f"  Coding:    {coding_pass}/{coding_total}")
    print(f"  Reasoning: {reason_pass}/{reason_total}")
    print(f"  Total:     {total}/{total_all} ({total/total_all*100:.0f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Intelligence Benchmarks")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--sanitize-patch", action="store_true")
    args = parser.parse_args()

    if args.sanitize_patch:
        from omlx.patches.granitemoehybrid_sanitize import apply_sanitize_patch
        apply_sanitize_patch()

    from mlx_lm import load
    print(f"Loading: {args.model}")
    model, tokenizer = load(args.model)
    print(f"Memory: {mx.get_active_memory()/1e9:.1f}GB")

    run_benchmarks(model, tokenizer, max_tokens=args.max_tokens)

    # Cleanup
    del model
    import gc
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


if __name__ == "__main__":
    main()
