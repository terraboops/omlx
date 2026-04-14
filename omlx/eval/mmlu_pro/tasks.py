# SPDX-License-Identifier: Apache-2.0
"""MMLU-Pro task loader and scoring (arXiv:2406.01574).

MMLU-Pro is a harder MMLU variant with 10 answer choices (A-J) and
chain-of-thought reasoning. 14 categories covering STEM, humanities,
social sciences, and more.

Usage:
    questions = load_mmlu_pro(categories=["computer_science", "math"], n=25)
    for q in questions:
        prompt = format_prompt(q)
        response = model.generate(prompt)
        answer = extract_answer(response)
        correct = answer == q["answer"]
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Categories relevant for a coding model
QUICK_CATEGORIES = ["computer_science", "math"]

# All 14 MMLU-Pro categories
ALL_CATEGORIES = [
    "biology", "business", "chemistry", "computer_science",
    "economics", "engineering", "health", "history",
    "law", "math", "philosophy", "physics",
    "psychology", "other",
]


def load_mmlu_pro(
    categories: Optional[list[str]] = None,
    n: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Load MMLU-Pro questions from HuggingFace.

    Args:
        categories: Which categories to include (default: all)
        n: Max questions to sample (default: all in selected categories)
        seed: Random seed for sampling

    Returns:
        List of question dicts with keys:
          question, options, answer, category, answer_index
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package not installed: pip install datasets")
        return []

    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception as e:
        logger.error(f"Failed to load MMLU-Pro dataset: {e}")
        return []

    cats = set(categories or ALL_CATEGORIES)
    questions = []

    for row in ds:
        cat = row.get("category", "other")
        if cat not in cats:
            continue

        # MMLU-Pro has 10 options (A-J)
        options = row.get("options", [])
        answer = row.get("answer", "")
        question_text = row.get("question", "")

        if not question_text or not options or not answer:
            continue

        questions.append({
            "question": question_text,
            "options": options,
            "answer": answer,  # Letter like "A", "B", etc.
            "category": cat,
        })

    if n is not None and n < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, n)

    logger.info(f"MMLU-Pro: loaded {len(questions)} questions from {len(cats)} categories")
    return questions


def format_prompt(q: dict) -> str:
    """Format an MMLU-Pro question as a chain-of-thought prompt.

    Uses the paper's recommended format: show all options, ask for
    reasoning then final answer.
    """
    options_str = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(q["options"])
    )

    return (
        f"The following is a multiple choice question about {q['category']}. "
        f"Think step by step and then output the answer in the format of "
        f"\"The answer is (X)\" at the end.\n\n"
        f"Question: {q['question']}\n"
        f"Options:\n{options_str}\n\n"
        f"Let me think step by step."
    )


def extract_answer(response: str) -> str:
    """Extract the answer letter from a model response.

    Looks for patterns like "The answer is (A)", "answer is B",
    "Answer: C", or just a standalone letter at the end.
    """
    # Pattern 1: "The answer is (X)" or "the answer is X"
    m = re.search(r"[Tt]he answer is \(?([A-J])\)?", response)
    if m:
        return m.group(1)

    # Pattern 2: "Answer: X" or "answer: X"
    m = re.search(r"[Aa]nswer:\s*\(?([A-J])\)?", response)
    if m:
        return m.group(1)

    # Pattern 3: Last standalone letter A-J in the response
    m = re.findall(r"\b([A-J])\b", response)
    if m:
        return m[-1]

    return ""
