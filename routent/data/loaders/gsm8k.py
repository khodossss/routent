"""GSM8K loader: grade school math word problems (numeric answers).

Schema per item:
    id: int
    question: str           - the word problem
    answer: str             - the final numeric answer (string form)
    category: str           - "math"
    metadata: dict          - {"raw_answer": str} full chain-of-thought solution
"""

import re
from typing import List, Optional


def _extract_gsm8k_answer(raw_answer: str) -> str:
    """Extract the final numeric answer from GSM8K's '####' format."""
    if "####" in raw_answer:
        final = raw_answer.split("####")[-1].strip()
        final = final.replace(",", "")
        return final
    numbers = re.findall(r"-?\d+\.?\d*", raw_answer)
    return numbers[-1] if numbers else raw_answer.strip()


def load(
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> List[dict]:
    """Load GSM8K (openai/gsm8k, config='main')."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load GSM8K. "
            "Install via: pip install datasets"
        ) from e

    try:
        ds = load_dataset("openai/gsm8k", "main", split=split, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load openai/gsm8k (split={split}): {e}"
        ) from e

    items = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        question = row["question"]
        raw_answer = row["answer"]
        final_answer = _extract_gsm8k_answer(raw_answer)

        items.append({
            "id": i + 1,
            "question": question,
            "answer": final_answer,
            "category": "math",
            "metadata": {"raw_answer": raw_answer},
        })

    return items
