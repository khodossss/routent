"""MMLU loader: massive multitask multiple-choice classification.

Schema per item:
    id: int
    question: str           - question formatted with A/B/C/D choices
    answer: str             - correct letter "A" | "B" | "C" | "D"
    choices: List[str]      - ["A", "B", "C", "D"]
    category: str           - the MMLU subject (e.g. "high_school_physics")
    metadata: dict          - {"raw_choices": [c0, c1, c2, c3], "subject": str}
"""

from typing import List, Optional


_LETTERS = ["A", "B", "C", "D"]


def load(
    split: str = "test",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> List[dict]:
    """Load MMLU (cais/mmlu, config='all')."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load MMLU. "
            "Install via: pip install datasets"
        ) from e

    try:
        ds = load_dataset("cais/mmlu", "all", split=split, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load cais/mmlu (split={split}): {e}"
        ) from e

    items = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        question_text = row["question"]
        raw_choices = list(row["choices"])
        answer_idx = int(row["answer"])
        subject = row.get("subject", "mmlu")

        formatted = (
            f"{question_text}\n"
            f"A: {raw_choices[0]}\n"
            f"B: {raw_choices[1]}\n"
            f"C: {raw_choices[2]}\n"
            f"D: {raw_choices[3]}"
        )

        items.append({
            "id": i + 1,
            "question": formatted,
            "answer": _LETTERS[answer_idx],
            "choices": list(_LETTERS),
            "category": subject,
            "metadata": {"raw_choices": raw_choices, "subject": subject},
        })

    return items
