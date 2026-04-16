"""STS-B loader: semantic textual similarity (regression, 0-5).

Schema per item:
    id: int
    question: str           - "Similarity between:\nA: {s1}\nB: {s2}"
    answer: str             - similarity score formatted to two decimals
    choices: List[str]      - [] (regression task)
    category: str           - "similarity"
    metadata: dict          - {"sentence1": str, "sentence2": str, "label": float}
"""

from typing import List, Optional


def load(
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> List[dict]:
    """Load STS-B (glue/stsb)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load STS-B. "
            "Install via: pip install datasets"
        ) from e

    try:
        ds = load_dataset("glue", "stsb", split=split, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load glue/stsb (split={split}): {e}"
        ) from e

    items = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        s1 = row["sentence1"]
        s2 = row["sentence2"]
        label = float(row["label"])
        # GLUE test split has label=-1 (unlabeled); skip those.
        if label < 0:
            continue

        question = f"Similarity between:\nA: {s1}\nB: {s2}"

        items.append({
            "id": i + 1,
            "question": question,
            "answer": f"{label:.2f}",
            "choices": [],
            "category": "similarity",
            "metadata": {"sentence1": s1, "sentence2": s2, "label": label},
        })

    return items
