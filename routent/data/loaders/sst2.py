"""SST-2 loader: binary sentiment classification.

Schema per item:
    id: int
    question: str           - the input sentence
    answer: str             - "negative" or "positive"
    choices: List[str]      - ["negative", "positive"]
    category: str           - "sentiment"
    metadata: dict          - {"label": int}
"""

from typing import List, Optional


_LABELS = ["negative", "positive"]


def load(
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> List[dict]:
    """Load SST-2 (stanfordnlp/sst2, falling back to glue/sst2)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load SST-2. "
            "Install via: pip install datasets"
        ) from e

    ds = None
    last_err = None
    for loader_args in (("stanfordnlp/sst2",), ("glue", "sst2")):
        try:
            ds = load_dataset(*loader_args, split=split, token=token)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(
            f"Failed to load SST-2 (split={split}): {last_err}"
        )

    items = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        sentence = row["sentence"]
        label = int(row["label"])
        # GLUE test splits sometimes have label=-1 (unlabeled); skip those.
        if label < 0 or label >= len(_LABELS):
            continue

        items.append({
            "id": i + 1,
            "question": sentence,
            "answer": _LABELS[label],
            "choices": list(_LABELS),
            "category": "sentiment",
            "metadata": {"label": label},
        })

    return items
