"""GoEmotions loader: multi-label emotion classification.

Schema per item:
    id: int
    question: str           - the input text
    answer: str             - comma-joined emotion labels (e.g. "joy,admiration")
    choices: List[str]      - all possible emotion label names
    category: str           - "emotion"
    metadata: dict          - {"label_ids": List[int], "label_names": List[str]}
"""

from typing import List, Optional


def load(
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> List[dict]:
    """Load GoEmotions (google-research-datasets/go_emotions, config='simplified')."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load GoEmotions. "
            "Install via: pip install datasets"
        ) from e

    try:
        ds = load_dataset(
            "google-research-datasets/go_emotions",
            "simplified",
            split=split,
            token=token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load go_emotions (split={split}): {e}"
        ) from e

    # Pull the full label-name vocabulary from the feature schema.
    try:
        all_labels = list(ds.features["labels"].feature.names)
    except Exception:
        all_labels = []

    items = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        text = row["text"]
        label_ids = [int(x) for x in row["labels"]]
        if all_labels:
            label_names = [all_labels[j] for j in label_ids if 0 <= j < len(all_labels)]
        else:
            label_names = [str(j) for j in label_ids]

        items.append({
            "id": i + 1,
            "question": text,
            "answer": ",".join(label_names),
            "choices": list(all_labels),
            "category": "emotion",
            "metadata": {"label_ids": label_ids, "label_names": label_names},
        })

    return items
