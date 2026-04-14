"""Dataset loader for HuggingFace benchmarks."""

import os
import re
from typing import List, Optional


def load_dataset_hf(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[dict]:
    """Load a HuggingFace dataset and convert to benchmark format.

    Each item has: id, question, answer, category.

    Supported datasets:
        - openai/gsm8k: grade school math

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split ("train" or "test").
        max_samples: Maximum number of samples to load.

    Returns:
        List of benchmark dicts.
    """
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")

    if "gsm8k" in dataset_name.lower():
        return _load_gsm8k(dataset_name, split, max_samples, token)
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not yet supported. "
            f"Supported: openai/gsm8k. "
            f"Add a loader in dataset_loader.py."
        )


def _load_gsm8k(
    dataset_name: str,
    split: str,
    max_samples: Optional[int],
    token: Optional[str] = None,
) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, "main", split=split, token=token)

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
        })

    return items


def _extract_gsm8k_answer(raw_answer: str) -> str:
    """Extract the final numeric answer from GSM8K's '####' format."""
    if "####" in raw_answer:
        final = raw_answer.split("####")[-1].strip()
        final = final.replace(",", "")
        return final
    numbers = re.findall(r"-?\d+\.?\d*", raw_answer)
    return numbers[-1] if numbers else raw_answer.strip()


def load_benchmark(config) -> tuple:
    """Load benchmark from HuggingFace dataset.

    Returns:
        (train_data, test_data)
    """
    total_needed = config.train_size + config.test_size
    print(f"Loading dataset: {config.dataset} (split={config.dataset_split}, "
          f"train_size={config.train_size}, test_size={config.test_size})")
    data = load_dataset_hf(
        config.dataset,
        split=config.dataset_split,
        max_samples=total_needed,
    )

    train_data = data[:config.train_size]
    test_data = data[config.train_size:config.train_size + config.test_size]
    print(f"  {len(train_data)} train / {len(test_data)} test")
    return train_data, test_data
