"""Dataset loader dispatcher.

Routes a dataset name to the appropriate modular loader in
`routent.data.loaders.*`. Each loader implements:

    def load(split: str = "train",
             max_samples: Optional[int] = None,
             token: Optional[str] = None) -> List[dict]: ...

and returns benchmark items with (at least) the keys: id, question, answer,
category, metadata. Classification/regression loaders additionally provide
`choices`.
"""

import os
from typing import List, Optional


def load_dataset_hf(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[dict]:
    """Load a HuggingFace dataset and convert to benchmark format.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "openai/gsm8k",
            "cais/mmlu", "glue/sst2", "glue/stsb", "lmsys/mt_bench_human_judgments",
            "google-research-datasets/go_emotions").
        split: Dataset split ("train", "validation", "test").
        max_samples: Maximum number of samples to load.

    Returns:
        List of benchmark dicts.

    Raises:
        ValueError: if the dataset name is not recognized.
    """
    token = os.environ.get("HF_TOKEN")
    name = dataset_name.lower()

    if "gsm8k" in name:
        from routent.data.loaders.gsm8k import load as _load
    elif "mmlu" in name:
        from routent.data.loaders.mmlu import load as _load
    elif "sst2" in name or "sst-2" in name or "glue/sst2" in name:
        from routent.data.loaders.sst2 import load as _load
    elif "stsb" in name or "sts-b" in name or "glue/stsb" in name:
        from routent.data.loaders.stsb import load as _load
    elif "mt_bench" in name or "mt-bench" in name:
        from routent.data.loaders.mt_bench import load as _load
    elif "goemotions" in name or "go_emotions" in name:
        from routent.data.loaders.goemotions import load as _load
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Supported: gsm8k, mmlu, "
            f"sst2, stsb, mt_bench, goemotions. Add a loader in "
            f"routent/data/loaders/ and register it in dataset_loader.py."
        )

    return _load(split=split, max_samples=max_samples, token=token)


def load_benchmark(config) -> tuple:
    """Load benchmark from HuggingFace dataset and split into train/test.

    Returns:
        (train_data, test_data)
    """
    total_needed = config.train_size + config.test_size
    print(
        f"Loading dataset: {config.dataset} (split={config.dataset_split}, "
        f"train_size={config.train_size}, test_size={config.test_size})"
    )
    data = load_dataset_hf(
        config.dataset,
        split=config.dataset_split,
        max_samples=total_needed,
    )

    train_data = data[: config.train_size]
    test_data = data[config.train_size : config.train_size + config.test_size]
    print(f"  {len(train_data)} train / {len(test_data)} test")
    return train_data, test_data
