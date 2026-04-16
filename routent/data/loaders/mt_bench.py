"""MT-Bench loader: open-ended LLM-judge style prompts.

Schema per item:
    id: int
    question: str           - first-turn prompt
    answer: str             - reference answer if available, else placeholder
    choices: List[str]      - [] (open-ended)
    category: str           - row["category"] if present, else "general"
    metadata: dict          - {"question_id": int, "turns": List[str],
                               "has_reference": bool}

Note:
    MT-Bench is an open-ended benchmark. Evaluation typically uses an LLM judge
    that either (a) compares the model response to a reference answer, or
    (b) compares two model responses head-to-head. When no reference is
    available in the dataset, `answer` is set to a placeholder string and
    `metadata["has_reference"]` is False; downstream code should then route
    to judge-based scoring rather than exact/fuzzy match.
"""

from typing import List, Optional


_PLACEHOLDER = "[no reference answer - use LLM judge]"


def load(
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> List[dict]:
    """Load MT-Bench (tries lmsys/mt_bench_human_judgments, then HuggingFaceH4/mt_bench_prompts)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load MT-Bench. "
            "Install via: pip install datasets"
        ) from e

    ds = None
    last_err = None
    for name in ("lmsys/mt_bench_human_judgments", "HuggingFaceH4/mt_bench_prompts"):
        try:
            ds = load_dataset(name, split=split, token=token)
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(
            f"Failed to load MT-Bench (split={split}): {last_err}"
        )

    items = []
    seen_qids = set()
    for i, row in enumerate(ds):
        if max_samples is not None and len(items) >= max_samples:
            break

        # Deduplicate on question_id when present (judgments datasets repeat qs).
        qid = row.get("question_id", i + 1)
        if qid in seen_qids:
            continue
        seen_qids.add(qid)

        turns = row.get("turns") or row.get("prompt") or []
        if isinstance(turns, str):
            turns = [turns]
        if not turns:
            continue
        question = turns[0]

        reference = row.get("reference")
        if isinstance(reference, list):
            reference = reference[0] if reference else None
        has_reference = bool(reference)
        answer = reference if has_reference else _PLACEHOLDER

        category = row.get("category", "general")

        items.append({
            "id": int(qid) if isinstance(qid, int) else i + 1,
            "question": question,
            "answer": answer,
            "choices": [],
            "category": category,
            "metadata": {
                "question_id": qid,
                "turns": list(turns),
                "has_reference": has_reference,
            },
        })

    return items
