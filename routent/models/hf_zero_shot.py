"""Zero-shot classifier wrapper (e.g. facebook/bart-large-mnli) as a BaseLLM."""

from __future__ import annotations

import time
from typing import List, Tuple

from routent.models.base import BaseLLM
from routent.models.generative import print_download_notice


class HFZeroShotLLM(BaseLLM):
    """Wraps a HuggingFace zero-shot NLI classifier as a routable model.

    The user supplies candidate labels at construction time; at inference,
    the pipeline scores each label against the prompt using an entailment
    hypothesis template. No fine-tuning is required.
    """

    def __init__(
        self,
        model_name: str,
        labels: List[str],
        hypothesis_template: str = "This example is {}.",
        multilabel: bool = False,
        threshold: float = 0.5,
        device: int = -1,
        cost_per_1m_input: float = 0.0,
    ) -> None:
        if not labels:
            raise ValueError("HFZeroShotLLM requires a non-empty `labels` list.")

        from transformers import pipeline  # lazy import

        print_download_notice(model_name)

        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )

        self.name = f"hf_zero_shot/{model_name}"
        self.model_name = model_name
        self.labels = list(labels)
        self.hypothesis_template = hypothesis_template
        self.multilabel = multilabel
        self.threshold = threshold
        self.cost_per_1m_input = cost_per_1m_input
        self.cost_per_1m_output = 0.0

    def generate(
        self,
        prompt: str,
        answer: str = "",
        difficulty: str = "",
        category: str = "",
    ) -> Tuple[str, float, float]:
        start = time.perf_counter()
        try:
            result = self._pipe(
                prompt,
                candidate_labels=self.labels,
                hypothesis_template=self.hypothesis_template,
                multi_label=self.multilabel,
            )

            labels_sorted = result["labels"]
            scores_sorted = result["scores"]

            if self.multilabel:
                kept = [
                    lab
                    for lab, sc in zip(labels_sorted, scores_sorted)
                    if sc >= self.threshold
                ]
                prediction = ", ".join(kept)
            else:
                prediction = labels_sorted[0]

            latency_ms = (time.perf_counter() - start) * 1000.0
            return prediction, latency_ms, 0.0
        except Exception as e:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return f"ERROR: {e}", latency_ms, 0.0

    def batch_generate(self, prompts):
        """Sequential calls so per-item latency is measured accurately."""
        return [self.generate(p, "", "", "") for p in prompts]
