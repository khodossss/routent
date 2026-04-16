"""HuggingFace sequence classification model wrapped as a BaseLLM."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from routent.models.base import BaseLLM


class HFClassifierLLM(BaseLLM):
    """A BERT-style classifier (AutoModelForSequenceClassification) exposed
    through the BaseLLM interface.

    Supports both single-label (argmax) and multilabel (sigmoid + threshold)
    classification. Inference runs locally, so cost is zero.
    """

    def __init__(
        self,
        model_name: str,
        label_map: Optional[Dict[int, str]] = None,
        multilabel: bool = False,
        threshold: float = 0.5,
        device: int = -1,
        cost_per_1m_input: float = 0.0,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from routent.models.generative import print_download_notice

        print_download_notice(model_name)

        self._torch = torch
        self.model_name = model_name
        self.multilabel = multilabel
        self.threshold = threshold
        self.device = device
        self.max_length = max_length

        # BaseLLM attributes
        self.name = f"hf_classifier/{model_name}"
        self.cost_per_1m_input = cost_per_1m_input
        self.cost_per_1m_output = 0.0

        # Load tokenizer + model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

        if device is not None and device >= 0:
            self._torch_device = torch.device(f"cuda:{device}")
            self._model.to(self._torch_device)
        else:
            self._torch_device = torch.device("cpu")

        # Resolve label map
        if label_map is not None:
            self.label_map = label_map
        else:
            # model.config.id2label keys may be str or int — normalize to int
            raw = getattr(self._model.config, "id2label", {})
            self.label_map = {int(k): str(v) for k, v in raw.items()}

    def generate(
        self,
        prompt: str,
        answer: str = "",
        difficulty: str = "",
        category: str = "",
    ) -> Tuple[str, float, float]:
        torch = self._torch
        start = time.perf_counter()
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
            logits = outputs.logits[0]  # shape: (num_labels,)

            if self.multilabel:
                probs = torch.sigmoid(logits)
                selected = [
                    self.label_map.get(i, str(i))
                    for i, p in enumerate(probs.tolist())
                    if p >= self.threshold
                ]
                prediction = ",".join(selected)
            else:
                idx = int(torch.argmax(logits).item())
                prediction = self.label_map.get(idx, str(idx))

            latency_ms = (time.perf_counter() - start) * 1000.0
            return prediction, latency_ms, 0.0
        except Exception as e:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return f"ERROR: {e}", latency_ms, 0.0

    def batch_generate(
        self,
        prompts: List[str],
        answers: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        # Sequential keeps per-item latency accurate; classifier inference is fast.
        return [self.generate(p, "", "", "") for p in prompts]
