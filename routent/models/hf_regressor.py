"""HuggingFace regression model wrapper (e.g. STS-B semantic similarity)."""

from __future__ import annotations

import time
from typing import List, Tuple

import torch

from routent.models.base import BaseLLM
from routent.models.generative import print_download_notice


class HFRegressorLLM(BaseLLM):
    """BERT-style regressor wrapped as a BaseLLM.

    Produces a single float prediction (e.g. STS-B similarity score) which is
    returned as a formatted string for compatibility with the BaseLLM interface.
    """

    def __init__(
        self,
        model_name: str,
        device: int = -1,
        cost_per_1m_input: float = 0.0,
        max_length: int = 512,
    ) -> None:
        print_download_notice(model_name)

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_name = model_name
        self.device = device
        self.cost_per_1m_input = cost_per_1m_input
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.model.eval()

        if device >= 0:
            self._torch_device = torch.device(f"cuda:{device}")
            self.model.to(self._torch_device)
        else:
            self._torch_device = torch.device("cpu")

        self.name = f"hf_regressor/{model_name}"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        assistant_prefix: str = "",
        stop_sequence: str = "",
    ) -> Tuple[str, float, float]:
        start = time.perf_counter()
        try:
            inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device >= 0:
                inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            value = outputs.logits.squeeze().item()
            prediction_str = f"{value:.4f}"
            latency_ms = (time.perf_counter() - start) * 1000.0
            return prediction_str, latency_ms, 0.0
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return f"ERROR: {e}", latency_ms, 0.0

    def batch_generate(
        self, prompts: List[str]
    ) -> List[Tuple[str, float, float]]:
        return [self.generate(p, "", "", "") for p in prompts]
