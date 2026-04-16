"""Base abstract class for all model types in the routing pool."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseLLM(ABC):
    """Abstract interface for any model that can be routed to.

    Implementations:
    - GenerativeLLM: text generation via LangChain (OpenAI, Gemini, HF, ...)
    - HFClassifierLLM: BERT-style classifier (single or multi-label)
    - HFZeroShotLLM: zero-shot classifier (bart-large-mnli style)
    - HFRegressorLLM: BERT-style regressor (float output)

    All implementations must return (prediction, latency_ms, cost) where
    prediction is always a string (labels stringified, numbers formatted).
    """

    name: str
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    @abstractmethod
    def generate(
        self, prompt: str, answer: str, difficulty: str, category: str
    ) -> Tuple[str, float, float]:
        """Produce a prediction for the given prompt.

        Args:
            prompt: Input question/text.
            answer: Ground truth (used only by simulated models).
            difficulty: Optional metadata.
            category: Optional metadata.

        Returns:
            (prediction_str, latency_ms, cost_usd)
        """
        pass

    def batch_generate(self, prompts):
        """Default batch implementation: sequential calls.

        Subclasses may override for true batching.
        """
        return [self.generate(p, "", "", "") for p in prompts]
