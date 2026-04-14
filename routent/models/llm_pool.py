"""Base class for LLM models."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseLLM(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    def generate(
        self, prompt: str, answer: str, difficulty: str, category: str
    ) -> Tuple[str, float, float]:
        """Generate a response for the given prompt.

        Args:
            prompt: The input question/prompt.
            answer: The ground truth answer (used for simulated models).
            difficulty: Question difficulty metadata.
            category: Question category metadata.

        Returns:
            Tuple of (predicted_answer, latency_ms, cost).
        """
        pass
