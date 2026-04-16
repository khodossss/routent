"""Feature extraction for converting prompt strings into fixed-size torch tensors."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit the extractor on a corpus of prompt strings."""
        pass

    @abstractmethod
    def transform(self, prompt: str) -> torch.Tensor:
        """Transform a single prompt string into a feature tensor."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of the output feature vector."""
        pass


class SentenceEmbeddingFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using sentence-transformers embeddings.

    Produces semantic embeddings that capture meaning far better than TF-IDF,
    especially for cross-domain routing.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, CPU-friendly).
    Other good options: all-mpnet-base-v2 (768d, higher quality).

    fit() computes per-dimension mean and std over the training corpus so that
    transform() returns zero-mean, unit-variance vectors — essential for the
    linear reward model inside LinUCB (and beneficial for any policy network).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from routent.models.generative import print_download_notice
        print_download_notice(model_name)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        # Prefer new API, fall back to old one for older sentence-transformers
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = self._model.get_embedding_dimension()
        else:
            self._dim = self._model.get_sentence_embedding_dimension()
        self._is_fitted = False
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def feature_dim(self) -> int:
        return self._dim

    def fit(self, corpus: List[str]) -> None:
        """Encode corpus and compute per-dimension mean/std for centering."""
        if not corpus:
            raise ValueError("Cannot fit on an empty corpus.")
        emb = self._model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
        self._mean = emb.mean(axis=0).astype(np.float32)
        self._std = emb.std(axis=0).astype(np.float32)
        # Avoid division by zero on constant dimensions
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        self._is_fitted = True

    def load_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Load pre-computed centering stats (from checkpoint) instead of calling fit()."""
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self._is_fitted = True

    def transform(self, prompt: str) -> torch.Tensor:
        """Transform a single prompt into a centered, unit-variance embedding."""
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor has not been fitted. Call fit() first.")
        emb = self._model.encode(prompt, convert_to_numpy=True, show_progress_bar=False)
        emb = (emb - self._mean) / self._std
        return torch.tensor(emb, dtype=torch.float32)

    def transform_batch(self, prompts: List[str]) -> torch.Tensor:
        """Batch encode with centering — much faster than per-prompt transform()."""
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor has not been fitted. Call fit() first.")
        emb = self._model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
        emb = (emb - self._mean) / self._std
        return torch.tensor(emb, dtype=torch.float32)
