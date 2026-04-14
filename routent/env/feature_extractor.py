"""Feature extraction for converting prompt strings into fixed-size torch tensors."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


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


class TfidfFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor combining TF-IDF features with handcrafted features.

    Produces a tensor of shape (tfidf_max_features + 10,) where the first
    tfidf_max_features dimensions are TF-IDF features and the last 10 are
    handcrafted features min-max normalized to [0, 1].
    """

    MATH_SYMBOLS = set("+-*/=^")
    CODE_KEYWORDS = {
        "function", "def", "class", "return", "print", "import", "if",
        "else", "for", "while", "var", "let", "const", "int", "float",
        "string", "void", "public", "private", "static",
    }

    def __init__(self, tfidf_max_features: int = 100) -> None:
        self._tfidf_max_features = tfidf_max_features
        self._vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        self._is_fitted = False

        # Min/max stats for handcrafted features, computed during fit().
        # Each array has length 10 (one entry per handcrafted feature).
        self._hc_min: Optional[np.ndarray] = None
        self._hc_max: Optional[np.ndarray] = None

    @property
    def feature_dim(self) -> int:
        if self._is_fitted:
            return len(self._vectorizer.vocabulary_) + 10
        return self._tfidf_max_features + 10

    # ------------------------------------------------------------------
    # Handcrafted feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences by splitting on sentence-ending punctuation."""
        parts = re.split(r"[.!?]+", text)
        # Filter out empty strings that result from splitting
        return max(len([p for p in parts if p.strip()]), 1)

    def _extract_handcrafted(self, prompt: str) -> np.ndarray:
        """Return a length-10 numpy array of raw (un-normalised) handcrafted features."""
        words = prompt.split()
        num_tokens = len(words) if words else 1
        num_chars = len(prompt)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
        num_sentences = self._count_sentences(prompt)
        has_math = float(any(ch in self.MATH_SYMBOLS for ch in prompt))
        lower_words = set(w.lower().strip("(){}[];:,.'\"") for w in words)
        has_code = float(bool(lower_words & self.CODE_KEYWORDS))
        has_question_mark = float("?" in prompt)
        num_numbers = sum(1 for w in words if re.search(r"\d", w))
        unique_words = len(set(w.lower() for w in words))
        vocabulary_richness = unique_words / num_tokens if num_tokens else 0.0
        max_word_length = max((len(w) for w in words), default=0)

        return np.array(
            [
                num_tokens,
                num_chars,
                avg_word_length,
                num_sentences,
                has_math,
                has_code,
                has_question_mark,
                num_numbers,
                vocabulary_richness,
                max_word_length,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str]) -> None:
        """Fit TF-IDF vectorizer and compute min/max stats for handcrafted features."""
        if not corpus:
            raise ValueError("Cannot fit on an empty corpus.")

        # Fit TF-IDF
        self._vectorizer.fit(corpus)

        # Compute handcrafted features for the whole corpus to get min/max
        hc_matrix = np.stack([self._extract_handcrafted(p) for p in corpus])
        self._hc_min = hc_matrix.min(axis=0)
        self._hc_max = hc_matrix.max(axis=0)

        self._is_fitted = True

    def transform(self, prompt: str) -> torch.Tensor:
        """Transform a single prompt into a feature tensor of shape (feature_dim,)."""
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureExtractor has not been fitted. Call fit() first."
            )

        # TF-IDF features (sparse -> dense)
        tfidf_sparse = self._vectorizer.transform([prompt])
        tfidf_dense = np.asarray(tfidf_sparse.todense(), dtype=np.float32).flatten()

        # Handcrafted features (raw -> min-max normalised)
        hc_raw = self._extract_handcrafted(prompt)
        denom = self._hc_max - self._hc_min
        # Avoid division by zero: if min == max the feature is constant -> 0
        denom = np.where(denom == 0, 1.0, denom)
        hc_norm = (hc_raw - self._hc_min) / denom
        hc_norm = np.clip(hc_norm, 0.0, 1.0)

        combined = np.concatenate([tfidf_dense, hc_norm])
        return torch.tensor(combined, dtype=torch.float32)


class SentenceEmbeddingFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using sentence-transformers embeddings.

    Produces semantic embeddings that capture meaning far better than TF-IDF,
    especially for cross-domain routing.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, CPU-friendly).
    Other good options: all-mpnet-base-v2 (768d, higher quality).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from routent.models.real_llm import _print_download_notice
        _print_download_notice(model_name)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._is_fitted = False

    @property
    def feature_dim(self) -> int:
        return self._dim

    def fit(self, corpus: List[str]) -> None:
        """No fitting needed — embeddings are pretrained."""
        self._is_fitted = True

    def transform(self, prompt: str) -> torch.Tensor:
        """Transform a single prompt into a sentence embedding."""
        emb = self._model.encode(prompt, convert_to_numpy=True, show_progress_bar=False)
        return torch.tensor(emb, dtype=torch.float32)

    def transform_batch(self, prompts: List[str]) -> torch.Tensor:
        """Batch encode — much faster than per-prompt transform()."""
        emb = self._model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
        return torch.tensor(emb, dtype=torch.float32)
