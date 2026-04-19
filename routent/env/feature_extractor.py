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
    """Feature extractor producing LinUCB-ready feature vectors from prompts.

    Pipeline applied to every prompt:

        raw sentence embedding   (MiniLM, default 384-d)
            ↓ subtract mean, divide by std  (zero-mean / unit-variance)
            ↓ optional PCA projection       (``pca_dim`` components)
            ↓ optional constant-1 bias      (``prepend_bias``)
        final feature vector used by LinUCB

    **Bias term.** Zero-mean centred embeddings strip out the per-arm
    baseline reward signal: without an explicit bias dimension, LinUCB's
    linear predictor ``θ·x`` averages to zero at the corpus mean and cannot
    encode "arm A is uniformly better than arm B". Prepending a constant 1.0
    recovers the intercept. Default on; turn off only to reproduce legacy
    runs that shipped before this fix.

    **PCA.** Reduces dimensionality so LinUCB's O(d²) per-arm matrices stay
    well-conditioned on small training sets. ``pca_dim=32`` is a robust
    default for MiniLM 384-d embeddings.

    fit() computes per-dimension mean/std over the training corpus and, if
    ``pca_dim`` is set, fits a PCA on the centred embeddings. The complete
    state (mean, std, PCA components, PCA internal mean, bias flag) is
    serialisable so the same transform can be replayed at inference time.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, CPU-friendly).
    Other good options: all-mpnet-base-v2 (768d, higher quality).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        pca_dim: Optional[int] = None,
        prepend_bias: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from routent.models.generative import print_download_notice
        print_download_notice(model_name)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        # Prefer new API, fall back to old one for older sentence-transformers
        if hasattr(self._model, "get_embedding_dimension"):
            self._raw_dim = self._model.get_embedding_dimension()
        else:
            self._raw_dim = self._model.get_sentence_embedding_dimension()
        self._pca_dim = int(pca_dim) if pca_dim else None
        self._prepend_bias = bool(prepend_bias)

        self._is_fitted = False
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None  # (pca_dim, raw_dim)
        self._pca_input_mean: Optional[np.ndarray] = None  # (raw_dim,)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        dim = self._pca_dim if self._pca_dim is not None else self._raw_dim
        if self._prepend_bias:
            dim += 1
        return dim

    @property
    def raw_dim(self) -> int:
        """Dimensionality of the underlying sentence-embedding model (pre-PCA)."""
        return self._raw_dim

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str]) -> None:
        """Encode corpus, compute centering stats, and optionally fit PCA."""
        if not corpus:
            raise ValueError("Cannot fit on an empty corpus.")
        emb = self._model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
        self._mean = emb.mean(axis=0).astype(np.float32)
        self._std = emb.std(axis=0).astype(np.float32)
        # Avoid division by zero on constant dimensions
        self._std = np.where(self._std < 1e-8, 1.0, self._std)

        if self._pca_dim is not None:
            if self._pca_dim > self._raw_dim:
                raise ValueError(
                    f"pca_dim={self._pca_dim} exceeds embedding dim {self._raw_dim}"
                )
            from sklearn.decomposition import PCA
            centered = (emb - self._mean) / self._std
            pca = PCA(n_components=self._pca_dim, random_state=0)
            pca.fit(centered)
            self._pca_components = pca.components_.astype(np.float32)  # (k, d)
            self._pca_input_mean = pca.mean_.astype(np.float32)         # (d,)

        self._is_fitted = True

    def load_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        pca_components: Optional[np.ndarray] = None,
        pca_input_mean: Optional[np.ndarray] = None,
    ) -> None:
        """Load pre-computed state (from checkpoint) instead of calling fit()."""
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        if pca_components is not None:
            self._pca_components = np.asarray(pca_components, dtype=np.float32)
            if self._pca_dim is None:
                self._pca_dim = int(self._pca_components.shape[0])
        if pca_input_mean is not None:
            self._pca_input_mean = np.asarray(pca_input_mean, dtype=np.float32)
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _apply_post_encoding(self, emb: np.ndarray) -> np.ndarray:
        """Common centering + PCA + bias pipeline shared by single/batch paths."""
        assert self._mean is not None and self._std is not None, \
            "FeatureExtractor has not been fitted."
        centered = (emb - self._mean) / self._std
        if self._pca_components is not None:
            # Replicate sklearn PCA.transform(X) = (X - pca.mean_) @ components_.T
            shifted = centered - self._pca_input_mean
            projected = shifted @ self._pca_components.T
            centered = projected
        if self._prepend_bias:
            if centered.ndim == 1:
                centered = np.concatenate([[1.0], centered])
            else:
                ones = np.ones((centered.shape[0], 1), dtype=centered.dtype)
                centered = np.concatenate([ones, centered], axis=1)
        return centered

    def transform(self, prompt: str) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor has not been fitted. Call fit() first.")
        emb = self._model.encode(prompt, convert_to_numpy=True, show_progress_bar=False)
        out = self._apply_post_encoding(emb)
        return torch.tensor(out, dtype=torch.float32)

    def transform_batch(self, prompts: List[str]) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor has not been fitted. Call fit() first.")
        emb = self._model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
        out = self._apply_post_encoding(emb)
        return torch.tensor(out, dtype=torch.float32)
