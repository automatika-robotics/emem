from typing import List, Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        :param texts: Input strings to embed.
        :returns: Array of shape ``(N, dim)``.
        :rtype: numpy.ndarray
        """
        ...


class NullEmbeddingProvider:
    """Returns zero vectors. Use when embeddings are pre-computed."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), self._dim), dtype=np.float32)


class SentenceTransformerProvider:
    """Wraps sentence-transformers for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install with: pip install emem[embeddings]"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
