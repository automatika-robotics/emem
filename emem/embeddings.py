from typing import Callable, List, Optional, Protocol

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


class CallableEmbeddingProvider:
    """Wraps an embedding function into an :class:`EmbeddingProvider`.

    The function should accept a string or list of strings and return a list
    of embedding vectors (``List[List[float]]``).  This matches the signature
    of ``OllamaClient._embed()`` in EmbodiedAgents.

    Example::

        from emem.embeddings import CallableEmbeddingProvider

        provider = CallableEmbeddingProvider(ollama_client._embed)
        mem = SpatioTemporalMemory(embedding_provider=provider)

    :param embed_fn: Embedding function.
    :param dim: Embedding vector dimension.  When *None* (the default) the
        dimension is discovered by embedding a probe string.
    """

    def __init__(
        self,
        embed_fn: Callable,
        dim: Optional[int] = None,
    ):
        self._fn = embed_fn
        if dim is None:
            probe = self._fn(["hello"])
            self._dim = len(probe[0])
        else:
            self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        result = self._fn(texts)
        return np.array(result, dtype=np.float32)


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
