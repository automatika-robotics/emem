import os

import numpy as np

from harness.providers.http import post_json_with_retry

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiEmbeddingProvider:
    """Embedding provider backed by the Gemini Embedding API.

    Implements the :class:`~emem.embeddings.EmbeddingProvider` protocol.
    Uses ``batchEmbedContents`` for efficient batching.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        dim: int = 768,
        batch_size: int = 50,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "Gemini API key required: pass api_key= or set GEMINI_API_KEY"
            )
        self._dim = dim
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts via Gemini batch API.

        :param texts: Input strings.
        :returns: Array of shape ``(len(texts), dim)``.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            vecs.extend(self._batch_embed(texts[i : i + self._batch_size]))
        return np.array(vecs, dtype=np.float32)

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        model_path = f"models/{self._model}"
        url = f"{_BASE}/{model_path}:batchEmbedContents?key={self._api_key}"
        requests = [
            {
                "model": model_path,
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self._dim,
            }
            for t in texts
        ]
        data = post_json_with_retry(url, {"requests": requests})
        return [emb["values"] for emb in data["embeddings"]]
