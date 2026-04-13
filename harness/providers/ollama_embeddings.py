from __future__ import annotations

import numpy as np

from harness.providers.http import post_json


class OllamaEmbeddingProvider:
    """Embedding provider backed by an Ollama model.

    Implements the :class:`~emem.embeddings.EmbeddingProvider` protocol.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text-v2-moe:latest",
        base_url: str = "http://localhost:11434",
        batch_size: int = 50,
    ):
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._batch_size = batch_size
        probe = self._request(["hello"])
        self._dim = len(probe[0])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        :param texts: Input strings.
        :returns: Array of shape ``(len(texts), dim)``.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            vecs.extend(self._request(texts[i : i + self._batch_size]))
        return np.array(vecs, dtype=np.float32)

    def _request(self, texts: list[str]) -> list[list[float]]:
        data = post_json(self._url, {"model": self._model, "input": texts})
        return data["embeddings"]
