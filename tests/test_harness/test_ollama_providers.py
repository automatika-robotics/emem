"""Tests for Ollama providers — requires running Ollama server."""

import numpy as np
import pytest

pytestmark = pytest.mark.ollama


class TestOllamaEmbeddingProvider:
    def test_dim_property(self):
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        assert isinstance(provider.dim, int)
        assert provider.dim > 0

    def test_embed_single(self):
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        result = provider.embed(["hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, provider.dim)
        assert result.dtype == np.float32

    def test_embed_batch(self):
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        texts = ["hello", "world", "testing embeddings"]
        result = provider.embed(texts)
        assert result.shape == (3, provider.dim)

    def test_embed_empty(self):
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        result = provider.embed([])
        assert result.shape == (0, provider.dim)

    def test_similar_texts_closer(self):
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider()
        vecs = provider.embed(["cat", "kitten", "database server"])
        # cat and kitten should be more similar than cat and database
        sim_close = np.dot(vecs[0], vecs[1])
        sim_far = np.dot(vecs[0], vecs[2])
        assert sim_close > sim_far


class TestOllamaLLMClient:
    def test_summarize(self):
        from harness.providers.ollama_llm import OllamaLLMClient

        client = OllamaLLMClient()
        result = client.summarize(["I see a red door.", "There is a table nearby."])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_synthesize(self):
        from harness.providers.ollama_llm import OllamaLLMClient

        client = OllamaLLMClient()
        result = client.synthesize(
            {
                "description": ["A small room with a door."],
                "place": ["kitchen"],
            }
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_entities(self):
        from harness.providers.ollama_llm import OllamaLLMClient

        client = OllamaLLMClient()
        result = client.extract_entities(
            ["I see a red door and a wooden table."]
        )
        assert isinstance(result, list)
        # Should find at least one entity
        if result:
            assert "name" in result[0]
