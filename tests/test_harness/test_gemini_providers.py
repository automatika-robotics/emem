"""Tests for Gemini providers — requires GEMINI_API_KEY."""

import numpy as np
import pytest

pytestmark = pytest.mark.gemini


class TestGeminiEmbeddingProvider:
    def test_dim_property(self):
        from harness.providers.gemini_embeddings import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(dim=768)
        assert provider.dim == 768

    def test_embed_single(self):
        from harness.providers.gemini_embeddings import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider()
        result = provider.embed(["hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, provider.dim)
        assert result.dtype == np.float32

    def test_embed_batch(self):
        from harness.providers.gemini_embeddings import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider()
        result = provider.embed(["hello", "world", "test"])
        assert result.shape == (3, provider.dim)

    def test_embed_empty(self):
        from harness.providers.gemini_embeddings import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider()
        result = provider.embed([])
        assert result.shape == (0, provider.dim)

    def test_similar_texts_closer(self):
        from harness.providers.gemini_embeddings import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider()
        vecs = provider.embed(["cat", "kitten", "database server"])
        sim_close = np.dot(vecs[0], vecs[1])
        sim_far = np.dot(vecs[0], vecs[2])
        assert sim_close > sim_far


class TestGeminiLLMClient:
    def test_summarize(self):
        from harness.providers.gemini_llm import GeminiLLMClient

        client = GeminiLLMClient()
        result = client.summarize(["I see a red door.", "There is a table nearby."])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_image(self):
        from harness.providers.gemini_vlm import GeminiVLM

        vlm = GeminiVLM()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:32, :, 0] = 255  # red top
        image[32:, :, 2] = 255  # blue bottom

        result = vlm.describe(image, "Describe what you see.")
        assert isinstance(result, str)
        assert len(result) > 0
