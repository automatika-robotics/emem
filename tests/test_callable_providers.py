"""Tests for CallableEmbeddingProvider and InferenceLLMClient."""

import numpy as np
import pytest

from emem.consolidation import InferenceLLMClient, _parse_entities
from emem.embeddings import CallableEmbeddingProvider


# ── CallableEmbeddingProvider ────────────────────────────────────


class TestCallableEmbeddingProvider:
    def _mock_embed(self, input):
        """Mock embedding function returning 4-dim vectors."""
        if isinstance(input, str):
            input = [input]
        return [[1.0, 2.0, 3.0, 4.0] for _ in input]

    def test_dim_probed(self):
        provider = CallableEmbeddingProvider(self._mock_embed)
        assert provider.dim == 4

    def test_dim_explicit(self):
        calls = []

        def tracking_embed(texts):
            calls.append(texts)
            return [[0.0] * 8 for _ in texts]

        provider = CallableEmbeddingProvider(tracking_embed, dim=8)
        assert provider.dim == 8
        assert len(calls) == 0  # no probe call

    def test_embed_returns_ndarray(self):
        provider = CallableEmbeddingProvider(self._mock_embed)
        result = provider.embed(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)
        assert result.dtype == np.float32

    def test_embed_empty_list(self):
        provider = CallableEmbeddingProvider(self._mock_embed)
        result = provider.embed([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 4)

    def test_embed_single_text(self):
        provider = CallableEmbeddingProvider(self._mock_embed)
        result = provider.embed(["single"])
        assert result.shape == (1, 4)


# ── InferenceLLMClient ───────────────────────────────────────────


class TestInferenceLLMClient:
    def _mock_inference(self, input_dict):
        """Mock inference that echoes the prompt."""
        messages = input_dict["query"]
        prompt = messages[-1]["content"]
        return {"output": f"Summary of: {prompt[:50]}"}

    def test_summarize_formats_prompt(self):
        calls = []

        def capturing_inference(input_dict):
            calls.append(input_dict)
            return {"output": "A concise summary."}

        client = InferenceLLMClient(capturing_inference)
        result = client.summarize(["obs 1", "obs 2", "obs 3"])

        assert result == "A concise summary."
        assert len(calls) == 1
        prompt = calls[0]["query"][0]["content"]
        assert "1. obs 1" in prompt
        assert "2. obs 2" in prompt
        assert "3. obs 3" in prompt
        assert "Summarize" in prompt
        assert calls[0]["stream"] is False

    def test_synthesize_formats_layers(self):
        calls = []

        def capturing_inference(input_dict):
            calls.append(input_dict)
            return {"output": "Cross-layer synthesis."}

        client = InferenceLLMClient(capturing_inference)
        result = client.synthesize({
            "vlm": ["A kitchen with cabinets"],
            "detections": ["chair, table"],
        })

        assert result == "Cross-layer synthesis."
        prompt = calls[0]["query"][0]["content"]
        assert "[vlm]:" in prompt
        assert "[detections]:" in prompt
        assert "Synthesize" in prompt

    def test_extract_entities_parses_json(self):
        def json_inference(input_dict):
            return {
                "output": 'Here are the entities: [{"name": "red chair", '
                '"entity_type": "furniture", "confidence": 0.9}]'
            }

        client = InferenceLLMClient(json_inference)
        entities = client.extract_entities(["A red chair near the table"])

        assert len(entities) == 1
        assert entities[0]["name"] == "red chair"
        assert entities[0]["entity_type"] == "furniture"
        assert entities[0]["confidence"] == 0.9

    def test_extract_entities_handles_no_json(self):
        def no_json_inference(input_dict):
            return {"output": "I couldn't find any entities."}

        client = InferenceLLMClient(no_json_inference)
        entities = client.extract_entities(["some text"])
        assert entities == []

    def test_strips_think_tokens(self):
        def thinking_inference(input_dict):
            return {
                "output": "<think>Let me think about this...</think>The answer is 42."
            }

        client = InferenceLLMClient(thinking_inference)
        result = client.summarize(["obs 1"])
        assert "<think>" not in result
        assert "The answer is 42." in result

    def test_handles_none_result(self):
        def failing_inference(input_dict):
            return None

        client = InferenceLLMClient(failing_inference)
        result = client.summarize(["obs 1"])
        assert result == ""

    def test_handles_missing_output_key(self):
        def bad_inference(input_dict):
            return {"error": "something went wrong"}

        client = InferenceLLMClient(bad_inference)
        result = client.summarize(["obs 1"])
        assert result == ""

    def test_custom_temperature_and_tokens(self):
        calls = []

        def capturing_inference(input_dict):
            calls.append(input_dict)
            return {"output": "ok"}

        client = InferenceLLMClient(
            capturing_inference, temperature=0.7, max_new_tokens=1000
        )
        client.summarize(["text"])

        assert calls[0]["temperature"] == 0.7
        assert calls[0]["max_new_tokens"] == 1000


# ── _parse_entities ──────────────────────────────────────────────


class TestParseEntities:
    def test_valid_json(self):
        raw = '[{"name": "chair", "entity_type": "furniture", "confidence": 0.95}]'
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0]["name"] == "chair"
        assert result[0]["entity_type"] == "furniture"
        assert result[0]["confidence"] == 0.95

    def test_json_embedded_in_text(self):
        raw = 'Here are the entities: [{"name": "door"}] and more text'
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0]["name"] == "door"
        assert result[0]["confidence"] == 1.0  # default

    def test_no_json(self):
        result = _parse_entities("No entities found here.")
        assert result == []

    def test_invalid_json(self):
        result = _parse_entities("[not valid json]")
        assert result == []

    def test_missing_name_key(self):
        raw = '[{"entity_type": "furniture"}]'
        result = _parse_entities(raw)
        assert result == []

    def test_multiple_entities(self):
        raw = '[{"name": "a"}, {"name": "b", "entity_type": "x"}]'
        result = _parse_entities(raw)
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[1]["entity_type"] == "x"

    def test_empty_array(self):
        result = _parse_entities("[]")
        assert result == []
