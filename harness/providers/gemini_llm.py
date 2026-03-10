import json
import os
import re
from typing import Any

from harness.providers.http import post_json_with_retry, strip_think_tags

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiLLMClient:
    """LLM client backed by the Gemini generateContent API.

    Implements the :class:`~emem.consolidation.LLMClient` protocol.
    Also serves as VLM via :meth:`describe`.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        api_key: str | None = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "Gemini API key required: pass api_key= or set GEMINI_API_KEY"
            )
        self._url = f"{_BASE}/models/{self._model}:generateContent?key={self._api_key}"

    def summarize(self, texts: list[str]) -> str:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return self._generate(
            "Summarize the following observations into a concise paragraph. "
            "Preserve spatial and temporal details.\n\n" + numbered
        )

    def synthesize(self, layer_texts: dict[str, list[str]]) -> str:
        block = "\n".join(
            f"[{layer}]: {'; '.join(texts)}"
            for layer, texts in layer_texts.items()
        )
        return self._generate(
            "Synthesize the following observations grouped by perception layer "
            "into a coherent summary. Highlight agreements and contradictions.\n\n"
            + block
        )

    def extract_entities(self, texts: list[str]) -> list[dict[str, Any]]:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        raw = self._generate(
            "Extract named entities (objects, places, people) from these observations. "
            "Return ONLY a JSON array where each element has keys: "
            '"name" (string), "entity_type" (string or null), "confidence" (float 0-1).\n\n'
            + numbered
        )
        return _parse_entities(raw)

    def describe(
        self,
        image_b64: str,
        prompt: str,
        mime_type: str = "image/png",
        max_tokens: int | None = None,
    ) -> str:
        """Send a base64-encoded image with a text prompt.

        :param image_b64: Base64-encoded image data.
        :param prompt: Text prompt.
        :param mime_type: Image MIME type.
        :param max_tokens: Maximum tokens to generate.
        :returns: Model's text response.
        """
        payload: dict[str, Any] = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                    {"text": prompt},
                ],
            }],
        }
        if max_tokens is not None:
            payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        data = post_json_with_retry(self._url, payload)
        return _extract_text(data)

    def _generate(self, prompt: str) -> str:
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        data = post_json_with_retry(self._url, payload)
        return _extract_text(data)


def _extract_text(data: dict) -> str:
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return ""
    return strip_think_tags(text)


def _parse_entities(raw: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        entities = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return [
        {
            "name": str(e["name"]),
            "entity_type": e.get("entity_type"),
            "confidence": float(e.get("confidence", 1.0)),
        }
        for e in entities
        if isinstance(e, dict) and "name" in e
    ]
