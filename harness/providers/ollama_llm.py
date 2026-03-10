import json
import re
from typing import Any

from harness.providers.http import post_json, strip_think_tags


class OllamaLLMClient:
    """LLM client backed by an Ollama chat model.

    Implements the :class:`~emem.consolidation.LLMClient` protocol
    (required: ``summarize``; optional: ``synthesize``, ``extract_entities``).
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"

    def summarize(self, texts: list[str]) -> str:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return self._chat(
            "Summarize the following observations into a concise paragraph. "
            "Preserve spatial and temporal details.\n\n" + numbered
        )

    def synthesize(self, layer_texts: dict[str, list[str]]) -> str:
        block = "\n".join(
            f"[{layer}]: {'; '.join(texts)}"
            for layer, texts in layer_texts.items()
        )
        return self._chat(
            "Synthesize the following observations grouped by perception layer "
            "into a coherent summary. Highlight agreements and contradictions.\n\n"
            + block
        )

    def extract_entities(self, texts: list[str]) -> list[dict[str, Any]]:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        raw = self._chat(
            "Extract named entities (objects, places, people) from these observations. "
            "Return ONLY a JSON array where each element has keys: "
            '"name" (string), "entity_type" (string or null), "confidence" (float 0-1).\n\n'
            + numbered
        )
        return _parse_entities(raw)

    def _chat(self, prompt: str, max_tokens: int | None = None) -> str:
        body: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if max_tokens is not None:
            body["options"] = {"num_predict": max_tokens}
        data = post_json(self._url, body, timeout=300)
        return strip_think_tags(data["message"]["content"])


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
