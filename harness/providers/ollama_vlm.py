import numpy as np

from harness.providers.http import encode_image_b64, post_json, strip_think_tags


def _is_thinking_model(model: str) -> bool:
    """Check if model uses <think> reasoning (qwen3+, deepseek-r1, etc.)."""
    m = model.lower()
    return any(k in m for k in ("qwen3", "deepseek-r1", "qwq"))


class OllamaVLM:
    """Image-to-text via Ollama's chat API with vision."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._thinks = _is_thinking_model(model)

    def describe(
        self,
        image: np.ndarray,
        prompt: str,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        """Send an RGB image with a prompt and return the text response.

        :param image: ``(H, W, 3)`` uint8 numpy array (RGB).
        :param prompt: Text prompt for the VLM.
        :param max_tokens: Maximum tokens to generate.
        :param think: Force thinking on/off.  ``None`` = auto (disable for
            thinking models when *max_tokens* ≤ 200 to avoid wasting budget).
        :returns: Model's text response.
        """
        # Auto-disable thinking for short outputs on thinking models
        if think is None and self._thinks:
            think = False if (max_tokens is not None and max_tokens <= 200) else True
        if think is False and self._thinks:
            prompt = prompt.rstrip() + " /no_think"

        b64 = encode_image_b64(image)
        body: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
        }
        if max_tokens is not None:
            body["options"] = {"num_predict": max_tokens}
        data = post_json(self._url, body, timeout=300)
        return strip_think_tags(data["message"]["content"])
