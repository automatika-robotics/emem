import numpy as np

from harness.providers.http import encode_image_b64, post_json, strip_think_tags


class OllamaVLM:
    """Image-to-text via Ollama's chat API with vision."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"

    def describe(self, image: np.ndarray, prompt: str) -> str:
        """Send an RGB image with a prompt and return the text response.

        :param image: ``(H, W, 3)`` uint8 numpy array (RGB).
        :param prompt: Text prompt for the VLM.
        :returns: Model's text response.
        """
        b64 = encode_image_b64(image)
        data = post_json(self._url, {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
        }, timeout=300)
        return strip_think_tags(data["message"]["content"])
