from __future__ import annotations

import numpy as np

from harness.providers.gemini_llm import GeminiLLMClient
from harness.providers.http import encode_image_b64


class GeminiVLM:
    """Image-to-text via Gemini API.

    Thin wrapper around :class:`GeminiLLMClient` that handles
    numpy array to base64 PNG encoding.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        api_key: str | None = None,
    ):
        self._client = GeminiLLMClient(model=model, api_key=api_key)

    def describe(
        self,
        image: np.ndarray,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Send an RGB image with a prompt and return the text response.

        :param image: ``(H, W, 3)`` uint8 numpy array (RGB).
        :param prompt: Text prompt for the VLM.
        :param max_tokens: Maximum tokens to generate (passed to Gemini).
        :returns: Model's text response.
        """
        b64 = encode_image_b64(image)
        return self._client.describe(
            b64,
            prompt,
            mime_type="image/png",
            max_tokens=max_tokens,
        )
