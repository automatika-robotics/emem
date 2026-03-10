import numpy as np
import pytest

pytestmark = [pytest.mark.ollama, pytest.mark.slow]


class TestOllamaVLM:
    def test_describe_synthetic_image(self):
        from harness.providers.ollama_vlm import OllamaVLM

        vlm = OllamaVLM()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:32, :, 0] = 255  # red top half
        image[32:, :, 2] = 255  # blue bottom half

        result = vlm.describe(image, "Describe what you see.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_place_prompt(self):
        from harness.providers.ollama_vlm import OllamaVLM

        vlm = OllamaVLM()
        image = np.full((64, 64, 3), 128, dtype=np.uint8)

        result = vlm.describe(
            image, "What type of place or room is this? Answer in one word."
        )
        assert isinstance(result, str)
        assert len(result) > 0
