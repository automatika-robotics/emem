import os
import tempfile

from harness.benchmarks.academic.caption_cache import CaptionCache


class TestCaptionCache:
    def test_put_and_get(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cache = CaptionCache(path)
            cache.put("img.png", "describe", "model1", "A cat sitting")
            assert cache.get("img.png", "describe", "model1") == "A cat sitting"
        finally:
            os.unlink(path)

    def test_cache_miss(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cache = CaptionCache(path)
            assert cache.get("nonexistent.png", "prompt", "model") is None
        finally:
            os.unlink(path)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cache1 = CaptionCache(path)
            cache1.put("img.png", "describe", "model1", "A dog")

            cache2 = CaptionCache(path)
            assert cache2.get("img.png", "describe", "model1") == "A dog"
        finally:
            os.unlink(path)

    def test_different_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cache = CaptionCache(path)
            cache.put("img.png", "describe", "model1", "caption1")
            cache.put("img.png", "describe", "model2", "caption2")
            assert cache.get("img.png", "describe", "model1") == "caption1"
            assert cache.get("img.png", "describe", "model2") == "caption2"
        finally:
            os.unlink(path)

    def test_no_duplicate_writes(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cache = CaptionCache(path)
            cache.put("img.png", "p", "m", "cap")
            cache.put("img.png", "p", "m", "cap")

            with open(path) as f:
                lines = [line for line in f.readlines() if line.strip()]
            assert len(lines) == 1
        finally:
            os.unlink(path)
