from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional


class CaptionCache:
    """JSONL-backed cache for VLM captions.

    Avoids re-computing expensive VLM captions across ablation reruns.
    """

    def __init__(self, cache_path: str):
        """
        :param cache_path: Path to the JSONL cache file (created if absent).
        """
        self._path = cache_path
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._cache[entry["key"]] = entry["caption"]

    @staticmethod
    def _make_key(image_path: str, prompt: str, model: str) -> str:
        raw = f"{image_path}|{prompt}|{model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, image_path: str, prompt: str, model: str) -> Optional[str]:
        """Look up a cached caption.

        :param image_path: Path to the source image.
        :param prompt: VLM prompt used for captioning.
        :param model: VLM model name.
        :returns: Cached caption string, or ``None`` on miss.
        """
        return self._cache.get(self._make_key(image_path, prompt, model))

    def put(self, image_path: str, prompt: str, model: str, caption: str) -> None:
        """Store a caption in the cache.

        Duplicate puts (same key) are silently ignored.

        :param image_path: Path to the source image.
        :param prompt: VLM prompt used for captioning.
        :param model: VLM model name.
        :param caption: The generated caption to cache.
        """
        key = self._make_key(image_path, prompt, model)
        if key in self._cache:
            return
        self._cache[key] = caption
        with open(self._path, "a") as f:
            f.write(json.dumps({"key": key, "caption": caption}) + "\n")
