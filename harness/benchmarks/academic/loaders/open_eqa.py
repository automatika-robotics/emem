import glob
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from harness.benchmarks.academic.caption_cache import CaptionCache
from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)

log = logging.getLogger(__name__)


class OpenEQALoader:
    """Loader for the OpenEQA Episodic Memory benchmark.

    Questions are grouped by ``episode_history`` so each trajectory is
    loaded once and all its questions are batched together.
    """

    def __init__(
        self,
        data_dir: str,
        n_frames: int = 20,
        caption_cache: Optional[CaptionCache] = None,
        vlm: Any = None,
        caption_prompt: str = "Describe what you see in this image in 1-2 sentences.",
        vlm_model_name: str = "default",
    ):
        """
        :param data_dir: Root data directory containing ``open-eqa-v0.json``
            and a ``frames/`` subdirectory.
        :param n_frames: Number of frames to uniformly sample per episode.
        :param caption_cache: Optional cache to avoid re-running VLM.
        :param vlm: Optional VLM instance with a ``.describe(img, prompt)``
            method. Used when ``caption_cache`` misses. Requires ``cv2``.
        :param caption_prompt: Prompt passed to the VLM.
        :param vlm_model_name: Model name for cache key generation.
        """
        self._data_dir = data_dir
        self._n_frames = n_frames
        self._cache = caption_cache
        self._vlm = vlm
        self._caption_prompt = caption_prompt
        self._vlm_model = vlm_model_name

    @property
    def name(self) -> str:
        return "open-eqa"

    def load(self) -> Iterator[BenchmarkSample]:
        """Yield one :class:`BenchmarkSample` per episode (with all its questions)."""
        meta_path = os.path.join(self._data_dir, "open-eqa-v0.json")
        with open(meta_path) as f:
            entries = json.load(f)

        by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            by_episode[entry.get("episode_history", "")].append(entry)

        for ep_history, ep_entries in by_episode.items():
            trajectory = self._load_trajectory(ep_history)
            if not trajectory:
                continue

            questions = [
                BenchmarkQuestion(
                    question_id=str(entry.get("question_id", "")),
                    question=entry["question"],
                    answer=entry.get("answer", ""),
                    category=entry.get("category", ""),
                )
                for entry in ep_entries
            ]

            yield BenchmarkSample(
                sample_id=ep_history,
                scene_id=ep_history,
                trajectory=trajectory,
                questions=questions,
            )

    def _load_trajectory(self, episode_history: str) -> List[TrajectoryFrame]:
        """Load and uniformly sample frames from an episode directory.

        :param episode_history: Relative path to the episode within ``frames/``.
        :returns: Sampled trajectory frames. Empty if directory missing.
        """
        frame_dir = os.path.join(self._data_dir, "frames", episode_history)
        if not os.path.isdir(frame_dir):
            return []

        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*-rgb.png")))
        if not frame_paths:
            return []

        indices = np.linspace(0, len(frame_paths) - 1, min(self._n_frames, len(frame_paths)))
        indices = np.unique(indices.astype(int))
        sampled = [frame_paths[i] for i in indices]

        frames: List[TrajectoryFrame] = []
        for i, img_path in enumerate(sampled):
            position = self._load_pose(img_path)
            text = self._get_caption(img_path)

            frames.append(TrajectoryFrame(
                frame_id=os.path.basename(img_path),
                position=position,
                timestamp=float(i),
                text=text,
                image_path=img_path,
            ))

        return frames

    def _load_pose(self, img_path: str) -> Tuple[float, float, float]:
        """Load camera pose for a frame.

        Expects a ``*-pose.txt`` file alongside the RGB frame containing
        a 4x4 camera extrinsics matrix. Falls back to ``(0, 0, 0)`` if
        the pose file is missing or malformed.

        :param img_path: Path to the RGB frame image.
        :returns: ``(x, y, z)`` translation from the pose matrix.
        """
        pose_path = re.sub(r"-rgb\.png$", "-pose.txt", img_path)
        if os.path.exists(pose_path):
            try:
                pose = np.loadtxt(pose_path).reshape(4, 4)
                return (float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]))
            except Exception:
                log.warning("Failed to parse pose file: %s", pose_path)
        return (0.0, 0.0, 0.0)

    def _get_caption(self, img_path: str) -> str:
        """Get caption from cache or generate via VLM.

        :param img_path: Path to the RGB frame image.
        :returns: Caption text. Falls back to the filename if no VLM available.
        """
        if self._cache is not None:
            cached = self._cache.get(img_path, self._caption_prompt, self._vlm_model)
            if cached is not None:
                return cached

        if self._vlm is not None:
            try:
                import cv2
            except ImportError:
                log.warning("cv2 not installed; cannot caption frames via VLM")
                return f"Frame: {os.path.basename(img_path)}"

            img = cv2.imread(img_path)
            if img is not None:
                caption = self._vlm.describe(img, self._caption_prompt)
                if self._cache is not None:
                    self._cache.put(img_path, self._caption_prompt, self._vlm_model, caption)
                return caption

        return f"Frame: {os.path.basename(img_path)}"
