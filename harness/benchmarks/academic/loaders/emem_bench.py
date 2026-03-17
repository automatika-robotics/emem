"""Loader for the eMEM-Bench custom embodied memory benchmark.

Loads multi-layer trajectory data with interoception and tool-expected
annotations from the eMEM-Bench JSON format.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)

log = logging.getLogger(__name__)


class EMEMBenchLoader:
    """Loader for eMEM-Bench benchmark data.

    Each sample is an exploration episode with multi-layer observations,
    interoception data, and questions annotated with expected tool usage.

    :param data_dir: Root directory containing ``emem-bench-v0.json`` and
        source subdirectories (``ai2thor/``, ``robot/``).
    :param max_samples: Maximum samples to yield.
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self._data_dir = data_dir
        self._max_samples = max_samples

    @property
    def name(self) -> str:
        return "emem-bench"

    def load(self) -> Iterator[BenchmarkSample]:
        """Yield one :class:`BenchmarkSample` per exploration episode."""
        index = self._load_index()
        count = 0

        for entry in index:
            if self._max_samples is not None and count >= self._max_samples:
                break

            sample = self._load_sample(entry)
            if sample is None:
                continue

            yield sample
            count += 1

    def _load_index(self) -> List[Dict[str, Any]]:
        """Load the benchmark index file.

        Tries ``emem-bench-v0.json`` first, then ``emem-bench.json``.

        :returns: List of sample entries from the index.
        """
        for filename in ("emem-bench-v0.json", "emem-bench.json"):
            path = os.path.join(self._data_dir, filename)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if "samples" in data:
                    return data["samples"]
                return [data]

        # Fallback: scan for individual sample JSON files
        samples: List[Dict[str, Any]] = []
        for source_dir in ("ai2thor", "robot"):
            source_path = os.path.join(self._data_dir, source_dir)
            if not os.path.isdir(source_path):
                continue
            for scene_dir in sorted(os.listdir(source_path)):
                traj_path = os.path.join(source_path, scene_dir, "trajectory.json")
                if os.path.exists(traj_path):
                    with open(traj_path) as f:
                        samples.append(json.load(f))
        return samples

    def _load_sample(self, entry: Dict[str, Any]) -> Optional[BenchmarkSample]:
        """Convert a sample entry dict into a :class:`BenchmarkSample`.

        If the entry has a ``trajectory_path`` key pointing to an external
        file, load it from disk. Otherwise, expect trajectory data inline.

        :param entry: Sample entry from the index or standalone file.
        :returns: BenchmarkSample, or ``None`` if data is incomplete.
        """
        # Load external trajectory if referenced
        if "trajectory_path" in entry:
            traj_path = os.path.join(self._data_dir, entry["trajectory_path"])
            if not os.path.exists(traj_path):
                log.warning("Trajectory file not found: %s", traj_path)
                return None
            with open(traj_path) as f:
                traj_data = json.load(f)
            # Merge trajectory data into entry
            entry = {**traj_data, **entry}

        trajectory = self._build_trajectory(entry)
        questions = self._build_questions(entry)

        if not trajectory or not questions:
            log.warning(
                "Skipping sample %s: %d frames, %d questions",
                entry.get("sample_id", "?"),
                len(trajectory),
                len(questions),
            )
            return None

        return BenchmarkSample(
            sample_id=str(entry.get("sample_id", entry.get("scene_id", "unknown"))),
            scene_id=str(entry.get("scene_id", entry.get("sample_id", "unknown"))),
            trajectory=trajectory,
            questions=questions,
        )

    @staticmethod
    def _build_trajectory(entry: Dict[str, Any]) -> List[TrajectoryFrame]:
        """Build trajectory frames from multi-layer observations.

        Each frame in the trajectory has multiple layers. We create one
        :class:`TrajectoryFrame` per layer per position, sharing coordinates
        and timestamps.

        Interoception data is also converted to frames with
        ``layer_name="interoception:<key>"``.

        :param entry: Sample dict with ``"trajectory"`` and optional
            ``"interoception"`` keys.
        :returns: List of trajectory frames.
        """
        frames: List[TrajectoryFrame] = []
        trajectory = entry.get("trajectory", [])

        for waypoint in trajectory:
            frame_id = waypoint.get("frame_id", "")
            pos = waypoint.get("position", [0.0, 0.0, 0.0])
            # Ensure 3D position
            if len(pos) == 2:
                pos = [pos[0], pos[1], 0.0]
            position = (float(pos[0]), float(pos[1]), float(pos[2]))
            timestamp = float(waypoint.get("timestamp", 0.0))
            image_path = waypoint.get("image_path")

            layers = waypoint.get("layers", {})
            if not layers:
                # Fallback: single text field
                text = waypoint.get("text", "")
                if text:
                    layers = {"description": text}

            for layer_name, text in layers.items():
                if not text:
                    continue
                frames.append(TrajectoryFrame(
                    frame_id=f"{frame_id}_{layer_name}",
                    position=position,
                    timestamp=timestamp,
                    text=str(text),
                    layer_name=layer_name,
                    image_path=image_path,
                ))

        # Add interoception data as special frames
        for body_state in entry.get("interoception", []):
            ts = float(body_state.get("timestamp", 0.0))
            # Find the closest trajectory position at this timestamp
            closest_pos = (0.0, 0.0, 0.0)
            if trajectory:
                closest = min(
                    trajectory,
                    key=lambda w: abs(float(w.get("timestamp", 0.0)) - ts),
                )
                p = closest.get("position", [0.0, 0.0, 0.0])
                if len(p) == 2:
                    p = [p[0], p[1], 0.0]
                closest_pos = (float(p[0]), float(p[1]), float(p[2]))

            for key, value in body_state.items():
                if key == "timestamp":
                    continue
                frames.append(TrajectoryFrame(
                    frame_id=f"interoception_{key}_{ts:.0f}",
                    position=closest_pos,
                    timestamp=ts,
                    text=str(value),
                    layer_name=f"interoception:{key}",
                ))

        return frames

    @staticmethod
    def _build_questions(entry: Dict[str, Any]) -> List[BenchmarkQuestion]:
        """Extract questions from the sample entry.

        :param entry: Sample dict with ``"questions"`` key.
        :returns: List of benchmark questions.
        """
        questions: List[BenchmarkQuestion] = []
        for qa in entry.get("questions", []):
            questions.append(BenchmarkQuestion(
                question_id=str(qa.get("question_id", qa.get("id", len(questions)))),
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                category=str(qa.get("category", "")),
            ))
        return questions
