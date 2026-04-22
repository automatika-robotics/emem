"""Loader: scenes.jsonl manifest → :class:`Schedule` stream.

Reads the A7 collection output (``<data_dir>/scenes.jsonl`` plus
per-house ``trajectory.json`` files) and yields one :class:`Schedule`
per manifest entry. Paradigm generators (A14b/c) are expected to
post-process these into richer multi-phase schedules; for A14a the
baseline conversion is a single :class:`IngestPhase` containing every
trajectory frame as an :class:`Observation`, with no probes.

That baseline conversion is exactly what's needed to smoke-test the
runner against real data before paradigms exist; once A14b adds a
paradigm, it can build schedules directly (or wrap this loader and
mutate / replace phases).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from harness.benchmarks.academic.emem_bench_v1.schedule import (
    IngestPhase,
    Observation,
    Schedule,
)

log = logging.getLogger(__name__)


class SceneManifestLoader:
    """Load scenes.jsonl + per-house trajectories as ingest-only Schedules.

    :param data_dir: Directory containing ``scenes.jsonl`` and the
        per-house subdirectories referenced by ``trajectory_path``.
    :param max_samples: Cap on samples to yield.
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self._data_dir = data_dir
        self._max_samples = max_samples

    @property
    def name(self) -> str:
        return "emem-bench-v1"

    def load(self) -> Iterator[Schedule]:
        """Yield one ingest-only :class:`Schedule` per manifest entry."""
        manifest_path = os.path.join(self._data_dir, "scenes.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"scenes.jsonl not found at {manifest_path!r}; did you run the "
                "ProcTHOR collector (harness.benchmarks.emem_bench.collect_procthor)?"
            )

        count = 0
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if self._max_samples is not None and count >= self._max_samples:
                    break
                entry = json.loads(line)
                schedule = self._load_schedule(entry)
                if schedule is None:
                    continue
                yield schedule
                count += 1

    def _load_schedule(self, entry: Dict[str, Any]) -> Optional[Schedule]:
        """Convert one manifest entry into an ingest-only Schedule."""
        traj_rel = entry.get("trajectory_path")
        if not traj_rel:
            log.warning("manifest entry missing trajectory_path: %s", entry)
            return None
        traj_path = os.path.join(self._data_dir, traj_rel)
        if not os.path.exists(traj_path):
            log.warning("trajectory file not found: %s", traj_path)
            return None
        with open(traj_path) as f:
            trajectory = json.load(f)

        observations = _observations_from_trajectory(trajectory)
        if not observations:
            log.warning("no observations in %s", traj_rel)
            return None

        start_time = min(o.timestamp for o in observations)
        return Schedule(
            sample_id=str(entry.get("sample_id") or entry.get("scene_id") or "unknown"),
            scene_id=str(entry.get("scene_id") or entry.get("sample_id") or "unknown"),
            start_time=start_time,
            phases=[IngestPhase(episode_name="ingest", observations=observations)],
        )


def _observations_from_trajectory(trajectory: Dict[str, Any]) -> List[Observation]:
    """Flatten a v1 trajectory.json into :class:`Observation` records.

    Each waypoint produces one Observation per non-empty layer (vlm /
    detections / place). Interoception entries produce one Observation
    per body-state key (battery, cpu_temp, …) with
    ``is_interoception=True``. The closest trajectory waypoint's
    position is reused for interoception timestamps so the spatial
    coordinate is non-degenerate even for body-state entries.
    """
    observations: List[Observation] = []
    frames = trajectory.get("trajectory") or []
    for wp in frames:
        position = _pos3(wp.get("position", [0.0, 0.0, 0.0]))
        timestamp = float(wp.get("timestamp", 0.0))
        for layer, text in (wp.get("layers") or {}).items():
            if not text:
                continue
            observations.append(
                Observation(
                    text=str(text),
                    position=position,
                    timestamp=timestamp,
                    layer_name=str(layer),
                )
            )

    for body in trajectory.get("interoception") or []:
        ts = float(body.get("timestamp", 0.0))
        closest_pos = _closest_waypoint_position(frames, ts)
        for key, value in body.items():
            if key == "timestamp":
                continue
            observations.append(
                Observation(
                    text=str(value),
                    position=closest_pos,
                    timestamp=ts,
                    layer_name=str(key),
                    is_interoception=True,
                )
            )

    return observations


def _pos3(pos: Any) -> tuple:
    """Coerce a 2- or 3-element position list into a 3-tuple."""
    if len(pos) == 2:
        return (float(pos[0]), float(pos[1]), 0.0)
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _closest_waypoint_position(frames: List[Dict[str, Any]], ts: float) -> tuple:
    """Return the position of the waypoint with nearest timestamp."""
    if not frames:
        return (0.0, 0.0, 0.0)
    closest = min(frames, key=lambda w: abs(float(w.get("timestamp", 0.0)) - ts))
    return _pos3(closest.get("position", [0.0, 0.0, 0.0]))
