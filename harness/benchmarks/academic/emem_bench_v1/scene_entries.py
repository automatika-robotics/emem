"""Read scenes.jsonl + per-house trajectory.json into merged scene entries.

:class:`SceneManifestLoader` yields :class:`Schedule` objects for the
benchmark runner; the paradigm generators need richer scene data
(trajectory frames with room tags, scene_objects metadata) to author
questions. This module reads the manifest + trajectories and returns
flat dicts that paradigm generators can consume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def load_scene_entries(
    data_dir: Path, max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Return one dict per manifest entry, trajectory + objects merged in.

    Each entry looks like the manifest row with these extra keys:
      * ``trajectory`` — list of waypoint dicts
      * ``scene_objects`` — list from trajectory.json metadata
      * ``interoception`` — list of body-state snapshots

    :param data_dir: Directory containing ``scenes.jsonl`` and the
        per-house trajectory paths referenced by it.
    :param max_samples: Optional cap.
    :returns: Flat list of merged entries.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "scenes.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"scenes.jsonl not found at {manifest_path!r}")

    entries: List[Dict[str, Any]] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and len(entries) >= max_samples:
                break
            meta = json.loads(line)
            traj_rel = meta.get("trajectory_path")
            if not traj_rel:
                log.warning("manifest entry missing trajectory_path: %s", meta)
                continue
            traj_path = data_dir / traj_rel
            if not traj_path.exists():
                log.warning("trajectory file not found: %s", traj_path)
                continue
            with traj_path.open() as t:
                trajectory = json.load(t)
            merged = dict(meta)
            merged["trajectory"] = trajectory.get("trajectory") or []
            merged["interoception"] = trajectory.get("interoception") or []
            merged["scene_objects"] = (trajectory.get("metadata") or {}).get(
                "scene_objects", []
            )
            entries.append(merged)
    return entries


def group_waypoints_by_room(
    trajectory: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Group trajectory waypoints by ``room_type``.

    For each room type we aggregate the ``object_detection``,
    ``place``, and ``vlm`` layer text plus the count of waypoints that
    landed there. Waypoints with no ``room_type`` (agent standing in a
    doorway / outside any polygon) are bucketed under ``"outside"``.

    :param trajectory: List of waypoint dicts from a merged scene entry.
    :returns: Mapping ``room_type -> {"n_waypoints", "object_detection",
        "vlm_descriptions", "places"}``.
    """
    agg: Dict[str, Dict[str, Any]] = {}
    for wp in trajectory:
        room_type = wp.get("room_type") or "outside"
        bucket = agg.setdefault(
            room_type,
            {
                "n_waypoints": 0,
                "object_detection": [],
                "vlm_descriptions": [],
                "places": [],
            },
        )
        bucket["n_waypoints"] += 1
        layers = wp.get("layers") or {}
        for key, dest in (
            ("object_detection", "object_detection"),
            ("vlm", "vlm_descriptions"),
            ("place", "places"),
        ):
            val = layers.get(key)
            if val:
                bucket[dest].append(str(val))
    return agg
