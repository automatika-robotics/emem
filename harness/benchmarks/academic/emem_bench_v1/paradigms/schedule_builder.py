"""Shared schedule-building helpers for paradigms.

Paradigm modules own the question-generation logic; the schedule
shape that the runner executes is usually ``ingest the scene →
probe``. This module provides that shape as a reusable helper so
each paradigm doesn't repeat itself. Schedule shapes that require
interleaved clock advancement (retention decay, prospective memory
— A14c territory) build their own schedules directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    IngestPhase,
    Observation,
    ProbePhase,
    Schedule,
)
from harness.benchmarks.academic.trajectory import BenchmarkQuestion


def build_ingest_then_probe_schedule(
    candidate: CandidateQuestion,
    scene: Dict[str, Any],
) -> Schedule:
    """Build an ingest-all → probe-once schedule from one candidate.

    Used by one-shot paradigms (DRM, pattern completion, source
    monitoring, etc.) whose mechanism is "encode the whole scene,
    then ask one question". The probe fires immediately after the
    final ingested observation's timestamp.

    :param candidate: The curated :class:`CandidateQuestion`.
    :param scene: Merged scene entry from
        :func:`...scene_entries.load_scene_entries` — must include a
        ``trajectory`` list of waypoint dicts.
    :returns: A two-phase schedule (IngestPhase → ProbePhase).
    """
    observations = _observations_from_trajectory(scene.get("trajectory") or [])
    if not observations:
        raise ValueError(
            f"scene {scene.get('sample_id')!r} has no ingestible observations"
        )
    start_time = min(o.timestamp for o in observations)
    end_time = max(o.timestamp for o in observations)
    probe = BenchmarkQuestion(
        question_id=candidate.question_id,
        question=candidate.question,
        answer=candidate.answer,
        category=candidate.category,
        tools_expected=list(candidate.tools_expected),
    )
    return Schedule(
        sample_id=f"{candidate.paradigm}::{candidate.question_id}",
        scene_id=str(scene.get("scene_id") or scene.get("sample_id") or ""),
        start_time=start_time,
        phases=[
            IngestPhase(
                episode_name=f"{candidate.paradigm}_encode",
                observations=observations,
            ),
            ProbePhase(
                at_time=end_time + 1.0,
                probe_id=candidate.paradigm,
                query_set=[probe],
            ),
        ],
    )


def _observations_from_trajectory(
    frames: List[Dict[str, Any]],
) -> List[Observation]:
    """Flatten waypoint layers + interoception into Observations.

    Mirrors the v1 SceneManifestLoader logic so the paradigm path and
    the baseline loader agree on what "the scene" means.
    """
    observations: List[Observation] = []
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
    return observations


def _pos3(pos: Any) -> tuple:
    """Coerce 2- or 3-element position list into a 3-tuple."""
    if len(pos) == 2:
        return (float(pos[0]), float(pos[1]), 0.0)
    return (float(pos[0]), float(pos[1]), float(pos[2]))
