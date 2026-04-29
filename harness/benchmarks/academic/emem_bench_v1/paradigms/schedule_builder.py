"""Shared schedule-building helpers for paradigms.

Paradigm modules own the question-generation logic; the schedule
shape that the runner executes is usually ``ingest the scene →
probe``. This module provides that shape as a reusable helper so
each paradigm doesn't repeat itself. Schedule shapes that require
interleaved clock advancement (retention decay, prospective memory
— A14c territory) build their own schedules directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    AdvanceClockPhase,
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
    text_prefix: str = "",
) -> List[Observation]:
    """Flatten waypoint layers + interoception into Observations.

    Mirrors the v1 SceneManifestLoader logic so the paradigm path and
    the baseline loader agree on what "the scene" means.

    :param text_prefix: When non-empty, prepended to every
        observation's text (e.g. ``"In house Alpha: "`` for
        pattern-separation ingest phases). Gives the agent a stable
        natural-language label it can bind features to.
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
                    text=text_prefix + str(text),
                    position=position,
                    timestamp=timestamp,
                    layer_name=str(layer),
                )
            )
    return observations


def build_two_house_schedule(
    candidate: CandidateQuestion,
    scenes_by_id: Dict[str, Dict[str, Any]],
) -> Schedule:
    """Build an ingest-A → ingest-B → probe schedule for pattern separation.

    Both houses of the pair are ingested back-to-back as separate
    episodes. Every observation is prefixed with the agent-facing
    house label (e.g. ``"In house Alpha: "``) so the agent can bind
    features to the house without needing to know ProcTHOR sample
    ids. The probe fires after the last observation of house B.

    :param candidate: Curated pattern-separation candidate. Must
        carry ``paradigm_metadata.label_to_sample_id`` (mapping
        ``Alpha``/``Beta`` to the pair's two sample ids).
    :param scenes_by_id: Lookup from sample_id to merged scene dict
        (the runner builds this from ``load_scene_entries``).
    :returns: A three-phase :class:`Schedule` the runner can execute.
    """
    label_map: Dict[str, str] = (
        candidate.paradigm_metadata.get("label_to_sample_id") or {}
    )
    if set(label_map) != {"Alpha", "Beta"}:
        raise ValueError(
            f"Pattern-separation candidate {candidate.question_id!r} missing "
            f"label_to_sample_id mapping; found {label_map!r}"
        )
    sid_a = label_map["Alpha"]
    sid_b = label_map["Beta"]
    scene_a = scenes_by_id.get(sid_a)
    scene_b = scenes_by_id.get(sid_b)
    if scene_a is None or scene_b is None:
        raise ValueError(
            f"Pattern-separation candidate {candidate.question_id!r} "
            f"references scenes {sid_a!r}/{sid_b!r} not in scenes_by_id"
        )
    obs_a = _observations_from_trajectory(
        scene_a.get("trajectory") or [], text_prefix="In house Alpha: "
    )
    obs_b = _observations_from_trajectory(
        scene_b.get("trajectory") or [], text_prefix="In house Beta: "
    )
    if not obs_a or not obs_b:
        raise ValueError(
            f"Pattern-separation candidate {candidate.question_id!r}: one "
            "or both houses have no ingestible observations"
        )
    start_time = min(min(o.timestamp for o in obs_a), min(o.timestamp for o in obs_b))
    last_a = max(o.timestamp for o in obs_a)
    # Offset house B so its timestamps come strictly after house A's,
    # matching the schedule-is-temporal semantics of the v1 runner.
    offset_b = (last_a - min(o.timestamp for o in obs_b)) + 1.0
    obs_b_shifted = [
        Observation(
            text=o.text,
            position=o.position,
            timestamp=o.timestamp + offset_b,
            layer_name=o.layer_name,
            is_interoception=o.is_interoception,
        )
        for o in obs_b
    ]
    probe_question = BenchmarkQuestion(
        question_id=candidate.question_id,
        question=candidate.question,
        answer=candidate.answer,
        category=candidate.category,
        tools_expected=list(candidate.tools_expected),
    )
    last_b = max(o.timestamp for o in obs_b_shifted)
    return Schedule(
        sample_id=f"{candidate.paradigm}::{candidate.question_id}",
        scene_id=f"{sid_a}+{sid_b}",
        start_time=start_time,
        phases=[
            IngestPhase(
                episode_name=f"{candidate.paradigm}_encode_alpha",
                observations=obs_a,
            ),
            IngestPhase(
                episode_name=f"{candidate.paradigm}_encode_beta",
                observations=obs_b_shifted,
            ),
            ProbePhase(
                at_time=last_b + 1.0,
                probe_id=candidate.paradigm,
                query_set=[probe_question],
            ),
        ],
    )


def build_retention_decay_schedule(
    candidate: CandidateQuestion,
    scene: Dict[str, Any],
) -> Schedule:
    """Build an ingest → advance(delay, maintenance) → probe schedule.

    Used by the retention-interval-decay paradigm. The candidate
    carries ``paradigm_metadata.retention_delay_seconds`` indicating
    how far into the future the probe should fire (and therefore how
    much consolidation/archival pressure the system has been under).

    :param candidate: A retention-decay :class:`CandidateQuestion`.
    :param scene: Merged scene entry with a ``trajectory`` list.
    :returns: A three-phase Schedule: IngestPhase →
        AdvanceClockPhase(run_maintenance=True) → ProbePhase.
    :raises ValueError: If the scene has no observations, or the
        candidate metadata is missing ``retention_delay_seconds``.
    """
    delay_seconds = candidate.paradigm_metadata.get("retention_delay_seconds")
    if delay_seconds is None:
        raise ValueError(
            f"Retention-decay candidate {candidate.question_id!r} missing "
            f"retention_delay_seconds in paradigm_metadata"
        )
    observations = _observations_from_trajectory(scene.get("trajectory") or [])
    if not observations:
        raise ValueError(
            f"scene {scene.get('sample_id')!r} has no ingestible observations"
        )
    start_time = min(o.timestamp for o in observations)
    end_time = max(o.timestamp for o in observations)
    probe_time = end_time + float(delay_seconds)
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
            AdvanceClockPhase(
                delta_seconds=float(delay_seconds),
                run_maintenance=True,
            ),
            ProbePhase(
                at_time=probe_time,
                probe_id=str(
                    candidate.paradigm_metadata.get(
                        "retention_delay_bucket", candidate.paradigm
                    )
                ),
                query_set=[probe],
            ),
        ],
    )


def build_context_dependent_schedule(
    candidate: CandidateQuestion,
    scene: Dict[str, Any],
) -> Schedule:
    """Build an ingest → set-position → probe schedule.

    Used by the context-dependent-retrieval paradigm. After ingesting
    the trajectory normally, an extra single-observation IngestPhase
    drops an ``"Agent is currently here."`` observation at the
    candidate's specified ``agent_position`` (layer
    ``agent_position``), so the agent's tools see the position as a
    recent contextual cue when the probe runs.

    :param candidate: A context-dependent-retrieval
        :class:`CandidateQuestion`. Must carry
        ``paradigm_metadata.agent_position`` as a 3-element list.
    :param scene: Merged scene entry with a ``trajectory`` list.
    :returns: A four-phase Schedule: IngestPhase(scene) →
        IngestPhase(agent_position) → ProbePhase.
    :raises ValueError: If the scene has no observations, or the
        candidate metadata is missing ``agent_position``.
    """
    pos = candidate.paradigm_metadata.get("agent_position")
    if not pos or len(pos) != 3:
        raise ValueError(
            f"Context-dependent candidate {candidate.question_id!r} missing "
            f"a 3-element agent_position in paradigm_metadata"
        )
    observations = _observations_from_trajectory(scene.get("trajectory") or [])
    if not observations:
        raise ValueError(
            f"scene {scene.get('sample_id')!r} has no ingestible observations"
        )
    start_time = min(o.timestamp for o in observations)
    end_time = max(o.timestamp for o in observations)
    pos_obs = Observation(
        text="The agent is currently here.",
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        timestamp=end_time + 0.5,
        layer_name="agent_position",
    )
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
            IngestPhase(
                episode_name=f"{candidate.paradigm}_agent_position",
                observations=[pos_obs],
            ),
            ProbePhase(
                at_time=end_time + 1.0,
                probe_id=str(candidate.paradigm_metadata.get("context", "")),
                query_set=[probe],
            ),
        ],
    )


def build_prospective_memory_schedule(
    candidate: CandidateQuestion,
    scene: Dict[str, Any],
) -> Schedule:
    """Build instruction → distractor → probe schedule for prospective memory.

    Three phases plus a small clock advance between encode and tour::

        IngestPhase  — single instruction observation, layer "instruction"
        AdvanceClock — 60s, no maintenance (instruction stays warm)
        IngestPhase  — full scene trajectory; the trigger fires
                       naturally when the agent's perception touches
                       the item via the object_detection layer
        ProbePhase   — "what were you instructed to do when you see X?"

    :param candidate: A prospective-memory :class:`CandidateQuestion`.
        Must carry ``paradigm_metadata.instruction_text`` and
        ``paradigm_metadata.trigger_object``.
    :param scene: Merged scene entry with a ``trajectory`` list.
    :returns: A four-phase Schedule.
    :raises ValueError: If the scene has no observations or the
        candidate is missing required metadata.
    """
    instruction_text = candidate.paradigm_metadata.get("instruction_text")
    if not instruction_text:
        raise ValueError(
            f"Prospective-memory candidate {candidate.question_id!r} missing "
            f"instruction_text in paradigm_metadata"
        )
    observations = _observations_from_trajectory(scene.get("trajectory") or [])
    if not observations:
        raise ValueError(
            f"scene {scene.get('sample_id')!r} has no ingestible observations"
        )

    traj_start = min(o.timestamp for o in observations)
    traj_end = max(o.timestamp for o in observations)
    instruction_obs = Observation(
        text=str(instruction_text),
        position=(0.0, 0.0, 0.0),
        timestamp=traj_start - 60.0,
        layer_name="instruction",
    )
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
        start_time=instruction_obs.timestamp,
        phases=[
            IngestPhase(
                episode_name=f"{candidate.paradigm}_instruction",
                observations=[instruction_obs],
            ),
            AdvanceClockPhase(delta_seconds=60.0, run_maintenance=False),
            IngestPhase(
                episode_name=f"{candidate.paradigm}_distractor_trajectory",
                observations=observations,
            ),
            ProbePhase(
                at_time=traj_end + 1.0,
                probe_id=candidate.paradigm,
                query_set=[probe],
            ),
        ],
    )


def build_long_horizon_interference_schedule(
    candidate: CandidateQuestion,
    scenes_by_id: Dict[str, Dict[str, Any]],
) -> Schedule:
    """Build encode-A → gap-with-maintenance → encode-B → probe schedule.

    Used by the long-horizon-interference paradigm. Two house
    trajectories are ingested as separate episodes (prefixed
    "In house Alpha: " / "In house Beta: ") with an
    :class:`AdvanceClockPhase` between them that fires consolidation
    + archival at the new clock value. The probe then asks about
    items from House Alpha only.

    :param candidate: An LHI :class:`CandidateQuestion`. Must carry
        ``paradigm_metadata.label_to_sample_id`` mapping
        ``Alpha``/``Beta`` to the pair's two sample ids, and
        ``paradigm_metadata.gap_seconds`` for the inter-episode gap.
    :param scenes_by_id: Lookup from sample_id to merged scene dict.
    :returns: A four-phase Schedule.
    :raises ValueError: If required metadata is missing or scenes
        can't be resolved.
    """
    label_map: Dict[str, str] = (
        candidate.paradigm_metadata.get("label_to_sample_id") or {}
    )
    if set(label_map) != {"Alpha", "Beta"}:
        raise ValueError(
            f"LHI candidate {candidate.question_id!r} missing "
            f"label_to_sample_id mapping; found {label_map!r}"
        )
    gap_seconds = candidate.paradigm_metadata.get("gap_seconds")
    if gap_seconds is None:
        raise ValueError(
            f"LHI candidate {candidate.question_id!r} missing "
            f"gap_seconds in paradigm_metadata"
        )
    sid_a = label_map["Alpha"]
    sid_b = label_map["Beta"]
    scene_a = scenes_by_id.get(sid_a)
    scene_b = scenes_by_id.get(sid_b)
    if scene_a is None or scene_b is None:
        raise ValueError(
            f"LHI candidate {candidate.question_id!r} references scenes "
            f"{sid_a!r}/{sid_b!r} not in scenes_by_id"
        )

    obs_a = _observations_from_trajectory(
        scene_a.get("trajectory") or [], text_prefix="In house Alpha: "
    )
    if not obs_a:
        raise ValueError(
            f"LHI candidate {candidate.question_id!r}: house Alpha "
            f"({sid_a!r}) has no ingestible observations"
        )
    obs_b = _observations_from_trajectory(
        scene_b.get("trajectory") or [], text_prefix="In house Beta: "
    )
    if not obs_b:
        raise ValueError(
            f"LHI candidate {candidate.question_id!r}: house Beta "
            f"({sid_b!r}) has no ingestible observations"
        )

    start_time = min(o.timestamp for o in obs_a)
    last_a = max(o.timestamp for o in obs_a)
    # The gap fires AFTER house A's ingestion but BEFORE house B's;
    # consolidation+archival therefore see only A's content. Then
    # B's observations land at clock = last_a + gap_seconds + delta.
    b_clock_floor = last_a + float(gap_seconds)
    obs_b_offset = b_clock_floor - min(o.timestamp for o in obs_b) + 1.0
    obs_b_shifted = [
        Observation(
            text=o.text,
            position=o.position,
            timestamp=o.timestamp + obs_b_offset,
            layer_name=o.layer_name,
            is_interoception=o.is_interoception,
        )
        for o in obs_b
    ]
    last_b = max(o.timestamp for o in obs_b_shifted)
    probe = BenchmarkQuestion(
        question_id=candidate.question_id,
        question=candidate.question,
        answer=candidate.answer,
        category=candidate.category,
        tools_expected=list(candidate.tools_expected),
    )
    return Schedule(
        sample_id=f"{candidate.paradigm}::{candidate.question_id}",
        scene_id=f"{sid_a}+{sid_b}",
        start_time=start_time,
        phases=[
            IngestPhase(
                episode_name=f"{candidate.paradigm}_encode_alpha",
                observations=obs_a,
            ),
            AdvanceClockPhase(
                delta_seconds=float(gap_seconds),
                run_maintenance=True,
            ),
            IngestPhase(
                episode_name=f"{candidate.paradigm}_encode_beta",
                observations=obs_b_shifted,
            ),
            ProbePhase(
                at_time=last_b + 1.0,
                probe_id=str(candidate.paradigm_metadata.get("item_origin", "")),
                query_set=[probe],
            ),
        ],
    )


def build_serial_position_schedule(
    candidate: CandidateQuestion,
    scenes_by_id: Dict[str, Dict[str, Any]],
) -> Schedule:
    """Build a chained-houses super-trajectory schedule for serial position.

    The chain's house trajectories are ingested as separate episodes
    *without* clock advances or maintenance between them — one
    continuous tour. Trajectory timestamps are shifted so each
    chain[i+1] starts strictly after chain[i] ends. The probe fires
    after the last observation of the final chain position.

    :param candidate: A serial-position :class:`CandidateQuestion`.
        Must carry ``paradigm_metadata.chain_sample_ids`` listing
        the chain's sample_ids in order.
    :param scenes_by_id: Lookup from sample_id to merged scene dict.
    :returns: A multi-phase Schedule (one IngestPhase per chain
        position + one ProbePhase).
    :raises ValueError: If chain metadata is missing or any chain
        scene can't be resolved.
    """
    chain_ids: List[str] = list(
        candidate.paradigm_metadata.get("chain_sample_ids") or []
    )
    if not chain_ids:
        raise ValueError(
            f"Serial-position candidate {candidate.question_id!r} missing "
            f"chain_sample_ids in paradigm_metadata"
        )
    chain_scenes = []
    for sid in chain_ids:
        scene = scenes_by_id.get(sid)
        if scene is None:
            raise ValueError(
                f"Serial-position candidate {candidate.question_id!r} "
                f"references scene {sid!r} not in scenes_by_id"
            )
        chain_scenes.append(scene)

    # Build observations per chain position; shift timestamps so
    # each subsequent house starts after the previous one ends.
    phases: List[Any] = []
    running_clock = 0.0
    start_time: Optional[float] = None
    last_obs_ts = 0.0
    for position, scene in enumerate(chain_scenes):
        obs = _observations_from_trajectory(scene.get("trajectory") or [])
        if not obs:
            raise ValueError(
                f"Serial-position candidate {candidate.question_id!r}: "
                f"chain position {position} (sample {scene.get('sample_id')!r}) "
                f"has no ingestible observations"
            )
        # First call: anchor running_clock to the first house's
        # earliest observation. Subsequent houses shift forward to
        # start ≥ running_clock + 1s after the previous house's end.
        original_min = min(o.timestamp for o in obs)
        original_max = max(o.timestamp for o in obs)
        if position == 0:
            shift = 0.0
            start_time = original_min
            running_clock = original_max
        else:
            shift = (running_clock + 1.0) - original_min
            running_clock = original_max + shift
        shifted = [
            Observation(
                text=o.text,
                position=o.position,
                timestamp=o.timestamp + shift,
                layer_name=o.layer_name,
                is_interoception=o.is_interoception,
            )
            for o in obs
        ]
        last_obs_ts = max(o.timestamp for o in shifted)
        phases.append(
            IngestPhase(
                episode_name=f"{candidate.paradigm}_chain_pos{position}",
                observations=shifted,
            )
        )

    probe = BenchmarkQuestion(
        question_id=candidate.question_id,
        question=candidate.question,
        answer=candidate.answer,
        category=candidate.category,
        tools_expected=list(candidate.tools_expected),
    )
    phases.append(
        ProbePhase(
            at_time=last_obs_ts + 1.0,
            probe_id=str(candidate.paradigm_metadata.get("bin_label", "")),
            query_set=[probe],
        )
    )
    return Schedule(
        sample_id=f"{candidate.paradigm}::{candidate.question_id}",
        scene_id=str(candidate.paradigm_metadata.get("chain_id", "")),
        start_time=float(start_time or 0.0),
        phases=phases,
    )


def _pos3(pos: Any) -> tuple:
    """Coerce 2- or 3-element position list into a 3-tuple."""
    if len(pos) == 2:
        return (float(pos[0]), float(pos[1]), 0.0)
    return (float(pos[0]), float(pos[1]), float(pos[2]))
