"""Schedule dataclasses for eMEM-Bench v1 paradigms.

A :class:`Schedule` is an ordered list of phases the runner walks
through in sequence. Three phase types cover the paradigms in A14b/c:

- :class:`IngestPhase` — drop a batch of observations into memory,
  each at its own timestamp. Used for encoding.
- :class:`AdvanceClockPhase` — advance the runner's virtual clock by
  some delta, optionally firing consolidation + archival ("maintenance")
  at the new clock value. Used for retention-decay / prospective
  paradigms where the gap between encode and probe matters.
- :class:`ProbePhase` — run one or more :class:`BenchmarkQuestion`
  queries against whatever's in memory at a specified wall time.
  Used for retrieval.

The schema is intentionally light — no paradigm-specific fields —
because paradigm semantics live in how the phases are ordered, not
in the dataclasses themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Union

from harness.benchmarks.academic.trajectory import BenchmarkQuestion


@dataclass
class Observation:
    """A single memory observation, intended for an IngestPhase.

    Mirrors the fields that :meth:`emem.memory.SpatioTemporalMemory.add`
    accepts so the runner can forward without translation.
    """

    text: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: float = 0.0
    layer_name: str = "default"
    is_interoception: bool = False


@dataclass
class IngestPhase:
    """Ingest a batch of observations into a fresh episode.

    The runner opens an episode at the start of the phase and closes
    it at the end with consolidation disabled (so maintenance only
    fires from :class:`AdvanceClockPhase`, not implicitly here).

    :param observations: Observations to ingest, in order.
    :param episode_name: Label for the opened episode (appears in
        memory's episode table; useful for downstream paradigm
        queries that filter by episode).
    """

    observations: List[Observation] = field(default_factory=list)
    episode_name: str = "schedule"


@dataclass
class AdvanceClockPhase:
    """Advance the virtual clock by ``delta_seconds``.

    If ``run_maintenance`` is True, after the clock advances the
    runner fires ``consolidate_time_window(reference_time=new_clock)``
    followed by ``archive_long_term(reference_time=new_clock)`` —
    the same pair of calls a long-lived deployment would make
    periodically in the background.

    :param delta_seconds: How many seconds to advance.
    :param run_maintenance: Whether to run consolidation + archival
        at the new clock value.
    """

    delta_seconds: float
    run_maintenance: bool = False


@dataclass
class ProbePhase:
    """Fire a query set against memory at virtual time ``at_time``.

    The runner ensures the virtual clock is at ``at_time`` before
    the probe runs. Each :class:`BenchmarkQuestion` in ``query_set``
    is dispatched through the configured agent; the predictions are
    collected for later scoring.

    :param at_time: Virtual wall time (seconds) at which the probe
        fires. Must be >= the clock value reached so far.
    :param query_set: Questions to ask at this probe.
    :param probe_id: Optional label to group per-probe results
        (e.g. ``"retention_1h"``, ``"retention_24h"``) for paradigm
        reporting.
    """

    at_time: float
    query_set: List[BenchmarkQuestion] = field(default_factory=list)
    probe_id: str = ""


Phase = Union[IngestPhase, AdvanceClockPhase, ProbePhase]


@dataclass
class Schedule:
    """A sample's full paradigm: start time + ordered phases.

    :param sample_id: Stable identifier for this schedule (typically
        mirrors the scene / house id plus a paradigm tag).
    :param scene_id: Identifier of the source scene (so multiple
        schedules can share a scene).
    :param phases: Ordered list of phases for the runner to walk.
    :param start_time: Virtual clock value at the start of phase 0.
    """

    sample_id: str
    scene_id: str = ""
    phases: List[Phase] = field(default_factory=list)
    start_time: float = 0.0
