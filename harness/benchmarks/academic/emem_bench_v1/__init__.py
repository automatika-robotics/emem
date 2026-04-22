"""eMEM-Bench v1: scheduled paradigm runner for embodied-memory tests.

This package holds the v1 runtime used by the cognitive-paradigm
evaluations defined in A14b/c. The v0 harness ran every benchmark as
"ingest everything, then ask all questions". v1 lets a paradigm
interleave ingestion, clock advancement (with optional consolidation /
archival), and probes, so paradigms like retention decay or
prospective memory can be expressed directly.
"""

from harness.benchmarks.academic.emem_bench_v1.schedule import (
    AdvanceClockPhase,
    IngestPhase,
    Observation,
    ProbePhase,
    Schedule,
)

__all__ = [
    "AdvanceClockPhase",
    "IngestPhase",
    "Observation",
    "ProbePhase",
    "Schedule",
]
