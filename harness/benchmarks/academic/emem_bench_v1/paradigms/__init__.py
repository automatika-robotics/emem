"""Cognitive-paradigm generators for eMEM-Bench v1.

Each paradigm lives in its own submodule (`drm`, `pattern_separation`,
…). A generator takes the ProcTHOR scene inventory + config and emits
a list of :class:`CandidateQuestion` records. Candidates flow through
an optional LLM pre-filter, then a human review CLI, and end up in
``<paradigm>.jsonl`` under this package.
"""

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
    ReviewDecision,
    load_candidates,
    save_candidates,
)

__all__ = [
    "CandidateQuestion",
    "ParadigmGenerator",
    "ReviewDecision",
    "load_candidates",
    "save_candidates",
]
