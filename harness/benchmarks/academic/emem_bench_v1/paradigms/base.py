"""Base types for paradigm generators.

A paradigm implementation provides:

1. A subclass of :class:`ParadigmGenerator` that implements
   ``generate(...)`` to emit candidate questions for a batch of scenes.
2. A ``prefilter_rubric()`` classmethod returning the yes / ambiguous /
   no rubric prompt the pre-filter will apply.
3. A ``build_schedule(candidate, scenes)`` staticmethod that converts a
   curated candidate into a :class:`~...emem_bench_v1.schedule.Schedule`
   at benchmark run time.

The CLI (:mod:`.review`) and the pre-filter (:mod:`.prefilter`) consume
these dataclasses.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ReviewDecision(str, Enum):
    """Outcomes of the per-candidate human review step."""

    KEEP = "keep"
    EDITED = "edited"
    DISCARD = "discard"


@dataclass
class CandidateQuestion:
    """A generator-emitted question with enough context for curation.

    The ``paradigm_metadata`` dict is paradigm-specific — it carries
    whatever the generator needs the schedule-builder and the human
    reviewer to see (which scene, which room, which object was the
    probe target, etc.). Kept as a free-form dict so each paradigm
    can evolve its own schema without changing this base class.

    The optional ``review_decision`` / ``review_reason`` fields are
    populated by the human-review CLI; they're empty at generation
    time and serialised alongside the rest so reviewer intent is
    preserved even in rejected sets.
    """

    question_id: str
    question: str
    answer: str
    category: str
    paradigm: str
    scene_ids: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    paradigm_metadata: Dict[str, Any] = field(default_factory=dict)
    generator_notes: str = ""
    review_decision: Optional[str] = None
    review_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSONL serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateQuestion":
        """Reconstruct from a JSONL row, tolerating missing optional fields."""
        return cls(
            question_id=str(data["question_id"]),
            question=str(data["question"]),
            answer=str(data["answer"]),
            category=str(data.get("category", data.get("paradigm", ""))),
            paradigm=str(data.get("paradigm", "")),
            scene_ids=list(data.get("scene_ids") or []),
            tools_expected=list(data.get("tools_expected") or []),
            paradigm_metadata=dict(data.get("paradigm_metadata") or {}),
            generator_notes=str(data.get("generator_notes") or ""),
            review_decision=data.get("review_decision"),
            review_reason=str(data.get("review_reason") or ""),
        )


def save_candidates(path: Path, candidates: Iterable[CandidateQuestion]) -> int:
    """Write candidates as newline-delimited JSON; return the count.

    Overwrites the target file. Parent directory is created if missing.

    :param path: Destination JSONL path.
    :param candidates: Iterable of CandidateQuestion records.
    :returns: Number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict()) + "\n")
            n += 1
    return n


def load_candidates(path: Path) -> List[CandidateQuestion]:
    """Read a JSONL file of candidates; return an empty list if missing.

    :param path: Source JSONL path.
    :returns: Parsed candidates in file order.
    """
    if not path.exists():
        return []
    out: List[CandidateQuestion] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(CandidateQuestion.from_dict(json.loads(line)))
    return out


class ParadigmGenerator(ABC):
    """Abstract base class for paradigm-specific generators.

    Subclasses own the scene-filtering and LLM-prompting logic for
    their paradigm. The pre-filter and review CLI consume the
    :class:`CandidateQuestion` records produced by
    :meth:`generate`.

    Subclass contract:

    - ``name``: class-level string, used for filenames and tags.
    - :meth:`generate`: produce candidates for a batch of scenes.
    - :meth:`prefilter_rubric`: the yes / ambiguous / no rubric
      prompt the pre-filter will apply.
    """

    name: str = ""

    @abstractmethod
    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        """Emit candidate questions for ``scenes``.

        :param scenes: List of scene-manifest rows (per-entry output of
            :class:`~...emem_bench_v1.loader.SceneManifestLoader` with
            the trajectory merged in).
        :param n_per_scene: Target number of candidates to emit per
            scene; generators are expected to hit roughly 2× the final
            curated count because pre-filter + human review will drop
            about half.
        :param seed: Deterministic seed for any sampling the generator
            does.
        :returns: Candidates in generation order.
        """

    @classmethod
    @abstractmethod
    def prefilter_rubric(cls) -> str:
        """Return the rubric prompt used by :mod:`.prefilter`.

        The rubric must instruct the LLM to emit exactly one of the
        words ``yes``, ``ambiguous``, or ``no`` so the filter can
        partition candidates deterministically.
        """
