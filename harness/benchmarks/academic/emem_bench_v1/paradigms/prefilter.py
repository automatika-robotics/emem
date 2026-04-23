"""LLM-based pre-filter for paradigm candidates.

Applies a per-paradigm rubric prompt (obtained via
``Generator.prefilter_rubric()``) to each candidate, parses the
model's one-word verdict, and partitions the input into
``kept`` (yes + ambiguous) and ``rejected`` (no) lists. Designed so
the expensive human review only touches candidates that plausibly
test the paradigm.

The LLM is provided via a minimal ``chat(prompt: str) -> str``
callable, matching the ``_chat`` / ``_generate`` interface that the
OllamaLLMClient and GeminiLLMClient expose.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import CandidateQuestion

log = logging.getLogger(__name__)

# Match the first standalone "yes" / "no" / "ambiguous" token in the
# model's output. Case-insensitive; allows surrounding punctuation or
# whitespace. Anything else becomes "ambiguous" so candidates aren't
# silently dropped when a judge emits unexpected prose.
_VERDICT_RE = re.compile(r"\b(yes|no|ambiguous)\b", re.IGNORECASE)


def _parse_verdict(raw: str) -> str:
    """Return one of ``"yes"`` / ``"no"`` / ``"ambiguous"``."""
    if not raw:
        return "ambiguous"
    match = _VERDICT_RE.search(raw)
    if not match:
        return "ambiguous"
    return match.group(1).lower()


@dataclass
class PreFilterResult:
    """Partition of input candidates and verdict counts."""

    kept: List[CandidateQuestion] = field(default_factory=list)
    rejected: List[CandidateQuestion] = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def prefilter_candidates(
    candidates: List[CandidateQuestion],
    rubric: str,
    llm_chat: Callable[[str], str],
    *,
    keep_ambiguous: bool = True,
) -> PreFilterResult:
    """Partition candidates into kept / rejected via the rubric.

    :param candidates: Candidates to judge.
    :param rubric: The paradigm-specific rubric prompt; the builder
        appends the candidate's question + answer + metadata.
    :param llm_chat: A ``(prompt: str) -> str`` callable that returns
        the model's response text.
    :param keep_ambiguous: If True, candidates the model labels
        ``ambiguous`` flow to the human reviewer (``kept``). If False,
        ambiguous lands in ``rejected``. Default True — the plan has
        humans adjudicate borderline cases.
    :returns: A :class:`PreFilterResult`; each rejected candidate has
        its ``review_decision`` set to ``"discard"`` and
        ``review_reason`` stamped with ``"prefilter:<verdict>"``.
    """
    kept: List[CandidateQuestion] = []
    rejected: List[CandidateQuestion] = []
    counts = {"yes": 0, "ambiguous": 0, "no": 0}

    for c in candidates:
        prompt = _build_prompt(rubric, c)
        raw = llm_chat(prompt)
        verdict = _parse_verdict(raw)
        counts[verdict] = counts.get(verdict, 0) + 1

        if verdict == "no" or (verdict == "ambiguous" and not keep_ambiguous):
            c.review_decision = "discard"
            c.review_reason = f"prefilter:{verdict}"
            rejected.append(c)
        else:
            kept.append(c)

    log.info(
        "pre-filter: yes=%d ambiguous=%d no=%d -> kept=%d rejected=%d",
        counts["yes"],
        counts["ambiguous"],
        counts["no"],
        len(kept),
        len(rejected),
    )
    return PreFilterResult(kept=kept, rejected=rejected, counts=counts)


def _build_prompt(rubric: str, candidate: CandidateQuestion) -> str:
    """Append the candidate's fields to the paradigm rubric.

    The final prompt ends with an explicit one-word instruction so
    noisy chat models default to the expected format. Keeping the
    candidate fields in a fixed order makes the prompts reproducible
    and cacheable.
    """
    meta_lines = [f"  {k}: {v}" for k, v in sorted(candidate.paradigm_metadata.items())]
    meta_block = ("\n" + "\n".join(meta_lines)) if meta_lines else ""
    return (
        f"{rubric.strip()}\n\n"
        f"Candidate question: {candidate.question}\n"
        f"Expected answer: {candidate.answer}\n"
        f"Scene ids: {', '.join(candidate.scene_ids) or '-'}\n"
        f"Paradigm metadata:{meta_block or ' -'}\n\n"
        "Answer with exactly one word from: yes, ambiguous, no."
    )


def rubric_summary(result: PreFilterResult) -> Tuple[int, int, int]:
    """Return ``(yes, ambiguous, no)`` counts from a :class:`PreFilterResult`.

    Mostly useful for test assertions; callers can also read
    ``result.counts`` directly.
    """
    return (
        result.counts.get("yes", 0),
        result.counts.get("ambiguous", 0),
        result.counts.get("no", 0),
    )
