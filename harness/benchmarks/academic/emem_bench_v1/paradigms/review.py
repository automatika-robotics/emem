"""Interactive review CLI for paradigm candidates.

Iterates a JSONL of candidates and prompts the reviewer per entry:

  [k]eep        — accept as-is, append to curated output
  [e]dit        — edit the question / answer inline, then keep
  [d]iscard     — reject with an optional reason code
  [s]kip        — defer; candidate stays unreviewed for next run
  [n]otes       — type a note (stored in generator_notes) without
                  changing the decision
  [?]           — re-display the candidate
  [q]uit        — stop the session; progress is persisted

On every decision the CLI rewrites three JSONL files so the session
is fully resumable:

  <basename>.curated.jsonl   — keep + edited
  <basename>.rejected.jsonl  — discarded (with reason codes)
  <basename>.inprogress.jsonl — decisions so far; consumed on resume

Designed to be driven by :func:`run_review`; :func:`main` wires it
to argparse so a generator can be reviewed straight from the CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ReviewDecision,
    load_candidates,
    save_candidates,
)

log = logging.getLogger(__name__)


def _render(candidate: CandidateQuestion, idx: int, total: int) -> str:
    """Format a candidate for terminal display."""
    meta_lines = [
        f"    {k}: {v}" for k, v in sorted(candidate.paradigm_metadata.items())
    ]
    meta_block = "\n".join(meta_lines) if meta_lines else "    (none)"
    return (
        f"\n────────── candidate {idx + 1}/{total} ──────────\n"
        f"  paradigm: {candidate.paradigm}\n"
        f"  scenes:   {', '.join(candidate.scene_ids) or '-'}\n"
        f"  question: {candidate.question}\n"
        f"  answer:   {candidate.answer}\n"
        f"  tools:    {', '.join(candidate.tools_expected) or '-'}\n"
        f"  metadata:\n{meta_block}\n"
        f"  notes:    {candidate.generator_notes or '-'}\n"
    )


def _prompt_input(prompt: str) -> str:
    """Indirection for :func:`input` so tests can monkeypatch easily."""
    return input(prompt)


def _split_reviewed(
    cands: Iterable[CandidateQuestion],
) -> Tuple[List[CandidateQuestion], List[CandidateQuestion]]:
    """Partition a stream into (curated, rejected) based on review_decision."""
    curated: List[CandidateQuestion] = []
    rejected: List[CandidateQuestion] = []
    for c in cands:
        if c.review_decision in (
            ReviewDecision.KEEP.value,
            ReviewDecision.EDITED.value,
        ):
            curated.append(c)
        elif c.review_decision == ReviewDecision.DISCARD.value:
            rejected.append(c)
    return curated, rejected


def _output_paths(base: Path) -> Tuple[Path, Path, Path]:
    """Derive the three artefact paths from a base path.

    The base is typically the candidates file (e.g.
    ``drm.candidates.jsonl``); the derived paths live in the same
    directory.
    """
    stem = base.name
    for suffix in (".candidates.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    d = base.parent
    return (
        d / f"{stem}.curated.jsonl",
        d / f"{stem}.rejected.jsonl",
        d / f"{stem}.inprogress.jsonl",
    )


def run_review(
    candidates_path: Path,
    *,
    input_fn: Callable[[str], str] = _prompt_input,
    output_fn: Callable[[str], None] = print,
) -> Tuple[int, int]:
    """Drive an interactive review session.

    :param candidates_path: JSONL of candidates to review; will be
        merged with any prior in-progress session at the same base.
    :param input_fn: Indirection for ``input()`` so tests can feed
        canned keypresses.
    :param output_fn: Indirection for ``print()`` — tests capture by
        redirecting, but keeping this explicit makes the hot loop
        easy to mock.
    :returns: ``(n_curated, n_rejected)`` after the session ends.
    """
    curated_path, rejected_path, inprogress_path = _output_paths(candidates_path)
    all_candidates = load_candidates(candidates_path)
    previously_reviewed = load_candidates(inprogress_path)
    decided_by_id = {
        c.question_id: c for c in previously_reviewed if c.review_decision is not None
    }
    reviewed: List[CandidateQuestion] = list(decided_by_id.values())

    # Queue = candidates that haven't been decided yet, in their
    # original order. We never mutate this list after construction;
    # decided candidates go into ``reviewed``, pending ones stay in
    # ``pending[idx_holder[0]:]``.
    pending: List[CandidateQuestion] = [
        c for c in all_candidates if c.question_id not in decided_by_id
    ]
    idx_holder = [0]

    output_fn(
        f"Starting review: {len(pending)} pending, "
        f"{len(reviewed)} already decided from prior session."
    )

    def _persist() -> None:
        # inprogress = every decided candidate plus everything still
        # pending — no duplicates, resumable.
        save_candidates(inprogress_path, reviewed + pending[idx_holder[0] :])
        curated, rejected = _split_reviewed(reviewed)
        save_candidates(curated_path, curated)
        save_candidates(rejected_path, rejected)

    while idx_holder[0] < len(pending):
        cand = pending[idx_holder[0]]
        output_fn(_render(cand, idx_holder[0], len(pending)))
        raw = input_fn("  decision [k/e/d/s/n/?/q]: ").strip().lower() or "s"
        if raw == "q":
            output_fn("  quitting; progress saved.")
            break
        if raw == "?":
            continue
        if raw == "s":
            idx_holder[0] += 1
            continue
        if raw == "n":
            note = input_fn("    note: ").strip()
            if note:
                cand.generator_notes = (cand.generator_notes + " | " + note).strip(" |")
            _persist()
            continue
        if raw == "k":
            cand.review_decision = ReviewDecision.KEEP.value
        elif raw == "e":
            new_q = input_fn(f"    question [{cand.question}]: ").strip()
            new_a = input_fn(f"    answer   [{cand.answer}]: ").strip()
            cand = replace(
                cand,
                question=new_q or cand.question,
                answer=new_a or cand.answer,
                review_decision=ReviewDecision.EDITED.value,
            )
        elif raw == "d":
            reason = input_fn("    reason code: ").strip() or "manual_discard"
            cand.review_decision = ReviewDecision.DISCARD.value
            cand.review_reason = reason
        else:
            output_fn(f"  unknown command {raw!r}; skipping")
            idx_holder[0] += 1
            continue

        reviewed.append(cand)
        _persist()
        idx_holder[0] += 1

    _persist()
    curated, rejected = _split_reviewed(reviewed)
    save_candidates(curated_path, curated)
    save_candidates(rejected_path, rejected)
    output_fn(
        f"Session done: curated={len(curated)} rejected={len(rejected)} "
        f"pending={len(pending) - idx_holder[0]}"
    )
    return len(curated), len(rejected)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point — ``.venv/bin/python -m ... review <candidates>``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidates",
        type=Path,
        help="Path to the paradigm candidates JSONL to review.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if not args.candidates.exists():
        print(f"error: candidates file not found: {args.candidates}", file=sys.stderr)
        return 2
    run_review(args.candidates)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
