"""Unified CLI for paradigm generation + curation.

Subcommands:

  generate <paradigm>   — emit candidate questions for a paradigm
  review <candidates>   — interactive human review (keep / edit / …)
  prefilter <candidates> — apply the paradigm's rubric via an LLM

Run ``python -m harness.benchmarks.academic.emem_bench_v1.paradigms.cli
--help`` for the full argument surface.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    ParadigmGenerator,
    load_candidates,
    save_candidates,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
    DRMGenerator,
    make_prior_house_lookup,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.pattern_completion import (
    PatternCompletionGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.context_dependent_retrieval import (
    ContextDependentRetrievalGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.long_horizon_interference import (
    LongHorizonInterferenceGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.prospective_memory import (
    ProspectiveMemoryGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.serial_position import (
    SerialPositionGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.pattern_separation import (
    PatternSeparationGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.retention_interval_decay import (
    RetentionIntervalDecayGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.source_monitoring import (
    SourceMonitoringGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.prefilter import (
    prefilter_candidates,
    rubric_summary,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.review import run_review
from harness.benchmarks.academic.emem_bench_v1.scene_entries import (
    load_scene_entries,
)


def _make_drm_generator(chat: Callable[[str], str], total: int) -> ParadigmGenerator:
    """Build the DRM generator with a ProcTHOR-backed house lookup.

    The house lookup re-uses the cached ProcTHOR-10K download so
    generation pulls real per-room object lists from the simulator
    rather than VLM captions.
    """
    return DRMGenerator(
        llm_chat=chat,
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_pattern_separation_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the pattern-separation generator with a ProcTHOR-backed house lookup."""
    return PatternSeparationGenerator(
        llm_chat=chat,
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_pattern_completion_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the pattern-completion generator with a ProcTHOR-backed house lookup."""
    return PatternCompletionGenerator(
        llm_chat=chat,
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_source_monitoring_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the source-monitoring generator.

    Source-monitoring works on the trajectory's per-frame layer text
    directly (vlm + object_detection), so it doesn't need a ProcTHOR
    house lookup the way DRM / pattern_separation /
    pattern_completion do.
    """
    return SourceMonitoringGenerator(llm_chat=chat, target_total=total)


def _make_retention_interval_decay_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the retention-interval-decay generator with a ProcTHOR house lookup."""
    return RetentionIntervalDecayGenerator(
        llm_chat=chat,
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_context_dependent_retrieval_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the context-dependent-retrieval generator with a ProcTHOR house lookup."""
    return ContextDependentRetrievalGenerator(
        llm_chat=chat,
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_prospective_memory_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the prospective-memory generator with a ProcTHOR house lookup.

    Doesn't actually use ``chat`` — PM generation is deterministic
    (templated instructions, ProcTHOR-derived items, rotated action
    vocabulary). The factory still accepts ``chat`` to match the
    registry signature.
    """
    del chat
    return ProspectiveMemoryGenerator(
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_long_horizon_interference_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the long-horizon-interference generator with a ProcTHOR house lookup.

    Like prospective_memory, generation is deterministic (set
    arithmetic on ground-truth object types per house pair) so
    ``chat`` is unused.
    """
    del chat
    return LongHorizonInterferenceGenerator(
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _make_serial_position_generator(
    chat: Callable[[str], str], total: int
) -> ParadigmGenerator:
    """Build the serial-position generator with a ProcTHOR house lookup.

    Deterministic (chain partitioning + per-house unique-set
    arithmetic) so ``chat`` is unused.
    """
    del chat
    return SerialPositionGenerator(
        house_lookup=make_prior_house_lookup(),
        target_total=total,
    )


def _load_exclude_probes(
    paths: List[Path],
) -> Dict[str, set]:
    """Read prior candidate JSONL files and collect probe objects per group.

    Used by the ``generate`` command's ``--exclude-from`` flag so a
    supplemental run can avoid proposing anything the reviewer has
    already seen (whether they kept or rejected it). The grouping key
    is paradigm-specific: ``room_type`` for DRM / pattern-completion,
    ``source`` for source-monitoring. Candidates missing
    ``paradigm_metadata`` (or both grouping keys) are skipped.
    """
    out: Dict[str, set] = {}
    for path in paths:
        if not path.exists():
            print(
                f"warning: --exclude-from {path} not found; skipping", file=sys.stderr
            )
            continue
        for cand in load_candidates(path):
            md = cand.paradigm_metadata
            group = md.get("room_type") or md.get("source")
            probe = md.get("probe_object")
            if group and probe:
                out.setdefault(str(group), set()).add(str(probe))
    return out


# Registry of paradigm factories keyed by name. Factories take a
# single ``llm_chat`` callable + target-total and return a configured
# generator. Adding a new paradigm is a one-line change here.
_PARADIGM_REGISTRY: Dict[
    str, Callable[[Callable[[str], str], int], ParadigmGenerator]
] = {
    "drm": _make_drm_generator,
    "pattern_separation": _make_pattern_separation_generator,
    "pattern_completion": _make_pattern_completion_generator,
    "source_monitoring": _make_source_monitoring_generator,
    "retention_interval_decay": _make_retention_interval_decay_generator,
    "context_dependent_retrieval": _make_context_dependent_retrieval_generator,
    "prospective_memory": _make_prospective_memory_generator,
    "long_horizon_interference": _make_long_horizon_interference_generator,
    "serial_position": _make_serial_position_generator,
}


def _build_llm_chat(
    provider: str, model: str, url: str, seed: Optional[int]
) -> Callable[[str], str]:
    """Return a ``(prompt) -> str`` callable backed by the chosen provider."""
    if provider == "ollama":
        from harness.providers.ollama_llm import OllamaLLMClient

        client = OllamaLLMClient(model=model, base_url=url, seed=seed)
        return client._chat
    if provider == "gemini":
        import os

        from harness.providers.gemini_llm import GeminiLLMClient

        client = GeminiLLMClient(model=model, api_key=os.environ.get("GEMINI_API_KEY"))
        return client._generate
    raise ValueError(f"Unknown provider: {provider!r}")


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate candidate questions for ``args.paradigm``.

    Walks scenes one at a time and appends that scene's candidates to
    the output JSONL before moving to the next scene. So long
    multi-hour runs stay resumable (on crash or kill, everything up
    to the last completed scene is already on disk) and progress is
    visible per-scene.
    """
    import json as _json  # local — tests never reach this path

    if args.paradigm not in _PARADIGM_REGISTRY:
        print(
            f"error: unknown paradigm {args.paradigm!r}. "
            f"Known: {sorted(_PARADIGM_REGISTRY)}",
            file=sys.stderr,
        )
        return 2
    scenes = load_scene_entries(Path(args.data_dir), max_samples=args.max_samples)
    if args.only_scenes:
        wanted = set(args.only_scenes)
        scenes = [s for s in scenes if s.get("sample_id") in wanted]
        missing = wanted - {s.get("sample_id") for s in scenes}
        if missing:
            print(
                f"warning: --only-scenes had no match for: {sorted(missing)}",
                file=sys.stderr,
            )
    if not scenes:
        print(f"error: no scenes found in {args.data_dir}", file=sys.stderr)
        return 2
    llm_chat = _build_llm_chat(
        args.provider, args.llm_model, args.ollama_url, args.seed
    )
    factory = _PARADIGM_REGISTRY[args.paradigm]
    # Target-total is enforced outside, so pass a large cap into the
    # generator and control budget via the per-scene loop.
    generator = factory(llm_chat, args.target_total * 2)
    if args.exclude_from:
        exclude_paths = [Path(p) for p in args.exclude_from]
        excluded = _load_exclude_probes(exclude_paths)
        if excluded and hasattr(generator, "_excluded"):
            # Seed the generator's exclusion set with the union of
            # probes from every --exclude-from file.
            for room, probes in excluded.items():
                generator._excluded.setdefault(room, set()).update(
                    p.lower() for p in probes
                )
            print(
                f"exclusion list loaded: "
                f"{sum(len(v) for v in excluded.values())} probes across "
                f"{len(excluded)} rooms",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fresh start: overwrite any prior partial file. If resume is
    # needed later, we'd key on question_id.
    out_path.write_text("")
    total = 0
    seen_ids: set = set()
    units = generator.scene_units(scenes)
    for i, unit in enumerate(units, start=1):
        if total >= args.target_total:
            break
        remaining = args.target_total - total
        budget = min(args.n_per_scene, remaining)
        cands = generator.generate(unit, n_per_scene=budget, seed=args.seed or 0)
        with out_path.open("a") as f:
            written = 0
            for c in cands:
                if c.question_id in seen_ids:
                    continue  # dedupe across units with identical probes
                seen_ids.add(c.question_id)
                f.write(_json.dumps(c.to_dict()) + "\n")
                written += 1
                total += 1
                if total >= args.target_total:
                    break
        label = (
            unit[0].get("sample_id", "?")
            if len(unit) == 1
            else " + ".join(s.get("sample_id", "?") for s in unit)
        )
        print(
            f"unit {i}/{len(units)} {label}: "
            f"+{written} candidates (total: {total}/{args.target_total})",
            flush=True,
        )
    print(f"done: {total} candidates -> {out_path}", flush=True)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Run the interactive review CLI on a candidates file.

    With ``--data-dir``, the reviewer also sees the full observation
    list for each candidate's probe room so they can verify the
    ground-truth answer without trusting the post-filter blindly.
    """
    path = Path(args.candidates)
    if not path.exists():
        print(f"error: candidates file not found: {path}", file=sys.stderr)
        return 2
    data_dir = Path(args.data_dir) if args.data_dir else None
    run_review(path, data_dir=data_dir)
    return 0


def cmd_prefilter(args: argparse.Namespace) -> int:
    """Apply a paradigm's rubric over a candidates file.

    Writes a ``.prefilter-kept.jsonl`` and ``.prefilter-rejected.jsonl``
    beside the input so the downstream review step sees only kept
    candidates.
    """
    if args.paradigm not in _PARADIGM_REGISTRY:
        print(f"error: unknown paradigm {args.paradigm!r}", file=sys.stderr)
        return 2
    candidates = load_candidates(Path(args.candidates))
    if not candidates:
        print(f"error: no candidates at {args.candidates}", file=sys.stderr)
        return 2
    paradigm_cls = type(_PARADIGM_REGISTRY[args.paradigm](lambda _p: "", 0))
    rubric = paradigm_cls.prefilter_rubric()
    llm_chat = _build_llm_chat(
        args.provider, args.llm_model, args.ollama_url, args.seed
    )
    result = prefilter_candidates(
        candidates, rubric=rubric, llm_chat=llm_chat, keep_ambiguous=True
    )
    y, amb, no = rubric_summary(result)
    print(f"pre-filter: yes={y} ambiguous={amb} no={no}")
    base = Path(args.candidates)
    kept_path = base.with_suffix(".prefilter-kept.jsonl")
    rej_path = base.with_suffix(".prefilter-rejected.jsonl")
    save_candidates(kept_path, result.kept)
    save_candidates(rej_path, result.rejected)
    print(f"  kept:     {kept_path}")
    print(f"  rejected: {rej_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="generate candidate questions")
    gen.add_argument("paradigm", choices=sorted(_PARADIGM_REGISTRY))
    gen.add_argument("--data-dir", required=True, help="Path to v1 scenes directory")
    gen.add_argument("--out", required=True, help="Output candidates.jsonl path")
    gen.add_argument("--n-per-scene", type=int, default=6)
    gen.add_argument("--target-total", type=int, default=120)
    gen.add_argument("--max-samples", type=int, default=None)
    gen.add_argument(
        "--only-scenes",
        action="append",
        default=[],
        help=(
            "Restrict generation to the named sample_ids (repeatable). "
            "Applied after --max-samples. Useful for resuming a single "
            "scene after a killed run without rebuilding the manifest."
        ),
    )
    gen.add_argument(
        "--exclude-from",
        action="append",
        default=[],
        help=(
            "Path to a prior candidates/curated/rejected JSONL. Probe "
            "objects from those files are fed into the generator as an "
            "exclusion set per room, so supplemental runs propose new "
            "items instead of repeating what the reviewer has seen. "
            "Can be passed multiple times."
        ),
    )
    gen.add_argument("--provider", default="ollama", choices=["ollama", "gemini"])
    gen.add_argument("--llm-model", default="qwen3.6:27b")
    gen.add_argument("--ollama-url", default="http://localhost:11434")
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=cmd_generate)

    rev = subparsers.add_parser("review", help="interactive human review")
    rev.add_argument("candidates")
    rev.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Optional path to the v1 scenes directory. When set, each "
            "candidate's full room-observation list is shown in the "
            "review panel so the reviewer can verify the ground truth."
        ),
    )
    rev.set_defaults(func=cmd_review)

    pre = subparsers.add_parser("prefilter", help="apply the paradigm's rubric via LLM")
    pre.add_argument("paradigm", choices=sorted(_PARADIGM_REGISTRY))
    pre.add_argument("candidates")
    pre.add_argument("--provider", default="ollama", choices=["ollama", "gemini"])
    pre.add_argument("--llm-model", default="gemma4:31b")
    pre.add_argument("--ollama-url", default="http://localhost:11434")
    pre.add_argument("--seed", type=int, default=42)
    pre.set_defaults(func=cmd_prefilter)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI dispatch."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
