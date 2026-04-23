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
from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import DRMGenerator
from harness.benchmarks.academic.emem_bench_v1.paradigms.prefilter import (
    prefilter_candidates,
    rubric_summary,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.review import run_review
from harness.benchmarks.academic.emem_bench_v1.scene_entries import (
    load_scene_entries,
)


# Registry of paradigm factories keyed by name. Factories take a
# single ``llm_chat`` callable and return a configured generator.
# Adding a new paradigm is a one-line change here.
_PARADIGM_REGISTRY: Dict[
    str, Callable[[Callable[[str], str], int], ParadigmGenerator]
] = {
    "drm": lambda chat, total: DRMGenerator(llm_chat=chat, target_total=total),
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fresh start: overwrite any prior partial file. If resume is
    # needed later, we'd key on question_id.
    out_path.write_text("")
    total = 0
    seen_ids: set = set()
    for i, scene in enumerate(scenes, start=1):
        if total >= args.target_total:
            break
        remaining = args.target_total - total
        budget = min(args.n_per_scene, remaining)
        cands = generator.generate([scene], n_per_scene=budget, seed=args.seed or 0)
        with out_path.open("a") as f:
            written = 0
            for c in cands:
                if c.question_id in seen_ids:
                    continue  # dedupe across scenes with identical probes
                seen_ids.add(c.question_id)
                f.write(_json.dumps(c.to_dict()) + "\n")
                written += 1
                total += 1
                if total >= args.target_total:
                    break
        print(
            f"scene {i}/{len(scenes)} {scene.get('sample_id', '?')}: "
            f"+{written} candidates (total: {total}/{args.target_total})",
            flush=True,
        )
    print(f"done: {total} candidates -> {out_path}", flush=True)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Run the interactive review CLI on a candidates file."""
    path = Path(args.candidates)
    if not path.exists():
        print(f"error: candidates file not found: {path}", file=sys.stderr)
        return 2
    run_review(path)
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
    gen.add_argument("--provider", default="ollama", choices=["ollama", "gemini"])
    gen.add_argument("--llm-model", default="qwen3.6:27b")
    gen.add_argument("--ollama-url", default="http://localhost:11434")
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=cmd_generate)

    rev = subparsers.add_parser("review", help="interactive human review")
    rev.add_argument("candidates")
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
