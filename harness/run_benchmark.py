import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from harness.benchmarks.academic.ablation import ABLATIONS
from harness.benchmarks.academic.replay_runner import BenchmarkReport, BenchmarkRunner


def _make_loader(dataset: str, data_dir: str) -> Any:
    """Create the appropriate dataset loader.

    :param dataset: One of ``"sqa3d"``, ``"locomo"``, ``"open-eqa"``.
    :param data_dir: Path to the dataset directory.
    :returns: Loader instance.
    """
    if dataset == "sqa3d":
        from harness.benchmarks.academic.loaders.sqa3d import SQA3DLoader
        return SQA3DLoader(data_dir)
    if dataset == "locomo":
        from harness.benchmarks.academic.loaders.locomo import LoCoMoLoader
        return LoCoMoLoader(data_dir)
    if dataset == "open-eqa":
        from harness.benchmarks.academic.loaders.open_eqa import OpenEQALoader
        return OpenEQALoader(data_dir)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _make_scorer(dataset: str, **kwargs: Any) -> Any:
    """Create the appropriate scorer for the given dataset.

    :param dataset: One of ``"sqa3d"``, ``"locomo"``, ``"open-eqa"``.
    :returns: Scorer instance.
    """
    if dataset == "sqa3d":
        from harness.benchmarks.academic.scorers.exact_match import ExactMatchScorer
        return ExactMatchScorer()
    if dataset == "locomo":
        from harness.benchmarks.academic.scorers.f1 import F1Scorer
        return F1Scorer()
    if dataset == "open-eqa":
        from harness.benchmarks.academic.scorers.llm_match import LLMMatchScorer
        from harness.providers.ollama_llm import OllamaLLMClient
        llm = OllamaLLMClient(
            model=kwargs.get("judge_model", "qwen3.5:4b"),
            base_url=kwargs.get("ollama_url", "http://localhost:11434"),
        )
        return LLMMatchScorer(llm_chat=llm._chat)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _make_providers(provider: str, embed_model: str, llm_model: str, **kwargs: Any) -> tuple:
    """Create ``(embedder, llm_client, agent_kwargs)`` for the given provider.

    :param provider: ``"ollama"`` or ``"gemini"``.
    :param embed_model: Embedding model name.
    :param llm_model: LLM model name.
    :returns: Tuple of (embedding_provider, llm_client, agent_kwargs_dict).
    """
    if provider == "ollama":
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider
        from harness.providers.ollama_llm import OllamaLLMClient

        url = kwargs.get("ollama_url", "http://localhost:11434")
        return (
            OllamaEmbeddingProvider(embed_model, url),
            OllamaLLMClient(llm_model, url),
            {"model": llm_model, "base_url": url},
        )
    if provider == "gemini":
        from harness.providers.gemini_embeddings import GeminiEmbeddingProvider
        from harness.providers.gemini_llm import GeminiLLMClient

        key = kwargs.get("gemini_api_key")
        return (
            GeminiEmbeddingProvider(model=embed_model, api_key=key),
            GeminiLLMClient(model=llm_model, api_key=key),
            {"model": llm_model, "api_key": key},
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def _make_agent_factory(
    provider: str,
    agent_kwargs: Dict[str, Any],
    system_preamble: Optional[str] = None,
) -> Any:
    """Create an agent factory function for the given provider.

    :param provider: ``"ollama"`` or ``"gemini"``.
    :param agent_kwargs: Keyword arguments passed to the agent constructor.
    :param system_preamble: Custom system prompt preamble. If ``None``, the
        agent uses the default preamble.
    :returns: Callable that takes a memory instance and returns an agent.
    """
    def factory(mem: Any) -> Any:
        system_prompt = None
        if system_preamble is not None:
            from harness.agent.prompts import build_system_prompt
            system_prompt = build_system_prompt(
                mem.get_tool_definitions(), preamble=system_preamble,
            )
        if provider == "gemini":
            from harness.agent.react_agent import GeminiReactAgent
            return GeminiReactAgent(mem, system_prompt=system_prompt, **agent_kwargs)
        from harness.agent.react_agent import ReactAgent
        return ReactAgent(mem, system_prompt=system_prompt, **agent_kwargs)
    return factory


_QUESTION_TEMPLATES: Dict[str, str] = {
    "locomo": (
        "Based on the conversation history stored in memory, answer this question.\n"
        "Give a short, direct answer — just the key fact, no explanation.\n\n"
        "Question: {question}"
    ),
    "sqa3d": (
        "Based on the 3D scene objects stored in memory, answer this question.\n"
        "Give a short answer — a single word or brief phrase.\n\n"
        "Question: {question}"
    ),
    "open-eqa": (
        "Based on the observations stored in memory, answer this question.\n"
        "Give a concise answer in one sentence or less.\n\n"
        "Question: {question}"
    ),
}

_SYSTEM_PREAMBLES: Dict[str, str] = {
    "locomo": (
        "You are a conversational memory assistant. You have access to a memory "
        "system that stores past conversations. Use the tools to search your "
        "memory and answer questions about what was discussed. Give short, "
        "factual answers — just the key information, no elaboration. "
        "You have access to the following tools:"
    ),
    "sqa3d": (
        "You are a situated 3D question answering assistant. You have access to "
        "a memory system that stores objects and their 3D positions in a scene. "
        "Use the tools to find objects and answer spatial questions. Give short "
        "answers — a single word or brief phrase. "
        "You have access to the following tools:"
    ),
    "open-eqa": (
        "You are an embodied question answering assistant. You have access to a "
        "memory system that stores observations from exploring an environment. "
        "Use the tools to recall what was observed and answer questions. Give "
        "concise answers. You have access to the following tools:"
    ),
}


def _print_report(report: BenchmarkReport) -> None:
    """Print a human-readable summary of a benchmark report.

    :param report: The benchmark report to display.
    """
    s = report.summary()
    print(f"\n{'=' * 60}")
    print(f"  {s['dataset'].upper()} -- {s['ablation']}")
    print(f"  {s['n_questions']} questions across {s['n_samples']} samples")
    print(f"  Time: {s['total_time_s']}s")
    print(f"{'=' * 60}")
    for k, v in s["metrics"].items():
        print(f"  {k:20s}: {v:6.2f}")
    print()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for academic benchmark evaluation.

    :param argv: Command-line arguments (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Academic benchmark evaluation for eMEM")
    parser.add_argument("--dataset", required=True, choices=["sqa3d", "locomo", "open-eqa"])
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--ablation", default="full", help="Comma-separated ablation names")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini"])
    parser.add_argument("--embed-model", default="nomic-embed-text-v2-moe:latest")
    parser.add_argument("--llm-model", default="qwen3.5:4b")
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ablation_names = [a.strip() for a in args.ablation.split(",")]
    for name in ablation_names:
        if name not in ABLATIONS:
            print(f"Unknown ablation: {name!r}. Available: {list(ABLATIONS)}", file=sys.stderr)
            sys.exit(1)

    embedder, llm, agent_kwargs = _make_providers(
        args.provider, args.embed_model, args.llm_model,
        ollama_url=args.ollama_url, gemini_api_key=args.gemini_api_key,
    )
    agent_factory = _make_agent_factory(
        args.provider, agent_kwargs,
        system_preamble=_SYSTEM_PREAMBLES.get(args.dataset),
    )
    scorer = _make_scorer(
        args.dataset, judge_model=args.judge_model, ollama_url=args.ollama_url,
    )

    reports: List[BenchmarkReport] = []
    for abl_name in ablation_names:
        ablation = ABLATIONS[abl_name]
        loader = _make_loader(args.dataset, args.data_dir)

        runner = BenchmarkRunner(
            loader=loader,
            scorer=scorer,
            ablation=ablation,
            embedding_provider=embedder,
            llm_client=llm,
            agent_factory=agent_factory,
            max_samples=args.max_samples,
            question_template=_QUESTION_TEMPLATES.get(args.dataset),
        )

        report = runner.run()
        reports.append(report)

        if not args.json:
            _print_report(report)

    if args.json:
        print(json.dumps([r.summary() for r in reports], indent=2))


if __name__ == "__main__":
    main()
