from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from harness.benchmarks.academic.ablation import ABLATIONS, DATASET_TOOL_FILTERS
from harness.benchmarks.academic.replay_runner import BenchmarkReport, BenchmarkRunner


def _make_loader(dataset: str, data_dir: str) -> Any:
    """Create the appropriate dataset loader.

    :param dataset: One of ``"sqa3d"``, ``"locomo"``.
    :param data_dir: Path to the dataset directory.
    :returns: Loader instance.
    """
    if dataset == "sqa3d":
        from harness.benchmarks.academic.loaders.sqa3d import SQA3DLoader

        return SQA3DLoader(data_dir)
    if dataset == "locomo":
        from harness.benchmarks.academic.loaders.locomo import LoCoMoLoader

        return LoCoMoLoader(data_dir)
    if dataset == "emem-bench":
        from harness.benchmarks.academic.loaders.emem_bench import EMEMBenchLoader

        return EMEMBenchLoader(data_dir)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _make_scorer(dataset: str, **kwargs: Any) -> Any:
    """Create the appropriate scorer for the given dataset.

    :param dataset: One of ``"sqa3d"``, ``"locomo"``, ``"emem-bench"``.
    :returns: Scorer instance.
    """
    if dataset == "sqa3d":
        from harness.benchmarks.academic.scorers.exact_match import ExactMatchScorer

        return ExactMatchScorer()
    if dataset == "locomo":
        from harness.benchmarks.academic.scorers.f1 import F1Scorer

        return F1Scorer()
    if dataset == "emem-bench":
        from harness.benchmarks.academic.scorers.emem_bench import EMEMBenchScorer

        judge_client = kwargs.get("judge_client") or kwargs.get("llm_client")
        provider = kwargs.get("judge_provider") or kwargs.get("provider", "ollama")
        llm_chat = (
            judge_client._generate if provider == "gemini" else judge_client._chat
        )
        return EMEMBenchScorer(llm_chat=llm_chat)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _make_providers(
    provider: str, embed_model: str, llm_model: str, **kwargs: Any
) -> tuple:
    """Create ``(embedder, llm_client, agent_kwargs)`` for the given provider.

    :param provider: ``"ollama"`` or ``"gemini"``.
    :param embed_model: Embedding model name.
    :param llm_model: LLM model name.
    :keyword seed: Optional integer seed passed to the LLM sampler (Ollama
        only); supplied via the ``seed`` option to make outputs
        reproducible.
    :returns: Tuple of (embedding_provider, llm_client, agent_kwargs_dict).
    """
    if provider == "ollama":
        from harness.providers.ollama_embeddings import OllamaEmbeddingProvider
        from harness.providers.ollama_llm import OllamaLLMClient

        url = kwargs.get("ollama_url", "http://localhost:11434")
        seed = kwargs.get("seed")
        agent_kwargs = {"model": llm_model, "base_url": url}
        if seed is not None:
            agent_kwargs["seed"] = seed
        return (
            OllamaEmbeddingProvider(embed_model, url),
            OllamaLLMClient(llm_model, url, seed=seed),
            agent_kwargs,
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


def _dataset_tool_filter(dataset: str) -> Optional[List[str]]:
    """Return the dataset-specific tool allow-list, or ``None`` for no filter.

    LoCoMo runs over text conversations so spatial/body-state tools are
    filtered out to shrink the choice space for the agent's tool
    selector. eMEM-Bench exercises all 10 tools and gets no filter.
    """
    filt = DATASET_TOOL_FILTERS.get(dataset)
    return list(filt) if filt is not None else None


def _apply_seed(seed: Optional[int]) -> None:
    """Seed Python's ``random`` and NumPy. The Ollama sampler seed is
    applied per-request via the LLM client configuration elsewhere.

    :param seed: Integer seed, or ``None`` to leave RNGs untouched.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _aggregate_summary(reports: List[BenchmarkReport]) -> Dict[str, Any]:
    """Aggregate per-category scores across runs of the same ablation.

    :param reports: List of ``BenchmarkReport`` from repeated runs.
    :returns: Dict with ``n_runs`` and, per category, mean and
        standard error (stderr = stdev / sqrt(n)).
    """
    summaries = [r.summary() for r in reports]
    n = len(summaries)
    per_cat: Dict[str, List[float]] = defaultdict(list)
    overalls: List[float] = []
    for s in summaries:
        score = s.get("metrics", {}).get("score")
        if score is not None:
            overalls.append(float(score))
        for cat, info in s.get("per_category", {}).items():
            per_cat[cat].append(float(info["mean_score"]))

    def _mean_stderr(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "stderr": 0.0}
        if len(xs) == 1:
            return {"mean": xs[0], "stderr": 0.0}
        mean = statistics.fmean(xs)
        stderr = statistics.stdev(xs) / math.sqrt(len(xs))
        return {"mean": round(mean, 2), "stderr": round(stderr, 2)}

    agg: Dict[str, Any] = {"n_runs": n}
    agg["per_category"] = {cat: _mean_stderr(xs) for cat, xs in per_cat.items()}
    if overalls:
        agg["overall"] = _mean_stderr(overalls)
    return agg


def _print_aggregate(ablation: str, reports: List[BenchmarkReport]) -> None:
    """Print a mean ± stderr summary across runs for one ablation.

    :param ablation: Ablation name for the header.
    :param reports: Per-run reports for this ablation.
    """
    agg = _aggregate_summary(reports)
    n = agg["n_runs"]
    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE ({ablation}) over {n} runs")
    print(f"{'=' * 60}")
    if "overall" in agg:
        o = agg["overall"]
        print(f"  overall       : {o['mean']:6.2f} ± {o['stderr']:.2f}")
    print(f"\n  {'Category':<12} {'Mean':>8} {'StdErr':>8}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8}")
    for cat, info in agg["per_category"].items():
        print(f"  {cat:<12} {info['mean']:>8.2f} {info['stderr']:>8.2f}")
    print()


def _make_agent_factory(
    provider: str,
    agent_kwargs: Dict[str, Any],
    system_preamble: Optional[str] = None,
    think: bool = False,
    agent_mode: str = "native",
) -> Any:
    """Create an agent factory function for the given provider.

    :param provider: ``"ollama"`` or ``"gemini"``.
    :param agent_kwargs: Keyword arguments passed to the agent constructor.
    :param system_preamble: Custom system prompt preamble. If ``None``, the
        agent uses the default preamble.
    :param think: Enable thinking for Ollama models (ReAct mode only).
    :param agent_mode: ``"native"`` (Ollama native tool-calling via
        ``/api/chat`` ``tools=[...]``) or ``"react"`` (text-mode ReAct
        loop). Ollama-only; Gemini always uses its ReAct variant.
    :returns: Callable that takes a memory instance and returns an agent.
    """

    def _react_prompt(mem: Any) -> Optional[str]:
        if system_preamble is None:
            return None
        from harness.agent.prompts import build_system_prompt

        return build_system_prompt(mem.get_tool_definitions(), preamble=system_preamble)

    def factory(mem: Any) -> Any:
        if provider == "gemini":
            from harness.agent.react_agent import GeminiReactAgent

            return GeminiReactAgent(
                mem, system_prompt=_react_prompt(mem), **agent_kwargs
            )

        if agent_mode == "native":
            from harness.agent.react_agent import NativeToolCallAgent

            native_kwargs = {k: v for k, v in agent_kwargs.items() if k != "think"}
            return NativeToolCallAgent(
                mem,
                system_prompt=system_preamble,
                **native_kwargs,
            )

        from harness.agent.react_agent import ReactAgent

        return ReactAgent(
            mem, system_prompt=_react_prompt(mem), think=think, **agent_kwargs
        )

    return factory


_QUESTION_TEMPLATES: Dict[str, str] = {
    "locomo": (
        "Answer this question using the conversation history in memory.\n"
        "Your answer MUST be 5 words or fewer — the key fact only, no "
        "explanation.\n"
        "Convert any relative dates to absolute dates.\n"
        "Commit to your best answer whenever retrieved observations are "
        "relevant to the question, even if no single observation states "
        "the answer verbatim; combining and paraphrasing across "
        "observations is expected. If the question uses 'would', "
        "'might', 'likely', 'is it possible', reason from what was "
        "retrieved and give a direct yes/no or short phrase.\n"
        "Only respond with UNANSWERABLE if the retrieved observations "
        "contain nothing about the topic of the question.\n\n"
        "Question: {question}"
    ),
    "sqa3d": (
        "Based on the 3D scene objects stored in memory, answer this question.\n"
        "Give a short answer — a single word or brief phrase.\n\n"
        "Question: {question}"
    ),
    "emem-bench": (
        "You are an embodied agent with a spatio-temporal memory of your exploration.\n"
        "Use your memory tools to answer the following question.\n"
        "Give a concise answer — a short phrase or sentence.\n"
        "Use spatial coordinates when relevant.\n\n"
        "Question: {question}"
    ),
}

_SYSTEM_PREAMBLES: Dict[str, str] = {
    "locomo": (
        "You are a conversational memory assistant with tools that search "
        "past conversations. Pick the tool that matches the question's "
        "shape: temporal questions (when, what date, last X days) call "
        "temporal_query; questions about a specific named entity call "
        "entity_query; general topic lookups call semantic_search; "
        "long-horizon summaries call search_gists. "
        "After a tool returns relevant observations, commit to your best "
        "answer in 5 words or fewer. Extract the most likely fact from "
        "the retrieved text even when it does not repeat the question "
        "verbatim; memory answers live across multiple observations and "
        "require combining / paraphrasing. "
        "Always convert relative dates to absolute dates. "
        "Only emit UNANSWERABLE if the retrieved observations contain "
        "nothing about the topic — not merely because the wording "
        "differs from the question."
    ),
    "sqa3d": (
        "You are a situated 3D question answering assistant. You have access to "
        "a memory system that stores objects and their 3D positions in a scene. "
        "Use the tools to find objects and answer spatial questions. Give short "
        "answers — a single word or brief phrase. "
        "You have access to the following tools:"
    ),
    "emem-bench": (
        "You are an embodied robot assistant with a spatio-temporal memory system. "
        "You have explored an environment and stored observations across multiple "
        "perception layers (visual descriptions, object detections, place labels) "
        "as well as body state (battery, temperature). "
        "Use the available tools to search your memory and answer questions about "
        "what you observed, where things are, and your body state. "
        "Give concise answers. Use spatial coordinates when they help. "
        "You have access to the following tools:"
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

    if "per_category" in s:
        print(f"\n  {'Category':<12} {'N':>5} {'Score':>7} {'Zero':>5} {'Perfect':>8}")
        print(f"  {'-' * 12} {'-' * 5} {'-' * 7} {'-' * 5} {'-' * 8}")
        for cat, info in s["per_category"].items():
            print(
                f"  {cat:<12} {info['n']:>5} {info['mean_score']:>7.1f}"
                f" {info['n_zero']:>5} {info['n_perfect']:>8}"
            )
    print()


def _print_details(report: BenchmarkReport) -> None:
    """Print detailed per-question results for failure analysis.

    :param report: The benchmark report to display.
    """
    for sr in report.sample_results:
        print(
            f"\n--- Sample: {sr.sample_id} ({len(sr.question_results)} questions) ---"
        )
        for qr in sr.question_results:
            score = qr.scores.get("score", 0)
            tools = ",".join(qr.tools_used) if qr.tools_used else "none"
            marker = "OK" if score >= 50 else "LOW" if score > 0 else "FAIL"
            gt_display = qr.ground_truth if qr.ground_truth else "(empty)"
            print(
                f"\n  [{marker}] q={qr.question_id} cat={qr.category} score={score:.0f} tools=[{tools}]"
            )
            print(f"    Q: {qr.question}")
            print(f"    GT: {gt_display}")
            print(f"    Pred: {qr.prediction[:200]}")
    print()


def main(argv: Optional[List[str]] = None) -> None:  # noqa: C901  # TODO: split into helpers
    """CLI entry point for academic benchmark evaluation.

    :param argv: Command-line arguments (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Academic benchmark evaluation for eMEM"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["sqa3d", "locomo", "emem-bench"]
    )
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument(
        "--ablation", default="full", help="Comma-separated ablation names"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-questions-per-sample",
        type=int,
        default=None,
        help=(
            "Cap on questions evaluated per sample. Useful for fast "
            "subsample validation (e.g. --max-questions-per-sample 40 "
            "to probe tool-selection behaviour quickly)."
        ),
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini"])
    parser.add_argument("--embed-model", default="nomic-embed-text-v2-moe:latest")
    parser.add_argument(
        "--llm-model",
        default="qwen3.5:27b",
        help="Agent LLM model (default: qwen3.5:27b)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "LLM model for eMEM-Bench judge scoring. If omitted, reuses "
            "--llm-model. Recommended to use a different family (e.g. "
            "gemma4:31b) to avoid self-grading bias."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base random seed. Applied to Python random, NumPy, and the "
            "Ollama sampler (for reproducibility). With --n-runs > 1, each "
            "run uses seed + run_index."
        ),
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help=(
            "Repeat the benchmark N times and report mean and standard "
            "error per category. Requires --seed to be deterministic."
        ),
    )
    parser.add_argument(
        "--agent",
        choices=["native", "react"],
        default="native",
        help=(
            "Agent loop: 'native' uses Ollama's native tool-calling API "
            "(default, recommended for Qwen 3.5 / Gemma 4 / Llama 3.x) "
            "and 'react' uses text-mode ReAct prompting (fallback for "
            "models without tools capability). If 'native' is selected "
            "and the model does not advertise the tools capability, the "
            "runner falls back to 'react' with a warning."
        ),
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument(
        "--think", action="store_true", help="Enable thinking for Ollama models"
    )
    parser.add_argument(
        "--details", action="store_true", help="Print full Q/A/GT for each question"
    )
    parser.add_argument(
        "--recency-weight",
        type=float,
        default=0.0,
        help="Recency weighting alpha (0=disabled, try 0.3)",
    )
    parser.add_argument(
        "--recency-halflife",
        type=float,
        default=2592000.0,
        help="Recency halflife in seconds (default 30 days)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ablation_names = [a.strip() for a in args.ablation.split(",")]
    for name in ablation_names:
        if name not in ABLATIONS:
            print(
                f"Unknown ablation: {name!r}. Available: {list(ABLATIONS)}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.n_runs > 1 and args.seed is None:
        print(
            "Warning: --n-runs > 1 without --seed produces non-deterministic "
            "variance; pass --seed N for reproducible repeats.",
            file=sys.stderr,
        )

    mem_config_overrides: Dict[str, Any] = {}
    if args.recency_weight > 0:
        mem_config_overrides["recency_weight"] = args.recency_weight
        mem_config_overrides["recency_halflife"] = args.recency_halflife

    all_reports: List[List[BenchmarkReport]] = []  # all_reports[abl_idx][run_idx]
    for _ in ablation_names:
        all_reports.append([])

    for run_idx in range(args.n_runs):
        run_seed = None if args.seed is None else args.seed + run_idx
        _apply_seed(run_seed)

        embedder, llm, agent_kwargs = _make_providers(
            args.provider,
            args.embed_model,
            args.llm_model,
            ollama_url=args.ollama_url,
            gemini_api_key=args.gemini_api_key,
            seed=run_seed,
        )
        judge_model = args.judge_model or args.llm_model
        if args.dataset == "emem-bench" and judge_model != args.llm_model:
            _, judge_llm, _ = _make_providers(
                args.provider,
                args.embed_model,
                judge_model,
                ollama_url=args.ollama_url,
                gemini_api_key=args.gemini_api_key,
                seed=run_seed,
            )
        else:
            judge_llm = llm

        agent_mode = args.agent
        if agent_mode == "native" and args.provider == "ollama" and run_idx == 0:
            from harness.agent.react_agent import ollama_model_supports_tools

            if not ollama_model_supports_tools(args.llm_model, args.ollama_url):
                print(
                    f"Warning: model {args.llm_model!r} does not advertise the "
                    "'tools' capability; falling back to --agent react.",
                    file=sys.stderr,
                )
                agent_mode = "react"

        agent_factory = _make_agent_factory(
            args.provider,
            agent_kwargs,
            system_preamble=_SYSTEM_PREAMBLES.get(args.dataset),
            think=args.think,
            agent_mode=agent_mode,
        )
        scorer = _make_scorer(
            args.dataset,
            llm_client=llm,
            judge_client=judge_llm,
            provider=args.provider,
        )

        for abl_idx, abl_name in enumerate(ablation_names):
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
                max_questions_per_sample=args.max_questions_per_sample,
                dataset_tool_filter=_dataset_tool_filter(args.dataset),
                question_template=_QUESTION_TEMPLATES.get(args.dataset),
                mem_config_overrides=mem_config_overrides,
            )

            report = runner.run()
            all_reports[abl_idx].append(report)

            if not args.json:
                header = f"run {run_idx + 1}/{args.n_runs}" if args.n_runs > 1 else ""
                if header:
                    print(f"\n[{header}]")
                _print_report(report)
                if args.details:
                    _print_details(report)

    if args.n_runs > 1 and not args.json:
        for abl_idx, abl_name in enumerate(ablation_names):
            _print_aggregate(abl_name, all_reports[abl_idx])

    if args.json:
        out: List[Dict[str, Any]] = []
        for abl_idx, abl_name in enumerate(ablation_names):
            per_run = [r.summary() for r in all_reports[abl_idx]]
            entry: Dict[str, Any] = {"ablation": abl_name, "runs": per_run}
            if args.n_runs > 1:
                entry["aggregate"] = _aggregate_summary(all_reports[abl_idx])
            out.append(entry)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
