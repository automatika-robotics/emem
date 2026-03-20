import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from harness.benchmarks.academic.ablation import AblationConfig
from harness.benchmarks.academic.trajectory import BenchmarkQuestion, BenchmarkSample

log = logging.getLogger("harness.academic")


@dataclass
class QuestionResult:
    """Result for a single benchmark question."""

    question_id: str
    question: str
    ground_truth: str
    prediction: str
    category: str = ""
    scores: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class SampleResult:
    """Result for a single benchmark sample (scene/episode)."""

    sample_id: str
    scene_id: str
    question_results: List[QuestionResult] = field(default_factory=list)
    ingestion_time_s: float = 0.0
    n_observations: int = 0


@dataclass
class BenchmarkReport:
    """Full report from a benchmark run."""

    dataset: str
    ablation: str
    sample_results: List[SampleResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def all_scores(self) -> List[Dict[str, Any]]:
        return [qr.scores for sr in self.sample_results for qr in sr.question_results]

    def mean_score(self, key: str = "score") -> float:
        """Compute the mean of a score key across all questions.

        :param key: Score dict key to average (default ``"score"``).
        :returns: Mean value, or 0 if no scores.
        """
        scores = [s[key] for s in self.all_scores if key in s]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def all_question_results(self) -> List["QuestionResult"]:
        return [qr for sr in self.sample_results for qr in sr.question_results]

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary dict suitable for JSON serialization.

        :returns: Dict with dataset, ablation, counts, mean metrics, and
            per-category breakdown.
        """
        all_s = self.all_scores
        keys: set[str] = set()
        for s in all_s:
            keys.update(s.keys())
        means: Dict[str, float] = {}
        for k in sorted(keys):
            vals = [s[k] for s in all_s if k in s and isinstance(s[k], (int, float))]
            if vals:
                means[k] = sum(vals) / len(vals)

        # Per-category breakdown
        per_category: Dict[str, Dict[str, Any]] = {}
        for qr in self.all_question_results:
            cat = qr.category or "unknown"
            if cat not in per_category:
                per_category[cat] = {"n": 0, "scores": [], "tools_failed": 0, "tool_jaccard": []}
            per_category[cat]["n"] += 1
            per_category[cat]["scores"].append(qr.scores.get("score", 0))
            if not qr.tools_used or "none" in qr.tools_used:
                per_category[cat]["tools_failed"] += 1
            if qr.tools_expected:
                expected = set(qr.tools_expected)
                used = set(qr.tools_used)
                union = expected | used
                jaccard = len(expected & used) / len(union) if union else 1.0
                per_category[cat]["tool_jaccard"].append(jaccard)
        cat_summary: Dict[str, Dict[str, Any]] = {}
        for cat, info in sorted(per_category.items()):
            scores = info["scores"]
            tj = info["tool_jaccard"]
            cat_entry: Dict[str, Any] = {
                "n": info["n"],
                "mean_score": round(sum(scores) / len(scores), 2) if scores else 0,
                "n_zero": sum(1 for s in scores if s == 0),
                "n_perfect": sum(1 for s in scores if s == 100),
            }
            if tj:
                cat_entry["tool_accuracy"] = round(sum(tj) / len(tj), 2)
            cat_summary[cat] = cat_entry

        # Aggregate tool selection accuracy across all categories
        all_tj: List[float] = []
        for info in per_category.values():
            all_tj.extend(info["tool_jaccard"])
        metrics = {k: round(v, 2) for k, v in means.items()}
        if all_tj:
            metrics["tool_selection_accuracy"] = round(sum(all_tj) / len(all_tj), 2)

        return {
            "dataset": self.dataset,
            "ablation": self.ablation,
            "n_questions": len(all_s),
            "n_samples": len(self.sample_results),
            "total_time_s": round(self.total_time_s, 1),
            "metrics": metrics,
            "per_category": cat_summary,
        }


_UNANSWERABLE_RE = re.compile(
    r"^("
    r"unanswerable"
    r"|(?:no |not )?"
    r"(?:information|info|record|records|data|details?|mention|evidence|result|results)"
    r"(?:\s+(?:is\s+)?(?:not\s+)?(?:found|available|stored|recorded|specified|mentioned|documented|known|in memory))*"
    r".*"
    r"|not (?:found|available|stored|recorded|specified|mentioned|documented|known)(?: (?:in|from) (?:memory|records?|our records?|stored memories|conversation history?))?\.?"
    r"|(?:no (?:such )?(?:record|information|info|data|details?|mention|evidence|result|results) (?:found|available|in memory).*)"
    r"|(?:insufficient information.*)"
    r"|(?:unknown.*)"
    r"|not in (?:memory|records?|our records?|conversation history)"
    r"|none(?: recorded| specified| found)?"
    r")$",
    re.IGNORECASE,
)


def _clean_answer(answer: str) -> str:
    """Post-process an agent answer to remove leaked thinking and verbosity.

    Strips meta-commentary (e.g. "Wait,", "Thought:", "Let me"), takes only
    the first meaningful line, and maps "not found" / unanswerable responses
    to empty string so they score correctly against empty ground truth.

    :param answer: Raw agent answer.
    :returns: Cleaned answer string.
    """
    # Strip leaked thinking / meta-commentary (match lines with or without
    # trailing newline so the pattern catches the last line too)
    answer = re.sub(
        r"^(?:Thought|Wait|Hmm|Let me|Action|Action Input|Observation|So,|Based on)[:\s].*?(?:\n|$)",
        "",
        answer,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Remove "Final Answer:" prefix if present
    answer = re.sub(r"^Final Answer:\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip()
    # Take first non-empty line only
    for line in answer.split("\n"):
        line = line.strip()
        if line:
            answer = line
            break
    # Map "not found" / unanswerable responses to empty string
    if _UNANSWERABLE_RE.match(answer.strip().rstrip(".")):
        return ""
    return answer


class _AblatedMemory:
    """Wrapper that restricts tool access based on ablation config."""

    def __init__(self, mem: Any, ablation: AblationConfig):
        self._mem = mem
        self._ablation = ablation

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return self._ablation.filter_tool_definitions(
            self._mem.get_tool_definitions()
        )

    def dispatch_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self._ablation.available_tools:
            return f"Tool '{tool_name}' is not available in this configuration."
        return self._mem.dispatch_tool_call(tool_name, arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mem, name)


class BenchmarkRunner:
    """Runs academic benchmark evaluation via trajectory replay."""

    def __init__(
        self,
        loader: Any,
        scorer: Any,
        ablation: AblationConfig,
        embedding_provider: Any,
        llm_client: Any = None,
        agent_factory: Optional[Callable[[Any], Any]] = None,
        max_samples: Optional[int] = None,
        question_template: Optional[str] = None,
        system_preamble: Optional[str] = None,
        mem_config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        :param loader: Dataset loader yielding :class:`BenchmarkSample` instances.
        :param scorer: Scorer implementing the :class:`Scorer` protocol.
        :param ablation: Ablation configuration to apply.
        :param embedding_provider: Embedding provider for eMEM.
        :param llm_client: LLM client for consolidation (optional).
        :param agent_factory: ``fn(mem) -> agent`` with a ``.run(query)`` method.
            Defaults to :class:`ReactAgent` via Ollama.
        :param max_samples: Maximum number of samples to evaluate.
        :param question_template: Template for wrapping questions before passing
            to the agent. Use ``{question}`` as placeholder. If ``None``, the raw
            question is passed directly.
        :param system_preamble: Custom preamble for the agent's system prompt,
            placed before the tool definitions. If ``None``, uses the default.
        :param mem_config_overrides: Dict of field overrides applied to the
            default :class:`SpatioTemporalMemoryConfig` for each sample.
        """
        self._loader = loader
        self._scorer = scorer
        self._ablation = ablation
        self._embedder = embedding_provider
        self._llm = llm_client
        self._agent_factory = agent_factory
        self._max_samples = max_samples
        self._question_template = question_template
        self._system_preamble = system_preamble
        self._mem_config_overrides = mem_config_overrides or {}

    def run(self) -> BenchmarkReport:
        """Run the full benchmark evaluation.

        :returns: Report with per-sample and aggregate results.
        """
        from emem import SpatioTemporalMemory

        report = BenchmarkReport(
            dataset=self._loader.name,
            ablation=self._ablation.name,
        )
        t_start = time.monotonic()
        count = 0

        for sample in self._loader.load():
            if self._max_samples is not None and count >= self._max_samples:
                break

            log.info(
                "[%s] sample=%s scene=%s frames=%d questions=%d",
                self._ablation.name, sample.sample_id, sample.scene_id,
                len(sample.trajectory), len(sample.questions),
            )

            sr = self._run_sample(sample, SpatioTemporalMemory)
            report.sample_results.append(sr)
            count += 1

        report.total_time_s = time.monotonic() - t_start
        log.info(
            "[%s] Done: %d samples, mean_score=%.1f, time=%.1fs",
            self._ablation.name, count,
            report.mean_score(), report.total_time_s,
        )
        return report

    def _run_sample(self, sample: BenchmarkSample, mem_cls: type) -> SampleResult:
        """Ingest a single sample's trajectory, answer its questions, and score.

        :param sample: The benchmark sample to evaluate.
        :param mem_cls: SpatioTemporalMemory class (passed to allow testing).
        :returns: Per-sample result with question scores.
        """
        from emem.config import SpatioTemporalMemoryConfig
        from pathlib import Path

        db_path = tempfile.mktemp(suffix=".db")
        config_kwargs: Dict[str, Any] = {
            "db_path": db_path,
            "hnsw_path": str(Path(db_path).with_suffix(".hnsw.bin")),
        }
        config_kwargs.update(self._mem_config_overrides)
        config = SpatioTemporalMemoryConfig(**config_kwargs)
        replay_time = [0.0]
        mem = mem_cls(
            db_path=db_path,
            config=config,
            embedding_provider=self._embedder,
            llm_client=self._llm,
            get_current_time=lambda: replay_time[0],
        )

        sr = SampleResult(sample_id=sample.sample_id, scene_id=sample.scene_id)

        try:
            t_ingest = time.monotonic()
            mem.start_episode(sample.scene_id)

            for frame in sample.trajectory:
                is_interoception = frame.layer_name.startswith("interoception:")
                layer = frame.layer_name
                if not self._ablation.use_multi_layer and not is_interoception:
                    layer = "default"
                if is_interoception:
                    mem.add_body_state(
                        text=frame.text,
                        layer_name=layer,
                        timestamp=frame.timestamp,
                    )
                else:
                    mem.add(
                        text=frame.text,
                        x=frame.position[0],
                        y=frame.position[1],
                        z=frame.position[2],
                        timestamp=frame.timestamp,
                        layer_name=layer,
                    )
                replay_time[0] = max(replay_time[0], frame.timestamp)
                sr.n_observations += 1

            n_intero = sum(1 for f in sample.trajectory
                          if f.layer_name.startswith("interoception:"))
            log.info(
                "  ingested %d frames (%d interoception)",
                sr.n_observations, n_intero,
            )

            mem.end_episode(consolidate=self._ablation.use_consolidation)
            sr.ingestion_time_s = time.monotonic() - t_ingest

            if sample.agent_position is not None:
                mem.add(
                    text=f"Agent is currently here. {sample.agent_situation}",
                    x=sample.agent_position[0],
                    y=sample.agent_position[1],
                    z=sample.agent_position[2],
                    layer_name="agent_position",
                )

            ablated_mem = _AblatedMemory(mem, self._ablation)
            agent = self._make_agent(ablated_mem)

            for bq in sample.questions:
                qr = self._run_question(agent, bq)
                sr.question_results.append(qr)
                tools_str = ",".join(qr.tools_used) if qr.tools_used else "none"
                gt_short = qr.ground_truth[:40] if qr.ground_truth else "(empty)"
                log.info(
                    "  q=%s cat=%s score=%.0f tools=[%s] gt=%s pred=%s",
                    bq.question_id, bq.category or "?",
                    qr.scores.get("score", -1),
                    tools_str, gt_short,
                    qr.prediction[:80],
                )

        finally:
            mem.close()

        return sr

    def _run_question(self, agent: Any, bq: BenchmarkQuestion) -> QuestionResult:
        """Run the agent on a single question and score its answer.

        :param agent: Agent instance with a ``.run(query)`` method.
        :param bq: The benchmark question.
        :returns: Scored question result.
        """
        query = bq.question
        if self._question_template:
            query = self._question_template.format(question=query)

        t0 = time.monotonic()
        result = agent.run(query)
        latency = time.monotonic() - t0

        answer = _clean_answer(result.answer)

        if hasattr(self._scorer, 'score_with_category'):
            scores = self._scorer.score_with_category(bq.question, answer, bq.answer, bq.category)
        else:
            scores = self._scorer.score(bq.question, answer, bq.answer)

        return QuestionResult(
            question_id=bq.question_id,
            question=bq.question,
            ground_truth=bq.answer,
            prediction=answer,
            category=bq.category,
            scores=scores,
            tools_used=result.tools_used,
            tools_expected=bq.tools_expected,
            latency_s=latency,
        )

    def _make_agent(self, mem: Any) -> Any:
        """Create an agent for the given memory instance.

        :param mem: Memory instance (possibly ablated).
        :returns: Agent with a ``.run(query)`` method.
        """
        if self._agent_factory is not None:
            return self._agent_factory(mem)
        from harness.agent.react_agent import ReactAgent
        return ReactAgent(mem, system_prompt=self._build_system_prompt(mem))

    def _build_system_prompt(self, mem: Any) -> str | None:
        """Build a system prompt with a custom preamble, if set.

        :param mem: Memory instance for tool definitions.
        :returns: Custom system prompt, or ``None`` to use the default.
        """
        if self._system_preamble is None:
            return None
        from harness.agent.prompts import build_system_prompt
        return build_system_prompt(
            mem.get_tool_definitions(), preamble=self._system_preamble,
        )
