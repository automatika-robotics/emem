"""ScheduleRunner — executes eMEM-Bench v1 :class:`Schedule` objects.

Walks the ordered phases of a schedule, driving a fresh memory
instance per sample. Ingestion phases open + close episodes without
consolidation; advance-clock phases optionally fire
``consolidate_time_window`` + ``archive_long_term`` at the new
clock value; probe phases dispatch questions through an agent and
collect predictions. Scoring is deferred to a second pass so the
judge LLM loads at most once per benchmark, mirroring the
``replay_runner.BenchmarkRunner`` pattern.

Reuses :class:`QuestionResult`, :class:`SampleResult`,
:class:`BenchmarkReport`, :class:`_AblatedMemory`, and
:func:`clean_answer` from the LoCoMo path — v1 is a sibling runner,
not a fork of the LoCoMo runner.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, List, Optional

from harness.benchmarks.academic.ablation import AblationConfig
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    AdvanceClockPhase,
    IngestPhase,
    Observation,
    ProbePhase,
    Schedule,
)
from harness.benchmarks.academic.replay_runner import (
    BenchmarkReport,
    QuestionResult,
    SampleResult,
    _AblatedMemory,
)
from harness.benchmarks.academic.trajectory import BenchmarkQuestion
from harness.postprocess import clean_answer

log = logging.getLogger("harness.academic.v1")


class ScheduleRunner:
    """Drive a stream of :class:`Schedule` samples end-to-end.

    The runner mirrors :class:`BenchmarkRunner` at the seams — same
    ablation plumbing, same dataset-tool filter, same deferred
    scoring — but the per-sample body iterates phases instead of
    treating the whole sample as one ingest-then-query pass.

    :param schedules: Iterable of Schedule objects to execute.
    :param scorer: Scorer with ``.score(question, prediction, gt)``
        (or ``.score_with_category(..., category)`` if available).
    :param ablation: AblationConfig (tool restriction + memory config
        overrides).
    :param memory_factory: ``fn(time_getter) -> mem`` producing a fresh
        memory instance per sample. The runner passes a ``time_getter``
        closure that reflects the current virtual clock; the factory is
        expected to wire it into the memory as ``get_current_time`` so
        that ``consolidate_time_window()`` and ``maintenance()`` see the
        runner's clock rather than wall time. The returned object must
        support ``.add(text, x, y, timestamp, layer_name)``,
        ``.start_episode(name)``, ``.end_episode(consolidate)``,
        ``.consolidate_time_window()``, and ``.maintenance()``.
    :param agent_factory: ``fn(mem_wrapper) -> agent`` where ``agent``
        exposes ``.run(query) -> AgentResult``.
    :param dataset_tool_filter: Optional extra allow-list of tool
        names on top of the ablation's filter.
    :param max_samples: Cap on samples evaluated.
    :param dataset_name: Dataset label used in the emitted
        :class:`BenchmarkReport`; defaults to ``"emem-bench-v1"``.
    """

    def __init__(
        self,
        schedules: Iterable[Schedule],
        scorer: Any,
        ablation: AblationConfig,
        memory_factory: Callable[[Callable[[], float]], Any],
        agent_factory: Callable[[Any], Any],
        dataset_tool_filter: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        dataset_name: str = "emem-bench-v1",
    ):
        self._schedules = schedules
        self._scorer = scorer
        self._ablation = ablation
        self._memory_factory = memory_factory
        self._agent_factory = agent_factory
        self._dataset_tool_filter = dataset_tool_filter
        self._max_samples = max_samples
        self._dataset_name = dataset_name

    def run(self) -> BenchmarkReport:
        """Run every schedule and return a scored report.

        :returns: BenchmarkReport with one SampleResult per schedule.
        """
        report = BenchmarkReport(
            dataset=self._dataset_name, ablation=self._ablation.name
        )
        t_start = time.monotonic()
        count = 0
        for schedule in self._schedules:
            if self._max_samples is not None and count >= self._max_samples:
                break
            log.info(
                "[%s/%s] starting schedule sample=%s phases=%d",
                self._dataset_name,
                self._ablation.name,
                schedule.sample_id,
                len(schedule.phases),
            )
            sr = self._run_schedule(schedule)
            report.sample_results.append(sr)
            count += 1

        self._score_samples(report.sample_results)
        report.total_time_s = time.monotonic() - t_start
        log.info(
            "[%s/%s] done: %d samples, mean_score=%.1f, time=%.1fs",
            self._dataset_name,
            self._ablation.name,
            len(report.sample_results),
            report.mean_score(),
            report.total_time_s,
        )
        return report

    def _run_schedule(self, schedule: Schedule) -> SampleResult:
        """Execute a single schedule end-to-end."""
        sr = SampleResult(
            sample_id=schedule.sample_id,
            scene_id=schedule.scene_id or schedule.sample_id,
        )
        t_sample_start = time.monotonic()

        # Mutable clock holder: the memory queries this via its
        # ``get_current_time`` hook, so any call to the memory's
        # consolidation / maintenance methods picks up the runner's
        # virtual clock automatically.
        clock_holder = [schedule.start_time]
        mem = self._memory_factory(lambda: clock_holder[0])
        ablated = _AblatedMemory(
            mem,
            self._ablation,
            dataset_tool_filter=self._dataset_tool_filter,
        )
        agent = self._agent_factory(ablated)

        n_ingested = 0
        t_ingest = 0.0

        try:
            for phase_idx, phase in enumerate(schedule.phases):
                if isinstance(phase, IngestPhase):
                    t0 = time.monotonic()
                    n_added = self._ingest(mem, phase, clock_holder)
                    n_ingested += n_added
                    t_ingest += time.monotonic() - t0
                elif isinstance(phase, AdvanceClockPhase):
                    self._advance(mem, phase, clock_holder)
                elif isinstance(phase, ProbePhase):
                    self._probe(agent, phase, clock_holder, sr, phase_idx)
                else:
                    raise TypeError(
                        f"Unknown phase type at index {phase_idx}: {type(phase).__name__}"
                    )
        finally:
            closer = getattr(mem, "close", None)
            if callable(closer):
                closer()

        sr.ingestion_time_s = t_ingest
        sr.n_observations = n_ingested
        log.info(
            "  sample %s: %d observations, %d probes, ingest=%.1fs, total=%.1fs",
            schedule.sample_id,
            n_ingested,
            len(sr.question_results),
            t_ingest,
            time.monotonic() - t_sample_start,
        )
        return sr

    def _ingest(self, mem: Any, phase: IngestPhase, clock_holder: List[float]) -> int:
        """Run an IngestPhase; return the number of observations added.

        Opens a named episode, adds every observation at its own
        timestamp, then closes the episode with ``consolidate=False``
        so consolidation only fires from AdvanceClockPhase. Advances
        ``clock_holder[0]`` to the largest observation timestamp
        seen, so the memory's injected time-getter reflects where
        ingestion left off.
        """
        mem.start_episode(phase.episode_name)
        max_ts = clock_holder[0]
        for obs in phase.observations:
            self._add_observation(mem, obs)
            max_ts = max(max_ts, obs.timestamp)
        mem.end_episode(consolidate=False)
        clock_holder[0] = max_ts
        return len(phase.observations)

    @staticmethod
    def _add_observation(mem: Any, obs: Observation) -> None:
        """Dispatch a single Observation into memory.

        Body-state observations go to :meth:`add_body_state` so
        interoception ends up on the expected code path; normal
        observations go to :meth:`add`.
        """
        if obs.is_interoception:
            mem.add_body_state(
                obs.text,
                timestamp=obs.timestamp,
                layer_name=obs.layer_name,
            )
            return
        x, y, _ = obs.position
        mem.add(
            obs.text,
            x=x,
            y=y,
            timestamp=obs.timestamp,
            layer_name=obs.layer_name,
        )

    @staticmethod
    def _advance(mem: Any, phase: AdvanceClockPhase, clock_holder: List[float]) -> None:
        """Advance the virtual clock, firing consolidation + archival if asked.

        Both ``consolidate_time_window()`` and ``maintenance()`` consult
        the memory's injected time-getter, which tracks
        ``clock_holder[0]`` — so bumping the holder in place before the
        calls is how the runner tells memory "it is now later".
        """
        clock_holder[0] += phase.delta_seconds
        if phase.run_maintenance:
            mem.consolidate_time_window()
            mem.maintenance()

    def _probe(
        self,
        agent: Any,
        phase: ProbePhase,
        clock_holder: List[float],
        sr: SampleResult,
        phase_idx: int,
    ) -> None:
        """Run a ProbePhase's query set and append the predictions to ``sr``.

        Updates ``clock_holder[0]`` to ``phase.at_time`` before the agent
        runs so any tool call that consults the memory's clock sees the
        right wall time.
        """
        if phase.at_time < clock_holder[0]:
            raise ValueError(
                f"ProbePhase.at_time={phase.at_time} precedes clock={clock_holder[0]}"
            )
        clock_holder[0] = phase.at_time
        for bq in phase.query_set:
            qr = self._predict_question(agent, bq, phase.probe_id)
            sr.question_results.append(qr)
            tools_str = ",".join(qr.tools_used) if qr.tools_used else "none"
            gt_short = qr.ground_truth[:40] if qr.ground_truth else "(empty)"
            log.info(
                "  phase %d probe=%s q=%s cat=%s tools=[%s] gt=%s pred=%s",
                phase_idx,
                phase.probe_id or "-",
                bq.question_id,
                bq.category or "?",
                tools_str,
                gt_short,
                qr.prediction[:80],
            )

    def _predict_question(
        self, agent: Any, bq: BenchmarkQuestion, probe_id: str
    ) -> QuestionResult:
        """Run the agent on one question, return an unscored QuestionResult."""
        query = bq.question
        t0 = time.monotonic()
        try:
            result = agent.run(query)
            latency = time.monotonic() - t0
            answer = clean_answer(result.answer)
            tools_used = result.tools_used
        except RuntimeError as exc:
            latency = time.monotonic() - t0
            log.warning(
                "[%s] agent.run failed on q=%s: %s",
                self._ablation.name,
                bq.question_id,
                exc,
            )
            answer = ""
            tools_used = []

        return QuestionResult(
            question_id=bq.question_id,
            question=bq.question,
            ground_truth=bq.answer,
            prediction=answer,
            category=bq.category or probe_id or "",
            scores={},
            tools_used=tools_used,
            tools_expected=bq.tools_expected,
            latency_s=latency,
        )

    def _score_samples(self, sample_results: List[SampleResult]) -> None:
        """Score every prediction collected across all samples in one pass."""
        use_category = hasattr(self._scorer, "score_with_category")
        n_scored = 0
        t0 = time.monotonic()
        for sr in sample_results:
            for qr in sr.question_results:
                if use_category:
                    qr.scores = self._scorer.score_with_category(
                        qr.question, qr.prediction, qr.ground_truth, qr.category
                    )
                else:
                    qr.scores = self._scorer.score(
                        qr.question, qr.prediction, qr.ground_truth
                    )
                n_scored += 1
        log.info(
            "[%s] scored %d question results in %.1fs",
            self._ablation.name,
            n_scored,
            time.monotonic() - t0,
        )


__all__ = ["ScheduleRunner"]
