"""Unit tests for the eMEM-Bench v1 ScheduleRunner (A14a).

Exercises the runner against an in-memory fake so we can assert:
- the virtual clock advances correctly across phases
- maintenance fires exactly in AdvanceClockPhases that request it
  (not when run_maintenance=False, not during IngestPhase)
- probes are dispatched at the right clock value and scored against
  their own query sets
- ProbePhase with at_time < current clock raises
- unknown phase types raise TypeError
- deferred scoring populates QuestionResult.scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harness.benchmarks.academic.ablation import ABLATIONS
from harness.benchmarks.academic.emem_bench_v1 import (
    AdvanceClockPhase,
    IngestPhase,
    Observation,
    ProbePhase,
    Schedule,
)
from harness.benchmarks.academic.emem_bench_v1.runner import ScheduleRunner
from harness.benchmarks.academic.trajectory import BenchmarkQuestion


@dataclass
class _CallLog:
    """Everything the fake memory did, in order."""

    events: List[tuple] = field(default_factory=list)


class _FakeMemory:
    """Minimal memory stub satisfying ScheduleRunner's requirements.

    Records a time-getter at construction so tests can assert the
    runner's virtual clock is visible to the memory when
    maintenance / consolidation fire.
    """

    def __init__(self, call_log: _CallLog, time_getter: Any = None):
        self._log = call_log
        self._time_getter = time_getter or (lambda: 0.0)
        self._active_episode: Optional[str] = None
        self.observations: List[tuple] = []
        self.closed = False

    def start_episode(self, name: str) -> str:
        self._active_episode = name
        self._log.events.append(("start_episode", name))
        return name

    def end_episode(self, consolidate: bool = True) -> None:
        self._log.events.append(("end_episode", self._active_episode, consolidate))
        self._active_episode = None

    def add(
        self,
        text: str,
        x: float = 0.0,
        y: float = 0.0,
        timestamp: float = 0.0,
        layer_name: str = "default",
    ) -> None:
        self.observations.append((text, x, y, timestamp, layer_name))
        self._log.events.append(("add", text, timestamp, layer_name))

    def add_body_state(
        self, text: str, timestamp: float = 0.0, layer_name: str = "interoception"
    ) -> None:
        self._log.events.append(("add_body_state", text, timestamp, layer_name))

    def consolidate_time_window(self) -> list:
        self._log.events.append(("consolidate", self._time_getter()))
        return []

    def maintenance(self) -> int:
        self._log.events.append(("archive", self._time_getter()))
        return 0

    def close(self) -> None:
        self.closed = True
        self._log.events.append(("close",))

    def get_tool_definitions(self) -> List[dict]:
        return [{"type": "function", "function": {"name": "semantic_search"}}]

    def dispatch_tool_call(self, name: str, args: dict) -> str:
        return f"called {name}"


@dataclass
class _FakeAgentResult:
    answer: str
    tools_used: List[str] = field(default_factory=list)


class _FakeAgent:
    """Agent that echoes a canned answer per question id."""

    def __init__(self, mem: Any, answer_map: Dict[str, str]):
        self._mem = mem
        self._answer_map = answer_map
        self.calls: List[str] = []

    def run(self, query: str) -> _FakeAgentResult:
        self.calls.append(query)
        # Match on substring — tests build queries that contain the qid.
        for qid, answer in self._answer_map.items():
            if qid in query:
                return _FakeAgentResult(answer=answer, tools_used=["semantic_search"])
        return _FakeAgentResult(answer="")


class _ExactScorer:
    """Score 100 for exact match, 0 otherwise."""

    def score(
        self, question: str, prediction: str, ground_truth: str
    ) -> Dict[str, float]:
        return {"score": 100.0 if prediction.strip() == ground_truth.strip() else 0.0}


def _obs(text: str, ts: float, *, layer: str = "description") -> Observation:
    return Observation(
        text=text, position=(0.0, 0.0, 0.0), timestamp=ts, layer_name=layer
    )


def _q(qid: str, answer: str, *, category: str = "") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=qid,
        question=f"probe {qid}",
        answer=answer,
        category=category,
    )


def _factories(log: _CallLog, answers: Dict[str, str]):
    mem_holder: List[_FakeMemory] = []

    def mem_factory(time_getter: Any) -> _FakeMemory:
        mem = _FakeMemory(log, time_getter=time_getter)
        mem_holder.append(mem)
        return mem

    agent_holder: List[_FakeAgent] = []

    def agent_factory(wrapped: Any) -> _FakeAgent:
        agent = _FakeAgent(wrapped, answers)
        agent_holder.append(agent)
        return agent

    return mem_factory, agent_factory, mem_holder, agent_holder


class TestScheduleRunner:
    def test_three_phase_example_end_to_end(self):
        """Encode → advance 3600s with maintenance → probe."""
        log = _CallLog()
        schedule = Schedule(
            sample_id="sample_0",
            scene_id="scene_0",
            start_time=0.0,
            phases=[
                IngestPhase(
                    episode_name="ep0",
                    observations=[
                        _obs("I see a red door", ts=1.0),
                        _obs("I see a chair", ts=2.0),
                    ],
                ),
                AdvanceClockPhase(delta_seconds=3600.0, run_maintenance=True),
                ProbePhase(
                    at_time=3602.0,
                    probe_id="retention_1h",
                    query_set=[_q("q1", "red door", category="single")],
                ),
            ],
        )

        mem_factory, agent_factory, mem_holder, _ = _factories(log, {"q1": "red door"})
        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        report = runner.run()

        # One sample, one question, scored correctly
        assert len(report.sample_results) == 1
        sr = report.sample_results[0]
        assert sr.n_observations == 2
        assert len(sr.question_results) == 1
        qr = sr.question_results[0]
        assert qr.prediction == "red door"
        assert qr.scores["score"] == 100.0

        # Event order: episode open, two adds, close (no consolidate),
        # consolidate + archive at 3602, no post-probe consolidate, close.
        event_names = [e[0] for e in log.events]
        assert event_names == [
            "start_episode",
            "add",
            "add",
            "end_episode",
            "consolidate",
            "archive",
            "close",
        ]
        # Maintenance fires at end of the advance phase — clock=3602.
        consolidate_event = next(e for e in log.events if e[0] == "consolidate")
        assert consolidate_event[1] == 3602.0
        # end_episode was asked NOT to consolidate (False in tuple).
        end_ep = next(e for e in log.events if e[0] == "end_episode")
        assert end_ep[2] is False
        # Memory was closed.
        assert mem_holder[0].closed is True

    def test_maintenance_does_not_fire_when_flag_off(self):
        log = _CallLog()
        schedule = Schedule(
            sample_id="no_maint",
            phases=[
                IngestPhase(observations=[_obs("x", ts=1.0)]),
                AdvanceClockPhase(delta_seconds=60.0, run_maintenance=False),
                AdvanceClockPhase(delta_seconds=60.0, run_maintenance=True),
            ],
        )
        mem_factory, agent_factory, _, _ = _factories(log, {})
        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        runner.run()
        # Exactly one consolidate + one archive — from the second advance.
        assert sum(1 for e in log.events if e[0] == "consolidate") == 1
        assert sum(1 for e in log.events if e[0] == "archive") == 1

    def test_multiple_probes_with_different_query_sets(self):
        """Two probes at different clock points, different qs."""
        log = _CallLog()
        schedule = Schedule(
            sample_id="multi_probe",
            phases=[
                IngestPhase(
                    observations=[
                        _obs("fact A", ts=1.0),
                        _obs("fact B", ts=2.0),
                    ]
                ),
                ProbePhase(
                    at_time=10.0,
                    probe_id="early",
                    query_set=[_q("qa", "fact A")],
                ),
                AdvanceClockPhase(delta_seconds=100.0, run_maintenance=True),
                ProbePhase(
                    at_time=110.0,
                    probe_id="late",
                    query_set=[_q("qb", "fact B"), _q("qc", "fact Z")],
                ),
            ],
        )
        mem_factory, agent_factory, _, _ = _factories(
            log, {"qa": "fact A", "qb": "fact B", "qc": "wrong"}
        )
        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        report = runner.run()

        results = report.sample_results[0].question_results
        assert [qr.question_id for qr in results] == ["qa", "qb", "qc"]
        assert [qr.scores["score"] for qr in results] == [100.0, 100.0, 0.0]
        # Probe categories fell back to probe_id since the questions had
        # no explicit category.
        assert [qr.category for qr in results] == ["early", "late", "late"]

    def test_probe_before_clock_raises(self):
        log = _CallLog()
        schedule = Schedule(
            sample_id="bad_probe",
            phases=[
                AdvanceClockPhase(delta_seconds=100.0),
                ProbePhase(at_time=50.0, query_set=[_q("q1", "x")]),
            ],
        )
        mem_factory, agent_factory, _, _ = _factories(log, {"q1": "x"})
        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        try:
            runner.run()
        except ValueError as e:
            assert "precedes clock" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_ablation_restricts_tool_definitions(self):
        """An ablation that allows only one tool should pass through
        as the wrapper's effective allow-list."""
        log = _CallLog()
        schedule = Schedule(
            sample_id="abl",
            phases=[ProbePhase(at_time=0.0, query_set=[_q("q1", "x")])],
        )

        captured_tools: List[list] = []

        def agent_factory(wrapped: Any):
            captured_tools.append(wrapped.get_tool_definitions())
            return _FakeAgent(wrapped, {"q1": "x"})

        def mem_factory(time_getter: Any) -> _FakeMemory:
            return _FakeMemory(log, time_getter=time_getter)

        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["vector_only"],  # semantic_search only
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        runner.run()
        assert len(captured_tools) == 1
        names = [t["function"]["name"] for t in captured_tools[0]]
        assert names == ["semantic_search"]

    def test_unknown_phase_type_raises(self):
        log = _CallLog()

        class _BogusPhase:
            pass

        schedule = Schedule(sample_id="weird", phases=[_BogusPhase()])  # type: ignore[list-item]
        mem_factory, agent_factory, _, _ = _factories(log, {})
        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        try:
            runner.run()
        except TypeError as e:
            assert "Unknown phase type" in str(e)
        else:
            raise AssertionError("expected TypeError")

    def test_interoception_routes_to_add_body_state(self):
        log = _CallLog()
        schedule = Schedule(
            sample_id="intero",
            phases=[
                IngestPhase(
                    observations=[
                        Observation(
                            text="battery: 85%",
                            timestamp=1.0,
                            layer_name="battery",
                            is_interoception=True,
                        ),
                        _obs("I see a chair", ts=2.0),
                    ]
                ),
            ],
        )
        mem_factory, agent_factory, _, _ = _factories(log, {})
        runner = ScheduleRunner(
            schedules=[schedule],
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
        )
        runner.run()
        kinds = [e[0] for e in log.events if e[0] in ("add", "add_body_state")]
        assert kinds == ["add_body_state", "add"]

    def test_max_samples_limits_output(self):
        log = _CallLog()
        schedules = [
            Schedule(
                sample_id=f"s{i}",
                phases=[IngestPhase(observations=[_obs("x", ts=1.0)])],
            )
            for i in range(5)
        ]
        mem_factory, agent_factory, _, _ = _factories(log, {})
        runner = ScheduleRunner(
            schedules=schedules,
            scorer=_ExactScorer(),
            ablation=ABLATIONS["full"],
            memory_factory=mem_factory,
            agent_factory=agent_factory,
            max_samples=2,
        )
        report = runner.run()
        assert len(report.sample_results) == 2
