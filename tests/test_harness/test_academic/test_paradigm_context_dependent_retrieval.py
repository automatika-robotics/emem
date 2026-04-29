"""Unit tests for the context-dependent retrieval paradigm (A14c.2).

Synthetic ProcTHOR-shaped house dict + mocked LLM picker. Covers:

  * Generator emits one (item, match) and one (item, mismatch)
    candidate per chosen item.
  * Hallucinated picks (LLM proposes items absent from the unique
    pool) are dropped.
  * Each candidate carries the correct ``context``,
    ``agent_position``, and ``answer`` (yes/no).
  * Scenes with only one eligible room can't form mismatch probes
    and are skipped.
  * Pre-filter rubric mentions yes / ambiguous / no.
  * The schedule_builder helper emits a 3-phase schedule
    (ingest scene → ingest agent_position → probe).
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.context_dependent_retrieval import (
    CONTEXT_MATCH,
    CONTEXT_MISMATCH,
    ContextDependentRetrievalGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_context_dependent_schedule,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    IngestPhase,
    ProbePhase,
)


def _unit_square(x0: float, z0: float, side: float = 4.0) -> List[Dict[str, float]]:
    return [
        {"x": x0, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0 + side},
        {"x": x0, "y": 0.0, "z": z0 + side},
    ]


def _fake_house() -> Dict[str, Any]:
    """3-room house with one HousePlant shared and the rest unique."""
    return {
        "rooms": [
            {
                "id": "room|0",
                "roomType": "Kitchen",
                "floorPolygon": _unit_square(0.0, 0.0),
            },
            {
                "id": "room|1",
                "roomType": "Bedroom",
                "floorPolygon": _unit_square(5.0, 0.0),
            },
            {
                "id": "room|2",
                "roomType": "Bathroom",
                "floorPolygon": _unit_square(10.0, 0.0),
            },
        ],
        "objects": [
            {"id": "Fridge|0|0", "position": {"x": 1.0, "z": 1.0}},
            {"id": "Microwave|0|1", "position": {"x": 2.0, "z": 2.0}},
            {"id": "HousePlant|0|2", "position": {"x": 3.0, "z": 3.0}},
            {"id": "Bed|1|0", "position": {"x": 6.0, "z": 2.0}},
            {"id": "HousePlant|1|1", "position": {"x": 7.0, "z": 2.0}},
            {"id": "Lamp|1|2", "position": {"x": 8.0, "z": 2.0}},
            {"id": "Toilet|2|0", "position": {"x": 11.0, "z": 2.0}},
            {"id": "Sink|2|1", "position": {"x": 12.0, "z": 2.0}},
            {"id": "Mirror|2|2", "position": {"x": 13.0, "z": 2.0}},
        ],
    }


def _scene(sample_id: str = "house_0") -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "procthor_dataset_index": 42,
        "trajectory": [
            {
                "frame_id": "f0",
                "position": [1.0, 0.0, 1.0],
                "timestamp": 100.0,
                "room_id": "room|0",
                "room_type": "Kitchen",
                "layers": {"vlm": "kitchen view", "object_detection": "fridge"},
            },
            {
                "frame_id": "f1",
                "position": [6.0, 0.0, 2.0],
                "timestamp": 200.0,
                "room_id": "room|1",
                "room_type": "Bedroom",
                "layers": {"vlm": "bedroom view", "object_detection": "bed, lamp"},
            },
        ],
        "interoception": [],
        "scene_objects": [],
    }


class TestContextDependentRetrievalGenerator:
    def _llm_stub(self, picks: List[str]):
        import json as _json

        def chat(prompt: str) -> str:
            return _json.dumps(picks)

        return chat

    def test_emits_match_and_mismatch_per_item(self):
        llm = self._llm_stub(["Lamp"])
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=8,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        assert len(cands) == 2
        contexts = sorted(c.paradigm_metadata["context"] for c in cands)
        assert contexts == [CONTEXT_MATCH, CONTEXT_MISMATCH]
        # Match candidate has answer "yes", mismatch has "no".
        by_ctx = {c.paradigm_metadata["context"]: c for c in cands}
        assert by_ctx[CONTEXT_MATCH].answer == "yes"
        assert by_ctx[CONTEXT_MISMATCH].answer == "no"
        # Both share the same probe object.
        assert all(c.paradigm_metadata["probe_object"] == "lamp" for c in cands)

    def test_hallucinated_picks_are_dropped(self):
        llm = self._llm_stub(["NotARealItem", "Lamp"])
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=8,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "lamp" in probes
        assert "notarealitem" not in probes
        assert len(cands) == 2  # 1 valid item × 2 contexts

    def test_shared_house_plant_is_not_picked(self):
        llm = self._llm_stub(["HousePlant", "Lamp"])
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=8,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "house plant" not in probes
        assert "lamp" in probes

    def test_question_format(self):
        llm = self._llm_stub(["Lamp"])
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=8,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        for c in cands:
            assert c.question == "Is there a lamp near your current location?"

    def test_match_position_in_item_room(self):
        """Match probe's agent_position should fall in the item's room polygon."""
        llm = self._llm_stub(["Lamp"])
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=8,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        match = next(
            c for c in cands if c.paradigm_metadata["context"] == CONTEXT_MATCH
        )
        # Bedroom polygon is x in [5, 9], z in [0, 4]; centroid at (7, 2).
        x, _y, z = match.paradigm_metadata["agent_position"]
        assert 5.0 <= x <= 9.0
        assert 0.0 <= z <= 4.0

    def test_skips_scene_with_one_eligible_room(self):
        """Single-room house has no mismatch room → no candidates."""
        house = {
            "rooms": [
                {
                    "id": "room|0",
                    "roomType": "Kitchen",
                    "floorPolygon": _unit_square(0.0, 0.0),
                }
            ],
            "objects": [
                {"id": "Lamp|0|0", "position": {"x": 1.0, "z": 1.0}},
                {"id": "Fridge|0|1", "position": {"x": 2.0, "z": 2.0}},
            ],
        }
        gen = ContextDependentRetrievalGenerator(
            llm_chat=self._llm_stub(["Lamp"]),
            house_lookup=lambda _i: house,
            target_total=8,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        assert cands == []

    def test_exclude_probes_suppresses_duplicates(self):
        llm = self._llm_stub(["Fridge", "Lamp"])
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=8,
            exclude_probes={"__ctxdep__": {"fridge"}},
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "fridge" not in probes
        assert "lamp" in probes

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = ContextDependentRetrievalGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestContextDependentScheduleBuilder:
    def test_three_phase_schedule_with_agent_position(self):
        """Schedule = IngestPhase(scene) → IngestPhase(pos) → ProbePhase."""
        llm = lambda _p: '["Lamp"]'  # noqa: E731
        gen = ContextDependentRetrievalGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=2,
        )
        cands = gen.generate([_scene()], n_per_scene=2)
        match = next(
            c for c in cands if c.paradigm_metadata["context"] == CONTEXT_MATCH
        )
        sched = build_context_dependent_schedule(match, _scene())
        assert len(sched.phases) == 3
        assert isinstance(sched.phases[0], IngestPhase)
        assert isinstance(sched.phases[1], IngestPhase)
        assert isinstance(sched.phases[2], ProbePhase)
        # The agent_position phase carries exactly one observation
        # tagged with the agent_position layer.
        agent_phase = sched.phases[1]
        assert len(agent_phase.observations) == 1
        agent_obs = agent_phase.observations[0]
        assert agent_obs.layer_name == "agent_position"
        # Position matches the candidate metadata.
        expected_pos = match.paradigm_metadata["agent_position"]
        assert list(agent_obs.position) == [
            float(expected_pos[0]),
            float(expected_pos[1]),
            float(expected_pos[2]),
        ]

    def test_missing_agent_position_raises(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        bad = CandidateQuestion(
            question_id="x",
            question="?",
            answer="yes",
            category="context_dependent_retrieval",
            paradigm="context_dependent_retrieval",
            scene_ids=["house_0"],
            paradigm_metadata={},  # missing agent_position
        )
        try:
            build_context_dependent_schedule(bad, _scene())
        except ValueError as exc:
            assert "agent_position" in str(exc)
        else:
            raise AssertionError("expected ValueError")
