"""Unit tests for the prospective-memory paradigm (A14c.3).

Synthetic ProcTHOR-shaped house dict; no LLM picker (PM generation
is deterministic). Covers:

  * Generator picks unique-to-one-room objects and emits one
    candidate per item with a deterministic rotated action.
  * Each candidate carries instruction_text, instructed_action,
    trigger_object, and ground_truth_objects_in_room.
  * Question and answer text are well-formed.
  * Pre-filter rubric mentions yes / ambiguous / no.
  * Schedule_builder helper emits a 4-phase schedule:
    instruction → AdvanceClock → trajectory → probe.
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.prospective_memory import (
    ACTIONS,
    ProspectiveMemoryGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_prospective_memory_schedule,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    AdvanceClockPhase,
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


class TestProspectiveMemoryGenerator:
    def test_emits_one_candidate_per_item_with_rotated_action(self):
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
        )
        cands = gen.generate([_scene()], n_per_scene=5)
        assert len(cands) == 5
        # All triggers should be unique (no duplicate items)
        triggers = [c.paradigm_metadata["trigger_object"] for c in cands]
        assert len(set(triggers)) == 5
        # Action sequence rotates through ACTIONS deterministically.
        actions = [c.paradigm_metadata["instructed_action"] for c in cands]
        assert actions[0] == ACTIONS[0]
        assert actions[1] == ACTIONS[1]
        assert actions[-1] == ACTIONS[(len(actions) - 1) % len(ACTIONS)]

    def test_shared_house_plant_is_not_picked(self):
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
        )
        cands = gen.generate([_scene()], n_per_scene=10)
        triggers = {c.paradigm_metadata["trigger_object"] for c in cands}
        assert "house plant" not in triggers

    def test_question_and_answer(self):
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
        )
        cands = gen.generate([_scene()], n_per_scene=1)
        assert len(cands) == 1
        c = cands[0]
        trigger = c.paradigm_metadata["trigger_object"]
        action = c.paradigm_metadata["instructed_action"]
        assert (
            c.question == f"What were you instructed to do when you see the {trigger}?"
        )
        assert c.answer == action
        assert c.paradigm_metadata["instruction_text"] == (
            f"Instruction: when you see the {trigger}, {action}."
        )

    def test_metadata_carries_ground_truth(self):
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
        )
        cands = gen.generate([_scene()], n_per_scene=3)
        for c in cands:
            md = c.paradigm_metadata
            assert "room_type" in md
            assert "ground_truth_objects_in_room" in md
            assert md["trigger_object"] in {
                t.lower() for t in md["ground_truth_objects_in_room"]
            } | {t.lower().replace(" ", "") for t in md["ground_truth_objects_in_room"]}

    def test_skips_scene_with_no_unique_objects(self):
        house = {
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
            ],
            "objects": [
                {"id": "Lamp|0|0", "position": {"x": 1.0, "z": 1.0}},
                {"id": "Lamp|1|0", "position": {"x": 6.0, "z": 1.0}},
            ],
        }
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: house,
            target_total=12,
        )
        cands = gen.generate([_scene()], n_per_scene=3)
        assert cands == []

    def test_exclude_probes_suppresses_duplicates(self):
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
            exclude_probes={"__prosmem__": {"fridge"}},
        )
        cands = gen.generate([_scene()], n_per_scene=10)
        triggers = {c.paradigm_metadata["trigger_object"] for c in cands}
        assert "fridge" not in triggers

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = ProspectiveMemoryGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestProspectiveMemoryScheduleBuilder:
    def test_four_phase_schedule(self):
        gen = ProspectiveMemoryGenerator(
            house_lookup=lambda _i: _fake_house(),
            target_total=1,
        )
        cand = gen.generate([_scene()], n_per_scene=1)[0]
        sched = build_prospective_memory_schedule(cand, _scene())
        assert len(sched.phases) == 4
        assert isinstance(sched.phases[0], IngestPhase)
        assert isinstance(sched.phases[1], AdvanceClockPhase)
        assert isinstance(sched.phases[2], IngestPhase)
        assert isinstance(sched.phases[3], ProbePhase)
        # Phase 0 has the instruction observation, layer "instruction".
        instr_phase = sched.phases[0]
        assert len(instr_phase.observations) == 1
        assert instr_phase.observations[0].layer_name == "instruction"
        assert (
            instr_phase.observations[0].text
            == cand.paradigm_metadata["instruction_text"]
        )
        # Phase 1 advances clock without firing maintenance.
        adv = sched.phases[1]
        assert adv.delta_seconds == 60.0
        assert adv.run_maintenance is False
        # Phase 2 carries the trajectory.
        traj_phase = sched.phases[2]
        assert len(traj_phase.observations) > 0
        # Probe at end with the candidate's question.
        probe = sched.phases[3]
        assert len(probe.query_set) == 1
        assert probe.query_set[0].question == cand.question
        assert probe.query_set[0].answer == cand.answer

    def test_missing_instruction_text_raises(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        bad = CandidateQuestion(
            question_id="x",
            question="?",
            answer="check it",
            category="prospective_memory",
            paradigm="prospective_memory",
            scene_ids=["house_0"],
            paradigm_metadata={},
        )
        try:
            build_prospective_memory_schedule(bad, _scene())
        except ValueError as exc:
            assert "instruction_text" in str(exc)
        else:
            raise AssertionError("expected ValueError")
