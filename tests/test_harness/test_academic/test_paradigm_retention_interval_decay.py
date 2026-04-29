"""Unit tests for the retention-interval-decay paradigm (A14c.1).

Uses a synthetic ProcTHOR-shaped house dict + a mocked LLM picker.
Covers:

  * Generator picks unique-to-one-room objects and emits one
    candidate per (item, delay) pair.
  * Hallucinated picks (LLM proposes items absent from the unique
    pool) are dropped.
  * Each candidate carries the correct ``retention_delay_bucket`` /
    ``retention_delay_seconds`` metadata.
  * Question text and ``answer="yes"`` are correct.
  * Pre-filter rubric mentions yes / ambiguous / no.
  * The schedule_builder helper emits a 3-phase schedule (ingest →
    advance with maintenance → probe) at the expected probe time.
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.retention_interval_decay import (
    RETENTION_DELAYS,
    RetentionIntervalDecayGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_retention_decay_schedule,
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


def _scene_with_trajectory(sample_id: str = "house_0") -> Dict[str, Any]:
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


def _scene_no_trajectory(sample_id: str = "house_0") -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "procthor_dataset_index": 42,
        "trajectory": [],
        "interoception": [],
        "scene_objects": [],
    }


class TestRetentionIntervalDecayGenerator:
    def _llm_stub(self, picks: List[str]):
        import json as _json

        def chat(prompt: str) -> str:
            return _json.dumps(picks)

        return chat

    def test_emits_one_candidate_per_item_per_delay(self):
        """Picking 2 items with 6 delays -> 12 candidates."""
        llm = self._llm_stub(["Fridge", "Lamp"])
        gen = RetentionIntervalDecayGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=24,
        )
        # n_per_scene = 2 items × 6 delays.
        cands = gen.generate([_scene_no_trajectory()], n_per_scene=12)
        assert len(cands) == 12
        from collections import Counter

        buckets = Counter(c.paradigm_metadata["retention_delay_bucket"] for c in cands)
        # Every bucket appears exactly twice (once per item).
        for name, _ in RETENTION_DELAYS:
            assert buckets[name] == 2

    def test_hallucinated_picks_are_dropped(self):
        llm = self._llm_stub(["NotARealItem", "Lamp"])
        gen = RetentionIntervalDecayGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=24,
        )
        cands = gen.generate([_scene_no_trajectory()], n_per_scene=12)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "lamp" in probes
        assert "notarealitem" not in probes
        # 1 valid item × 6 delays
        assert len(cands) == len(RETENTION_DELAYS)

    def test_shared_house_plant_is_not_picked(self):
        """HousePlant is in two rooms — must not appear as a probe."""
        llm = self._llm_stub(["HousePlant", "Lamp"])
        gen = RetentionIntervalDecayGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
        )
        cands = gen.generate([_scene_no_trajectory()], n_per_scene=6)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "house plant" not in probes
        assert "lamp" in probes

    def test_question_format_and_answer(self):
        llm = self._llm_stub(["Fridge"])
        gen = RetentionIntervalDecayGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=24,
        )
        cands = gen.generate([_scene_no_trajectory()], n_per_scene=6)
        # All candidates for the same item have the same question and answer.
        assert all(c.question == "Did you see the fridge?" for c in cands)
        assert all(c.answer == "yes" for c in cands)
        # Metadata records the right delay seconds for each bucket.
        seconds_by_bucket = {
            c.paradigm_metadata["retention_delay_bucket"]: c.paradigm_metadata[
                "retention_delay_seconds"
            ]
            for c in cands
        }
        assert seconds_by_bucket == dict(RETENTION_DELAYS)

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
        gen = RetentionIntervalDecayGenerator(
            llm_chat=self._llm_stub([]),
            house_lookup=lambda _i: house,
            target_total=12,
        )
        cands = gen.generate([_scene_no_trajectory()], n_per_scene=6)
        assert cands == []

    def test_exclude_probes_suppresses_duplicates(self):
        llm = self._llm_stub(["Fridge", "Lamp"])
        gen = RetentionIntervalDecayGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=12,
            exclude_probes={"__retdec__": {"fridge"}},
        )
        cands = gen.generate([_scene_no_trajectory()], n_per_scene=6)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "fridge" not in probes
        assert "lamp" in probes

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = RetentionIntervalDecayGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestRetentionDecayScheduleBuilder:
    def test_three_phase_schedule_with_correct_advance(self):
        """Schedule = IngestPhase → AdvanceClockPhase(maint=True) → ProbePhase."""
        llm = lambda _p: '["Fridge"]'  # noqa: E731
        gen = RetentionIntervalDecayGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=3,
        )
        cands = gen.generate([_scene_with_trajectory()], n_per_scene=3)
        # Pick the 1d candidate.
        cand_1d = next(
            c for c in cands if c.paradigm_metadata["retention_delay_bucket"] == "1d"
        )
        sched = build_retention_decay_schedule(cand_1d, _scene_with_trajectory())
        assert len(sched.phases) == 3
        assert isinstance(sched.phases[0], IngestPhase)
        assert isinstance(sched.phases[1], AdvanceClockPhase)
        assert isinstance(sched.phases[2], ProbePhase)
        assert sched.phases[1].run_maintenance is True
        assert sched.phases[1].delta_seconds == 86400  # 1 day
        # Probe should fire at end_of_ingest + 86400.
        assert sched.phases[2].at_time == 200.0 + 86400
        assert sched.phases[2].probe_id == "1d"
        assert len(sched.phases[2].query_set) == 1

    def test_missing_delay_metadata_raises(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        bad = CandidateQuestion(
            question_id="x",
            question="?",
            answer="yes",
            category="retention_interval_decay",
            paradigm="retention_interval_decay",
            scene_ids=["house_0"],
            paradigm_metadata={},  # missing retention_delay_seconds
        )
        try:
            build_retention_decay_schedule(bad, _scene_with_trajectory())
        except ValueError as exc:
            assert "retention_delay_seconds" in str(exc)
        else:
            raise AssertionError("expected ValueError")
