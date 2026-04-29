"""Unit tests for the pattern-completion paradigm (A14b.4).

Uses synthetic ProcTHOR-shaped house dicts + mocked LLM. Covers:

  * Generator picks objects unique to one room (not in any other room)
  * Question text uses lowercase + correct article
  * Ground-truth answer is the room phrase
  * Hallucinated picks (LLM proposes items not in the list) are dropped
  * Pre-filter rubric mentions yes / ambiguous / no
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.pattern_completion import (
    PatternCompletionGenerator,
)


def _unit_square(x0: float, z0: float, side: float = 4.0) -> List[Dict[str, float]]:
    return [
        {"x": x0, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0 + side},
        {"x": x0, "y": 0.0, "z": z0 + side},
    ]


def _fake_house() -> Dict[str, Any]:
    """A house with overlapping + unique objects across rooms.

    Layout:
      Kitchen (0..4, 0..4): Fridge, Microwave, HousePlant
      Bedroom (5..9, 0..4): Bed, HousePlant, Lamp
      Bathroom (10..14, 0..4): Toilet, Sink, Mirror

    HousePlant appears in 2 rooms — should NOT be a candidate.
    Fridge / Microwave / Bed / Lamp / Toilet / Sink / Mirror appear in
    exactly one room — they ARE candidates.
    """
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
        "trajectory": [],
        "interoception": [],
        "scene_objects": [],
    }


class TestPatternCompletionGenerator:
    def _llm_stub(self, responses_by_room: Dict[str, str]):
        def chat(prompt: str) -> str:
            for room_phrase, response in responses_by_room.items():
                if f"unique to the {room_phrase}" in prompt:
                    return response
            return "[]"

        return chat

    def test_generates_candidates_only_for_unique_room_objects(self):
        """HousePlant is in two rooms — generator must drop it even
        if the LLM returns it. Lamp is unique to bedroom — kept."""
        llm = self._llm_stub({
            "kitchen": '["Fridge", "HousePlant"]',
            "bedroom": '["Lamp", "HousePlant"]',
            "bathroom": '["Mirror"]',
        })
        gen = PatternCompletionGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
        )
        cands = gen.generate([_scene()], n_per_scene=6)
        probes = [c.paradigm_metadata["probe_object"] for c in cands]
        assert "house plant" not in probes
        assert "fridge" in probes
        assert "lamp" in probes
        assert "mirror" in probes

    def test_question_format_and_answer(self):
        llm = self._llm_stub({
            "kitchen": '["Fridge"]',
            "bedroom": '["Lamp"]',
            "bathroom": '["Mirror"]',
        })
        gen = PatternCompletionGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
        )
        cands = gen.generate([_scene()], n_per_scene=3)
        by_obj = {c.paradigm_metadata["probe_object"]: c for c in cands}
        # Question text uses lowercase + correct article
        assert by_obj["fridge"].question == "What room did you see a fridge in?"
        assert by_obj["lamp"].question == "What room did you see a lamp in?"
        # Answers map to the room phrase
        assert by_obj["fridge"].answer == "kitchen"
        assert by_obj["lamp"].answer == "bedroom"
        assert by_obj["mirror"].answer == "bathroom"

    def test_hallucinated_picks_are_dropped(self):
        """LLM proposes an item not in any unique-list. The generator
        must drop it instead of building a candidate."""
        llm = self._llm_stub({
            "kitchen": '["NotARealItem"]',  # not in unique list
            "bedroom": '["Lamp"]',
            "bathroom": '["Mirror"]',
        })
        gen = PatternCompletionGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
        )
        cands = gen.generate([_scene()], n_per_scene=3)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "notarealitem" not in probes
        assert "lamp" in probes
        assert "mirror" in probes

    def test_skips_scene_with_no_unique_objects(self):
        """A house where every object type appears in multiple rooms
        produces zero candidates."""
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
        gen = PatternCompletionGenerator(
            llm_chat=self._llm_stub({}),
            house_lookup=lambda _i: house,
            target_total=10,
        )
        cands = gen.generate([_scene()], n_per_scene=3)
        assert cands == []

    def test_exclude_probes_suppresses_duplicates(self):
        llm = self._llm_stub({
            "kitchen": '["Fridge", "Microwave"]',
            "bedroom": '["Lamp"]',
            "bathroom": '["Mirror"]',
        })
        gen = PatternCompletionGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
            exclude_probes={"Kitchen": {"fridge"}},
        )
        cands = gen.generate([_scene()], n_per_scene=4)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "fridge" not in probes
        assert "microwave" in probes

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = PatternCompletionGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low

    def test_unique_per_room_in_metadata(self):
        """Each candidate carries the per-room unique-objects map so
        the reviewer can verify ground truth."""
        llm = self._llm_stub({
            "kitchen": '["Fridge"]',
            "bedroom": '["Lamp"]',
            "bathroom": '["Mirror"]',
        })
        gen = PatternCompletionGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
        )
        cands = gen.generate([_scene()], n_per_scene=3)
        md = cands[0].paradigm_metadata
        assert "unique_per_room" in md
        # House plant is shared, must NOT be in any room's unique list
        for room_objs in md["unique_per_room"].values():
            assert "HousePlant" not in room_objs
