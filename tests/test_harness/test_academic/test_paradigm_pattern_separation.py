"""Unit tests for the pattern-separation paradigm (A14b.3).

Uses synthetic ProcTHOR-shaped house dicts + mocked LLM so no
dataset download, no Ollama server required. Covers:

  * scene_units groups by similarity_pair_id, drops singletons,
    returns 2-element lists in deterministic order
  * Generator produces candidates only for objects that are in
    exactly one of the two houses' rooms
  * correct_house label maps to the house that actually contains
    the probe object
  * Common / shared objects are never proposed (only "only_in_a"
    and "only_in_b" lists are shown to the LLM)
  * Two-house schedule builder emits IngestPhase(A) →
    IngestPhase(B) → ProbePhase, with B's timestamps shifted past A
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.pattern_separation import (
    HOUSE_LABEL_A,
    HOUSE_LABEL_B,
    PatternSeparationGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_two_house_schedule,
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


def _make_house_a() -> Dict[str, Any]:
    """Kitchen with Fridge + Toaster, Bedroom with Bed + Lamp."""
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
        ],
        "objects": [
            {"id": "Fridge|0|0", "position": {"x": 1.0, "z": 1.0}},
            {"id": "Toaster|0|1", "position": {"x": 2.0, "z": 2.0}},
            {"id": "CounterTop|0|2", "position": {"x": 3.0, "z": 3.0}},
            {"id": "Bed|1|0", "position": {"x": 6.0, "z": 2.0}},
            {"id": "Lamp|1|1", "position": {"x": 7.0, "z": 2.0}},
        ],
    }


def _make_house_b() -> Dict[str, Any]:
    """Same room layout but different contents: Microwave + Oven in
    Kitchen, Pillow + CeilingLight in Bedroom. Shares CounterTop."""
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
        ],
        "objects": [
            {"id": "Microwave|0|0", "position": {"x": 1.0, "z": 1.0}},
            {"id": "Oven|0|1", "position": {"x": 2.0, "z": 2.0}},
            {"id": "CounterTop|0|2", "position": {"x": 3.0, "z": 3.0}},  # shared
            {"id": "Pillow|1|0", "position": {"x": 6.0, "z": 2.0}},
            {"id": "CeilingLight|1|1", "position": {"x": 7.0, "z": 2.0}},
        ],
    }


def _scene(sample_id: str, pair_id: str, dataset_idx: int) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "similarity_pair_id": pair_id,
        "procthor_dataset_index": dataset_idx,
        "trajectory": [
            {
                "frame_id": "f0",
                "position": [1.0, 1.0, 0.0],
                "timestamp": 100.0,
                "room_type": "Kitchen",
                "layers": {"detections": "fridge", "vlm": "", "place": ""},
            },
            {
                "frame_id": "f1",
                "position": [6.0, 2.0, 0.0],
                "timestamp": 105.0,
                "room_type": "Bedroom",
                "layers": {"detections": "bed", "vlm": "", "place": ""},
            },
        ],
        "interoception": [],
        "scene_objects": [],
    }


class TestSceneUnits:
    def test_groups_by_pair_and_drops_singletons(self):
        gen = PatternSeparationGenerator(
            llm_chat=lambda _p: "[]",
            house_lookup=lambda _i: None,
        )
        scenes = [
            _scene("house_0", "pair_0", 0),
            _scene("house_1", "pair_0", 1),
            _scene("house_2", "pair_1", 2),  # singleton — drop
            _scene("house_3", "pair_2", 3),
            _scene("house_4", "pair_2", 4),
            {"sample_id": "house_orphan"},  # no pair id — drop
        ]
        units = gen.scene_units(scenes)
        assert len(units) == 2
        pair_ids = [u[0]["similarity_pair_id"] for u in units]
        assert pair_ids == ["pair_0", "pair_2"]  # sorted alphabetically
        for u in units:
            assert len(u) == 2

    def test_alpha_beta_rotation_spreads_over_pairs(self):
        """Across many pairs the hash-parity rotation should give
        both sample_ids a roughly-equal chance at being Alpha."""
        gen = PatternSeparationGenerator(
            llm_chat=lambda _p: "[]",
            house_lookup=lambda _i: None,
        )
        alpha_is_first = 0
        total = 40
        for i in range(total):
            scenes = [
                _scene("house_a", f"pair_{i:03d}", 2 * i),
                _scene("house_b", f"pair_{i:03d}", 2 * i + 1),
            ]
            units = gen.scene_units(scenes)
            if units[0][0]["sample_id"] == "house_a":
                alpha_is_first += 1
        # Expect roughly 50/50 split with slack for 40 samples.
        assert 10 <= alpha_is_first <= 30


class TestPatternSeparationGenerator:
    def _lookup_for(self, a: Dict[str, Any], b: Dict[str, Any]):
        def lookup(idx: int) -> Dict[str, Any]:
            return a if idx == 0 else b

        return lookup

    def _llm_stub(self, responses_by_room: Dict[str, str]):
        def chat(prompt: str) -> str:
            for room_phrase, response in responses_by_room.items():
                if room_phrase in prompt:
                    return response
            return "[]"

        return chat

    def test_correct_house_label(self):
        """'Toaster' is in house A's kitchen; 'Oven' is in house B's.
        The generator should label each with the right correct_house."""
        llm = self._llm_stub({"kitchen": '["Toaster", "Oven"]'})
        gen = PatternSeparationGenerator(
            llm_chat=llm,
            house_lookup=self._lookup_for(_make_house_a(), _make_house_b()),
            target_total=10,
        )
        scene_a = _scene("house_0", "pair_0", 0)
        scene_b = _scene("house_1", "pair_0", 1)
        cands = gen.generate([scene_a, scene_b], n_per_scene=4)
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        by_obj = {c.paradigm_metadata["probe_object"]: c for c in kitchen}
        assert "toaster" in by_obj
        assert "oven" in by_obj
        assert by_obj["toaster"].answer == HOUSE_LABEL_A
        assert by_obj["oven"].answer == HOUSE_LABEL_B
        assert by_obj["toaster"].paradigm_metadata["correct_house"] == HOUSE_LABEL_A

    def test_shared_objects_are_dropped(self):
        """'CounterTop' is in both kitchens — not a valid probe."""
        llm = self._llm_stub({"kitchen": '["CounterTop", "Toaster"]'})
        gen = PatternSeparationGenerator(
            llm_chat=llm,
            house_lookup=self._lookup_for(_make_house_a(), _make_house_b()),
            target_total=10,
        )
        cands = gen.generate(
            [_scene("house_0", "pair_0", 0), _scene("house_1", "pair_0", 1)],
            n_per_scene=2,
        )
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        objs = {c.paradigm_metadata["probe_object"] for c in kitchen}
        assert "toaster" in objs
        assert "countertop" not in objs

    def test_question_text_uses_labels_and_lowercase(self):
        llm = self._llm_stub({"kitchen": '["Oven"]'})
        gen = PatternSeparationGenerator(
            llm_chat=llm,
            house_lookup=self._lookup_for(_make_house_a(), _make_house_b()),
            target_total=10,
        )
        cands = gen.generate(
            [_scene("house_0", "pair_0", 0), _scene("house_1", "pair_0", 1)],
            n_per_scene=1,
        )
        assert cands
        q = cands[0].question
        assert HOUSE_LABEL_A in q and HOUSE_LABEL_B in q
        assert "an oven" in q  # lowercase + vowel-initial article
        assert "in the kitchen" in q

    def test_scene_ids_are_paired(self):
        llm = self._llm_stub({"kitchen": '["Toaster"]'})
        gen = PatternSeparationGenerator(
            llm_chat=llm,
            house_lookup=self._lookup_for(_make_house_a(), _make_house_b()),
            target_total=10,
        )
        cands = gen.generate(
            [_scene("alpha_house", "pair_0", 0), _scene("beta_house", "pair_0", 1)],
            n_per_scene=1,
        )
        assert cands[0].scene_ids == ["alpha_house", "beta_house"]
        mapping = cands[0].paradigm_metadata["label_to_sample_id"]
        assert mapping == {
            HOUSE_LABEL_A: "alpha_house",
            HOUSE_LABEL_B: "beta_house",
        }

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = PatternSeparationGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low

    def test_exclude_probes_suppresses_duplicate(self):
        llm = self._llm_stub({"kitchen": '["Toaster", "Oven"]'})
        gen = PatternSeparationGenerator(
            llm_chat=llm,
            house_lookup=self._lookup_for(_make_house_a(), _make_house_b()),
            target_total=10,
            exclude_probes={"Kitchen": {"toaster"}},
        )
        cands = gen.generate(
            [_scene("house_0", "pair_0", 0), _scene("house_1", "pair_0", 1)],
            n_per_scene=2,
        )
        objs = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "toaster" not in objs
        assert "oven" in objs


class TestBuildTwoHouseSchedule:
    def _candidate(self) -> CandidateQuestion:
        return CandidateQuestion(
            question_id="patsep_q1",
            question=(
                f"You explored two houses, {HOUSE_LABEL_A} and "
                f"{HOUSE_LABEL_B}. In which house did you see a toaster "
                f"in the kitchen?"
            ),
            answer=HOUSE_LABEL_A,
            category="pattern_separation",
            paradigm="pattern_separation",
            scene_ids=["alpha_house", "beta_house"],
            tools_expected=["semantic_search"],
            paradigm_metadata={
                "pair_id": "pair_0",
                "room_type": "Kitchen",
                "probe_object": "toaster",
                "correct_house": HOUSE_LABEL_A,
                "label_to_sample_id": {
                    HOUSE_LABEL_A: "alpha_house",
                    HOUSE_LABEL_B: "beta_house",
                },
            },
        )

    def test_two_ingests_then_probe(self):
        scenes_by_id = {
            "alpha_house": _scene("alpha_house", "pair_0", 0),
            "beta_house": _scene("beta_house", "pair_0", 1),
        }
        sched = build_two_house_schedule(self._candidate(), scenes_by_id)
        assert len(sched.phases) == 3
        ing_a, ing_b, probe = sched.phases
        assert isinstance(ing_a, IngestPhase)
        assert isinstance(ing_b, IngestPhase)
        assert isinstance(probe, ProbePhase)
        assert ing_a.episode_name.endswith("alpha")
        assert ing_b.episode_name.endswith("beta")
        # Every observation carries its house prefix.
        assert all(o.text.startswith("In house Alpha: ") for o in ing_a.observations)
        assert all(o.text.startswith("In house Beta: ") for o in ing_b.observations)
        # House B's timestamps are strictly after house A's.
        max_a_ts = max(o.timestamp for o in ing_a.observations)
        min_b_ts = min(o.timestamp for o in ing_b.observations)
        assert min_b_ts > max_a_ts
        # Probe fires after house B.
        assert probe.at_time > max(o.timestamp for o in ing_b.observations)

    def test_missing_scene_raises(self):
        scenes_by_id = {"alpha_house": _scene("alpha_house", "pair_0", 0)}
        with pytest.raises(ValueError, match="not in scenes_by_id"):
            build_two_house_schedule(self._candidate(), scenes_by_id)

    def test_empty_trajectory_raises(self):
        a = _scene("alpha_house", "pair_0", 0)
        a["trajectory"] = []
        scenes_by_id = {
            "alpha_house": a,
            "beta_house": _scene("beta_house", "pair_0", 1),
        }
        with pytest.raises(ValueError, match="no ingestible observations"):
            build_two_house_schedule(self._candidate(), scenes_by_id)

    def test_missing_label_mapping_raises(self):
        cand = self._candidate()
        cand.paradigm_metadata.pop("label_to_sample_id")
        with pytest.raises(ValueError, match="missing label_to_sample_id"):
            build_two_house_schedule(cand, {})
