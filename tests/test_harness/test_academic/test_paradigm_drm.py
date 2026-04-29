"""Unit tests for the DRM generator, schedule builder, and scene loader.

Uses a synthetic ProcTHOR-shaped house dict + a mocked LLM so no
scenes.jsonl on disk, no Ollama server, no prior dataset download
required. Covers:

  * scene_entries.load_scene_entries merges manifest + trajectory
  * group_waypoints_by_room partitions by room_type (kept for review)
  * procthor_utils.objects_by_room buckets objects by floor polygon
  * DRMGenerator parses a well-formed JSON response, filters leakers
    against the ProcTHOR ground-truth list, caps at the per-scene
    budget, and stamps the ground-truth list on each CandidateQuestion
  * build_ingest_then_probe_schedule emits a 2-phase Schedule whose
    probe fires after the last observation timestamp
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
    DRMGenerator,
    _indefinite_article,
    _parse_absent_items,
    _probe_for_question,
    _probe_object_leaks,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_ingest_then_probe_schedule,
)
from harness.benchmarks.academic.emem_bench_v1.scene_entries import (
    group_waypoints_by_room,
    load_scene_entries,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    IngestPhase,
    ProbePhase,
)
from harness.environments.procthor_utils import objects_by_room


def _unit_square(x0: float, z0: float, side: float = 4.0) -> List[Dict[str, float]]:
    return [
        {"x": x0, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0 + side},
        {"x": x0, "y": 0.0, "z": z0 + side},
    ]


def _fake_house() -> Dict[str, Any]:
    """Synthetic ProcTHOR-shaped house matching the real dataset format.

    Objects carry ``id`` (with type as the first |-separated segment)
    rather than ``objectType``; a ``TVStand`` has a nested ``Television``
    child to exercise the tree-walking in ``objects_by_room``.
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
        ],
        "objects": [
            {"id": "Fridge|0|0", "position": {"x": 1.0, "y": 0.0, "z": 1.0}},
            {"id": "Microwave|0|1", "position": {"x": 2.0, "y": 1.0, "z": 2.0}},
            {"id": "Chair|0|2", "position": {"x": 3.0, "y": 0.0, "z": 3.0}},
            {
                "id": "TVStand|1|0",
                "position": {"x": 6.0, "y": 0.0, "z": 2.5},
                "children": [
                    {
                        "id": "Television|1|0|0",
                        "position": {"x": 6.0, "y": 0.6, "z": 2.5},
                    },
                    {
                        "id": "RemoteControl|1|0|1",
                        "position": {"x": 6.1, "y": 0.7, "z": 2.5},
                    },
                ],
            },
            {"id": "Bed|1|1", "position": {"x": 6.5, "y": 0.0, "z": 2.0}},
            {"id": "Lamp|1|2", "position": {"x": 7.0, "y": 0.5, "z": 2.0}},
            # Outside any room polygon: hallway fixture.
            {"id": "Doorway|9|0", "position": {"x": 50.0, "y": 0.0, "z": 50.0}},
        ],
    }


def _fake_scene(sample_id: str = "house_0") -> Dict[str, Any]:
    """Merged scene entry the paradigm generator consumes.

    Includes ``procthor_dataset_index`` so the generator can ask its
    house_lookup for the real house dict.
    """
    return {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "procthor_dataset_index": 42,
        "trajectory": [
            {
                "frame_id": "f0",
                "position": [1.0, 1.0, 0.0],
                "timestamp": 100.0,
                "room_type": "Kitchen",
                "layers": {
                    "detections": "fridge",
                    "vlm": "kitchen",
                    "place": "kitchen",
                },
            },
            {
                "frame_id": "f1",
                "position": [6.0, 2.0, 0.0],
                "timestamp": 104.0,
                "room_type": "Bedroom",
                "layers": {"detections": "bed", "vlm": "bedroom", "place": "bedroom"},
            },
        ],
        "interoception": [],
        "scene_objects": [],
    }


class TestSceneEntries:
    def test_load_merges_manifest_and_trajectory(self, tmp_path: Path):
        (tmp_path / "house_000").mkdir()
        traj = {
            "trajectory": [{"frame_id": "f0", "timestamp": 1.0, "layers": {}}],
            "interoception": [{"timestamp": 1.0, "battery": "battery: 99%"}],
            "metadata": {"scene_objects": [{"objectType": "Fridge"}]},
        }
        (tmp_path / "house_000" / "trajectory.json").write_text(json.dumps(traj))
        (tmp_path / "scenes.jsonl").write_text(
            json.dumps({
                "sample_id": "procthor_house_000",
                "scene_id": "procthor_house_000",
                "trajectory_path": "house_000/trajectory.json",
                "procthor_dataset_index": 3208,
                "room_count": 4,
            })
            + "\n"
        )

        entries = load_scene_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["sample_id"] == "procthor_house_000"
        assert e["procthor_dataset_index"] == 3208
        assert len(e["trajectory"]) == 1

    def test_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_scene_entries(tmp_path)

    def test_group_waypoints_by_room(self):
        trajectory = _fake_scene()["trajectory"]
        grouped = group_waypoints_by_room(trajectory)
        assert grouped["Kitchen"]["n_waypoints"] == 1
        assert grouped["Bedroom"]["n_waypoints"] == 1


class TestObjectsByRoom:
    def test_attributes_objects_via_polygon_and_walks_children(self):
        mapping = objects_by_room(_fake_house())
        # Kitchen: top-level furniture only.
        assert sorted(mapping["Kitchen"]) == ["Chair", "Fridge", "Microwave"]
        # Bedroom: includes the Television + RemoteControl that are children
        # of TVStand; the tree walk must catch them.
        assert sorted(mapping["Bedroom"]) == [
            "Bed",
            "Lamp",
            "RemoteControl",
            "TVStand",
            "Television",
        ]
        assert mapping.get("outside") == ["Doorway"]

    def test_object_without_id_is_skipped(self):
        house = {
            "rooms": [
                {
                    "id": "r0",
                    "roomType": "Kitchen",
                    "floorPolygon": _unit_square(0.0, 0.0),
                },
            ],
            "objects": [
                # No id -> no type to extract; skipped without crashing.
                {"position": {"x": 1.0, "z": 1.0}},
                {"id": "Fridge|0|0", "position": {"x": 1.0, "z": 1.0}},
            ],
        }
        mapping = objects_by_room(house)
        assert mapping["Kitchen"] == ["Fridge"]


class TestParseAbsentItems:
    def test_valid_json_array(self):
        raw = '["toaster", "coffee maker", "blender"]'
        assert _parse_absent_items(raw) == ["toaster", "coffee maker", "blender"]

    def test_embedded_in_prose(self):
        raw = 'Here are three items: ["toaster", "kettle"]. Hope that helps!'
        assert _parse_absent_items(raw) == ["toaster", "kettle"]

    def test_deduplicates_case_insensitive(self):
        raw = '["Toaster", "toaster", "kettle"]'
        assert _parse_absent_items(raw) == ["Toaster", "kettle"]

    def test_malformed_returns_empty(self):
        assert _parse_absent_items("nonsense text with no json") == []
        assert _parse_absent_items("[this is not json]") == []

    def test_non_string_items_skipped(self):
        raw = '["toaster", 42, null, "kettle"]'
        assert _parse_absent_items(raw) == ["toaster", "kettle"]

    def test_empty_input(self):
        assert _parse_absent_items("") == []


class TestProbeLeakFilter:
    """Ground-truth leak filter: probes whose tokens overlap any
    ground-truth objectType are dropped."""

    def test_token_match_blocks_synonym(self):
        gt = ["Fridge", "DiningTable", "Chair"]
        # "coffee table" shares "table" token with "DiningTable"
        assert _probe_object_leaks("coffee table", gt) is True
        # "mini fridge" shares "fridge" with "Fridge"
        assert _probe_object_leaks("mini fridge", gt) is True

    def test_unrelated_passes(self):
        gt = ["Fridge", "DiningTable", "Chair"]
        assert _probe_object_leaks("toaster", gt) is False
        assert _probe_object_leaks("coffee maker", gt) is False

    def test_camelcase_gt_splits_into_tokens(self):
        # "DiningTable" splits into ["Dining", "Table"]; probe
        # "dining chair" shares "Dining".
        assert _probe_object_leaks("dining chair", ["DiningTable"]) is True

    def test_short_tokens_ignored(self):
        # "tv" is 2 chars — won't false-positive against "Television"
        assert _probe_object_leaks("tv", ["Television"]) is False

    def test_case_insensitive(self):
        assert _probe_object_leaks("Table", ["dining_TABLE"]) is True

    def test_empty_ground_truth_passes_everything(self):
        assert _probe_object_leaks("anything", []) is False


class TestQuestionFormatting:
    """The question template must lowercase the probe and pick the
    right indefinite article so natural-language output is grammatical."""

    def test_article_a_for_consonant_initial(self):
        assert _indefinite_article("toaster") == "a"
        assert _indefinite_article("mirror") == "a"
        assert _indefinite_article("coffee table") == "a"

    def test_article_an_for_vowel_initial(self):
        assert _indefinite_article("oven") == "an"
        assert _indefinite_article("alarm clock") == "an"
        assert _indefinite_article("umbrella") == "an"

    def test_article_case_insensitive(self):
        assert _indefinite_article("Oven") == "an"
        assert _indefinite_article("Alarm Clock") == "an"

    def test_article_skips_leading_punctuation(self):
        assert _indefinite_article('"oven"') == "an"

    def test_probe_for_question_lowercases_and_strips(self):
        assert _probe_for_question("Alarm Clock") == "alarm clock"
        assert _probe_for_question("  Mirror  ") == "mirror"

    def test_probe_for_question_splits_camel_case(self):
        # Raw ProcTHOR objectType values are CamelCase. They must
        # render as natural phrases in the question text.
        assert _probe_for_question("ShelvingUnit") == "shelving unit"
        assert _probe_for_question("CoffeeMachine") == "coffee machine"
        assert _probe_for_question("TVStand") == "tv stand"
        # Already-split phrases are left alone.
        assert _probe_for_question("dental floss") == "dental floss"


class TestDRMGeneratorQuestionText:
    """End-to-end check: the generated question uses lowercase probe
    and the correct a/an article."""

    def test_generated_question_is_grammatical(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
            DRMGenerator,
        )

        def llm(prompt: str) -> str:
            # Propose two items: one vowel-initial and one consonant-initial.
            if "kitchen" in prompt:
                return '["Oven", "Toaster"]'
            return "[]"

        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
        )
        cands = gen.generate([_fake_scene()], n_per_scene=4)
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        texts = sorted(c.question for c in kitchen)
        assert texts == [
            "Did you see a toaster in the kitchen?",
            "Did you see an oven in the kitchen?",
        ]
        # Probe in metadata is also lowercase.
        assert {c.paradigm_metadata["probe_object"] for c in kitchen} == {
            "oven",
            "toaster",
        }


class TestDRMGeneratorExclusion:
    """Proposals that match an exclusion-set entry (case-insensitive,
    regardless of the LLM's output casing) must be rejected."""

    def test_exclude_probes_rejects_duplicates(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
            DRMGenerator,
        )

        def llm(prompt: str) -> str:
            # "Toaster" is excluded; "Mixer" is new → only Mixer survives.
            if "kitchen" in prompt:
                return '["Toaster", "Mixer"]'
            return "[]"

        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
            exclude_probes={"Kitchen": {"toaster"}},
        )
        cands = gen.generate([_fake_scene()], n_per_scene=2)
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        assert [c.paradigm_metadata["probe_object"] for c in kitchen] == ["mixer"]

    def test_exclude_is_case_insensitive(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.drm import (
            DRMGenerator,
        )

        def llm(prompt: str) -> str:
            return '["TOASTER"]' if "kitchen" in prompt else "[]"

        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: _fake_house(),
            target_total=10,
            exclude_probes={"Kitchen": {"Toaster"}},
        )
        cands = gen.generate([_fake_scene()], n_per_scene=2)
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        assert kitchen == []


class TestDRMGeneratorGroundTruth:
    def _llm_stub(self, responses_by_room: Dict[str, str]):
        def chat(prompt: str) -> str:
            for room_phrase, response in responses_by_room.items():
                if room_phrase in prompt:
                    return response
            return "[]"

        return chat

    def _house_lookup_stub(self, house: Dict[str, Any]):
        def lookup(idx: int) -> Dict[str, Any]:
            return house

        return lookup

    def test_candidate_shape_and_ground_truth_embedding(self):
        llm = self._llm_stub({
            "kitchen": '["toaster", "coffee maker"]',
            "bedroom": '["alarm clock", "curtains"]',
        })
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=self._house_lookup_stub(_fake_house()),
            target_total=10,
        )
        cands = gen.generate([_fake_scene()], n_per_scene=4)
        # 2 rooms × 2 items budget each = 4 candidates
        assert len(cands) == 4
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        assert len(kitchen) == 2
        k0 = kitchen[0]
        assert k0.answer == "no"
        assert k0.paradigm == "drm"
        assert k0.category == "drm"
        assert "toaster" in k0.question.lower()
        assert k0.tools_expected == ["semantic_search", "entity_query"]
        # Ground-truth object list is embedded on the candidate.
        gt = k0.paradigm_metadata["ground_truth_objects_in_room"]
        assert sorted(gt) == ["Chair", "Fridge", "Microwave"]

    def test_ground_truth_leaker_dropped(self):
        """LLM proposes an item that overlaps a ground-truth type
        — the filter must reject it."""
        # Kitchen ground truth has "Fridge"; LLM proposes "mini fridge"
        # (shares "fridge" token) + "toaster" (safe).
        llm = self._llm_stub({"kitchen": '["mini fridge", "toaster"]'})
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=self._house_lookup_stub(_fake_house()),
            target_total=10,
        )
        cands = gen.generate([_fake_scene()], n_per_scene=1)
        kitchen = [c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"]
        assert len(kitchen) == 1
        assert kitchen[0].paradigm_metadata["probe_object"] == "toaster"

    def test_skips_scene_without_dataset_index(self):
        llm = self._llm_stub({})
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=self._house_lookup_stub(_fake_house()),
            target_total=10,
        )
        bad_scene = {"sample_id": "x", "trajectory": []}
        cands = gen.generate([bad_scene], n_per_scene=4)
        assert cands == []

    def test_skips_when_house_lookup_returns_none(self):
        llm = self._llm_stub({})
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=lambda _i: None,
            target_total=10,
        )
        cands = gen.generate([_fake_scene()], n_per_scene=4)
        assert cands == []

    def test_skips_empty_rooms(self):
        """A house whose eligible rooms hold no objects produces 0 candidates."""
        empty_house = {
            "rooms": [
                {
                    "id": "r0",
                    "roomType": "Kitchen",
                    "floorPolygon": _unit_square(0.0, 0.0),
                }
            ],
            "objects": [],
        }
        gen = DRMGenerator(
            llm_chat=self._llm_stub({"kitchen": '["toaster"]'}),
            house_lookup=self._house_lookup_stub(empty_house),
            target_total=10,
        )
        cands = gen.generate([_fake_scene()], n_per_scene=4)
        assert cands == []

    def test_malformed_llm_output_produces_no_candidates(self):
        llm = self._llm_stub({
            "kitchen": "I refuse to answer because I am an AI.",
            "bedroom": "garbage",
        })
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=self._house_lookup_stub(_fake_house()),
            target_total=10,
        )
        cands = gen.generate([_fake_scene()], n_per_scene=4)
        assert cands == []

    def test_target_total_cap_applies_across_scenes(self):
        llm = self._llm_stub({
            "kitchen": '["a", "b", "c", "d"]',
            "bedroom": '["e", "f", "g", "h"]',
        })
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=self._house_lookup_stub(_fake_house()),
            target_total=3,
        )
        cands = gen.generate([_fake_scene("h0"), _fake_scene("h1")], n_per_scene=4)
        assert len(cands) == 3

    def test_question_id_stable_and_id_safe(self):
        llm = self._llm_stub({"kitchen": '["coffee maker"]'})
        gen = DRMGenerator(
            llm_chat=llm,
            house_lookup=self._house_lookup_stub(_fake_house()),
            target_total=10,
        )
        cands = gen.generate([_fake_scene("house_0")], n_per_scene=1)
        assert cands
        qid = cands[0].question_id
        assert " " not in qid
        assert qid.startswith("drm_house_0_kitchen_coffee_maker")

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = DRMGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestBuildIngestThenProbeSchedule:
    def _candidate(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        return CandidateQuestion(
            question_id="drm_q1",
            question="Did you see a toaster in the kitchen?",
            answer="no",
            category="drm",
            paradigm="drm",
            scene_ids=["house_0"],
            tools_expected=["semantic_search"],
        )

    def test_emits_ingest_then_probe(self):
        schedule = build_ingest_then_probe_schedule(
            self._candidate(), _fake_scene("house_0")
        )
        assert len(schedule.phases) == 2
        assert isinstance(schedule.phases[0], IngestPhase)
        assert isinstance(schedule.phases[1], ProbePhase)

        ingest = schedule.phases[0]
        # 2 text-bearing waypoints × 3 non-empty layers = 6 observations.
        assert len(ingest.observations) == 6
        assert ingest.episode_name == "drm_encode"

        probe = schedule.phases[1]
        assert probe.probe_id == "drm"
        assert len(probe.query_set) == 1
        assert probe.at_time > max(o.timestamp for o in ingest.observations)

    def test_empty_trajectory_raises(self):
        empty_scene = {"sample_id": "x", "trajectory": []}
        with pytest.raises(ValueError, match="no ingestible observations"):
            build_ingest_then_probe_schedule(self._candidate(), empty_scene)
