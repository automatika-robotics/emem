"""Unit tests for the DRM generator, schedule builder, and scene loader.

Uses a fake trajectory + mocked LLM so no scenes.jsonl on disk and no
Ollama server required. Covers:

  * scene_entries.load_scene_entries merges manifest + trajectory
  * group_waypoints_by_room partitions by room_type and captures layers
  * DRMGenerator parses a well-formed JSON response, rejects malformed
    LLM output gracefully, caps at the per-scene budget, and stamps
    expected fields on each CandidateQuestion
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
    _parse_absent_items,
    _probe_object_leaks,
    _room_observation_blob,
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


def _make_scene(sample_id: str = "house_0") -> Dict[str, Any]:
    """Minimal merged scene entry with kitchen + bedroom waypoints."""
    return {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "trajectory": [
            {
                "frame_id": "f0",
                "position": [1.0, 1.0, 0.0],
                "timestamp": 100.0,
                "room_type": "Kitchen",
                "layers": {
                    "detections": "fridge, counter, tile floor",
                    "vlm": "A bright kitchen with white cabinets.",
                    "place": "kitchen",
                },
            },
            {
                "frame_id": "f1",
                "position": [1.5, 1.5, 0.0],
                "timestamp": 102.0,
                "room_type": "Kitchen",
                "layers": {
                    "detections": "cabinet, table",
                    "vlm": "Wooden table in the kitchen.",
                    "place": "kitchen",
                },
            },
            {
                "frame_id": "f2",
                "position": [5.0, 5.0, 0.0],
                "timestamp": 104.0,
                "room_type": "Bedroom",
                "layers": {
                    "detections": "bed, lamp, dresser",
                    "vlm": "A small bedroom with a bed and lamp.",
                    "place": "bedroom",
                },
            },
            {
                "frame_id": "f3",
                "position": [6.0, 5.0, 0.0],
                "timestamp": 106.0,
                "room_type": None,  # doorway / outside
                "layers": {
                    "detections": "",
                    "vlm": "",
                    "place": "",
                },
            },
        ],
        "interoception": [],
        "scene_objects": [],
    }


class TestSceneEntries:
    def test_load_merges_manifest_and_trajectory(self, tmp_path: Path):
        # Build a minimal v1 data directory.
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
                "room_count": 4,
            })
            + "\n"
        )

        entries = load_scene_entries(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e["sample_id"] == "procthor_house_000"
        assert len(e["trajectory"]) == 1
        assert e["interoception"][0]["battery"] == "battery: 99%"
        assert e["scene_objects"][0]["objectType"] == "Fridge"

    def test_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_scene_entries(tmp_path)

    def test_group_waypoints_by_room(self):
        trajectory = _make_scene()["trajectory"]
        grouped = group_waypoints_by_room(trajectory)
        assert grouped["Kitchen"]["n_waypoints"] == 2
        assert grouped["Kitchen"]["detections"] == [
            "fridge, counter, tile floor",
            "cabinet, table",
        ]
        assert grouped["Bedroom"]["n_waypoints"] == 1
        # Empty-layer waypoint without room_type bucketed under "outside"
        # but contributes no text.
        assert "outside" in grouped
        assert grouped["outside"]["detections"] == []


class TestParseAbsentItems:
    def test_valid_json_array(self):
        raw = '["toaster", "coffee maker", "blender"]'
        assert _parse_absent_items(raw) == ["toaster", "coffee maker", "blender"]

    def test_embedded_in_prose(self):
        raw = 'Here are three items: ["toaster", "kettle"]. Hope that helps!'
        assert _parse_absent_items(raw) == ["toaster", "kettle"]

    def test_deduplicates_case_insensitive(self):
        raw = '["Toaster", "toaster", "kettle"]'
        # Second "toaster" dropped; first-seen casing preserved.
        assert _parse_absent_items(raw) == ["Toaster", "kettle"]

    def test_malformed_returns_empty(self):
        assert _parse_absent_items("nonsense text with no json") == []
        assert _parse_absent_items("[this is not json]") == []

    def test_non_string_items_skipped(self):
        raw = '["toaster", 42, null, "kettle"]'
        assert _parse_absent_items(raw) == ["toaster", "kettle"]

    def test_empty_input(self):
        assert _parse_absent_items("") == []


class TestDRMGenerator:
    def _llm(self, responses_by_room: Dict[str, str]):
        """Stub LLM that returns different responses per room_phrase."""

        def chat(prompt: str) -> str:
            for phrase, response in responses_by_room.items():
                if phrase in prompt:
                    return response
            return "[]"

        return chat

    def test_generate_shapes_candidates(self):
        llm = self._llm({
            "kitchen": '["toaster", "coffee maker"]',
            "bedroom": '["alarm clock", "curtains"]',
        })
        gen = DRMGenerator(llm_chat=llm, target_total=10)
        scenes = [_make_scene("house_0")]
        cands = gen.generate(scenes, n_per_scene=4)
        # Two eligible rooms × up to 2 items each (budget split); all kept.
        assert len(cands) == 4
        by_room = {}
        for c in cands:
            by_room.setdefault(c.paradigm_metadata["room_type"], []).append(c)
        assert len(by_room["Kitchen"]) == 2
        assert len(by_room["Bedroom"]) == 2
        kitchen_cand = by_room["Kitchen"][0]
        assert kitchen_cand.answer == "no"
        assert kitchen_cand.paradigm == "drm"
        assert kitchen_cand.category == "drm"
        assert "toaster" in kitchen_cand.question.lower()
        assert "kitchen" in kitchen_cand.question.lower()
        assert kitchen_cand.paradigm_metadata["probe_object"] == "toaster"
        assert kitchen_cand.tools_expected == ["semantic_search", "entity_query"]
        assert kitchen_cand.scene_ids == ["house_0"]

    def test_skips_scenes_with_no_eligible_rooms(self):
        llm = self._llm({})
        gen = DRMGenerator(llm_chat=llm, target_total=10)
        scene = {
            "sample_id": "x",
            "trajectory": [
                {
                    "frame_id": "f0",
                    "position": [0, 0, 0],
                    "timestamp": 1.0,
                    "room_type": None,
                    "layers": {"detections": "a, b", "vlm": "", "place": ""},
                }
            ],
        }
        cands = gen.generate([scene], n_per_scene=4)
        assert cands == []

    def test_malformed_llm_output_produces_no_candidates(self):
        llm = self._llm({
            "kitchen": "I refuse to answer because I am an AI.",
            "bedroom": "also refusal",
        })
        gen = DRMGenerator(llm_chat=llm, target_total=10)
        cands = gen.generate([_make_scene()], n_per_scene=4)
        assert cands == []

    def test_target_total_cap_applies_across_scenes(self):
        llm = self._llm({
            "kitchen": '["a", "b", "c", "d"]',
            "bedroom": '["e", "f", "g", "h"]',
        })
        gen = DRMGenerator(llm_chat=llm, target_total=3)
        scenes = [_make_scene("h0"), _make_scene("h1")]
        cands = gen.generate(scenes, n_per_scene=4)
        assert len(cands) == 3  # cap honoured

    def test_question_id_stable_and_id_safe(self):
        llm = self._llm({"kitchen": '["coffee maker"]'})
        gen = DRMGenerator(llm_chat=llm, target_total=10)
        cands = gen.generate([_make_scene("house_0")], n_per_scene=1)
        assert cands
        qid = cands[0].question_id
        # Spaces squashed into underscores; lower-cased.
        assert " " not in qid
        assert qid.startswith("drm_house_0_kitchen_coffee_maker")

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = DRMGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestProbeLeakFilter:
    def test_token_match_blocks_partial(self):
        blob = "wall, table, lamp, fridge"
        # "coffee table" shares the token "table" → leak
        assert _probe_object_leaks("coffee table", blob) is True
        # "wooden table" shares "table"
        assert _probe_object_leaks("wooden table", blob) is True

    def test_unrelated_passes(self):
        blob = "wall, table, lamp, fridge"
        assert _probe_object_leaks("toaster", blob) is False
        assert _probe_object_leaks("coffee maker", blob) is False

    def test_short_tokens_ignored(self):
        # "tv" is 2 chars so it shouldn't false-positive against
        # "television" (and vice versa on the other end).
        assert _probe_object_leaks("tv", "television stand") is False

    def test_word_boundary_not_substring(self):
        # "art" in "artwork" shouldn't count (word-boundary check).
        blob = "artwork on wall"
        assert _probe_object_leaks("art", blob) is False
        # But "art" in "art piece" should.
        assert _probe_object_leaks("art", "art piece on table") is True

    def test_case_insensitive(self):
        assert _probe_object_leaks("Table", "wooden TABLE") is True

    def test_room_blob_concats_all_layers(self):
        aggregate = {
            "n_waypoints": 3,
            "detections": ["fridge, wall"],
            "vlm_descriptions": ["A bright kitchen."],
            "places": ["kitchen"],
        }
        blob = _room_observation_blob(aggregate)
        assert "fridge" in blob
        assert "bright" in blob
        assert "kitchen" in blob
        assert blob == blob.lower()  # lowercased


class TestDRMGeneratorLeakFilter:
    def _llm(self, response: str):
        def chat(prompt: str) -> str:
            return response

        return chat

    def test_drops_leaking_items(self):
        """Scene has 'table' in detections; generator proposes 'coffee
        table' — must be dropped."""
        scene = _make_scene("house_0")  # detections include "table"
        gen = DRMGenerator(
            llm_chat=self._llm('["coffee table", "toaster", "wooden table"]'),
            target_total=10,
        )
        cands = gen.generate([scene], n_per_scene=2)
        # All three proposed; "coffee table" and "wooden table" leak
        # (scene detections mention "table"). Only "toaster" survives.
        kitchen_cands = [
            c for c in cands if c.paradigm_metadata["room_type"] == "Kitchen"
        ]
        object_names = {c.paradigm_metadata["probe_object"] for c in kitchen_cands}
        assert "toaster" in object_names
        assert "coffee table" not in object_names
        assert "wooden table" not in object_names


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
            self._candidate(), _make_scene("house_0")
        )
        assert len(schedule.phases) == 2
        assert isinstance(schedule.phases[0], IngestPhase)
        assert isinstance(schedule.phases[1], ProbePhase)

        ingest = schedule.phases[0]
        # 3 text-bearing waypoints × 3 non-empty layers = 9 observations.
        # The 4th waypoint has no room_type and empty layers, so it
        # contributes 0.
        assert len(ingest.observations) == 9
        assert ingest.episode_name == "drm_encode"

        probe = schedule.phases[1]
        assert probe.probe_id == "drm"
        assert len(probe.query_set) == 1
        # Probe fires after the last observation timestamp.
        assert probe.at_time > max(o.timestamp for o in ingest.observations)

    def test_empty_trajectory_raises(self):
        empty_scene = {"sample_id": "x", "trajectory": []}
        with pytest.raises(ValueError, match="no ingestible observations"):
            build_ingest_then_probe_schedule(self._candidate(), empty_scene)
