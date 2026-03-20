"""Tests for eMEM-Bench loader and question generation."""

import json
import os
import tempfile

import pytest

from harness.benchmarks.academic.loaders.emem_bench import EMEMBenchLoader
from harness.benchmarks.academic.trajectory import BenchmarkQuestion
from harness.benchmarks.emem_bench.generate_questions import (
    _is_valid_caption,
    _is_valid_place,
    generate_questions_for_sample,
)


def _make_sample():
    """Create a minimal sample dict for testing."""
    return {
        "sample_id": "test_scene_01",
        "scene_id": "TestScene1",
        "source": "ai2thor",
        "trajectory": [
            {
                "frame_id": "frame_0000",
                "position": [1.0, 2.0, 0.0],
                "timestamp": 1000.0,
                "layers": {
                    "vlm": "A kitchen counter with a red mug and a toaster",
                    "detections": "mug, toaster, counter",
                    "place": "kitchen",
                },
            },
            {
                "frame_id": "frame_0001",
                "position": [3.0, 4.0, 0.0],
                "timestamp": 1002.0,
                "layers": {
                    "vlm": "A living room with a sofa and TV",
                    "detections": "sofa, TV, coffee table",
                    "place": "living room",
                },
            },
            {
                "frame_id": "frame_0002",
                "position": [1.5, 2.5, 0.0],
                "timestamp": 1004.0,
                "layers": {
                    "vlm": "Kitchen sink area with dishes",
                    "detections": "sink, plate, fork",
                    "place": "kitchen",
                },
            },
        ],
        "interoception": [
            {
                "timestamp": 1000.0,
                "battery": "battery: 95%",
                "cpu_temp": "cpu_temp: 55C",
            },
            {
                "timestamp": 1004.0,
                "battery": "battery: 90%",
                "cpu_temp": "cpu_temp: 58C",
            },
        ],
        "questions": [
            {
                "question_id": "q001",
                "question": "Where is the red mug?",
                "answer": "kitchen counter near (1.0, 2.0)",
                "category": "spatial",
                "tools_expected": ["locate", "semantic_search"],
            },
            {
                "question_id": "q002",
                "question": "What is my current battery level?",
                "answer": "battery: 90%",
                "category": "interoception",
                "tools_expected": ["body_status"],
            },
        ],
        "metadata": {
            "scene_objects": [
                {
                    "objectId": "Mug|1|2|0",
                    "objectType": "Mug",
                    "name": "Mug",
                    "position": [1.0, 2.0, 0.0],
                    "visible": True,
                    "pickupable": True,
                    "receptacle": False,
                    "parentReceptacles": ["CounterTop"],
                },
                {
                    "objectId": "Toaster|1.2|2.1|0",
                    "objectType": "Toaster",
                    "name": "Toaster",
                    "position": [1.2, 2.1, 0.0],
                    "visible": True,
                    "pickupable": False,
                    "receptacle": False,
                    "parentReceptacles": ["CounterTop"],
                },
                {
                    "objectId": "CounterTop|1|2|0",
                    "objectType": "CounterTop",
                    "name": "CounterTop",
                    "position": [1.1, 2.0, 0.0],
                    "visible": True,
                    "pickupable": False,
                    "receptacle": True,
                    "parentReceptacles": [],
                },
            ],
        },
    }


class TestEMEMBenchLoader:
    def test_load_inline_sample(self, tmp_path):
        """Loader can read a sample from the index file (inline data)."""
        sample = _make_sample()
        index_path = tmp_path / "emem-bench-v0.json"
        index_path.write_text(json.dumps([sample]))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())

        assert len(samples) == 1
        s = samples[0]
        assert s.sample_id == "test_scene_01"
        assert s.scene_id == "TestScene1"
        assert len(s.questions) == 2
        # Trajectory: 3 frames * 3 layers + 2 interoception entries * 2 keys = 13
        assert len(s.trajectory) == 13

    def test_load_with_trajectory_path(self, tmp_path):
        """Loader resolves trajectory_path references."""
        sample = _make_sample()
        scene_dir = tmp_path / "ai2thor" / "testscene1"
        scene_dir.mkdir(parents=True)
        traj_path = scene_dir / "trajectory.json"
        traj_path.write_text(json.dumps(sample))

        index = [{
            "sample_id": "test_scene_01",
            "scene_id": "TestScene1",
            "source": "ai2thor",
            "trajectory_path": "ai2thor/testscene1/trajectory.json",
        }]
        (tmp_path / "emem-bench-v0.json").write_text(json.dumps(index))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())
        assert len(samples) == 1
        assert samples[0].sample_id == "test_scene_01"

    def test_multi_layer_frames(self, tmp_path):
        """Each layer in a waypoint produces a separate TrajectoryFrame."""
        sample = _make_sample()
        (tmp_path / "emem-bench-v0.json").write_text(json.dumps([sample]))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())
        s = samples[0]

        layer_names = {f.layer_name for f in s.trajectory}
        assert "vlm" in layer_names
        assert "detections" in layer_names
        assert "place" in layer_names

    def test_interoception_frames(self, tmp_path):
        """Interoception data is converted to frames with special layer names."""
        sample = _make_sample()
        (tmp_path / "emem-bench-v0.json").write_text(json.dumps([sample]))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())
        s = samples[0]

        intero_frames = [f for f in s.trajectory if f.is_interoception]
        assert len(intero_frames) == 4  # 2 entries * 2 keys (battery, cpu_temp)
        battery_frames = [f for f in intero_frames if f.layer_name == "battery"]
        assert len(battery_frames) == 2
        assert "battery: 95%" in battery_frames[0].text

    def test_max_samples(self, tmp_path):
        """max_samples limits the number of yielded samples."""
        sample1 = _make_sample()
        sample2 = {**_make_sample(), "sample_id": "test_scene_02"}
        (tmp_path / "emem-bench-v0.json").write_text(json.dumps([sample1, sample2]))

        loader = EMEMBenchLoader(str(tmp_path), max_samples=1)
        samples = list(loader.load())
        assert len(samples) == 1

    def test_fallback_scan(self, tmp_path):
        """Loader falls back to scanning scene directories."""
        scene_dir = tmp_path / "ai2thor" / "testscene1"
        scene_dir.mkdir(parents=True)
        sample = _make_sample()
        (scene_dir / "trajectory.json").write_text(json.dumps(sample))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())
        assert len(samples) == 1

    def test_2d_position_padded(self, tmp_path):
        """2D positions are padded to 3D."""
        sample = _make_sample()
        sample["trajectory"][0]["position"] = [1.0, 2.0]  # Only 2D
        (tmp_path / "emem-bench-v0.json").write_text(json.dumps([sample]))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())
        frame = samples[0].trajectory[0]
        assert len(frame.position) == 3
        assert frame.position[2] == 0.0


class TestQuestionGeneration:
    def test_generates_all_categories(self):
        """Question generation produces all 6 categories."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)

        categories = {q["category"] for q in questions}
        assert "spatial" in categories
        assert "temporal" in categories
        assert "cross_layer" in categories
        assert "entity" in categories
        assert "interoception" in categories
        assert "episodic" in categories

    def test_question_ids_assigned(self):
        """All generated questions get sequential IDs."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)

        for i, q in enumerate(questions):
            assert q["question_id"] == f"q{i:03d}"

    def test_tools_expected_present(self):
        """All generated questions have tools_expected."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)

        for q in questions:
            assert "tools_expected" in q
            assert isinstance(q["tools_expected"], list)
            assert len(q["tools_expected"]) > 0

    def test_spatial_questions_from_objects(self):
        """Spatial questions are generated from scene objects."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)
        spatial = [q for q in questions if q["category"] == "spatial"]
        assert len(spatial) >= 2

    def test_interoception_questions(self):
        """Interoception questions reference body state data."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)
        intero = [q for q in questions if q["category"] == "interoception"]
        assert len(intero) >= 1
        assert any("battery" in q["question"].lower() for q in intero)


class TestCaptionFiltering:
    def test_rejects_garbage_captions(self):
        assert not _is_valid_caption("")
        assert not _is_valid_caption("hello")  # too short
        assert not _is_valid_caption(
            "This image displays a uniform brown surface with no discernible objects"
        )
        assert not _is_valid_caption(
            "the image you uploaded consists solely of a uniform brown color"
        )
        assert not _is_valid_caption("I cannot identify any objects in this image")

    def test_accepts_good_captions(self):
        assert _is_valid_caption("A kitchen counter with a red mug and a toaster")
        assert _is_valid_caption("sofa, TV, coffee table, bookshelf")
        assert _is_valid_caption("A living room with a sofa and TV")

    def test_rejects_garbage_places(self):
        assert not _is_valid_place("abstract")
        assert not _is_valid_place("wallpaper")
        assert not _is_valid_place("wall")
        assert not _is_valid_place("")
        assert not _is_valid_place(
            "the image you uploaded consists solely of a uniform brown color"
        )

    def test_accepts_valid_places(self):
        assert _is_valid_place("kitchen")
        assert _is_valid_place("Kitchen")
        assert _is_valid_place("living room")
        assert _is_valid_place("bathroom")
        assert _is_valid_place("hallway")


class TestQuestionGenerationFixes:
    def test_no_structural_objects(self):
        """Spatial questions should not ask about Floor, Wall, etc."""
        sample = _make_sample()
        # Add structural objects
        sample["metadata"]["scene_objects"].extend([
            {
                "objectId": "Floor|0|0|0",
                "objectType": "Floor",
                "name": "Floor",
                "position": [0, 0, 0],
                "visible": True,
                "pickupable": False,
                "receptacle": False,
                "parentReceptacles": [],
            },
            {
                "objectId": "Wall|0|0|0",
                "objectType": "Wall",
                "name": "Wall",
                "position": [0, 0, 0],
                "visible": True,
                "pickupable": False,
                "receptacle": False,
                "parentReceptacles": [],
            },
        ])
        questions = generate_questions_for_sample(sample)
        spatial = [q for q in questions if q["category"] == "spatial"]
        for q in spatial:
            assert "floor" not in q["question"].lower() or "floorplan" in q["question"].lower()
            assert "Where is the wall?" != q["question"]

    def test_no_entity_count_questions(self):
        """No 'How many times have I seen X?' questions should be generated."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)
        for q in questions:
            assert "how many times" not in q["question"].lower()

    def test_no_duplicate_questions(self):
        """All question texts should be unique."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)
        texts = [q["question"] for q in questions]
        assert len(texts) == len(set(texts))

    def test_temporal_no_raw_timestamp(self):
        """'When did I last see X?' should not contain raw Unix timestamps."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)
        temporal = [q for q in questions if q["category"] == "temporal"]
        for q in temporal:
            if "when" in q["question"].lower() and "last" in q["question"].lower():
                assert "at timestamp" not in q["answer"]

    def test_describe_area_concise_gt(self):
        """Descriptive GT should be concise, not raw caption dumps."""
        sample = _make_sample()
        questions = generate_questions_for_sample(sample)
        for q in questions:
            if "describe" in q["question"].lower() or "tell me everything" in q["question"].lower():
                word_count = len(q["answer"].split())
                assert word_count < 100, f"GT too long ({word_count} words): {q['answer'][:80]}..."


class TestToolsExpected:
    def test_benchmark_question_has_tools_expected(self):
        """BenchmarkQuestion dataclass should have tools_expected field."""
        bq = BenchmarkQuestion(
            question_id="q1", question="test", answer="test",
        )
        assert bq.tools_expected == []

        bq2 = BenchmarkQuestion(
            question_id="q2", question="test", answer="test",
            tools_expected=["locate", "semantic_search"],
        )
        assert bq2.tools_expected == ["locate", "semantic_search"]

    def test_loader_extracts_tools_expected(self, tmp_path):
        """Loader should preserve tools_expected from JSON."""
        sample = _make_sample()
        (tmp_path / "emem-bench-v0.json").write_text(json.dumps([sample]))

        loader = EMEMBenchLoader(str(tmp_path))
        samples = list(loader.load())
        q1 = samples[0].questions[0]
        assert q1.tools_expected == ["locate", "semantic_search"]
        q2 = samples[0].questions[1]
        assert q2.tools_expected == ["body_status"]


class TestFormatObservation:
    def test_includes_timestamp(self):
        """_format_observation should include a timestamp."""
        from emem.tools import _format_observation
        from emem.types import ObservationNode
        import numpy as np

        obs = ObservationNode(
            text="A red mug on the counter",
            coordinates=np.array([1.0, 2.0, 0.0]),
            timestamp=1710000042.0,
            layer_name="vlm",
        )
        result = _format_observation(obs)
        assert "[vlm]" in result
        assert "(1.0,2.0)" in result
        assert "2024-03-09" in result  # UTC date for timestamp 1710000042
        assert "A red mug on the counter" in result
