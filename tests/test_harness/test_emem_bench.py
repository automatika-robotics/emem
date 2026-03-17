"""Tests for eMEM-Bench loader and question generation."""

import json
import os
import tempfile

import pytest

from harness.benchmarks.academic.loaders.emem_bench import EMEMBenchLoader
from harness.benchmarks.emem_bench.generate_questions import (
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

        intero_frames = [f for f in s.trajectory if f.layer_name.startswith("interoception:")]
        assert len(intero_frames) == 4  # 2 entries * 2 keys (battery, cpu_temp)
        battery_frames = [f for f in intero_frames if "battery" in f.layer_name]
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
