import json
import os
import tempfile

from harness.benchmarks.academic.loaders.locomo import LoCoMoLoader
from harness.benchmarks.academic.loaders.sqa3d import SQA3DLoader


class TestLoCoMoLoader:
    def _make_data(self, tmpdir, n_conversations=2, n_turns=5, n_questions=3):
        conversations = []
        for c in range(n_conversations):
            # Build session turns in the actual LoCoMo format
            turns = [
                {
                    "dia_id": f"d{c}_{t}",
                    "speaker": "Alice" if t % 2 == 0 else "Bob",
                    "text": f"Turn {t} of conversation {c}",
                }
                for t in range(n_turns)
            ]
            qa = [
                {
                    "question_id": f"q{c}_{q}",
                    "question": f"What did Alice say in turn {q * 2}?",
                    "answer": f"Turn {q * 2} of conversation {c}",
                    "category": "factual",
                }
                for q in range(n_questions)
            ]
            # Actual LoCoMo format: sessions as dict keys within "conversation"
            conversations.append({
                "sample_id": f"conv_{c}",
                "conversation": {
                    "session_1": turns,
                    "session_1_date_time": "1:00 pm on 8 May, 2023",
                },
                "qa": qa,
            })

        path = os.path.join(tmpdir, "locomo10.json")
        with open(path, "w") as f:
            json.dump(conversations, f)

    def test_load_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_conversations=2, n_turns=4, n_questions=2)
            loader = LoCoMoLoader(tmpdir)
            samples = list(loader.load())

            assert len(samples) == 2
            assert samples[0].sample_id == "conv_0"
            assert len(samples[0].trajectory) == 4
            assert len(samples[0].questions) == 2
            assert all(f.position == (0.0, 0.0, 0.0) for f in samples[0].trajectory)

    def test_max_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_conversations=5)
            loader = LoCoMoLoader(tmpdir, max_conversations=2)
            samples = list(loader.load())
            assert len(samples) == 2

    def test_trajectory_has_speaker_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_conversations=1, n_turns=2)
            loader = LoCoMoLoader(tmpdir)
            samples = list(loader.load())
            assert "[Alice]:" in samples[0].trajectory[0].text
            assert "[Bob]:" in samples[0].trajectory[1].text

    def test_conversation_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_conversations=1, n_turns=2)
            loader = LoCoMoLoader(tmpdir)
            samples = list(loader.load())
            assert all(f.layer_name == "conversation" for f in samples[0].trajectory)

    def test_session_date_in_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_conversations=1, n_turns=2)
            loader = LoCoMoLoader(tmpdir)
            samples = list(loader.load())
            text = samples[0].trajectory[0].text
            assert "[Session 1 — 1:00 pm on 8 May, 2023]" in text
            assert "[Alice]:" in text

    def test_name(self):
        loader = LoCoMoLoader("/nonexistent")
        assert loader.name == "locomo"


class TestSQA3DLoader:
    def _make_data(self, tmpdir, n_scenes=1, n_questions_per_scene=2, n_objects=3):
        import numpy as np

        questions = []
        annotations = []
        for s in range(n_scenes):
            scene_id = f"scene{s:04d}_00"
            for q in range(n_questions_per_scene):
                qid = s * 100 + q
                questions.append({
                    "question_id": qid,
                    "scene_id": scene_id,
                    "question": f"What is near position {q}?",
                    "situation": f"I am standing in scene {s}",
                })
                annotations.append({
                    "question_id": qid,
                    "scene_id": scene_id,
                    "answers": [
                        {"answer": "chair", "answer_confidence": "yes"},
                        {"answer": "table", "answer_confidence": "maybe"},
                    ],
                    "position": {"x": float(q), "y": 0.0, "z": 0.0},
                    "question_type": "spatial",
                })

            # Create scene object bboxes
            scene_dir = os.path.join(tmpdir, "scannet", scene_id)
            os.makedirs(scene_dir, exist_ok=True)

            bboxes = np.zeros((n_objects, 8), dtype=np.float64)
            for o in range(n_objects):
                bboxes[o, :3] = [float(o), float(o), 0.0]
                bboxes[o, 3:6] = [1.0, 1.0, 1.0]
                bboxes[o, 6] = o  # label_id
                bboxes[o, 7] = o  # obj_id

            np.save(os.path.join(scene_dir, f"{scene_id}_aligned_bbox.npy"), bboxes)

        # SQA3D file layout: sqa_task/balanced/
        sqa_dir = os.path.join(tmpdir, "sqa_task", "balanced")
        os.makedirs(sqa_dir, exist_ok=True)

        with open(os.path.join(sqa_dir, "v1_balanced_questions_val_scannetv2.json"), "w") as f:
            json.dump({"questions": questions}, f)
        with open(os.path.join(sqa_dir, "v1_balanced_sqa_annotations_val_scannetv2.json"), "w") as f:
            json.dump({"annotations": annotations}, f)

    def test_load_scenes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_scenes=1, n_questions_per_scene=2, n_objects=3)
            loader = SQA3DLoader(tmpdir, split="val")
            samples = list(loader.load())

            assert len(samples) == 2
            assert samples[0].scene_id.startswith("scene")
            assert len(samples[0].trajectory) == 3

    def test_trajectory_has_object_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_scenes=1, n_questions_per_scene=1, n_objects=2)
            loader = SQA3DLoader(tmpdir, split="val")
            samples = list(loader.load())

            texts = [f.text for f in samples[0].trajectory]
            assert "object_0" in texts
            assert "object_1" in texts

    def test_agent_position_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_scenes=1, n_questions_per_scene=1)
            loader = SQA3DLoader(tmpdir, split="val")
            samples = list(loader.load())
            assert samples[0].agent_position is not None

    def test_object_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_data(tmpdir, n_scenes=1, n_questions_per_scene=1, n_objects=2)
            loader = SQA3DLoader(tmpdir, split="val")
            samples = list(loader.load())
            assert all(f.layer_name == "object" for f in samples[0].trajectory)

    def test_name(self):
        loader = SQA3DLoader("/nonexistent")
        assert loader.name == "sqa3d"
