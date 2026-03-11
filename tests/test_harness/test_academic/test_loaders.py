import json
import os
import tempfile

from harness.benchmarks.academic.loaders.locomo import LoCoMoLoader
from harness.benchmarks.academic.loaders.sqa3d import SQA3DLoader


class TestLoCoMoLoader:
    def _make_data(self, tmpdir, n_conversations=2, n_turns=5, n_questions=3):
        conversations = []
        for c in range(n_conversations):
            turns = [
                {
                    "speaker": "Alice" if t % 2 == 0 else "Bob",
                    "text": f"Turn {t} of conversation {c}",
                    "timestamp": t * 60.0,
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
            conversations.append({
                "conversation_id": f"conv_{c}",
                "sessions": [{"turns": turns}],
                "qa_pairs": qa,
            })

        path = os.path.join(tmpdir, "locomo.json")
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
                    "position": {"x": float(q), "y": 0.0, "z": 0.0},
                })
                annotations.append({
                    "question_id": qid,
                    "answers": [
                        {"answer": "chair", "answer_confidence": "yes"},
                        {"answer": "table", "answer_confidence": "maybe"},
                    ],
                })

            scene_dir = os.path.join(tmpdir, "scannet", scene_id)
            os.makedirs(scene_dir, exist_ok=True)

            bboxes = np.zeros((n_objects, 8), dtype=np.float64)
            labels = {}
            for o in range(n_objects):
                bboxes[o, :3] = [float(o), float(o), 0.0]
                bboxes[o, 3:6] = [1.0, 1.0, 1.0]
                bboxes[o, 6] = o
                bboxes[o, 7] = o
                labels[o] = f"object_{o}"

            np.save(os.path.join(scene_dir, f"{scene_id}_aligned_bbox.npy"), bboxes)
            with open(os.path.join(scene_dir, f"{scene_id}_sem_labels.json"), "w") as f:
                json.dump(labels, f)

        q_dir = os.path.join(tmpdir, "question", "balanced")
        a_dir = os.path.join(tmpdir, "answer", "balanced")
        os.makedirs(q_dir, exist_ok=True)
        os.makedirs(a_dir, exist_ok=True)

        with open(os.path.join(q_dir, "v1_balanced_questions_val_scannetv2.json"), "w") as f:
            json.dump({"questions": questions}, f)
        with open(os.path.join(a_dir, "v1_balanced_answers_val_scannetv2.json"), "w") as f:
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


class TestOpenEQALoader:
    def test_name(self):
        from harness.benchmarks.academic.loaders.open_eqa import OpenEQALoader
        loader = OpenEQALoader("/nonexistent")
        assert loader.name == "open-eqa"

    def test_load_with_mock_data(self):
        from harness.benchmarks.academic.loaders.open_eqa import OpenEQALoader

        with tempfile.TemporaryDirectory() as tmpdir:
            ep_dir = os.path.join(tmpdir, "frames", "hm3d-v0", "episode1")
            os.makedirs(ep_dir)
            for i in range(5):
                open(os.path.join(ep_dir, f"{i:05d}-rgb.png"), "w").close()

            entries = [
                {
                    "question_id": "q1",
                    "question": "What color is the wall?",
                    "answer": "white",
                    "episode_history": "hm3d-v0/episode1",
                    "category": "color",
                },
                {
                    "question_id": "q2",
                    "question": "How many chairs?",
                    "answer": "two",
                    "episode_history": "hm3d-v0/episode1",
                },
            ]
            with open(os.path.join(tmpdir, "open-eqa-v0.json"), "w") as f:
                json.dump(entries, f)

            loader = OpenEQALoader(tmpdir, n_frames=3)
            samples = list(loader.load())

            assert len(samples) == 1
            assert len(samples[0].questions) == 2
            assert len(samples[0].trajectory) == 3
