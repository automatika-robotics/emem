import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)


class SQA3DLoader:
    """Loader for the SQA3D situated question answering dataset.

    Loads SQA3D questions and ScanNet object annotations, yielding one
    :class:`BenchmarkSample` per question (each with its own agent position).
    Questions sharing a scene share the same trajectory.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        max_scenes: Optional[int] = None,
    ):
        """
        :param data_dir: Root directory containing ``question/``, ``answer/``,
            and ``scannet/`` subdirectories.
        :param split: Dataset split (``"train"``, ``"val"``, or ``"test"``).
        :param max_scenes: Maximum number of scenes to load.
        """
        self._data_dir = data_dir
        self._split = split
        self._max_scenes = max_scenes

    @property
    def name(self) -> str:
        return "sqa3d"

    def load(self) -> Iterator[BenchmarkSample]:
        """Yield one :class:`BenchmarkSample` per question.

        Questions are grouped by scene internally so scene objects are
        loaded only once per scene.
        """
        questions_by_scene = self._load_questions()

        scene_count = 0
        for scene_id, scene_questions in questions_by_scene.items():
            if self._max_scenes is not None and scene_count >= self._max_scenes:
                break

            trajectory = self._load_scene_objects(scene_id)
            if not trajectory:
                continue

            for q_data in scene_questions:
                yield BenchmarkSample(
                    sample_id=f"{scene_id}_{q_data['question_id']}",
                    scene_id=scene_id,
                    trajectory=trajectory,
                    questions=[BenchmarkQuestion(
                        question_id=q_data["question_id"],
                        question=q_data["question"],
                        answer=q_data["answer"],
                        category=q_data.get("question_type", ""),
                        extra_answers=q_data.get("extra_answers", []),
                    )],
                    agent_position=q_data.get("position"),
                    agent_situation=q_data.get("situation", ""),
                )

            scene_count += 1

    def _load_questions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load and group questions by scene_id.

        :returns: Mapping of scene_id to list of question dicts.
        """
        q_path = os.path.join(
            self._data_dir, "question", "balanced",
            f"v1_balanced_questions_{self._split}_scannetv2.json",
        )
        a_path = os.path.join(
            self._data_dir, "answer", "balanced",
            f"v1_balanced_answers_{self._split}_scannetv2.json",
        )

        with open(q_path) as f:
            q_data = json.load(f)
        with open(a_path) as f:
            a_data = json.load(f)

        answers: Dict[str, Dict[str, Any]] = {}
        for ann in a_data.get("annotations", []):
            qid = str(ann["question_id"])
            ann_answers = ann.get("answers", [{"answer": ""}])
            best = max(
                ann_answers,
                key=lambda a: 1.0 if a.get("answer_confidence") == "yes" else 0.0,
            )
            extra = [a["answer"] for a in ann_answers if a["answer"] != best["answer"]]
            answers[qid] = {"answer": best["answer"], "extra_answers": extra}

        by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for q in q_data.get("questions", []):
            qid = str(q["question_id"])
            scene_id = q["scene_id"]

            position: Optional[Tuple[float, float, float]] = None
            if "position" in q:
                p = q["position"]
                position = (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))

            ans = answers.get(qid, {"answer": "", "extra_answers": []})
            by_scene[scene_id].append({
                "question_id": qid,
                "question": q["question"],
                "answer": ans["answer"],
                "extra_answers": ans["extra_answers"],
                "situation": q.get("situation", ""),
                "position": position,
                "question_type": q.get("question_type", ""),
            })

        return dict(by_scene)

    def _load_scene_objects(self, scene_id: str) -> List[TrajectoryFrame]:
        """Load ScanNet object annotations as trajectory frames.

        Each object's bounding box center becomes a :class:`TrajectoryFrame`
        with the object label as text.

        :param scene_id: ScanNet scene identifier.
        :returns: List of frames, one per scene object. Empty if files missing.
        """
        scene_dir = os.path.join(self._data_dir, "scannet", scene_id)
        bbox_path = os.path.join(scene_dir, f"{scene_id}_aligned_bbox.npy")
        labels_path = os.path.join(scene_dir, f"{scene_id}_sem_labels.json")

        if not os.path.exists(bbox_path):
            return []

        bboxes = np.load(bbox_path)
        label_map: Dict[int, str] = {}
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                raw = json.load(f)
                label_map = {int(k): v for k, v in raw.items()}

        frames: List[TrajectoryFrame] = []
        for i, bbox in enumerate(bboxes):
            cx, cy, cz = float(bbox[0]), float(bbox[1]), float(bbox[2])
            label_id = int(bbox[6])
            label = label_map.get(label_id, f"object_{label_id}")

            frames.append(TrajectoryFrame(
                frame_id=f"{scene_id}_obj{i}",
                position=(cx, cy, cz),
                timestamp=float(i),
                text=label,
                layer_name="object",
            ))

        return frames
