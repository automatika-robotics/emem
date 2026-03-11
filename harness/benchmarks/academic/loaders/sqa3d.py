import csv
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)

log = logging.getLogger(__name__)

# NYU40 label ID -> human-readable name, loaded once from the TSV
_nyu40_cache: Optional[Dict[int, str]] = None


def _load_nyu40_labels(tsv_path: str) -> Dict[int, str]:
    """Load NYU40 label mapping from ``scannetv2-labels.combined.tsv``.

    :param tsv_path: Path to the TSV file.
    :returns: Mapping of NYU40 label ID to label name.
    """
    global _nyu40_cache
    if _nyu40_cache is not None:
        return _nyu40_cache

    label_map: Dict[int, str] = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                nyu_id = int(row["nyu40id"])
                label_map[nyu_id] = row["nyu40class"]
            except (KeyError, ValueError):
                continue
    _nyu40_cache = label_map
    return label_map


class SQA3DLoader:
    """Loader for the SQA3D situated question answering dataset.

    Yields one :class:`BenchmarkSample` per question. Questions sharing a
    scene reuse the same trajectory to avoid redundant object loading.

    Expected data directory layout::

        data_dir/
            sqa_task/
                balanced/
                    v1_balanced_questions_{split}_scannetv2.json
                    v1_balanced_sqa_annotations_{split}_scannetv2.json
            scannet/
                {scene_id}/
                    {scene_id}_aligned_bbox.npy
            scannetv2-labels.combined.tsv
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        max_scenes: Optional[int] = None,
    ):
        """
        :param data_dir: Root directory containing ``sqa_task/`` and
            ``scannet/`` subdirectories.
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
                log.warning("No scene objects for %s, skipping", scene_id)
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
        """Load questions and annotations, group by scene_id.

        Questions come from the questions file; answers, positions, and
        rotations come from the annotations file (keyed by ``question_id``).

        :returns: Mapping of scene_id to list of question dicts.
        """
        q_path = os.path.join(
            self._data_dir, "sqa_task", "balanced",
            f"v1_balanced_questions_{self._split}_scannetv2.json",
        )
        a_path = os.path.join(
            self._data_dir, "sqa_task", "balanced",
            f"v1_balanced_sqa_annotations_{self._split}_scannetv2.json",
        )

        with open(q_path) as f:
            q_data = json.load(f)
        with open(a_path) as f:
            a_data = json.load(f)

        # Build annotation lookup: question_id -> {answer, position, ...}
        annotations: Dict[str, Dict[str, Any]] = {}
        for ann in a_data.get("annotations", []):
            qid = str(ann["question_id"])
            ann_answers = ann.get("answers", [{"answer": ""}])
            best = max(
                ann_answers,
                key=lambda a: 1.0 if a.get("answer_confidence") == "yes" else 0.0,
            )
            extra = [a["answer"] for a in ann_answers if a["answer"] != best["answer"]]

            # Position is in the annotations file
            position: Optional[Tuple[float, float, float]] = None
            if "position" in ann:
                p = ann["position"]
                position = (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))

            annotations[qid] = {
                "answer": best["answer"],
                "extra_answers": extra,
                "position": position,
                "question_type": ann.get("question_type", ""),
            }

        by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for q in q_data.get("questions", []):
            qid = str(q["question_id"])
            scene_id = q["scene_id"]
            ann = annotations.get(qid, {"answer": "", "extra_answers": [], "position": None, "question_type": ""})

            by_scene[scene_id].append({
                "question_id": qid,
                "question": q["question"],
                "answer": ann["answer"],
                "extra_answers": ann["extra_answers"],
                "situation": q.get("situation", ""),
                "position": ann["position"],
                "question_type": ann["question_type"],
            })

        return dict(by_scene)

    def _load_scene_objects(self, scene_id: str) -> List[TrajectoryFrame]:
        """Load ScanNet object annotations as trajectory frames.

        Each object's bounding box center becomes a :class:`TrajectoryFrame`
        with the NYU40 label name as text. The bbox array has shape
        ``(N, 8)``: ``cx, cy, cz, dx, dy, dz, label_id, obj_id``.

        :param scene_id: ScanNet scene identifier (e.g. ``"scene0380_00"``).
        :returns: List of frames, one per object. Empty if bbox file missing.
        """
        scene_dir = os.path.join(self._data_dir, "scannet", scene_id)
        bbox_path = os.path.join(scene_dir, f"{scene_id}_aligned_bbox.npy")

        if not os.path.exists(bbox_path):
            return []

        bboxes = np.load(bbox_path)

        # Load NYU40 label names
        label_map = self._get_label_map()

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

    def _get_label_map(self) -> Dict[int, str]:
        """Load NYU40 label mapping, trying multiple locations.

        :returns: Mapping of NYU40 label ID to human-readable name.
        """
        candidates = [
            os.path.join(self._data_dir, "scannetv2-labels.combined.tsv"),
            os.path.join(self._data_dir, "scannet", "meta_data", "scannetv2-labels.combined.tsv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return _load_nyu40_labels(path)
        log.warning("NYU40 label TSV not found, using numeric label IDs")
        return {}
