"""Question generation for eMEM-Bench from AI2-THOR trajectory data.

Generates questions across 6 categories using object metadata, trajectory
structure, and template-based generation.

Usage::

    python -m harness.benchmarks.emem_bench.generate_questions \\
        --data-dir data/emem-bench/ai2thor/ \\
        --output data/emem-bench/emem-bench-v0.json
"""

import argparse
import json
import logging
import math
import os
import random
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Question templates ───────────────────────────────────────────────

def _spatial_questions(
    objects: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate spatial questions from object metadata.

    :param objects: Scene objects with positions.
    :param trajectory: Trajectory frames.
    :returns: List of question dicts.
    """
    questions: List[Dict[str, Any]] = []

    # "Where is X?" questions
    visible_objects = [o for o in objects if o.get("visible")]
    if not visible_objects:
        visible_objects = objects[:20]

    for obj in random.sample(visible_objects, min(8, len(visible_objects))):
        pos = obj.get("position", [0, 0, 0])
        obj_name = _format_object_name(obj["objectType"])

        # Find the place layer caption nearest to this object
        place = _find_place_near(trajectory, pos[0], pos[1])
        answer = f"{place} near ({pos[0]:.1f}, {pos[1]:.1f})" if place else f"near ({pos[0]:.1f}, {pos[1]:.1f})"

        questions.append({
            "question": f"Where is the {obj_name}?",
            "answer": answer,
            "category": "spatial",
            "tools_expected": ["locate", "semantic_search"],
        })

    # "What objects are near (x, y)?" questions
    if trajectory:
        for _ in range(min(4, len(trajectory))):
            wp = random.choice(trajectory)
            pos = wp.get("position", [0, 0, 0])
            nearby = _objects_near(objects, pos[0], pos[1], radius=2.0)
            if nearby:
                obj_names = ", ".join(_format_object_name(o["objectType"]) for o in nearby[:5])
                questions.append({
                    "question": f"What objects are near ({pos[0]:.1f}, {pos[1]:.1f})?",
                    "answer": obj_names,
                    "category": "spatial",
                    "tools_expected": ["spatial_query"],
                })

    # "Which room has X?" questions
    receptacles = [o for o in objects if o.get("receptacle")]
    for obj in random.sample(receptacles, min(3, len(receptacles))):
        obj_name = _format_object_name(obj["objectType"])
        pos = obj.get("position", [0, 0, 0])
        place = _find_place_near(trajectory, pos[0], pos[1])
        if place:
            questions.append({
                "question": f"Which room has the {obj_name}?",
                "answer": place,
                "category": "spatial",
                "tools_expected": ["locate", "semantic_search"],
            })

    return questions


def _temporal_questions(
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate temporal questions from trajectory timestamps.

    :param trajectory: Trajectory frames with timestamps.
    :returns: List of question dicts.
    """
    questions: List[Dict[str, Any]] = []

    if not trajectory:
        return questions

    # "What did I see most recently?"
    last_frame = trajectory[-1]
    last_vlm = last_frame.get("layers", {}).get("vlm", "")
    if last_vlm:
        questions.append({
            "question": "What did I see most recently?",
            "answer": last_vlm,
            "category": "temporal",
            "tools_expected": ["temporal_query"],
        })

    # "What did I see first?"
    first_frame = trajectory[0]
    first_vlm = first_frame.get("layers", {}).get("vlm", "")
    if first_vlm:
        questions.append({
            "question": "What did I observe at the beginning of my exploration?",
            "answer": first_vlm,
            "category": "temporal",
            "tools_expected": ["temporal_query"],
        })

    # "When did I last see X?" — pick a random detection
    for frame in random.sample(trajectory, min(3, len(trajectory))):
        detections = frame.get("layers", {}).get("detections", "")
        if detections:
            items = [d.strip() for d in detections.split(",") if d.strip()]
            if items:
                obj = random.choice(items)
                ts = frame.get("timestamp", 0)
                questions.append({
                    "question": f"When did I last see the {obj}?",
                    "answer": f"at timestamp {ts:.0f}",
                    "category": "temporal",
                    "tools_expected": ["temporal_query", "semantic_search"],
                })

    return questions


def _cross_layer_questions(
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate cross-layer questions requiring information from different layers.

    :param trajectory: Trajectory frames with multi-layer observations.
    :returns: List of question dicts.
    """
    questions: List[Dict[str, Any]] = []

    if not trajectory:
        return questions

    # "What kind of place is the area near (x, y)?"
    for frame in random.sample(trajectory, min(4, len(trajectory))):
        pos = frame.get("position", [0, 0, 0])
        place = frame.get("layers", {}).get("place", "")
        if place:
            questions.append({
                "question": f"What kind of place is the area near ({pos[0]:.1f}, {pos[1]:.1f})?",
                "answer": place,
                "category": "cross_layer",
                "tools_expected": ["get_current_context", "semantic_search"],
            })

    # "What objects were detected in the X?"
    places: Dict[str, List[str]] = {}
    for frame in trajectory:
        place = frame.get("layers", {}).get("place", "").strip().lower()
        detections = frame.get("layers", {}).get("detections", "")
        if place and detections:
            places.setdefault(place, []).append(detections)

    for place, det_lists in list(places.items())[:3]:
        all_items = set()
        for det in det_lists:
            all_items.update(d.strip() for d in det.split(",") if d.strip())
        if all_items:
            questions.append({
                "question": f"What objects were detected in the {place}?",
                "answer": ", ".join(sorted(all_items)[:8]),
                "category": "cross_layer",
                "tools_expected": ["semantic_search"],
            })

    # "Describe the X area"
    for place in list(places.keys())[:2]:
        vlm_descs = []
        for frame in trajectory:
            if frame.get("layers", {}).get("place", "").strip().lower() == place:
                vlm = frame.get("layers", {}).get("vlm", "")
                if vlm:
                    vlm_descs.append(vlm)
        if vlm_descs:
            questions.append({
                "question": f"Describe the {place} area.",
                "answer": " ".join(vlm_descs[:3]),
                "category": "cross_layer",
                "tools_expected": ["semantic_search", "recall"],
            })

    return questions


def _entity_questions(
    objects: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate entity tracking questions.

    :param objects: Scene objects with metadata.
    :param trajectory: Trajectory frames.
    :returns: List of question dicts.
    """
    questions: List[Dict[str, Any]] = []

    if not trajectory:
        return questions

    # Count how many times each object type appears in detections
    obj_counts: Dict[str, int] = {}
    for frame in trajectory:
        detections = frame.get("layers", {}).get("detections", "")
        for item in detections.split(","):
            item = item.strip().lower()
            if item:
                obj_counts[item] = obj_counts.get(item, 0) + 1

    # "How many times have I seen the X?"
    frequent = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)
    for obj_name, count in frequent[:4]:
        questions.append({
            "question": f"How many times have I seen the {obj_name}?",
            "answer": f"{count} times",
            "category": "entity",
            "tools_expected": ["entity_query", "semantic_search"],
        })

    # "What objects appear together near X?"
    for frame in random.sample(trajectory, min(3, len(trajectory))):
        detections = frame.get("layers", {}).get("detections", "")
        items = [d.strip() for d in detections.split(",") if d.strip()]
        if len(items) >= 2:
            anchor = items[0]
            cooccur = ", ".join(items[1:4])
            questions.append({
                "question": f"What objects appear together with the {anchor}?",
                "answer": cooccur,
                "category": "entity",
                "tools_expected": ["entity_query", "semantic_search"],
            })

    return questions


def _interoception_questions(
    interoception: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate interoception questions.

    :param interoception: Body state data.
    :param trajectory: Trajectory frames.
    :returns: List of question dicts.
    """
    questions: List[Dict[str, Any]] = []

    if not interoception:
        return questions

    # "What is my current battery level?"
    latest = interoception[-1]
    battery = latest.get("battery", "")
    if battery:
        questions.append({
            "question": "What is my current battery level?",
            "answer": battery,
            "category": "interoception",
            "tools_expected": ["body_status"],
        })

    # "What was my CPU temperature when I was in the kitchen?"
    for entry in interoception:
        ts = entry.get("timestamp", 0)
        cpu = entry.get("cpu_temp", "")
        if not cpu:
            continue
        place = _find_place_at_time(trajectory, ts)
        if place:
            questions.append({
                "question": f"What was my CPU temperature when I was in the {place}?",
                "answer": cpu,
                "category": "interoception",
                "tools_expected": ["body_status"],
            })
            break  # One is enough

    return questions


def _episodic_questions(
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate episodic / consolidation questions.

    :param trajectory: Trajectory frames.
    :returns: List of question dicts.
    """
    questions: List[Dict[str, Any]] = []

    if not trajectory:
        return questions

    # "Summarize my exploration"
    places_visited = set()
    for frame in trajectory:
        place = frame.get("layers", {}).get("place", "").strip()
        if place:
            places_visited.add(place.lower())

    if places_visited:
        questions.append({
            "question": "Summarize what I explored.",
            "answer": f"Explored: {', '.join(sorted(places_visited))}",
            "category": "episodic",
            "tools_expected": ["episode_summary", "search_gists"],
        })

    # "Tell me everything about X"
    for place in list(places_visited)[:2]:
        vlm_descs = []
        for frame in trajectory:
            if frame.get("layers", {}).get("place", "").strip().lower() == place:
                vlm = frame.get("layers", {}).get("vlm", "")
                if vlm:
                    vlm_descs.append(vlm)
        if vlm_descs:
            questions.append({
                "question": f"Tell me everything about the {place}.",
                "answer": " ".join(vlm_descs[:3]),
                "category": "episodic",
                "tools_expected": ["recall", "search_gists"],
            })

    # "What did I do in my last episode?"
    questions.append({
        "question": "What did I do in my last episode?",
        "answer": f"Explored areas including: {', '.join(sorted(places_visited)[:5])}",
        "category": "episodic",
        "tools_expected": ["episode_summary"],
    })

    return questions


# ── Helpers ──────────────────────────────────────────────────────────

def _format_object_name(object_type: str) -> str:
    """Convert CamelCase AI2-THOR object type to readable name.

    :param object_type: e.g. ``"CoffeeMachine"``
    :returns: e.g. ``"coffee machine"``
    """
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", object_type)
    return name.lower()


def _find_place_near(
    trajectory: List[Dict[str, Any]],
    x: float,
    y: float,
    radius: float = 3.0,
) -> str:
    """Find the place label from the nearest trajectory frame.

    :returns: Place name, or empty string if none found.
    """
    best_dist = float("inf")
    best_place = ""
    for frame in trajectory:
        pos = frame.get("position", [0, 0, 0])
        dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
        if dist < best_dist and dist <= radius:
            place = frame.get("layers", {}).get("place", "").strip()
            if place:
                best_dist = dist
                best_place = place
    return best_place


def _find_place_at_time(
    trajectory: List[Dict[str, Any]],
    timestamp: float,
) -> str:
    """Find the place label from the frame closest in time.

    :returns: Place name, or empty string if none found.
    """
    if not trajectory:
        return ""
    closest = min(trajectory, key=lambda f: abs(f.get("timestamp", 0) - timestamp))
    return closest.get("layers", {}).get("place", "").strip().lower()


def _objects_near(
    objects: List[Dict[str, Any]],
    x: float,
    y: float,
    radius: float = 2.0,
) -> List[Dict[str, Any]]:
    """Find objects within radius of (x, y).

    :returns: List of nearby objects sorted by distance.
    """
    nearby = []
    for obj in objects:
        pos = obj.get("position", [0, 0, 0])
        dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
        if dist <= radius:
            nearby.append((dist, obj))
    nearby.sort(key=lambda pair: pair[0])
    return [obj for _, obj in nearby]


def generate_questions_for_sample(
    sample: Dict[str, Any],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate all question categories for a single sample.

    :param sample: Sample dict with trajectory, interoception, and metadata.
    :param seed: Random seed for reproducible question selection.
    :returns: List of question dicts with IDs assigned.
    """
    random.seed(seed)
    trajectory = sample.get("trajectory", [])
    interoception = sample.get("interoception", [])
    objects = sample.get("metadata", {}).get("scene_objects", [])

    all_questions: List[Dict[str, Any]] = []
    all_questions.extend(_spatial_questions(objects, trajectory))
    all_questions.extend(_temporal_questions(trajectory))
    all_questions.extend(_cross_layer_questions(trajectory))
    all_questions.extend(_entity_questions(objects, trajectory))
    all_questions.extend(_interoception_questions(interoception, trajectory))
    all_questions.extend(_episodic_questions(trajectory))

    # Assign question IDs
    for i, q in enumerate(all_questions):
        q["question_id"] = f"q{i:03d}"

    return all_questions


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for question generation.

    :param argv: Command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate eMEM-Bench questions from trajectory data",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory with scene subdirectories containing trajectory.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the benchmark index JSON",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load existing index if present
    index: List[Dict[str, Any]] = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            index = json.load(f)
        if isinstance(index, dict):
            index = index.get("samples", [])

    # Process each scene directory
    total_questions = 0
    for scene_dir in sorted(os.listdir(args.data_dir)):
        traj_path = os.path.join(args.data_dir, scene_dir, "trajectory.json")
        if not os.path.exists(traj_path):
            continue

        with open(traj_path) as f:
            sample = json.load(f)

        questions = generate_questions_for_sample(sample)
        sample["questions"] = questions
        total_questions += len(questions)

        # Write updated trajectory back
        with open(traj_path, "w") as f:
            json.dump(sample, f, indent=2)

        # Update index entry
        sample_id = sample.get("sample_id", scene_dir)
        found = False
        for entry in index:
            if entry.get("sample_id") == sample_id:
                entry["n_questions"] = len(questions)
                found = True
                break
        if not found:
            # Make trajectory_path relative to the index file's directory
            output_dir = os.path.dirname(os.path.abspath(args.output))
            rel_traj = os.path.relpath(
                os.path.abspath(traj_path), output_dir,
            )
            index.append({
                "sample_id": sample_id,
                "scene_id": sample.get("scene_id", scene_dir),
                "source": sample.get("source", "ai2thor"),
                "trajectory_path": rel_traj,
                "n_questions": len(questions),
            })

        log.info("Scene %s: %d questions generated", scene_dir, len(questions))

    # Write index
    with open(args.output, "w") as f:
        json.dump(index, f, indent=2)

    log.info("Total: %d questions across %d scenes. Index: %s",
             total_questions, len(index), args.output)


if __name__ == "__main__":
    main()
