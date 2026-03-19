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
from collections import Counter
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)


# ── Filtering utilities (Step 1) ─────────────────────────────────────

VALID_PLACES: Set[str] = {
    "kitchen",
    "living room",
    "livingroom",
    "bedroom",
    "bathroom",
    "hallway",
    "corridor",
    "dining room",
    "office",
    "laundry room",
    "garage",
    "closet",
    "pantry",
    "foyer",
    "entryway",
    "basement",
    "attic",
}

STRUCTURAL_OBJECTS: Set[str] = {
    "Wall",
    "Floor",
    "Ceiling",
    "Doorframe",
    "Door",
    "Window",
    "Doorway",
    "StandardWallSize",
    "LightSwitch",
}

_GARBAGE_CAPTION_RE = re.compile(
    r"(?i)"
    r"(?:^image\s+of\b)"
    r"|(?:^a\s+photo\s+of\b)"
    r"|(?:^this\s+is\s+a\s+picture\b)"
    r"|(?:^[\d\s\.\,]+$)"
    r"|(?:^n/?a$)"
    r"|(?:^none$)"
    r"|(?:^unknown$)"
    r"|(?:^null$)"
    r"|(?:^i'm sorry\b)"
    r"|(?:i cannot\b)"
    r"|(?:i can't\b)"
    r"|(?:^sorry,)"
    r"|(?:^as an ai\b)"
    r"|(?:i don't see\b)"
    r"|(?:i am unable\b)"
    r"|(?:^no image\b)"
    r"|(?:uniform\s+\w+\s+surface)"
    r"|(?:no discernible)"
    r"|(?:the image you uploaded)"
    r"|(?:this image displays a uniform)"
    r"|(?:solid color)"
    r"|(?:single color)"
    r"|(?:^blank\b)"
)


def _is_valid_caption(text: str) -> bool:
    """Return True if *text* is a usable VLM caption.

    Rejects empty strings, strings shorter than 10 characters, and captions
    matching common garbage patterns.
    """
    if not text or len(text.strip()) < 10:
        return False
    return _GARBAGE_CAPTION_RE.search(text.strip()) is None


def _is_valid_place(place: str) -> bool:
    """Return True if *place* is a recognised room name."""
    return place.strip().lower() in VALID_PLACES


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

    Only returns places that pass ``_is_valid_place``.

    :returns: Place name (lowercased), or empty string if none found.
    """
    best_dist = float("inf")
    best_place = ""
    for frame in trajectory:
        pos = frame.get("position", [0, 0, 0])
        dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
        if dist < best_dist and dist <= radius:
            place = frame.get("layers", {}).get("place", "").strip()
            if place and _is_valid_place(place):
                best_dist = dist
                best_place = place.strip().lower()
    return best_place


def _find_place_at_time(
    trajectory: List[Dict[str, Any]],
    timestamp: float,
) -> str:
    """Find the place label from the frame closest in time.

    Only returns places that pass ``_is_valid_place``.

    :returns: Place name (lowercased), or empty string if none found.
    """
    if not trajectory:
        return ""
    closest = min(
        trajectory, key=lambda f: abs(f.get("timestamp", 0) - timestamp),
    )
    place = closest.get("layers", {}).get("place", "").strip().lower()
    if _is_valid_place(place):
        return place
    return ""


def _detections_near(
    trajectory: List[Dict[str, Any]],
    x: float,
    y: float,
    radius: float = 2.0,
) -> List[str]:
    """Collect detection-layer items from trajectory frames near *(x, y)*.

    Returns a deduplicated list of detected object names sorted
    alphabetically.
    """
    items: Set[str] = set()
    for frame in trajectory:
        pos = frame.get("position", [0, 0, 0])
        dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
        if dist <= radius:
            detections = frame.get("layers", {}).get("detections", "")
            for d in detections.split(","):
                d = d.strip()
                if d:
                    items.add(d)
    return sorted(items)


def _trajectory_position_label(index: int, total: int) -> str:
    """Human-readable label for where *index* falls in the trajectory."""
    if total <= 0:
        return "During exploration"
    ratio = index / total
    if ratio < 0.25:
        return "Near the beginning of exploration"
    if ratio < 0.65:
        return "Around the middle of exploration"
    return "Near the end of exploration"


def _top_detection_items(
    trajectory: List[Dict[str, Any]],
    place: str,
    top_n: int = 8,
) -> List[str]:
    """Return the *top_n* most frequently mentioned detections at *place*."""
    counter: Counter = Counter()
    for frame in trajectory:
        fp = frame.get("layers", {}).get("place", "").strip().lower()
        if fp != place:
            continue
        detections = frame.get("layers", {}).get("detections", "")
        for d in detections.split(","):
            d = d.strip()
            if d:
                counter[d] += 1
    return [item for item, _ in counter.most_common(top_n)]


def _shortest_valid_caption(
    trajectory: List[Dict[str, Any]],
    place: str,
) -> str:
    """Return the shortest valid VLM caption observed at *place*."""
    best = ""
    for frame in trajectory:
        fp = frame.get("layers", {}).get("place", "").strip().lower()
        if fp != place:
            continue
        vlm = frame.get("layers", {}).get("vlm", "")
        if _is_valid_caption(vlm):
            if not best or len(vlm) < len(best):
                best = vlm
    return best


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

    # Filter out structural objects
    objects = [
        o for o in objects
        if o.get("objectType") not in STRUCTURAL_OBJECTS
    ]

    # "Where is X?" questions
    visible_objects = [o for o in objects if o.get("visible")]
    if not visible_objects:
        visible_objects = objects[:20]

    for obj in random.sample(visible_objects, min(8, len(visible_objects))):
        pos = obj.get("position", [0, 0, 0])
        obj_name = _format_object_name(obj["objectType"])

        place = _find_place_near(trajectory, pos[0], pos[1])
        answer = (
            f"{place} near ({pos[0]:.1f}, {pos[1]:.1f})"
            if place
            else f"near ({pos[0]:.1f}, {pos[1]:.1f})"
        )

        questions.append({
            "question": f"Where is the {obj_name}?",
            "answer": answer,
            "category": "spatial",
            "tools_expected": ["locate", "semantic_search"],
        })

    # "What objects are near (x, y)?" — use detections layer
    if trajectory:
        for _ in range(min(4, len(trajectory))):
            wp = random.choice(trajectory)
            pos = wp.get("position", [0, 0, 0])
            nearby = _detections_near(trajectory, pos[0], pos[1], radius=2.0)
            if nearby:
                obj_names = ", ".join(nearby[:5])
                questions.append({
                    "question": (
                        f"What objects are near "
                        f"({pos[0]:.1f}, {pos[1]:.1f})?"
                    ),
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
    if _is_valid_caption(last_vlm):
        questions.append({
            "question": "What did I see most recently?",
            "answer": last_vlm,
            "category": "temporal",
            "tools_expected": ["temporal_query"],
        })

    # "What did I see first?"
    first_frame = trajectory[0]
    first_vlm = first_frame.get("layers", {}).get("vlm", "")
    if _is_valid_caption(first_vlm):
        questions.append({
            "question": (
                "What did I observe at the beginning of my exploration?"
            ),
            "answer": first_vlm,
            "category": "temporal",
            "tools_expected": ["temporal_query"],
        })

    # "When did I last see X?" — iterate trajectory in REVERSE
    sampled_items_seen: Set[str] = set()
    for frame in random.sample(trajectory, min(5, len(trajectory))):
        detections = frame.get("layers", {}).get("detections", "")
        items = [d.strip() for d in detections.split(",") if d.strip()]
        if not items:
            continue
        obj = random.choice(items)
        if obj in sampled_items_seen:
            continue
        sampled_items_seen.add(obj)

        # Walk trajectory in reverse to find the actual last occurrence
        last_idx = None
        last_frame_found = None
        for idx in range(len(trajectory) - 1, -1, -1):
            det = trajectory[idx].get("layers", {}).get("detections", "")
            if obj in [d.strip() for d in det.split(",")]:
                last_idx = idx
                last_frame_found = trajectory[idx]
                break

        if last_frame_found is not None and last_idx is not None:
            pos = last_frame_found.get("position", [0, 0, 0])
            place = _find_place_near(trajectory, pos[0], pos[1])
            pos_label = _trajectory_position_label(
                last_idx, len(trajectory),
            )
            if place:
                answer = (
                    f"{pos_label}, at ({pos[0]:.1f}, {pos[1]:.1f}) "
                    f"in the {place}"
                )
            else:
                answer = (
                    f"{pos_label}, at ({pos[0]:.1f}, {pos[1]:.1f})"
                )
            questions.append({
                "question": f"When did I last see the {obj}?",
                "answer": answer,
                "category": "temporal",
                "tools_expected": ["temporal_query", "semantic_search"],
            })

        if len(sampled_items_seen) >= 3:
            break

    return questions


def _cross_layer_questions(
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate cross-layer questions requiring information from multiple
    layers.

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
        if place and _is_valid_place(place):
            questions.append({
                "question": (
                    f"What kind of place is the area near "
                    f"({pos[0]:.1f}, {pos[1]:.1f})?"
                ),
                "answer": place.strip().lower(),
                "category": "cross_layer",
                "tools_expected": ["get_current_context", "semantic_search"],
            })

    # "What objects were detected in the X?" — valid places only
    places: Dict[str, List[str]] = {}
    for frame in trajectory:
        place = frame.get("layers", {}).get("place", "").strip().lower()
        detections = frame.get("layers", {}).get("detections", "")
        if place and _is_valid_place(place) and detections:
            places.setdefault(place, []).append(detections)

    for place, det_lists in list(places.items())[:3]:
        all_items: Set[str] = set()
        for det in det_lists:
            all_items.update(
                d.strip() for d in det.split(",") if d.strip()
            )
        if all_items:
            questions.append({
                "question": (
                    f"What objects were detected in the {place}?"
                ),
                "answer": ", ".join(sorted(all_items)[:8]),
                "category": "cross_layer",
                "tools_expected": ["semantic_search"],
            })

    # "Describe the X area" — top objects + shortest valid caption
    for place in list(places.keys())[:2]:
        top_objs = _top_detection_items(trajectory, place, top_n=8)
        caption = _shortest_valid_caption(trajectory, place)
        if top_objs or caption:
            obj_str = ", ".join(top_objs[:8]) if top_objs else ""
            if obj_str and caption:
                answer = f"{place} area with {obj_str}. {caption}"
            elif obj_str:
                answer = f"{place} area with {obj_str}."
            else:
                answer = f"{place} area. {caption}"
            questions.append({
                "question": f"Describe the {place} area.",
                "answer": answer,
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

    # Build per-object stats from detections layer
    obj_counts: Counter = Counter()
    obj_first_pos: Dict[str, Dict[str, Any]] = {}
    obj_places: Dict[str, Set[str]] = {}

    for frame in trajectory:
        detections = frame.get("layers", {}).get("detections", "")
        pos = frame.get("position", [0, 0, 0])
        place = frame.get("layers", {}).get("place", "").strip().lower()
        for item in detections.split(","):
            item = item.strip().lower()
            if not item:
                continue
            obj_counts[item] += 1
            if item not in obj_first_pos:
                obj_first_pos[item] = {"position": pos, "place": place}
            if place and _is_valid_place(place):
                obj_places.setdefault(item, set()).add(place)

    frequent = obj_counts.most_common()

    # "Have you seen X during your exploration?"
    for obj_name, _count in frequent[:3]:
        info = obj_first_pos.get(obj_name, {})
        pos = info.get("position", [0, 0, 0])
        place = _find_place_near(trajectory, pos[0], pos[1])
        if place:
            answer = (
                f"Yes, near ({pos[0]:.1f}, {pos[1]:.1f}) in the {place}"
            )
        else:
            answer = f"Yes, near ({pos[0]:.1f}, {pos[1]:.1f})"
        questions.append({
            "question": (
                f"Have you seen {obj_name} during your exploration?"
            ),
            "answer": answer,
            "category": "entity",
            "tools_expected": ["entity_query", "semantic_search"],
        })

    # "Where have you seen X?" — list distinct places
    for obj_name, _count in frequent[:3]:
        pl = obj_places.get(obj_name, set())
        if pl:
            questions.append({
                "question": f"Where have you seen {obj_name}?",
                "answer": ", ".join(sorted(pl)),
                "category": "entity",
                "tools_expected": ["entity_query", "semantic_search"],
            })

    # "Is X a common object in the scene?"
    if frequent:
        samples = []
        if len(frequent) >= 1:
            samples.append(frequent[0])
        if len(frequent) >= 2:
            samples.append(frequent[1])
        if len(frequent) >= 3:
            samples.append(frequent[-1])  # least frequent
        total_frames = len(trajectory)
        for obj_name, count in samples:
            freq_ratio = count / total_frames if total_frames > 0 else 0
            if freq_ratio > 0.15:
                answer = "Yes, frequently"
            else:
                answer = "No, rarely"
            questions.append({
                "question": (
                    f"Is {obj_name} a common object in the scene?"
                ),
                "answer": answer,
                "category": "entity",
                "tools_expected": ["entity_query", "semantic_search"],
            })

    # "What objects appear together with X?"
    for frame in random.sample(trajectory, min(3, len(trajectory))):
        detections = frame.get("layers", {}).get("detections", "")
        items = [d.strip() for d in detections.split(",") if d.strip()]
        if len(items) >= 2:
            anchor = items[0]
            cooccur = ", ".join(items[1:4])
            questions.append({
                "question": (
                    f"What objects appear together with the {anchor}?"
                ),
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
                "question": (
                    f"What was my CPU temperature when I was in "
                    f"the {place}?"
                ),
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

    # "Summarize my exploration" — filter valid places
    places_visited: Set[str] = set()
    for frame in trajectory:
        place = frame.get("layers", {}).get("place", "").strip()
        if place and _is_valid_place(place):
            places_visited.add(place.lower())

    if places_visited:
        questions.append({
            "question": "Summarize what I explored.",
            "answer": (
                f"Explored: {', '.join(sorted(places_visited))}"
            ),
            "category": "episodic",
            "tools_expected": ["episode_summary", "search_gists"],
        })

    # "Tell me everything about X" — top objects + shortest valid caption
    for place in sorted(places_visited)[:2]:
        top_objs = _top_detection_items(trajectory, place, top_n=8)
        caption = _shortest_valid_caption(trajectory, place)
        if top_objs or caption:
            obj_str = ", ".join(top_objs[:8]) if top_objs else ""
            if obj_str and caption:
                answer = f"{place} area with {obj_str}. {caption}"
            elif obj_str:
                answer = f"{place} area with {obj_str}."
            elif caption:
                answer = f"{place} area. {caption}"
            else:
                continue
            questions.append({
                "question": f"Tell me everything about the {place}.",
                "answer": answer,
                "category": "episodic",
                "tools_expected": ["recall", "search_gists"],
            })

    # "What did I do in my last episode?"
    if places_visited:
        questions.append({
            "question": "What did I do in my last episode?",
            "answer": (
                "Explored areas including: "
                f"{', '.join(sorted(places_visited)[:5])}"
            ),
            "category": "episodic",
            "tools_expected": ["episode_summary"],
        })

    return questions


# ── Top-level generation ─────────────────────────────────────────────

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

    # Deduplicate by question text
    seen_texts: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for q in all_questions:
        text = q["question"]
        if text not in seen_texts:
            seen_texts.add(text)
            deduped.append(q)

    # Assign question IDs
    for i, q in enumerate(deduped):
        q["question_id"] = f"q{i:03d}"

    return deduped


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
        traj_path = os.path.join(
            args.data_dir, scene_dir, "trajectory.json",
        )
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

        log.info(
            "Scene %s: %d questions generated", scene_dir, len(questions),
        )

    # Write index
    with open(args.output, "w") as f:
        json.dump(index, f, indent=2)

    log.info(
        "Total: %d questions across %d scenes. Index: %s",
        total_questions, len(index), args.output,
    )


if __name__ == "__main__":
    main()
