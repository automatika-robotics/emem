"""AI2-THOR data collection for eMEM-Bench.

Runs teleport exploration of AI2-THOR scenes, captures multi-layer VLM
captions, generates synthetic interoception data, and saves trajectories
in eMEM-Bench format.

Usage::

    python -m harness.benchmarks.emem_bench.collect_ai2thor \\
        --scenes FloorPlan1,FloorPlan2 \\
        --vlm-model gemma3:4b \\
        --output data/emem-bench/ai2thor/
"""

import argparse
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from harness.benchmarks.academic.caption_cache import CaptionCache
from harness.benchmarks.emem_bench.generate_questions import (
    _is_valid_caption,
    _is_valid_place,
)
from harness.environments.ai2thor_adapter import AI2ThorAdapter

log = logging.getLogger(__name__)

# Multi-layer VLM prompts
LAYER_PROMPTS: Dict[str, str] = {
    "vlm": (
        "Describe what you see in this image in one or two sentences. "
        "Focus on objects, their spatial arrangement, and any notable features."
    ),
    "detections": (
        "List all distinct objects visible in this image, separated by commas. "
        "Be specific (e.g. 'red mug' not just 'mug'). Only list objects, nothing else."
    ),
    "place": (
        "What kind of room or area is this? Answer with just the room/area name "
        "(e.g. 'kitchen', 'living room', 'hallway', 'bathroom', 'bedroom')."
    ),
}

# Default scenes covering different room types
DEFAULT_SCENES: List[str] = [
    # Kitchens (FloorPlan 1-30)
    "FloorPlan1",
    "FloorPlan2",
    "FloorPlan3",
    "FloorPlan5",
    "FloorPlan7",
    "FloorPlan10",
    "FloorPlan14",
    "FloorPlan18",
    "FloorPlan22",
    "FloorPlan26",
    # Living rooms (FloorPlan 201-230)
    "FloorPlan201",
    "FloorPlan202",
    "FloorPlan205",
    "FloorPlan208",
    "FloorPlan210",
    "FloorPlan215",
    "FloorPlan218",
    "FloorPlan222",
    "FloorPlan225",
    "FloorPlan228",
    # Bedrooms (FloorPlan 301-330)
    "FloorPlan301",
    "FloorPlan302",
    "FloorPlan305",
    "FloorPlan308",
    "FloorPlan310",
    "FloorPlan315",
    "FloorPlan318",
    "FloorPlan322",
    "FloorPlan325",
    "FloorPlan328",
]


def _make_vlm(model: str, base_url: str) -> Any:
    """Create a VLM client.

    :param model: Ollama model name.
    :param base_url: Ollama server URL.
    :returns: VLM client with ``describe()`` method.
    """
    from harness.providers.ollama_vlm import OllamaVLM

    return OllamaVLM(model=model, base_url=base_url)


def _generate_interoception(
    timestamps: List[float],
) -> List[Dict[str, Any]]:
    """Generate synthetic interoception data over the trajectory.

    Simulates battery drain and CPU temperature fluctuation.

    :param timestamps: Trajectory timestamps to sample from.
    :returns: List of interoception entries.
    """
    if not timestamps:
        return []

    entries: List[Dict[str, Any]] = []
    # Sample at ~10 evenly spaced points
    n_samples = min(10, len(timestamps))
    indices = np.linspace(0, len(timestamps) - 1, n_samples, dtype=int)

    battery_start = random.uniform(85, 100)
    battery_drain_rate = random.uniform(0.5, 2.0)  # % per minute

    for idx in indices:
        ts = timestamps[idx]
        elapsed_minutes = (ts - timestamps[0]) / 60.0

        battery = max(5.0, battery_start - battery_drain_rate * elapsed_minutes)
        cpu_temp = 55.0 + random.gauss(0, 5) + elapsed_minutes * 0.1

        entries.append({
            "timestamp": ts,
            "battery": f"battery: {battery:.0f}%",
            "cpu_temp": f"cpu_temp: {cpu_temp:.0f}C",
        })

    return entries


def _get_scene_objects(controller: Any) -> List[Dict[str, Any]]:
    """Extract object metadata from the AI2-THOR scene.

    :param controller: AI2-THOR controller with an active scene.
    :returns: List of object dicts with name, type, position, etc.
    """
    objects = []
    for obj in controller.last_event.metadata["objects"]:
        pos = obj["position"]
        objects.append({
            "objectId": obj["objectId"],
            "objectType": obj["objectType"],
            "name": obj.get("name", obj["objectType"]),
            "position": [pos["x"], pos["z"], pos.get("y", 0.0)],
            "visible": obj.get("visible", False),
            "pickupable": obj.get("pickupable", False),
            "receptacle": obj.get("receptacle", False),
            "parentReceptacles": obj.get("parentReceptacles", []),
        })
    return objects


def collect_scene(
    scene: str,
    vlm: Any,
    cache: CaptionCache,
    output_dir: str,
    max_waypoints: Optional[int] = None,
    headless: bool = True,
    save_frames: bool = False,
    vlm_model: str = "",
) -> Dict[str, Any]:
    """Collect trajectory data from a single AI2-THOR scene.

    :param scene: Scene name (e.g. ``"FloorPlan1"``).
    :param vlm: VLM client for captioning.
    :param cache: Caption cache to avoid re-captioning.
    :param output_dir: Directory to save this scene's data.
    :param max_waypoints: Maximum teleport waypoints.
    :param headless: Use headless rendering.
    :param save_frames: Save RGB frames to disk.
    :param vlm_model: VLM model name (for cache keys).
    :returns: Sample dict in eMEM-Bench format.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    log.info("Collecting scene: %s", scene)

    env = AI2ThorAdapter(
        scene=scene,
        exploration_mode="teleport",
        max_waypoints=max_waypoints,
        headless=headless,
        rotations_per_waypoint=4,
    )

    try:
        frame, pos = env.reset()
        # Extract scene objects for question generation.
        # Accesses private _controller since AI2ThorAdapter doesn't expose
        # scene metadata — acceptable for offline data collection.
        scene_objects = _get_scene_objects(env._controller)

        trajectory: List[Dict[str, Any]] = []
        timestamps: List[float] = []
        base_time = time.time()
        step_idx = 0

        # Process initial frame
        done = False
        while True:
            frame_id = f"frame_{step_idx:04d}"
            timestamp = base_time + step_idx * 2.0  # 2s between frames
            timestamps.append(timestamp)

            # Save frame if requested
            image_path = None
            if save_frames:
                image_path = f"frames/{frame_id}.jpg"
                full_path = os.path.join(output_dir, image_path)
                _save_frame(frame, full_path)

            # Multi-layer VLM captioning
            layers: Dict[str, str] = {}
            for layer_name, prompt in LAYER_PROMPTS.items():
                # Use cache key based on scene + position + rotation + prompt
                cache_key = f"{scene}_{frame_id}"
                cached = cache.get(cache_key, prompt, vlm_model)
                if cached is not None:
                    caption = cached
                else:
                    caption = vlm.describe(frame, prompt)
                    cache.put(cache_key, prompt, vlm_model, caption)

                # Filter invalid captions at collection time
                if layer_name == "place" and not _is_valid_place(caption):
                    caption = ""
                elif layer_name in ("vlm", "detections") and not _is_valid_caption(
                    caption
                ):
                    caption = ""

                layers[layer_name] = caption

            waypoint: Dict[str, Any] = {
                "frame_id": frame_id,
                "position": [pos[0], pos[1], 0.0],
                "timestamp": timestamp,
                "layers": layers,
            }
            if image_path:
                waypoint["image_path"] = image_path

            trajectory.append(waypoint)
            step_idx += 1

            if done:
                break

            # Step
            frame, pos, _, done, _ = env.step(0)

        interoception = _generate_interoception(timestamps)

        sample = {
            "sample_id": f"thor_{scene.lower()}",
            "scene_id": scene,
            "source": "ai2thor",
            "trajectory": trajectory,
            "interoception": interoception,
            "questions": [],  # Generated separately
            "metadata": {
                "n_waypoints": len(trajectory),
                "scene_objects": scene_objects,
            },
        }

        # Save trajectory
        traj_path = os.path.join(output_dir, "trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(sample, f, indent=2)

        log.info(
            "Scene %s: %d frames, %d objects",
            scene,
            len(trajectory),
            len(scene_objects),
        )
        return sample

    finally:
        env.close()


def _save_frame(frame: np.ndarray, path: str) -> None:
    """Save an RGB frame as JPEG.

    :param frame: ``(H, W, 3)`` uint8 array.
    :param path: Output file path.
    """
    from PIL import Image

    img = Image.fromarray(frame)
    img.save(path, quality=85)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for AI2-THOR data collection.

    :param argv: Command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect AI2-THOR data for eMEM-Bench",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene names (default: 30 diverse scenes)",
    )
    parser.add_argument("--vlm-model", default="qwen3.5:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--output", required=True, help="Output directory for scene data"
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Path for benchmark index JSON (default: <output>/../emem-bench-v0.json)",
    )
    parser.add_argument("--max-waypoints", type=int, default=None)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--no-headless", dest="headless", action="store_false", default=True
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    scenes = args.scenes.split(",") if args.scenes else DEFAULT_SCENES

    vlm = _make_vlm(args.vlm_model, args.ollama_url)
    cache_path = os.path.join(args.output, "caption_cache.jsonl")
    cache = CaptionCache(cache_path)

    all_samples: List[Dict[str, Any]] = []

    for scene in scenes:
        scene = scene.strip()
        scene_dir = os.path.join(args.output, scene.lower())
        try:
            sample = collect_scene(
                scene=scene,
                vlm=vlm,
                cache=cache,
                output_dir=scene_dir,
                max_waypoints=args.max_waypoints,
                headless=args.headless,
                save_frames=args.save_frames,
                vlm_model=args.vlm_model,
            )
            all_samples.append({
                "sample_id": sample["sample_id"],
                "scene_id": sample["scene_id"],
                "source": "ai2thor",
                "trajectory_path": f"{scene.lower()}/trajectory.json",
            })
        except Exception:
            log.exception("Failed to collect scene %s", scene)

    # Write index file
    if args.index:
        index_path = args.index
    else:
        index_path = os.path.normpath(
            os.path.join(args.output, "..", "emem-bench-v0.json")
        )

    # Make trajectory_path entries relative to the index file's directory
    index_dir = os.path.dirname(os.path.abspath(index_path))
    output_abs = os.path.abspath(args.output)
    for s in all_samples:
        abs_traj = os.path.join(output_abs, s["trajectory_path"])
        s["trajectory_path"] = os.path.relpath(abs_traj, index_dir)

    existing: List[Dict[str, Any]] = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            existing = existing.get("samples", [])

    # Merge: update existing entries or add new ones
    existing_ids = {s["sample_id"] for s in existing}
    for s in all_samples:
        if s["sample_id"] not in existing_ids:
            existing.append(s)
        else:
            for i, ex in enumerate(existing):
                if ex["sample_id"] == s["sample_id"]:
                    existing[i] = s
                    break

    with open(index_path, "w") as f:
        json.dump(existing, f, indent=2)

    log.info("Collected %d scenes. Index: %s", len(all_samples), index_path)


if __name__ == "__main__":
    main()
