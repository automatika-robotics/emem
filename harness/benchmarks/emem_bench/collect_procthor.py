"""ProcTHOR multi-room scene collection for eMEM-Bench v1.

Samples ``--n-houses`` houses from ProcTHOR-10K that satisfy a
``min_rooms``/``max_rooms`` range, teleport-explores each with
multi-layer VLM captioning, and emits:

  - per-house trajectories at ``<output>/house_<k>/trajectory.json``
  - a newline-delimited manifest at ``<output>/scenes.jsonl`` with
    ``room_types_present``, ``room_count``, ``similarity_pair_id``,
    and per-waypoint room tagging.

The AI2-THOR controller accepts ProcTHOR house dicts natively, so
the underlying :class:`AI2ThorAdapter` is reused as-is. Captions
are served via the shared :class:`CaptionCache` — re-running is free
for already-seen (position, prompt, model) triples.

Usage::

    python -m harness.benchmarks.emem_bench.collect_procthor \\
        --output data/emem-bench-v1/ \\
        --n-houses 20 --n-similarity-pairs 4 \\
        --vlm-model qwen3.5:27b --max-waypoints 15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from harness.benchmarks.academic.caption_cache import CaptionCache
from harness.benchmarks.emem_bench.collect_ai2thor import (
    LAYER_PROMPTS,
    _generate_interoception,
    _get_scene_objects,
    _make_vlm,
    _save_frame,
)
from harness.benchmarks.emem_bench.generate_questions import (
    _is_valid_caption,
    _is_valid_place,
)
from harness.environments.ai2thor_adapter import AI2ThorAdapter
from harness.environments.procthor_utils import (
    assign_similarity_pairs,
    house_metadata,
    room_at_position,
    select_houses,
)

log = logging.getLogger(__name__)


def collect_house(
    house_idx: int,
    dataset_idx: int,
    house: Dict[str, Any],
    similarity_pair_id: Optional[str],
    vlm: Any,
    cache: CaptionCache,
    output_dir: str,
    max_waypoints: Optional[int],
    rotations_per_waypoint: int,
    headless: bool,
    save_frames: bool,
    vlm_model: str,
) -> Dict[str, Any]:
    """Collect trajectory data from a single ProcTHOR house.

    :param house_idx: Index within the selected set (0..n_houses-1);
        used as the sample-id suffix.
    :param dataset_idx: Index into the ProcTHOR-10K split; recorded
        in metadata for reproducibility.
    :param house: The ProcTHOR house dict.
    :param similarity_pair_id: Pair tag from
        :func:`assign_similarity_pairs`, or ``None``.
    :param vlm: VLM client with ``describe(image, prompt)``.
    :param cache: Caption cache.
    :param output_dir: Per-house output directory.
    :param max_waypoints: Cap on teleport waypoints (``None`` = all).
    :param rotations_per_waypoint: Number of rotation stops per waypoint.
    :param headless: Use CloudRendering.
    :param save_frames: Save RGB frames to disk.
    :param vlm_model: VLM model name (for cache keys).
    :returns: Sample dict written to ``trajectory.json``.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    sample_id = f"procthor_house_{house_idx:03d}"
    md = house_metadata(house)
    log.info(
        "Collecting %s (dataset_idx=%d, %d rooms, types=%s, pair=%s)",
        sample_id,
        dataset_idx,
        md["room_count"],
        md["room_types_present"],
        similarity_pair_id or "-",
    )

    env = AI2ThorAdapter(
        scene=house,
        exploration_mode="teleport",
        max_waypoints=max_waypoints,
        headless=headless,
        rotations_per_waypoint=rotations_per_waypoint,
    )

    try:
        frame, pos = env.reset()
        scene_objects = _get_scene_objects(env._controller)

        trajectory: List[Dict[str, Any]] = []
        timestamps: List[float] = []
        base_time = time.time()
        step_idx = 0

        done = False
        while True:
            frame_id = f"frame_{step_idx:04d}"
            timestamp = base_time + step_idx * 2.0
            timestamps.append(timestamp)

            image_path = None
            if save_frames:
                image_path = f"frames/{frame_id}.jpg"
                full_path = os.path.join(output_dir, image_path)
                _save_frame(frame, full_path)

            layers: Dict[str, str] = {}
            for layer_name, prompt in LAYER_PROMPTS.items():
                cache_key = f"{sample_id}_{frame_id}"
                cached = cache.get(cache_key, prompt, vlm_model)
                if cached is not None:
                    caption = cached
                else:
                    caption = vlm.describe(frame, prompt)
                    cache.put(cache_key, prompt, vlm_model, caption)

                if layer_name == "place" and not _is_valid_place(caption):
                    caption = ""
                elif layer_name in ("vlm", "detections") and not _is_valid_caption(
                    caption
                ):
                    caption = ""

                layers[layer_name] = caption

            room = room_at_position(house, x=pos[0], z=pos[1])
            waypoint: Dict[str, Any] = {
                "frame_id": frame_id,
                "position": [pos[0], pos[1], 0.0],
                "timestamp": timestamp,
                "layers": layers,
                "room_id": room["id"] if room else None,
                "room_type": room["roomType"] if room else None,
            }
            if image_path:
                waypoint["image_path"] = image_path

            trajectory.append(waypoint)
            step_idx += 1

            if done:
                break

            frame, pos, _, done, _ = env.step(0)

        interoception = _generate_interoception(timestamps)

        sample: Dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": sample_id,
            "source": "procthor",
            "trajectory": trajectory,
            "interoception": interoception,
            "questions": [],
            "metadata": {
                **md,
                "procthor_dataset_index": dataset_idx,
                "similarity_pair_id": similarity_pair_id,
                "n_waypoints": len(trajectory),
                "scene_objects": scene_objects,
            },
        }

        traj_path = os.path.join(output_dir, "trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(sample, f, indent=2)

        # Per-waypoint room coverage summary for the log.
        room_hits: Dict[str, int] = {}
        for wp in trajectory:
            key = wp.get("room_type") or "outside"
            room_hits[key] = room_hits.get(key, 0) + 1
        log.info(
            "%s: %d frames, %d objects, room coverage=%s",
            sample_id,
            len(trajectory),
            len(scene_objects),
            room_hits,
        )
        return sample

    finally:
        env.close()


def _load_procthor(revision: Optional[str] = None) -> Any:
    """Load the ProcTHOR-10K dataset via the AllenAI ``prior`` loader.

    :param revision: Optional dataset revision pin.
    :returns: Dataset object with split-indexable access.
    """
    import prior

    if revision is not None:
        return prior.load_dataset("procthor-10k", revision=revision)
    return prior.load_dataset("procthor-10k")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for ProcTHOR scene collection.

    :param argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="Collect ProcTHOR-10K multi-room houses for eMEM-Bench v1",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-houses", type=int, default=20)
    parser.add_argument("--n-similarity-pairs", type=int, default=4)
    parser.add_argument("--min-rooms", type=int, default=4)
    parser.add_argument("--max-rooms", type=int, default=8)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help=(
            "Optional ProcTHOR-10K revision pin (commit hash). Useful to "
            "reproduce a prior run if the dataset upstream updates."
        ),
    )
    parser.add_argument("--vlm-model", default="qwen3.5:27b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=15,
        help="Cap on teleport waypoints per house (default: 15)",
    )
    parser.add_argument(
        "--rotations-per-waypoint",
        type=int,
        default=2,
        help="Rotations per waypoint (default: 2 = every 180°)",
    )
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--no-headless", dest="headless", action="store_false", default=True
    )
    parser.add_argument(
        "--dry-run-houses",
        type=int,
        default=None,
        help=(
            "Run on the first N selected houses only — useful for a "
            "fast budget / coverage sanity check before a full sweep."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    os.makedirs(args.output, exist_ok=True)
    cache_path = os.path.join(args.output, "caption_cache.jsonl")
    cache = CaptionCache(cache_path)
    vlm = _make_vlm(args.vlm_model, args.ollama_url)

    log.info("Loading ProcTHOR-10K (%s split)", args.split)
    dataset = _load_procthor(revision=args.dataset_revision)
    selected = select_houses(
        dataset,
        n_houses=args.n_houses,
        min_rooms=args.min_rooms,
        max_rooms=args.max_rooms,
        split=args.split,
        seed=args.seed,
    )
    pair_map = assign_similarity_pairs(selected, n_pairs=args.n_similarity_pairs)

    if args.dry_run_houses is not None:
        selected = selected[: args.dry_run_houses]
        log.info("Dry-run: limited to %d houses", len(selected))

    manifest_path = os.path.join(args.output, "scenes.jsonl")
    # Truncate the manifest so incomplete prior runs don't leak.
    open(manifest_path, "w").close()

    for local_idx, (dataset_idx, house) in enumerate(selected):
        house_dir = os.path.join(args.output, f"house_{local_idx:03d}")
        pair_id = pair_map.get(dataset_idx)
        try:
            sample = collect_house(
                house_idx=local_idx,
                dataset_idx=dataset_idx,
                house=house,
                similarity_pair_id=pair_id,
                vlm=vlm,
                cache=cache,
                output_dir=house_dir,
                max_waypoints=args.max_waypoints,
                rotations_per_waypoint=args.rotations_per_waypoint,
                headless=args.headless,
                save_frames=args.save_frames,
                vlm_model=args.vlm_model,
            )
        except Exception:
            log.exception(
                "House %d (dataset_idx=%d) failed; skipping", local_idx, dataset_idx
            )
            continue

        manifest_entry = {
            "sample_id": sample["sample_id"],
            "scene_id": sample["scene_id"],
            "source": "procthor",
            "trajectory_path": os.path.relpath(
                os.path.join(house_dir, "trajectory.json"), args.output
            ),
            "room_count": sample["metadata"]["room_count"],
            "room_types_present": sample["metadata"]["room_types_present"],
            "similarity_pair_id": pair_id,
            "n_waypoints": sample["metadata"]["n_waypoints"],
            "procthor_dataset_index": dataset_idx,
        }
        with open(manifest_path, "a") as f:
            f.write(json.dumps(manifest_entry) + "\n")

    log.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
