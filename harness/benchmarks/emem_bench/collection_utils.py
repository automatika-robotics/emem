"""Shared helpers used across eMEM-Bench collection scripts.

Consolidates the small utilities that used to live in
``collect_ai2thor.py`` and ``generate_questions.py`` — extracted here
so the ProcTHOR collector (and any future v1 collector) can reuse
them after the v0 modules are retired.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Set

import numpy as np

# Multi-layer VLM prompts used by every eMEM-Bench collector.
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


# Canonical vocabulary of "places" the place-layer VLM caption may report.
# Used by :func:`is_valid_place` to reject fabricated or off-ontology labels.
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
    r"|(?:single color)"
    r"|(?:^blank\b)"
)


def is_valid_caption(text: str) -> bool:
    """Return True if *text* is a usable VLM caption.

    Rejects empty strings, strings shorter than 10 characters, and
    captions matching common garbage patterns (apologetic refusals,
    pure-numeric output, ``"unknown"``, etc.).
    """
    if not text or len(text.strip()) < 10:
        return False
    return _GARBAGE_CAPTION_RE.search(text.strip()) is None


def is_valid_place(place: str) -> bool:
    """Return True if *place* is a recognised room/area name."""
    return place.strip().lower() in VALID_PLACES


def make_ollama_vlm(model: str, base_url: str) -> Any:
    """Construct an Ollama-backed VLM client.

    :param model: Ollama model tag (e.g. ``"qwen3.5:27b"``).
    :param base_url: Ollama server URL.
    :returns: VLM client exposing ``.describe(image, prompt)``.
    """
    from harness.providers.ollama_vlm import OllamaVLM

    return OllamaVLM(model=model, base_url=base_url)


def generate_synthetic_interoception(
    timestamps: List[float],
) -> List[Dict[str, Any]]:
    """Sample battery + CPU-temperature entries along a trajectory.

    Battery drains linearly from a random 85–100% start at a random
    0.5–2%/min rate; CPU temp fluctuates around 55°C with a small
    positive drift. Sampled at up to 10 evenly-spaced timestamps.

    :param timestamps: Trajectory timestamps (seconds).
    :returns: List of ``{"timestamp": t, "battery": "...", "cpu_temp": "..."}``.
    """
    if not timestamps:
        return []

    entries: List[Dict[str, Any]] = []
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


def extract_scene_objects(controller: Any) -> List[Dict[str, Any]]:
    """Extract object metadata from an active AI2-THOR / ProcTHOR scene.

    :param controller: AI2-THOR controller with an active scene.
    :returns: List of object dicts (name, type, position, flags).
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


def save_frame_jpeg(frame: np.ndarray, path: str) -> None:
    """Save an ``(H, W, 3)`` uint8 RGB frame as JPEG at quality 85.

    :param frame: RGB frame array.
    :param path: Output file path.
    """
    from PIL import Image

    Image.fromarray(frame).save(path, quality=85)
