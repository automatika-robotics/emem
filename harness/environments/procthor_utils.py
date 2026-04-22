"""ProcTHOR-specific helpers used by the A7 scene-collection pipeline.

These are thin utilities around the ProcTHOR-10K dataset format as
loaded by ``prior.load_dataset('procthor-10k')``. The core
navigation adapter lives in :class:`AI2ThorAdapter` — it accepts
ProcTHOR house dicts natively — so this module focuses on the
dataset-specific bits: filtering the 10K set to multi-room houses,
tagging each observation with the room the agent is standing in,
and producing a per-house manifest entry.

Nothing here initialises an AI2-THOR controller; the functions
operate on the offline house dicts so they can be called without a
running renderer.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Room-type vocabulary ProcTHOR-10K uses (empirically observed:
# Bedroom, Bathroom, LivingRoom, Kitchen). Kept as a tuple so callers
# can use it in ``in`` / equality checks without importing the dataset.
PROCTHOR_ROOM_TYPES: Tuple[str, ...] = (
    "Bedroom",
    "Bathroom",
    "LivingRoom",
    "Kitchen",
)


def _point_in_polygon(x: float, z: float, polygon: Sequence[Dict[str, float]]) -> bool:
    """Ray-casting point-in-polygon for the floor-plane (x, z).

    :param x: Floor-plane x coordinate.
    :param z: Floor-plane z coordinate.
    :param polygon: Ordered list of ``{"x": .., "z": ..}`` vertices;
        the polygon is treated as closed (last → first implicit).
    :returns: True if (x, z) lies inside the polygon.
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]["x"], polygon[i]["z"]
        xj, zj = polygon[j]["x"], polygon[j]["z"]
        if ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def room_at_position(
    house: Dict[str, Any], x: float, z: float
) -> Optional[Dict[str, str]]:
    """Return the room containing floor-plane point ``(x, z)``, if any.

    :param house: A ProcTHOR house dict (``prior.load_dataset`` row).
    :param x: Floor-plane x coordinate.
    :param z: Floor-plane z coordinate.
    :returns: ``{"id": room_id, "roomType": type}`` or ``None`` if
        the point falls outside every room polygon (e.g. doorways /
        minor numerical drift).
    """
    for room in house.get("rooms", []):
        polygon = room.get("floorPolygon", [])
        if _point_in_polygon(x, z, polygon):
            return {"id": room.get("id", ""), "roomType": room.get("roomType", "")}
    return None


def house_metadata(house: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise a ProcTHOR house for the scene manifest.

    :param house: A ProcTHOR house dict.
    :returns: Dict with ``room_count``, ``room_types_present``
        (sorted unique types), and ``n_objects``. Consumers add
        ``similarity_pair_id`` and any extra tags externally.
    """
    rooms = house.get("rooms", [])
    types = sorted({r.get("roomType", "") for r in rooms if r.get("roomType")})
    return {
        "room_count": len(rooms),
        "room_types_present": types,
        "n_objects": len(house.get("objects", [])),
    }


def select_houses(
    dataset: Any,
    n_houses: int,
    min_rooms: int = 4,
    max_rooms: int = 8,
    split: str = "train",
    seed: int = 0,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Sample ``n_houses`` distinct houses with room counts in range.

    Uses a deterministic hash-based stride so repeated calls with
    the same seed pick the same indices regardless of dataset
    shuffling. Filtering is eager because the ProcTHOR-10K rows are
    cheap (dicts, not scene states).

    :param dataset: Object returned by ``prior.load_dataset(...)``.
    :param n_houses: Number of houses to return.
    :param min_rooms: Inclusive lower bound on room count.
    :param max_rooms: Inclusive upper bound on room count.
    :param split: Dataset split name.
    :param seed: Deterministic seed.
    :returns: List of ``(index, house_dict)`` tuples.
    """
    split_data = dataset[split]
    candidate_ids: List[int] = []
    for i in range(len(split_data)):
        n = len(split_data[i].get("rooms", []))
        if min_rooms <= n <= max_rooms:
            candidate_ids.append(i)
    if len(candidate_ids) < n_houses:
        raise ValueError(
            f"Only {len(candidate_ids)} houses match rooms∈[{min_rooms}, {max_rooms}] "
            f"in split {split!r}; requested {n_houses}."
        )
    # Deterministic shuffle keyed on seed — avoids bias toward low-index
    # houses (all index 0 samples get picked first).

    def _key(idx: int) -> int:
        h = hashlib.md5(f"{seed}:{idx}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    ordered = sorted(candidate_ids, key=_key)
    picked = ordered[:n_houses]
    return [(i, split_data[i]) for i in picked]


def assign_similarity_pairs(
    selected: List[Tuple[int, Dict[str, Any]]], n_pairs: int
) -> Dict[int, Optional[str]]:
    """Group the first ``2 * n_pairs`` houses into visually-similar pairs.

    Two houses are treated as a similarity pair when they have the
    same multiset of ``roomType``s. We scan the selected list in
    order, pair up same-signature houses greedily, and stop once we
    have ``n_pairs`` pairs. Houses that don't end up paired get
    ``None``.

    :param selected: Output of :func:`select_houses`.
    :param n_pairs: Number of similarity pairs to emit.
    :returns: Mapping ``dataset_index -> pair_id`` where ``pair_id``
        is ``"pair_{k}"`` for the k-th pair and ``None`` otherwise.
    """
    pairs: Dict[int, Optional[str]] = {idx: None for idx, _ in selected}
    signatures: Dict[Tuple[str, ...], int] = {}
    next_pair = 0
    for idx, house in selected:
        if next_pair >= n_pairs:
            break
        sig = tuple(sorted(r.get("roomType", "") for r in house.get("rooms", [])))
        if sig in signatures:
            partner_idx = signatures.pop(sig)
            pair_id = f"pair_{next_pair}"
            pairs[idx] = pair_id
            pairs[partner_idx] = pair_id
            next_pair += 1
        else:
            signatures[sig] = idx
    return pairs
