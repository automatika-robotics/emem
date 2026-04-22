"""Unit tests for harness.environments.procthor_utils.

These exercise the pure-python helpers (room lookup, house metadata,
selection, similarity pairing) against synthetic ProcTHOR-shaped
dicts — no ProcTHOR dataset download or AI2-THOR controller needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.environments.procthor_utils import (
    assign_similarity_pairs,
    house_metadata,
    room_at_position,
    select_houses,
)


def _unit_square(x0: float, z0: float, side: float = 4.0) -> list[dict[str, float]]:
    """Rectangle with corners at (x0, z0) .. (x0+side, z0+side)."""
    return [
        {"x": x0, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0 + side},
        {"x": x0, "y": 0.0, "z": z0 + side},
    ]


def _make_house(room_specs: List[Dict[str, Any]], n_objects: int = 0) -> Dict[str, Any]:
    """Build a minimal ProcTHOR-shaped house dict for testing."""
    rooms = []
    for i, spec in enumerate(room_specs):
        rooms.append({
            "id": spec.get("id", f"room|{i}"),
            "roomType": spec["roomType"],
            "floorPolygon": spec.get("floorPolygon", _unit_square(0.0, 0.0)),
        })
    return {
        "rooms": rooms,
        "objects": [{"objectId": f"obj_{i}"} for i in range(n_objects)],
    }


class _FakeSplit:
    """List-like split stand-in for :func:`select_houses` tests."""

    def __init__(self, houses: List[Dict[str, Any]]):
        self._houses = houses

    def __len__(self) -> int:
        return len(self._houses)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self._houses[i]


class _FakeDataset:
    """``prior.load_dataset``-shaped object for tests."""

    def __init__(self, train_houses: List[Dict[str, Any]]):
        self._splits = {"train": _FakeSplit(train_houses)}

    def __getitem__(self, key: str) -> _FakeSplit:
        return self._splits[key]


class TestRoomAtPosition:
    def test_point_inside_room(self):
        house = _make_house([
            {"roomType": "Kitchen", "floorPolygon": _unit_square(0.0, 0.0)},
            {"roomType": "Bathroom", "floorPolygon": _unit_square(5.0, 0.0)},
        ])
        room = room_at_position(house, x=2.0, z=2.0)
        assert room is not None
        assert room["roomType"] == "Kitchen"

        room = room_at_position(house, x=6.0, z=2.0)
        assert room is not None
        assert room["roomType"] == "Bathroom"

    def test_point_outside_all_rooms(self):
        house = _make_house([
            {"roomType": "Kitchen", "floorPolygon": _unit_square(0.0, 0.0)},
        ])
        assert room_at_position(house, x=50.0, z=50.0) is None

    def test_empty_rooms_returns_none(self):
        assert room_at_position({"rooms": []}, x=0.0, z=0.0) is None

    def test_degenerate_polygon_is_none(self):
        # Polygon with <3 vertices can't contain any point.
        house = {
            "rooms": [
                {
                    "id": "r0",
                    "roomType": "Kitchen",
                    "floorPolygon": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 1, "y": 0, "z": 0},
                    ],
                }
            ]
        }
        assert room_at_position(house, x=0.5, z=0.0) is None


class TestHouseMetadata:
    def test_metadata_fields(self):
        house = _make_house(
            [
                {"roomType": "Kitchen"},
                {"roomType": "Bedroom"},
                {"roomType": "Kitchen"},
            ],
            n_objects=7,
        )
        md = house_metadata(house)
        assert md["room_count"] == 3
        assert md["room_types_present"] == ["Bedroom", "Kitchen"]  # unique + sorted
        assert md["n_objects"] == 7

    def test_empty_house(self):
        md = house_metadata({"rooms": [], "objects": []})
        assert md["room_count"] == 0
        assert md["room_types_present"] == []
        assert md["n_objects"] == 0


class TestSelectHouses:
    def test_filters_by_room_range(self):
        houses = [
            _make_house([{"roomType": "Kitchen"}]),  # 1 room — out
            _make_house([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}]),  # 2 — out
            _make_house([{"roomType": "Kitchen"}] * 4),  # 4 — in
            _make_house([{"roomType": "Kitchen"}] * 5),  # 5 — in
            _make_house([{"roomType": "Kitchen"}] * 9),  # 9 — out
        ]
        ds = _FakeDataset(houses)
        selected = select_houses(ds, n_houses=2, min_rooms=4, max_rooms=8, seed=0)
        assert len(selected) == 2
        for _, house in selected:
            assert 4 <= len(house["rooms"]) <= 8

    def test_deterministic_given_seed(self):
        houses = [_make_house([{"roomType": "Kitchen"}] * 4) for _ in range(10)]
        ds = _FakeDataset(houses)
        a = select_houses(ds, n_houses=3, seed=42)
        b = select_houses(ds, n_houses=3, seed=42)
        assert [i for i, _ in a] == [i for i, _ in b]

    def test_raises_when_insufficient_candidates(self):
        houses = [_make_house([{"roomType": "Kitchen"}])]  # only 1-room houses
        ds = _FakeDataset(houses)
        try:
            select_houses(ds, n_houses=2, min_rooms=4, seed=0)
        except ValueError as e:
            assert "houses match" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestAssignSimilarityPairs:
    def test_same_room_signature_pairs(self):
        # Two houses with identical room-type multisets should pair.
        same_sig = [
            (0, _make_house([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}])),
            (1, _make_house([{"roomType": "Bedroom"}, {"roomType": "Kitchen"}])),
            (2, _make_house([{"roomType": "Bathroom"}])),
        ]
        pairs = assign_similarity_pairs(same_sig, n_pairs=1)
        assert pairs[0] == pairs[1]
        assert pairs[0] is not None
        assert pairs[2] is None

    def test_no_matching_signatures_no_pairs(self):
        unique = [
            (0, _make_house([{"roomType": "Kitchen"}])),
            (1, _make_house([{"roomType": "Bedroom"}])),
            (2, _make_house([{"roomType": "Bathroom"}])),
        ]
        pairs = assign_similarity_pairs(unique, n_pairs=2)
        assert all(v is None for v in pairs.values())

    def test_stops_at_n_pairs(self):
        houses = [
            (i, _make_house([{"roomType": "Kitchen"}, {"roomType": "Bedroom"}]))
            for i in range(6)
        ]
        # Six same-signature houses could yield three pairs; cap at 1.
        pairs = assign_similarity_pairs(houses, n_pairs=1)
        assigned = [v for v in pairs.values() if v is not None]
        assert len(assigned) == 2  # exactly one pair = 2 houses
