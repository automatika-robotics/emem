"""Unit tests for the serial-position paradigm (A14c.5)."""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_serial_position_schedule,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.serial_position import (
    ACTIVE_POSITIONS,
    BIN_LABELS,
    CHAIN_LENGTH,
    SerialPositionGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    IngestPhase,
    ProbePhase,
)


def _unit_square(x0: float, z0: float, side: float = 4.0) -> List[Dict[str, float]]:
    return [
        {"x": x0, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0},
        {"x": x0 + side, "y": 0.0, "z": z0 + side},
        {"x": x0, "y": 0.0, "z": z0 + side},
    ]


def _house_with(items: List[str]) -> Dict[str, Any]:
    return {
        "rooms": [
            {
                "id": "room|0",
                "roomType": "Kitchen",
                "floorPolygon": _unit_square(0.0, 0.0),
            },
            {
                "id": "room|1",
                "roomType": "Bedroom",
                "floorPolygon": _unit_square(5.0, 0.0),
            },
        ],
        "objects": [
            # Strictly inside the polygons (avoid x=4 / z=4 boundary).
            {"id": f"{t}|0|{i}", "position": {"x": 1.0 + (i % 3) * 0.5, "z": 1.5}}
            if i < len(items) // 2
            else {"id": f"{t}|1|{i}", "position": {"x": 6.0 + (i % 3) * 0.5, "z": 1.5}}
            for i, t in enumerate(items)
        ],
    }


def _scene(
    sample_id: str, procthor_idx: int, *, with_trajectory: bool = False
) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "procthor_dataset_index": procthor_idx,
        "interoception": [],
        "scene_objects": [],
        "trajectory": [],
    }
    if with_trajectory:
        # Two-frame trajectory; timestamps within a single house are
        # contiguous; serial-position chains shift them per chain
        # position so the cross-house ordering is preserved.
        s["trajectory"] = [
            {
                "frame_id": "f0",
                "position": [1.0, 0.0, 1.5],
                "timestamp": 100.0 + procthor_idx * 1000.0,
                "room_id": "room|0",
                "room_type": "Kitchen",
                "layers": {"vlm": "kitchen view", "object_detection": "fridge"},
            },
            {
                "frame_id": "f1",
                "position": [6.0, 0.0, 1.5],
                "timestamp": 200.0 + procthor_idx * 1000.0,
                "room_id": "room|1",
                "room_type": "Bedroom",
                "layers": {"vlm": "bedroom view", "object_detection": "bed, lamp"},
            },
        ]
    return s


# 5 distinct houses with disjoint object sets so each position's
# items can be cleanly traced to its house.
HOUSE_OBJECTS = [
    ["AlarmClock", "Vase"],  # position 0  (early)
    ["Boots", "Backpack"],  # position 1  (buffer)
    ["Painting", "Statue"],  # position 2  (middle)
    ["Newspaper", "Box"],  # position 3  (buffer)
    ["DogBed", "Cart"],  # position 4  (late)
]


class TestSerialPositionGenerator:
    def _lookup(self):
        houses = {i: _house_with(HOUSE_OBJECTS[i]) for i in range(CHAIN_LENGTH)}
        return lambda i: houses.get(i)

    def _chain(self):
        return [_scene(f"s{i}", i) for i in range(CHAIN_LENGTH)]

    def test_emits_only_from_active_positions(self):
        gen = SerialPositionGenerator(house_lookup=self._lookup(), target_total=20)
        cands = gen.generate(self._chain(), n_per_scene=18)
        # Items from positions 0, 2, 4 only (1 and 3 are buffers).
        positions = {c.paradigm_metadata["position_in_chain"] for c in cands}
        assert positions == set(ACTIVE_POSITIONS)
        # No items from positions 1 or 3.
        for c in cands:
            assert c.paradigm_metadata["source_house_id"] not in {"s1", "s3"}

    def test_bin_labels_match_position(self):
        gen = SerialPositionGenerator(house_lookup=self._lookup(), target_total=20)
        cands = gen.generate(self._chain(), n_per_scene=18)
        for c in cands:
            md = c.paradigm_metadata
            assert md["bin_label"] == BIN_LABELS[md["position_in_chain"]]
            assert c.answer == md["bin_label"]

    def test_chain_metadata_present(self):
        gen = SerialPositionGenerator(house_lookup=self._lookup(), target_total=20)
        cands = gen.generate(self._chain(), n_per_scene=18)
        for c in cands:
            md = c.paradigm_metadata
            # chain_id derives from the first chain scene's sample_id
            # so each chain has a globally-unique identifier.
            assert md["chain_id"] == "chain_s0"
            assert md["chain_length"] == CHAIN_LENGTH
            assert len(md["chain_sample_ids"]) == CHAIN_LENGTH
            assert "ground_truth_objects_in_room" in md

    def test_skips_short_scene_lists(self):
        gen = SerialPositionGenerator(house_lookup=self._lookup(), target_total=20)
        # Only 4 scenes — not enough for a full 5-house chain.
        cands = gen.generate(self._chain()[:4], n_per_scene=18)
        assert cands == []

    def test_question_format(self):
        gen = SerialPositionGenerator(house_lookup=self._lookup(), target_total=20)
        cands = gen.generate(self._chain(), n_per_scene=18)
        for c in cands:
            assert c.question.startswith("Did you see the ")
            assert "early, in the middle, or late" in c.question

    def test_exclude_probes_suppresses_duplicates(self):
        gen = SerialPositionGenerator(
            house_lookup=self._lookup(),
            target_total=20,
            exclude_probes={"__serpos__": {"alarm clock"}},
        )
        cands = gen.generate(self._chain(), n_per_scene=18)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "alarm clock" not in probes

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = SerialPositionGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestSerialPositionScheduleBuilder:
    def _lookup(self):
        houses = {i: _house_with(HOUSE_OBJECTS[i]) for i in range(CHAIN_LENGTH)}
        return lambda i: houses.get(i)

    def test_chain_phases_with_monotonic_timestamps(self):
        gen = SerialPositionGenerator(house_lookup=self._lookup(), target_total=20)
        chain = [_scene(f"s{i}", i, with_trajectory=True) for i in range(CHAIN_LENGTH)]
        cand = gen.generate(chain, n_per_scene=3)[0]
        scenes_by_id = {s["sample_id"]: s for s in chain}
        sched = build_serial_position_schedule(cand, scenes_by_id)

        # CHAIN_LENGTH ingest phases + 1 probe phase.
        assert len(sched.phases) == CHAIN_LENGTH + 1
        for i in range(CHAIN_LENGTH):
            assert isinstance(sched.phases[i], IngestPhase)
        assert isinstance(sched.phases[-1], ProbePhase)
        # Each subsequent phase's earliest observation > previous
        # phase's latest observation (the cross-house chain shift).
        prev_max = float("-inf")
        for i in range(CHAIN_LENGTH):
            phase = sched.phases[i]
            ts_min = min(o.timestamp for o in phase.observations)
            ts_max = max(o.timestamp for o in phase.observations)
            assert ts_min > prev_max
            prev_max = ts_max

    def test_missing_chain_metadata_raises(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        bad = CandidateQuestion(
            question_id="x",
            question="?",
            answer="early",
            category="serial_position",
            paradigm="serial_position",
            scene_ids=[],
            paradigm_metadata={},
        )
        try:
            build_serial_position_schedule(bad, {})
        except ValueError as exc:
            assert "chain_sample_ids" in str(exc)
        else:
            raise AssertionError("expected ValueError")
