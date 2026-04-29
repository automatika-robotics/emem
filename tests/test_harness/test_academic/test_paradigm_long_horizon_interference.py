"""Unit tests for the long-horizon-interference paradigm (A14c.4).

Synthetic ProcTHOR-shaped house dicts; no LLM picker (LHI is
deterministic). Covers:

  * Generator pairs scenes sequentially and emits A-only (yes) +
    B-only (no) candidates per pair.
  * Single-house pairs with no exclusive items are skipped.
  * Each candidate carries item_origin, label_to_sample_id, and
    gap_seconds metadata.
  * Pre-filter rubric mentions yes / ambiguous / no.
  * Schedule builder emits the 4-phase shape:
    encode-A → AdvanceClock(maint=True) → encode-B → probe.
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.long_horizon_interference import (
    DEFAULT_GAP_SECONDS,
    ITEM_ORIGIN_A,
    ITEM_ORIGIN_B,
    LongHorizonInterferenceGenerator,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.schedule_builder import (
    build_long_horizon_interference_schedule,
)
from harness.benchmarks.academic.emem_bench_v1.schedule import (
    AdvanceClockPhase,
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
    """Build a synthetic single-room house with the given object types."""
    return {
        "rooms": [
            {
                "id": "room|0",
                "roomType": "Kitchen",
                "floorPolygon": _unit_square(0.0, 0.0),
            }
        ],
        "objects": [
            # Position items strictly inside the polygon (avoid the
            # x=4 / z=4 boundary which point-in-polygon treats as
            # outside under ray-casting).
            {"id": f"{t}|0|{i}", "position": {"x": 1.0 + i * 0.5, "z": 1.5}}
            for i, t in enumerate(items)
        ],
    }


def _scene(
    sample_id: str,
    procthor_idx: int = 0,
    *,
    with_trajectory: bool = False,
) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "procthor_dataset_index": procthor_idx,
        "interoception": [],
        "scene_objects": [],
    }
    if with_trajectory:
        s["trajectory"] = [
            {
                "frame_id": "f0",
                "position": [1.0, 0.0, 1.0],
                "timestamp": 100.0,
                "room_id": "room|0",
                "room_type": "Kitchen",
                "layers": {"vlm": "kitchen view", "object_detection": "fridge"},
            },
        ]
    else:
        s["trajectory"] = []
    return s


# Two houses with overlapping + distinctive items.
HOUSE_A_OBJECTS = ["Fridge", "Microwave", "AlarmClock", "Vase"]
HOUSE_B_OBJECTS = ["Fridge", "Microwave", "TeddyBear", "BasketBall"]
# A-only: AlarmClock, Vase
# B-only: TeddyBear, BasketBall
# Shared: Fridge, Microwave


class TestLongHorizonInterferenceGenerator:
    def _lookup(self):
        # Pair 0: idx 0 (A) and idx 1 (B). Pair 1: idx 2 / idx 3 (empty).
        houses = {
            0: _house_with(HOUSE_A_OBJECTS),
            1: _house_with(HOUSE_B_OBJECTS),
            2: _house_with(["Lamp", "Bed"]),
            3: _house_with(["Lamp", "Bed"]),  # all-shared with idx 2 → no candidates
        }
        return lambda i: houses.get(i)

    def test_emits_a_only_and_b_only_pairs(self):
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(), target_total=20
        )
        scenes = [_scene("a", 0), _scene("b", 1)]
        cands = gen.generate(scenes, n_per_scene=4)
        # 2 A-only + 2 B-only = 4 candidates per pair
        assert len(cands) == 4
        origins = sorted(c.paradigm_metadata["item_origin"] for c in cands)
        assert origins == [ITEM_ORIGIN_A, ITEM_ORIGIN_A, ITEM_ORIGIN_B, ITEM_ORIGIN_B]
        a_only = [
            c for c in cands if c.paradigm_metadata["item_origin"] == ITEM_ORIGIN_A
        ]
        b_only = [
            c for c in cands if c.paradigm_metadata["item_origin"] == ITEM_ORIGIN_B
        ]
        # A-only items answer "yes"; B-only "no".
        assert all(c.answer == "yes" for c in a_only)
        assert all(c.answer == "no" for c in b_only)

    def test_skips_pair_with_no_exclusive_items(self):
        """A pair where all objects are shared yields zero candidates."""
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(), target_total=20
        )
        # Scenes 2,3 share all objects → empty
        scenes = [_scene("c", 2), _scene("d", 3)]
        cands = gen.generate(scenes, n_per_scene=4)
        assert cands == []

    def test_question_format_and_answer(self):
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(), target_total=20
        )
        scenes = [_scene("a", 0), _scene("b", 1)]
        cands = gen.generate(scenes, n_per_scene=4)
        # Pick one A-only candidate.
        a_cand = next(
            c for c in cands if c.paradigm_metadata["item_origin"] == ITEM_ORIGIN_A
        )
        item = a_cand.paradigm_metadata["probe_object"]
        assert "House Alpha" in a_cand.question
        assert item in a_cand.question
        assert a_cand.question.startswith("Earlier you toured House Alpha. Did you see")

    def test_metadata_carries_label_map_and_gap(self):
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(), target_total=20, gap_seconds=12345
        )
        scenes = [_scene("a", 0), _scene("b", 1)]
        cands = gen.generate(scenes, n_per_scene=4)
        for c in cands:
            md = c.paradigm_metadata
            assert md["label_to_sample_id"] == {"Alpha": "a", "Beta": "b"}
            assert md["house_alpha_id"] == "a"
            assert md["house_beta_id"] == "b"
            assert md["gap_seconds"] == 12345
            assert "ground_truth_objects_in_alpha" in md
            assert "ground_truth_objects_in_beta" in md

    def test_default_gap_is_one_day(self):
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(), target_total=20
        )
        scenes = [_scene("a", 0), _scene("b", 1)]
        cands = gen.generate(scenes, n_per_scene=4)
        assert all(
            c.paradigm_metadata["gap_seconds"] == DEFAULT_GAP_SECONDS for c in cands
        )

    def test_exclude_probes_suppresses_duplicates(self):
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(),
            target_total=20,
            exclude_probes={"__lhi__": {"alarm clock"}},
        )
        scenes = [_scene("a", 0), _scene("b", 1)]
        cands = gen.generate(scenes, n_per_scene=4)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "alarm clock" not in probes

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = LongHorizonInterferenceGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low


class TestLongHorizonInterferenceScheduleBuilder:
    def _lookup(self):
        houses = {
            0: _house_with(HOUSE_A_OBJECTS),
            1: _house_with(HOUSE_B_OBJECTS),
        }
        return lambda i: houses.get(i)

    def test_four_phase_schedule_with_maintenance_gap(self):
        gen = LongHorizonInterferenceGenerator(
            house_lookup=self._lookup(), target_total=4
        )
        scene_a = _scene("a", 0, with_trajectory=True)
        scene_b = _scene("b", 1, with_trajectory=True)
        cand = gen.generate([scene_a, scene_b], n_per_scene=2)[0]
        scenes_by_id = {"a": scene_a, "b": scene_b}
        sched = build_long_horizon_interference_schedule(cand, scenes_by_id)

        assert len(sched.phases) == 4
        assert isinstance(sched.phases[0], IngestPhase)
        assert isinstance(sched.phases[1], AdvanceClockPhase)
        assert isinstance(sched.phases[2], IngestPhase)
        assert isinstance(sched.phases[3], ProbePhase)
        # Gap fires maintenance and matches candidate's gap_seconds.
        adv = sched.phases[1]
        assert adv.run_maintenance is True
        assert adv.delta_seconds == cand.paradigm_metadata["gap_seconds"]
        # Alpha's observations carry the prefix; Beta's likewise.
        a_phase = sched.phases[0]
        b_phase = sched.phases[2]
        assert all("In house Alpha:" in obs.text for obs in a_phase.observations)
        assert all("In house Beta:" in obs.text for obs in b_phase.observations)
        # Beta's observations are timestamped strictly after Alpha + gap.
        last_a = max(obs.timestamp for obs in a_phase.observations)
        first_b = min(obs.timestamp for obs in b_phase.observations)
        assert first_b > last_a + cand.paradigm_metadata["gap_seconds"] - 1e-6

    def test_missing_label_map_raises(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        bad = CandidateQuestion(
            question_id="x",
            question="?",
            answer="yes",
            category="long_horizon_interference",
            paradigm="long_horizon_interference",
            scene_ids=["a", "b"],
            paradigm_metadata={"gap_seconds": 10},  # missing label_to_sample_id
        )
        try:
            build_long_horizon_interference_schedule(bad, {})
        except ValueError as exc:
            assert "label_to_sample_id" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_missing_gap_seconds_raises(self):
        from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
            CandidateQuestion,
        )

        bad = CandidateQuestion(
            question_id="x",
            question="?",
            answer="yes",
            category="long_horizon_interference",
            paradigm="long_horizon_interference",
            scene_ids=["a", "b"],
            paradigm_metadata={"label_to_sample_id": {"Alpha": "a", "Beta": "b"}},
        )
        try:
            build_long_horizon_interference_schedule(bad, {})
        except ValueError as exc:
            assert "gap_seconds" in str(exc)
        else:
            raise AssertionError("expected ValueError")
