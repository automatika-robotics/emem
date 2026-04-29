"""Unit tests for the source-monitoring paradigm (A14b.5).

Uses synthetic trajectory frames + a mocked LLM picker. Covers:

  * Picks attributed to "scene description" must appear in the VLM
    corpus and not in the OD object-name token union.
  * Picks attributed to "object detection" must not overlap any VLM
    corpus token.
  * Hallucinated picks (LLM proposes items absent from both streams)
    are dropped.
  * Question text uses lowercase + correct article.
  * Pre-filter rubric mentions yes / ambiguous / no.
"""

from __future__ import annotations

from typing import Any, Dict, List

from harness.benchmarks.academic.emem_bench_v1.paradigms.source_monitoring import (
    SOURCE_OD,
    SOURCE_VLM,
    SourceMonitoringGenerator,
)


def _frame(
    fid: str,
    vlm: str,
    od: str,
    *,
    pos=(0.0, 0.0, 0.0),
    timestamp: float = 0.0,
    room_id: str = "room|0",
    room_type: str = "Kitchen",
) -> Dict[str, Any]:
    return {
        "frame_id": fid,
        "position": list(pos),
        "timestamp": timestamp,
        "room_id": room_id,
        "room_type": room_type,
        "layers": {"vlm": vlm, "object_detection": od, "place": room_type.lower()},
    }


def _scene(
    sample_id: str = "house_0",
    *,
    trajectory: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scene_id": sample_id,
        "procthor_dataset_index": 42,
        "trajectory": trajectory or [],
        "interoception": [],
        "scene_objects": [],
    }


_BASE_FRAMES = [
    _frame(
        "f0",
        vlm=(
            "A minimalist kitchen with pale green walls and a wooden dining "
            "table. Framed pictures hang on the far wall."
        ),
        od="fridge, microwave, dining table, picture",
        room_id="room|7",
        room_type="Kitchen",
    ),
    _frame(
        "f1",
        vlm="A small bedroom with a single bed and a tall lamp beside it.",
        od="bed, lamp, alarm clock, dresser",
        room_id="room|8",
        room_type="Bedroom",
    ),
]


class TestSourceMonitoringGenerator:
    def _llm_stub(self, vlm_picks: List[str], od_picks: List[str]):
        """Mocked picker: return VLM picks when asked for the VLM source,
        OD picks when asked for the OD source."""
        import json as _json

        def chat(prompt: str) -> str:
            if "DO NOT appear in the object detection stream" in prompt:
                return _json.dumps(vlm_picks)
            if "DO NOT appear in the scene description stream" in prompt:
                return _json.dumps(od_picks)
            return "[]"

        return chat

    def test_vlm_pick_must_be_in_vlm_and_not_in_od(self):
        """'pale green walls' is in VLM caption, no OD overlap → kept.
        'fridge' is in OD → must be dropped from VLM picks even if LLM
        proposes it."""
        llm = self._llm_stub(
            vlm_picks=["pale green walls", "fridge"],
            od_picks=["alarm clock"],
        )
        gen = SourceMonitoringGenerator(llm_chat=llm, target_total=10)
        cands = gen.generate([_scene(trajectory=_BASE_FRAMES)], n_per_scene=4)
        vlm_probes = [
            c.paradigm_metadata["probe_object"] for c in cands if c.answer == SOURCE_VLM
        ]
        assert "pale green walls" in vlm_probes
        assert "fridge" not in vlm_probes

    def test_od_pick_must_not_overlap_vlm_tokens(self):
        """'alarm clock' is OD-only (neither word in any caption) → kept.
        'dining table' overlaps the caption text ('wooden dining table')
        → must be dropped from OD picks even if LLM proposes it."""
        llm = self._llm_stub(
            vlm_picks=["pale green walls"],
            od_picks=["alarm clock", "dining table"],
        )
        gen = SourceMonitoringGenerator(llm_chat=llm, target_total=10)
        cands = gen.generate([_scene(trajectory=_BASE_FRAMES)], n_per_scene=4)
        od_probes = [
            c.paradigm_metadata["probe_object"] for c in cands if c.answer == SOURCE_OD
        ]
        assert "alarm clock" in od_probes
        assert "dining table" not in od_probes

    def test_hallucinated_picks_are_dropped(self):
        """LLM proposes items absent from both streams. Must be dropped."""
        llm = self._llm_stub(
            vlm_picks=["pale green walls", "imaginary widget"],
            od_picks=["alarm clock", "made up gadget"],
        )
        gen = SourceMonitoringGenerator(llm_chat=llm, target_total=10)
        cands = gen.generate([_scene(trajectory=_BASE_FRAMES)], n_per_scene=4)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "imaginary widget" not in probes
        assert "made up gadget" not in probes
        assert "pale green walls" in probes
        assert "alarm clock" in probes

    def test_question_format_and_answer(self):
        llm = self._llm_stub(
            vlm_picks=["pale green walls"],
            od_picks=["alarm clock"],
        )
        gen = SourceMonitoringGenerator(llm_chat=llm, target_total=10)
        cands = gen.generate([_scene(trajectory=_BASE_FRAMES)], n_per_scene=4)
        by_obj = {c.paradigm_metadata["probe_object"]: c for c in cands}
        # ``the`` reads correctly for singular, plural, and mass nouns.
        assert by_obj["pale green walls"].question.startswith(
            "Did you learn about the pale green walls"
        )
        assert by_obj["pale green walls"].answer == SOURCE_VLM
        assert by_obj["alarm clock"].question.startswith(
            "Did you learn about the alarm clock"
        )
        assert by_obj["alarm clock"].answer == SOURCE_OD

    def test_skips_scene_with_no_trajectory(self):
        gen = SourceMonitoringGenerator(
            llm_chat=self._llm_stub([], []), target_total=10
        )
        cands = gen.generate([_scene(trajectory=[])], n_per_scene=4)
        assert cands == []

    def test_exclude_probes_suppresses_duplicates(self):
        llm = self._llm_stub(
            vlm_picks=["pale green walls", "framed pictures"],
            od_picks=["alarm clock"],
        )
        gen = SourceMonitoringGenerator(
            llm_chat=llm,
            target_total=10,
            exclude_probes={SOURCE_VLM: {"pale green walls"}},
        )
        cands = gen.generate([_scene(trajectory=_BASE_FRAMES)], n_per_scene=4)
        probes = {c.paradigm_metadata["probe_object"] for c in cands}
        assert "pale green walls" not in probes
        assert "framed pictures" in probes

    def test_rubric_mentions_yes_no_ambiguous(self):
        rubric = SourceMonitoringGenerator.prefilter_rubric()
        low = rubric.lower()
        assert "yes" in low and "no" in low and "ambiguous" in low
