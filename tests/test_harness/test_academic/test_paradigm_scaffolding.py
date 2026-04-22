"""Unit tests for the paradigm scaffolding (A14b.1).

Covers:
  * CandidateQuestion dataclass round-trips through JSONL and
    tolerates missing optional fields
  * ParadigmGenerator ABC + a trivial concrete paradigm for shaping
  * LLM pre-filter partitioning (yes / ambiguous / no) and the
    keep-ambiguous toggle
  * Review CLI decision branches (k / e / d / s / n / q) with a
    scripted input feed, including resume from .inprogress.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from harness.benchmarks.academic.emem_bench_v1.paradigms.base import (
    CandidateQuestion,
    ParadigmGenerator,
    ReviewDecision,
    load_candidates,
    save_candidates,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.prefilter import (
    PreFilterResult,
    prefilter_candidates,
    rubric_summary,
)
from harness.benchmarks.academic.emem_bench_v1.paradigms.review import run_review


def _candidate(qid: str, **overrides: Any) -> CandidateQuestion:
    defaults: Dict[str, Any] = {
        "question_id": qid,
        "question": f"Is {qid} present?",
        "answer": "no",
        "category": "dummy",
        "paradigm": "dummy",
        "scene_ids": ["scene_0"],
        "tools_expected": ["semantic_search"],
        "paradigm_metadata": {"probe_object": qid, "room": "kitchen"},
    }
    defaults.update(overrides)
    return CandidateQuestion(**defaults)


class _DummyParadigm(ParadigmGenerator):
    """Trivial paradigm for shape testing."""

    name = "dummy"

    def generate(
        self,
        scenes: List[Dict[str, Any]],
        *,
        n_per_scene: int,
        seed: int = 0,
    ) -> List[CandidateQuestion]:
        out: List[CandidateQuestion] = []
        for s in scenes:
            for i in range(n_per_scene):
                out.append(
                    _candidate(
                        f"{s['sample_id']}_{i}",
                        scene_ids=[s["sample_id"]],
                    )
                )
        return out

    @classmethod
    def prefilter_rubric(cls) -> str:
        return "Does this question test the Dummy paradigm? yes / ambiguous / no."


class TestCandidateQuestion:
    def test_round_trip_jsonl(self, tmp_path: Path):
        path = tmp_path / "cands.jsonl"
        originals = [_candidate("a"), _candidate("b", answer="yes")]
        n = save_candidates(path, originals)
        assert n == 2
        loaded = load_candidates(path)
        assert len(loaded) == 2
        assert loaded[0].question_id == "a"
        assert loaded[1].answer == "yes"
        assert loaded[0].paradigm_metadata["probe_object"] == "a"

    def test_from_dict_tolerates_missing_optional_fields(self):
        minimal = {
            "question_id": "x",
            "question": "q",
            "answer": "a",
            "category": "drm",
            "paradigm": "drm",
        }
        c = CandidateQuestion.from_dict(minimal)
        assert c.scene_ids == []
        assert c.paradigm_metadata == {}
        assert c.review_decision is None

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        assert load_candidates(tmp_path / "nope.jsonl") == []


class TestDummyParadigm:
    def test_generate_shapes(self):
        gen = _DummyParadigm()
        scenes = [{"sample_id": "s0"}, {"sample_id": "s1"}]
        out = gen.generate(scenes, n_per_scene=3)
        assert len(out) == 6
        ids = [c.question_id for c in out]
        assert ids == ["s0_0", "s0_1", "s0_2", "s1_0", "s1_1", "s1_2"]
        assert all(c.paradigm == "dummy" for c in out)

    def test_rubric_present(self):
        assert "yes" in _DummyParadigm.prefilter_rubric().lower()


class TestPrefilter:
    def _stub_chat(self, verdicts_by_question: Dict[str, str]):
        """Return a fake chat() that looks for each candidate's
        question text (which is unique per candidate) in the prompt
        and emits the corresponding scripted verdict."""

        def chat(prompt: str) -> str:
            for question, verdict in verdicts_by_question.items():
                if question in prompt:
                    return verdict
            return "ambiguous"

        return chat

    def test_partitions_on_verdict(self):
        cands = [_candidate("A_zzz"), _candidate("B_zzz"), _candidate("C_zzz")]
        chat = self._stub_chat({
            "Is A_zzz present?": "yes",
            "Is B_zzz present?": "no",
            "Is C_zzz present?": "ambiguous",
        })
        result = prefilter_candidates(cands, rubric="rubric", llm_chat=chat)
        assert {c.question_id for c in result.kept} == {"A_zzz", "C_zzz"}
        assert {c.question_id for c in result.rejected} == {"B_zzz"}
        y, amb, no = rubric_summary(result)
        assert (y, amb, no) == (1, 1, 1)
        # Rejected candidates are stamped with a reason.
        rej = result.rejected[0]
        assert rej.review_decision == ReviewDecision.DISCARD.value
        assert rej.review_reason == "prefilter:no"

    def test_ambiguous_rejected_when_toggle_off(self):
        cands = [_candidate("A_zzz"), _candidate("B_zzz")]
        chat = self._stub_chat({
            "Is A_zzz present?": "yes",
            "Is B_zzz present?": "ambiguous",
        })
        result = prefilter_candidates(
            cands, rubric="rubric", llm_chat=chat, keep_ambiguous=False
        )
        assert [c.question_id for c in result.kept] == ["A_zzz"]
        assert [c.question_id for c in result.rejected] == ["B_zzz"]
        assert result.rejected[0].review_reason == "prefilter:ambiguous"

    def test_garbage_output_becomes_ambiguous(self):
        cands = [_candidate("a")]

        def chat(prompt: str) -> str:
            return "I refuse to answer because I am an AI."

        result = prefilter_candidates(cands, rubric="r", llm_chat=chat)
        # "ambiguous" by default flows to kept — safe behaviour.
        assert len(result.kept) == 1
        assert result.counts["ambiguous"] == 1

    def test_prefilter_result_shape(self):
        r = PreFilterResult()
        assert r.kept == [] and r.rejected == []


class TestReviewCli:
    def _run(
        self, tmp_path: Path, candidates: List[CandidateQuestion], inputs: List[str]
    ):
        candidates_path = tmp_path / "dummy.candidates.jsonl"
        save_candidates(candidates_path, candidates)

        inputs_iter = iter(inputs)
        outputs: List[str] = []

        def fake_input(prompt: str) -> str:
            try:
                return next(inputs_iter)
            except StopIteration:
                return "q"

        def fake_output(msg: str) -> None:
            outputs.append(msg)

        n_curated, n_rejected = run_review(
            candidates_path, input_fn=fake_input, output_fn=fake_output
        )
        return n_curated, n_rejected, outputs, candidates_path

    def test_keep_discard_skip(self, tmp_path: Path):
        cands = [_candidate("a"), _candidate("b"), _candidate("c")]
        inputs = [
            "k",  # keep a
            "d",
            "weak_distractor",  # discard b with reason
            "s",  # skip c (no decision)
        ]
        n_c, n_r, _, base = self._run(tmp_path, cands, inputs)
        assert (n_c, n_r) == (1, 1)
        curated = load_candidates(base.parent / "dummy.curated.jsonl")
        rejected = load_candidates(base.parent / "dummy.rejected.jsonl")
        assert [c.question_id for c in curated] == ["a"]
        assert [c.question_id for c in rejected] == ["b"]
        assert rejected[0].review_reason == "weak_distractor"

    def test_edit_overrides_question_and_answer(self, tmp_path: Path):
        cands = [_candidate("a")]
        inputs = [
            "e",
            "What colour is the ball?",  # new question
            "red",  # new answer
        ]
        _, _, _, base = self._run(tmp_path, cands, inputs)
        curated = load_candidates(base.parent / "dummy.curated.jsonl")
        assert len(curated) == 1
        assert curated[0].question == "What colour is the ball?"
        assert curated[0].answer == "red"
        assert curated[0].review_decision == ReviewDecision.EDITED.value

    def test_note_preserves_candidate_pending(self, tmp_path: Path):
        cands = [_candidate("a")]
        # note, then re-display, then keep. The note path shouldn't
        # decide anything — only the final k does.
        inputs = ["n", "revisit after DRM rubric calibration", "?", "k"]
        n_c, _, _, base = self._run(tmp_path, cands, inputs)
        assert n_c == 1
        curated = load_candidates(base.parent / "dummy.curated.jsonl")
        assert curated[0].review_decision == ReviewDecision.KEEP.value
        assert "revisit after DRM rubric calibration" in curated[0].generator_notes

    def test_quit_persists_partial_progress(self, tmp_path: Path):
        cands = [_candidate("a"), _candidate("b")]
        inputs = ["k", "q"]
        n_c, n_r, _, base = self._run(tmp_path, cands, inputs)
        assert (n_c, n_r) == (1, 0)
        # Resume: run again with only the remaining candidate to review.
        inputs = ["d", "too_easy"]
        n_c, n_r, _, _ = self._run(tmp_path, cands, inputs)
        assert (n_c, n_r) == (1, 1)
        curated = load_candidates(base.parent / "dummy.curated.jsonl")
        rejected = load_candidates(base.parent / "dummy.rejected.jsonl")
        assert [c.question_id for c in curated] == ["a"]
        assert [c.question_id for c in rejected] == ["b"]


def test_paradigm_generator_is_abstract():
    """ABC enforcement: can't instantiate ParadigmGenerator directly."""
    with pytest.raises(TypeError):
        ParadigmGenerator()  # type: ignore[abstract]
