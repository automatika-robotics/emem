import pytest

from harness.benchmarks.academic.scorers.exact_match import ExactMatchScorer
from harness.benchmarks.academic.scorers.f1 import F1Scorer
from harness.benchmarks.academic.scorers.llm_match import LLMMatchScorer


class TestExactMatchScorer:
    def test_exact_match(self):
        s = ExactMatchScorer()
        r = s.score("q", "chair", "chair")
        assert r["score"] == 100.0
        assert r["match"] is True

    def test_case_insensitive(self):
        s = ExactMatchScorer()
        r = s.score("q", "Chair", "chair")
        assert r["score"] == 100.0

    def test_strip_articles(self):
        s = ExactMatchScorer()
        r = s.score("q", "a red chair", "the red chair")
        assert r["score"] == 100.0

    def test_strip_punctuation(self):
        s = ExactMatchScorer()
        r = s.score("q", "yes.", "yes")
        assert r["score"] == 100.0

    def test_mismatch(self):
        s = ExactMatchScorer()
        r = s.score("q", "table", "chair")
        assert r["score"] == 0.0
        assert r["match"] is False

    def test_whitespace_collapse(self):
        s = ExactMatchScorer()
        r = s.score("q", "red   chair", "red chair")
        assert r["score"] == 100.0


class TestF1Scorer:
    def test_perfect_match(self):
        s = F1Scorer()
        r = s.score("q", "the cat sat on the mat", "the cat sat on the mat")
        assert r["f1"] == pytest.approx(1.0)

    def test_partial_overlap(self):
        s = F1Scorer()
        r = s.score("q", "the cat sat", "the cat sat on the mat")
        assert 0 < r["f1"] < 1.0

    def test_no_overlap(self):
        s = F1Scorer()
        r = s.score("q", "hello world", "foo bar")
        assert r["f1"] == 0.0

    def test_empty_prediction(self):
        s = F1Scorer()
        r = s.score("q", "", "some answer")
        assert r["f1"] == 0.0
        assert r["bleu1"] == 0.0

    def test_bleu1_computed(self):
        s = F1Scorer()
        r = s.score("q", "the cat sat on the mat", "the cat sat on the mat")
        assert r["bleu1"] > 0

    def test_score_is_f1_times_100(self):
        s = F1Scorer()
        r = s.score("q", "some answer text", "answer text here")
        assert r["score"] == pytest.approx(r["f1"] * 100)


class TestLLMMatchScorer:
    def test_parse_mark(self):
        assert LLMMatchScorer._parse_mark("Your mark: 5") == 5
        assert LLMMatchScorer._parse_mark("3") == 3
        assert LLMMatchScorer._parse_mark("I give this a 4 out of 5") == 4
        assert LLMMatchScorer._parse_mark("no number here") == 1

    def test_score_with_mock_llm(self):
        scorer = LLMMatchScorer(llm_chat=lambda p: "Your mark: 4")
        r = scorer.score("What color?", "red", "The car is red")
        assert r["score"] == pytest.approx(75.0)
        assert r["raw_mark"] == 4

    def test_normalization_bounds(self):
        scorer_low = LLMMatchScorer(llm_chat=lambda p: "1")
        assert scorer_low.score("q", "a", "b")["score"] == 0.0

        scorer_high = LLMMatchScorer(llm_chat=lambda p: "5")
        assert scorer_high.score("q", "a", "b")["score"] == 100.0
