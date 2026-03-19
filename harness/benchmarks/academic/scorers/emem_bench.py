"""Category-aware scorer for eMEM-Bench.

Uses LLM-based scoring for most categories but falls back to exact match
for interoception questions where answers are structured strings
(e.g. ``"battery: 85%"``).
"""

from typing import Any, Callable, Dict

from harness.benchmarks.academic.scorers.exact_match import ExactMatchScorer
from harness.benchmarks.academic.scorers.llm_match import LLMMatchScorer


class EMEMBenchScorer:
    """Category-aware scorer for eMEM-Bench."""

    def __init__(self, llm_chat: Callable[[str], str]):
        self._llm_scorer = LLMMatchScorer(llm_chat=llm_chat)
        self._exact_scorer = ExactMatchScorer()

    @property
    def name(self) -> str:
        return "emem_bench"

    def score(self, question: str, prediction: str, ground_truth: str) -> Dict[str, Any]:
        """Default scoring via LLM judge.

        :param question: The benchmark question.
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :returns: Score dict.
        """
        return self._llm_scorer.score(question, prediction, ground_truth)

    def score_with_category(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        category: str,
    ) -> Dict[str, Any]:
        """Score using a category-appropriate strategy.

        Interoception questions use exact match; all others use the LLM judge.

        :param question: The benchmark question.
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :param category: Question category (e.g. ``"interoception"``).
        :returns: Score dict.
        """
        if category == "interoception":
            return self._exact_scorer.score(question, prediction, ground_truth)
        return self._llm_scorer.score(question, prediction, ground_truth)
