"""Category-aware scorer for eMEM-Bench.

Uses LLM-based scoring for all categories.  Interoception answers are
structured strings (e.g. ``"battery: 85%"``) but the agent naturally
paraphrases them ("Your battery level is 85%"), so the LLM judge is
needed to match the value correctly.
"""

from typing import Any, Callable, Dict

from harness.benchmarks.academic.scorers.llm_match import LLMMatchScorer


class EMEMBenchScorer:
    """Category-aware scorer for eMEM-Bench."""

    def __init__(self, llm_chat: Callable[[str], str]):
        self._llm_scorer = LLMMatchScorer(llm_chat=llm_chat)

    @property
    def name(self) -> str:
        return "emem_bench"

    def score(
        self, question: str, prediction: str, ground_truth: str
    ) -> Dict[str, Any]:
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
        """Score using the LLM judge for all categories.

        :param question: The benchmark question.
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :param category: Question category (unused — LLM judge for all).
        :returns: Score dict.
        """
        return self._llm_scorer.score(question, prediction, ground_truth)
