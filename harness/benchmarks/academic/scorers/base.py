from typing import Any, Dict, Protocol


class Scorer(Protocol):
    """Protocol for scoring benchmark predictions against ground truth."""

    def score(
        self, question: str, prediction: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Score a single prediction.

        :param question: The benchmark question text.
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :returns: Dict with at least a ``"score"`` key (float 0-100).
        """
        ...

    @property
    def name(self) -> str:
        """Scorer name for reporting."""
        ...
