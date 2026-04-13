from __future__ import annotations

import re
import string
from typing import Any, Dict


class ExactMatchScorer:
    """Scores predictions by exact match after normalization."""

    @property
    def name(self) -> str:
        return "exact_match"

    def score(
        self, question: str, prediction: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Score by exact match after normalizing both strings.

        :param question: The benchmark question (unused, kept for protocol).
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :returns: Dict with ``"score"`` (0 or 100) and ``"match"`` (bool).
        """
        pred_norm = self._normalize(prediction)
        gt_norm = self._normalize(ground_truth)
        match = pred_norm == gt_norm
        return {"score": 100.0 if match else 0.0, "match": match}

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip articles/punctuation, collapse whitespace.

        :param text: Raw text to normalize.
        :returns: Normalized text.
        """
        text = text.lower().strip()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text
