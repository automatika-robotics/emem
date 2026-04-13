from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List


class F1Scorer:
    """Computes token-based F1 and BLEU-1 following the REMem protocol."""

    @property
    def name(self) -> str:
        return "f1"

    def score(
        self, question: str, prediction: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Compute token F1 and BLEU-1 scores.

        :param question: The benchmark question (unused, kept for protocol).
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :returns: Dict with ``"score"`` (F1*100), ``"f1"``, and ``"bleu1"``.
        """
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(ground_truth)

        f1 = self._compute_f1(pred_tokens, gt_tokens)
        bleu1 = self._compute_bleu1(pred_tokens, gt_tokens)

        return {"score": f1 * 100, "f1": f1, "bleu1": bleu1}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Split text into lowercase word tokens.

        :param text: Raw text.
        :returns: List of lowercase tokens.
        """
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _compute_f1(pred: List[str], gold: List[str]) -> float:
        """Compute token-level F1 between prediction and gold tokens.

        :param pred: Predicted tokens.
        :param gold: Gold reference tokens.
        :returns: F1 score in [0, 1].
        """
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        pred_counts = Counter(pred)
        gold_counts = Counter(gold)
        overlap = sum((pred_counts & gold_counts).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _compute_bleu1(pred: List[str], gold: List[str]) -> float:
        """Compute unigram BLEU with brevity penalty.

        :param pred: Predicted tokens.
        :param gold: Gold reference tokens.
        :returns: BLEU-1 score in [0, 1].
        """
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        pred_counts = Counter(pred)
        gold_counts = Counter(gold)
        clipped = sum(min(pred_counts[w], gold_counts[w]) for w in pred_counts)
        precision = clipped / len(pred)
        bp = math.exp(min(0, 1 - len(gold) / len(pred)))
        return bp * precision
