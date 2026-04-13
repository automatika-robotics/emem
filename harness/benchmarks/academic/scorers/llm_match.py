from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict

log = logging.getLogger(__name__)

LLM_MATCH_PROMPT = """\
You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Here are some examples:
Question: What color is the car?
Answer: Red
Response: The car is red.
Your mark: 5

Question: How many chairs are there?
Answer: 3
Response: There are four chairs in the room.
Your mark: 2

Question: What is on the table?
Answer: A laptop and a cup
Response: I can see a laptop on the table.
Your mark: 3

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}
Your mark:"""


class LLMMatchScorer:
    """Scores predictions using an LLM judge (1-5 scale, normalized to 0-100)."""

    def __init__(self, llm_chat: Callable[[str], str]):
        """
        :param llm_chat: Function that takes a prompt string and returns the
            LLM's text response.
        """
        self._llm_chat = llm_chat

    @property
    def name(self) -> str:
        return "llm_match"

    def score(
        self, question: str, prediction: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Score prediction via LLM judge on a 1-5 scale, normalized to 0-100.

        :param question: The benchmark question.
        :param prediction: The model's predicted answer.
        :param ground_truth: The reference answer.
        :returns: Dict with ``"score"`` (0-100) and ``"raw_mark"`` (1-5).
        """
        prompt = LLM_MATCH_PROMPT.format(
            question=question,
            answer=ground_truth,
            prediction=prediction,
        )
        response = self._llm_chat(prompt)
        raw = self._parse_mark(response)
        normalized = 100.0 * (max(1, min(5, raw)) - 1) / 4
        return {"score": normalized, "raw_mark": raw}

    @staticmethod
    def _parse_mark(text: str) -> int:
        """Extract integer mark (1-5) from LLM response.

        :param text: Raw LLM response text.
        :returns: Integer mark 1-5. Defaults to 1 if unparseable.
        """
        match = re.search(r"\b([1-5])\b", text)
        if match:
            return int(match.group(1))
        log.warning("Could not parse LLM judge mark from: %s", text[:100])
        return 1
