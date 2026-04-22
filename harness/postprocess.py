"""Post-processing for agent answers on academic benchmarks.

This module extracts a single responsibility previously inlined in
``replay_runner.py``: cleaning raw agent output before scoring. The rules
applied here match what was used for the reported LoCoMo and
eMEM-Bench numbers.

Rules applied to the raw agent output, in order:

1. Strip leaked meta-commentary lines that ReAct agents sometimes emit
   (``Thought:``, ``Wait``, ``Hmm``, ``Let me``, ``Action``,
   ``Action Input``, ``Observation``, ``So,``, ``Based on``).
2. Strip a leading ``Final Answer:`` prefix.
3. Take only the first non-empty line (benchmark answers are short).
4. Map unanswerable / not-found responses to the empty string so they
   score correctly against empty ground truth on LoCoMo adversarial
   questions.

The regex and rule set are intentionally conservative: they fire on
obvious leakage but do not rewrite answer content.
"""

from __future__ import annotations

import re


UNANSWERABLE_RE = re.compile(
    r"^("
    r"unanswerable"
    r"|(?:no |not )?"
    r"(?:information|info|record|records|data|details?|mention|evidence|result|results)"
    r"(?:\s+(?:is\s+)?(?:not\s+)?(?:found|available|stored|recorded|specified|mentioned|documented|known|in memory))*"
    r".*"
    r"|not (?:found|available|stored|recorded|specified|mentioned|documented|known)(?: (?:in|from) (?:memory|records?|our records?|stored memories|conversation history?))?\.?"
    r"|(?:no (?:such )?(?:record|information|info|data|details?|mention|evidence|result|results) (?:found|available|in memory).*)"
    r"|(?:insufficient information.*)"
    r"|(?:unknown.*)"
    r"|not in (?:memory|records?|our records?|conversation history)"
    r"|none(?: recorded| specified| found)?"
    r")$",
    re.IGNORECASE,
)

_LEAK_RE = re.compile(
    r"^(?:Thought|Wait|Hmm|Let me|Action|Action Input|Observation|So,|Based on)[:\s].*?(?:\n|$)",
    re.MULTILINE | re.IGNORECASE,
)

_FINAL_PREFIX_RE = re.compile(r"^Final Answer:\s*", re.IGNORECASE)


def clean_answer(answer: str) -> str:
    """Post-process an agent answer for scoring.

    :param answer: Raw agent answer as emitted by the ReAct loop.
    :returns: Cleaned answer string; empty string if the answer maps to
        "unanswerable / not found / unknown".
    """
    answer = _LEAK_RE.sub("", answer)
    answer = _FINAL_PREFIX_RE.sub("", answer)
    answer = answer.strip()
    for line in answer.split("\n"):
        line = line.strip()
        if line:
            answer = line
            break
    if UNANSWERABLE_RE.match(answer.strip().rstrip(".")):
        return ""
    return answer
