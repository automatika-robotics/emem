"""Unit tests for the LoCoMo / eMEM-Bench answer post-processor."""

from __future__ import annotations

from harness.postprocess import clean_answer


def test_plain_answer_passes_through() -> None:
    assert clean_answer("Paris") == "Paris"
    assert clean_answer("3 apples") == "3 apples"


def test_strips_thought_prefix() -> None:
    raw = "Thought: I need to search for this.\nFinal Answer: Paris"
    assert clean_answer(raw) == "Paris"


def test_strips_action_and_observation_lines() -> None:
    raw = (
        "Action: semantic_search\n"
        'Action Input: {"query": "capital"}\n'
        "Observation: Paris is the capital of France.\n"
        "Paris"
    )
    assert clean_answer(raw) == "Paris"


def test_final_answer_prefix_stripped() -> None:
    assert clean_answer("Final Answer: 42") == "42"
    assert clean_answer("final answer: 42") == "42"


def test_takes_first_nonempty_line() -> None:
    raw = "\n\nRed\nAdditional commentary we do not want"
    assert clean_answer(raw) == "Red"


def test_unanswerable_maps_to_empty() -> None:
    assert clean_answer("UNANSWERABLE") == ""
    assert clean_answer("Unanswerable.") == ""
    assert clean_answer("unknown") == ""


def test_not_found_phrases_map_to_empty() -> None:
    assert clean_answer("Not found in memory") == ""
    assert clean_answer("No information available") == ""
    assert clean_answer("No record found in memory") == ""
    assert clean_answer("not in our records") == ""
    assert clean_answer("Insufficient information to answer") == ""


def test_none_recorded_maps_to_empty() -> None:
    assert clean_answer("None recorded") == ""


def test_real_answer_containing_no_is_preserved() -> None:
    # Answers that mention "no" but aren't unanswerable-patterns should pass.
    assert clean_answer("No, she did not attend.") == "No, she did not attend."


def test_whitespace_only_input_is_empty() -> None:
    assert clean_answer("   \n\t\n") == ""


def test_leaked_thought_on_multiple_lines_stripped() -> None:
    # The regex strips lines that begin with one of the trigger tokens
    # followed by ":" or whitespace — matches the ReAct output format.
    raw = (
        "Thought: I should check memory.\n"
        "Action: semantic_search\n"
        "Observation: found it\n"
        "The answer is blue."
    )
    assert clean_answer(raw) == "The answer is blue."
