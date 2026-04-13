from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)

log = logging.getLogger(__name__)


def _parse_locomo_timestamp(ts_str: str) -> float:
    """Parse a LoCoMo timestamp string like ``"1:56 pm on 8 May, 2023"``.

    :param ts_str: Raw timestamp string from the dataset.
    :returns: Unix-style timestamp (seconds). Falls back to 0.0 on failure.
    """
    try:
        # "1:56 pm on 8 May, 2023" -> parse with datetime
        cleaned = ts_str.replace(" on ", " ").strip()
        dt = datetime.strptime(cleaned, "%I:%M %p %d %B, %Y")
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


class LoCoMoLoader:
    """Loader for the LoCoMo conversational memory benchmark.

    All observations are placed at origin ``(0, 0, 0)`` since LoCoMo
    tests temporal/semantic retrieval only (no spatial data).

    The dataset file ``locomo10.json`` is a list of 10 conversations.
    Each conversation has sessions keyed as ``session_1``, ``session_2``,
    etc., with corresponding ``session_1_date_time`` timestamps. QA pairs
    are in the ``qa`` key.
    """

    def __init__(self, data_dir: str, max_conversations: Optional[int] = None):
        """
        :param data_dir: Directory containing ``locomo10.json`` (or ``locomo.json``).
        :param max_conversations: Maximum number of conversations to load.
        """
        self._data_dir = data_dir
        self._max_conversations = max_conversations

    @property
    def name(self) -> str:
        return "locomo"

    def load(self) -> Iterator[BenchmarkSample]:
        """Yield one :class:`BenchmarkSample` per conversation."""
        conversations = self._load_data()
        count = 0

        for conv in conversations:
            if self._max_conversations is not None and count >= self._max_conversations:
                break

            conv_id = str(conv.get("sample_id", count))
            trajectory = self._build_trajectory(conv)
            questions = self._build_questions(conv)

            if not trajectory or not questions:
                continue

            yield BenchmarkSample(
                sample_id=conv_id,
                scene_id=conv_id,
                trajectory=trajectory,
                questions=questions,
            )
            count += 1

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load conversation data from disk.

        Tries ``locomo10.json`` first (official filename), then ``locomo.json``,
        then falls back to loading individual JSON files.

        :returns: List of conversation dicts.
        """
        for filename in ("locomo10.json", "locomo.json"):
            path = os.path.join(self._data_dir, filename)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if "conversations" in data:
                    return data["conversations"]
                return [data]

        conversations: List[Dict[str, Any]] = []
        for fname in sorted(os.listdir(self._data_dir)):
            if fname.endswith(".json") and fname != "metadata.json":
                with open(os.path.join(self._data_dir, fname)) as f:
                    conversations.append(json.load(f))
        return conversations

    @staticmethod
    def _build_trajectory(conv: Dict[str, Any]) -> List[TrajectoryFrame]:
        """Convert conversation sessions into trajectory frames at origin.

        Handles the LoCoMo format where sessions are keyed as ``session_1``,
        ``session_2``, etc. within the ``conversation`` dict, with timestamps
        in ``session_1_date_time``, etc.

        :param conv: Conversation dict with ``"conversation"`` key.
        :returns: List of frames, one per turn.
        """
        frames: List[TrajectoryFrame] = []
        conversation = conv.get("conversation", conv)

        # Find session keys: session_1, session_2, ...
        session_nums = sorted(
            int(m.group(1))
            for key in conversation
            if (m := re.match(r"session_(\d+)$", key))
        )

        global_turn = 0
        for num in session_nums:
            session_key = f"session_{num}"
            ts_key = f"session_{num}_date_time"

            turns = conversation.get(session_key, [])
            raw_ts = conversation.get(ts_key, "")
            session_ts = _parse_locomo_timestamp(raw_ts)

            # Build a readable date prefix for this session
            date_prefix = f"[Session {num} — {raw_ts}] " if raw_ts else ""

            for turn_idx, turn in enumerate(turns):
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                # Use session timestamp + offset per turn within session
                ts = (
                    session_ts + turn_idx * 30.0
                    if session_ts > 0
                    else global_turn * 60.0
                )

                frames.append(
                    TrajectoryFrame(
                        frame_id=turn.get("dia_id", f"turn_{global_turn}"),
                        position=(0.0, 0.0, 0.0),
                        timestamp=ts,
                        text=f"{date_prefix}[{speaker}]: {text}",
                        layer_name="conversation",
                    )
                )
                global_turn += 1

        return frames

    @staticmethod
    def _build_questions(conv: Dict[str, Any]) -> List[BenchmarkQuestion]:
        """Extract QA pairs from conversation data.

        :param conv: Conversation dict with ``"qa"`` key.
        :returns: List of benchmark questions.
        """
        questions: List[BenchmarkQuestion] = []
        qa_pairs = conv.get("qa", conv.get("qa_pairs", conv.get("questions", [])))

        for i, qa in enumerate(qa_pairs):
            questions.append(
                BenchmarkQuestion(
                    question_id=str(qa.get("question_id", qa.get("id", i))),
                    question=str(qa.get("question", qa.get("query", ""))),
                    answer=str(qa.get("answer", qa.get("response", ""))),
                    category=str(qa.get("category", qa.get("type", ""))),
                )
            )

        return questions
