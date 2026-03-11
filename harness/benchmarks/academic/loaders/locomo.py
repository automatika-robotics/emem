import json
import os
from typing import Any, Dict, Iterator, List, Optional

from harness.benchmarks.academic.trajectory import (
    BenchmarkQuestion,
    BenchmarkSample,
    TrajectoryFrame,
)


class LoCoMoLoader:
    """Loader for the LoCoMo conversational memory benchmark.

    All observations are placed at origin ``(0, 0, 0)`` since LoCoMo
    tests temporal/semantic retrieval only (no spatial data).
    """

    def __init__(self, data_dir: str, max_conversations: Optional[int] = None):
        """
        :param data_dir: Directory containing ``locomo.json`` or individual
            conversation JSON files.
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

            conv_id = str(conv.get("conversation_id", count))
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

        Supports a single ``locomo.json`` file (array or ``{"conversations": [...]}``
        format) or a directory of individual JSON files.

        :returns: List of conversation dicts.
        """
        single_path = os.path.join(self._data_dir, "locomo.json")
        if os.path.exists(single_path):
            with open(single_path) as f:
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
        """Convert conversation turns into trajectory frames at origin.

        :param conv: Conversation dict with ``"sessions"`` or ``"conversation"`` key.
        :returns: List of frames, one per turn.
        """
        frames: List[TrajectoryFrame] = []
        sessions = conv.get("sessions", [])
        if not sessions and "conversation" in conv:
            sessions = [{"turns": conv["conversation"]}]

        global_turn = 0
        for session in sessions:
            turns = session.get("turns", session.get("messages", []))
            for turn in turns:
                speaker = turn.get("speaker", turn.get("role", "unknown"))
                text = turn.get("text", turn.get("content", ""))
                ts = float(turn.get("timestamp", global_turn * 60.0))

                frames.append(TrajectoryFrame(
                    frame_id=f"turn_{global_turn}",
                    position=(0.0, 0.0, 0.0),
                    timestamp=ts,
                    text=f"[{speaker}]: {text}",
                    layer_name="conversation",
                ))
                global_turn += 1

        return frames

    @staticmethod
    def _build_questions(conv: Dict[str, Any]) -> List[BenchmarkQuestion]:
        """Extract QA pairs from conversation data.

        :param conv: Conversation dict with ``"qa_pairs"`` or ``"questions"`` key.
        :returns: List of benchmark questions.
        """
        questions: List[BenchmarkQuestion] = []
        qa_pairs = conv.get("qa_pairs", conv.get("questions", []))

        for i, qa in enumerate(qa_pairs):
            questions.append(BenchmarkQuestion(
                question_id=str(qa.get("question_id", qa.get("id", i))),
                question=qa.get("question", qa.get("query", "")),
                answer=qa.get("answer", qa.get("response", "")),
                category=qa.get("category", qa.get("type", "")),
            ))

        return questions
