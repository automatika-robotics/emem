from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrajectoryFrame:
    """A single observation frame in a replay trajectory."""

    frame_id: str
    position: Tuple[float, float, float]
    timestamp: float
    text: str
    layer_name: str = "description"
    image_path: Optional[str] = None
    is_interoception: bool = False


@dataclass
class BenchmarkQuestion:
    """A question to evaluate against a trajectory."""

    question_id: str
    question: str
    answer: str
    category: str = ""
    extra_answers: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSample:
    """A complete evaluation sample: trajectory + questions."""

    sample_id: str
    scene_id: str
    trajectory: List[TrajectoryFrame]
    questions: List[BenchmarkQuestion]
    agent_position: Optional[Tuple[float, float, float]] = None
    agent_situation: str = ""
