import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class EdgeType(str, Enum):
    """Relationship kinds connecting memory nodes in the graph."""

    BELONGS_TO = "belongs_to"  # Observation -> Episode
    FOLLOWS = "follows"  # Episode -> Episode (temporal ordering)
    SUBTASK_OF = "subtask_of"  # Episode -> Episode (hierarchical)
    SUMMARIZES = "summarizes"  # Gist -> Observation(s)
    OBSERVED_IN = "observed_in"  # Entity -> Observation
    COOCCURS_WITH = "cooccurs_with"  # Entity <-> Entity


class Tier(str, Enum):
    """Storage tier assigned to a memory item by its consolidation state."""

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    ARCHIVED = "archived"


class EpisodeStatus(str, Enum):
    """Lifecycle state of an episode."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


def _new_id() -> str:
    """Generate a unique hex identifier for a new node or edge."""
    return uuid.uuid4().hex


@dataclass
class ObservationNode:
    """Single timestamped observation tied to a spatial location."""

    text: str
    coordinates: np.ndarray
    timestamp: float
    layer_name: str = "default"
    source_type: str = "manual"
    confidence: float = 1.0
    episode_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: str = Tier.SHORT_TERM.value
    id: str = field(default_factory=_new_id)
    embedding: Optional[np.ndarray] = None


@dataclass
class EpisodeNode:
    """Temporal grouping of observations representing a coherent episode."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = EpisodeStatus.ACTIVE.value
    gist: str = ""
    gist_embedding: Optional[np.ndarray] = None
    parent_episode_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)


@dataclass
class GistNode:
    """Consolidated summary of a cluster of related observations."""

    text: str
    center_position: np.ndarray
    radius: float
    time_start: float
    time_end: float
    source_observation_count: int
    source_observation_ids: List[str]
    layer_name: Optional[str] = None
    episode_id: Optional[str] = None
    id: str = field(default_factory=_new_id)
    embedding: Optional[np.ndarray] = None


@dataclass
class EntityNode:
    """Named entity tracked across observations with location and recency."""

    name: str
    coordinates: np.ndarray
    last_seen: float
    first_seen: float
    observation_count: int = 1
    confidence: float = 1.0
    entity_type: Optional[str] = None
    layer_name: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    embedding: Optional[np.ndarray] = None


@dataclass
class Edge:
    """Typed directed edge connecting two memory nodes."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
