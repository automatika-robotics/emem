"""Tests for core data types."""

import numpy as np
import pytest

from emem.types import (
    Edge,
    EdgeType,
    EpisodeNode,
    EpisodeStatus,
    GistNode,
    ObservationNode,
    Tier,
)


class TestObservationNode:
    def test_create_minimal(self):
        obs = ObservationNode(
            text="A red chair",
            coordinates=np.array([1.0, 2.0, 0.0]),
            timestamp=1000.0,
        )
        assert obs.text == "A red chair"
        assert obs.id  # auto-generated
        assert obs.layer_name == "default"
        assert obs.source_type == "manual"
        assert obs.confidence == 1.0
        assert obs.tier == Tier.SHORT_TERM.value
        assert obs.embedding is None
        assert obs.episode_id is None

    def test_create_full(self):
        emb = np.random.randn(384).astype(np.float32)
        obs = ObservationNode(
            text="Detected person",
            coordinates=np.array([5.0, 10.0, 1.5]),
            timestamp=2000.0,
            layer_name="detections",
            source_type="detection",
            confidence=0.95,
            episode_id="ep-123",
            metadata={"model": "yolo"},
            tier=Tier.WORKING.value,
            embedding=emb,
        )
        assert obs.layer_name == "detections"
        assert obs.confidence == 0.95
        assert obs.embedding is not None
        assert obs.metadata["model"] == "yolo"

    def test_unique_ids(self):
        a = ObservationNode(text="a", coordinates=np.zeros(3), timestamp=0)
        b = ObservationNode(text="b", coordinates=np.zeros(3), timestamp=0)
        assert a.id != b.id


class TestEpisodeNode:
    def test_create(self):
        ep = EpisodeNode(name="patrol_zone_a", start_time=1000.0)
        assert ep.name == "patrol_zone_a"
        assert ep.status == EpisodeStatus.ACTIVE.value
        assert ep.end_time is None
        assert ep.gist == ""

    def test_hierarchical(self):
        parent = EpisodeNode(name="mission", start_time=0)
        child = EpisodeNode(name="subtask", start_time=10, parent_episode_id=parent.id)
        assert child.parent_episode_id == parent.id


class TestGistNode:
    def test_create(self):
        gist = GistNode(
            text="Area around (3,4) contains chairs and tables",
            center_position=np.array([3.0, 4.0, 0.0]),
            radius=2.5,
            time_start=1000.0,
            time_end=2000.0,
            source_observation_count=5,
            source_observation_ids=["a", "b", "c", "d", "e"],
        )
        assert gist.source_observation_count == 5
        assert gist.radius == 2.5


class TestEdge:
    def test_create(self):
        edge = Edge(
            source_id="obs-1",
            target_id="ep-1",
            edge_type=EdgeType.BELONGS_TO,
        )
        assert edge.edge_type == EdgeType.BELONGS_TO
        assert edge.id  # auto-generated


class TestEdgeType:
    def test_values(self):
        assert EdgeType.BELONGS_TO.value == "belongs_to"
        assert EdgeType.FOLLOWS.value == "follows"
        assert EdgeType.SUBTASK_OF.value == "subtask_of"
        assert EdgeType.SUMMARIZES.value == "summarizes"
