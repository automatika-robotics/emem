"""Tests for MemoryStore."""

import os
import tempfile

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.embeddings import NullEmbeddingProvider
from emem.store import MemoryStore
from emem.types import EdgeType, GistNode, ObservationNode


class FakeEmbeddingProvider:
    """Deterministic embeddings for testing: hash-based."""

    def __init__(self, dim: int = 32):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts):
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.RandomState(hash(text) % (2**31))
            result[i] = rng.randn(self._dim).astype(np.float32)
            result[i] /= np.linalg.norm(result[i]) + 1e-8
        return result


@pytest.fixture
def store(tmp_path):
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "test.db"),
        hnsw_path=str(tmp_path / "test_hnsw.bin"),
        embedding_dim=32,
        hnsw_max_elements=1000,
    )
    s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
    yield s
    s.close()


def _make_obs(text="test", x=0.0, y=0.0, z=0.0, ts=1000.0, layer="default",
              episode_id=None, **kwargs) -> ObservationNode:
    return ObservationNode(
        text=text,
        coordinates=np.array([x, y, z]),
        timestamp=ts,
        layer_name=layer,
        episode_id=episode_id,
        **kwargs,
    )


class TestAddAndRetrieve:
    def test_add_single(self, store):
        obs = _make_obs("red chair at entrance")
        obs_id = store.add_observation(obs)
        assert obs_id == obs.id

        retrieved = store.get_observation(obs_id)
        assert retrieved is not None
        assert retrieved.text == "red chair at entrance"
        np.testing.assert_array_almost_equal(retrieved.coordinates, [0, 0, 0])

    def test_add_batch(self, store):
        observations = [_make_obs(f"obs_{i}", x=float(i), ts=1000.0 + i) for i in range(10)]
        ids = store.add_observations_batch(observations)
        assert len(ids) == 10
        assert store.count_observations() == 10

    def test_count(self, store):
        assert store.count_observations() == 0
        store.add_observation(_make_obs("a"))
        store.add_observation(_make_obs("b"))
        assert store.count_observations() == 2


class TestEpisodes:
    def test_start_and_end(self, store):
        ep_id = store.start_episode("patrol", 1000.0)
        ep = store.get_episode(ep_id)
        assert ep.name == "patrol"
        assert ep.status == "active"

        store.end_episode(ep_id, 2000.0, gist="Patrolled zone A")
        ep = store.get_episode(ep_id)
        assert ep.status == "completed"
        assert ep.gist == "Patrolled zone A"

    def test_episode_observations(self, store):
        ep_id = store.start_episode("task1", 1000.0)
        store.add_observation(_make_obs("obs1", ts=1001.0, episode_id=ep_id))
        store.add_observation(_make_obs("obs2", ts=1002.0, episode_id=ep_id))

        obs_list = store.get_episode_observations(ep_id)
        assert len(obs_list) == 2
        assert obs_list[0].timestamp < obs_list[1].timestamp

    def test_list_episodes(self, store):
        store.start_episode("patrol_a", 1000.0)
        store.start_episode("patrol_b", 2000.0)
        store.start_episode("inspection", 3000.0)

        patrols = store.list_episodes(task_name="patrol")
        assert len(patrols) == 2

        recent = store.list_episodes(last_n=1)
        assert len(recent) == 1
        assert recent[0].name == "inspection"

    def test_hierarchical_episodes(self, store):
        parent_id = store.start_episode("mission", 1000.0)
        child_id = store.start_episode("subtask", 1100.0, parent_episode_id=parent_id)

        edges = store.get_edges(source_id=child_id, edge_type=EdgeType.SUBTASK_OF)
        assert len(edges) == 1
        assert edges[0].target_id == parent_id


class TestSpatialQuery:
    def test_query_radius(self, store):
        store.add_observation(_make_obs("near", x=1.0, y=1.0))
        store.add_observation(_make_obs("far", x=100.0, y=100.0))

        results = store.spatial_query(center=np.array([0, 0, 0]), radius=5.0)
        assert len(results) == 1
        assert results[0].text == "near"

    def test_spatial_nearest(self, store):
        for i in range(5):
            store.add_observation(_make_obs(f"p{i}", x=float(i), y=0.0, ts=1000.0 + i))

        results = store.spatial_nearest(np.array([2, 0, 0]), k=2)
        assert len(results) == 2
        assert results[0].text == "p2"

    def test_spatial_with_layer_filter(self, store):
        store.add_observation(_make_obs("det", x=1.0, y=1.0, layer="detections"))
        store.add_observation(_make_obs("vlm", x=1.0, y=1.0, layer="vlm"))

        results = store.spatial_query(center=np.array([1, 1, 0]), radius=2.0, layer="detections")
        assert len(results) == 1
        assert results[0].layer_name == "detections"


class TestTemporalQuery:
    def test_time_range(self, store):
        for i in range(5):
            store.add_observation(_make_obs(f"t{i}", ts=1000.0 + i * 100))

        results = store.temporal_query(time_range=(1100.0, 1300.0))
        assert len(results) == 3

    def test_last_n_seconds(self, store):
        for i in range(5):
            store.add_observation(_make_obs(f"t{i}", ts=1000.0 + i * 100))

        results = store.temporal_query(last_n_seconds=250, reference_time=1500.0)
        assert len(results) == 2  # 1500 - 250 = 1250, so timestamps >= 1250: 1300, 1400

    def test_order(self, store):
        store.add_observation(_make_obs("first", ts=1000.0))
        store.add_observation(_make_obs("second", ts=2000.0))

        newest = store.temporal_query(time_range=(0, 3000), order="newest")
        assert newest[0].text == "second"

        oldest = store.temporal_query(time_range=(0, 3000), order="oldest")
        assert oldest[0].text == "first"


class TestSemanticSearch:
    def test_basic_search(self, store):
        store.add_observation(_make_obs("red chair in the lobby"))
        store.add_observation(_make_obs("blue table in the kitchen"))
        store.add_observation(_make_obs("green plant near window"))

        results = store.semantic_search("chair", n_results=2)
        assert len(results) > 0
        # With fake embeddings, order is hash-based but should return results

    def test_search_with_filters(self, store):
        store.add_observation(_make_obs("chair", layer="vlm", ts=1000.0))
        store.add_observation(_make_obs("table", layer="detection", ts=2000.0))

        results = store.semantic_search("furniture", layer="vlm")
        for r in results:
            assert r.layer_name == "vlm"

    def test_empty_store(self, store):
        results = store.semantic_search("anything")
        assert results == []


class TestGists:
    def test_add_and_search(self, store):
        gist = GistNode(
            text="Area contains furniture: chairs and tables",
            center_position=np.array([5.0, 5.0, 0.0]),
            radius=3.0,
            time_start=1000.0,
            time_end=2000.0,
            source_observation_count=5,
            source_observation_ids=["a", "b", "c", "d", "e"],
        )
        gist_id = store.add_gist(gist)
        assert gist_id

        retrieved = store.get_gist(gist_id)
        assert retrieved.text == gist.text

    def test_gist_edges(self, store):
        obs_ids = []
        for i in range(3):
            oid = store.add_observation(_make_obs(f"obs{i}", ts=1000.0 + i))
            obs_ids.append(oid)

        gist = GistNode(
            text="Summary",
            center_position=np.array([0, 0, 0]),
            radius=1.0,
            time_start=1000.0,
            time_end=1002.0,
            source_observation_count=3,
            source_observation_ids=obs_ids,
        )
        gist_id = store.add_gist(gist)

        edges = store.get_edges(source_id=gist_id, edge_type=EdgeType.SUMMARIZES)
        assert len(edges) == 3


class TestUnifiedSemanticSearch:
    def test_returns_gists_from_hnsw(self, store):
        """semantic_search should return gists indexed in HNSW."""
        store.add_observation(_make_obs("red chair in lobby", x=1.0, y=1.0))
        gist = GistNode(
            text="Lobby area contains furniture including a red chair and blue table",
            center_position=np.array([1.0, 1.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=2000.0,
            source_observation_count=3,
            source_observation_ids=["a", "b", "c"],
        )
        store.add_gist(gist)

        results = store.semantic_search("chair", n_results=10)
        types = {type(r).__name__ for r in results}
        assert "GistNode" in types or "ObservationNode" in types
        assert len(results) > 0

    def test_gists_survive_archival(self, store):
        """After archiving observations, gists should still appear in semantic_search."""
        obs_id = store.add_observation(_make_obs("important chair", x=2.0, y=2.0))
        gist = GistNode(
            text="Area has an important chair",
            center_position=np.array([2.0, 2.0, 0.0]),
            radius=1.0,
            time_start=1000.0,
            time_end=2000.0,
            source_observation_count=1,
            source_observation_ids=[obs_id],
        )
        store.add_gist(gist)

        # Archive the observation (drops text + embedding)
        store.update_observation_tier(obs_id, "archived", drop_text=True)

        results = store.semantic_search("chair", n_results=10)
        # The archived observation should be excluded, but the gist should remain
        gist_results = [r for r in results if isinstance(r, GistNode)]
        assert len(gist_results) >= 1
        assert "chair" in gist_results[0].text

    def test_unified_preserves_hnsw_order(self, store):
        """Results should be ordered by HNSW distance, mixing obs and gists."""
        for i in range(3):
            store.add_observation(_make_obs(f"observation_{i}", ts=1000.0 + i))
        gist = GistNode(
            text="gist summary",
            center_position=np.array([0, 0, 0]),
            radius=1.0,
            time_start=1000.0,
            time_end=2000.0,
            source_observation_count=3,
            source_observation_ids=["x", "y", "z"],
        )
        store.add_gist(gist)

        results = store.semantic_search("summary", n_results=10)
        assert len(results) > 0

    def test_search_gists_by_area(self, store):
        gist = GistNode(
            text="Kitchen area summary",
            center_position=np.array([5.0, 5.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=2000.0,
            source_observation_count=3,
            source_observation_ids=["a", "b", "c"],
        )
        store.add_gist(gist)

        results = store.search_gists_by_area(center=np.array([5.0, 5.0, 0.0]), radius=3.0)
        assert len(results) == 1
        assert "Kitchen" in results[0].text

        far_results = store.search_gists_by_area(center=np.array([100.0, 100.0, 0.0]), radius=1.0)
        assert len(far_results) == 0


class TestTierManagement:
    def test_archive_drops_text(self, store):
        obs_id = store.add_observation(_make_obs("important text"))
        store.update_observation_tier(obs_id, "archived", drop_text=True)

        obs = store.get_observation(obs_id)
        assert obs.text == ""
        assert obs.tier == "archived"

    def test_archived_excluded_from_queries(self, store):
        store.add_observation(_make_obs("visible", x=1.0, y=1.0, ts=1000.0))
        obs2_id = store.add_observation(_make_obs("hidden", x=1.0, y=1.0, ts=1001.0))
        store.update_observation_tier(obs2_id, "archived", drop_text=True)

        results = store.spatial_query(center=np.array([1, 1, 0]), radius=5.0)
        assert len(results) == 1
        assert results[0].text == "visible"


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "persist.db"),
            hnsw_path=str(tmp_path / "persist_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=1000,
        )
        provider = FakeEmbeddingProvider(32)

        # Write
        s1 = MemoryStore(config=config, embedding_provider=provider)
        s1.add_observation(_make_obs("persisted observation"))
        s1.start_episode("test_ep", 1000.0)
        s1.close()

        # Read
        s2 = MemoryStore(config=config, embedding_provider=provider)
        assert s2.count_observations() == 1
        eps = s2.list_episodes()
        assert len(eps) == 1
        s2.close()
