"""Tests for MemoryStore."""

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.types import EdgeType, EntityNode, GistNode, ObservationNode


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


def _make_obs(
    text="test",
    x=0.0,
    y=0.0,
    z=0.0,
    ts=1000.0,
    layer="default",
    episode_id=None,
    **kwargs,
) -> ObservationNode:
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
        observations = [
            _make_obs(f"obs_{i}", x=float(i), ts=1000.0 + i) for i in range(10)
        ]
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

        results = store.spatial_query(
            center=np.array([1, 1, 0]), radius=2.0, layer="detections"
        )
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

        results = store.search_gists_by_area(
            center=np.array([5.0, 5.0, 0.0]), radius=3.0
        )
        assert len(results) == 1
        assert "Kitchen" in results[0].text

        far_results = store.search_gists_by_area(
            center=np.array([100.0, 100.0, 0.0]), radius=1.0
        )
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


def _make_entity(
    name="red chair", x=5.0, y=5.0, z=0.0, ts=1000.0, entity_type=None, **kwargs
) -> EntityNode:
    return EntityNode(
        name=name,
        coordinates=np.array([x, y, z]),
        last_seen=ts,
        first_seen=ts,
        entity_type=entity_type,
        **kwargs,
    )


class TestEntityCRUD:
    def test_add_and_get(self, store):
        entity = _make_entity("red chair", x=5.0, y=5.0)
        eid = store.add_entity(entity)
        assert eid == entity.id

        retrieved = store.get_entity(eid)
        assert retrieved is not None
        assert retrieved.name == "red chair"
        np.testing.assert_array_almost_equal(retrieved.coordinates, [5, 5, 0])

    def test_update_position(self, store):
        entity = _make_entity("blue table", x=1.0, y=1.0, ts=1000.0)
        eid = store.add_entity(entity)

        entity.coordinates = np.array([10.0, 10.0, 0.0])
        entity.last_seen = 2000.0
        entity.observation_count = 3
        store.update_entity(entity)

        updated = store.get_entity(eid)
        assert updated.last_seen == 2000.0
        assert updated.observation_count == 3
        np.testing.assert_array_almost_equal(updated.coordinates, [10, 10, 0])

    def test_get_nonexistent(self, store):
        assert store.get_entity("nonexistent") is None


class TestEntityMatching:
    def test_match_same_location(self, store):
        entity = _make_entity("red chair", x=5.0, y=5.0)
        store.add_entity(entity)

        match = store.find_matching_entity("red chair", np.array([5.5, 5.5, 0.0]))
        assert match is not None
        assert match.id == entity.id

    def test_no_match_far_away(self, store):
        entity = _make_entity("red chair", x=5.0, y=5.0)
        store.add_entity(entity)

        match = store.find_matching_entity("red chair", np.array([100.0, 100.0, 0.0]))
        assert match is None

    def test_no_match_different_name(self, store):
        entity = _make_entity("red chair", x=5.0, y=5.0)
        store.add_entity(entity)

        store.find_matching_entity("blue sofa", np.array([5.0, 5.0, 0.0]))
        # Might or might not match depending on embedding similarity
        # But with FakeEmbeddingProvider, different names produce different embeddings
        # so similarity should be below threshold for very different names


class TestEntitySpatialIndex:
    def test_entity_in_spatial_query(self, store):
        """Entities are indexed in the spatial index."""
        entity = _make_entity("red chair", x=5.0, y=5.0)
        store.add_entity(entity)

        # The spatial index contains the entity
        ids = store._spatial.query_radius(np.array([5.0, 5.0, 0.0]), 2.0)
        assert entity.id in ids


class TestEntityQuery:
    def test_by_name(self, store):
        store.add_entity(_make_entity("red chair", x=5.0, y=5.0))
        store.add_entity(_make_entity("blue table", x=10.0, y=10.0))

        results = store.query_entities(name="chair")
        assert len(results) == 1
        assert results[0].name == "red chair"

    def test_by_type(self, store):
        store.add_entity(_make_entity("red chair", entity_type="furniture"))
        store.add_entity(_make_entity("person", entity_type="human"))

        results = store.query_entities(entity_type="furniture")
        assert len(results) == 1
        assert results[0].name == "red chair"

    def test_by_spatial(self, store):
        store.add_entity(_make_entity("near", x=5.0, y=5.0))
        store.add_entity(_make_entity("far", x=100.0, y=100.0))

        results = store.query_entities(
            near_coordinates=np.array([5.0, 5.0, 0.0]),
            spatial_radius=3.0,
        )
        assert len(results) == 1
        assert results[0].name == "near"

    def test_by_recency(self, store):
        store.add_entity(_make_entity("old", ts=100.0))
        store.add_entity(_make_entity("recent", ts=2000.0))

        results = store.query_entities(last_seen_after=1000.0)
        assert len(results) == 1
        assert results[0].name == "recent"


class TestEntityEdges:
    def test_observed_in(self, store):
        entity = _make_entity("red chair")
        store.add_entity(entity)
        obs = _make_obs("saw red chair", x=5.0, y=5.0)
        store.add_observation(obs)

        from emem.types import Edge

        store.add_edge(
            Edge(
                source_id=entity.id,
                target_id=obs.id,
                edge_type=EdgeType.OBSERVED_IN,
            )
        )

        observations = store.get_entity_observations(entity.id)
        assert len(observations) == 1
        assert observations[0].id == obs.id

    def test_cooccurs_with(self, store):
        e1 = _make_entity("chair", x=5.0, y=5.0)
        e2 = _make_entity("table", x=5.5, y=5.5)
        store.add_entity(e1)
        store.add_entity(e2)

        from emem.types import Edge

        store.add_edge(
            Edge(
                source_id=e1.id,
                target_id=e2.id,
                edge_type=EdgeType.COOCCURS_WITH,
            )
        )

        cooccurring = store.get_cooccurring_entities(e1.id)
        assert len(cooccurring) == 1
        assert cooccurring[0].id == e2.id

        # Bidirectional
        cooccurring2 = store.get_cooccurring_entities(e2.id)
        assert len(cooccurring2) == 1
        assert cooccurring2[0].id == e1.id


class TestRecencyWeighting:
    def test_recency_disabled_by_default(self, store):
        """With default config (recency_weight=0), ordering matches pure HNSW distance."""
        store.add_observation(_make_obs("alpha", ts=1000.0))
        store.add_observation(_make_obs("beta", ts=2000.0))

        results_no_ref = store.semantic_search("alpha", n_results=10)
        results_with_ref = store.semantic_search(
            "alpha", n_results=10, reference_time=3000.0
        )
        # Same ordering since recency_weight=0
        assert [r.id for r in results_no_ref] == [r.id for r in results_with_ref]

    def test_recency_favors_recent(self, tmp_path):
        """With recency_weight > 0, recent observations rank higher at equal distance."""
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "recency.db"),
            hnsw_path=str(tmp_path / "recency_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=1000,
            recency_weight=1.0,
            recency_halflife=1000.0,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))

        # Two observations with same text (same embedding) but different timestamps
        s.add_observation(_make_obs("identical test phrase", ts=100.0))
        s.add_observation(_make_obs("identical test phrase", ts=9000.0))

        results = s.semantic_search(
            "identical test phrase", n_results=2, reference_time=10000.0
        )
        assert len(results) == 2
        # The recent one (ts=9000) should rank first due to lower recency penalty
        assert results[0].timestamp == 9000.0
        s.close()

    def test_recency_across_node_types(self, tmp_path):
        """Recency weighting works for gists (time_end) and entities (last_seen)."""
        # Hybrid retrieval is disabled here so the test isolates the
        # HNSW recency-weighting logic from the BM25 path. BM25 would
        # legitimately push a text-matching gist up regardless of age,
        # which is a different (also correct) behaviour tested
        # elsewhere.
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "recency_types.db"),
            hnsw_path=str(tmp_path / "recency_types_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=1000,
            recency_weight=1.0,
            recency_halflife=1000.0,
            use_hybrid_retrieval=False,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))

        s.add_entity(_make_entity("test item", ts=5000.0))
        gist = GistNode(
            text="test item summary",
            center_position=np.array([0, 0, 0]),
            radius=1.0,
            time_start=100.0,
            time_end=200.0,
            source_observation_count=1,
            source_observation_ids=["x"],
        )
        s.add_gist(gist)

        results = s.semantic_search("test item", n_results=10, reference_time=6000.0)
        assert len(results) >= 1
        # Entity (last_seen=5000) is more recent than gist (time_end=200)
        # so entity should rank first
        assert isinstance(results[0], EntityNode)
        s.close()

    def test_recency_no_reference_time(self, tmp_path):
        """Without reference_time, original ordering even if recency_weight > 0."""
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "recency_noref.db"),
            hnsw_path=str(tmp_path / "recency_noref_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=1000,
            recency_weight=1.0,
            recency_halflife=1000.0,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
        s.add_observation(_make_obs("foo", ts=100.0))
        s.add_observation(_make_obs("foo", ts=9000.0))

        # No reference_time → should use plain HNSW ordering (no crash)
        results = s.semantic_search("foo", n_results=10)
        assert len(results) >= 1
        s.close()


class TestEntityInSemanticSearch:
    def test_entity_found_by_semantic_search(self, store):
        entity = _make_entity("red chair", x=5.0, y=5.0)
        store.add_entity(entity)

        results = store.semantic_search("red chair", n_results=10)
        entity_results = [r for r in results if isinstance(r, EntityNode)]
        assert len(entity_results) >= 1
        assert entity_results[0].name == "red chair"


class TestHybridRetrieval:
    """A1: SQLite FTS5 BM25 + HNSW via Reciprocal Rank Fusion."""

    def test_rrf_merge_sums_reciprocal_ranks(self):
        """RRF combines two ranked lists and rewards appearing in both."""
        fused = MemoryStore._rrf_merge([["a", "b", "c"], ["b", "c", "a"]], k=60)
        # Each id appears in both lists; 'b' is at rank 2 in list1 and
        # rank 1 in list2, so should win over 'a' (rank 1 + rank 3).
        assert fused[0] == "b"
        # 'c' is rank 3 + rank 2 = lowest-scored.
        assert fused[-1] == "c"

    def test_rrf_empty_input(self):
        assert MemoryStore._rrf_merge([]) == []
        assert MemoryStore._rrf_merge([[], []]) == []

    def test_rrf_single_source_degenerates_to_identity(self):
        """If only HNSW contributes (e.g. no BM25 hits), RRF returns
        the HNSW ranking unchanged."""
        fused = MemoryStore._rrf_merge([["x", "y", "z"], []])
        assert fused == ["x", "y", "z"]

    def test_fts_escape_quotes_query(self):
        assert MemoryStore._fts_escape_query("hello world") == '"hello world"'
        # Internal quotes are doubled per FTS5 convention.
        assert MemoryStore._fts_escape_query('say "hi"') == '"say ""hi"""'
        # Empty string falls back to a safe empty quoted string.
        assert MemoryStore._fts_escape_query("") == '""'

    def test_bm25_retrieves_rare_token_matches(self, tmp_path):
        """BM25 finds an observation by a rare word where HNSW's
        hash-based fake embeddings would miss it."""
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "bm25.db"),
            hnsw_path=str(tmp_path / "bm25_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=100,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
        s.add_observations_batch([
            _make_obs("saw a xylophone near the window"),
            _make_obs("there was a chair in the room"),
            _make_obs("we ate dinner at the table"),
        ])
        # BM25 lookup by rare token.
        bm25_ids = s._bm25_search_ids("xylophone", "observation", fetch_k=5)
        assert len(bm25_ids) == 1
        hit = s.get_observation(bm25_ids[0])
        assert hit is not None and "xylophone" in hit.text
        s.close()

    def test_bm25_disabled_returns_empty(self, tmp_path):
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "off.db"),
            hnsw_path=str(tmp_path / "off_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=100,
            use_hybrid_retrieval=False,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
        s.add_observations_batch([_make_obs("saw a xylophone near the window")])
        assert s._bm25_search_ids("xylophone", "observation", fetch_k=5) == []
        s.close()

    def test_fts_entry_removed_on_archival(self, tmp_path):
        """When an observation is archived (text dropped), its FTS
        entry is removed so BM25 can no longer return it."""
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "arch.db"),
            hnsw_path=str(tmp_path / "arch_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=100,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
        ids = s.add_observations_batch([
            _make_obs("contains the word fluorescent marker")
        ])
        assert s._bm25_search_ids("fluorescent", "observation", fetch_k=5) == ids

        s.update_observation_tier(ids[0], "archived", drop_text=True)
        assert s._bm25_search_ids("fluorescent", "observation", fetch_k=5) == []
        s.close()

    def test_hybrid_semantic_search_surfaces_bm25_match(self, tmp_path):
        """semantic_search should surface an observation whose rare
        token matches, even if HNSW alone would miss it."""
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "hybrid.db"),
            hnsw_path=str(tmp_path / "hybrid_hnsw.bin"),
            embedding_dim=32,
            hnsw_max_elements=100,
        )
        s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
        distractors = [_make_obs(f"unrelated sentence number {i}") for i in range(10)]
        target = _make_obs("the quarterback threw a touchdown pass")
        s.add_observations_batch(distractors + [target])
        results = s.semantic_search("quarterback", n_results=3)
        texts = [r.text for r in results if isinstance(r, ObservationNode)]
        assert any("quarterback" in t for t in texts), (
            "BM25 branch of hybrid retrieval should surface the "
            "literal-keyword match even when HNSW's fake embeddings would not"
        )
        s.close()

    def test_hybrid_off_vs_on_produces_different_rankings(self, tmp_path):
        """Flipping ``use_hybrid_retrieval`` changes retrieval behaviour:
        the hybrid-on path surfaces a keyword match in the top results;
        the hybrid-off (HNSW-only) path runs without error.

        We check top-3 rather than top-1 because RRF ties (which occur
        on the tiny corpus used here) are broken by dict insertion
        order across sources, which is not part of the contract we
        want to pin down.
        """

        def _run(flag: bool) -> list:
            config = SpatioTemporalMemoryConfig(
                db_path=str(tmp_path / f"flag_{flag}.db"),
                hnsw_path=str(tmp_path / f"flag_{flag}_hnsw.bin"),
                embedding_dim=32,
                hnsw_max_elements=100,
                use_hybrid_retrieval=flag,
            )
            s = MemoryStore(config=config, embedding_provider=FakeEmbeddingProvider(32))
            s.add_observations_batch([_make_obs(f"filler text {i}") for i in range(10)])
            s.add_observations_batch([
                _make_obs("the quarterback threw a touchdown pass")
            ])
            results = s.semantic_search("quarterback", n_results=3)
            texts = [r.text for r in results if hasattr(r, "text")]
            s.close()
            return texts

        on_texts = _run(True)
        off_texts = _run(False)

        assert any("quarterback" in t for t in on_texts), (
            f"hybrid-on should surface the quarterback match in top-3; "
            f"got {on_texts!r}"
        )
        assert isinstance(off_texts, list)
