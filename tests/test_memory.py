"""Tests for the SpatioTemporalMemory facade."""

import numpy as np
import pytest

from emem import SpatioTemporalMemory, SpatioTemporalMemoryConfig


class FakeEmbedder:
    def __init__(self, dim=32):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def embed(self, texts):
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.RandomState(hash(text) % (2**31))
            result[i] = rng.randn(self._dim).astype(np.float32)
            result[i] /= np.linalg.norm(result[i]) + 1e-8
        return result


@pytest.fixture
def mem(tmp_path):
    m = SpatioTemporalMemory(db_path=str(tmp_path / "test.db"))
    yield m
    m.close()


class TestAdd:
    def test_basic(self, mem):
        obs_id = mem.add("red chair", x=1.0, y=2.0)
        assert obs_id
        # Observation is buffered; query auto-flushes
        result = mem.spatial_query(x=1.0, y=2.0, radius=1.0)
        assert "red chair" in result

    def test_with_all_args(self, mem):
        obs_id = mem.add(
            "detected person",
            x=5.0,
            y=10.0,
            z=1.5,
            layer_name="detections",
            source_type="detection",
            confidence=0.95,
            metadata={"model": "yolo"},
        )
        mem.save()  # flush to store
        obs = mem.store.get_observation(obs_id)
        assert obs.layer_name == "detections"
        assert obs.confidence == 0.95


class TestEpisodeLifecycle:
    def test_start_and_end(self, mem):
        ep_id = mem.start_episode("patrol")
        assert mem.active_episode_id == ep_id

        mem.add("observation 1", x=1.0, y=1.0)
        mem.add("observation 2", x=2.0, y=2.0)

        returned_id = mem.end_episode()
        assert returned_id == ep_id
        assert mem.active_episode_id is None

        ep = mem.store.get_episode(ep_id)
        assert ep.status == "completed"

    def test_auto_assigns_episode(self, mem):
        ep_id = mem.start_episode("test")
        obs_id = mem.add("something", x=0.0, y=0.0)
        mem.end_episode()

        obs = mem.store.get_observation(obs_id)
        assert obs.episode_id == ep_id

    def test_auto_consolidates_on_end(self, mem):
        ep_id = mem.start_episode("patrol")
        mem.add("chair", x=1.0, y=1.0)
        mem.add("table", x=1.5, y=1.5)
        mem.end_episode()

        # Observations should be demoted to long_term (consolidated)
        obs_list = mem.store.get_episode_observations(ep_id)
        for obs in obs_list:
            assert obs.tier == "long_term"

    def test_no_consolidation_when_disabled(self, mem):
        ep_id = mem.start_episode("patrol")
        mem.add("chair", x=1.0, y=1.0)
        mem.end_episode(consolidate=False)

        obs_list = mem.store.get_episode_observations(ep_id)
        for obs in obs_list:
            assert obs.tier == "short_term"

    def test_end_without_active(self, mem):
        assert mem.end_episode() is None


class TestQueryAutoFlush:
    def test_spatial_query_sees_buffered(self, mem):
        # Won't auto-flush due to high thresholds — but query should force it
        mem.add("nearby", x=1.0, y=1.0)

        result = mem.spatial_query(x=1.0, y=1.0, radius=5.0)
        assert "nearby" in result

    def test_temporal_query_sees_buffered(self, mem):
        mem.add("recent thing", x=0.0, y=0.0)
        result = mem.temporal_query(last_n_minutes=60)
        assert "recent thing" in result


class TestToolDispatch:
    def test_dispatch(self, mem):
        mem.start_episode("test")
        mem.add("hello", x=0.0, y=0.0)
        mem.end_episode()

        result = mem.dispatch_tool_call("episode_summary", {"last_n": 1})
        assert "test" in result

    def test_tool_definitions(self, mem):
        defs = mem.get_tool_definitions()
        assert len(defs) == 10


class TestContextManager:
    def test_context_manager(self, tmp_path):
        with SpatioTemporalMemory(db_path=str(tmp_path / "ctx.db")) as mem:
            mem.add("test", x=0.0, y=0.0)


class TestPosition:
    def test_tracks_position(self, mem):
        assert mem.current_position is None
        mem.add("a", x=5.0, y=10.0)
        np.testing.assert_array_almost_equal(mem.current_position, [5.0, 10.0, 0.0])


class TestUnifiedSearchE2E:
    def test_consolidated_gist_found_by_semantic_search(self, tmp_path):
        """E2E: add observations → consolidate → semantic_search finds gist."""
        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "unified.db"),
            hnsw_path=str(tmp_path / "unified_hnsw.bin"),
            embedding_dim=32,
        )
        mem = SpatioTemporalMemory(
            config=config,
            embedding_provider=FakeEmbedder(32),
        )
        mem.start_episode("furniture_scan")
        mem.add("red chair at entrance", x=1.0, y=1.0)
        mem.add("blue table near wall", x=1.5, y=1.5)
        mem.add("wooden shelf by door", x=2.0, y=2.0)
        mem.end_episode()  # auto-consolidates → archives observations, creates gist

        # After consolidation, observations are archived. semantic_search should
        # still find relevant content via the gist.
        result = mem.dispatch_tool_call(
            "semantic_search", {"query": "chair", "n_results": 10}
        )
        # Should find something (gist) even though observations are archived
        assert "No results found" not in result
        mem.close()


class TestRecent:
    def test_get_recent(self, mem):
        for i in range(5):
            mem.add(f"obs_{i}", x=float(i), y=0.0)
        recent = mem.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].text == "obs_4"


class TestEntityFacade:
    def test_add_entity(self, mem):
        eid = mem.add_entity("red chair", x=5.0, y=5.0, entity_type="furniture")
        assert eid
        entity = mem.store.get_entity(eid)
        assert entity.name == "red chair"
        assert entity.entity_type == "furniture"

    def test_entity_query(self, mem):
        mem.add_entity("red chair", x=5.0, y=5.0, entity_type="furniture")
        mem.add_entity("blue table", x=10.0, y=10.0, entity_type="furniture")

        result = mem.entity_query(name="chair")
        assert "red chair" in result
        assert "blue table" not in result

    def test_entity_query_dispatch(self, mem):
        mem.add_entity("lamp", x=1.0, y=1.0)
        result = mem.dispatch_tool_call("entity_query", {"name": "lamp"})
        assert "lamp" in result
