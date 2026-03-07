"""Tests for MemoryTools."""

import time

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.tools import MemoryTools, _parse_relative_time
from emem.types import GistNode, ObservationNode


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
def store(tmp_path):
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "tools_test.db"),
        hnsw_path=str(tmp_path / "tools_hnsw.bin"),
        embedding_dim=32,
        hnsw_max_elements=1000,
    )
    s = MemoryStore(config=config, embedding_provider=FakeEmbedder(32))
    yield s
    s.close()


@pytest.fixture
def tools(store):
    current_pos = np.array([5.0, 5.0, 0.0])
    return MemoryTools(
        store=store,
        get_current_time=lambda: 2000.0,
        get_current_position=lambda: current_pos,
    )


def _obs(text, x=0.0, y=0.0, ts=1000.0, layer="default"):
    return ObservationNode(
        text=text,
        coordinates=np.array([x, y, 0.0]),
        timestamp=ts,
        layer_name=layer,
    )


class TestRelativeTimeParsing:
    def test_minutes(self):
        result = _parse_relative_time("-10m", reference_time=1000.0)
        assert result == 400.0

    def test_hours(self):
        result = _parse_relative_time("-1h", reference_time=7200.0)
        assert result == 3600.0

    def test_days(self):
        result = _parse_relative_time("-2d", reference_time=200000.0)
        assert result == 200000.0 - 172800.0

    def test_seconds(self):
        result = _parse_relative_time("-30s", reference_time=1000.0)
        assert result == 970.0

    def test_absolute_passthrough(self):
        result = _parse_relative_time("1500.0")
        assert result == 1500.0


class TestSpatialQuery:
    def test_basic(self, store, tools):
        store.add_observation(_obs("near robot", x=5.0, y=5.0, ts=1500.0))
        store.add_observation(_obs("far away", x=100.0, y=100.0, ts=1500.0))

        result = tools.spatial_query(x=5.0, y=5.0, radius=3.0)
        assert "near robot" in result
        assert "far away" not in result


class TestTemporalQuery:
    def test_last_n_minutes(self, store, tools):
        store.add_observation(_obs("old", ts=100.0))
        store.add_observation(_obs("recent", ts=1900.0))

        result = tools.temporal_query(last_n_minutes=5)
        assert "recent" in result
        # old is at t=100, current=2000, 5min=300s ago => cutoff=1700
        assert "old" not in result

    def test_with_time_after(self, store, tools):
        store.add_observation(_obs("a", ts=500.0))
        store.add_observation(_obs("b", ts=1500.0))

        result = tools.temporal_query(time_after="1000.0")
        assert "b" in result


class TestEpisodeSummary:
    def test_basic(self, store, tools):
        ep_id = store.start_episode("patrol", 1000.0)
        store.end_episode(ep_id, 2000.0, gist="Patrolled zone A, found nothing")

        result = tools.episode_summary(episode_id=ep_id)
        assert "patrol" in result
        assert "Patrolled" in result

    def test_by_name(self, store, tools):
        store.start_episode("patrol_a", 1000.0)
        result = tools.episode_summary(task_name="patrol")
        assert "patrol_a" in result

    def test_not_found(self, store, tools):
        result = tools.episode_summary(episode_id="nonexistent")
        assert "not found" in result


class TestCurrentContext:
    def test_basic(self, store, tools):
        store.add_observation(_obs("nearby thing", x=5.0, y=5.5, ts=1950.0))
        result = tools.get_current_context()
        assert "Position: (5.0, 5.0)" in result
        assert "nearby thing" in result

    def test_empty(self, tools):
        result = tools.get_current_context()
        # Should still show position even with no data
        assert "Position" in result


class TestSearchGists:
    def test_basic(self, store, tools):
        gist = GistNode(
            text="Furniture cluster: chairs and tables",
            center_position=np.array([3.0, 3.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=1500.0,
            source_observation_count=5,
            source_observation_ids=["a", "b", "c", "d", "e"],
        )
        store.add_gist(gist)
        result = tools.search_gists(query="furniture")
        assert "Furniture" in result or "chairs" in result


class TestSemanticSearchUnified:
    def test_formats_gist_results(self, store, tools):
        gist = GistNode(
            text="Furniture cluster: chairs and tables",
            center_position=np.array([3.0, 3.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=1500.0,
            source_observation_count=5,
            source_observation_ids=["a", "b", "c", "d", "e"],
        )
        store.add_gist(gist)

        result = tools.semantic_search(query="furniture")
        # Should contain gist formatting markers
        assert "gist/" in result or "Furniture" in result or "chairs" in result

    def test_mixed_obs_and_gist(self, store, tools):
        store.add_observation(_obs("red chair nearby", x=5.0, y=5.0, ts=1500.0))
        gist = GistNode(
            text="Area has furniture including chairs",
            center_position=np.array([5.0, 5.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=1500.0,
            source_observation_count=3,
            source_observation_ids=["a", "b", "c"],
        )
        store.add_gist(gist)

        result = tools.semantic_search(query="chair", n_results=10)
        assert "No results found" not in result


class TestCurrentContextWithGists:
    def test_includes_area_gists(self, store, tools):
        gist = GistNode(
            text="This area was patrolled and found clear",
            center_position=np.array([5.0, 5.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=1500.0,
            source_observation_count=3,
            source_observation_ids=["a", "b", "c"],
        )
        store.add_gist(gist)

        result = tools.get_current_context(radius=3.0)
        assert "Area summaries:" in result
        assert "patrolled" in result


class TestDispatch:
    def test_dispatch_known_tool(self, tools):
        result = tools.dispatch_tool_call("episode_summary", {"task_name": "nonexistent"})
        assert "No episodes" in result

    def test_dispatch_unknown_tool(self, tools):
        result = tools.dispatch_tool_call("nonexistent_tool", {})
        assert "Unknown tool" in result


class TestToolDefinitions:
    def test_returns_six_tools(self, tools):
        defs = tools.get_tool_definitions()
        assert len(defs) == 6
        names = {d["name"] for d in defs}
        assert names == {
            "semantic_search", "spatial_query", "temporal_query",
            "episode_summary", "get_current_context", "search_gists",
        }
