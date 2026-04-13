"""Tests for MemoryTools."""

import time

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.tools import MemoryTools, _parse_relative_time
from emem.types import EntityNode, GistNode, ObservationNode


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

    def test_gist_fallback_after_consolidation(self, store, tools):
        """When observations are archived, temporal_query falls back to gists."""
        # Add a gist covering a recent time range (within last 5 min of t=2000)
        gist = GistNode(
            text="Explored a maze with gray pathways",
            center_position=np.array([5.0, 5.0, 0.0]),
            radius=2.0,
            time_start=1800.0,
            time_end=1950.0,
            source_observation_count=4,
            source_observation_ids=["o1", "o2", "o3", "o4"],
        )
        store.add_gist(gist)

        # No raw observations exist — should fall back to gists
        result = tools.temporal_query(last_n_minutes=5)
        assert "consolidated" in result.lower()
        assert "maze" in result
        assert "gray pathways" in result

    def test_gist_fallback_not_triggered_when_observations_exist(self, store, tools):
        """Gist fallback should NOT trigger when observations are found."""
        store.add_observation(_obs("live observation", ts=1900.0))
        gist = GistNode(
            text="This gist should not appear",
            center_position=np.array([5.0, 5.0, 0.0]),
            radius=1.0,
            time_start=1800.0,
            time_end=1950.0,
            source_observation_count=2,
            source_observation_ids=["x1", "x2"],
        )
        store.add_gist(gist)

        result = tools.temporal_query(last_n_minutes=5)
        assert "live observation" in result
        assert "consolidated" not in result.lower()


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
        result = tools.dispatch_tool_call(
            "episode_summary", {"task_name": "nonexistent"}
        )
        assert "No episodes" in result

    def test_dispatch_unknown_tool(self, tools):
        result = tools.dispatch_tool_call("nonexistent_tool", {})
        assert "Unknown tool" in result


class TestToolDefinitions:
    def test_returns_ten_tools_in_openai_format(self, tools):
        defs = tools.get_tool_definitions()
        assert len(defs) == 10
        for d in defs:
            assert d["type"] == "function"
            assert "name" in d["function"]
        names = {d["function"]["name"] for d in defs}
        assert names == {
            "semantic_search",
            "spatial_query",
            "temporal_query",
            "episode_summary",
            "get_current_context",
            "search_gists",
            "entity_query",
            "locate",
            "recall",
            "body_status",
        }


class TestEntityQuery:
    def test_basic(self, store, tools):
        entity = EntityNode(
            name="red chair",
            coordinates=np.array([5.0, 5.0, 0.0]),
            last_seen=1500.0,
            first_seen=1000.0,
            observation_count=3,
            entity_type="furniture",
        )
        store.add_entity(entity)

        result = tools.entity_query(name="chair")
        assert "red chair" in result
        assert "entity/furniture" in result
        assert "seen 3x" in result

    def test_empty(self, tools):
        result = tools.entity_query(name="nonexistent")
        assert "No entities found" in result

    def test_dispatch(self, store, tools):
        entity = EntityNode(
            name="blue table",
            coordinates=np.array([5.0, 5.0, 0.0]),
            last_seen=1500.0,
            first_seen=1000.0,
        )
        store.add_entity(entity)

        result = tools.dispatch_tool_call("entity_query", {"name": "table"})
        assert "blue table" in result

    def test_with_spatial_filter(self, store, tools):
        entity = EntityNode(
            name="lamp",
            coordinates=np.array([5.0, 5.0, 0.0]),
            last_seen=1500.0,
            first_seen=1000.0,
        )
        store.add_entity(entity)

        result = tools.entity_query(near_x=5.0, near_y=5.0, spatial_radius=2.0)
        assert "lamp" in result

        result2 = tools.entity_query(near_x=100.0, near_y=100.0, spatial_radius=2.0)
        assert "No entities found" in result2


class TestCurrentContextWithEntities:
    def test_includes_nearby_entities(self, store, tools):
        entity = EntityNode(
            name="red chair",
            coordinates=np.array([5.0, 5.5, 0.0]),
            last_seen=1500.0,
            first_seen=1000.0,
            observation_count=3,
            entity_type="furniture",
        )
        store.add_entity(entity)

        result = tools.get_current_context(radius=3.0)
        assert "Nearby entities:" in result
        assert "red chair" in result


class TestContextGrouping:
    def test_groups_nearby_by_layer(self, store, tools):
        store.add_observation(
            _obs("white cabinets", x=5.0, y=5.0, ts=1500.0, layer="vlm")
        )
        store.add_observation(
            _obs("chair, table", x=5.0, y=5.0, ts=1500.0, layer="detections")
        )
        store.add_observation(_obs("kitchen", x=5.0, y=5.0, ts=1500.0, layer="place"))

        result = tools.get_current_context(radius=3.0)
        assert "[detections]" in result
        assert "[vlm]" in result
        assert "[place]" in result
        # Verify layer headers appear before their content
        det_idx = result.index("[detections]")
        chair_idx = result.index("chair, table")
        assert det_idx < chair_idx

    def test_single_layer_groups(self, store, tools):
        store.add_observation(_obs("obs1", x=5.0, y=5.0, ts=1500.0, layer="vlm"))
        store.add_observation(_obs("obs2", x=5.0, y=5.1, ts=1501.0, layer="vlm"))

        result = tools.get_current_context(radius=3.0)
        assert "[vlm]" in result
        assert "obs1" in result
        assert "obs2" in result


class TestLocate:
    def test_basic_locate(self, store, tools):
        store.add_observation(_obs("kitchen area", x=10.0, y=10.0, ts=1500.0))
        store.add_observation(_obs("kitchen counter", x=11.0, y=10.0, ts=1501.0))
        store.add_observation(_obs("kitchen sink", x=10.0, y=11.0, ts=1502.0))

        result = tools.locate(concept="kitchen")
        assert "Location:" in result
        assert "Based on:" in result

    def test_no_results(self, tools):
        result = tools.locate(concept="nonexistent_place_xyz")
        assert "Could not locate" in result

    def test_includes_gists(self, store, tools):
        gist = GistNode(
            text="Kitchen area with appliances",
            center_position=np.array([10.0, 10.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=1500.0,
            source_observation_count=3,
            source_observation_ids=["a", "b", "c"],
        )
        store.add_gist(gist)

        result = tools.locate(concept="kitchen appliances")
        assert "Location:" in result

    def test_dispatch(self, store, tools):
        store.add_observation(_obs("office desk", x=20.0, y=20.0, ts=1500.0))

        result = tools.dispatch_tool_call("locate", {"concept": "office"})
        assert "Location:" in result or "Could not locate" in result


class TestRecall:
    def test_basic_recall(self, store, tools):
        store.add_observation(
            _obs("kitchen with white cabinets", x=10.0, y=10.0, ts=1500.0, layer="vlm")
        )
        store.add_observation(
            _obs("chair, table, fridge", x=10.0, y=10.0, ts=1500.0, layer="detections")
        )
        entity = EntityNode(
            name="refrigerator",
            coordinates=np.array([10.0, 10.0, 0.0]),
            last_seen=1500.0,
            first_seen=1500.0,
            observation_count=1,
            entity_type="appliance",
        )
        store.add_entity(entity)

        result = tools.recall(query="kitchen")
        assert "About 'kitchen'" in result
        assert "Observations:" in result
        assert "Entities:" in result

    def test_no_results(self, tools):
        result = tools.recall(query="nonexistent_xyz")
        assert "No memories found" in result

    def test_includes_gists(self, store, tools):
        store.add_observation(_obs("office area", x=20.0, y=20.0, ts=1500.0))
        gist = GistNode(
            text="Office area with desks and computers",
            center_position=np.array([20.0, 20.0, 0.0]),
            radius=2.0,
            time_start=1000.0,
            time_end=1500.0,
            source_observation_count=5,
            source_observation_ids=["a", "b", "c", "d", "e"],
        )
        store.add_gist(gist)

        result = tools.recall(query="office")
        assert "About 'office'" in result
        assert "Summaries:" in result

    def test_dispatch(self, store, tools):
        store.add_observation(_obs("corridor lights", x=30.0, y=30.0, ts=1500.0))

        result = tools.dispatch_tool_call("recall", {"query": "corridor"})
        assert "About 'corridor'" in result or "No memories found" in result
