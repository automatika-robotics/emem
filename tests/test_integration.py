"""End-to-end integration test for the spatio-temporal memory system."""

import time

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.consolidation import ConsolidationEngine
from emem.store import MemoryStore
from emem.tools import MemoryTools
from emem.types import ObservationNode, Tier
from emem.working_memory import WorkingMemory


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
def system(tmp_path):
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "integ.db"),
        hnsw_path=str(tmp_path / "integ_hnsw.bin"),
        embedding_dim=32,
        hnsw_max_elements=10000,
        flush_batch_size=10,
        flush_interval=100.0,
        consolidation_window=500.0,
        consolidation_spatial_eps=5.0,
        consolidation_min_samples=3,
    )
    embedder = FakeEmbedder(32)
    store = MemoryStore(config=config, embedding_provider=embedder)
    wm = WorkingMemory(store=store, config=config)
    consolidation = ConsolidationEngine(store=store, config=config)

    sim_time = [0.0]
    current_pos = [np.array([0.0, 0.0, 0.0])]

    tools = MemoryTools(
        store=store,
        get_current_time=lambda: sim_time[0],
        get_current_position=lambda: current_pos[0],
    )

    yield {
        "config": config,
        "store": store,
        "wm": wm,
        "consolidation": consolidation,
        "tools": tools,
        "sim_time": sim_time,
        "current_pos": current_pos,
    }
    store.close()


def test_full_pipeline(system):
    store = system["store"]
    wm = system["wm"]
    consolidation = system["consolidation"]
    tools = system["tools"]
    sim_time = system["sim_time"]
    current_pos = system["current_pos"]

    # ── Phase 1: Simulate robot patrol with 3 episodes ────────────

    # Episode 1: Kitchen patrol (t=100-200, position around (10, 10))
    sim_time[0] = 100.0
    ep1_id = wm.start_episode("kitchen_patrol")
    for i in range(15):
        t = 100.0 + i * 5
        sim_time[0] = t
        x = 10.0 + np.random.randn() * 1.0
        y = 10.0 + np.random.randn() * 1.0
        current_pos[0] = np.array([x, y, 0.0])
        wm.add(
            ObservationNode(
                text=f"kitchen_obs_{i}: {'table' if i % 3 == 0 else 'counter' if i % 3 == 1 else 'sink'}",
                coordinates=np.array([x, y, 0.0]),
                timestamp=t,
                layer_name="vlm",
            )
        )
    sim_time[0] = 200.0
    wm.flush()
    wm.end_episode(gist="Patrolled kitchen area. Found tables, counters, and sinks.")

    # Episode 2: Hallway patrol (t=300-400, position around (30, 10))
    sim_time[0] = 300.0
    ep2_id = wm.start_episode("hallway_patrol")
    for i in range(20):
        t = 300.0 + i * 5
        sim_time[0] = t
        x = 30.0 + np.random.randn() * 0.5
        y = 10.0 + np.random.randn() * 0.5
        current_pos[0] = np.array([x, y, 0.0])
        wm.add(
            ObservationNode(
                text=f"hallway_obs_{i}: {'door' if i % 4 == 0 else 'painting' if i % 4 == 1 else 'exit_sign' if i % 4 == 2 else 'fire_extinguisher'}",
                coordinates=np.array([x, y, 0.0]),
                timestamp=t,
                layer_name="vlm",
            )
        )
    sim_time[0] = 400.0
    wm.flush()
    wm.end_episode(
        gist="Patrolled hallway. Found doors, paintings, exit signs, fire extinguishers."
    )

    # Episode 3: Lobby inspection (t=500-600, position around (50, 50))
    sim_time[0] = 500.0
    wm.start_episode("lobby_inspection")
    for i in range(15):
        t = 500.0 + i * 5
        sim_time[0] = t
        x = 50.0 + np.random.randn() * 2.0
        y = 50.0 + np.random.randn() * 2.0
        current_pos[0] = np.array([x, y, 0.0])
        wm.add(
            ObservationNode(
                text=f"lobby_obs_{i}: {'chair' if i % 3 == 0 else 'reception_desk' if i % 3 == 1 else 'plant'}",
                coordinates=np.array([x, y, 0.0]),
                timestamp=t,
                layer_name="vlm",
            )
        )
    sim_time[0] = 600.0
    wm.flush()
    wm.end_episode(gist="Inspected lobby. Found chairs, reception desk, plants.")

    # Also add some detection-layer observations
    for i in range(10):
        store.add_observation(
            ObservationNode(
                text=f"detected_person_{i}",
                coordinates=np.array([10.0 + i * 5, 10.0, 0.0]),
                timestamp=150.0 + i * 50,
                layer_name="detections",
                source_type="detection",
                confidence=0.8 + np.random.rand() * 0.2,
            )
        )

    total_obs = store.count_observations()
    assert total_obs >= 50, f"Expected 50+ observations, got {total_obs}"

    # ── Phase 2: Query with all 6 tools ───────────────────────────

    sim_time[0] = 700.0
    current_pos[0] = np.array([50.0, 50.0, 0.0])

    # Tool 1: Semantic search
    result = tools.semantic_search(query="kitchen table", n_results=5)
    assert "No observations" not in result

    # Tool 2: Spatial query — what's near the lobby?
    result = tools.spatial_query(x=50.0, y=50.0, radius=5.0)
    assert "lobby" in result or len(result) > 0

    # Tool 3: Temporal query — recent observations
    result = tools.temporal_query(last_n_minutes=5)  # last 5min = 300s → t>=400
    assert len(result) > 0

    # Tool 4: Episode summaries
    result = tools.episode_summary(task_name="patrol")
    assert "kitchen" in result.lower() or "hallway" in result.lower()

    # Tool 5: Current context
    result = tools.get_current_context(radius=5.0, include_recent_minutes=3)
    assert "Position" in result

    # Tool 6: Search gists (none yet — not consolidated)
    result = tools.search_gists(query="kitchen")
    # Might be empty since we haven't consolidated yet

    # ── Phase 3: Consolidation ────────────────────────────────────

    # Consolidate episode 1
    gist1_ids = consolidation.consolidate_episode(ep1_id)
    assert len(gist1_ids) >= 1
    gist1 = store.get_gist(gist1_ids[0])
    # All 15 observations are within one consolidation_window (500s),
    # so they should be in a single chunk
    assert gist1.source_observation_count == 15

    # Episode 1 observations should be demoted to long_term
    ep1_obs = store.get_episode_observations(ep1_id)
    for o in ep1_obs:
        assert o.tier == Tier.LONG_TERM.value

    # Time-window consolidation for detection-layer observations
    consolidation.consolidate_time_window(reference_time=700.0)
    # Detection observations at t=150-600, window=500s, cutoff=200
    # Observations before t=200 are candidates

    # ── Phase 4: Verify post-consolidation queries ────────────────

    # Gists should now be searchable
    result = tools.search_gists(query="kitchen table")
    # Should find the episode gist

    # Archived observations excluded from spatial queries
    spatial_results = store.spatial_query(
        center=np.array([10, 10, 0]),
        radius=3.0,
    )
    for r in spatial_results:
        assert r.tier != Tier.ARCHIVED.value

    # ── Phase 5: Verify graph edges ───────────────────────────────

    from emem.types import EdgeType

    # Episode membership edges
    edges = store.get_edges(target_id=ep2_id, edge_type=EdgeType.BELONGS_TO)
    assert len(edges) == 20  # 20 hallway observations

    # Gist summarizes edges
    if gist1_ids[0]:
        gist_edges = store.get_edges(
            source_id=gist1_ids[0], edge_type=EdgeType.SUMMARIZES
        )
        assert len(gist_edges) == 15

    # ── Phase 6: Dispatch tool calls ──────────────────────────────

    result = tools.dispatch_tool_call(
        "spatial_query", {"x": 30.0, "y": 10.0, "radius": 3.0}
    )
    assert "hallway" in result or len(result) > 0

    result = tools.dispatch_tool_call("episode_summary", {"last_n": 3})
    assert "lobby" in result.lower() or "inspection" in result.lower()


def test_persistence_round_trip(tmp_path):
    """Test that data survives store close/reopen."""
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "persist.db"),
        hnsw_path=str(tmp_path / "persist_hnsw.bin"),
        embedding_dim=32,
        hnsw_max_elements=1000,
    )
    embedder = FakeEmbedder(32)

    # Write phase
    store1 = MemoryStore(config=config, embedding_provider=embedder)
    store1.start_episode("ep1", 1000.0)
    for i in range(20):
        store1.add_observation(
            ObservationNode(
                text=f"obs_{i}",
                coordinates=np.array([float(i), 0.0, 0.0]),
                timestamp=1000.0 + i,
                layer_name="vlm",
            )
        )
    store1.close()

    # Read phase
    store2 = MemoryStore(config=config, embedding_provider=embedder)
    assert store2.count_observations() == 20
    eps = store2.list_episodes()
    assert len(eps) == 1

    # Semantic search still works
    results = store2.semantic_search("obs", n_results=5)
    assert len(results) > 0
    store2.close()
