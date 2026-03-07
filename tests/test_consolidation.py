"""Tests for ConsolidationEngine."""

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.consolidation import ConcatenationSummarizer, ConsolidationEngine
from emem.store import MemoryStore
from emem.types import ObservationNode, Tier


@pytest.fixture
def store(tmp_path):
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "cons_test.db"),
        hnsw_path=str(tmp_path / "cons_hnsw.bin"),
        embedding_dim=32,
        hnsw_max_elements=1000,
    )
    s = MemoryStore(config=config)
    yield s
    s.close()


@pytest.fixture
def engine(store):
    config = SpatioTemporalMemoryConfig(
        consolidation_window=100.0,  # 100s window for testing
        consolidation_spatial_eps=3.0,
        consolidation_min_samples=2,
    )
    return ConsolidationEngine(store=store, config=config)


def _obs(text, x=0.0, y=0.0, ts=1000.0, episode_id=None):
    return ObservationNode(
        text=text,
        coordinates=np.array([x, y, 0.0]),
        timestamp=ts,
        episode_id=episode_id,
    )


class TestEpisodeConsolidation:
    def test_consolidate_episode(self, store, engine):
        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(_obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id))
        store.add_observation(_obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        gist_id = engine.consolidate_episode(ep_id)
        assert gist_id is not None

        gist = store.get_gist(gist_id)
        assert "chair" in gist.text
        assert "table" in gist.text
        assert gist.source_observation_count == 2

        # Source observations should be archived
        obs = store.get_episode_observations(ep_id)
        for o in obs:
            assert o.tier == Tier.ARCHIVED.value
            assert o.text == ""

    def test_consolidate_empty_episode(self, store, engine):
        ep_id = store.start_episode("empty", 1000.0)
        result = engine.consolidate_episode(ep_id)
        assert result is None


class TestTimeWindowConsolidation:
    def test_clusters_and_consolidates(self, store, engine):
        # Cluster 1: observations near (1, 1)
        store.add_observation(_obs("chair1", x=1.0, y=1.0, ts=100.0))
        store.add_observation(_obs("chair2", x=1.5, y=1.0, ts=101.0))
        store.add_observation(_obs("chair3", x=1.0, y=1.5, ts=102.0))

        # Cluster 2: observations near (50, 50) — far away
        store.add_observation(_obs("table1", x=50.0, y=50.0, ts=103.0))
        store.add_observation(_obs("table2", x=50.5, y=50.0, ts=104.0))
        store.add_observation(_obs("table3", x=50.0, y=50.5, ts=105.0))

        # Recent observation — should NOT be consolidated (within window)
        store.add_observation(_obs("recent", x=1.0, y=1.0, ts=990.0))

        gist_ids = engine.consolidate_time_window(reference_time=1000.0)
        # Window=100s, so cutoff=900. Observations at ts 100-105 qualify.
        assert len(gist_ids) == 2

        # Verify gists exist
        for gid in gist_ids:
            g = store.get_gist(gid)
            assert g is not None
            assert g.source_observation_count >= 2

    def test_no_candidates(self, store, engine):
        store.add_observation(_obs("recent", ts=999.0))
        gist_ids = engine.consolidate_time_window(reference_time=1000.0)
        assert gist_ids == []

    def test_noise_points_not_consolidated(self, store, engine):
        # Single isolated observation — noise in DBSCAN
        store.add_observation(_obs("lonely", x=100.0, y=100.0, ts=100.0))
        gist_ids = engine.consolidate_time_window(reference_time=1000.0)
        assert gist_ids == []
        # Observation should still be in short_term
        assert store.count_observations(tier=Tier.SHORT_TERM.value) == 1


class TestSummarizer:
    def test_concatenation_fallback(self):
        s = ConcatenationSummarizer()
        result = s.summarize(["saw chair", "saw table", "heard noise"])
        assert "chair" in result
        assert "table" in result
        assert "noise" in result
