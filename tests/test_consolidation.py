"""Tests for ConsolidationEngine."""

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.consolidation import ConcatenationSummarizer, ConsolidationEngine
from emem.store import MemoryStore
from emem.types import EdgeType, EntityNode, ObservationNode, Tier


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

        gist_ids = engine.consolidate_episode(ep_id)
        assert len(gist_ids) == 1

        gist = store.get_gist(gist_ids[0])
        assert "chair" in gist.text
        assert "table" in gist.text
        assert gist.source_observation_count == 2

        # Source observations should be demoted to long_term (text preserved)
        obs = store.get_episode_observations(ep_id)
        for o in obs:
            assert o.tier == Tier.LONG_TERM.value
            assert o.text != ""

    def test_consolidate_empty_episode(self, store, engine):
        ep_id = store.start_episode("empty", 1000.0)
        result = engine.consolidate_episode(ep_id)
        assert result == []


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

    def test_concatenation_synthesize_format(self):
        s = ConcatenationSummarizer()
        layer_texts = {
            "vlm": ["white cabinets", "wooden table"],
            "detections": ["chair", "table", "fridge"],
        }
        result = s.synthesize(layer_texts)
        assert "[detections]" in result
        assert "[vlm]" in result
        assert "||" in result
        assert "chair" in result
        assert "white cabinets" in result


class TestCrossLayerSynthesis:
    def _obs_with_layer(self, text, layer, x=1.0, y=1.0, ts=1001.0, episode_id=None):
        return ObservationNode(
            text=text,
            coordinates=np.array([x, y, 0.0]),
            timestamp=ts,
            layer_name=layer,
            episode_id=episode_id,
        )

    def test_multi_layer_uses_synthesize(self, store):
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
        )
        engine = ConsolidationEngine(store=store, config=config)

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(self._obs_with_layer("white cabinets", "vlm", ts=1001.0, episode_id=ep_id))
        store.add_observation(self._obs_with_layer("chair, table", "detections", ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        gist_ids = engine.consolidate_episode(ep_id)
        gist = store.get_gist(gist_ids[0])
        # Should use synthesize format with layer headers
        assert "[vlm]" in gist.text
        assert "[detections]" in gist.text
        assert gist.layer_name is None  # cross-layer

    def test_single_layer_uses_summarize(self, store):
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
        )
        engine = ConsolidationEngine(store=store, config=config)

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(self._obs_with_layer("saw chair", "vlm", ts=1001.0, episode_id=ep_id))
        store.add_observation(self._obs_with_layer("saw table", "vlm", ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        gist_ids = engine.consolidate_episode(ep_id)
        gist = store.get_gist(gist_ids[0])
        # Single layer: uses summarize (pipe-separated)
        assert "saw chair" in gist.text
        assert "saw table" in gist.text
        assert gist.layer_name == "vlm"


class FakeEntityExtractor:
    """Summarizer that also implements extract_entities."""

    def summarize(self, texts):
        return " | ".join(texts)

    def extract_entities(self, texts):
        entities = []
        for text in texts:
            for word in text.split():
                if word in ("chair", "table", "lamp"):
                    entities.append({"name": word, "entity_type": "furniture"})
        # Deduplicate by name
        seen = set()
        result = []
        for e in entities:
            if e["name"] not in seen:
                seen.add(e["name"])
                result.append(e)
        return result


class TestEntityExtraction:
    def test_episode_extracts_entities(self, store):
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
        )
        engine = ConsolidationEngine(store=store, config=config, llm_client=FakeEntityExtractor())

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(_obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id))
        store.add_observation(_obs("saw a table", x=5.5, y=5.5, ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        # Entities should have been created
        entities = store.query_entities()
        assert len(entities) >= 2
        names = {e.name for e in entities}
        assert "chair" in names
        assert "table" in names

        # OBSERVED_IN edges should exist
        for ent in entities:
            edges = store.get_edges(source_id=ent.id, edge_type=EdgeType.OBSERVED_IN)
            assert len(edges) > 0

        # COOCCURS_WITH edges between chair and table
        chair_ent = [e for e in entities if e.name == "chair"][0]
        cooccurring = store.get_cooccurring_entities(chair_ent.id)
        assert len(cooccurring) >= 1

    def test_no_extraction_with_fallback_summarizer(self, store, engine):
        """ConcatenationSummarizer returns no entities."""
        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(_obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        entities = store.query_entities()
        assert len(entities) == 0

    def test_merge_on_reobservation(self, store):
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
        )
        engine = ConsolidationEngine(store=store, config=config, llm_client=FakeEntityExtractor())

        # First episode — creates entity
        ep1 = store.start_episode("ep1", 1000.0)
        store.add_observation(_obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep1))
        store.add_observation(_obs("found a chair", x=5.2, y=5.2, ts=1002.0, episode_id=ep1))
        store.end_episode(ep1, 1003.0)
        engine.consolidate_episode(ep1)

        entities_before = store.query_entities(name="chair")
        assert len(entities_before) == 1
        count_before = entities_before[0].observation_count

        # Second episode — should merge into existing entity
        ep2 = store.start_episode("ep2", 2000.0)
        store.add_observation(_obs("chair still here", x=5.1, y=5.1, ts=2001.0, episode_id=ep2))
        store.add_observation(_obs("chair looks same", x=5.0, y=5.0, ts=2002.0, episode_id=ep2))
        store.end_episode(ep2, 2003.0)
        engine.consolidate_episode(ep2)

        entities_after = store.query_entities(name="chair")
        assert len(entities_after) == 1
        assert entities_after[0].observation_count > count_before


class TestTwoPhaseConsolidation:
    def test_long_term_still_searchable(self, store):
        """After consolidate_episode, observations are long_term and still findable."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
        )
        engine = ConsolidationEngine(store=store, config=config)

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(_obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id))
        store.add_observation(_obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        # Observations should be long_term with text preserved
        obs = store.get_episode_observations(ep_id)
        for o in obs:
            assert o.tier == Tier.LONG_TERM.value
            assert o.text != ""

        # Should still be findable via spatial_query (long_term != archived)
        spatial = store.spatial_query(center=np.array([1.0, 1.0, 0.0]), radius=3.0)
        assert len(spatial) > 0

        # Should still be findable via temporal_query
        temporal = store.temporal_query(time_range=(1000.0, 1010.0))
        assert len(temporal) > 0

    def test_archive_long_term(self, store):
        """After sufficient time, archive_long_term archives observations."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
            archive_after_seconds=500.0,
        )
        engine = ConsolidationEngine(store=store, config=config)

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(_obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id))
        store.add_observation(_obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        # Now archive — reference_time far in the future
        count = engine.archive_long_term(reference_time=2000.0)
        assert count == 2

        obs = store.get_episode_observations(ep_id)
        for o in obs:
            assert o.tier == Tier.ARCHIVED.value
            assert o.text == ""

    def test_archive_long_term_respects_threshold(self, store):
        """Young long_term observations are NOT archived."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
            archive_after_seconds=500.0,
        )
        engine = ConsolidationEngine(store=store, config=config)

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(_obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id))
        store.add_observation(_obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id))
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        # reference_time only slightly after — should NOT archive
        count = engine.archive_long_term(reference_time=1100.0)
        assert count == 0

        obs = store.get_episode_observations(ep_id)
        for o in obs:
            assert o.tier == Tier.LONG_TERM.value

    def test_maintenance_facade(self, tmp_path):
        """Test mem.maintenance() end-to-end."""
        from emem import SpatioTemporalMemory

        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "maint.db"),
            hnsw_path=str(tmp_path / "maint_hnsw.bin"),
            embedding_dim=32,
            archive_after_seconds=500.0,
        )
        sim_time = [1000.0]
        mem = SpatioTemporalMemory(
            config=config,
            get_current_time=lambda: sim_time[0],
        )

        ep_id = mem.start_episode("test")
        mem.add("chair", x=1.0, y=1.0, timestamp=1001.0)
        mem.add("table", x=1.5, y=1.5, timestamp=1002.0)
        sim_time[0] = 1003.0
        mem.end_episode()

        # Too early to archive
        sim_time[0] = 1100.0
        assert mem.maintenance() == 0

        # Now far enough
        sim_time[0] = 2000.0
        assert mem.maintenance() == 2
        mem.close()


class TestEarlyEntityExtraction:
    def test_entities_extracted_at_flush(self, tmp_path):
        """Entities are extracted at flush time, before end_episode."""
        from emem import SpatioTemporalMemory

        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "early_ent.db"),
            hnsw_path=str(tmp_path / "early_ent_hnsw.bin"),
            embedding_dim=32,
            flush_batch_size=2,
            entity_extract_flush_interval=1,
        )
        mem = SpatioTemporalMemory(
            config=config,
            llm_client=FakeEntityExtractor(),
        )

        ep_id = mem.start_episode("test")
        mem.add("saw a chair", x=5.0, y=5.0, timestamp=1001.0)
        mem.add("saw a table", x=5.5, y=5.5, timestamp=1002.0)
        # flush_batch_size=2 triggers auto-flush after 2nd add

        # Entities should exist BEFORE end_episode
        entities = mem.store.query_entities()
        assert len(entities) >= 2
        names = {e.name for e in entities}
        assert "chair" in names
        assert "table" in names

        mem.end_episode()
        mem.close()

    def test_no_double_extraction(self, tmp_path):
        """After flush-time extraction, consolidate_episode doesn't double entities."""
        from emem import SpatioTemporalMemory

        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "no_double.db"),
            hnsw_path=str(tmp_path / "no_double_hnsw.bin"),
            embedding_dim=32,
            flush_batch_size=2,
            entity_extract_flush_interval=1,
        )
        mem = SpatioTemporalMemory(
            config=config,
            llm_client=FakeEntityExtractor(),
        )

        ep_id = mem.start_episode("test")
        mem.add("saw a chair", x=5.0, y=5.0, timestamp=1001.0)
        mem.add("saw a table", x=5.5, y=5.5, timestamp=1002.0)

        # Entities exist from flush
        entities_before = mem.store.query_entities()
        chair_before = [e for e in entities_before if e.name == "chair"]
        assert len(chair_before) == 1
        count_before = chair_before[0].observation_count

        # end_episode triggers consolidate_episode which calls _extract_and_merge_entities
        # The dedup guard should prevent re-extraction
        mem.end_episode()

        entities_after = mem.store.query_entities(name="chair")
        assert len(entities_after) == 1
        # observation_count should NOT have doubled
        assert entities_after[0].observation_count == count_before

        mem.close()

    def test_graceful_fallback_summarizer(self, tmp_path):
        """ConcatenationSummarizer produces no entities, flush works normally."""
        from emem import SpatioTemporalMemory

        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "no_ext.db"),
            hnsw_path=str(tmp_path / "no_ext_hnsw.bin"),
            embedding_dim=32,
            flush_batch_size=2,
        )
        mem = SpatioTemporalMemory(config=config)

        ep_id = mem.start_episode("test")
        mem.add("saw a chair", x=5.0, y=5.0, timestamp=1001.0)
        mem.add("saw a table", x=5.5, y=5.5, timestamp=1002.0)

        # Should work fine, no entities
        entities = mem.store.query_entities()
        assert len(entities) == 0

        mem.end_episode()
        mem.close()
