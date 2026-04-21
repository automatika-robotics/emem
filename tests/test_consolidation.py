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
        store.add_observation(
            _obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id)
        )
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
        store.add_observation(
            self._obs_with_layer("white cabinets", "vlm", ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            self._obs_with_layer(
                "chair, table", "detections", ts=1002.0, episode_id=ep_id
            )
        )
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
        store.add_observation(
            self._obs_with_layer("saw chair", "vlm", ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            self._obs_with_layer("saw table", "vlm", ts=1002.0, episode_id=ep_id)
        )
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
        engine = ConsolidationEngine(
            store=store, config=config, llm_client=FakeEntityExtractor()
        )

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(
            _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("saw a table", x=5.5, y=5.5, ts=1002.0, episode_id=ep_id)
        )
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
        store.add_observation(
            _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id)
        )
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        entities = store.query_entities()
        assert len(entities) == 0

    def test_merge_on_reobservation(self, store):
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0,
            consolidation_min_samples=2,
        )
        engine = ConsolidationEngine(
            store=store, config=config, llm_client=FakeEntityExtractor()
        )

        # First episode — creates entity
        ep1 = store.start_episode("ep1", 1000.0)
        store.add_observation(
            _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep1)
        )
        store.add_observation(
            _obs("found a chair", x=5.2, y=5.2, ts=1002.0, episode_id=ep1)
        )
        store.end_episode(ep1, 1003.0)
        engine.consolidate_episode(ep1)

        entities_before = store.query_entities(name="chair")
        assert len(entities_before) == 1
        count_before = entities_before[0].observation_count

        # Second episode — should merge into existing entity
        ep2 = store.start_episode("ep2", 2000.0)
        store.add_observation(
            _obs("chair still here", x=5.1, y=5.1, ts=2001.0, episode_id=ep2)
        )
        store.add_observation(
            _obs("chair looks same", x=5.0, y=5.0, ts=2002.0, episode_id=ep2)
        )
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
        store.add_observation(
            _obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id)
        )
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
        store.add_observation(
            _obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id)
        )
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
        store.add_observation(
            _obs("saw a chair", x=1.0, y=1.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("saw a table", x=1.5, y=1.5, ts=1002.0, episode_id=ep_id)
        )
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

        mem.start_episode("test")
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

        mem.start_episode("test")
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

        mem.start_episode("test")
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

        mem.start_episode("test")
        mem.add("saw a chair", x=5.0, y=5.0, timestamp=1001.0)
        mem.add("saw a table", x=5.5, y=5.5, timestamp=1002.0)

        # Should work fine, no entities
        entities = mem.store.query_entities()
        assert len(entities) == 0

        mem.end_episode()
        mem.close()


class AttributedEntityExtractor:
    """Fake summarizer that returns per-observation entity attribution.

    Mirrors the contract that ``LLMClient.extract_entities`` returns
    after parsing: each entity carries a 0-based ``observation_index``
    pointing into the batch. The LLM prompt uses 1-based indices for
    human readability; ``_parse_entities`` converts them to 0-based
    before this layer sees them.
    """

    def summarize(self, texts):
        return " | ".join(texts)

    def extract_entities(self, texts):
        keywords = ("chair", "table", "lamp", "fridge")
        out = []
        for i, text in enumerate(texts):
            for word in text.split():
                if word in keywords:
                    out.append({
                        "name": word,
                        "entity_type": "furniture",
                        "confidence": 0.9,
                        "observation_index": i,  # 0-based post-parser
                    })
        return out


class TestPerObservationAttribution:
    """A2: per-observation entity attribution + batch-link fallback."""

    def test_parser_reads_observation_index(self):
        from emem.consolidation import _parse_entities

        raw = (
            '[{"name":"chair","entity_type":"furniture","confidence":0.9,'
            '"observation_index":2}]'
        )
        parsed = _parse_entities(raw)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "chair"
        assert parsed[0]["observation_index"] == 1  # 1-based -> 0-based

    def test_parser_missing_index_is_none(self):
        from emem.consolidation import _parse_entities

        raw = '[{"name":"chair","entity_type":"furniture","confidence":0.9}]'
        parsed = _parse_entities(raw)
        assert parsed[0]["observation_index"] is None

    def test_parser_rejects_bad_index_string(self):
        from emem.consolidation import _parse_entities

        raw = '[{"name":"chair","observation_index":"one"}]'
        parsed = _parse_entities(raw)
        assert parsed[0]["observation_index"] is None

    def test_parser_rejects_zero_index(self):
        """1-based input must be >= 1; 0 or negative means missing."""
        from emem.consolidation import _parse_entities

        raw = '[{"name":"chair","observation_index":0}]'
        parsed = _parse_entities(raw)
        assert parsed[0]["observation_index"] is None

    def test_attributed_entity_links_to_single_observation(self, store):
        """With per-obs attribution, OBSERVED_IN points only at the
        attributed observation, not at every observation in the batch."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0, consolidation_min_samples=2
        )
        engine = ConsolidationEngine(
            store=store, config=config, llm_client=AttributedEntityExtractor()
        )

        ep_id = store.start_episode("test", 1000.0)
        chair_obs = _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id)
        table_obs = _obs("spotted a table", x=10.0, y=10.0, ts=1002.0, episode_id=ep_id)
        store.add_observation(chair_obs)
        store.add_observation(table_obs)
        store.end_episode(ep_id, 1003.0)

        engine.consolidate_episode(ep_id)

        chair = [e for e in store.query_entities() if e.name == "chair"][0]
        table = [e for e in store.query_entities() if e.name == "table"][0]

        chair_edges = store.get_edges(
            source_id=chair.id, edge_type=EdgeType.OBSERVED_IN
        )
        table_edges = store.get_edges(
            source_id=table.id, edge_type=EdgeType.OBSERVED_IN
        )
        assert len(chair_edges) == 1
        assert chair_edges[0].target_id == chair_obs.id
        assert len(table_edges) == 1
        assert table_edges[0].target_id == table_obs.id

    def test_attributed_entity_uses_observation_coords(self, store):
        """Entity coordinates should come from its source observation,
        not the batch centroid."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0, consolidation_min_samples=2
        )
        engine = ConsolidationEngine(
            store=store, config=config, llm_client=AttributedEntityExtractor()
        )

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(
            _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("spotted a table", x=50.0, y=50.0, ts=1002.0, episode_id=ep_id)
        )
        store.end_episode(ep_id, 1003.0)
        engine.consolidate_episode(ep_id)

        chair = [e for e in store.query_entities() if e.name == "chair"][0]
        assert abs(chair.coordinates[0] - 5.0) < 0.01
        assert abs(chair.coordinates[1] - 5.0) < 0.01

        table = [e for e in store.query_entities() if e.name == "table"][0]
        assert abs(table.coordinates[0] - 50.0) < 0.01

    def test_attributed_entity_count_starts_at_one(self, store):
        """Per-observation attribution means observation_count starts at 1,
        not at batch size."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0, consolidation_min_samples=2
        )
        engine = ConsolidationEngine(
            store=store, config=config, llm_client=AttributedEntityExtractor()
        )

        ep_id = store.start_episode("test", 1000.0)
        for i in range(4):
            store.add_observation(
                _obs(
                    f"object {i}",
                    x=float(i),
                    y=0.0,
                    ts=1001.0 + i,
                    episode_id=ep_id,
                )
            )
        store.add_observation(
            _obs("saw a chair", x=100.0, y=100.0, ts=1005.0, episode_id=ep_id)
        )
        store.end_episode(ep_id, 1006.0)
        engine.consolidate_episode(ep_id)

        chair = [e for e in store.query_entities() if e.name == "chair"][0]
        # Only one observation mentions "chair", so count should be 1.
        assert chair.observation_count == 1

    def test_fallback_batch_links_when_index_missing(self, store, caplog):
        """When the LLM omits observation_index, fall back to linking
        every entity to every observation in the batch (previous
        behaviour), and emit a warning."""
        import logging

        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0, consolidation_min_samples=2
        )
        engine = ConsolidationEngine(
            store=store,
            config=config,
            llm_client=FakeEntityExtractor(),  # no observation_index
        )

        ep_id = store.start_episode("test", 1000.0)
        store.add_observation(
            _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("nothing here", x=5.5, y=5.5, ts=1002.0, episode_id=ep_id)
        )
        store.end_episode(ep_id, 1003.0)

        with caplog.at_level(logging.WARNING, logger="emem.consolidation"):
            engine.consolidate_episode(ep_id)

        chair = [e for e in store.query_entities() if e.name == "chair"][0]
        edges = store.get_edges(source_id=chair.id, edge_type=EdgeType.OBSERVED_IN)
        # Fallback batch-links to both observations.
        assert len(edges) == 2
        assert any("observation_index" in r.message for r in caplog.records)

    def test_cooccurrence_requires_same_observation(self, store):
        """With attribution, two entities co-occur only if they came
        from the same observation."""
        config = SpatioTemporalMemoryConfig(
            consolidation_window=100.0, consolidation_min_samples=2
        )
        engine = ConsolidationEngine(
            store=store, config=config, llm_client=AttributedEntityExtractor()
        )

        ep_id = store.start_episode("test", 1000.0)
        # Two observations; chair in the first, table in the second —
        # they must NOT be marked as co-occurring.
        store.add_observation(
            _obs("saw a chair", x=5.0, y=5.0, ts=1001.0, episode_id=ep_id)
        )
        store.add_observation(
            _obs("spotted a table", x=10.0, y=10.0, ts=1002.0, episode_id=ep_id)
        )
        store.end_episode(ep_id, 1003.0)
        engine.consolidate_episode(ep_id)

        chair = [e for e in store.query_entities() if e.name == "chair"][0]
        cooc = store.get_cooccurring_entities(chair.id)
        assert len(cooc) == 0

        # Now a second episode where both are seen in the same observation
        # (they co-occur).
        ep2 = store.start_episode("together", 2000.0)
        store.add_observation(
            _obs(
                "the chair and the table together",
                x=1.0,
                y=1.0,
                ts=2001.0,
                episode_id=ep2,
            )
        )
        store.end_episode(ep2, 2002.0)
        engine.consolidate_episode(ep2)

        chair2 = [e for e in store.query_entities() if e.name == "chair"][0]
        cooc2 = store.get_cooccurring_entities(chair2.id)
        assert any(e.name == "table" for e in cooc2)


class TestContextAwareEntityMerge:
    """A3: merges gated on observation-text cosine, not name cosine alone."""

    def _merge_and_list(self, tmp_path, threshold, obs_a_text, obs_b_text):
        """Helper: ingest two observations that share the entity name
        ``chair`` under different contexts, run entity extraction, and
        return the final list of ``chair`` entities in the store."""
        from emem import SpatioTemporalMemory

        config = SpatioTemporalMemoryConfig(
            db_path=str(tmp_path / "ctx.db"),
            hnsw_path=str(tmp_path / "ctx_hnsw.bin"),
            embedding_dim=4,
            flush_batch_size=1,  # each obs extracted immediately
            entity_text_similarity_threshold=threshold,
            entity_spatial_radius=10.0,
        )

        class _ControlledEmbedder:
            """Deterministic embedder: name embeds share one direction,
            observation texts split into two clusters so context cosine
            distinguishes them."""

            dim = 4

            def embed(self, texts):
                out = np.zeros((len(texts), 4), dtype=np.float32)
                for i, t in enumerate(texts):
                    if t == "chair":
                        out[i] = np.array([1.0, 0.0, 0.0, 0.0])
                    elif t == obs_a_text:
                        out[i] = np.array([0.0, 1.0, 0.0, 0.0])
                    elif t == obs_b_text:
                        out[i] = np.array([0.0, 0.0, 1.0, 0.0])
                    else:
                        out[i] = np.array([0.0, 0.0, 0.0, 1.0])
                return out

        class _FakeLLM:
            def summarize(self, texts):
                return " | ".join(texts)

            def extract_entities(self, texts):
                # One entity per observation — attributed.
                return [
                    {
                        "name": "chair",
                        "entity_type": "furniture",
                        "confidence": 0.9,
                        "observation_index": i,
                    }
                    for i in range(len(texts))
                ]

        mem = SpatioTemporalMemory(
            config=config,
            embedding_provider=_ControlledEmbedder(),
            llm_client=_FakeLLM(),
        )
        mem.start_episode("ctx_test")
        mem.add(obs_a_text, x=0.0, y=0.0, timestamp=1000.0)
        mem.add(obs_b_text, x=1.0, y=1.0, timestamp=1001.0)
        mem.end_episode(consolidate=False)
        entities = mem.store.query_entities(name="chair")
        mem.close()
        return entities

    def test_context_dissimilar_prevents_merge(self, tmp_path):
        """Two 'chair' observations whose embeddings are orthogonal
        (cosine=0) must NOT merge when the threshold is above zero."""
        entities = self._merge_and_list(
            tmp_path,
            threshold=0.5,
            obs_a_text="kitchen chair",
            obs_b_text="bedroom chair",
        )
        # Should be two distinct chair entities, not one.
        assert len(entities) == 2

    def test_context_zero_threshold_allows_merge(self, tmp_path):
        """Threshold at zero should reproduce pre-A3 behaviour: name +
        spatial alone decide, so the two chairs collapse into one."""
        entities = self._merge_and_list(
            tmp_path,
            threshold=0.0,
            obs_a_text="kitchen chair",
            obs_b_text="bedroom chair",
        )
        assert len(entities) == 1
        assert entities[0].observation_count == 2

    def test_context_similar_allows_merge(self, tmp_path):
        """Two observations with the same context text should merge."""
        entities = self._merge_and_list(
            tmp_path,
            threshold=0.5,
            obs_a_text="kitchen chair",
            obs_b_text="kitchen chair",  # identical => cosine = 1
        )
        assert len(entities) == 1
        assert entities[0].observation_count == 2
