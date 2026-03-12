"""Tests for interoception / body state support."""

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


_T = 1000.0


@pytest.fixture
def mem(tmp_path):
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "intero.db"),
        hnsw_path=str(tmp_path / "intero_hnsw.bin"),
        embedding_dim=32,
    )
    m = SpatioTemporalMemory(
        config=config,
        embedding_provider=FakeEmbedder(32),
        get_current_time=lambda: _T,
    )
    yield m
    m.close()


class TestAddBodyState:
    def test_add_body_state(self, mem):
        obs_id = mem.add_body_state("battery: 45%", layer_name="battery")
        assert obs_id
        mem.save()
        obs = mem.store.get_observation(obs_id)
        assert obs.source_type == "interoception"
        assert obs.layer_name == "battery"
        assert obs.text == "battery: 45%"

    def test_body_state_uses_current_position(self, mem):
        mem.add("world observation", x=10.0, y=10.0)
        obs_id = mem.add_body_state("battery: 45%", layer_name="battery")
        mem.save()
        obs = mem.store.get_observation(obs_id)
        np.testing.assert_array_almost_equal(obs.coordinates, [10.0, 10.0, 0.0])

    def test_body_state_no_position_fallback(self, mem):
        assert mem.current_position is None
        obs_id = mem.add_body_state("battery: 45%", layer_name="battery")
        mem.save()
        obs = mem.store.get_observation(obs_id)
        np.testing.assert_array_almost_equal(obs.coordinates, [0.0, 0.0, 0.0])

    def test_position_not_updated_by_body_state(self, mem):
        mem.add("world obs", x=10.0, y=10.0)
        np.testing.assert_array_almost_equal(mem.current_position, [10.0, 10.0, 0.0])
        mem.add_body_state("battery: 45%", layer_name="battery")
        np.testing.assert_array_almost_equal(mem.current_position, [10.0, 10.0, 0.0])


class TestBodyStatusTool:
    def test_body_status(self, mem):
        mem.add_body_state("45% remaining", layer_name="battery", timestamp=_T - 120)
        mem.add_body_state("72C across 4 cores", layer_name="cpu_temp", timestamp=_T - 30)
        result = mem.body_status()
        assert "Body Status:" in result
        assert "45% remaining" in result
        assert "72C across 4 cores" in result

    def test_body_status_layer_filter(self, mem):
        mem.add_body_state("45% remaining", layer_name="battery")
        mem.add_body_state("72C", layer_name="cpu_temp")
        result = mem.body_status(layers=["battery"])
        assert "45% remaining" in result
        assert "72C" not in result

    def test_body_status_empty(self, mem):
        result = mem.body_status()
        assert "No body state" in result

    def test_body_status_returns_latest(self, mem):
        mem.add_body_state("80% remaining", layer_name="battery", timestamp=_T - 300)
        mem.add_body_state("45% remaining", layer_name="battery", timestamp=_T - 30)
        result = mem.body_status()
        assert "45% remaining" in result
        # The old reading should not appear (only latest per layer)
        assert "80% remaining" not in result


class TestContextIncludesBodyStatus:
    def test_context_includes_body_status(self, mem):
        mem.add("world obs", x=5.0, y=5.0)
        mem.add_body_state("45% remaining", layer_name="battery")
        result = mem.get_current_context()
        assert "Body Status:" in result
        assert "45% remaining" in result


class TestSourceTypeFilters:
    def test_temporal_query_source_type_filter(self, mem):
        mem.add("world obs", x=5.0, y=5.0, timestamp=_T - 60)
        mem.add_body_state("45% remaining", layer_name="battery", timestamp=_T - 30)
        mem.save()

        result = mem.store.temporal_query(
            last_n_seconds=300,
            reference_time=_T,
            source_type="interoception",
        )
        assert len(result) == 1
        assert result[0].source_type == "interoception"

    def test_spatial_query_exclude_source_type(self, mem):
        mem.add("world obs", x=5.0, y=5.0, timestamp=_T - 60)
        mem.add_body_state("45% remaining", layer_name="battery", timestamp=_T - 30)
        mem.save()

        result = mem.store.spatial_query(
            center=np.array([5.0, 5.0, 0.0]),
            radius=10.0,
            exclude_source_type="interoception",
        )
        for obs in result:
            assert obs.source_type != "interoception"
        assert any(obs.text == "world obs" for obs in result)


class TestConsolidationMixed:
    def test_consolidation_mixed_episode(self, mem):
        ep_id = mem.start_episode("patrol")
        mem.add("rough terrain ahead", x=5.0, y=5.0)
        mem.add_body_state("battery: 70%", layer_name="battery")
        mem.add_body_state("cpu: 65C", layer_name="cpu_temp")
        mem.end_episode()

        ep = mem.store.get_episode(ep_id)
        assert ep.status == "completed"
        # Observations should be demoted to long_term after consolidation
        obs_list = mem.store.get_episode_observations(ep_id)
        for obs in obs_list:
            assert obs.tier == "long_term"
