"""Tests for WorkingMemory."""

import time

import numpy as np
import pytest

from emem.config import SpatioTemporalMemoryConfig
from emem.store import MemoryStore
from emem.types import ObservationNode
from emem.working_memory import WorkingMemory


@pytest.fixture
def store(tmp_path):
    config = SpatioTemporalMemoryConfig(
        db_path=str(tmp_path / "wm_test.db"),
        hnsw_path=str(tmp_path / "wm_hnsw.bin"),
        embedding_dim=32,
        hnsw_max_elements=1000,
    )
    s = MemoryStore(config=config)
    yield s
    s.close()


@pytest.fixture
def wm(store):
    config = SpatioTemporalMemoryConfig(
        flush_batch_size=3,
        flush_interval=100.0,  # Long interval so we control flushes manually
    )
    return WorkingMemory(store=store, config=config)


def _obs(text="test", x=0.0, y=0.0, ts=None):
    return ObservationNode(
        text=text,
        coordinates=np.array([x, y, 0.0]),
        timestamp=ts or time.time(),
    )


class TestBuffering:
    def test_add_buffers(self, wm, store):
        wm.add(_obs("a"))
        wm.add(_obs("b"))
        assert wm.buffer_size == 2
        assert store.count_observations() == 0  # Not flushed yet

    def test_auto_flush_on_batch_size(self, wm, store):
        wm.add(_obs("a"))
        wm.add(_obs("b"))
        wm.add(_obs("c"))  # batch_size=3 triggers flush
        assert wm.buffer_size == 0
        assert store.count_observations() == 3

    def test_manual_flush(self, wm, store):
        wm.add(_obs("a"))
        count = wm.flush()
        assert count == 1
        assert store.count_observations() == 1

    def test_flush_empty(self, wm):
        assert wm.flush() == 0

    def test_tier_upgraded_on_flush(self, wm, store):
        wm.add(_obs("a"))
        wm.flush()
        obs = list(store.temporal_query(time_range=(0, time.time() + 1000)))
        assert obs[0].tier == "short_term"


class TestRecent:
    def test_get_recent(self, wm):
        for i in range(5):
            wm.add(_obs(f"obs_{i}"))
        recent = wm.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].text == "obs_4"

    def test_recent_maxlen(self, store):
        config = SpatioTemporalMemoryConfig(
            working_memory_size=3,
            flush_batch_size=100,
            flush_interval=100.0,
        )
        wm = WorkingMemory(store=store, config=config)
        for i in range(10):
            wm.add(_obs(f"obs_{i}"))
        assert wm.recent_count == 3


class TestEpisodes:
    def test_episode_lifecycle(self, wm, store):
        ep_id = wm.start_episode("test_task")
        assert wm.active_episode_id == ep_id

        wm.add(_obs("during episode"))
        wm.flush()

        obs = store.get_episode_observations(ep_id)
        assert len(obs) == 1
        assert obs[0].episode_id == ep_id

        wm.end_episode(gist="Did some testing")
        assert wm.active_episode_id is None

        ep = store.get_episode(ep_id)
        assert ep.status == "completed"
        assert ep.gist == "Did some testing"

    def test_auto_assigns_episode(self, wm):
        ep_id = wm.start_episode("auto")
        obs = _obs("no episode set")
        wm.add(obs)
        # Should auto-assign active episode
        recent = wm.get_recent(1)
        assert recent[0].episode_id == ep_id

    def test_end_without_active(self, wm):
        result = wm.end_episode()
        assert result is None


class TestPosition:
    def test_tracks_position(self, wm):
        assert wm.current_position is None
        wm.add(_obs("a", x=5.0, y=10.0))
        np.testing.assert_array_almost_equal(wm.current_position, [5.0, 10.0, 0.0])
