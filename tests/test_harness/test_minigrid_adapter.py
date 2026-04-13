"""Tests for MiniGrid adapter — requires minigrid installed."""

import numpy as np
import pytest

pytestmark = pytest.mark.minigrid


class TestMiniGridAdapter:
    def test_reset_returns_frame_and_pos(self):
        from harness.environments.minigrid_adapter import MiniGridAdapter

        adapter = MiniGridAdapter()
        frame, pos = adapter.reset()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        adapter.close()

    def test_step_returns_correct_tuple(self):
        from harness.environments.minigrid_adapter import MiniGridAdapter

        adapter = MiniGridAdapter()
        adapter.reset()
        frame, pos, reward, done, info = adapter.step(2)  # forward
        assert isinstance(frame, np.ndarray)
        assert isinstance(pos, tuple)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        adapter.close()

    def test_available_actions(self):
        from harness.environments.minigrid_adapter import MiniGridAdapter

        adapter = MiniGridAdapter()
        actions = adapter.available_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert 0 in actions  # left
        assert 2 in actions  # forward
        adapter.close()

    def test_multiple_steps(self):
        from harness.environments.minigrid_adapter import MiniGridAdapter

        adapter = MiniGridAdapter()
        adapter.reset()
        for _ in range(20):
            action = 2  # forward
            _, _, _, done, _ = adapter.step(action)
            if done:
                adapter.reset()
        adapter.close()
