"""Tests for AI2-THOR adapter — requires ai2thor installed."""

import numpy as np
import pytest

pytestmark = pytest.mark.ai2thor


class TestAI2ThorAdapter:
    def test_reset_returns_frame_and_pos(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(scene="FloorPlan1")
        frame, pos = adapter.reset()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.dtype == np.uint8
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert all(isinstance(v, float) for v in pos)
        adapter.close()

    def test_step_returns_correct_tuple(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(scene="FloorPlan1")
        adapter.reset()
        frame, pos, reward, done, info = adapter.step(0)  # MoveAhead
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert isinstance(reward, float)
        assert reward == 0.0
        assert isinstance(done, bool)
        assert done is False
        assert isinstance(info, dict)
        assert "success" in info
        adapter.close()

    def test_available_actions(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(scene="FloorPlan1")
        actions = adapter.available_actions()
        assert isinstance(actions, list)
        assert len(actions) == 6
        assert actions == [0, 1, 2, 3, 4, 5]
        adapter.close()

    def test_multiple_steps(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(scene="FloorPlan1")
        adapter.reset()
        for action in [0, 2, 0, 3, 0]:  # move, rotate, move, rotate, move
            frame, pos, reward, done, info = adapter.step(action)
            assert frame.shape[2] == 3
        adapter.close()
