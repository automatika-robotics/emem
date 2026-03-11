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


class TestTeleportTour:
    def test_teleport_mode_reset_builds_tour(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(scene="FloorPlan1", exploration_mode="teleport")
        frame, pos = adapter.reset()
        assert isinstance(frame, np.ndarray)
        assert len(adapter._tour) > 0
        adapter.close()

    def test_teleport_steps_change_position(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(
            scene="FloorPlan1", exploration_mode="teleport",
            rotations_per_waypoint=4,
        )
        frame, start_pos = adapter.reset()

        # First 3 steps are rotations at the same waypoint
        positions = [start_pos]
        for _ in range(3):
            frame, pos, reward, done, info = adapter.step(0)
            positions.append(pos)

        # Rotations should keep the same position
        for p in positions[1:4]:
            assert abs(p[0] - start_pos[0]) < 0.01
            assert abs(p[1] - start_pos[1]) < 0.01

        # Step 4 should teleport to next waypoint (different position)
        frame, new_pos, reward, done, info = adapter.step(0)
        dist = ((new_pos[0] - start_pos[0])**2 + (new_pos[1] - start_pos[1])**2)**0.5
        assert dist > 0.1, f"Expected teleport to new position, got {new_pos} vs {start_pos}"
        adapter.close()

    def test_teleport_tour_completes(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(
            scene="FloorPlan1", exploration_mode="teleport",
            max_waypoints=3, rotations_per_waypoint=2,
        )
        adapter.reset()

        # 3 waypoints x 2 rotations = 6 steps before done
        # (first waypoint rendered on reset, so steps cover rotations + teleports)
        done = False
        steps = 0
        while not done and steps < 20:
            _, _, _, done, info = adapter.step(0)
            steps += 1

        assert done, f"Tour should complete within 20 steps, took {steps}"
        adapter.close()

    def test_available_actions_teleport_mode(self):
        from harness.environments.ai2thor_adapter import AI2ThorAdapter

        adapter = AI2ThorAdapter(scene="FloorPlan1", exploration_mode="teleport")
        actions = adapter.available_actions()
        assert actions == [0]
        adapter.close()

    def test_farthest_point_sampling(self):
        from harness.environments.ai2thor_adapter import _farthest_point_sample

        positions = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 10.0, "y": 0.0, "z": 0.0},
            {"x": 5.0, "y": 0.0, "z": 5.0},
            {"x": 5.0, "y": 0.0, "z": -5.0},
        ]
        order = _farthest_point_sample(positions, start_idx=0)
        assert order[0] == 0
        # Second point should be the farthest from (0,0) = (10,0)
        assert order[1] == 1
        assert len(order) == 4

    def test_farthest_point_sampling_max_points(self):
        from harness.environments.ai2thor_adapter import _farthest_point_sample

        positions = [
            {"x": float(i), "y": 0.0, "z": 0.0} for i in range(10)
        ]
        order = _farthest_point_sample(positions, start_idx=0, max_points=3)
        assert len(order) == 3
        assert order[0] == 0
