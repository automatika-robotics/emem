from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("harness")


def _farthest_point_sample(
    positions: list[dict[str, float]],
    start_idx: int = 0,
    max_points: int | None = None,
) -> list[int]:
    """Order positions by farthest-point sampling for maximum spatial coverage.

    Greedy algorithm: start from *start_idx*, each next point is the one
    whose minimum distance to all already-selected points is largest.

    :param positions: List of ``{"x": ..., "y": ..., "z": ...}`` dicts.
    :param start_idx: Index of the starting position.
    :param max_points: Maximum number of points to select (``None`` = all).
    :returns: Ordered list of indices into *positions*.
    """
    n = len(positions)
    if n == 0:
        return []
    max_points = max_points or n

    coords = np.array([[p["x"], p["z"]] for p in positions])
    selected = [start_idx]
    # min_dist[i] = min distance from position i to any selected point
    min_dist = np.full(n, np.inf)
    min_dist[start_idx] = -1.0  # exclude start

    for _ in range(min(max_points, n) - 1):
        last = coords[selected[-1]]
        dists = np.linalg.norm(coords - last, axis=1)
        np.minimum(min_dist, dists, out=min_dist)
        min_dist[np.array(selected)] = -1.0
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            break
        selected.append(next_idx)

    return selected


class AI2ThorAdapter:
    """Wraps an AI2-THOR scene to produce RGB frames and agent coordinates.

    Uses floor-plane ``(x, z)`` from AI2-THOR as ``(x, y)`` to match the
    2D convention used by MiniGridAdapter and eMEM.

    :param scene: AI2-THOR scene name (e.g. ``"FloorPlan1"``).
    :param grid_size: Navigation grid spacing in metres.
    :param width: Frame width in pixels.
    :param height: Frame height in pixels.
    :param headless: Use CloudRendering (no display required).
    :param exploration_mode: ``"random"`` for standard actions,
        ``"teleport"`` for systematic teleport tour with rotation sweeps.
    :param rotations_per_waypoint: Number of rotation stops per waypoint
        in teleport mode (default 4 = every 90°).
    :param max_waypoints: Maximum waypoints to visit in teleport mode
        (``None`` = all reachable positions).
    """

    ACTIONS = [
        "MoveAhead",
        "MoveBack",
        "RotateRight",
        "RotateLeft",
        "LookUp",
        "LookDown",
    ]

    def __init__(
        self,
        scene: str = "FloorPlan1",
        grid_size: float = 0.25,
        width: int = 300,
        height: int = 300,
        headless: bool = False,
        exploration_mode: str = "random",
        rotations_per_waypoint: int = 4,
        max_waypoints: int | None = None,
    ):
        from ai2thor.controller import Controller

        self._scene = scene
        self._grid_size = grid_size
        self._exploration_mode = exploration_mode
        self._rotations_per_wp = rotations_per_waypoint
        self._max_waypoints = max_waypoints
        self._rotation_step = 360.0 / rotations_per_waypoint

        kwargs: dict[str, Any] = {
            "scene": scene,
            "gridSize": grid_size,
            "width": width,
            "height": height,
        }
        if headless:
            from ai2thor.platform import CloudRendering

            kwargs["platform"] = CloudRendering

        self._controller = Controller(**kwargs)

        # Teleport tour state
        self._tour: list[dict[str, float]] = []
        self._tour_idx: int = 0
        self._rotation_idx: int = 0

    def _pos(self) -> tuple[float, float]:
        """Extract floor-plane (x, z) from agent metadata."""
        p = self._controller.last_event.metadata["agent"]["position"]
        return (p["x"], p["z"])

    def _build_tour(self) -> None:
        """Compute the teleport tour using farthest-point sampling."""
        event = self._controller.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        if not positions:
            log.warning("GetReachablePositions returned empty list")
            self._tour = []
            return

        # Find the index closest to the agent's current position
        agent_pos = self._controller.last_event.metadata["agent"]["position"]
        coords = np.array([[p["x"], p["z"]] for p in positions])
        agent_xz = np.array([agent_pos["x"], agent_pos["z"]])
        start_idx = int(np.argmin(np.linalg.norm(coords - agent_xz, axis=1)))

        order = _farthest_point_sample(
            positions,
            start_idx=start_idx,
            max_points=self._max_waypoints,
        )
        self._tour = [positions[i] for i in order]
        self._tour_idx = 0
        self._rotation_idx = 0

        log.info(
            "Teleport tour: %d waypoints from %d reachable positions",
            len(self._tour),
            len(positions),
        )

    def _teleport_to_waypoint(self, wp_idx: int, rotation: float = 0.0) -> Any:
        """Teleport to a waypoint with given rotation."""
        wp = self._tour[wp_idx]
        event = self._controller.step(
            action="TeleportFull",
            x=wp["x"],
            y=wp["y"],
            z=wp["z"],
            rotation={"x": 0, "y": rotation, "z": 0},
            horizon=0.0,
            standing=True,
        )
        return event

    def reset(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Reset the scene.

        In teleport mode, computes the tour from reachable positions.

        :returns: ``(rgb_frame, (x, y))``.
        """
        self._controller.reset(self._scene)
        event = self._controller.step(
            action="Initialize",
            gridSize=self._grid_size,
        )

        if self._exploration_mode == "teleport":
            self._build_tour()
            if self._tour:
                event = self._teleport_to_waypoint(0, rotation=0.0)

        return event.frame, self._pos()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, tuple[float, float], float, bool, dict[str, Any]]:
        """Take a navigation action.

        In ``"random"`` mode, executes the given action directly.
        In ``"teleport"`` mode, the *action* parameter is ignored —
        the adapter cycles through waypoints with rotation sweeps.

        :param action: Action index (ignored in teleport mode).
        :returns: ``(rgb_frame, (x, y), reward, done, info)``.
        """
        if self._exploration_mode == "teleport":
            return self._teleport_step()

        action_name = self.ACTIONS[action]
        event = self._controller.step(action=action_name)
        info = {"success": event.metadata["lastActionSuccess"]}
        return event.frame, self._pos(), 0.0, False, info

    def _teleport_step(
        self,
    ) -> tuple[np.ndarray, tuple[float, float], float, bool, dict[str, Any]]:
        """Advance the teleport tour by one step."""
        if not self._tour:
            frame = self._controller.last_event.frame
            return frame, self._pos(), 0.0, True, {"success": False}

        self._rotation_idx += 1

        if self._rotation_idx >= self._rotations_per_wp:
            # Move to the next waypoint
            self._rotation_idx = 0
            self._tour_idx += 1

            if self._tour_idx >= len(self._tour):
                # Tour complete
                frame = self._controller.last_event.frame
                return (
                    frame,
                    self._pos(),
                    0.0,
                    True,
                    {
                        "success": True,
                        "tour_complete": True,
                    },
                )

            rotation = 0.0
            event = self._teleport_to_waypoint(self._tour_idx, rotation)
        else:
            # Rotate at current waypoint
            rotation = self._rotation_idx * self._rotation_step
            event = self._teleport_to_waypoint(self._tour_idx, rotation)

        info = {
            "success": event.metadata["lastActionSuccess"],
            "waypoint": self._tour_idx,
            "rotation": rotation,
            "waypoints_total": len(self._tour),
        }
        return event.frame, self._pos(), 0.0, False, info

    def available_actions(self) -> list[int]:
        """Return list of valid action indices."""
        if self._exploration_mode == "teleport":
            return [0]  # single dummy action; step() ignores it
        return list(range(len(self.ACTIONS)))

    def close(self):
        self._controller.stop()
