from typing import Any

import numpy as np


class AI2ThorAdapter:
    """Wraps an AI2-THOR scene to produce RGB frames and agent coordinates.

    Uses floor-plane ``(x, z)`` from AI2-THOR as ``(x, y)`` to match the
    2D convention used by MiniGridAdapter and eMEM.
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
    ):
        from ai2thor.controller import Controller

        self._scene = scene
        self._grid_size = grid_size

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

    def _pos(self) -> tuple[float, float]:
        """Extract floor-plane (x, z) from agent metadata."""
        p = self._controller.last_event.metadata["agent"]["position"]
        return (p["x"], p["z"])

    def reset(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Reset the scene.

        :returns: ``(rgb_frame, (x, y))``.
        """
        self._controller.reset(self._scene)
        event = self._controller.step(
            action="Initialize", gridSize=self._grid_size,
        )
        return event.frame, self._pos()

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, tuple[float, float], float, bool, dict[str, Any]]:
        """Take a navigation action.

        :param action: Action index (0=MoveAhead, 1=MoveBack, 2=RotateRight,
            3=RotateLeft, 4=LookUp, 5=LookDown).
        :returns: ``(rgb_frame, (x, y), reward, done, info)``.
        """
        action_name = self.ACTIONS[action]
        event = self._controller.step(action=action_name)
        info = {"success": event.metadata["lastActionSuccess"]}
        return event.frame, self._pos(), 0.0, False, info

    def available_actions(self) -> list[int]:
        """Return list of valid action indices."""
        return list(range(len(self.ACTIONS)))

    def close(self):
        self._controller.stop()
