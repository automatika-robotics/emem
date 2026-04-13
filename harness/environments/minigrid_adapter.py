from typing import Any

import numpy as np


class MiniGridAdapter:
    """Wraps a MiniGrid environment to produce RGB frames and agent coordinates."""

    def __init__(
        self,
        env_name: str = "MiniGrid-MultiRoom-N6-v0",
        tile_size: int = 32,
    ):
        import gymnasium
        from minigrid.wrappers import RGBImgObsWrapper

        self.env = RGBImgObsWrapper(
            gymnasium.make(env_name, render_mode="rgb_array"),
            tile_size=tile_size,
        )

    def reset(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Reset the environment.

        :returns: ``(rgb_frame, (x, y))``.
        """
        obs, _info = self.env.reset()
        return obs["image"], tuple(self.env.unwrapped.agent_pos)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, tuple[int, int], float, bool, dict[str, Any]]:
        """Take an action.

        :param action: Action index (0=left, 1=right, 2=forward, 3=pickup,
            4=drop, 5=toggle, 6=done).
        :returns: ``(rgb_frame, (x, y), reward, done, info)``.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = tuple(self.env.unwrapped.agent_pos)
        return obs["image"], pos, float(reward), terminated or truncated, info

    def available_actions(self) -> list[int]:
        """Return list of valid action indices."""
        return list(range(self.env.action_space.n))

    def close(self):
        self.env.close()
