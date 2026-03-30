import gymnasium as gym
import numpy as np
from typing import Any, TypeVar

ObsType = TypeVar("ObsType")


class ClipRewardVec(gym.vector.VectorWrapper):
    """
    A wrapper that clips each step's reward to the [-1, 1] range.

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environments to apply the wrapper to.

        enabled (bool) : Whether clipping is active. Can be toggled at runtime.
    """

    def __init__(self, env: gym.vector.VectorEnv, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled

    def step(self, actions):
        state, reward, terminated, truncated, info = self.env.step(actions)
        if self.enabled:
            reward = np.clip(reward, -1.0, 1.0)
        return state, reward, terminated, truncated, info
