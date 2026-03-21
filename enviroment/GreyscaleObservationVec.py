import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Any

from enviroment.GreyscaleObservation import GreyscaleObservation


class GreyscaleObservationVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that converts RGB observations to greyscale (single channel).

    Uses ITU-R BT.601 luminance: Y = 0.299R + 0.587G + 0.114B

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environment to apply the wrapper to.
    """

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        n = env.observation_space.shape[0]
        h, w = env.observation_space.shape[1:3]
        self.observation_space = Box(
            low=0, high=255, shape=(n, h, w, 1), dtype=np.uint8
        )

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return GreyscaleObservation._to_greyscale(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return GreyscaleObservation._to_greyscale(obs), info
