import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Any

from enviroment.HSLObservation import HSLObservation


class HSLObservationVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that converts RGB observations to HSL (Hue, Saturation, Lightness).

    All three channels are encoded as uint8 (0-255):
        H: 0-360 degrees mapped to 0-255
        S: 0-100% mapped to 0-255
        L: 0-100% mapped to 0-255

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environment to apply the wrapper to.
    """

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        n = env.observation_space.shape[0]
        h, w = env.observation_space.shape[1:3]
        self.observation_space = Box(
            low=0, high=255, shape=(n, h, w, 3), dtype=np.uint8
        )

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return HSLObservation._to_hsl(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return HSLObservation._to_hsl(obs), info
