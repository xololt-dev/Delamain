import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class GreyscaleObservation(gym.Wrapper):
    """
    A wrapper that converts RGB observations to greyscale (single channel).

    Uses ITU-R BT.601 luminance: Y = 0.299R + 0.587G + 0.114B

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.
    """

    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    @staticmethod
    def _to_greyscale(obs: np.ndarray) -> np.ndarray:
        grey = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return grey[..., np.newaxis]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_greyscale(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_greyscale(obs), info
