import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from enviroment.GaussianAntialiasObservation import GaussianAntialiasObservation


class GaussianAntialiasObservationVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that applies Gaussian blur antialiasing to RGB observations.

    Uses separable 1D Gaussian convolution for efficient per-channel smoothing.

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environment to apply the wrapper to.
        kernel_size (int)                : Size of the Gaussian kernel (odd integer). Default 3.
        sigma (float)                    : Standard deviation of the Gaussian. Default 0.8.
    """

    def __init__(self, env: gym.vector.VectorEnv, kernel_size=3, sigma=0.8):
        super().__init__(env)
        n = env.observation_space.shape[0]
        h, w, c = env.observation_space.shape[1:]
        self.observation_space = Box(
            low=0, high=255, shape=(n, h, w, c), dtype=np.uint8
        )
        self._kernel_size = kernel_size
        self._sigma = sigma

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return GaussianAntialiasObservation._antialias(obs, self._kernel_size, self._sigma), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return GaussianAntialiasObservation._antialias(obs, self._kernel_size, self._sigma), info
