import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from enviroment.wrappers.EdgeAntialiasObservation import EdgeAntialiasObservation


class EdgeAntialiasObservationVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that applies edge-aware antialiasing to RGB observations.

    Detects edges via luminance gradient comparison and blends pixels along
    detected edge directions. Non-edge pixels are left untouched.

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environment to apply the wrapper to.
        edge_threshold (float)           : Luminance difference threshold (0-1 scale). Default 0.08.
        strength (float)                 : Blend strength (0-1). Default 0.5.
    """

    def __init__(self, env: gym.vector.VectorEnv, edge_threshold=0.08, strength=0.5):
        super().__init__(env)
        n = env.observation_space.shape[0]
        h, w, c = env.observation_space.shape[1:]
        self.observation_space = Box(
            low=0, high=255, shape=(n, h, w, c), dtype=np.uint8
        )
        self._edge_threshold = edge_threshold
        self._strength = strength

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return (
            EdgeAntialiasObservation._antialias(
                obs, self._edge_threshold, self._strength
            ),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return (
            EdgeAntialiasObservation._antialias(
                obs, self._edge_threshold, self._strength
            ),
            info,
        )
