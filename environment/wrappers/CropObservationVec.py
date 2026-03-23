import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Any

from environment.wrappers.CropObservation import CropObservation


class CropObservationVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that crops observations to a target size.

    Horizontally crops evenly (equal pixels removed from left and right).
    Vertically crops from the bottom only (top stays unchanged).

    Default target is 84x84, removing the bottom black info bar from
    CarRacing-v3's 96x96 frames.

    Should be placed after color-space transforms but before SkipFrameVec:
        gym.make_vec(...) -> HSLObservationVec -> CropObservationVec -> SkipFrameVec

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environment to apply the wrapper to.
        target_h (int)                   : Target height. Default 84.
        target_w (int)                   : Target width. Default 84.
    """

    def __init__(self, env: gym.vector.VectorEnv, target_h=84, target_w=84):
        super().__init__(env)
        n = env.observation_space.shape[0]
        h, w = env.observation_space.shape[1:3]
        c = env.observation_space.shape[3]

        assert (
            target_h <= h and target_w <= w
        ), f"Target size ({target_h}, {target_w}) exceeds observation size ({h}, {w})"

        self._crop_top = 0
        self._crop_bottom = h - target_h
        self._crop_left = (w - target_w) // 2
        self._crop_right = w - target_w - self._crop_left

        self.observation_space = Box(
            low=0, high=255, shape=(n, target_h, target_w, c), dtype=np.uint8
        )

    def _crop(self, obs: np.ndarray) -> np.ndarray:
        h = obs.shape[1]
        w = obs.shape[2]
        return obs[
            :,
            self._crop_top : h - self._crop_bottom,
            self._crop_left : w - self._crop_right,
            :,
        ]

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._crop(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._crop(obs), info
