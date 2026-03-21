import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Any

from enviroment.wrappers.OpticalFlowObservation import OpticalFlowObservation


class OpticalFlowObservationVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that computes optical flow between oldest and newest frames
    in a stacked frame buffer, replacing the frame stack with the most recent
    frame plus optical flow (dx, dy) channels.

    Should be placed after SkipFrameVec in the wrapper chain:
        gym.make_vec(...) -> GreyscaleObservationVec -> SkipFrameVec -> OpticalFlowObservationVec

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environment to apply the wrapper to.
                                           Expects stacked frames from SkipFrameVec.
        skip (int)                       : Number of frames in the stack (must match SkipFrameVec skip value).
        channels (int)                   : Number of channels per frame (e.g. 1 for greyscale, 3 for HSL/RGB).
                                          Must match the channels parameter passed to SkipFrameVec.
    """

    def __init__(self, env: gym.vector.VectorEnv, skip: int, channels: int):
        super().__init__(env)
        n = env.observation_space.shape[0]
        h, w = env.observation_space.shape[1:3]
        self._skip = skip
        self._channels = channels

        self.observation_space = Box(
            low=0, high=255, shape=(n, h, w, channels + 2), dtype=np.uint8
        )

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        transformed = np.array(
            [
                OpticalFlowObservation._transform_single(obs[i], self._channels)
                for i in range(obs.shape[0])
            ]
        )
        return transformed, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        transformed = np.array(
            [
                OpticalFlowObservation._transform_single(obs[i], self._channels)
                for i in range(obs.shape[0])
            ]
        )
        return transformed, info
