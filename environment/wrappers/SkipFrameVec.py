import numpy as np
import gymnasium as gym
from typing import Any, TypeVar

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class SkipFrameVec(gym.vector.VectorWrapper):
    """
    A wrapper for skipping frames in the environment to speed up training.

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environments to apply the wrapper to.

        skip (int) : The number of frames to skip.

        channels (int): Number of channels per frame (e.g. 1 for greyscale, 3 for HSL/RGB).
    """

    def __init__(self, env: gym.vector.VectorEnv, skip: int, channels: int = 3):
        super().__init__(env)
        self._skip = skip
        self._channels = channels
        n = env.observation_space.shape[0]
        h, w = env.observation_space.shape[1:3]
        self.frames = np.zeros((n, h, w, skip * channels), dtype=np.uint8)
        self.rewards = np.zeros((n, skip), dtype=np.float32)

    def step(self, actions: ActType):
        # Executes the action for the specified number of frames, accumulating rewards.
        for _ in range(self._skip):
            self.frames[:, :, :, : -self._channels] = self.frames[
                :, :, :, self._channels :
            ]
            self.rewards[:, :-1] = self.rewards[:, 1:]
            state, reward, terminated, truncated, info = self.env.step(actions)

            self.rewards[:, -1] = reward
            self.frames[:, :, :, -self._channels :] = state

            if np.any(terminated):
                break

        return (
            self.frames.copy(),
            np.sum(self.rewards, dtype=np.float32, axis=1),
            terminated,
            truncated,
            info,
        )

    def reset(
        self, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.rewards.fill(0)
        self.frames.fill(0)
        state, info = self.env.reset(seed=seed, options=options)

        self.frames = np.tile(state, (1, 1, 1, self._skip))

        return self.frames.copy(), info
