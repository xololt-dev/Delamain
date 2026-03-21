import gymnasium as gym
import numpy as np


class SkipFrame(gym.Wrapper):
    """
    A wrapper for skipping frames in the environment to speed up training.

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.

        skip (int) : The number of frames to skip.
    """

    def __init__(self, env, skip, channels=3):
        super().__init__(env)
        self._skip = skip
        self._channels = channels
        self.frames = np.zeros((96, 96, skip * channels), dtype=np.uint8)
        self.rewards = np.zeros(skip, dtype=np.float32)

    def step(self, action):
        # Executes the action for the specified number of frames, accumulating rewards.
        for _ in range(self._skip):
            self.frames[:, :, : -self._channels] = self.frames[:, :, self._channels :]
            self.rewards[:-1] = self.rewards[1:]
            state, reward, terminated, truncated, info = self.env.step(action)
            self.rewards[-1] = reward
            self.frames[:, :, -self._channels :] = state
            if terminated:
                break

        return (
            self.frames.copy(),
            np.sum(self.rewards, dtype=np.float32),
            terminated,
            truncated,
            info,
        )

    def reset(self, seed: int | None = None, options: dict[str] | None = None):
        self.rewards.fill(0)
        state, info = self.env.reset(seed=seed, options=options)

        self.frames = np.tile(state, self._skip)

        return self.frames.copy(), info
