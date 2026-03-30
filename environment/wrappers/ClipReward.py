import gymnasium as gym
import numpy as np


class ClipReward(gym.Wrapper):
    """
    A wrapper that clips each step's reward to the [-1, 1] range.

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.

        enabled (bool) : Whether clipping is active. Can be toggled at runtime.
    """

    def __init__(self, env: gym.Env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.enabled:
            reward = np.clip(reward, -1.0, 1.0)
        return state, reward, terminated, truncated, info
