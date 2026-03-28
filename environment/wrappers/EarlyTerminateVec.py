import gymnasium as gym
import numpy as np
from typing import Any, TypeVar

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class EarlyTerminateVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that terminates episodes early when the cumulative reward
    for any environment drops below its peak by more than a set threshold.

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environments to apply the wrapper to.

        threshold (float) : The maximum allowed drop from the peak cumulative reward
            before the episode is terminated.

        penalty (float) : A penalty subtracted from the reward when early termination
            triggers. Default is 0.0 (no penalty).
    """

    def __init__(self, env: gym.vector.VectorEnv, threshold: float, penalty: float = 0.0):
        super().__init__(env)
        self._threshold = threshold
        self._penalty = penalty
        n = env.observation_space.shape[0]
        self._episode_rewards = np.zeros(n, dtype=np.float32)
        self._max_episode_rewards = np.zeros(n, dtype=np.float32)

    def step(self, actions: ActType):
        state, reward, terminated, truncated, info = self.env.step(actions)

        self._episode_rewards += reward
        np.maximum(self._episode_rewards, self._max_episode_rewards, out=self._max_episode_rewards)

        delta = self._max_episode_rewards - self._episode_rewards
        early = delta > self._threshold

        reward = np.where(early, reward - self._penalty, reward)
        terminated = np.logical_or(terminated, early)

        return state, reward, terminated, truncated, info

    def reset(
        self, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._episode_rewards.fill(0)
        self._max_episode_rewards.fill(0)
        return self.env.reset(seed=seed, options=options)
