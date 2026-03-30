import gymnasium as gym
import numpy as np
from typing import Any, TypeVar

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class RepeatActionPenaltyVec(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that penalizes the agent for repeating the same action
    consecutively.

    Parameters:
        env (gymnasium.vector.VectorEnv) : The vector environments to apply the
            wrapper to.

        thresholds (dict[int, int]) : Mapping of action index to the number of
            consecutive repeats required before the penalty kicks in. Actions not
            present in the dict are never penalized.

        penalty (float) : The amount subtracted from the reward each step once
            the repetition threshold is met.

        enabled (bool) : Whether the penalty is active. Can be toggled at runtime.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        thresholds: dict[int, int],
        penalty: float,
        enabled: bool = True,
    ):
        super().__init__(env)
        self._thresholds = thresholds
        self._penalty = penalty
        self.enabled = enabled
        n = env.observation_space.shape[0]
        self._prev_actions = np.full(n, -1, dtype=np.int32)
        self._repeat_counts = np.zeros(n, dtype=np.int32)

    def step(self, actions: ActType):
        state, reward, terminated, truncated, info = self.env.step(actions)

        if self.enabled:
            actions = np.asarray(actions)

            same = actions == self._prev_actions
            self._repeat_counts = np.where(same, self._repeat_counts + 1, 1)
            self._prev_actions = actions.copy()

            penalty_mask = np.zeros(len(actions), dtype=bool)
            for action_idx, threshold in self._thresholds.items():
                hit = (actions == action_idx) & (self._repeat_counts >= threshold)
                penalty_mask |= hit

            reward = np.where(penalty_mask, reward - self._penalty, reward)

        return state, reward, terminated, truncated, info

    def reset(
        self, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._prev_actions.fill(-1)
        self._repeat_counts.fill(0)
        return self.env.reset(seed=seed, options=options)
