import gymnasium as gym


class RepeatActionPenalty(gym.Wrapper):
    """
    A wrapper that penalizes the agent for repeating the same action consecutively.

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.

        thresholds (dict[int, int]) : Mapping of action index to the number of
            consecutive repeats required before the penalty kicks in. Actions not
            present in the dict are never penalized.

        penalty (float) : The amount subtracted from the reward each step once
            the repetition threshold is met.

        enabled (bool) : Whether the penalty is active. Can be toggled at runtime.
    """

    def __init__(
        self,
        env: gym.Env,
        thresholds: dict[int, int],
        penalty: float,
        enabled: bool = True,
    ):
        super().__init__(env)
        self._thresholds = thresholds
        self._penalty = penalty
        self.enabled = enabled
        self._prev_action = None
        self._repeat_count = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        if self.enabled:
            if action == self._prev_action:
                self._repeat_count += 1
                if (
                    action in self._thresholds
                    and self._repeat_count >= self._thresholds[action]
                ):
                    reward -= self._penalty
            else:
                self._prev_action = action
                self._repeat_count = 1

        return state, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, str | None] = None):
        self._prev_action = None
        self._repeat_count = 0
        return self.env.reset(seed=seed, options=options)
