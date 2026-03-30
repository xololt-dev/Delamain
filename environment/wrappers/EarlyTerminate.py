import gymnasium as gym


class EarlyTerminate(gym.Wrapper):
    """
    A wrapper that terminates the episode early when the cumulative reward
    drops below its peak by more than a set threshold.

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.

        threshold (float) : The maximum allowed drop from the peak cumulative reward
            before the episode is terminated.

        penalty (float) : A penalty subtracted from the reward when early termination
            triggers. Default is 0.0 (no penalty).
    """

    def __init__(self, env: gym.Env, threshold: float, penalty: float = 0.0):
        super().__init__(env)
        self._threshold = threshold
        self._penalty = penalty
        self._episode_reward = 0.0
        self._max_episode_reward = 0.0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        self._episode_reward += reward
        if self._episode_reward > self._max_episode_reward:
            self._max_episode_reward = self._episode_reward

        delta = self._max_episode_reward - self._episode_reward
        if delta > self._threshold:
            reward -= self._penalty
            terminated = True

        return state, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str] | None = None):
        self._episode_reward = 0.0
        self._max_episode_reward = 0.0
        return self.env.reset(seed=seed, options=options)
