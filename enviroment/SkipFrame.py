import gymnasium as gym

class SkipFrame(gym.Wrapper):
    """
    A wrapper for skipping frames in the environment to speed up training.

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.

        skip (int) : The number of frames to skip.
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # Executes the action for the specified number of frames, accumulating rewards.
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info