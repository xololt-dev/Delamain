import gymnasium as gym
import torch
import numpy

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
        # total_reward = 0.0
        # for _ in range(self._skip):
        #     state, reward, terminated, truncated, info = self.env.step(action)
        #     total_reward += reward
        #     if terminated:
        #         break
        # return state, total_reward, terminated, truncated, info
        total_reward = 0.0
        total_state = None
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
            if total_state is None:
                total_state = state
            else:
                total_state = numpy.concatenate((total_state, state), 2)
                # total_state = torch.cat((total_state, state), 0)
        return total_state, total_reward, terminated, truncated, info