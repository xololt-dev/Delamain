import gymnasium as gym
import torch
import numpy as np

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
        self.frames = np.zeros((96, 96, 12), dtype=np.uint8)
        self.rewards = np.zeros(4, dtype=np.float32)

    def step(self, action):
        # Executes the action for the specified number of frames, accumulating rewards.
        # total_reward = 0.0
        for _ in range(self._skip):
            self.frames = np.roll(self.frames, -3)
            self.rewards = np.roll(self.rewards, -1)
            state, reward, terminated, truncated, info = self.env.step(action)
            # total_reward += reward
            self.rewards[3] = reward
            self.frames[:, :, 9:12] = state
            if terminated:
                break

        return self.frames, np.sum(self.rewards, dtype=np.float32), terminated, truncated, info
    
        # total_reward = 0.0
        # frames = []
        # for _ in range(self._skip):
        #     state, reward, terminated, truncated, info = self.env.step(action)
        #     total_reward += reward
        #     frames.append(state)
        #     if terminated:
        #         break

        # while len(frames) < self._skip:
        #     frames.append(frames[-1])
        # total_state = numpy.concatenate(frames, axis=2)

        # return total_state, total_reward, terminated, truncated, info
    def reset(self, seed: int | None = None, options: dict[str] | None = None):
        self.frames.fill(0)
        self.rewards.fill(0)
        state, info = self.env.reset(seed=seed, options=options)
        self.frames[:, :, :3] = state
        self.frames[:, :, 3:6] = state
        self.frames[:, :, 6:9] = state
        self.frames[:, :, 9:12] = state
        # self.frames = np.tile(state, (1, 1, 4))
        # print(np.concatenate([state, state, state, state], out=self.frames, axis=2))
        # print(state[:, 48, :])
        print("state shape:", state.shape)
        print("frames shape:", self.frames.shape)
        print("frames[:, 48, :]:", self.frames[:, 48, :])

        return self.frames, info
