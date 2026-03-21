import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class GaussianAntialiasObservation(gym.Wrapper):
    """
    A wrapper that applies Gaussian blur antialiasing to RGB observations.

    Uses separable 1D Gaussian convolution (horizontal then vertical pass)
    for efficient per-channel smoothing. Pure numpy, no extra dependencies.

    Should be placed before color-space transforms in the wrapper chain:
        gym.make("CarRacing-v3") -> GaussianAntialiasObservation -> HSLObservation -> SkipFrame

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.
        kernel_size (int)   : Size of the Gaussian kernel (odd integer). Default 3.
        sigma (float)       : Standard deviation of the Gaussian. Default 0.8.
    """

    def __init__(self, env, kernel_size=3, sigma=0.8):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )
        self._kernel_size = kernel_size
        self._sigma = sigma

    @staticmethod
    def _make_gaussian_kernel(kernel_size, sigma):
        half = kernel_size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    @staticmethod
    def _antialias(obs: np.ndarray, kernel_size=3, sigma=0.8) -> np.ndarray:
        kernel = GaussianAntialiasObservation._make_gaussian_kernel(kernel_size, sigma)
        half = kernel_size // 2
        float_obs = obs.astype(np.float64)

        # Build padding for width axis (second-to-last) and height axis (third-to-last)
        # Works for both 3D (H,W,C) and 4D (N,H,W,C)
        ndim = float_obs.ndim
        width_pad = [(0, 0)] * ndim
        width_pad[-2] = (half, half)
        height_pad = [(0, 0)] * ndim
        height_pad[-3] = (half, half)

        w_size = float_obs.shape[-2]
        h_size = float_obs.shape[-3]

        # Horizontal pass (along width axis)
        padded = np.pad(float_obs, width_pad, mode="edge")
        result = np.zeros_like(float_obs)
        for i in range(kernel_size):
            slc = [slice(None)] * ndim
            slc[-2] = slice(i, i + w_size)
            result += kernel[i] * padded[tuple(slc)]

        # Vertical pass (along height axis)
        padded = np.pad(result, height_pad, mode="edge")
        result = np.zeros_like(float_obs)
        for i in range(kernel_size):
            slc = [slice(None)] * ndim
            slc[-3] = slice(i, i + h_size)
            result += kernel[i] * padded[tuple(slc)]

        return np.clip(result + 0.5, 0, 255).astype(np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._antialias(obs, self._kernel_size, self._sigma), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._antialias(obs, self._kernel_size, self._sigma), info
