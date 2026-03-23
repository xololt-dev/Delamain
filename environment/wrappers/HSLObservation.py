import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class HSLObservation(gym.Wrapper):
    """
    A wrapper that converts RGB observations to HSL (Hue, Saturation, Lightness).

    All three channels are encoded as uint8 (0-255):
        H: 0-360 degrees mapped to 0-255
        S: 0-100% mapped to 0-255
        L: 0-100% mapped to 0-255

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.
    """

    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )

    @staticmethod
    def _to_hsl(obs: np.ndarray) -> np.ndarray:
        r = obs[..., 0].astype(np.float32) / 255.0
        g = obs[..., 1].astype(np.float32) / 255.0
        b = obs[..., 2].astype(np.float32) / 255.0

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue
        h = np.zeros_like(delta)
        mask = delta != 0

        # Red is max
        mask_r = mask & (cmax == r)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360

        # Green is max
        mask_g = mask & (cmax == g)
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360

        # Blue is max
        mask_b = mask & (cmax == b)
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

        # Lightness
        l = (cmax + cmin) / 2.0

        # Saturation
        s = np.zeros_like(delta)
        mask_sl = (l > 0) & (l < 1)
        denom = 1.0 - np.abs(2.0 * l - 1.0)
        s[mask_sl] = delta[mask_sl] / denom[mask_sl]

        # Map to uint8: H 0-360 -> 0-255, S/L 0-1 -> 0-255
        h_out = (h / 360.0 * 255.0).astype(np.uint8)
        s_out = (s * 255.0).astype(np.uint8)
        l_out = (l * 255.0).astype(np.uint8)

        return np.stack([h_out, s_out, l_out], axis=-1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_hsl(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_hsl(obs), info
