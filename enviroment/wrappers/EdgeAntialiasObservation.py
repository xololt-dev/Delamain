import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class EdgeAntialiasObservation(gym.Wrapper):
    """
    A wrapper that applies edge-aware antialiasing to RGB observations.

    Detects edges via luminance gradient comparison and blends pixels along
    detected edge directions. Non-edge pixels are left untouched.

    Should be placed before color-space transforms in the wrapper chain:
        gym.make("CarRacing-v3") -> EdgeAntialiasObservation -> HSLObservation -> SkipFrame

    Parameters:
        env (gymnasium.Env)     : The environment to apply the wrapper to.
        edge_threshold (float)  : Luminance difference threshold (0-1 scale, 0-255 effective).
                                  Pixels differing more than this from neighbors are edge pixels. Default 0.08.
        strength (float)        : Blend strength (0-1). 0 = no effect, 1 = full blend. Default 0.5.
    """

    def __init__(self, env, edge_threshold=0.08, strength=0.5):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )
        self._edge_threshold = edge_threshold
        self._strength = strength

    @staticmethod
    def _antialias(obs: np.ndarray, edge_threshold=0.08, strength=0.5) -> np.ndarray:
        float_obs = obs.astype(np.float64)

        # Compute luminance (BT.601)
        lum = 0.299 * float_obs[..., 0] + 0.587 * float_obs[..., 1] + 0.114 * float_obs[..., 2]

        # Luminance differences with right and bottom neighbors
        diff_right = np.abs(lum[:, 1:] - lum[:, :-1])
        diff_down = np.abs(lum[1:, :] - lum[:-1, :])

        threshold = edge_threshold * 255.0

        # Edge maps: True where luminance difference exceeds threshold
        edge_h = diff_right > threshold  # (H, W-1)
        edge_v = diff_down > threshold   # (H-1, W)

        result = float_obs.copy()

        # Horizontal edge blending (for vertical edges — blend left-right)
        h_count = edge_h.sum()
        if h_count > 0:
            left = float_obs[:, :-1, :][edge_h]
            right = float_obs[:, 1:, :][edge_h]
            blended = 0.5 * left + 0.5 * right
            result[:, :-1, :][edge_h] = (
                result[:, :-1, :][edge_h] * (1.0 - strength) + blended * strength
            )
            result[:, 1:, :][edge_h] = (
                result[:, 1:, :][edge_h] * (1.0 - strength) + blended * strength
            )

        # Vertical edge blending (for horizontal edges — blend up-down)
        v_count = edge_v.sum()
        if v_count > 0:
            up = float_obs[:-1, :, :][edge_v]
            down = float_obs[1:, :, :][edge_v]
            blended = 0.5 * up + 0.5 * down
            result[:-1, :, :][edge_v] = (
                result[:-1, :, :][edge_v] * (1.0 - strength) + blended * strength
            )
            result[1:, :, :][edge_v] = (
                result[1:, :, :][edge_v] * (1.0 - strength) + blended * strength
            )

        return np.clip(result + 0.5, 0, 255).astype(np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._antialias(obs, self._edge_threshold, self._strength), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._antialias(obs, self._edge_threshold, self._strength), info
