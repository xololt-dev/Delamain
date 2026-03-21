import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box


class OpticalFlowObservation(gym.Wrapper):
    """
    A wrapper that computes optical flow between oldest and newest frames
    in a stacked frame buffer, replacing the frame stack with the most recent
    frame plus optical flow (dx, dy) channels.

    Should be placed after SkipFrame in the wrapper chain:
        gym.make("CarRacing-v3") -> GreyscaleObservation -> SkipFrame -> OpticalFlowObservation

    Parameters:
        env (gymnasium.Env) : The environment to apply the wrapper to.
                              Expects stacked frames from SkipFrame.
        skip (int)          : Number of frames in the stack (must match SkipFrame skip value).
        channels (int)      : Number of channels per frame (e.g. 1 for greyscale, 3 for HSL/RGB).
                              Must match the channels parameter passed to SkipFrame.
    """

    def __init__(self, env, skip, channels):
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self._skip = skip
        self._channels = channels

        self.observation_space = Box(
            low=0, high=255,
            shape=(h, w, channels + 2),
            dtype=np.uint8
        )

    @staticmethod
    def _to_greyscale(frame: np.ndarray) -> np.ndarray:
        """Convert a multi-channel frame to single-channel greyscale."""
        if frame.shape[-1] == 1:
            return frame[..., 0]
        return np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    @staticmethod
    def _compute_flow(oldest: np.ndarray, newest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow between two greyscale frames using Farneback.

        Returns:
            (dx, dy) each as uint8 arrays normalized to 0-255.
        """
        oldest_grey = OpticalFlowObservation._to_greyscale(oldest)
        newest_grey = OpticalFlowObservation._to_greyscale(newest)

        oldest_u8 = np.clip(oldest_grey, 0, 255).astype(np.uint8)
        newest_u8 = np.clip(newest_grey, 0, 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            oldest_u8, newest_u8, None,
            pyr_scale=0.5,
            levels=1,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        dx = flow[..., 0]
        dy = flow[..., 1]

        # Normalize to 0-255 uint8 range
        # Use symmetric normalization around zero: map [-max, +max] -> [0, 255]
        max_val = max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
        dx_norm = ((dx / max_val) * 127.5 + 127.5).astype(np.uint8)
        dy_norm = ((dy / max_val) * 127.5 + 127.5).astype(np.uint8)

        return dx_norm, dy_norm

    @staticmethod
    def _transform_single(obs: np.ndarray, channels: int) -> np.ndarray:
        """Extract newest frame and compute optical flow from stacked frames."""
        oldest = obs[:, :, :channels]
        newest = obs[:, :, -channels:]

        dx, dy = OpticalFlowObservation._compute_flow(oldest, newest)

        return np.concatenate([newest, dx[..., np.newaxis], dy[..., np.newaxis]], axis=-1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_single(obs, self._channels), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._transform_single(obs, self._channels), info
