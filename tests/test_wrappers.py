import os
import pytest
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.image as mpimg

from enviroment.wrappers.SkipFrame import SkipFrame
from enviroment.wrappers.SkipFrameVec import SkipFrameVec
from enviroment.wrappers.GreyscaleObservation import GreyscaleObservation
from enviroment.wrappers.HSLObservation import HSLObservation
from enviroment.wrappers.GreyscaleObservationVec import GreyscaleObservationVec
from enviroment.wrappers.HSLObservationVec import HSLObservationVec
from enviroment.wrappers.GaussianAntialiasObservation import (
    GaussianAntialiasObservation,
)
from enviroment.wrappers.GaussianAntialiasObservationVec import (
    GaussianAntialiasObservationVec,
)
from enviroment.wrappers.EdgeAntialiasObservation import EdgeAntialiasObservation
from enviroment.wrappers.EdgeAntialiasObservationVec import EdgeAntialiasObservationVec
from enviroment.wrappers.OpticalFlowObservation import OpticalFlowObservation
from enviroment.wrappers.OpticalFlowObservationVec import OpticalFlowObservationVec


SKIP = 4


@pytest.fixture
def skip_env():
    """Scalar CarRacing env wrapped with SkipFrame."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = SkipFrame(env, skip=SKIP)
    return env


@pytest.fixture
def skip_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with SkipFrameVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=2,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = SkipFrameVec(env, skip=SKIP)
    return env


class TestSkipFrame:
    def test_reset_returns_correct_shape(self, skip_env):
        frames, info = skip_env.reset(seed=42)
        assert frames.shape == (96, 96, SKIP * 3)
        assert frames.dtype == np.uint8

    def test_step_returns_correct_shape(self, skip_env):
        skip_env.reset(seed=42)
        action = 3  # gas
        frames, reward, terminated, truncated, info = skip_env.step(action)
        assert frames.shape == (96, 96, SKIP * 3)
        assert frames.dtype == np.uint8
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_accumulates_rewards(self, skip_env):
        skip_env.reset(seed=42)
        total_reward = 0.0
        for _ in range(3):
            action = 3  # gas
            frames, reward, terminated, truncated, info = skip_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        # Just verify reward is a finite number (can't predict exact value)
        assert np.isfinite(total_reward)

    def test_reset_fills_frames(self, skip_env):
        frames, _ = skip_env.reset(seed=42)
        # All 4 frame slots should be filled with the same initial frame
        # (reset copies the initial state skip times)
        assert not np.all(frames == 0)

    def test_multiple_episodes(self, skip_env):
        """Verify reset/step cycle works across multiple episodes."""
        for ep in range(2):
            frames, _ = skip_env.reset(seed=42 + ep)
            assert frames.shape == (96, 96, SKIP * 3)
            action = 0  # do nothing
            frames, _, terminated, truncated, _ = skip_env.step(action)
            assert frames.shape == (96, 96, SKIP * 3)
        skip_env.close()


class TestSkipFrameVec:
    def test_reset_returns_correct_shape(self, skip_vec_env):
        frames, info = skip_vec_env.reset(seed=[42, 43])
        assert frames.shape == (2, 96, 96, SKIP * 3)
        assert frames.dtype == np.uint8

    def test_step_returns_correct_shape(self, skip_vec_env):
        skip_vec_env.reset(seed=[42, 43])
        actions = [3, 3]  # gas for both envs
        frames, reward, terminated, truncated, info = skip_vec_env.step(actions)
        assert frames.shape == (2, 96, 96, SKIP * 3)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)

    def test_step_accumulates_rewards(self, skip_vec_env):
        skip_vec_env.reset(seed=[42, 43])
        actions = [3, 3]
        frames, reward, _, _, _ = skip_vec_env.step(actions)
        # Just verify rewards are finite
        assert np.all(np.isfinite(reward))

    def test_multiple_steps(self, skip_vec_env):
        skip_vec_env.reset(seed=[42, 43])
        for _ in range(3):
            actions = [0, 3]  # do nothing, gas
            frames, reward, terminated, truncated, info = skip_vec_env.step(actions)
            assert frames.shape == (2, 96, 96, SKIP * 3)
        skip_vec_env.close()


@pytest.fixture
def grey_env():
    """Scalar CarRacing env wrapped with GreyscaleObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = GreyscaleObservation(env)
    return env


@pytest.fixture
def grey_skip_env():
    """Scalar CarRacing env wrapped with GreyscaleObservation + SkipFrame."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = GreyscaleObservation(env)
    env = SkipFrame(env, skip=SKIP, channels=1)
    return env


class TestGreyscaleObservation:
    def test_reset_returns_correct_shape(self, grey_env):
        obs, info = grey_env.reset(seed=42)
        assert obs.shape == (96, 96, 1)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, grey_env):
        grey_env.reset(seed=42)
        obs, reward, terminated, truncated, info = grey_env.step(3)
        assert obs.shape == (96, 96, 1)
        assert obs.dtype == np.uint8

    def test_observation_space(self, grey_env):
        assert grey_env.observation_space.shape == (96, 96, 1)
        assert grey_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, grey_env):
        grey_env.reset(seed=42)
        obs, _, _, _, _ = grey_env.step(3)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_non_zero_after_reset(self, grey_env):
        obs, _ = grey_env.reset(seed=42)
        assert not np.all(obs == 0)

    def test_with_skipframe(self, grey_skip_env):
        frames, _ = grey_skip_env.reset(seed=42)
        assert frames.shape == (96, 96, SKIP)
        assert frames.dtype == np.uint8

        frames, _, terminated, truncated, _ = grey_skip_env.step(3)
        assert frames.shape == (96, 96, SKIP)
        grey_skip_env.close()


@pytest.fixture
def hsl_env():
    """Scalar CarRacing env wrapped with HSLObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = HSLObservation(env)
    return env


@pytest.fixture
def hsl_skip_env():
    """Scalar CarRacing env wrapped with HSLObservation + SkipFrame."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = HSLObservation(env)
    env = SkipFrame(env, skip=SKIP, channels=3)
    return env


class TestHSLObservation:
    def test_reset_returns_correct_shape(self, hsl_env):
        obs, info = hsl_env.reset(seed=42)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, hsl_env):
        hsl_env.reset(seed=42)
        obs, reward, terminated, truncated, info = hsl_env.step(3)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8

    def test_observation_space(self, hsl_env):
        assert hsl_env.observation_space.shape == (96, 96, 3)
        assert hsl_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, hsl_env):
        hsl_env.reset(seed=42)
        obs, _, _, _, _ = hsl_env.step(3)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_non_zero_after_reset(self, hsl_env):
        obs, _ = hsl_env.reset(seed=42)
        assert not np.all(obs == 0)

    def test_with_skipframe(self, hsl_skip_env):
        frames, _ = hsl_skip_env.reset(seed=42)
        assert frames.shape == (96, 96, SKIP * 3)
        assert frames.dtype == np.uint8

        frames, _, terminated, truncated, _ = hsl_skip_env.step(3)
        assert frames.shape == (96, 96, SKIP * 3)
        hsl_skip_env.close()


NUM_ENVS = 2


@pytest.fixture
def grey_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with GreyscaleObservationVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = GreyscaleObservationVec(env)
    return env


@pytest.fixture
def grey_skip_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with GreyscaleObservationVec + SkipFrameVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = GreyscaleObservationVec(env)
    env = SkipFrameVec(env, skip=SKIP, channels=1)
    return env


class TestGreyscaleObservationVec:
    def test_reset_returns_correct_shape(self, grey_vec_env):
        obs, info = grey_vec_env.reset(seed=[42, 43])
        assert obs.shape == (NUM_ENVS, 96, 96, 1)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, grey_vec_env):
        grey_vec_env.reset(seed=[42, 43])
        obs, reward, terminated, truncated, info = grey_vec_env.step([3, 3])
        assert obs.shape == (NUM_ENVS, 96, 96, 1)
        assert obs.dtype == np.uint8

    def test_observation_space(self, grey_vec_env):
        assert grey_vec_env.observation_space.shape == (NUM_ENVS, 96, 96, 1)
        assert grey_vec_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, grey_vec_env):
        grey_vec_env.reset(seed=[42, 43])
        obs, _, _, _, _ = grey_vec_env.step([3, 3])
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_with_skipframe_vec(self, grey_skip_vec_env):
        frames, _ = grey_skip_vec_env.reset(seed=[42, 43])
        assert frames.shape == (NUM_ENVS, 96, 96, SKIP)
        assert frames.dtype == np.uint8

        frames, _, _, _, _ = grey_skip_vec_env.step([3, 3])
        assert frames.shape == (NUM_ENVS, 96, 96, SKIP)
        grey_skip_vec_env.close()


@pytest.fixture
def hsl_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with HSLObservationVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = HSLObservationVec(env)
    return env


@pytest.fixture
def hsl_skip_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with HSLObservationVec + SkipFrameVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = HSLObservationVec(env)
    env = SkipFrameVec(env, skip=SKIP, channels=3)
    return env


class TestHSLObservationVec:
    def test_reset_returns_correct_shape(self, hsl_vec_env):
        obs, info = hsl_vec_env.reset(seed=[42, 43])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, hsl_vec_env):
        hsl_vec_env.reset(seed=[42, 43])
        obs, reward, terminated, truncated, info = hsl_vec_env.step([3, 3])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert obs.dtype == np.uint8

    def test_observation_space(self, hsl_vec_env):
        assert hsl_vec_env.observation_space.shape == (NUM_ENVS, 96, 96, 3)
        assert hsl_vec_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, hsl_vec_env):
        hsl_vec_env.reset(seed=[42, 43])
        obs, _, _, _, _ = hsl_vec_env.step([3, 3])
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_with_skipframe_vec(self, hsl_skip_vec_env):
        frames, _ = hsl_skip_vec_env.reset(seed=[42, 43])
        assert frames.shape == (NUM_ENVS, 96, 96, SKIP * 3)
        assert frames.dtype == np.uint8

        frames, _, _, _, _ = hsl_skip_vec_env.step([3, 3])
        assert frames.shape == (NUM_ENVS, 96, 96, SKIP * 3)
        hsl_skip_vec_env.close()


# --- Visual snapshot tests ---
# These save PNGs to tests/images/ for manual inspection.
# Run with: venv/bin/python -m pytest tests/test_wrappers.py::TestVisualSnapshots -v -s

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


class TestVisualSnapshots:
    def test_greyscale_snapshot(self):
        """Save original RGB and greyscale-converted frame as PNGs."""
        out_dir = os.path.join(IMAGES_DIR, "greyscale")
        os.makedirs(out_dir, exist_ok=True)

        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        obs, _ = env.reset(seed=42)
        obs, _, _, _, _ = env.step(3)

        # Save original RGB
        rgb_path = os.path.join(out_dir, "original_rgb.png")
        mpimg.imsave(rgb_path, obs)

        # Convert and save greyscale
        grey = GreyscaleObservation._to_greyscale(obs)
        grey_path = os.path.join(out_dir, "greyscale.png")
        mpimg.imsave(grey_path, grey[:, :, 0], cmap="gray")

        # Save greyscale with 3 channels (for comparison display)
        grey_3ch = np.repeat(grey, 3, axis=-1)
        grey_3ch_path = os.path.join(out_dir, "greyscale_3ch.png")
        mpimg.imsave(grey_3ch_path, grey_3ch)

        print(f"\n  Saved: {os.path.abspath(rgb_path)}")
        print(f"  Saved: {os.path.abspath(grey_path)}")
        print(f"  Saved: {os.path.abspath(grey_3ch_path)}")

        env.close()

    def test_hsl_snapshot(self):
        """Save original RGB and each HSL channel as individual PNGs."""
        out_dir = os.path.join(IMAGES_DIR, "hsl")
        os.makedirs(out_dir, exist_ok=True)

        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        obs, _ = env.reset(seed=42)
        obs, _, _, _, _ = env.step(3)

        # Save original RGB
        rgb_path = os.path.join(out_dir, "original_rgb.png")
        mpimg.imsave(rgb_path, obs)

        # Convert to HSL and save each channel
        hsl = HSLObservation._to_hsl(obs)

        h_path = os.path.join(out_dir, "hsl_hue.png")
        mpimg.imsave(h_path, hsl[:, :, 0], cmap="hsv")

        s_path = os.path.join(out_dir, "hsl_saturation.png")
        mpimg.imsave(s_path, hsl[:, :, 1], cmap="gray")

        l_path = os.path.join(out_dir, "hsl_lightness.png")
        mpimg.imsave(l_path, hsl[:, :, 2], cmap="gray")

        # Save all 3 channels stacked as one image for direct RGB-like viewing
        hsl_path = os.path.join(out_dir, "hsl_combined.png")
        mpimg.imsave(hsl_path, hsl)

        print(f"\n  Saved: {os.path.abspath(rgb_path)}")
        print(f"  Saved: {os.path.abspath(h_path)}")
        print(f"  Saved: {os.path.abspath(s_path)}")
        print(f"  Saved: {os.path.abspath(l_path)}")
        print(f"  Saved: {os.path.abspath(hsl_path)}")

        env.close()

    def test_greyscale_vec_snapshot(self):
        """Save original RGB and greyscale-converted frames from Vec env as PNGs."""
        out_dir = os.path.join(IMAGES_DIR, "greyscale_vec")
        os.makedirs(out_dir, exist_ok=True)

        seeds = [42, 43]

        # Unwrapped vec env for original RGB
        env_raw = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        obs_raw, _ = env_raw.reset(seed=seeds)
        obs_raw, _, _, _, _ = env_raw.step([3, 3])

        # Wrapped vec env for greyscale
        env_grey = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        env_grey = GreyscaleObservationVec(env_grey)
        obs_grey, _ = env_grey.reset(seed=seeds)
        obs_grey, _, _, _, _ = env_grey.step([3, 3])

        for i in range(2):
            # Save original RGB
            rgb_path = os.path.join(out_dir, f"original_env{i}.png")
            mpimg.imsave(rgb_path, obs_raw[i])

            # Save greyscale (single channel with gray cmap)
            grey_path = os.path.join(out_dir, f"greyscale_env{i}.png")
            mpimg.imsave(grey_path, obs_grey[i, :, :, 0], cmap="gray")

            # Save greyscale as 3 channels for easy side-by-side comparison
            grey_3ch = np.repeat(obs_grey[i], 3, axis=-1)
            grey_3ch_path = os.path.join(out_dir, f"greyscale_3ch_env{i}.png")
            mpimg.imsave(grey_3ch_path, grey_3ch)

            print(f"\n  Saved: {os.path.abspath(rgb_path)}")
            print(f"  Saved: {os.path.abspath(grey_path)}")
            print(f"  Saved: {os.path.abspath(grey_3ch_path)}")

        env_raw.close()
        env_grey.close()

    def test_hsl_vec_snapshot(self):
        """Save original RGB and HSL-converted frames from Vec env as PNGs."""
        out_dir = os.path.join(IMAGES_DIR, "hsl_vec")
        os.makedirs(out_dir, exist_ok=True)

        seeds = [42, 43]

        # Unwrapped vec env for original RGB
        env_raw = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        obs_raw, _ = env_raw.reset(seed=seeds)
        obs_raw, _, _, _, _ = env_raw.step([3, 3])

        # Wrapped vec env for HSL
        env_hsl = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        env_hsl = HSLObservationVec(env_hsl)
        obs_hsl, _ = env_hsl.reset(seed=seeds)
        obs_hsl, _, _, _, _ = env_hsl.step([3, 3])

        for i in range(2):
            # Save original RGB
            rgb_path = os.path.join(out_dir, f"original_env{i}.png")
            mpimg.imsave(rgb_path, obs_raw[i])

            # Save each HSL channel
            h_path = os.path.join(out_dir, f"hsl_hue_env{i}.png")
            mpimg.imsave(h_path, obs_hsl[i, :, :, 0], cmap="hsv")

            s_path = os.path.join(out_dir, f"hsl_saturation_env{i}.png")
            mpimg.imsave(s_path, obs_hsl[i, :, :, 1], cmap="gray")

            l_path = os.path.join(out_dir, f"hsl_lightness_env{i}.png")
            mpimg.imsave(l_path, obs_hsl[i, :, :, 2], cmap="gray")

            # Save combined
            hsl_path = os.path.join(out_dir, f"hsl_combined_env{i}.png")
            mpimg.imsave(hsl_path, obs_hsl[i])

            print(f"\n  Saved: {os.path.abspath(rgb_path)}")
            print(f"  Saved: {os.path.abspath(h_path)}")
            print(f"  Saved: {os.path.abspath(s_path)}")
            print(f"  Saved: {os.path.abspath(l_path)}")
            print(f"  Saved: {os.path.abspath(hsl_path)}")

        env_raw.close()
        env_hsl.close()


# =============================================================================
# Antialiasing wrapper tests
# =============================================================================

# --- GaussianAntialias scalar tests ---


@pytest.fixture
def gauss_env():
    """Scalar CarRacing env wrapped with GaussianAntialiasObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = GaussianAntialiasObservation(env)
    return env


@pytest.fixture
def gauss_hsl_env():
    """Scalar CarRacing env: GaussianAntialias -> HSLObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = GaussianAntialiasObservation(env)
    env = HSLObservation(env)
    return env


class TestGaussianAntialiasObservation:
    def test_reset_returns_correct_shape(self, gauss_env):
        obs, info = gauss_env.reset(seed=42)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, gauss_env):
        gauss_env.reset(seed=42)
        obs, reward, terminated, truncated, info = gauss_env.step(3)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_observation_space(self, gauss_env):
        assert gauss_env.observation_space.shape == (96, 96, 3)
        assert gauss_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, gauss_env):
        gauss_env.reset(seed=42)
        obs, _, _, _, _ = gauss_env.step(3)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_non_zero_after_reset(self, gauss_env):
        obs, _ = gauss_env.reset(seed=42)
        assert not np.all(obs == 0)

    def test_smoothing_effect(self, gauss_env):
        """Antialiased frame should have lower high-frequency variance than raw."""
        env_raw = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        raw, _ = env_raw.reset(seed=42)
        raw, _, _, _, _ = env_raw.step(3)
        aa, _ = gauss_env.reset(seed=42)
        aa, _, _, _, _ = gauss_env.step(3)
        raw_var = np.var(raw.astype(np.float64))
        aa_var = np.var(aa.astype(np.float64))
        assert aa_var <= raw_var
        env_raw.close()

    def test_with_hsl(self, gauss_hsl_env):
        obs, _ = gauss_hsl_env.reset(seed=42)
        assert obs.shape == (96, 96, 3)
        obs, _, _, _, _ = gauss_hsl_env.step(3)
        assert obs.shape == (96, 96, 3)
        gauss_hsl_env.close()


# --- GaussianAntialias vec tests ---


@pytest.fixture
def gauss_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with GaussianAntialiasObservationVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = GaussianAntialiasObservationVec(env)
    return env


class TestGaussianAntialiasObservationVec:
    def test_reset_returns_correct_shape(self, gauss_vec_env):
        obs, info = gauss_vec_env.reset(seed=[42, 43])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, gauss_vec_env):
        gauss_vec_env.reset(seed=[42, 43])
        obs, reward, terminated, truncated, info = gauss_vec_env.step([3, 3])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert reward.shape == (NUM_ENVS,)
        assert terminated.shape == (NUM_ENVS,)
        assert truncated.shape == (NUM_ENVS,)

    def test_observation_space(self, gauss_vec_env):
        assert gauss_vec_env.observation_space.shape == (NUM_ENVS, 96, 96, 3)
        assert gauss_vec_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, gauss_vec_env):
        gauss_vec_env.reset(seed=[42, 43])
        obs, _, _, _, _ = gauss_vec_env.step([3, 3])
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_multiple_steps(self, gauss_vec_env):
        gauss_vec_env.reset(seed=[42, 43])
        for _ in range(3):
            obs, _, _, _, _ = gauss_vec_env.step([3, 3])
            assert obs.shape == (NUM_ENVS, 96, 96, 3)
        gauss_vec_env.close()


# --- EdgeAntialias scalar tests ---


@pytest.fixture
def edge_env():
    """Scalar CarRacing env wrapped with EdgeAntialiasObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = EdgeAntialiasObservation(env)
    return env


@pytest.fixture
def edge_hsl_env():
    """Scalar CarRacing env: EdgeAntialias -> HSLObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = EdgeAntialiasObservation(env)
    env = HSLObservation(env)
    return env


class TestEdgeAntialiasObservation:
    def test_reset_returns_correct_shape(self, edge_env):
        obs, info = edge_env.reset(seed=42)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, edge_env):
        edge_env.reset(seed=42)
        obs, reward, terminated, truncated, info = edge_env.step(3)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_observation_space(self, edge_env):
        assert edge_env.observation_space.shape == (96, 96, 3)
        assert edge_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, edge_env):
        edge_env.reset(seed=42)
        obs, _, _, _, _ = edge_env.step(3)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_non_zero_after_reset(self, edge_env):
        obs, _ = edge_env.reset(seed=42)
        assert not np.all(obs == 0)

    def test_preserves_non_edge_pixels(self, edge_env):
        """With high threshold, most pixels should be unchanged."""
        env_high_thresh = gym.make(
            "CarRacing-v3", continuous=False, render_mode="rgb_array"
        )
        env_high_thresh = EdgeAntialiasObservation(
            env_high_thresh, edge_threshold=0.5, strength=0.5
        )
        raw_env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        raw, _ = raw_env.reset(seed=42)
        raw, _, _, _, _ = raw_env.step(3)
        aa, _ = env_high_thresh.reset(seed=42)
        aa, _, _, _, _ = env_high_thresh.step(3)
        match_ratio = np.mean(raw == aa)
        assert match_ratio > 0.5
        raw_env.close()
        env_high_thresh.close()

    def test_with_hsl(self, edge_hsl_env):
        obs, _ = edge_hsl_env.reset(seed=42)
        assert obs.shape == (96, 96, 3)
        obs, _, _, _, _ = edge_hsl_env.step(3)
        assert obs.shape == (96, 96, 3)
        edge_hsl_env.close()


# --- EdgeAntialias vec tests ---


@pytest.fixture
def edge_vec_env():
    """Vectorized CarRacing env (2 envs) wrapped with EdgeAntialiasObservationVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = EdgeAntialiasObservationVec(env)
    return env


class TestEdgeAntialiasObservationVec:
    def test_reset_returns_correct_shape(self, edge_vec_env):
        obs, info = edge_vec_env.reset(seed=[42, 43])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape(self, edge_vec_env):
        edge_vec_env.reset(seed=[42, 43])
        obs, reward, terminated, truncated, info = edge_vec_env.step([3, 3])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert reward.shape == (NUM_ENVS,)
        assert terminated.shape == (NUM_ENVS,)
        assert truncated.shape == (NUM_ENVS,)

    def test_observation_space(self, edge_vec_env):
        assert edge_vec_env.observation_space.shape == (NUM_ENVS, 96, 96, 3)
        assert edge_vec_env.observation_space.dtype == np.uint8

    def test_values_in_range(self, edge_vec_env):
        edge_vec_env.reset(seed=[42, 43])
        obs, _, _, _, _ = edge_vec_env.step([3, 3])
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_multiple_steps(self, edge_vec_env):
        edge_vec_env.reset(seed=[42, 43])
        for _ in range(3):
            obs, _, _, _, _ = edge_vec_env.step([3, 3])
            assert obs.shape == (NUM_ENVS, 96, 96, 3)
        edge_vec_env.close()


# --- Visual snapshot tests (antialiasing) ---
# Run with: venv/bin/python -m pytest tests/test_wrappers.py::TestAntialiasVisualSnapshots -v -s


def _warmup_env(env, seed):
    """Step through environment to let camera settle into position."""
    for _ in range(4):
        env.step(3)
    for _ in range(2):
        env.step(0)


class TestAntialiasVisualSnapshots:
    def test_gaussian_antialias_snapshot(self):
        """Save original RGB and Gaussian-antialiased frame as PNGs."""
        out_dir = os.path.join(IMAGES_DIR, "gaussian_antialias")
        os.makedirs(out_dir, exist_ok=True)

        env_raw = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env_raw.reset(seed=42)
        _warmup_env(env_raw, 42)
        raw, _, _, _, _ = env_raw.step(3)

        env_aa = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env_aa = GaussianAntialiasObservation(env_aa)
        env_aa.reset(seed=42)
        _warmup_env(env_aa, 42)
        aa, _, _, _, _ = env_aa.step(3)

        raw_path = os.path.join(out_dir, "original_rgb.png")
        mpimg.imsave(raw_path, raw)
        aa_path = os.path.join(out_dir, "gaussian_antialiased.png")
        mpimg.imsave(aa_path, aa)

        side_by_side = np.concatenate([raw, aa], axis=1)
        sb_path = os.path.join(out_dir, "comparison_side_by_side.png")
        mpimg.imsave(sb_path, side_by_side)

        print(f"\n  Saved: {os.path.abspath(raw_path)}")
        print(f"  Saved: {os.path.abspath(aa_path)}")
        print(f"  Saved: {os.path.abspath(sb_path)}")

        env_raw.close()
        env_aa.close()

    def test_edge_antialias_snapshot(self):
        """Save original RGB and edge-aware-antialiased frame as PNGs."""
        out_dir = os.path.join(IMAGES_DIR, "edge_antialias")
        os.makedirs(out_dir, exist_ok=True)

        env_raw = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env_raw.reset(seed=42)
        _warmup_env(env_raw, 42)
        raw, _, _, _, _ = env_raw.step(3)

        env_aa = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env_aa = EdgeAntialiasObservation(env_aa)
        env_aa.reset(seed=42)
        _warmup_env(env_aa, 42)
        aa, _, _, _, _ = env_aa.step(3)

        raw_path = os.path.join(out_dir, "original_rgb.png")
        mpimg.imsave(raw_path, raw)
        aa_path = os.path.join(out_dir, "edge_antialiased.png")
        mpimg.imsave(aa_path, aa)

        side_by_side = np.concatenate([raw, aa], axis=1)
        sb_path = os.path.join(out_dir, "comparison_side_by_side.png")
        mpimg.imsave(sb_path, side_by_side)

        print(f"\n  Saved: {os.path.abspath(raw_path)}")
        print(f"  Saved: {os.path.abspath(aa_path)}")
        print(f"  Saved: {os.path.abspath(sb_path)}")

        env_raw.close()
        env_aa.close()

    def test_gaussian_vec_snapshot(self):
        """Save Gaussian-antialiased frames from Vec env."""
        out_dir = os.path.join(IMAGES_DIR, "gaussian_antialias_vec")
        os.makedirs(out_dir, exist_ok=True)

        env_raw = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        env_raw.reset(seed=[42, 43])
        for _ in range(4):
            env_raw.step([3, 3])
        for _ in range(2):
            env_raw.step([0, 0])
        raw, _, _, _, _ = env_raw.step([3, 3])

        env_aa = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        env_aa = GaussianAntialiasObservationVec(env_aa)
        env_aa.reset(seed=[42, 43])
        for _ in range(4):
            env_aa.step([3, 3])
        for _ in range(2):
            env_aa.step([0, 0])
        aa, _, _, _, _ = env_aa.step([3, 3])

        for i in range(2):
            raw_path = os.path.join(out_dir, f"original_env{i}.png")
            mpimg.imsave(raw_path, raw[i])
            aa_path = os.path.join(out_dir, f"gaussian_env{i}.png")
            mpimg.imsave(aa_path, aa[i])
            side_by_side = np.concatenate([raw[i], aa[i]], axis=1)
            sb_path = os.path.join(out_dir, f"comparison_env{i}.png")
            mpimg.imsave(sb_path, side_by_side)

            print(f"\n  Saved: {os.path.abspath(raw_path)}")
            print(f"  Saved: {os.path.abspath(aa_path)}")
            print(f"  Saved: {os.path.abspath(sb_path)}")

        env_raw.close()
        env_aa.close()

    def test_edge_vec_snapshot(self):
        """Save edge-aware-antialiased frames from Vec env."""
        out_dir = os.path.join(IMAGES_DIR, "edge_antialias_vec")
        os.makedirs(out_dir, exist_ok=True)

        env_raw = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        env_raw.reset(seed=[42, 43])
        for _ in range(4):
            env_raw.step([3, 3])
        for _ in range(2):
            env_raw.step([0, 0])
        raw, _, _, _, _ = env_raw.step([3, 3])

        env_aa = gym.make_vec(
            "CarRacing-v3",
            num_envs=2,
            vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False,
            render_mode="rgb_array",
        )
        env_aa = EdgeAntialiasObservationVec(env_aa)
        env_aa.reset(seed=[42, 43])
        for _ in range(4):
            env_aa.step([3, 3])
        for _ in range(2):
            env_aa.step([0, 0])
        aa, _, _, _, _ = env_aa.step([3, 3])

        for i in range(2):
            raw_path = os.path.join(out_dir, f"original_env{i}.png")
            mpimg.imsave(raw_path, raw[i])
            aa_path = os.path.join(out_dir, f"edge_env{i}.png")
            mpimg.imsave(aa_path, aa[i])
            side_by_side = np.concatenate([raw[i], aa[i]], axis=1)
            sb_path = os.path.join(out_dir, f"comparison_env{i}.png")
            mpimg.imsave(sb_path, side_by_side)

            print(f"\n  Saved: {os.path.abspath(raw_path)}")
            print(f"  Saved: {os.path.abspath(aa_path)}")
            print(f"  Saved: {os.path.abspath(sb_path)}")

        env_raw.close()
        env_aa.close()


# =============================================================================
# Optical flow wrapper tests
# =============================================================================

# --- OpticalFlow scalar tests ---


@pytest.fixture
def optical_flow_hsl_env():
    """Scalar CarRacing env: HSLObservation -> SkipFrame -> OpticalFlowObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = HSLObservation(env)
    env = SkipFrame(env, skip=SKIP, channels=3)
    env = OpticalFlowObservation(env, skip=SKIP, channels=3)
    return env


@pytest.fixture
def optical_flow_grey_env():
    """Scalar CarRacing env: GreyscaleObservation -> SkipFrame -> OpticalFlowObservation."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = GreyscaleObservation(env)
    env = SkipFrame(env, skip=SKIP, channels=1)
    env = OpticalFlowObservation(env, skip=SKIP, channels=1)
    return env


class TestOpticalFlowObservation:
    def test_reset_returns_correct_shape_hsl(self, optical_flow_hsl_env):
        obs, info = optical_flow_hsl_env.reset(seed=42)
        # 3 HSL channels + 2 flow channels = 5
        assert obs.shape == (96, 96, 5)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape_hsl(self, optical_flow_hsl_env):
        optical_flow_hsl_env.reset(seed=42)
        obs, reward, terminated, truncated, info = optical_flow_hsl_env.step(3)
        assert obs.shape == (96, 96, 5)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_reset_returns_correct_shape_grey(self, optical_flow_grey_env):
        obs, info = optical_flow_grey_env.reset(seed=42)
        # 1 greyscale channel + 2 flow channels = 3
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape_grey(self, optical_flow_grey_env):
        optical_flow_grey_env.reset(seed=42)
        obs, reward, terminated, truncated, info = optical_flow_grey_env.step(3)
        assert obs.shape == (96, 96, 3)
        assert obs.dtype == np.uint8

    def test_observation_space_hsl(self, optical_flow_hsl_env):
        assert optical_flow_hsl_env.observation_space.shape == (96, 96, 5)
        assert optical_flow_hsl_env.observation_space.dtype == np.uint8

    def test_observation_space_grey(self, optical_flow_grey_env):
        assert optical_flow_grey_env.observation_space.shape == (96, 96, 3)
        assert optical_flow_grey_env.observation_space.dtype == np.uint8

    def test_values_in_range_hsl(self, optical_flow_hsl_env):
        optical_flow_hsl_env.reset(seed=42)
        obs, _, _, _, _ = optical_flow_hsl_env.step(3)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_values_in_range_grey(self, optical_flow_grey_env):
        optical_flow_grey_env.reset(seed=42)
        obs, _, _, _, _ = optical_flow_grey_env.step(3)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_non_zero_after_reset_hsl(self, optical_flow_hsl_env):
        obs, _ = optical_flow_hsl_env.reset(seed=42)
        assert not np.all(obs == 0)

    def test_non_zero_after_reset_grey(self, optical_flow_grey_env):
        obs, _ = optical_flow_grey_env.reset(seed=42)
        assert not np.all(obs == 0)

    def test_multiple_steps_hsl(self, optical_flow_hsl_env):
        optical_flow_hsl_env.reset(seed=42)
        for _ in range(3):
            obs, _, _, _, _ = optical_flow_hsl_env.step(3)
            assert obs.shape == (96, 96, 5)
        optical_flow_hsl_env.close()

    def test_multiple_steps_grey(self, optical_flow_grey_env):
        optical_flow_grey_env.reset(seed=42)
        for _ in range(3):
            obs, _, _, _, _ = optical_flow_grey_env.step(3)
            assert obs.shape == (96, 96, 3)
        optical_flow_grey_env.close()


# --- OpticalFlow vec tests ---


@pytest.fixture
def optical_flow_hsl_vec_env():
    """Vectorized CarRacing env (2 envs): HSLObservationVec -> SkipFrameVec -> OpticalFlowObservationVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = HSLObservationVec(env)
    env = SkipFrameVec(env, skip=SKIP, channels=3)
    env = OpticalFlowObservationVec(env, skip=SKIP, channels=3)
    return env


@pytest.fixture
def optical_flow_grey_vec_env():
    """Vectorized CarRacing env (2 envs): GreyscaleObservationVec -> SkipFrameVec -> OpticalFlowObservationVec."""
    env = gym.make_vec(
        "CarRacing-v3",
        num_envs=NUM_ENVS,
        vectorization_mode=gym.VectorizeMode.ASYNC,
        continuous=False,
        render_mode="rgb_array",
    )
    env = GreyscaleObservationVec(env)
    env = SkipFrameVec(env, skip=SKIP, channels=1)
    env = OpticalFlowObservationVec(env, skip=SKIP, channels=1)
    return env


class TestOpticalFlowObservationVec:
    def test_reset_returns_correct_shape_hsl(self, optical_flow_hsl_vec_env):
        obs, info = optical_flow_hsl_vec_env.reset(seed=[42, 43])
        assert obs.shape == (NUM_ENVS, 96, 96, 5)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape_hsl(self, optical_flow_hsl_vec_env):
        optical_flow_hsl_vec_env.reset(seed=[42, 43])
        obs, reward, terminated, truncated, info = optical_flow_hsl_vec_env.step([3, 3])
        assert obs.shape == (NUM_ENVS, 96, 96, 5)
        assert reward.shape == (NUM_ENVS,)
        assert terminated.shape == (NUM_ENVS,)
        assert truncated.shape == (NUM_ENVS,)

    def test_reset_returns_correct_shape_grey(self, optical_flow_grey_vec_env):
        obs, info = optical_flow_grey_vec_env.reset(seed=[42, 43])
        assert obs.shape == (NUM_ENVS, 96, 96, 3)
        assert obs.dtype == np.uint8

    def test_step_returns_correct_shape_grey(self, optical_flow_grey_vec_env):
        optical_flow_grey_vec_env.reset(seed=[42, 43])
        obs, reward, terminated, truncated, info = optical_flow_grey_vec_env.step(
            [3, 3]
        )
        assert obs.shape == (NUM_ENVS, 96, 96, 3)

    def test_values_in_range_hsl(self, optical_flow_hsl_vec_env):
        optical_flow_hsl_vec_env.reset(seed=[42, 43])
        obs, _, _, _, _ = optical_flow_hsl_vec_env.step([3, 3])
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_values_in_range_grey(self, optical_flow_grey_vec_env):
        optical_flow_grey_vec_env.reset(seed=[42, 43])
        obs, _, _, _, _ = optical_flow_grey_vec_env.step([3, 3])
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_multiple_steps_hsl(self, optical_flow_hsl_vec_env):
        optical_flow_hsl_vec_env.reset(seed=[42, 43])
        for _ in range(3):
            obs, _, _, _, _ = optical_flow_hsl_vec_env.step([3, 3])
            assert obs.shape == (NUM_ENVS, 96, 96, 5)
        optical_flow_hsl_vec_env.close()

    def test_multiple_steps_grey(self, optical_flow_grey_vec_env):
        optical_flow_grey_vec_env.reset(seed=[42, 43])
        for _ in range(3):
            obs, _, _, _, _ = optical_flow_grey_vec_env.step([3, 3])
            assert obs.shape == (NUM_ENVS, 96, 96, 3)
        optical_flow_grey_vec_env.close()


# --- OpticalFlow visual snapshot tests ---
# Run with: venv/bin/python -m pytest tests/test_wrappers.py::TestOpticalFlowVisualSnapshots -v -s


class TestOpticalFlowVisualSnapshots:
    @staticmethod
    def _flow_to_hsv(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Convert dx/dy flow to HSV color-coded RGB image.

        Hue = flow direction, Saturation = 1, Value = flow magnitude.
        This is the standard optical flow visualization.
        """
        # Convert from uint8 (0-255) back to signed range
        dx_signed = (dx.astype(np.float32) - 127.5) / 127.5
        dy_signed = (dy.astype(np.float32) - 127.5) / 127.5

        mag, ang = cv2.cartToPolar(dx_signed, dy_signed)

        hsv = np.zeros((dx.shape[0], dx.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: direction
        hsv[..., 1] = 255  # Saturation: full
        hsv[..., 2] = cv2.normalize(
            mag, None, 0, 255, cv2.NORM_MINMAX
        )  # Value: magnitude

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def test_optical_flow_hsl_snapshot(self):
        """Save HSL + optical flow channels as individual PNGs and HSV visualization."""
        out_dir = os.path.join(IMAGES_DIR, "optical_flow_hsl")
        os.makedirs(out_dir, exist_ok=True)

        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env = HSLObservation(env)
        env = SkipFrame(env, skip=SKIP, channels=3)
        env = OpticalFlowObservation(env, skip=SKIP, channels=3)

        env.reset(seed=42)
        # Warm up
        for _ in range(4):
            env.step(3)
        for _ in range(2):
            env.step(0)
        obs, _, _, _, _ = env.step(3)

        # Save individual channels
        h_path = os.path.join(out_dir, "hsl_h.png")
        mpimg.imsave(h_path, obs[:, :, 0], cmap="gray")

        s_path = os.path.join(out_dir, "hsl_s.png")
        mpimg.imsave(s_path, obs[:, :, 1], cmap="gray")

        l_path = os.path.join(out_dir, "hsl_l.png")
        mpimg.imsave(l_path, obs[:, :, 2], cmap="gray")

        dx_path = os.path.join(out_dir, "flow_dx.png")
        mpimg.imsave(dx_path, obs[:, :, 3], cmap="gray")

        dy_path = os.path.join(out_dir, "flow_dy.png")
        mpimg.imsave(dy_path, obs[:, :, 4], cmap="gray")

        # Save HSV color-coded flow visualization
        hsv_flow = self._flow_to_hsv(obs[:, :, 3], obs[:, :, 4])
        hsv_path = os.path.join(out_dir, "flow_hsv.png")
        mpimg.imsave(hsv_path, hsv_flow)

        # Save side-by-side: L channel + HSV flow
        l_3ch = np.repeat(obs[:, :, 2:3], 3, axis=-1)
        side_by_side = np.concatenate([l_3ch, hsv_flow], axis=1)
        sb_path = os.path.join(out_dir, "lightness_vs_flow.png")
        mpimg.imsave(sb_path, side_by_side)

        print(f"\n  Saved: {os.path.abspath(h_path)}")
        print(f"  Saved: {os.path.abspath(s_path)}")
        print(f"  Saved: {os.path.abspath(l_path)}")
        print(f"  Saved: {os.path.abspath(dx_path)}")
        print(f"  Saved: {os.path.abspath(dy_path)}")
        print(f"  Saved: {os.path.abspath(hsv_path)}")
        print(f"  Saved: {os.path.abspath(sb_path)}")

        env.close()

    def test_optical_flow_grey_snapshot(self):
        """Save greyscale + optical flow channels as individual PNGs and HSV visualization."""
        out_dir = os.path.join(IMAGES_DIR, "optical_flow_grey")
        os.makedirs(out_dir, exist_ok=True)

        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env = GreyscaleObservation(env)
        env = SkipFrame(env, skip=SKIP, channels=1)
        env = OpticalFlowObservation(env, skip=SKIP, channels=1)

        env.reset(seed=42)
        for _ in range(4):
            env.step(3)
        for _ in range(2):
            env.step(0)
        obs, _, _, _, _ = env.step(3)

        grey_path = os.path.join(out_dir, "greyscale.png")
        mpimg.imsave(grey_path, obs[:, :, 0], cmap="gray")

        dx_path = os.path.join(out_dir, "flow_dx.png")
        mpimg.imsave(dx_path, obs[:, :, 1], cmap="gray")

        dy_path = os.path.join(out_dir, "flow_dy.png")
        mpimg.imsave(dy_path, obs[:, :, 2], cmap="gray")

        # Save HSV color-coded flow visualization
        hsv_flow = self._flow_to_hsv(obs[:, :, 1], obs[:, :, 2])
        hsv_path = os.path.join(out_dir, "flow_hsv.png")
        mpimg.imsave(hsv_path, hsv_flow)

        # Save side-by-side: greyscale + HSV flow
        grey_3ch = np.repeat(obs[:, :, 0:1], 3, axis=-1)
        side_by_side = np.concatenate([grey_3ch, hsv_flow], axis=1)
        sb_path = os.path.join(out_dir, "greyscale_vs_flow.png")
        mpimg.imsave(sb_path, side_by_side)

        print(f"\n  Saved: {os.path.abspath(grey_path)}")
        print(f"  Saved: {os.path.abspath(dx_path)}")
        print(f"  Saved: {os.path.abspath(dy_path)}")
        print(f"  Saved: {os.path.abspath(hsv_path)}")
        print(f"  Saved: {os.path.abspath(sb_path)}")

        env.close()

    def test_optical_flow_sequence(self):
        """Save flow across multiple steps to see motion over time."""
        out_dir = os.path.join(IMAGES_DIR, "optical_flow_sequence")
        os.makedirs(out_dir, exist_ok=True)

        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env = GreyscaleObservation(env)
        env = SkipFrame(env, skip=SKIP, channels=1)
        env = OpticalFlowObservation(env, skip=SKIP, channels=1)

        env.reset(seed=42)
        # Warm up with gas
        for _ in range(4):
            env.step(3)

        # Capture 6 frames of driving
        for step in range(6):
            obs, _, _, _, _ = env.step(3)  # gas

            grey_3ch = np.repeat(obs[:, :, 0:1], 3, axis=-1)
            hsv_flow = self._flow_to_hsv(obs[:, :, 1], obs[:, :, 2])
            combined = np.concatenate([grey_3ch, hsv_flow], axis=1)

            path = os.path.join(out_dir, f"step_{step}.png")
            mpimg.imsave(path, combined)
            print(f"  Saved: {os.path.abspath(path)}")

        env.close()
