import os
import pytest
import gymnasium as gym
import numpy as np
import matplotlib.image as mpimg

from enviroment.SkipFrame import SkipFrame
from enviroment.SkipFrameVec import SkipFrameVec
from enviroment.GreyscaleObservation import GreyscaleObservation
from enviroment.HSLObservation import HSLObservation
from enviroment.GreyscaleObservationVec import GreyscaleObservationVec
from enviroment.HSLObservationVec import HSLObservationVec


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
            "CarRacing-v3", num_envs=2, vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False, render_mode="rgb_array",
        )
        obs_raw, _ = env_raw.reset(seed=seeds)
        obs_raw, _, _, _, _ = env_raw.step([3, 3])

        # Wrapped vec env for greyscale
        env_grey = gym.make_vec(
            "CarRacing-v3", num_envs=2, vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False, render_mode="rgb_array",
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
            "CarRacing-v3", num_envs=2, vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False, render_mode="rgb_array",
        )
        obs_raw, _ = env_raw.reset(seed=seeds)
        obs_raw, _, _, _, _ = env_raw.step([3, 3])

        # Wrapped vec env for HSL
        env_hsl = gym.make_vec(
            "CarRacing-v3", num_envs=2, vectorization_mode=gym.VectorizeMode.ASYNC,
            continuous=False, render_mode="rgb_array",
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
