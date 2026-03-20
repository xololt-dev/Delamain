import pytest
import gymnasium as gym
import numpy as np

from enviroment.SkipFrame import SkipFrame
from enviroment.SkipFrameVec import SkipFrameVec


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
