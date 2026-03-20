import pytest
import numpy as np
import torch

from tests.helpers import DEVICES, ACTION_N, STATE_SHAPE_D21
from tests.helpers import make_dqn_agent, make_ddqn_agent, make_ppo_agent


# --- Fixtures ---


@pytest.fixture(params=DEVICES, ids=lambda d: d)
def device(request):
    return request.param


@pytest.fixture
def action_n():
    return ACTION_N


@pytest.fixture
def tmp_save_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_state_raw():
    """(batch=2, 96, 96, 3) uint8 tensor — input for Delamain."""
    return torch.randint(0, 256, (2, 96, 96, 3), dtype=torch.uint8)


@pytest.fixture
def sample_state_d2():
    """(batch=2, 96, 96, 6) uint8 tensor — input for Delamain_2."""
    return torch.randint(0, 256, (2, 96, 96, 6), dtype=torch.uint8)


@pytest.fixture
def sample_state_d21():
    """(batch=2, 96, 96, 12) uint8 tensor — input for Delamain_2_1."""
    return torch.randint(0, 256, (2, 96, 96, 12), dtype=torch.uint8)


@pytest.fixture
def sample_state():
    """(batch=2, 96, 96, 12) uint8 tensor — input for Delamain_2_5."""
    return torch.randint(0, 256, (2, 96, 96, 12), dtype=torch.uint8)


# --- Agent fixtures (parametrized by device) ---


@pytest.fixture(params=DEVICES, ids=lambda d: d)
def dqn_agent(request):
    return make_dqn_agent(device=request.param)


@pytest.fixture(params=DEVICES, ids=lambda d: d)
def ddqn_agent(request):
    return make_ddqn_agent(device=request.param)


@pytest.fixture(params=DEVICES, ids=lambda d: d)
def ppo_agent(request):
    return make_ppo_agent(device=request.param)


# --- Buffer helpers ---


def _fill_agent_buffer(agent, n=8):
    """Store n random transitions into DQN/DDQN agent's buffer."""
    for _ in range(n):
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        new_state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        agent.store(state, np.random.randint(0, ACTION_N), 1.0, new_state, False)


def _fill_ppo_buffer(agent, n=8):
    """Store n transitions into PPO agent's buffer."""
    for _ in range(n):
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        new_state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        agent.store(state, 3, 1.0, new_state, False, log_prob=torch.tensor(-0.5))


@pytest.fixture
def fill_agent_buffer():
    return _fill_agent_buffer


@pytest.fixture
def fill_ppo_buffer():
    return _fill_ppo_buffer
