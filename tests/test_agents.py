import pytest
import os
import csv
import numpy as np
import torch

from enviroment.Agent import Agent
from enviroment.AgentDDQN import AgentDDQN
from enviroment.AgentPPO import AgentPPO
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO
from tests.helpers import DEVICES, make_dqn_agent, make_ddqn_agent, make_ppo_agent


ACTION_N = 5


# ======================================================================
# Parametrized tests for Agent (DQN) and AgentDDQN
# ======================================================================

AGENT_CLASSES = [Agent, AgentDDQN]
AGENT_IDS = ["DQN", "DDQN"]
AGENT_MAKERS = {"DQN": make_dqn_agent, "DDQN": make_ddqn_agent}


# --- Tests using device-parametrized fixtures (auto GPU if available) ---


class TestAgentInit:
    def test_creates_nets(self, dqn_agent):
        """Parametrized by device via dqn_agent fixture."""
        assert dqn_agent.policy_net is not None
        assert dqn_agent.target_net is not None

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_creates_optimizer(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=100,
            device="cpu",
        )
        assert isinstance(agent.optimizer, torch.optim.Adam)

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_creates_buffer(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=100,
            device="cpu",
        )
        assert agent.buffer is not None

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_default_hyperparams(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=100,
            device="cpu",
        )
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.05
        assert agent.action_n == ACTION_N

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_custom_hyperparams(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            gamma=0.99,
            epsilon=0.5,
            epsilon_end=0.01,
            lr=0.01,
            buffer_size=100,
            device="cpu",
        )
        assert agent.gamma == 0.99
        assert agent.epsilon == 0.5
        assert agent.epsilon_end == 0.01


class TestAgentStore:
    def test_store_increases_buffer(self, dqn_agent, fill_agent_buffer):
        """Parametrized by device via dqn_agent fixture."""
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        new_state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        dqn_agent.store(state, 0, 1.0, new_state, False)
        fill_agent_buffer(dqn_agent, n=10)
        states, actions, rewards, new_states, terminateds = dqn_agent.get_samples(5)
        assert states.shape[0] == 5

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_store_with_tensor_state(self, agent_cls, fill_agent_buffer):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=200,
            device="cpu",
        )
        state = torch.randint(0, 256, (96, 96, 12), dtype=torch.uint8)
        new_state = torch.randint(0, 256, (96, 96, 12), dtype=torch.uint8)
        agent.store(state, 2, -1.0, new_state, True)
        fill_agent_buffer(agent, n=10)
        states, actions, rewards, new_states, terminateds = agent.get_samples(5)
        assert states.shape[0] == 5


class TestAgentGetSamples:
    def test_sample_shapes(self, dqn_agent, fill_agent_buffer):
        """Parametrized by device via dqn_agent fixture."""
        fill_agent_buffer(dqn_agent, n=20)
        batch_size = 8
        states, actions, rewards, new_states, terminateds = dqn_agent.get_samples(
            batch_size
        )
        assert states.shape == (batch_size, 96, 96, 12)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert new_states.shape == (batch_size, 96, 96, 12)
        assert terminateds.shape == (batch_size,)

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_sample_dtypes(self, agent_cls, fill_agent_buffer):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=200,
            device="cpu",
        )
        fill_agent_buffer(agent, n=20)
        states, actions, rewards, new_states, terminateds = agent.get_samples(4)
        assert states.dtype == torch.uint8
        assert actions.dtype == torch.long
        assert rewards.dtype == torch.float32
        assert terminateds.dtype == torch.bool


class TestAgentTakeAction:
    def test_returns_valid_int(self, dqn_agent):
        """Parametrized by device via dqn_agent fixture."""
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        action = dqn_agent.take_action(state)
        assert isinstance(action, int)
        assert 0 <= action < ACTION_N

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_accepts_tensor_state(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=100,
            device="cpu",
        )
        state = torch.randint(0, 256, (96, 96, 12), dtype=torch.uint8)
        action = agent.take_action(state)
        assert 0 <= action < ACTION_N

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_epsilon_decays(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            epsilon=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9,
            buffer_size=100,
            device="cpu",
        )
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        initial_epsilon = agent.epsilon
        agent.take_action(state)
        assert agent.epsilon < initial_epsilon
        assert agent.act_taken == 1

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_epsilon_floor(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            epsilon=0.06,
            epsilon_end=0.05,
            epsilon_decay=0.9,
            buffer_size=100,
            device="cpu",
        )
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        for _ in range(20):
            agent.take_action(state)
        assert agent.epsilon >= agent.epsilon_end

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_greedy_with_zero_epsilon(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            epsilon=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_size=100,
            device="cpu",
        )
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        action1 = agent.take_action(state)
        action2 = agent.take_action(state)
        assert action1 == action2


class TestAgentUpdateNet:
    @pytest.mark.parametrize("agent_maker", ["DQN", "DDQN"], ids=AGENT_IDS)
    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    def test_returns_td_est_and_loss(self, agent_maker, fill_agent_buffer, device):
        """Parametrized by both agent class and device."""
        agent = AGENT_MAKERS[agent_maker](device=device)
        fill_agent_buffer(agent, n=20)
        td_est, loss = agent.update_net(batch_size=8)
        assert td_est is not None
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_increments_n_updates(self, agent_cls, fill_agent_buffer):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=200,
            device="cpu",
        )
        fill_agent_buffer(agent, n=20)
        assert agent.n_updates == 0
        agent.update_net(batch_size=8)
        assert agent.n_updates == 1
        agent.update_net(batch_size=8)
        assert agent.n_updates == 2


class TestAgentSaveLoad:
    def test_save_creates_file(self, dqn_agent, tmp_save_dir):
        """Parametrized by device via dqn_agent fixture."""
        dqn_agent.save(tmp_save_dir, "test_model")
        files = os.listdir(tmp_save_dir)
        assert any("test_model" in f for f in files)
        assert any(f.endswith(".pt") for f in files)

    @pytest.mark.parametrize("agent_maker", ["DQN", "DDQN"], ids=AGENT_IDS)
    @pytest.mark.parametrize("device", DEVICES, ids=lambda d: d)
    def test_save_load_round_trip(self, agent_maker, tmp_save_dir, device):
        """Parametrized by both agent class and device."""
        agent = AGENT_MAKERS[agent_maker](device=device)
        agent.epsilon = 0.5
        agent.epsilon_decay = 0.99

        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        for _ in range(5):
            agent.take_action(state)
        saved_epsilon = agent.epsilon
        saved_act_taken = agent.act_taken

        agent.save(tmp_save_dir, "test_roundtrip")

        agent2 = AGENT_MAKERS[agent_maker](device=device)
        agent2.load_state = "train"
        saved_file = [f for f in os.listdir(tmp_save_dir) if "test_roundtrip" in f][0]
        agent2.load(tmp_save_dir, saved_file)

        assert agent2.epsilon == saved_epsilon
        assert agent2.act_taken == saved_act_taken


class TestAgentWriteLog:
    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_creates_csv(self, agent_cls, tmp_save_dir):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=100,
            device="cpu",
        )
        agent.LOG_DIR = tmp_save_dir + "/"
        dates = ["2024-01-01", "2024-01-02"]
        times = ["10:00:00", "10:01:00"]
        rewards = [100.0, 200.0]
        lengths = [50, 60]
        losses = [0.5, 0.3]
        epsilons = [0.9, 0.8]
        lrs = [0.001, 0.001]

        agent.write_log(
            dates, times, rewards, lengths, losses, epsilons, lrs,
            log_filename="test_log.csv",
        )

        log_path = os.path.join(tmp_save_dir, "test_log.csv")
        assert os.path.exists(log_path)

        with open(log_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 7
        assert rows[0][0] == "date"
        assert rows[2][0] == "reward"
        assert rows[2][1:] == ["100.0", "200.0"]

    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_with_fuel_efficiency(self, agent_cls, tmp_save_dir):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            buffer_size=100,
            device="cpu",
        )
        agent.LOG_DIR = tmp_save_dir + "/"
        dates = ["2024-01-01"]
        times = ["10:00:00"]
        rewards = [100.0]
        lengths = [50]
        losses = [0.5]
        epsilons = [0.9]
        lrs = [0.001]
        fuel = [12.5]

        agent.write_log(
            dates, times, rewards, lengths, losses, epsilons, lrs,
            fuel_efficiency_list=fuel, log_filename="test_fuel.csv",
        )

        log_path = os.path.join(tmp_save_dir, "test_fuel.csv")
        with open(log_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 8
        assert rows[7][0] == "fuel_efficiency"
        assert rows[7][1] == "12.5"


class TestAgentGetLr:
    @pytest.mark.parametrize("agent_cls", AGENT_CLASSES, ids=AGENT_IDS)
    def test_returns_float(self, agent_cls):
        agent = agent_cls(
            state_space_shape=(2, 96, 96, 12),
            action_n=ACTION_N,
            model=Delamain_2_5,
            lr=0.002,
            buffer_size=100,
            device="cpu",
        )
        lr = agent.get_lr()
        assert isinstance(lr, float)
        assert lr > 0


# ======================================================================
# AgentPPO-specific tests
# All use ppo_agent fixture which is parametrized by device in conftest.
# ======================================================================


class TestPPOInit:
    def test_ppo_flags(self, ppo_agent):
        assert ppo_agent.is_ppo is True
        assert ppo_agent.eps_clip == 0.2
        assert ppo_agent.K_epochs == 4

    def test_actor_exists(self, ppo_agent):
        assert ppo_agent.actor is not None
        assert ppo_agent.policy_net is ppo_agent.actor

    def test_default_hyperparams(self, ppo_agent):
        assert ppo_agent.gamma == 0.99
        assert ppo_agent.epsilon == 0.0


class TestPPOStore:
    def test_store_appends_to_buffer(self, ppo_agent, fill_ppo_buffer):
        fill_ppo_buffer(ppo_agent, n=5)
        assert len(ppo_agent.buffer) == 5

    def test_store_manual(self, ppo_agent):
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        new_state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        ppo_agent.store(state, 3, 1.0, new_state, False, log_prob=torch.tensor(-0.5))
        assert len(ppo_agent.buffer) == 1


class TestPPOTakeAction:
    def test_scalar_returns_tuple(self, ppo_agent):
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        action_out = ppo_agent.take_action(state)
        action, log_prob = action_out
        assert isinstance(action, int)
        assert isinstance(log_prob, torch.Tensor)
        assert 0 <= action < ACTION_N
        assert ppo_agent.act_taken == 1

    def test_eval_mode_deterministic(self, ppo_agent):
        ppo_agent.load_state = "eval"
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        action1, _ = ppo_agent.take_action(state)
        action2, _ = ppo_agent.take_action(state)
        assert action1 == action2
        ppo_agent.load_state = "train"

    def test_accepts_tensor(self, ppo_agent):
        state = torch.randint(0, 256, (96, 96, 12), dtype=torch.uint8)
        action, log_prob = ppo_agent.take_action(state)
        assert isinstance(action, int)
        assert 0 <= action < ACTION_N


class TestPPOUpdateNet:
    def test_empty_buffer(self, ppo_agent):
        q, loss = ppo_agent.update_net()
        assert q is None
        assert loss.item() == 0.0

    def test_returns_finite_loss(self, ppo_agent, fill_ppo_buffer):
        fill_ppo_buffer(ppo_agent, n=16)
        q, loss = ppo_agent.update_net()
        assert q is None
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_clears_buffer(self, ppo_agent, fill_ppo_buffer):
        fill_ppo_buffer(ppo_agent, n=16)
        ppo_agent.update_net()
        assert len(ppo_agent.buffer) == 0

    def test_increments_n_updates(self, ppo_agent, fill_ppo_buffer):
        fill_ppo_buffer(ppo_agent, n=16)
        assert ppo_agent.n_updates == 0
        ppo_agent.update_net()
        assert ppo_agent.n_updates == 1


class TestPPOSaveLoad:
    def test_save_creates_file(self, ppo_agent, tmp_save_dir):
        ppo_agent.save(tmp_save_dir, "test_ppo")
        files = os.listdir(tmp_save_dir)
        assert any("test_ppo" in f for f in files)
        assert any(f.endswith(".pt") for f in files)

    def test_save_load_round_trip(self, ppo_agent, tmp_save_dir):
        state = np.random.randint(0, 256, (96, 96, 12), dtype=np.uint8)
        for _ in range(5):
            ppo_agent.take_action(state)
        saved_act_taken = ppo_agent.act_taken

        ppo_agent.save(tmp_save_dir, "test_ppo_rt")

        agent2 = make_ppo_agent(device=ppo_agent.device)
        agent2.load_state = "train"
        saved_file = [f for f in os.listdir(tmp_save_dir) if "test_ppo_rt" in f][0]
        agent2.load(tmp_save_dir, saved_file)

        assert agent2.act_taken == saved_act_taken


class TestPPOLog:
    def test_creates_csv(self, ppo_agent, tmp_save_dir):
        ppo_agent.LOG_DIR = tmp_save_dir + "/"
        dates = ["2024-01-01"]
        times = ["10:00:00"]
        rewards = [100.0]
        lengths = [50]
        losses = [0.5]
        epsilons = [0.0]
        lrs = [0.0003]

        ppo_agent.write_log(
            dates, times, rewards, lengths, losses, epsilons, lrs,
            log_filename="ppo_log.csv",
        )
        log_path = os.path.join(tmp_save_dir, "ppo_log.csv")
        assert os.path.exists(log_path)

    def test_get_lr(self, ppo_agent):
        lr = ppo_agent.get_lr()
        assert isinstance(lr, float)
        assert lr > 0
