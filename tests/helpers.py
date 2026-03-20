"""Shared test utilities — not a conftest (pytest doesn't auto-import this)."""
import torch

from enviroment.Agent import Agent
from enviroment.AgentDDQN import AgentDDQN
from enviroment.AgentPPO import AgentPPO
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO


ACTION_N = 5
STATE_SHAPE_RAW = (2, 96, 96, 3)
STATE_SHAPE_D2 = (2, 96, 96, 6)
STATE_SHAPE_D21 = (2, 96, 96, 12)

HAS_CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])


def make_dqn_agent(device="cpu"):
    return Agent(
        state_space_shape=STATE_SHAPE_D21,
        action_n=ACTION_N,
        model=Delamain_2_5,
        gamma=0.95,
        epsilon=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        lr=0.001,
        buffer_size=200,
        device=device,
    )


def make_ddqn_agent(device="cpu"):
    return AgentDDQN(
        state_space_shape=STATE_SHAPE_D21,
        action_n=ACTION_N,
        model=Delamain_2_5,
        gamma=0.95,
        epsilon=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        lr=0.001,
        buffer_size=200,
        device=device,
    )


def make_ppo_agent(device="cpu"):
    return AgentPPO(
        state_space_shape=STATE_SHAPE_D21,
        action_n=ACTION_N,
        model=Delamain_2_5_PPO,
        gamma=0.99,
        lr=0.0003,
        buffer_size=1024,
        device=device,
    )
