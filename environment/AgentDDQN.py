from __future__ import annotations

import numpy as np

import torch
from .AgentDQN import AgentDQN


class AgentDDQN(AgentDQN):
    def __init__(
        self,
        state_space_shape,
        action_n,
        model,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9999925,
        lr: float = 0.0002,
        lr_decay: float = 1.0,
        n_step: int = 1,
        buffer_size: int = 300000,
        skip_frames: int = 4,
        play_n_episodes: int = 3000,
        **kwargs,  # Catch-all
    ):
        super().__init__(
            state_space_shape,
            action_n,
            model,
            gamma,
            epsilon,
            epsilon_end,
            epsilon_decay,
            lr,
            lr_decay,
            n_step,
            buffer_size,
            skip_frames,
            **kwargs,
        )

    def update_net(self, batch_size: int):
        """
        Updates the Q-network using a batch of transitions.

        Parameters:
            batch_size (int) : The number of transitions to use for training
            the Q-network.

        Returns:
            td_est (torch.Tensor) : The temporal difference estimates for
            the sampled batch.

            loss (torch.Tensor) : The computed loss for the batch.
        """
        self.n_updates += 1
        states, actions, rewards, new_states, terminateds, info = self.get_samples(
            batch_size
        )
        if states == None:
            return 0.0, 0.0

        action_values = self.target_net(states)
        td_est = action_values[np.arange(batch_size), actions]
        with torch.no_grad():
            next_actions = torch.argmax(self.target_net(new_states), axis=1)
            tar_action_values = self.policy_net(new_states)
        td_tar = (
            rewards
            + (1 - terminateds.float())
            * self.gamma**self.n_step
            * tar_action_values[np.arange(batch_size), next_actions]
        )

        loss = self.loss_fn(td_est, td_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.buffer.update_priority(info["index"], loss)
        self.optimizer.step()
        self.scheduler.step()
        loss = loss.detach().cpu().item()

        return td_est, loss
