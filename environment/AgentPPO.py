from __future__ import annotations

import numpy as np
import os
import csv

import torch
import torch.nn as nn
from torch.distributions import Categorical
from .Agent import Agent


class AgentPPO(Agent):
    SAVE_DIR = "training/saved_models/"
    LOG_DIR = "training/logs/"

    def __init__(
        self,
        state_space_shape,
        action_n,
        model,
        gamma: float = 0.99,
        lr: float = 0.0003,
        lr_decay: float = 1.0,
        buffer_size: int = 4096,  # Typically larger for PPO rollouts
        **kwargs,  # Catch-all for DQN kwargs passed by TrainingGround that PPO ignores
    ):
        super().__init__(state_space_shape, action_n, gamma, **kwargs)

        self.vec = kwargs.get("vec", False)

        # PPO specific hyperparameters
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.entropy_coeff = 0.01
        self.critic_coeff = 0.5

        # Dummy variables to prevent TrainingGround logging/eval from crashing
        self.epsilon = 0.0
        self.epsilon_end = 0.0

        # Replay Buffer
        self.buffer = []

        self.actor = model().to(device=self.device, non_blocking=True)
        self.actor.compile()
        # Alias actor to policy_net so TrainingGround's eval mode toggle doesn't break
        self.policy_net = self.actor
        self.target_net = self.actor

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-7)
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay
        )
        self.loss_fn = nn.MSELoss()

    def store(self, state, action, reward, new_state, terminated, log_prob):
        """Stores a transition sequentially in the rollout buffer."""
        if isinstance(state, np.ndarray):
            state_t = torch.as_tensor(state, dtype=torch.float32)
        else:
            state_t = state.clone().detach().to(dtype=torch.float32)

        self.buffer.append(
            (
                state_t,
                torch.as_tensor(action, dtype=torch.long),
                torch.as_tensor(log_prob, dtype=torch.float32),
                torch.as_tensor(reward, dtype=torch.float32),
                torch.as_tensor(terminated, dtype=torch.bool),
            )
        )

    def take_action(self, state: np.ndarray | torch.Tensor):
        """Chooses an action based on the actor's probability distribution."""
        if isinstance(state, np.ndarray):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(dtype=torch.float32, device=self.device)

        if self.vec:
            return self.take_action_vec(state_t)
        else:
            return self.take_action_scalar(state_t)

    def take_action_scalar(self, state: torch.Tensor):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits, state_values = self.actor(state)
            dist = Categorical(logits=logits)

            # Deterministic evaluation if in eval mode, otherwise sample
            if self.load_state == "eval":
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        self.act_taken += 1
        return action.item(), log_prob.detach()

    def take_action_vec(self, state: torch.Tensor):
        with torch.no_grad():
            if self.target_net.action_mode_switch():
                self.target_net.eval()
            logits, state_values = self.actor(state)
            if self.load_state == "train" and self.target_net.action_mode_switch():
                self.target_net.train()
            dist = Categorical(logits=logits)

            # Deterministic evaluation if in eval mode, otherwise sample
            if self.load_state == "eval":
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        self.act_taken += 1
        return action.tolist(), log_prob.detach()

    def update_net(self, batch_size: int = None):
        self.n_updates += 1

        if len(self.buffer) == 0:
            return None, torch.tensor(0.0)

        if self.vec:
            # return self.update_net_vec_whole(batch_size)
            # return self.update_net_vec_gae(batch_size)
            return self.update_net_vec(batch_size)
        else:
            return self.update_net_scalar(batch_size)

    def update_net_scalar(self, batch_size: int = None):
        """Updates the Actor and Critic networks using the collected sequential rollout."""

        # Unpack the sequential buffer
        states = torch.stack([x[0] for x in self.buffer]).to(self.device)
        actions = torch.stack([x[1] for x in self.buffer]).to(self.device)
        old_log_probs = torch.stack([x[2] for x in self.buffer]).to(self.device)
        rewards = torch.stack([x[3] for x in self.buffer]).to(self.device)
        terminateds = torch.stack([x[4] for x in self.buffer]).to(self.device)

        # Calculate Monte Carlo estimates of rewards sequentially (Backwards)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(terminateds)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        final_loss = None

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate current actions
            logits, state_values = self.actor(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Critic evaluation
            state_values = state_values.squeeze()

            # Find ratios (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Find Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values, returns)

            # Total Loss (includes entropy bonus for exploration)
            loss = (
                actor_loss
                + self.critic_coeff * critic_loss
                - self.entropy_coeff * entropy.mean()
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if final_loss == None:
                final_loss = loss.detach().view(1)
            else:
                final_loss = torch.cat((final_loss, loss.detach().view(1)))

        self.scheduler.step()
        self.buffer.clear()  # Clear rollout buffer after update
        final_loss = final_loss.mean().cpu()

        return None, final_loss

    def update_net_vec_whole(self, batch_size: int = None):
        """
        Updates the Actor and Critic networks using collected vectorized rollouts.
        """
        self.n_updates += 1

        if len(self.buffer) == 0:
            return None, torch.tensor(0.0)

        # Stack into shape [T, n, ...] where T is steps per update and n is num envs
        states = torch.stack([x[0] for x in self.buffer]).to(self.device)
        actions = torch.stack([x[1] for x in self.buffer]).to(self.device)
        old_log_probs = torch.stack([x[2] for x in self.buffer]).to(self.device)
        rewards = torch.stack([x[3] for x in self.buffer]).to(self.device)
        terminateds = torch.stack([x[4] for x in self.buffer]).to(self.device)

        T, n = rewards.shape[0], rewards.shape[1]

        # Calculate Monte Carlo estimates of rewards per-environment
        returns = torch.zeros_like(rewards).to(self.device)
        discounted_reward = torch.zeros(n, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            # ~terminateds[t] is True (1.0) if the env didn't crash, False (0.0) if it did.
            # This cleanly severs the reward connection only for environments that ended.
            discounted_reward = discounted_reward * (~terminateds[t]).float()
            discounted_reward = rewards[t] + (self.gamma * discounted_reward)
            returns[t] = discounted_reward

        # Flatten the Time (T) and Environment (n) dimensions into a single batch dimension
        # [T, n, ...] becomes [T * n, ...]
        states = states.view(T * n, *states.shape[2:])
        actions = actions.view(T * n)
        old_log_probs = old_log_probs.view(T * n)
        returns = returns.view(T * n)

        # Normalize returns across all environments and timesteps
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        final_loss = None

        # Optimize policy for K epochs over the entire flattened batch
        for _ in range(self.K_epochs):
            logits, state_values = self.actor(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # state_values = self.critic_head(self.critic_base(states)).squeeze()
            state_values = state_values.squeeze()

            ratios = torch.exp(log_probs - old_log_probs)
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values, returns)

            loss = (
                actor_loss
                + self.critic_coeff * critic_loss
                - self.entropy_coeff * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if final_loss == None:
                final_loss = loss.detach().view(1)
            else:
                final_loss = torch.cat((final_loss, loss.detach().view(1)))

        self.scheduler.step()
        self.buffer.clear()  # Clear rollout buffer after update
        final_loss = final_loss.mean().cpu()

        return None, final_loss

    def update_net_vec(self, batch_size: int = 64):
        """
        Updates the Actor and Critic networks using collected vectorized rollouts.
        """
        self.n_updates += 1

        if len(self.buffer) == 0:
            return None, torch.tensor(0.0)

        # Stack into shape [T, n, ...] where T is steps per update and n is num envs
        states = torch.stack([x[0] for x in self.buffer]).to(self.device)
        actions = torch.stack([x[1] for x in self.buffer]).to(self.device)
        old_log_probs = torch.stack([x[2] for x in self.buffer]).to(self.device)
        rewards = torch.stack([x[3] for x in self.buffer]).to(self.device)
        terminateds = torch.stack([x[4] for x in self.buffer]).to(self.device)

        T, n = rewards.shape[0], rewards.shape[1]

        # Calculate Monte Carlo estimates of rewards per-environment
        returns = torch.zeros_like(rewards).to(self.device)
        discounted_reward = torch.zeros(n, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            # ~terminateds[t] is True (1.0) if the env didn't crash, False (0.0) if it did.
            discounted_reward = discounted_reward * (~terminateds[t]).float()
            discounted_reward = rewards[t] + (self.gamma * discounted_reward)
            returns[t] = discounted_reward

        # Flatten the Time (T) and Environment (n) dimensions into a single batch dimension
        states = states.view(T * n, *states.shape[2:])
        actions = actions.view(T * n)
        old_log_probs = old_log_probs.view(T * n)
        returns = returns.view(T * n)

        # Normalize returns across all environments and timesteps
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        final_loss = None
        total_batch_size = T * n

        # Fallback if batch_size isn't passed or is somehow larger than the rollout
        if batch_size is None or batch_size > total_batch_size:
            batch_size = 64

        # Optimize policy for K epochs over the flattened batch using MINI-BATCHES
        for _ in range(self.K_epochs):
            # Shuffle indices for this epoch
            indices = torch.randperm(total_batch_size).to(self.device)

            for start_idx in range(0, total_batch_size, batch_size):
                end_idx = start_idx + batch_size
                mb_indices = indices[start_idx:end_idx]

                # Extract mini-batch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]

                # Run network on MINI-BATCH only
                logits, state_values = self.actor(mb_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # state_values = self.critic_head(self.critic_base(mb_states)).squeeze()
                state_values = state_values.squeeze()

                ratios = torch.exp(log_probs - mb_old_log_probs)
                advantages = mb_returns - state_values.detach()

                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.loss_fn(state_values, mb_returns)

                loss = (
                    actor_loss
                    + self.critic_coeff * critic_loss
                    - self.entropy_coeff * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if final_loss == None:
                    final_loss = loss.detach().view(1)
                else:
                    final_loss = torch.cat((final_loss, loss.detach().view(1)))

                self.scheduler.step()
        self.buffer.clear()  # Clear rollout buffer after update
        final_loss = final_loss.mean().cpu()

        return None, final_loss

    def update_net_vec_gae(self, batch_size: int = 64):
        """
        Updates the Actor and Critic networks using collected vectorized rollouts.
        """
        self.n_updates += 1

        if len(self.buffer) == 0:
            return None, torch.tensor(0.0)

        # Stack into shape [T, n, ...] where T is steps per update and n is num envs
        states = torch.stack([x[0] for x in self.buffer]).to(self.device)
        actions = torch.stack([x[1] for x in self.buffer]).to(self.device)
        old_log_probs = torch.stack([x[2] for x in self.buffer]).to(self.device)
        rewards = torch.stack([x[3] for x in self.buffer]).to(self.device)
        terminateds = torch.stack([x[4] for x in self.buffer]).to(self.device)

        T, n = rewards.shape[0], rewards.shape[1]

        # --- THE GOLD STANDARD PPO ADVANTAGE CALCULATION ---

        # 1. Get Critic values for the entire sequence first
        with torch.no_grad():
            logits, values = self.actor(states)
            values = values.squeeze()

        # 2. Setup GAE tracking tensors
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = torch.zeros(n, dtype=torch.float32, device=self.device)

        # Usually defined in __init__, lambda controls the variance/bias tradeoff (0.95 is standard)
        gae_lambda = 0.95

        for t in reversed(range(T)):
            if t == T - 1:
                # BOOTSTRAPPING: At the very end of the rollout, if the env didn't crash,
                # we don't have the "next" state. A common approximation is to just
                # use the current state's value as a guess for the future.
                next_nonterminal = (~terminateds[t]).float()
                next_values = values[t]  # Bootstrapping happens here!
            else:
                next_nonterminal = (~terminateds[t]).float()
                next_values = values[t + 1]

            # Calculate the Temporal Difference Error (Delta)
            delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]

            # Calculate GAE
            advantages[t] = last_gae_lam = (
                delta + self.gamma * gae_lambda * next_nonterminal * last_gae_lam
            )

        # The target return for the Critic to learn from is simply Advantage + Value
        returns = advantages + values

        # Flatten tensors as before
        states = states.view(T * n, *states.shape[2:])
        actions = actions.view(T * n)
        old_log_probs = old_log_probs.view(T * n)
        returns = returns.view(T * n)
        advantages = advantages.view(T * n)  # We also flatten advantages

        # In standard PPO, we normalize ADVANTAGES, not returns
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        final_loss = None
        total_batch_size = T * n

        # Fallback if batch_size isn't passed or is somehow larger than the rollout
        if batch_size is None or batch_size > total_batch_size:
            batch_size = 64

        # Optimize policy for K epochs over the flattened batch using MINI-BATCHES
        for _ in range(self.K_epochs):
            # Shuffle indices for this epoch
            indices = torch.randperm(total_batch_size).to(self.device)

            for start_idx in range(0, total_batch_size, batch_size):
                end_idx = start_idx + batch_size
                mb_indices = indices[start_idx:end_idx]

                # Extract mini-batch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Run network on MINI-BATCH only
                logits, state_values = self.actor(mb_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # state_values = self.critic_head(self.critic_base(mb_states)).squeeze()
                state_values = state_values.squeeze()

                ratios = torch.exp(log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mb_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.loss_fn(state_values, mb_returns)

                loss = (
                    actor_loss
                    + self.critic_coeff * critic_loss
                    - self.entropy_coeff * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if final_loss == None:
                    final_loss = loss.detach().view(1)
                else:
                    final_loss = torch.cat((final_loss, loss.detach().view(1)))

        self.scheduler.step()
        self.buffer.clear()  # Clear rollout buffer after update
        final_loss = final_loss.mean().cpu()

        return None, final_loss

    def save(self, save_dir: str, save_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name + f"_PPO_{self.act_taken}.pt")

        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "action_number": self.act_taken,
            },
            save_path,
        )
        print(f"PPO Model saved to {save_path} at step {self.act_taken}")

    def load(self, load_dir: str, model_name: str):
        save_path = os.path.join(load_dir, model_name)
        loaded_model = torch.load(
            save_path, map_location=self.device, weights_only=False
        )

        self.actor.load_state_dict(loaded_model["actor_state_dict"])
        self.optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
        if "scheduler_state_dict" in loaded_model:
            self.scheduler.load_state_dict(loaded_model["scheduler_state_dict"])

        if self.load_state == "eval" or self.load_state == "kernel_vis":
            self.actor.eval()
        elif self.load_state in ["train", "fine_tune"]:
            self.actor.train()
            self.act_taken = loaded_model["action_number"]

        print(f"PPO Model {model_name} from {load_dir} loaded")

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]
