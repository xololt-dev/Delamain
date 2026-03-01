import numpy as np
from typing import Union
import os
import csv

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
from tensordict import TensorDict

class AgentPPO():
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
        buffer_size: int = 4096, # Typically larger for PPO rollouts
        **kwargs # Catch-all for DQN kwargs passed by TrainingGround that PPO ignores
    ):
        self.gamma = gamma
        self.action_n = action_n
        self.state_space_shape = state_space_shape
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vec = kwargs.get("vec", False)
        
        # PPO specific hyperparameters
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.is_ppo = True # Flag for TrainingGround logic

        # Dummy variables to prevent TrainingGround logging/eval from crashing
        self.epsilon = 0.0
        self.epsilon_end = 0.0
        self.act_taken = 0
        self.n_updates = 0
        self.load_state = 'train'
        self.save_dir = './training/saved_models/'

        # Replay Buffer
        self.buffer = []
            
        self.bufferLoss = TensorDictReplayBuffer(storage=LazyTensorStorage(250, device=self.device))

        # Networks
        self.actor = model().to(device=self.device, non_blocking=True)
        self.critic_base = model().to(device=self.device, non_blocking=True)
        self.critic_head = nn.Linear(action_n, 1).to(device=self.device, non_blocking=True)
        
        # Alias actor to policy_net so TrainingGround's eval mode toggle doesn't break
        self.policy_net = self.actor

        # Optimizers (Combining Actor and Critic parameters)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic_base.parameters(), 'lr': lr},
            {'params': self.critic_head.parameters(), 'lr': lr}
        ], eps=1e-7)
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: lr_decay)
        self.loss_fn = nn.MSELoss()

    def store(self, state, action, reward, new_state, terminated, log_prob):
        """Stores a transition sequentially in the rollout buffer."""
        if isinstance(state, np.ndarray):
            state_t = torch.as_tensor(state, dtype=torch.float32)
        else:
            state_t = state.clone().detach().to(dtype=torch.float32)

        self.buffer.append((
            state_t,
            torch.as_tensor(action, dtype=torch.long),
            torch.as_tensor(log_prob, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool)
        ))

    def take_action(self, state: Union[np.ndarray, torch.Tensor]):
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
            logits = self.actor(state)
            dist = Categorical(logits=logits)
            
            # Deterministic evaluation if in eval mode, otherwise sample
            if self.load_state == 'eval':
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        self.act_taken += 1
        return action.item(), log_prob.detach()

    def take_action_vec(self, state: torch.Tensor):
        with torch.no_grad():
            logits = self.actor(state)
            dist = Categorical(logits=logits)
            
            # Deterministic evaluation if in eval mode, otherwise sample
            if self.load_state == 'eval':
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
            logits = self.actor(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Critic evaluation
            state_values = self.critic_head(self.critic_base(states)).squeeze()

            # Find ratios (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Find Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values, returns)
            
            # Total Loss (includes entropy bonus for exploration)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            final_loss = loss.detach()

        self.scheduler.step()
        # self.buffer.empty() # Clear rollout buffer after update
        self.buffer.clear() # Clear rollout buffer after update

        # Save loss for TrainingGround compatibility
        self.bufferLoss.add(
            TensorDict({"loss": final_loss.clone().to(device=self.device, non_blocking=True)}, batch_size=[])
        )
        return None, final_loss

    # def update_net_vec(self, batch_size: int = None):
    #     """
    #     Updates the Actor and Critic networks using collected vectorized rollouts.
    #     """
    #     self.n_updates += 1
        
    #     if len(self.buffer) == 0:
    #         return None, torch.tensor(0.0)
            
    #     # Stack into shape [T, n, ...] where T is steps per update and n is num envs
    #     states = torch.stack([x[0] for x in self.buffer]).to(self.device)
    #     actions = torch.stack([x[1] for x in self.buffer]).to(self.device)
    #     old_log_probs = torch.stack([x[2] for x in self.buffer]).to(self.device)
    #     rewards = torch.stack([x[3] for x in self.buffer]).to(self.device)
    #     terminateds = torch.stack([x[4] for x in self.buffer]).to(self.device)

    #     T, n = rewards.shape[0], rewards.shape[1]

    #     # Calculate Monte Carlo estimates of rewards per-environment
    #     returns = torch.zeros_like(rewards).to(self.device)
    #     discounted_reward = torch.zeros(n, dtype=torch.float32, device=self.device)
        
    #     for t in reversed(range(T)):
    #         # ~terminateds[t] is True (1.0) if the env didn't crash, False (0.0) if it did.
    #         # This cleanly severs the reward connection only for environments that ended.
    #         discounted_reward = discounted_reward * (~terminateds[t]).float()
    #         discounted_reward = rewards[t] + (self.gamma * discounted_reward)
    #         returns[t] = discounted_reward
            
    #     # Flatten the Time (T) and Environment (n) dimensions into a single batch dimension
    #     # [T, n, ...] becomes [T * n, ...]
    #     states = states.view(T * n, *states.shape[2:])
    #     actions = actions.view(T * n)
    #     old_log_probs = old_log_probs.view(T * n)
    #     returns = returns.view(T * n)
        
    #     # Normalize returns across all environments and timesteps
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    #     final_loss = None

    #     # Optimize policy for K epochs over the entire flattened batch
    #     for _ in range(self.K_epochs):
    #         logits = self.actor(states)
    #         dist = Categorical(logits=logits)
    #         log_probs = dist.log_prob(actions)
    #         entropy = dist.entropy()

    #         state_values = self.critic_head(self.critic_base(states)).squeeze()

    #         ratios = torch.exp(log_probs - old_log_probs)
    #         advantages = returns - state_values.detach()
            
    #         surr1 = ratios * advantages
    #         surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

    #         actor_loss = -torch.min(surr1, surr2).mean()
    #         critic_loss = self.loss_fn(state_values, returns)
            
    #         loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         final_loss = loss.detach()

    #     self.scheduler.step()
    #     self.buffer.clear() # Clear rollout buffer after update

    #     # Log the loss
    #     self.bufferLoss.add(
    #         TensorDict({"loss": final_loss.clone().to(device=self.device, non_blocking=True)}, batch_size=[])
    #     )
    #     return None, final_loss
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
                logits = self.actor(mb_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                state_values = self.critic_head(self.critic_base(mb_states)).squeeze()

                ratios = torch.exp(log_probs - mb_old_log_probs)
                advantages = mb_returns - state_values.detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.loss_fn(state_values, mb_returns)
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                final_loss = loss.detach()

        self.scheduler.step()
        self.buffer.clear() # Clear rollout buffer after update

        # Log the loss (using the final mini-batch loss of the final epoch)
        self.bufferLoss.add(
            TensorDict({"loss": final_loss.clone().to(device=self.device, non_blocking=True)}, batch_size=[])
        )
        return None, final_loss

    def save(self, save_dir: str, save_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name + f"_PPO_{self.act_taken}.pt")

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_base_state_dict': self.critic_base.state_dict(),
            'critic_head_state_dict': self.critic_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'action_number': self.act_taken,
        }, save_path)
        print(f"PPO Model saved to {save_path} at step {self.act_taken}")

    def load(self, load_dir: str, model_name: str):
        save_path = os.path.join(load_dir, model_name)
        loaded_model = torch.load(save_path)
        
        self.actor.load_state_dict(loaded_model['actor_state_dict'])
        self.critic_base.load_state_dict(loaded_model['critic_base_state_dict'])
        self.critic_head.load_state_dict(loaded_model['critic_head_state_dict'])
        self.optimizer.load_state_dict(loaded_model['optimizer_state_dict'])
        if 'scheduler_state_dict' in loaded_model:
            self.scheduler.load_state_dict(loaded_model['scheduler_state_dict'])

        if self.load_state == 'eval' or self.load_state == 'kernel_vis':
            self.actor.eval()
            self.critic_base.eval()
            self.critic_head.eval()
        elif self.load_state in ['train', 'fine_tune']:
            self.actor.train()
            self.critic_base.train()
            self.critic_head.train()
            self.act_taken = loaded_model['action_number']
        
        print(f"PPO Model {model_name} from {load_dir} loaded")

    def write_log(self, date_list, time_list, reward_list, length_list, loss_list, epsilon_list, lr_list, log_filename='default_log.csv'):
        # Matches original Agent exactly
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        rows = [['date']+date_list, ['time']+time_list, ['reward']+reward_list,
                ['length']+length_list, ['loss']+loss_list, ['epsilon']+epsilon_list, ['lr']+lr_list]
        with open(self.LOG_DIR+log_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
    
    def get_lr(self):
        return self.scheduler.get_last_lr()[0]
