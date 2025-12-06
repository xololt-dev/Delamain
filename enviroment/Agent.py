import numpy as np
from typing import Union
import os
import csv
import time

import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage
from tensordict import TensorDict

class Agent():
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
    buffer_size: int = 300000,
    skip_frames: int = 4,
    play_n_episodes: int = 3000,
  ):
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_end = epsilon_end
    self.epsilon_decay = epsilon_decay
    self.action_n = action_n
    self.state_space_shape = state_space_shape
    self.skip_frames = skip_frames

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.buffer_size = buffer_size
    if self.device == 'cuda':
        self.buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(
                    buffer_size,
                    device=self.device))
    else:
        self.buffer = TensorDictReplayBuffer(
                storage=LazyMemmapStorage(
                    buffer_size,
                    device=self.device))
    self.bufferLoss = TensorDictReplayBuffer(
                storage=LazyTensorStorage(
                    250,
                    device=self.device))
    # if self.device == 'cuda':
    #     self.buffer.append_transform(lambda x: x.to(self.device))

    torch.set_float32_matmul_precision('high')
    self.policy_net = model().to(device=self.device, non_blocking=True)
    self.policy_net.compile()
    self.target_net = model().to(device=self.device, non_blocking=True)
    self.target_net.compile()

    self.optimizer = torch.optim.Adam(self.target_net.parameters(), lr, eps=1e-7)
    self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: lr_decay)
    # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, pct_start=0.1, max_lr=lr*2, steps_per_epoch=int(1000/self.skip_frames), epochs=play_n_episodes)
    self.loss_fn = torch.nn.SmoothL1Loss()

    self.act_taken = 0
    self.n_updates = 0

    self.load_state = 'train'

    self.save_dir = './training/saved_models/'
    self.log_dir = './training/logs/'

  def store(
    self,
    state: Union[np.ndarray, torch.Tensor],
    action: int,
    reward: float,
    new_state: Union[np.ndarray, torch.Tensor],
    terminated: bool
  ):
      """
      Stores a transition in the replay buffer.

      Parameters:
          state (numpy.ndarray | torch.Tensor) : The current state of
          the environment.

          action (int) : The action taken by the agent in the current state.

          reward (float) : The reward received after taking the action.

          new_state (numpy.ndarray | torch.Tensor) : The next state of
          the environment after the action.

          terminated (bool) : A boolean indicating whether the episode has ended.
      """
    #   self.buffer.add(
    #       TensorDict(
    #           {
    #               "state": torch.as_tensor(state),
    #             #   "state": state.detach().clone(),
    #             #   "state": state.detach().clone() if isinstance(state, torch.Tensor) else torch.tensor(state),
    #               "action": torch.tensor(action),
    #               "reward": torch.tensor(reward),
    #               "new_state": torch.as_tensor(new_state),
    #             #   "new_state": new_state.detach().clone(),
    #             #   "new_state": new_state.detach().clone() if isinstance(new_state, torch.Tensor) else torch.tensor(state),
    #               "terminated": torch.tensor(terminated)
    #           },
    #           batch_size=[]
    #       )
    #   )
      self.buffer.add(
        TensorDict(
            {
                "state": torch.as_tensor(state, dtype=torch.uint8),
                "action": torch.as_tensor(action, dtype=torch.uint8),
                "reward": torch.as_tensor(reward, dtype=torch.float32),
                "new_state": torch.as_tensor(new_state, dtype=torch.uint8),
                "terminated": torch.as_tensor(terminated, dtype=torch.bool)
            },
            batch_size=[]
        )
      )

  def get_samples(self, batch_size: int):
        """
        Samples a batch of transitions from the replay buffer.

        Parameters:
            batch_size (int) : The number of transitions to sample from
            the replay buffer.

        Returns:
            states (torch.Tensor) : A batch of sampled states.

            actions (torch.Tensor) : A batch of sampled actions.

            rewards (torch.Tensor) : A batch of sampled rewards.

            new_states (torch.Tensor) : A batch of sampled next states.

            terminateds (torch.Tensor) : A batch of sampled termination flags.
        """
        batch = self.buffer.sample(batch_size)
        states = batch.get('state').to(dtype=torch.float32, device=self.device)
        # states = batch.get('state').type(torch.Tensor).to(self.device)
        new_states = batch.get('new_state').to(dtype=torch.float32, device=self.device)
        # new_states = batch.get('new_state').type(torch.Tensor).to(self.device)
        actions = batch.get('action').squeeze().to(dtype=torch.long, device=self.device)
        rewards = batch.get('reward').squeeze().to(dtype=torch.float32, device=self.device)
        terminateds = batch.get('terminated').squeeze().to(self.device)
        return states, actions, rewards, new_states, terminateds

  def take_action(self, state: Union[np.ndarray, torch.Tensor]):
        """
        Chooses an action based on the epsilon-greedy policy.

        Parameters:
            state (numpy.ndarray | torch.Tensor) : The current state of
            the environment.

        Returns:
            action_idx (torch.Tensor) : The action chosen by the agent.
        """
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_n)
        else:
            # with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            else:
                state_t = state.to(dtype=torch.float32, device=self.device)

            # Add batch dimension
            state_t = state_t.unsqueeze(0)

            action_values = self.target_net(state_t)
            action_idx = torch.argmax(action_values, axis=1).item()
            # state = (state
            # # .detach().clone()
            # .to(dtype=torch.float, device=self.device) 
            #     if isinstance(state, torch.Tensor) else 
            #         torch.as_tensor(
            #         state,
            #         dtype=torch.float,
            #         device=self.device
            #     )).unsqueeze(0)
            # action_values = self.target_net(state)
            # action_idx = torch.argmax(action_values, axis=1).item()
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_end
        self.act_taken += 1
        return action_idx

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
        states, actions, rewards, \
            new_states, terminateds = self.get_samples(batch_size)
        # print('states:', states.shape, 'actions: ', actions)
        action_values = self.target_net(states)
        td_est = action_values[np.arange(batch_size), actions]
        with torch.no_grad():
            tar_action_values = self.policy_net(new_states)
        td_tar = rewards + (1 - terminateds.float()) * self.gamma*tar_action_values.max(1)[0]

        loss = self.loss_fn(td_est, td_tar)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.target_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        # t2 = time.perf_counter(), time.process_time()
        # print("Get steps")
        # print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
        # print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
        # print()
        # loss = loss.detach().clone().cpu().item()

        self.bufferLoss.add(
            TensorDict(
                {
                    "loss": loss.detach().clone().to(device=self.device, non_blocking=True),
                },
                batch_size=[]
            )
        )

        # loss = loss.item()
        # t2 = time.perf_counter(), time.process_time()
        # print("Get loss")
        # print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
        # print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
        # print()
        return td_est, loss

  def save(self, save_dir: str, save_name: str):
        """
        Saves the model, optimizer state, replay buffer, and other parameters.

        Parameters:
            save_dir (str) : The directory where the model should be saved.

            save_name (str) : The name of the file to save the model as.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name + f"_{self.act_taken}.pt")
        # buffer_save_path = os.path.join(save_dir, f"buffer_{self.act_taken}.pt")

        # Save models and optimizer state
        torch.save(
            {
                'upd_model_state_dict': self.target_net.state_dict(),
                'frz_model_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'action_number': self.act_taken,
                'epsilon': self.epsilon
            },
            save_path
        )

        # Save replay buffer data separately
        # torch.save(self.buffer._storage.data, buffer_save_path)
        # self.buffer.dumps(buffer_save_path)

        print(f"Model saved to {save_path} at step {self.act_taken}")
        # print(f"Replay buffer saved to {buffer_save_path}")

  def load(self, load_dir: str, model_name: str):
        """
        Loads a saved model and its parameters.

        Parameters:
            load_dir (str) : The directory from which the model should be loaded.

            model_name (str) : The name of the file containing the saved model.
        """
        save_path = os.path.join(load_dir, model_name)
        # buffer_save_path = os.path.join(load_dir, f"buffer_{self.act_taken}.pt") # This needs to be fixed to load the correct buffer

        loaded_model = torch.load(save_path)
        upd_net_param = loaded_model['upd_model_state_dict']
        frz_net_param = loaded_model['frz_model_state_dict']
        opt_param = loaded_model['optimizer_state_dict']
        sched_param = loaded_model['scheduler_state_dict']
        self.target_net.load_state_dict(upd_net_param)
        self.policy_net.load_state_dict(frz_net_param)
        self.optimizer.load_state_dict(opt_param)
        if sched_param:
            self.scheduler.load_state_dict(sched_param)

        # Load replay buffer data separately
        # self.buffer._storage.data =
        # torch.load(buffer_save_path)
        # self.buffer._storage._memmap = None # Reset memmap after loading

        if self.load_state == 'eval':
            self.target_net.eval()
            self.policy_net.eval()
            self.epsilon_min = 0
            self.epsilon = 0
        elif self.load_state == 'kernel_vis':
            self.target_net.eval()
            self.policy_net.eval()
            self.epsilon_min = 0
            self.epsilon = 0
        elif self.load_state == 'train':
            self.target_net.train()
            self.policy_net.train()
            self.act_taken = loaded_model['action_number']
            self.epsilon = loaded_model['epsilon']
            # self.buffer.loads(buffer_save_path)
        elif self.load_state == 'fine_tune':
            self.target_net.train()
            self.policy_net.train()
            self.act_taken = loaded_model['action_number']
            self.epsilon = loaded_model['epsilon']
        else:
            raise ValueError(f"Unknown load state. Should be either 'eval', 'kernel_vis', 'train' or 'fine_tune'.")
        
        print(f"Model {model_name} from {load_dir} loaded")

  def write_log(
        self,
        date_list: list,
        time_list: list,
        reward_list: list,
        length_list: list,
        loss_list: list,
        epsilon_list: list,
        lr_list: list,
        log_filename: str = 'default_log.csv'
    ):
        """
        Writes training logs to a CSV file.

        Parameters:
            date_list (list) : A list of dates corresponding to the episodes.

            time_list (list) : A list of times corresponding to the episodes.

            reward_list (list) : A list of rewards obtained in each episode.

            length_list (list) : A list of episode lengths (number of steps).

            loss_list (list) : A list of losses recorded during training.

            epsilon_list (list) : A list of epsilon values (exploration rates)
            during training.

            lr_list (list) : A list of learning rate values recorded during training.

            log_filename (str) : The name of the CSV file to save the logs.
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        rows = [['date']+date_list,
                ['time']+time_list,
                ['reward']+reward_list,
                ['length']+length_list,
                ['loss']+loss_list,
                ['epsilon']+epsilon_list,
                ['lr']+lr_list]
        with open(self.log_dir+log_filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
