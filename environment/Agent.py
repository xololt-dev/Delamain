from __future__ import annotations

import numpy as np
import os
import csv

import torch


class Agent:
    SAVE_DIR = "training/saved_models/"
    LOG_DIR = "training/logs/"

    def __init__(
        self,
        state_space_shape,
        action_n,
        gamma: float = 0.95,
        **kwargs,  # Catch-all
    ):
        self.gamma = gamma
        self.action_n = action_n
        self.state_space_shape = state_space_shape
        self.act_taken = 0
        self.n_updates = 0

        self.load_state = "train"

        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

    def store(
        self,
        state: np.ndarray | torch.Tensor,
        action: int,
        reward: float,
        new_state: np.ndarray | torch.Tensor,
        terminated: bool,
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def take_action(self, state: np.ndarray | torch.Tensor):
        """
        Chooses an action based on the epsilon-greedy policy.

        Parameters:
            state (numpy.ndarray | torch.Tensor) : The current state of
            the environment.

        Returns:
            action_idx (torch.Tensor) : The action chosen by the agent.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def save(self, save_dir: str, save_name: str):
        """
        Saves the model, optimizer state, replay buffer, and other parameters.

        Parameters:
            save_dir (str) : The directory where the model should be saved.

            save_name (str) : The name of the file to save the model as.
        """
        raise NotImplementedError()

    def load(self, load_dir: str, model_name: str):
        """
        Loads a saved model and its parameters.

        Parameters:
            load_dir (str) : The directory from which the model should be loaded.

            model_name (str) : The name of the file containing the saved model.
        """
        raise NotImplementedError()

    def write_log(
        self,
        date_list: list,
        time_list: list,
        reward_list: list,
        length_list: list,
        loss_list: list,
        epsilon_list: list,
        lr_list: list,
        actions_in_row_list: list | None = None,
        fuel_efficiency_list: list | None = None,
        log_filename: str = "default_log.csv",
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

            actions_in_row_list (list) : A list of actions in row values recorded during training.

            fuel_efficiency_list (list) : A list of fuel efficiency values recorded during training.

            log_filename (str) : The name of the CSV file to save the logs.
        """
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        rows = [
            ["date"] + date_list,
            ["time"] + time_list,
            ["reward"] + reward_list,
            ["length"] + length_list,
            ["loss"] + loss_list,
            ["epsilon"] + epsilon_list,
            ["lr"] + lr_list,
        ]
        if actions_in_row_list is not None:
            rows.append(["actions_in_row"] + actions_in_row_list)
        if fuel_efficiency_list is not None:
            rows.append(["fuel_efficiency"] + fuel_efficiency_list)
        with open(self.LOG_DIR + log_filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)

    def get_lr(self):
        raise NotImplementedError()
