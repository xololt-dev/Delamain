import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class Delamain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 21 * 21, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 5)

    def forward(self, x):
        # Permute the dimensions to have channels first (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def plot_reward(episode_num: int, reward_list: list, n_steps: int):
        """
        Plots the reward progression over episodes.

        Parameters:
            episode_num (int) : The current episode number.

            reward_list (list) : A list of rewards obtained in all episodes so far.

            n_steps (int) : The number of steps taken so far.
        """
        plt.figure(1)
        rewards_tensor = torch.tensor(reward_list, dtype=torch.float)
        if len(rewards_tensor) >= 11:
            eval_reward = torch.clone(rewards_tensor[-10:])
            mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
            std_eval_reward = round(torch.std(eval_reward).item(), 2)
            plt.clf()
            plt.title(f'Episode #{episode_num}: {n_steps} steps, \
                      reward {mean_eval_reward}±{std_eval_reward}')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_tensor.numpy())
        if len(rewards_tensor) >= 50:
            reward_f = torch.clone(rewards_tensor[:50])
            means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.ones(49)*torch.mean(reward_f), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)