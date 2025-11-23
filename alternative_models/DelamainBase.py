import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DelamainBase(nn.Module):
    def __init__(self):
        super().__init__()

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
    
    def is_prev_frame_needed(self) -> bool:
        return False