import gymnasium as gym
import yaml
import numpy as np
import torch
import datetime
import torch.nn as nn

from enviroment.Agent import Agent
from enviroment.SkipFrame import SkipFrame
from alternative_models.Delamain import Delamain

class TrainingGround():
    def __init__(self):
        yamlValues = None
        with open("training_params.yaml") as stream:
            try:
                yamlValues = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.batch_n = yamlValues['train']['batch_n'] if yamlValues['train']['batch_n'] else 32
        self.play_n_episodes = yamlValues['train']['play_n_episodes'] if yamlValues['train']['play_n_episodes'] else 3000
        self.episode_epsilon_list = []
        self.episode_reward_list = []
        self.episode_length_list = []
        self.episode_loss_list = []
        self.episode_date_list = []
        self.episode_time_list = []
        self.episode = 0
        self.timestep_n = 0
        self.when2learn = yamlValues['reporting']['when2learn'] if yamlValues['reporting']['when2learn'] else 4 # in timesteps
        self.when2sync = yamlValues['reporting']['when2sync'] if yamlValues['reporting']['when2sync'] else 5000 # in timesteps
        self.when2save = yamlValues['reporting']['when2save'] if yamlValues['reporting']['when2save'] else 50000 # in timesteps
        self.when2report = yamlValues['reporting']['when2report'] if yamlValues['reporting']['when2report'] else 5000 # in timesteps
        self.when2eval = yamlValues['reporting']['when2eval'] if yamlValues['reporting']['when2eval'] else 50000 # in timesteps
        self.when2log = yamlValues['reporting']['when2log'] if yamlValues['reporting']['when2log'] else 10 # in episodes
        self.report_type = yamlValues['reporting']['report_type']
        
        self.eval_tracks = yamlValues['eval']['tracks']

        self.env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array", domain_randomize=yamlValues['env']['random_colors'] if yamlValues['env']['random_colors'] else False)
        self.env = SkipFrame(self.env, skip=yamlValues['env']['skip_frames'] if yamlValues['env']['skip_frames'] else 4)
        if yamlValues['eval']['video']:
            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder="training/video",    # Folder to save videos
                name_prefix="eval",               # Prefix for video filenames
                episode_trigger=lambda x: x % self.when2eval == 0   # Record every episode
            )

        self.state, info = self.env.reset()
        action_n = self.env.action_space.n
        if yamlValues['env']['mode'] == 'eval':
            self.driver = Agent(self.state.shape, action_n, Delamain, buffer_size=0)
            if not yamlValues['model']['file_name']:
                raise Exception("No model file name passed in training params")

            self.driver.load_state = 'eval'
        else:
            agent_args = dict(gamma=yamlValues['train']['gamma'],
                epsilon=yamlValues['train']['epsilon'],
                epsilon_end=yamlValues['train']['epsilon_end'],
                epsilon_decay=yamlValues['train']['epsilon_decay'],
                lr=yamlValues['train']['lr'],
                buffer_size=yamlValues['train']['buffer_size'])
            self.driver = Agent(self.state.shape, 
                action_n, 
                self.parse_class_name(yamlValues['model']['class_name']),
                **{k: v for k, v in agent_args.items() if v is not None}
            )
        if yamlValues['model']['file_name']:
            self.driver.load('../training/saved_models', yamlValues['model']['file_name'])

    def parse_class_name(self, class_name: str | None) -> nn.Module:
        match class_name:
            case 'Delamain':
                return Delamain
            case _:
                return Delamain
    
    def start(self):
        if self.driver.load_state == 'eval':
            return self.eval()

        while self.episode <= self.play_n_episodes:
            self.episode += 1
            episode_reward = 0
            episode_length = 0
            updating = True
            loss_list = []
            self.episode_epsilon_list.append(self.driver.epsilon)

            while updating:
                self.timestep_n += 1
                episode_length += 1

                action = self.driver.take_action(self.state)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                self.driver.store(self.state, action, reward, new_state, terminated)
                # Move to the next state
                self.state = new_state
                updating = not (terminated or truncated)

                if self.timestep_n % self.when2sync == 0:
                    upd_net_param = self.driver.target_net.state_dict()
                    self.driver.policy_net.load_state_dict(upd_net_param)

                if self.timestep_n % self.when2save == 0:
                    self.driver.save(self.driver.save_dir, 'Delamain')

                if self.timestep_n % self.when2learn == 0:
                    q, loss = self.driver.update_net(self.batch_n)
                    loss_list.append(loss)


                if self.timestep_n % self.when2report == 0 and self.report_type == 'text':
                    print(f'Report: {self.timestep_n} timestep')
                    print(f'    episodes: {self.episode}')
                    print(f'    n_updates: {self.driver.n_updates}')
                    print(f'    epsilon: {self.driver.epsilon}')

                if self.timestep_n % self.when2eval == 0 and self.report_type == 'text':
                    rewards_tensor = torch.tensor(self.episode_reward_list,
                                                dtype=torch.float)
                    eval_reward = torch.clone(rewards_tensor[-50:])
                    mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
                    std_eval_reward = round(torch.std(eval_reward).item(), 2)

                    lengths_tensor = torch.tensor(self.episode_length_list,
                                                dtype=torch.float)
                    eval_length = torch.clone(lengths_tensor[-50:])
                    mean_eval_length = round(torch.mean(eval_length).item(), 2)
                    std_eval_length = round(torch.std(eval_length).item(), 2)

                    print(f'Evaluation: {self.timestep_n} timestep')
                    print(f'    reward {mean_eval_reward}±{std_eval_reward}')
                    print(f'    episode length {mean_eval_length}±{std_eval_length}')
                    print(f'    episodes: {self.episode}')
                    print(f'    n_updates: {self.driver.n_updates}')
                    print(f'    epsilon: {self.driver.epsilon}')
                    self.eval()

            self.state, info = self.env.reset()

            self.episode_reward_list.append(episode_reward)
            self.episode_length_list.append(episode_length)
            self.episode_loss_list.append(np.mean(loss_list))
            now_time = datetime.datetime.now()
            self.episode_date_list.append(now_time.date().strftime('%Y-%m-%d'))
            self.episode_time_list.append(now_time.time().strftime('%H:%M:%S'))

            if self.report_type == 'plot':
                draw_check = Delamain.plot_reward(self.episode, self.episode_reward_list, self.timestep_n)

            if self.episode % self.when2log == 0:
                self.driver.write_log(
                    self.episode_date_list,
                    self.episode_time_list,
                    self.episode_reward_list,
                    self.episode_length_list,
                    self.episode_loss_list,
                    self.episode_epsilon_list,
                    log_filename='Delamain_log_test.csv'
                    )

    def eval(self):
        if not self.eval_tracks:
            return

        if isinstance(self.eval_tracks, list):
            tracks_iterable = self.eval_tracks
        else:
            tracks_iterable = range(0, self.eval_tracks)

        for track in tracks_iterable:
            self.state, info = self.env.reset(seed=track if isinstance(self.eval_tracks, list) else None)

            episode_reward = 0
            episode_length = 0
            updating = True

            while updating:
                episode_length += 1
                
                action = self.driver.take_action(self.state)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                self.state = new_state
                updating = not (terminated or truncated)
            
            print(f'Track seed: {track}')
            print(f'    reward {episode_reward}')
            print(f'    episode length {episode_length}')
