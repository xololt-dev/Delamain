import gymnasium as gym
import yaml
import numpy as np
import torch
import torchvision
import datetime
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import time

from enviroment.Agent import Agent
from enviroment.SkipFrame import SkipFrame
from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

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
        self.episode_lr_list = []
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
        self.class_name = yamlValues['model']['class_name'] if yamlValues['model']['class_name'] else 'Delamain'

        self.env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array", domain_randomize=yamlValues['env']['random_colors'] if yamlValues['env']['random_colors'] else False)
        self.env = SkipFrame(self.env, skip=yamlValues['env']['skip_frames'] if yamlValues['env']['skip_frames'] else 4)
        if yamlValues['eval']['video']:
            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder="training/video",    # Folder to save videos
                name_prefix="eval",               # Prefix for video filenames
                episode_trigger=lambda x: True if yamlValues['env']['mode'] == 'eval' else lambda x: x % self.when2eval == 0   # Record every episode
            )

        self.state, info = self.env.reset()
        self.previous_state = self.state
        action_n = self.env.action_space.n
        if yamlValues['env']['mode'] == 'eval':
            self.driver = Agent(self.state.shape, action_n, self.parse_class_name(yamlValues['model']['class_name']), buffer_size=0)
            if not yamlValues['model']['file_name']:
                raise Exception("No model file name passed in training params")

            self.driver.load_state = 'eval'
            self.driver.epsilon_end = self.driver.epsilon = 0.0
        else:
            agent_args = dict(gamma=yamlValues['train']['gamma'],
                epsilon=yamlValues['train']['epsilon'],
                epsilon_end=yamlValues['train']['epsilon_end'],
                epsilon_decay=yamlValues['train']['epsilon_decay'],
                lr=yamlValues['train']['lr'],
                lr_decay=yamlValues['train']['lr_decay'],
                buffer_size=yamlValues['train']['buffer_size'],
                skip_frames=yamlValues['env']['skip_frames'] * yamlValues['reporting']['when2learn'],
                play_n_episodes=yamlValues['train']['play_n_episodes'])
            self.driver = Agent(self.state.shape, 
                action_n, 
                self.parse_class_name(yamlValues['model']['class_name']),
                **{k: v for k, v in agent_args.items() if v is not None}
            )
            if yamlValues['env']['mode']:
                self.driver.load_state = yamlValues['env']['mode']
        if yamlValues['model']['file_name']:
            self.driver.load('./training/saved_models', yamlValues['model']['file_name'])

    def parse_class_name(self, class_name: str | None) -> nn.Module:
        match class_name:
            case 'Delamain':
                return Delamain
            case 'Delamain_2':
                return Delamain_2
            case 'Delamain_2_1':
                return Delamain_2_1
            case 'Delamain_2_5':
                return Delamain_2_5
            case _:
                return Delamain

    def start(self):
        match self.driver.load_state:
            case 'eval':
                return self.eval()
            case 'train':
                return self.train()
            case 'fine_tune':
                raise Exception("Currently fine_tune not supported")
            case 'kernel_vis':
                return self.kernels()
            case _:
                return self.train()

    def train(self):
        while self.episode <= self.play_n_episodes:
            self.episode += 1
            episode_reward = 0
            episode_length = 0
            updating = True
            loss_list = []
            self.episode_epsilon_list.append(self.driver.epsilon)
            self.episode_lr_list.append(self.driver.scheduler.get_last_lr()[0])

            # t1 = time.perf_counter(), time.process_time()
            while updating:
                # t1 = time.perf_counter(), time.process_time()
                self.timestep_n += 1
                episode_length += 1

                # if self.driver.target_net.is_prev_frame_needed():
                    # action = self.driver.take_action(torch.tensor(np.array([self.state, self.previous_state])))
                    # action = self.driver.take_action(self.state)
                # else:
                action = self.driver.take_action(self.state)
                # torch.cuda.synchronize()
                # t1 = time.perf_counter(), time.process_time()
                new_state, reward, terminated, truncated, info = self.env.step(action)
                # t2 = time.perf_counter(), time.process_time()
                # print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
                # print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
                # print()
                episode_reward += reward

                self.driver.store(self.state, action, reward, new_state, terminated)

                # Move to the next state
                # self.previous_state = self.state
                self.state = new_state
                updating = not (terminated or truncated)

                if self.timestep_n % self.when2sync == 0:
                    upd_net_param = self.driver.target_net.state_dict()
                    self.driver.policy_net.load_state_dict(upd_net_param)

                if self.timestep_n % self.when2save == 0:
                    self.driver.save(self.driver.save_dir, self.class_name)

                if self.timestep_n % self.when2learn == 0:
                    q, loss = self.driver.update_net(self.batch_n)

                # t2 = time.perf_counter(), time.process_time()
                # print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
                # print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
                # print()


                if self.timestep_n % self.when2report == 0 and self.report_type == 'text':
                    print(f'Report: {self.timestep_n} timestep')
                    print(f'    episodes: {self.episode}')
                    print(f'    n_updates: {self.driver.n_updates}')
                    print(f'    epsilon: {self.driver.epsilon}')
                    print(f'    lr: {self.driver.scheduler.get_last_lr()[0]}')

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
                    print(f'    lr: {self.driver.scheduler.get_last_lr()[0]}')
                    epsl_end = self.driver.epsilon_end
                    curr_epsl = self.driver.epsilon
                    self.driver.epsilon_end = self.driver.epsilon = 0.0
                    self.driver.policy_net.eval()
                    with torch.no_grad():
                        self.eval()
                    self.driver.policy_net.train()
                    self.driver.epsilon_end = epsl_end
                    self.driver.epsilon = curr_epsl

            batch = self.driver.bufferLoss.sample(250).to('cpu')
            for i in [x for x in batch if x is not None]:
                loss_list.append(i.get('loss').item())
            self.driver.bufferLoss.empty()
            self.state, info = self.env.reset()
            # t2 = time.perf_counter(), time.process_time()
            # print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
            # print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
            # print()

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
                    self.episode_lr_list,
                    log_filename=f'{self.class_name}_log_test.csv'
                    )
        self.driver.save(self.driver.save_dir, self.class_name)

        self.env.close()
        plt.ioff()
        plt.show()

    def fine_tune(self):
        pass

    def eval(self):
        if not self.eval_tracks:
            return

        if isinstance(self.eval_tracks, list):
            tracks_iterable = self.eval_tracks
        else:
            tracks_iterable = range(0, self.eval_tracks)
        actions = np.zeros(5, dtype=np.uint8)

        for track in tracks_iterable:
            seed = track if isinstance(self.eval_tracks, list) else self.env.np_random.integers(0, 9223372036854775807, dtype=int)
            self.state, info = self.env.reset(seed=seed)
            actions.fill(0)

            episode_reward = 0
            episode_length = 0
            updating = True
            action = None

            while updating:
                episode_length += 1
                
                # if self.driver.target_net.is_prev_frame_needed():
                #     # action = self.driver.take_action(torch.tensor(np.array([self.state, self.previous_state])))
                #     action = self.driver.take_action(self.state)
                # else:
                action = self.driver.take_action(self.state)
                actions[action] += 1

                new_state, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward

                # self.previous_state = self.state
                self.state = new_state
                updating = not (terminated or truncated)
            
            print(f'Track seed: {seed}')
            print(f'    reward {episode_reward}')
            print(f'    episode length {episode_length}')
            print(f'    info {info}')
            print(f'    actions {actions}')

        self.env.close()

    def kernels(self):
        kernels = self.driver.policy_net.conv1.weight.detach().clone()

        #check size for sanity check
        print(kernels.size())

        # normalize to (0,1) range so that matplotlib
        # can plot them
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        filter_img = torchvision.utils.make_grid(kernels, nrow = 16)
        # change ordering since matplotlib requires images to 
        # be (H, W, C)
        plt.imshow(torch.Tensor.cpu(filter_img.permute(1, 2, 0)))

        # You can directly save the image as well using
        img = torchvision.utils.save_image(kernels, './encoder_conv1_filters.png', nrow = 16)
        print('kernel saved')