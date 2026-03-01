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
from enviroment.AgentPPO import AgentPPO
from enviroment.SkipFrame import SkipFrame
from enviroment.SkipFrameVec import SkipFrameVec
from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class TrainingGround:
    ENVIROMENT = "CarRacing-v3"
    VIDEO_DIR = "training/video"
    MODELS_DIR = "training/saved_models"
    PARAMS_FILE = "training_params.yaml"

    def __init__(self):
        yamlValues = None
        with open(self.PARAMS_FILE) as stream:
            try:
                yamlValues = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.batch_n = yamlValues["train"].get("batch_n", 32)
        self.play_n_episodes = yamlValues["train"].get(
            "play_n_episodes", 3000
        )
        self.episode_lr_list = []
        self.episode_epsilon_list = []
        self.episode_reward_list = []
        self.episode_length_list = []
        self.episode_loss_list = []
        self.episode_date_list = []
        self.episode_time_list = []
        self.episode = 0
        self.timestep_n = 0

        self.init_reporting(yamlValues["reporting"])

        self.eval_tracks = yamlValues["eval"].get("tracks")
        self.class_name = yamlValues["model"].get("class_name", "Delamain")

        self.env = gym.make_vec(
            self.ENVIROMENT, 
            num_envs=yamlValues["env"].get("envs_num", 4), 
            vectorization_mode=gym.VectorizeMode.ASYNC, 
            continuous=False, 
            render_mode="rgb_array", 
            domain_randomize=yamlValues["env"].get("random_colors", False),
        )
        
        self.env = SkipFrameVec(self.env, skip=yamlValues["env"].get("skip_frames", 4))
        # self.env = gym.make(
        #     self.ENVIROMENT,
        #     continuous=False,
        #     render_mode="rgb_array",
        #     domain_randomize=yamlValues["env"].get("random_colors", False),
        # )
        # self.env = SkipFrame(self.env, skip=yamlValues["env"].get("skip_frames", 4))
        

        if yamlValues["eval"]["video"] and False:
            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder=self.VIDEO_DIR,  # Folder to save videos
                name_prefix="eval",  # Prefix for video filenames
                episode_trigger=lambda x: (
                    True
                    if yamlValues["env"]["mode"] == "eval"
                    else x % self.when2eval == 0
                ),  # Record every episode
            )

        self.state, info = self.env.reset()
        self.previous_state = self.state
        action_n = self.env.single_action_space.n if yamlValues["env"].get("vec", False) else self.env.action_space.n
        if yamlValues["env"]["mode"] == "eval":
            self.driver = AgentPPO(
                self.state.shape,
                action_n,
                self.parse_class_name(yamlValues["model"]["class_name"]),
                buffer_size=0,
            )
            if not yamlValues["model"]["file_name"]:
                raise Exception("No model file name passed in training params")

            self.driver.load_state = "eval"
            self.driver.epsilon_end = self.driver.epsilon = 0.0
        else:
            agent_args = dict(
                gamma=yamlValues["train"]["gamma"],
                epsilon=yamlValues["train"]["epsilon"],
                epsilon_end=yamlValues["train"]["epsilon_end"],
                epsilon_decay=yamlValues["train"]["epsilon_decay"],
                lr=yamlValues["train"]["lr"],
                lr_decay=yamlValues["train"]["lr_decay"],
                buffer_size=yamlValues["train"]["buffer_size"],
                skip_frames=yamlValues["env"]["skip_frames"]
                * yamlValues["reporting"]["when2learn"],
                play_n_episodes=yamlValues["train"]["play_n_episodes"],
                vec=yamlValues["env"]["vec"],
            )
            self.driver = AgentPPO(
                self.state.shape,
                action_n,
                self.parse_class_name(yamlValues["model"]["class_name"]),
                **{k: v for k, v in agent_args.items() if v is not None},
            )
            if yamlValues["env"]["mode"]:
                self.driver.load_state = yamlValues["env"]["mode"]
        if yamlValues["model"]["file_name"]:
            self.driver.load(self.MODELS_DIR, yamlValues["model"]["file_name"])

    def init_reporting(self, section):
        self.when2learn = section.get("when2learn", 4)
        self.when2sync = section.get("when2sync", 5000)  # in timesteps
        self.when2save = section.get("when2save", 50000)  # in timesteps
        self.when2report = section.get("when2report", 5000)  # in timesteps
        self.when2eval = section.get("when2eval", 50000)  # in timesteps
        self.when2log = section.get("when2log", 10)  # in episodes
        self.report_type = section.get("report_type", "text")

    def parse_class_name(self, class_name: str | None) -> nn.Module:
        match class_name:
            case "Delamain":
                return Delamain
            case "Delamain_2":
                return Delamain_2
            case "Delamain_2_1":
                return Delamain_2_1
            case "Delamain_2_5":
                return Delamain_2_5
            case _:
                return Delamain

    def start(self):
        match self.driver.load_state:
            case "eval":
                return self.eval()
            case "train":
                return self.train()
            case "fine_tune":
                return self.fine_tune()
            case "kernel_vis":
                return self.kernels()
            case _:
                return self.train()

    def train(self):
        # [MODIFICATION] Vectorized tracking arrays (size n)
        # Assumes self.env is a vectorized environment (e.g., gym.vector.make)
        num_envs = self.env.num_envs 
        current_ep_rewards = np.zeros(num_envs)
        current_ep_lengths = np.zeros(num_envs)

        while self.episode <= self.play_n_episodes:
            self.episode += 1
            episode_reward = 0
            episode_length = 0
            updating = True
            loss_list = []
            self.episode_epsilon_list.append(self.driver.epsilon)
            self.episode_lr_list.append(self.driver.get_lr())

            while updating:
                self.timestep_n += 1
                episode_length += 1

                # --- ORIGINAL ---
                # action = self.driver.take_action(self.state)
                # new_state, reward, terminated, truncated, info = self.env.step(action)
                # episode_reward += reward
                # 
                # self.driver.store(self.state, action, reward, new_state, terminated)

                # --- MODIFICATION 1: Support PPO log_prob output while maintaining DQN compatibility ---
                action_out = self.driver.take_action(self.state)
                if isinstance(action_out, tuple): # PPO returns (action, log_prob)
                    action, log_prob = action_out
                else: # DQN returns just action
                    action = action_out
                    log_prob = None

                new_state, reward, terminated, truncated, info = self.env.step(action)
                # episode_reward += reward
                # Accumulate stats for each active environment
                current_ep_rewards += reward
                current_ep_lengths += 1

                # Pass log_prob only if using PPO
                if log_prob is not None:
                    self.driver.store(self.state, action, reward, new_state, terminated, log_prob=log_prob)
                else:
                    self.driver.store(self.state, action, reward, new_state, terminated)
                # --- END MODIFICATION 1 ---

                # Move to the next state
                # self.previous_state = self.state
                self.state = new_state
                updating = not (terminated.any() or truncated.any())
                # updating = not (terminated or truncated)

                dones = np.logical_or(terminated, truncated)
                for i in range(num_envs):
                    if dones[i]:
                        # self.episode += 1
                        
                        # Log the completed episode for this specific car
                        self.episode_reward_list.append(current_ep_rewards[i])
                        self.episode_length_list.append(current_ep_lengths[i])
                        self.episode_epsilon_list.append(self.driver.epsilon)
                        self.episode_lr_list.append(self.driver.scheduler.get_last_lr()[0])
                        
                        now_time = datetime.datetime.now()
                        self.episode_date_list.append(now_time.date().strftime('%Y-%m-%d'))
                        self.episode_time_list.append(now_time.time().strftime('%H:%M:%S'))

                        # Reset trackers for this specific car, as it auto-resets
                        current_ep_rewards[i] = 0
                        current_ep_lengths[i] = 0

                # --- ORIGINAL ---
                # if self.timestep_n % self.when2sync == 0:
                    # upd_net_param = self.driver.target_net.state_dict()
                    # self.driver.policy_net.load_state_dict(upd_net_param)
                
                # --- MODIFICATION 2: Only sync if using DQN (PPO doesn't use target_net) ---
                if self.timestep_n % self.when2sync == 0:
                    if getattr(self.driver, 'is_ppo', False) == False: 
                        upd_net_param = self.driver.target_net.state_dict()
                        self.driver.policy_net.load_state_dict(upd_net_param)
                # --- END MODIFICATION 2 ---

                if self.timestep_n % self.when2save == 0:
                    self.driver.save(self.driver.save_dir, self.class_name)

                if self.timestep_n % self.when2learn == 0:
                    q, loss = self.driver.update_net(self.batch_n)

                if (
                    self.timestep_n % self.when2report == 0
                    and self.report_type == "text"
                ):
                    print(f"Report: {self.timestep_n} timestep")
                    print(f"    episodes: {self.episode}")
                    print(f"    n_updates: {self.driver.n_updates}")
                    print(f"    epsilon: {self.driver.epsilon}")
                    print(f"    lr: {self.driver.get_lr()}")

                if self.timestep_n % self.when2eval == 0 and self.report_type == "text":
                    rewards_tensor = torch.tensor(
                        self.episode_reward_list, dtype=torch.float
                    )
                    eval_reward = torch.clone(rewards_tensor[-50:])
                    mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
                    std_eval_reward = round(torch.std(eval_reward).item(), 2)

                    lengths_tensor = torch.tensor(
                        self.episode_length_list, dtype=torch.float
                    )
                    eval_length = torch.clone(lengths_tensor[-50:])
                    mean_eval_length = round(torch.mean(eval_length).item(), 2)
                    std_eval_length = round(torch.std(eval_length).item(), 2)

                    print(f"Evaluation: {self.timestep_n} timestep")
                    print(f"    reward {mean_eval_reward}±{std_eval_reward}")
                    print(f"    episode length {mean_eval_length}±{std_eval_length}")
                    print(f"    episodes: {self.episode}")
                    print(f"    n_updates: {self.driver.n_updates}")
                    print(f"    epsilon: {self.driver.epsilon}")
                    print(f"    lr: {self.driver.get_lr()}")
                    epsl_end = self.driver.epsilon_end
                    curr_epsl = self.driver.epsilon
                    self.driver.epsilon_end = self.driver.epsilon = 0.0
                    self.driver.policy_net.eval()
                    self.driver.load_state = "eval"
                    with torch.no_grad():
                        self.eval()
                    self.driver.load_state = "train"
                    self.driver.policy_net.train()
                    self.driver.epsilon_end = epsl_end
                    self.driver.epsilon = curr_epsl

            # --- ORIGINAL ---
            # batch = self.driver.bufferLoss.sample(250).to("cpu")
            # for i in [x for x in batch if x is not None]:
            #     loss_list.append(i.get("loss").item())
            # self.driver.bufferLoss.empty()
            # --- MODIFICATION 3: Safely sample from bufferLoss (Protects against empty/small buffers) ---
            buffer_len = len(self.driver.bufferLoss)
            if buffer_len > 0:
                sample_size = min(250, buffer_len) # Sample 250, or whatever is available
                batch = self.driver.bufferLoss.sample(sample_size).to('cpu')
                for i in [x for x in batch if x is not None]:
                    loss_list.append(i.get('loss').item())
                self.driver.bufferLoss.empty()
            else:
                loss_list.append(0.0) # No network updates happened during this specific episode
            # --- END MODIFICATION 3 ---
            self.state, info = self.env.reset()

            # self.episode_reward_list.append(episode_reward)
            # self.episode_length_list.append(episode_length)
            # self.episode_loss_list.append(np.mean(loss_list))
            # now_time = datetime.datetime.now()
            # self.episode_date_list.append(now_time.date().strftime("%Y-%m-%d"))
            # self.episode_time_list.append(now_time.time().strftime("%H:%M:%S"))

            if self.report_type == "plot":
                draw_check = Delamain.plot_reward(
                    self.episode, self.episode_reward_list, self.timestep_n
                )

            if self.episode % self.when2log == 0:
                self.driver.write_log(
                    self.episode_date_list,
                    self.episode_time_list,
                    self.episode_reward_list,
                    self.episode_length_list,
                    self.episode_loss_list,
                    self.episode_epsilon_list,
                    self.episode_lr_list,
                    log_filename=f"{self.class_name}_log_test.csv",
                )
        self.driver.save(self.driver.SAVE_DIR, self.class_name)

        self.env.close()
        plt.ioff()
        plt.show()

    def fine_tune(self):
        raise Exception("Currently fine_tune not supported")

    def eval(self):
        if not self.eval_tracks:
            return

        if isinstance(self.eval_tracks, list):
            tracks_iterable = self.eval_tracks
        else:
            tracks_iterable = range(0, self.eval_tracks)
        actions = np.zeros(5, dtype=np.uint8)

        for track in tracks_iterable:
            seed: int = (
                track
                if isinstance(self.eval_tracks, list)
                else self.env.np_random.integers(0, 9223372036854775807, dtype=int)
            )
            seed_arr: list[int] = []
            for i in range(self.env.num_envs):
                seed_arr.append(seed)
            self.state, info = self.env.reset(seed=seed_arr)
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
                # --- ORIGINAL ---
                # action = self.driver.take_action(self.state)
                # actions[action] += 1

                action_out = self.driver.take_action(self.state)
                if isinstance(action_out, tuple): # PPO returns (action, log_prob)
                    action, log_prob = action_out
                else: # DQN returns just action
                    action = action_out
                    log_prob = None
                if isinstance(action, int):
                    actions[action] += 1
                elif isinstance(action, list):
                    actions[action[0]] += 1

                new_state, reward, terminated, truncated, info = self.env.step(action)

                if isinstance(reward, np.ndarray):
                    episode_reward += reward[0]
                else:
                    episode_reward += reward

                # self.previous_state = self.state
                self.state = new_state
                if isinstance(terminated, bool):
                    updating = not (terminated or truncated)
                else:
                    updating = not (terminated[0] or truncated[0])
                

            print(f"Track seed: {seed}")
            print(f"    reward {episode_reward}")
            print(f"    episode length {episode_length}")
            print(f"    info {info}")
            print(f"    actions {actions}")

        # self.env.close()

    def kernels(self):
        kernels = self.driver.policy_net.conv1.weight.detach().clone()

        # check size for sanity check
        print(kernels.size())

        # normalize to (0,1) range so that matplotlib
        # can plot them
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        filter_img = torchvision.utils.make_grid(kernels, nrow=16)
        # change ordering since matplotlib requires images to
        # be (H, W, C)
        plt.imshow(torch.Tensor.cpu(filter_img.permute(1, 2, 0)))

        # You can directly save the image as well using
        img = torchvision.utils.save_image(
            kernels, "./encoder_conv1_filters.png", nrow=16
        )
        print("kernel saved")
