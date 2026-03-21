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

from functools import partial

from enviroment.AgentDDQN import AgentDDQN
from enviroment.Agent import Agent
from enviroment.AgentPPO import AgentPPO
from enviroment.Algorithms import Algorithms
from enviroment.HSLObservation import HSLObservation
from enviroment.HSLObservationVec import HSLObservationVec
from enviroment.SkipFrame import SkipFrame
from enviroment.SkipFrameVec import SkipFrameVec
from enviroment.OpticalFlowObservation import OpticalFlowObservation
from enviroment.OpticalFlowObservationVec import OpticalFlowObservationVec
from alternative_models.Delamain import Delamain
from alternative_models.Delamain_2 import Delamain_2
from alternative_models.Delamain_2_1 import Delamain_2_1
from alternative_models.Delamain_2_5 import Delamain_2_5, Delamain_2_5_PPO
from alternative_models.Delamain_2_6 import Delamain_2_6, Delamain_2_6_PPO

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class TrainingGround:
    ENVIROMENT = "CarRacing-v3"
    VIDEO_DIR = "training/video"
    MODELS_DIR = "training/saved_models"
    PARAMS_FILE = "training_params.yaml"
    FUEL_PENALTY_ARR = [1.5, 2.0]

    def __init__(self):
        yamlValues = None
        with open(self.PARAMS_FILE) as stream:
            try:
                yamlValues = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.batch_n = yamlValues["train"].get("batch_n", 32)
        self.play_n_episodes = yamlValues["train"].get("play_n_episodes", 3000)
        self.episode_lr_list = []
        self.episode_epsilon_list = []
        self.episode_reward_list = []
        self.episode_length_list = []
        self.episode_loss_list = []
        self.episode_actions_in_row_list = []
        self.episode_date_list = []
        self.episode_time_list = []
        self.episode_fuel_efficiency_list = []
        self.episode = 0
        self.timestep_n = 0

        self.init_reporting(yamlValues["reporting"])

        self.eval_tracks = yamlValues["eval"].get("tracks")
        self.class_name = yamlValues["model"].get("class_name", "Delamain")
        self.algorithm: Algorithms = Algorithms[
            yamlValues["train"].get("algorithm", "DQN")
        ]
        self.vec: bool = yamlValues["env"].get("vec", False)
        self.envs_num: int = yamlValues["env"].get("envs_num", 1)
        self._skip_frames: int = yamlValues["env"].get("skip_frames", 4)
        self._optical_flow: bool = yamlValues["env"].get("optical_flow", False)

        if self.vec:
            self.env = gym.make_vec(
                self.ENVIROMENT,
                num_envs=self.envs_num,
                vectorization_mode=gym.VectorizeMode.ASYNC,
                # vector_kwargs={
                #     "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
                # },
                continuous=False,
                render_mode="rgb_array",
                domain_randomize=yamlValues["env"].get("random_colors", False),
            )
            self.env = HSLObservationVec(self.env)
            self.env = SkipFrameVec(
                self.env,
                skip=self._skip_frames,
            )
            if self._optical_flow:
                self.env = OpticalFlowObservationVec(
                    self.env,
                    skip=self._skip_frames,
                    channels=3,
                )
        else:
            self.env = gym.make(
                self.ENVIROMENT,
                continuous=False,
                render_mode="rgb_array",
                domain_randomize=yamlValues["env"].get("random_colors", False),
            )

            self.env = HSLObservation(self.env)
            self.env = SkipFrame(self.env, skip=self._skip_frames)
            if self._optical_flow:
                self.env = OpticalFlowObservation(
                    self.env,
                    skip=self._skip_frames,
                    channels=3,
                )

            if yamlValues["eval"]["video"]:
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
        action_n = (
            self.env.single_action_space.n if self.vec else self.env.action_space.n
        )
        driver_class = self.parse_algorithm(self.algorithm)

        if yamlValues["env"]["mode"] == "eval":
            if not yamlValues["model"]["file_name"]:
                raise Exception("No model file name passed in training params")

            agent_args = dict(device=yamlValues["env"].get("device", None))
            self.driver = driver_class(
                self.state.shape,
                action_n,
                self.parse_class_name(yamlValues["model"]["class_name"]),
                buffer_size=0,
                **{k: v for k, v in agent_args.items() if v is not None},
            )

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
                vec=self.vec,
                device=yamlValues["env"].get("device", None),
            )
            self.driver = driver_class(
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

    def _get_input_channels(self) -> int:
        """Compute the number of input channels based on the wrapper chain."""
        skip_frames = self._skip_frames
        optical_flow = self._optical_flow
        base_channels = 3  # HSLObservation always outputs 3 channels

        if optical_flow:
            return base_channels + 2  # HSL channels + dx + dy
        return skip_frames * base_channels

    def parse_class_name(self, class_name: str | None):
        match class_name:
            case "Delamain":
                return Delamain
            case "Delamain_2":
                return Delamain_2
            case "Delamain_2_1":
                return Delamain_2_1
            case "Delamain_2_5":
                if self.algorithm == Algorithms.PPO:
                    return Delamain_2_5_PPO
                return Delamain_2_5
            case "Delamain_2_6":
                in_channels = self._get_input_channels()
                if self.algorithm == Algorithms.PPO:
                    return partial(Delamain_2_6_PPO, in_channels=in_channels)
                return partial(Delamain_2_6, in_channels=in_channels)
            case _:
                return Delamain

    def parse_algorithm(
        self, algorithm: Algorithms | None
    ) -> type[Agent | AgentPPO | AgentDDQN]:
        match algorithm:
            case Algorithms.DQN:
                return Agent
            case Algorithms.PPO:
                return AgentPPO
            case Algorithms.DDQN:
                return AgentDDQN
            case _:
                return Agent

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
        if self.vec:
            self.train_vec()
        else:
            self.train_scalar()

    def train_scalar(self):
        while self.episode <= self.play_n_episodes:
            self.episode += 1
            episode_reward = 0
            episode_length = 0
            updating = True
            loss_list = []
            prev_action = None
            actions_in_row = [0]
            actions = np.zeros(5, dtype=np.uint8)
            self.episode_epsilon_list.append(self.driver.epsilon)
            self.episode_lr_list.append(self.driver.get_lr())

            while updating:
                self.timestep_n += 1
                episode_length += 1

                action_out = self.driver.take_action(self.state)
                if self.algorithm == Algorithms.PPO:  # PPO returns (action, log_prob)
                    action, log_prob = action_out
                else:  # DQN returns just action
                    action = action_out
                    log_prob = None

                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                actions[action] += 1
                if prev_action == action:
                    actions_in_row[-1] += 1
                else:
                    prev_action = action
                    if prev_action != None:
                        actions_in_row.append(0)

                # Pass log_prob only if using PPO
                if self.algorithm == Algorithms.PPO:
                    self.driver.store(
                        self.state,
                        action,
                        reward,
                        new_state,
                        terminated,
                        log_prob=log_prob,
                    )
                else:
                    self.driver.store(self.state, action, reward, new_state, terminated)

                # Move to the next state
                self.state = new_state
                updating = not (terminated or truncated)

                if self.timestep_n % self.when2sync == 0:
                    if self.algorithm != Algorithms.PPO:
                        upd_net_param = self.driver.target_net.state_dict()
                        self.driver.policy_net.load_state_dict(upd_net_param)

                if self.timestep_n % self.when2save == 0:
                    self.driver.save(self.driver.SAVE_DIR, self.class_name)

                if self.timestep_n % self.when2learn == 0:
                    q, loss = self.driver.update_net(self.batch_n)
                    loss_list.append(loss)

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

            self.state, info = self.env.reset()

            self.episode_reward_list.append(episode_reward)
            self.episode_length_list.append(episode_length)
            self.episode_loss_list.append(np.mean(loss_list))
            self.episode_actions_in_row_list.append(np.mean(actions_in_row))
            print("actions_in_row:", np.mean(actions_in_row))

            efficiency_bonus = 1.0 + np.sum(actions[:3]) * 0.01
            penalty = np.dot(actions[3:], self.FUEL_PENALTY_ARR)
            fuel_efficiency = (episode_reward * efficiency_bonus) / (penalty + 1.0)

            self.episode_fuel_efficiency_list.append(fuel_efficiency)
            print(f"Fuel efficiency: {fuel_efficiency:.3f}")

            now_time = datetime.datetime.now()
            self.episode_date_list.append(now_time.date().strftime("%Y-%m-%d"))
            self.episode_time_list.append(now_time.time().strftime("%H:%M:%S"))

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
                    self.episode_actions_in_row_list,
                    self.episode_fuel_efficiency_list,
                    log_filename=f"{self.class_name}_log_test.csv",
                )
        self.driver.save(self.driver.SAVE_DIR, self.class_name)

        self.env.close()
        plt.ioff()
        plt.show()

    def train_vec(self):
        assert (
            self.algorithm == Algorithms.PPO
        ), "Currently vectorized env works only on PPO!"

        current_ep_rewards = np.zeros(self.envs_num)
        current_ep_lengths = np.zeros(self.envs_num)
        last_loss = 0.0
        prev_action = np.full(self.envs_num, None)
        actions_in_row = []
        actions = np.zeros((self.envs_num, 5), dtype=np.uint8)

        while self.episode <= self.play_n_episodes:
            self.episode += 1
            current_ep_rewards.fill(0)
            current_ep_lengths.fill(0)
            prev_action.fill(None)
            for i in range(0, self.envs_num):
                actions_in_row.append([0])
            actions.fill(0)
            updating = True

            while updating:
                self.timestep_n += 1

                action_out = self.driver.take_action(self.state)
                if self.algorithm == Algorithms.PPO:  # PPO returns (action, log_prob)
                    action, log_prob = action_out
                else:
                    action = action_out
                    log_prob = None

                new_state, reward, terminated, truncated, info = self.env.step(action)
                rows = np.arange(len(action))
                np.add.at(actions, (rows, action), 1)

                for i in range(0, len(action)):
                    if prev_action[i] == action[i]:
                        actions_in_row[i][-1] += 1
                    else:
                        prev_action[i] = action[i]
                        if prev_action[i] != None:
                            actions_in_row[i].append(0)

                current_ep_rewards += reward
                current_ep_lengths += 1

                if self.algorithm == Algorithms.PPO:
                    self.driver.store(
                        self.state,
                        action,
                        reward,
                        new_state,
                        terminated,
                        log_prob=log_prob,
                    )
                else:
                    self.driver.store(self.state, action, reward, new_state, terminated)

                # Move to the next state
                self.state = new_state
                updating = not (terminated.any() or truncated.any())

                dones = np.logical_or(terminated, truncated)
                for i in range(self.envs_num):
                    if dones[i]:
                        # Log the completed episode for this specific car
                        self.episode_reward_list.append(current_ep_rewards[i])
                        self.episode_length_list.append(current_ep_lengths[i])
                        self.episode_loss_list.append(last_loss)
                        self.episode_epsilon_list.append(self.driver.epsilon)
                        self.episode_lr_list.append(self.driver.get_lr())
                        self.episode_actions_in_row_list.append(
                            np.mean(actions_in_row[i])
                        )
                        prev_action[i] = None
                        actions_in_row[i] = [0]

                        efficiency_bonus = 1.0 + np.sum(actions[i, :3]) * 0.01
                        penalty = np.dot(actions[i, 3:], self.FUEL_PENALTY_ARR)
                        fuel_efficiency = (current_ep_rewards[i] * efficiency_bonus) / (
                            penalty + 1.0
                        )
                        actions[i].fill(0)

                        self.episode_fuel_efficiency_list.append(fuel_efficiency)
                        print(f"Fuel efficiency: {fuel_efficiency:.3f}")

                        now_time = datetime.datetime.now()
                        self.episode_date_list.append(
                            now_time.date().strftime("%Y-%m-%d")
                        )
                        self.episode_time_list.append(
                            now_time.time().strftime("%H:%M:%S")
                        )

                        # Reset trackers for this specific car, as it auto-resets
                        current_ep_rewards[i] = 0
                        current_ep_lengths[i] = 0

                if self.timestep_n % self.when2sync == 0:
                    if self.algorithm != Algorithms.PPO:
                        upd_net_param = self.driver.target_net.state_dict()
                        self.driver.policy_net.load_state_dict(upd_net_param)

                if self.timestep_n % self.when2save == 0:
                    self.driver.save(self.driver.SAVE_DIR, self.class_name)

                if self.timestep_n % self.when2learn == 0:
                    q, loss = self.driver.update_net(self.batch_n)
                    last_loss = loss.cpu().numpy()

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
                    # Because eval resets env we need to stop current updating to force another env reset post eval
                    updating = False

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

            self.state, info = self.env.reset()

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
                    self.episode_actions_in_row_list,
                    self.episode_fuel_efficiency_list,
                    log_filename=f"{self.class_name}_log_test.csv",
                )
        self.driver.save(self.driver.SAVE_DIR, self.class_name)

        self.env.close()
        plt.ioff()
        plt.show()

    # AFAIK disabled auto-reset steps through not-to-be-reset enviroments with ? action, which makes this approach... wrong in a different way compared to train_vec
    def train_vec_2(self):
        assert (
            self.algorithm == Algorithms.PPO
        ), "Currently vectorized env works only on PPO!"

        current_ep_rewards = np.zeros(self.envs_num)
        current_ep_lengths = np.zeros(self.envs_num)
        last_loss = 0.0
        self.episode += self.envs_num

        while self.episode <= self.play_n_episodes:
            self.timestep_n += 1

            action_out = self.driver.take_action(self.state)
            if self.algorithm == Algorithms.PPO:  # PPO returns (action, log_prob)
                action, log_prob = action_out
            else:
                action = action_out
                log_prob = None

            new_state, reward, terminated, truncated, info = self.env.step(action)
            current_ep_rewards += reward
            current_ep_lengths += 1

            if self.algorithm == Algorithms.PPO:
                self.driver.store(
                    self.state,
                    action,
                    reward,
                    new_state,
                    terminated,
                    log_prob=log_prob,
                )
            else:
                self.driver.store(self.state, action, reward, new_state, terminated)

            # Move to the next state
            self.state = new_state
            dones = np.logical_or(terminated, truncated)

            if np.any(dones):
                for i in range(self.envs_num):
                    if dones[i]:
                        # Log the completed episode for this specific car
                        self.episode_reward_list.append(current_ep_rewards[i])
                        self.episode_length_list.append(current_ep_lengths[i])
                        self.episode_loss_list.append(last_loss)
                        self.episode_epsilon_list.append(self.driver.epsilon)
                        self.episode_lr_list.append(self.driver.get_lr())
                        now_time = datetime.datetime.now()
                        self.episode_date_list.append(
                            now_time.date().strftime("%Y-%m-%d")
                        )
                        self.episode_time_list.append(
                            now_time.time().strftime("%H:%M:%S")
                        )

                        # Reset trackers for this specific car, as it auto-resets
                        current_ep_rewards[i] = 0
                        current_ep_lengths[i] = 0
                        self.episode += 1

                self.state, info = self.env.reset(options={"reset_mask": dones})

            if self.timestep_n % self.when2sync == 0:
                if self.algorithm != Algorithms.PPO:
                    upd_net_param = self.driver.target_net.state_dict()
                    self.driver.policy_net.load_state_dict(upd_net_param)

            if self.timestep_n % self.when2save == 0:
                self.driver.save(self.driver.SAVE_DIR, self.class_name)

            if self.timestep_n % self.when2learn == 0:
                q, loss = self.driver.update_net(self.batch_n)
                last_loss = loss.cpu().numpy()

            if self.timestep_n % self.when2report == 0 and self.report_type == "text":
                print(f"Report: {self.timestep_n} timestep")
                print(f"    episodes: {self.episode}")
                print(f"    n_updates: {self.driver.n_updates}")
                print(f"    epsilon: {self.driver.epsilon}")
                print(f"    lr: {self.driver.get_lr()}")

            if (
                self.timestep_n % self.when2eval == 0
                and self.report_type == "text"
                and self.timestep_n != 0
            ):
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

                self.state, info = self.env.reset()

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
                    self.episode_actions_in_row_list,
                    self.episode_fuel_efficiency_list,
                    log_filename=f"{self.class_name}_log_test.csv",
                )
        self.driver.save(self.driver.SAVE_DIR, self.class_name)

        self.env.close()
        plt.ioff()
        plt.show()

    def fine_tune(self):
        raise Exception("Currently fine_tune not supported")

    def eval(self):
        if self.vec:
            self.eval_vec()
        else:
            self.eval_scalar()

    def eval_scalar(self):
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
            self.state, info = self.env.reset(seed=seed)
            actions.fill(0)

            episode_reward = 0
            episode_length = 0
            updating = True
            action = None
            prev_action = None
            actions_in_row = [0]

            while updating:
                episode_length += 1

                action_out = self.driver.take_action(self.state)
                if self.algorithm == Algorithms.PPO:  # PPO returns (action, log_prob)
                    action, log_prob = action_out
                else:  # DQN returns just action
                    action = action_out
                    log_prob = None

                actions[action] += 1
                if prev_action == action:
                    actions_in_row[-1] += 1
                else:
                    prev_action = action
                    if prev_action != None:
                        actions_in_row.append(0)

                new_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                self.state = new_state
                updating = not (terminated or truncated)

            print(f"Track seed: {seed}")
            print(f"    reward {episode_reward}")
            print(f"    episode length {episode_length}")
            print(f"    info {info}")
            print(f"    actions {actions}")
            print("actions_in_row:", np.mean(actions_in_row))

            efficiency_bonus = 1.0 + np.sum(actions[:3]) * 0.01
            penalty = np.dot(actions[3:], self.FUEL_PENALTY_ARR)
            fuel_efficiency = (episode_reward * efficiency_bonus) / (penalty + 1.0)

            self.episode_fuel_efficiency_list.append(fuel_efficiency)
            print(f"Fuel efficiency: {fuel_efficiency:.3f}")

    def eval_vec(self):
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
            for i in range(self.envs_num):
                seed_arr.append(seed)
            self.state, info = self.env.reset(seed=seed_arr)
            actions.fill(0)

            episode_reward = 0
            episode_length = 0
            updating = True
            action = None

            while updating:
                episode_length += 1

                action_out = self.driver.take_action(self.state)
                if self.algorithm == Algorithms.PPO:  # PPO returns (action, log_prob)
                    action, log_prob = action_out
                else:
                    action = action_out
                    log_prob = None
                if isinstance(action, int):
                    actions[action] += 1
                elif isinstance(action, list):
                    actions[action[0]] += 1

                new_state, reward, terminated, truncated, info = self.env.step(action)

                if self.vec:
                    episode_reward += reward[0]
                else:
                    episode_reward += reward

                self.state = new_state
                if not self.vec:
                    updating = not (terminated or truncated)
                else:
                    updating = not (terminated[0] or truncated[0])

            print(f"Track seed: {seed}")
            print(f"    reward {episode_reward}")
            print(f"    episode length {episode_length}")
            print(f"    info {info}")
            print(f"    actions {actions}")

            efficiency_bonus = 1.0 + np.sum(actions[:3]) * 0.01
            penalty = np.dot(actions[3:], self.FUEL_PENALTY_ARR)
            fuel_efficiency = (episode_reward * efficiency_bonus) / (penalty + 1.0)

            self.episode_fuel_efficiency_list.append(fuel_efficiency)
            print(f"Fuel efficiency: {fuel_efficiency:.3f}")

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
