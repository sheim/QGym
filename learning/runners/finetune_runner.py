from learning.algorithms import *  # noqa: F403
from learning.algorithms import StateEstimator
from learning.modules import Actor, SmoothActor, Critic, StateEstimatorNN
from learning.storage import DictStorage
from gym.envs.mini_cheetah.minimalist_cheetah import MinimalistCheetah
from .BaseRunner import BaseRunner

import torch
import os
import scipy.io
from tensordict import TensorDict

sim_storage = DictStorage()


class FineTuneRunner(BaseRunner):
    def __init__(
        self,
        env,
        train_cfg,
        log_dir,
        data_list,
        data_length=1500,
        data_name="SMOOTH_RL_CONTROLLER",
        se_path=None,
        use_simulator=True,
        exploration_scale=1.0,
        device="cpu",
    ):
        # Instead of super init, only set necessary attributes
        self.env = env
        self.parse_train_cfg(train_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]

        self.log_dir = log_dir
        self.data_list = data_list  # Describes structure of Robot-Software logs
        self.data_length = data_length  # Logs must contain at least this many steps
        self.data_name = data_name
        self.se_path = se_path
        self.use_simulator = use_simulator
        self.exploration_scale = exploration_scale
        self.device = device
        self._set_up_alg()

    def _set_up_alg(self):
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])
        if self.actor_cfg["smooth_exploration"]:
            actor = SmoothActor(
                num_obs=num_actor_obs,
                num_actions=num_actions,
                exploration_scale=self.exploration_scale,
                **self.actor_cfg,
            )
        else:
            actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)

        alg_name = self.cfg["algorithm_class_name"]
        alg_class = eval(alg_name)
        self.ipg = alg_name in ["PPO_IPG", "LinkedIPG"]

        if self.ipg:
            critic_v = Critic(num_critic_obs, **self.critic_cfg)
            critic_q = Critic(num_critic_obs + num_actions, **self.critic_cfg)
            target_critic_q = Critic(num_critic_obs + num_actions, **self.critic_cfg)
            self.alg = alg_class(
                actor,
                critic_v,
                critic_q,
                target_critic_q,
                device=self.device,
                **self.alg_cfg,
            )
        else:
            critic = Critic(num_critic_obs, **self.critic_cfg)
            self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

        if "state_estimator" in self.train_cfg.keys() and self.se_path is not None:
            self.se_cfg = self.train_cfg["state_estimator"]
            state_estimator_network = StateEstimatorNN(
                self.get_obs_size(self.se_cfg["obs"]),
                self.get_obs_size(self.se_cfg["targets"]),
                **self.se_cfg["network"],
            )
            self.SE = StateEstimator(
                state_estimator_network, device=self.device, **self.se_cfg
            )
            self.load_se(self.se_path)
        else:
            self.SE = None

    def parse_train_cfg(self, train_cfg):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.actor_cfg = train_cfg["actor"]
        self.critic_cfg = train_cfg["critic"]
        self.train_cfg = train_cfg

    def get_data_dict(self, offpol=False, load_path=None, save_path=None):
        # Concatenate data with loaded dict
        loaded_data_dict = torch.load(load_path) if load_path else None

        checkpoint = self.cfg["checkpoint"]
        if offpol:
            # All files up until checkpoint
            log_files = [
                file
                for file in os.listdir(self.log_dir)
                if file.endswith(".mat") and int(file.split(".")[0]) <= checkpoint
            ]
            log_files = sorted(log_files)
        else:
            # Single log file for checkpoint
            log_files = [str(checkpoint) + ".mat"]

        # Initialize data dict
        data = scipy.io.loadmat(os.path.join(self.log_dir, log_files[0]))
        batch_size = (self.data_length - 1, len(log_files))  # -1 for next_obs
        data_dict = TensorDict({}, device=self.device, batch_size=batch_size)

        # Collect all data
        actor_obs_all = torch.empty(0).to(self.device)
        critic_obs_all = torch.empty(0).to(self.device)
        actions_all = torch.empty(0).to(self.device)
        rewards_all = torch.empty(0).to(self.device)
        for log in log_files:
            data = scipy.io.loadmat(os.path.join(self.log_dir, log))
            self.data_struct = data[self.data_name][0][0]
            if self.SE:
                self.update_state_estimates()

            actor_obs = self.get_data_obs(self.actor_cfg["obs"], self.data_struct)
            critic_obs = self.get_data_obs(self.critic_cfg["obs"], self.data_struct)
            actor_obs_all = torch.cat((actor_obs_all, actor_obs), dim=1)
            critic_obs_all = torch.cat((critic_obs_all, critic_obs), dim=1)

            actions_idx = self.data_list.index("dof_pos_target")
            actions = (
                torch.tensor(self.data_struct[actions_idx]).to(self.device).float()
            )
            actions = actions[: self.data_length]
            actions = actions.reshape(
                (self.data_length, 1, -1)
            )  # shape (data_length, 1, n)
            actions_all = torch.cat((actions_all, actions), dim=1)

            reward_weights = self.critic_cfg["reward"]["weights"]
            rewards, _ = self.get_data_rewards(self.data_struct, reward_weights)
            rewards = rewards[: self.data_length]
            rewards = rewards.reshape((self.data_length, 1))  # shape (data_length, 1)
            rewards_all = torch.cat((rewards_all, rewards), dim=1)

        data_dict["actor_obs"] = actor_obs_all[:-1]
        data_dict["next_actor_obs"] = actor_obs_all[1:]
        data_dict["critic_obs"] = critic_obs_all[:-1]
        data_dict["next_critic_obs"] = critic_obs_all[1:]
        data_dict["actions"] = actions_all[:-1]
        data_dict["rewards"] = rewards_all[:-1]

        # No time outs and dones
        data_dict["timed_out"] = torch.zeros(batch_size, device=self.device, dtype=bool)
        data_dict["dones"] = torch.zeros(batch_size, device=self.device, dtype=bool)

        # Concatenate with loaded dict
        if loaded_data_dict is not None:
            loaded_batch_size = loaded_data_dict.batch_size
            assert loaded_batch_size[0] == batch_size[0]
            new_batch_size = (
                loaded_batch_size[0],
                loaded_batch_size[1] + batch_size[1],
            )
            data_dict = TensorDict(
                {
                    key: torch.cat((loaded_data_dict[key], data_dict[key]), dim=1)
                    for key in data_dict.keys()
                },
                device=self.device,
                batch_size=new_batch_size,
            )

        if save_path:
            torch.save(data_dict, save_path)

        return data_dict

    def get_data_obs(self, obs_list, data_struct):
        obs_all = torch.empty(0).to(self.device)
        for obs_name in obs_list:
            data_idx = self.data_list.index(obs_name)
            obs = torch.tensor(data_struct[data_idx]).to(self.device)
            obs = obs.squeeze()[: self.data_length]
            obs = obs.reshape((self.data_length, 1, -1))  # shape (data_length, 1, n)
            obs_all = torch.cat((obs_all, obs), dim=-1)

        return obs_all.float()

    def get_data_rewards(self, data_struct, reward_weights):
        ctrl_dt = 1.0 / self.env.cfg.control.ctrl_frequency
        minimalist_cheetah = MinimalistCheetah(ctrl_dt=ctrl_dt, device=self.device)
        rewards_dict = {name: [] for name in reward_weights.keys()}  # for plotting
        rewards_all = torch.empty(0).to(self.device)

        for i in range(self.data_length):
            minimalist_cheetah.set_states(
                base_height=data_struct[1][i],
                base_lin_vel=data_struct[2][i],
                base_ang_vel=data_struct[3][i],
                proj_gravity=data_struct[4][i],
                commands=data_struct[5][i],
                dof_pos_obs=data_struct[6][i],
                dof_vel=data_struct[7][i],
                phase_obs=data_struct[8][i],
                grf=data_struct[9][i],
                dof_pos_target=data_struct[10][i],
            )
            total_rewards = 0
            for name, weight in reward_weights.items():
                reward = weight * eval(f"minimalist_cheetah._reward_{name}()")
                rewards_dict[name].append(reward.item())
                total_rewards += reward
            rewards_all = torch.cat((rewards_all, total_rewards), dim=0)
            # Post process mini cheetah
            minimalist_cheetah.post_process()

        rewards_dict["total"] = rewards_all.tolist()
        rewards_all *= ctrl_dt  # scaled for alg update

        return rewards_all.float(), rewards_dict

    def update_state_estimates(self):
        se_obs = torch.empty(0).to(self.device)
        for obs in self.se_cfg["obs"]:
            data_idx = self.data_list.index(obs)
            data = torch.tensor(self.data_struct[data_idx]).to(self.device)
            data = data.squeeze()[: self.data_length]
            data = data.reshape((self.data_length, -1))
            se_obs = torch.cat((se_obs, data), dim=-1)

        se_targets = self.SE.estimate(se_obs.float())

        # Overwrite data struct with state estimates
        idx = 0
        for target in self.se_cfg["targets"]:
            data_idx = self.data_list.index(target)
            dim = self.data_struct[data_idx].shape[1]
            self.data_struct[data_idx] = (
                se_targets[:, idx : idx + dim].cpu().detach().numpy()
            )
            idx += dim

    def load_data(self, load_path=None, save_path=None):
        # Load on- and off-policy data
        if self.use_simulator:
            # Simulate on-policy data
            self.data_onpol = TensorDict(
                self.get_sim_data(),
                batch_size=(self.num_steps_per_env, self.env.num_envs),
                device=self.device,
            )
        else:
            self.data_onpol = self.get_data_dict()

        if self.ipg:
            self.data_offpol = self.get_data_dict(
                offpol=True, load_path=load_path, save_path=save_path
            )
        else:
            self.data_offpol = None

    def learn(self):
        self.alg.switch_to_train()

        # Single alg update on data
        if self.data_offpol is None:
            self.alg.update(self.data_onpol)
        else:
            self.alg.update(self.data_onpol, self.data_offpol)

    def get_sim_data(self):
        rewards_dict = {}
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])

        # * Initialize smooth exploration matrices
        if self.actor_cfg["smooth_exploration"]:
            self.alg.actor.sample_weights(batch_size=self.env.num_envs)

        # * Start up storage
        transition = TensorDict({}, batch_size=self.env.num_envs, device=self.device)
        transition.update(
            {
                "actor_obs": actor_obs,
                "next_actor_obs": actor_obs,
                "actions": self.alg.act(actor_obs),
                "critic_obs": critic_obs,
                "next_critic_obs": critic_obs,
                "rewards": self.get_rewards({"termination": 0.0})["termination"],
                "dones": self.get_timed_out(),
            }
        )
        sim_storage.initialize(
            transition,
            self.env.num_envs,
            self.env.num_envs * self.num_steps_per_env,
            device=self.device,
        )

        # * Rollout
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # * Re-sample noise matrix for smooth exploration
                sample_freq = self.actor_cfg["exploration_sample_freq"]
                if self.actor_cfg["smooth_exploration"] and i % sample_freq == 0:
                    self.alg.actor.sample_weights(batch_size=self.env.num_envs)

                actions = self.alg.act(actor_obs)
                self.set_actions(
                    self.actor_cfg["actions"],
                    actions,
                    self.actor_cfg["disable_actions"],
                )

                transition.update(
                    {
                        "actor_obs": actor_obs,
                        "actions": actions,
                        "critic_obs": critic_obs,
                    }
                )

                self.env.step()

                actor_obs = self.get_noisy_obs(
                    self.actor_cfg["obs"], self.actor_cfg["noise"]
                )
                critic_obs = self.get_obs(self.critic_cfg["obs"])

                # * get time_outs
                timed_out = self.get_timed_out()
                terminated = self.get_terminated()
                dones = timed_out | terminated

                self.update_rewards(rewards_dict, terminated)
                total_rewards = torch.stack(tuple(rewards_dict.values())).sum(dim=0)

                transition.update(
                    {
                        "next_actor_obs": actor_obs,
                        "next_critic_obs": critic_obs,
                        "rewards": total_rewards,
                        "timed_out": timed_out,
                        "dones": dones,
                    }
                )
                sim_storage.add_transitions(transition)

        return sim_storage.data

    def update_rewards(self, rewards_dict, terminated):
        rewards_dict.update(
            self.get_rewards(
                self.critic_cfg["reward"]["termination_weight"], mask=terminated
            )
        )
        rewards_dict.update(
            self.get_rewards(
                self.critic_cfg["reward"]["weights"],
                modifier=self.env.dt,
                mask=~terminated,
            )
        )

    def save(self, path):
        if self.ipg:
            torch.save(
                {
                    "actor_state_dict": self.alg.actor.state_dict(),
                    "critic_v_state_dict": self.alg.critic_v.state_dict(),
                    "critic_q_state_dict": self.alg.critic_q.state_dict(),
                    "target_critic_q_state_dict": self.alg.target_critic_q.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "critic_v_opt_state_dict": self.alg.critic_v_optimizer.state_dict(),
                    "critic_q_opt_state_dict": self.alg.critic_q_optimizer.state_dict(),
                },
                path,
            )
            return
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_state_dict": self.alg.critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])

        if self.ipg:
            self.alg.critic_v.load_state_dict(loaded_dict["critic_v_state_dict"])
            self.alg.critic_q.load_state_dict(loaded_dict["critic_q_state_dict"])
            self.alg.target_critic_q.load_state_dict(
                loaded_dict["target_critic_q_state_dict"]
            )
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.alg.critic_v_optimizer.load_state_dict(
                    loaded_dict["critic_v_opt_state_dict"]
                )
                self.alg.critic_q_optimizer.load_state_dict(
                    loaded_dict["critic_q_opt_state_dict"]
                )
        else:
            self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.alg.critic_optimizer.load_state_dict(
                    loaded_dict["critic_optimizer_state_dict"]
                )

    def load_se(self, se_path):
        se_dict = torch.load(se_path)
        self.SE.network.load_state_dict(se_dict["SE_state_dict"])

    def export(self, path):
        # Need to make a copy of actor
        if self.actor_cfg["smooth_exploration"]:
            actor_copy = SmoothActor(
                self.alg.actor.num_obs, self.alg.actor.num_actions, **self.actor_cfg
            )
        else:
            actor_copy = Actor(
                self.alg.actor.num_obs, self.alg.actor.num_actions, **self.actor_cfg
            )
        state_dict = {
            name: param.detach().clone()
            for name, param in self.alg.actor.state_dict().items()
        }
        actor_copy.load_state_dict(state_dict)
        actor_copy.export(path)
