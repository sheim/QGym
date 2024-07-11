import torch
from learning.algorithms import *  # noqa: F403
from learning.modules import Actor, Critic, SmoothActor
from learning.utils import remove_zero_weighted_rewards


class BaseRunner:
    def __init__(self, env, train_cfg, device="cpu"):
        self.device = device
        self.env = env
        self.parse_train_cfg(train_cfg)

        self.log_storage = self.cfg["log_storage"]
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_learning_iterations = self.cfg["max_iterations"]
        self.tot_timesteps = 0
        self.it = 0
        self.log_dir = train_cfg["log_dir"]
        self._set_up_alg()

    def _set_up_alg(self):
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])

        if self.actor_cfg["smooth_exploration"]:
            actor = SmoothActor(num_actor_obs, num_actions, **self.actor_cfg)
        else:
            actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)

        critic = Critic(num_critic_obs, **self.critic_cfg)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

    def parse_train_cfg(self, train_cfg):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        remove_zero_weighted_rewards(train_cfg["critic"]["reward"]["weights"])
        self.actor_cfg = train_cfg["actor"]
        self.critic_cfg = train_cfg["critic"]

    def init_storage(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def get_noise(self, obs_list, noise_dict):
        noise_vec = torch.zeros(self.get_obs_size(obs_list), device=self.device)
        obs_index = 0
        for obs in obs_list:
            obs_size = self.get_obs_size([obs])
            if obs in noise_dict.keys():
                noise_tensor = torch.ones(obs_size).to(self.device) * torch.tensor(
                    noise_dict[obs]
                ).to(self.device)
                if obs in self.env.scales.keys():
                    noise_tensor /= self.env.scales[obs]
                noise_vec[obs_index : obs_index + obs_size] = noise_tensor
            obs_index += obs_size
        return noise_vec * torch.randn(
            self.env.num_envs, len(noise_vec), device=self.device
        )

    def get_noisy_obs(self, obs_list, noise_dict):
        observation = self.get_obs(obs_list)
        return observation + self.get_noise(obs_list, noise_dict)

    def get_obs(self, obs_list):
        observation = self.env.get_states(obs_list).to(self.device)
        return observation

    def set_actions(self, actions_list, actions, disable_actions=False):
        if disable_actions:
            return
        self.env.set_states(actions_list, actions)

    def get_timed_out(self):
        return self.env.get_states(["timed_out"]).to(self.device)

    def get_terminated(self):
        return self.env.get_states(["terminated"]).to(self.device)

    def get_obs_size(self, obs_list):
        return self.get_obs(obs_list)[0].shape[0]

    def get_action_size(self, action_list):
        return self.env.get_states(action_list)[0].shape[0]

    def get_rewards(self, reward_weights, modifier=1, mask=None):
        rewards_dict = {}
        if mask is None:
            mask = 1.0
        for name, weight in reward_weights.items():
            rewards_dict[name] = mask * self._get_reward({name: weight}, modifier)
        return rewards_dict

    def _get_reward(self, name_weight, modifier=1):
        return modifier * self.env.compute_reward(name_weight).to(self.device)
