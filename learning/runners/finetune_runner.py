import torch
from learning.algorithms import *  # noqa: F403
from learning.modules import Actor, SmoothActor, Critic
from learning.utils import remove_zero_weighted_rewards


class FineTuneRunner:
    def __init__(self, train_cfg, data_dict, exploration_scale=1.0, device="cpu"):
        self.parse_train_cfg(train_cfg)
        self.data_dict = data_dict
        self.exploration_scale = exploration_scale
        self.device = device

    def _set_up_alg(self):
        num_actor_obs = self.data_dict["actor_obs"].shape[-1]
        num_actions = self.data_dict["actions"].shape[-1]
        num_critic_obs = self.data_dict["critic_obs"].shape[-1]
        if self.actor_cfg["smooth_exploration"]:
            actor = SmoothActor(
                num_obs=num_actor_obs,
                num_actions=num_actions,
                exploration_scale=self.exploration_scale,
                **self.actor_cfg,
            )
        else:
            actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)

        alg_class_name = self.cfg["algorithm_class_name"]
        alg_class = eval(alg_class_name)

        if alg_class_name == "PPO_IPG":
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

    def parse_train_cfg(self, train_cfg):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        remove_zero_weighted_rewards(train_cfg["critic"]["reward"]["weights"])
        self.actor_cfg = train_cfg["actor"]
        self.critic_cfg = train_cfg["critic"]

    def learn(self):
        # Single update on data dict
        if self.cfg["algorithm_class_name"] == "PPO_IPG":
            # TODO: How to handle off-policy data
            self.alg.update(self.data_dict, self.data_dict)
        else:
            self.alg.update(self.data_dict)

    def save(self, path):
        if self.cfg["algorithm_class_name"] == "PPO_IPG":
            torch.save(
                {
                    "actor_state_dict": self.alg.actor.state_dict(),
                    "critic_v_state_dict": self.alg.critic_v.state_dict(),
                    "critic_q_state_dict": self.alg.critic_q.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "critic_v_opt_state_dict": self.alg.critic_v_optimizer.state_dict(),
                    "critic_q_opt_state_dict": self.alg.critic_q_optimizer.state_dict(),
                    "iter": 1,  # only one iteration
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
                "iter": 1,  # only one iteration
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])

        if self.cfg["algorithm_class_name"] == "PPO_IPG":
            self.alg.critic_v.load_state_dict(loaded_dict["critic_v_state_dict"])
            self.alg.critic_q.load_state_dict(loaded_dict["critic_q_state_dict"])
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
