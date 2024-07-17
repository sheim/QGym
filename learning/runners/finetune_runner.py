import torch
from tensordict import TensorDict
from learning.algorithms import *  # noqa: F403
from learning.modules import Actor, SmoothActor, Critic
from learning.utils import remove_zero_weighted_rewards
from learning.storage import DictStorage

from .BaseRunner import BaseRunner

sim_storage = DictStorage()


class FineTuneRunner(BaseRunner):
    def __init__(
        self,
        env,
        train_cfg,
        data_onpol,
        data_offpol=None,
        exploration_scale=1.0,
        device="cpu",
    ):
        self.env = env
        self.parse_train_cfg(train_cfg)
        self.data_onpol = data_onpol
        self.data_offpol = data_offpol
        self.exploration_scale = exploration_scale
        self.device = device

    def _set_up_alg(self):
        num_actor_obs = self.data_onpol["actor_obs"].shape[-1]
        num_actions = self.data_onpol["actions"].shape[-1]
        num_critic_obs = self.data_onpol["critic_obs"].shape[-1]
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
        self.alg.switch_to_train()

        if self.env is not None:
            # Simulate 1 env, same number of steps as in data
            num_steps = self.data_onpol.shape[0]
            sim_data = self.get_sim_data(num_steps)
            # Concatenate data dict wtih sim data
            self.data_onpol = TensorDict(
                {
                    name: torch.cat((self.data_onpol[name], sim_data[name]), dim=1)
                    for name in self.data_onpol.keys()
                },
                batch_size=(num_steps, 2),
            )

        # Single alg update on data
        if self.data_offpol is None:
            self.alg.update(self.data_onpol)
        else:
            self.alg.update(self.data_onpol, self.data_offpol)

    def get_sim_data(self, num_steps):
        rewards_dict = {}
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])

        # * Initialize smooth exploration matrices
        if self.actor_cfg["smooth_exploration"]:
            self.alg.actor.sample_weights(batch_size=1)

        # * Start up storage
        transition = TensorDict({}, batch_size=1, device=self.device)
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
            num_envs=1,
            max_storage=num_steps,
            device=self.device,
        )

        # * Rollout
        with torch.inference_mode():
            for i in range(num_steps):
                # * Re-sample noise matrix for smooth exploration
                sample_freq = self.actor_cfg["exploration_sample_freq"]
                if self.actor_cfg["smooth_exploration"] and i % sample_freq == 0:
                    self.alg.actor.sample_weights(batch_size=1)

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
