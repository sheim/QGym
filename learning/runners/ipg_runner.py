import os
import torch
import torch.nn as nn
from tensordict import TensorDict

from learning.utils import Logger

from .BaseRunner import BaseRunner
from learning.modules import Actor, SmoothActor, Critic
from learning.storage import DictStorage, ReplayBuffer
from learning.algorithms import PPO_IPG

logger = Logger()
storage_onpol = DictStorage()
storage_offpol = ReplayBuffer()


class IPGRunner(BaseRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def _set_up_alg(self):
        alg_class = eval(self.cfg["algorithm_class_name"])
        if alg_class != PPO_IPG:
            raise ValueError("IPGRunner only supports PPO_IPG")

        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])
        if self.actor_cfg["smooth_exploration"]:
            actor = SmoothActor(num_actor_obs, num_actions, **self.actor_cfg)
        else:
            actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)
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

    def learn(self):
        self.set_up_logger()

        rewards_dict = {}

        self.alg.switch_to_train()
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        tot_iter = self.it + self.num_learning_iterations
        self.save()

        # * Initialize smooth exploration matrices
        if self.actor_cfg["smooth_exploration"]:
            self.alg.actor.sample_weights(batch_size=self.env.num_envs)

        # * start up both on- and off-policy storage
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
        storage_onpol.initialize(
            transition,
            self.env.num_envs,
            self.env.num_envs * self.num_steps_per_env,
            device=self.device,
        )
        storage_offpol.initialize(
            transition,
            self.env.num_envs,
            self.alg_cfg["storage_size"],
            device=self.device,
        )

        # burn in observation normalization.
        if self.actor_cfg["normalize_obs"] or self.critic_cfg["normalize_obs"]:
            self.burn_in_normalization()

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")
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
                    # add transition to both storages
                    storage_onpol.add_transitions(transition)
                    storage_offpol.add_transitions(transition)

                    logger.log_rewards(rewards_dict)
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(dones)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update(storage_onpol.data, storage_offpol.get_data())
            storage_onpol.clear()  # only clear on-policy storage
            logger.toc("learning")
            logger.log_all_categories()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.save()

    @torch.no_grad
    def burn_in_normalization(self, n_iterations=200):
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        for _ in range(n_iterations):
            actions = self.alg.act(actor_obs)
            self.set_actions(self.actor_cfg["actions"], actions)
            self.env.step()
            actor_obs = self.get_noisy_obs(
                self.actor_cfg["obs"], self.actor_cfg["noise"]
            )
            critic_obs = self.get_obs(self.critic_cfg["obs"])
            self.alg.critic_v.evaluate(critic_obs)
            q_input = torch.cat((critic_obs, actions), dim=-1)
            self.alg.critic_q.evaluate(q_input)
            self.alg.target_critic_q.evaluate(q_input)
        # self.env.reset()

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

    def set_up_logger(self):
        logger.register_rewards(list(self.critic_cfg["reward"]["weights"].keys()))
        logger.register_rewards(
            list(self.critic_cfg["reward"]["termination_weight"].keys())
        )
        logger.register_rewards(["total_rewards"])
        logger.register_category(
            "algorithm",
            self.alg,
            [
                "mean_value_loss",
                "mean_surrogate_loss",
                "learning_rate",
                # IPG specific
                "mean_q_loss",
                "mean_offpol_loss",
            ],
        )
        logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])

        logger.attach_torch_obj_to_wandb(
            (self.alg.actor, self.alg.critic_v, self.alg.critic_q)
        )

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_v_state_dict": self.alg.critic_v.state_dict(),
                "critic_q_state_dict": self.alg.critic_q.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "critic_v_opt_state_dict": self.alg.critic_v_optimizer.state_dict(),
                "critic_q_opt_state_dict": self.alg.critic_q_optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )

    def load(self, path, load_optimizer=True, load_actor_std=True):
        loaded_dict = torch.load(path)
        if load_actor_std:
            self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        else:
            std_init = self.alg.actor.std.detach().clone()
            self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
            self.alg.actor.std = nn.Parameter(std_init)
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
        self.it = loaded_dict["iter"]

    def switch_to_eval(self):
        self.alg.actor.eval()
        self.alg.critic_v.eval()
        self.alg.critic_q.eval()

    def get_inference_actions(self):
        obs = self.get_noisy_obs(self.actor_cfg["obs"], self.actor_cfg["noise"])
        return self.alg.actor.act_inference(obs)

    def export(self, path):
        self.alg.actor.export(path)
