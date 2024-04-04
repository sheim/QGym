import os
import numpy as np
import torch
from tensordict import TensorDict

from gym import LEGGED_GYM_ROOT_DIR
from learning.algorithms import *  # noqa: F403
from learning.modules import Actor, Critic
from learning.modules.lqrc import (
    Cholesky,
    CholeskyPlusConst,
    CholeskyOffset1,
    CholeskyOffset2
)
from learning.utils import Logger

from .on_policy_runner import OnPolicyRunner
from learning.storage import DictStorage


logger = Logger()
storage = DictStorage()


class CriticOnlyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def _set_up_alg(self):
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])
        critic_class_name = self.critic_cfg["critic_class_name"]
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)
        # critic = Critic(num_critic_obs, **self.critic_cfg)
        critic = eval(f"{critic_class_name}(num_critic_obs, **self.critic_cfg)")
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

    def learn(self):
        self.set_up_logger()

        rewards_dict = {}

        self.alg.switch_to_train()
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        tot_iter = self.it + self.num_learning_iterations
        self.all_obs = torch.zeros(self.env.num_envs * (tot_iter - self.it + 1), 2)
        self.all_obs[: self.env.num_envs, :] = actor_obs
        self.save()

        # * start up storage
        transition = TensorDict({}, batch_size=self.env.num_envs, device=self.device)
        transition.update(
            {
                "actor_obs": actor_obs,
                "actions": self.alg.act(actor_obs, critic_obs),
                "critic_obs": critic_obs,
                "rewards": self.get_rewards({"termination": 0.0})["termination"],
                "dones": self.get_timed_out(),
            }
        )
        storage.initialize(
            transition,
            self.env.num_envs,
            self.env.num_envs * self.num_steps_per_env,
            device=self.device,
        )

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
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
                    start = self.env.num_envs * self.it
                    end = self.env.num_envs * (self.it + 1)
                    self.all_obs[start:end, :] = actor_obs

                    # * get time_outs
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated

                    self.update_rewards(rewards_dict, terminated)
                    total_rewards = torch.stack(tuple(rewards_dict.values())).sum(dim=0)

                    transition.update(
                        {
                            "rewards": total_rewards,
                            "timed_out": timed_out,
                            "dones": dones,
                        }
                    )
                    storage.add_transitions(transition)

                    logger.log_rewards(rewards_dict)
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(dones)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update(storage.data)
            storage.clear()
            logger.toc("learning")
            logger.log_all_categories()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.all_obs = self.all_obs.detach().cpu().numpy()
        save_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "lqrc",
            "standard_training_data.npy"
            if self.critic_cfg["standard_critic_nn"]
            else "custom_training_data.npy",
        )

        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(save_path, self.all_obs)
        print(f"Saved training observations to {save_path}")
        self.save()

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_state_dict": self.alg.critic.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )

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
            "algorithm", self.alg, ["mean_value_loss", "mean_surrogate_loss"]
        )
        logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])

        logger.attach_torch_obj_to_wandb((self.alg.actor, self.alg.critic))
