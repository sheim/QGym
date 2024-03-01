import torch
import numpy as np
import os
from learning import LEGGED_GYM_LQRC_DIR

from learning.utils import Logger

from .on_policy_runner import OnPolicyRunner

logger = Logger()


class DataLoggingRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def learn(self):
        self.set_up_logger()

        rewards_dict = {}

        self.alg.actor_critic.train()
        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
        tot_iter = self.it + self.num_learning_iterations
        self.all_obs = torch.zeros(self.env.num_envs * (tot_iter - self.it + 1), 2)
        self.all_obs[: self.env.num_envs, :] = actor_obs

        self.save()

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)
                    self.set_actions(
                        self.policy_cfg["actions"],
                        actions,
                        self.policy_cfg["disable_actions"],
                    )

                    self.env.step()

                    actor_obs = self.get_noisy_obs(
                        self.policy_cfg["actor_obs"], self.policy_cfg["noise"]
                    )
                    critic_obs = self.get_obs(self.policy_cfg["critic_obs"])

                    start = self.env.num_envs * self.it
                    end = self.env.num_envs * (self.it + 1)
                    self.all_obs[start:end, :] = actor_obs

                    # * get time_outs
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated

                    self.update_rewards(rewards_dict, terminated)
                    total_rewards = torch.stack(tuple(rewards_dict.values())).sum(dim=0)

                    logger.log_rewards(rewards_dict)
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(dones)

                    self.alg.process_env_step(total_rewards, dones, timed_out)
                self.alg.compute_returns(critic_obs)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update()
            logger.toc("learning")
            logger.log_category()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.all_obs = self.all_obs.detach().cpu().numpy()
        save_path = os.path.join(
            LEGGED_GYM_LQRC_DIR,
            "logs",
            "standard_training_data.npy"
            if self.policy_cfg["standard_critic_nn"]
            else "custom_training_data.npy",
        )

        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(save_path, self.all_obs)
        print(f"Saved training observations to {save_path}")

        self.save()

    def update_rewards(self, rewards_dict, terminated):
        rewards_dict.update(
            self.get_rewards(
                self.policy_cfg["reward"]["termination_weight"], mask=terminated
            )
        )
        rewards_dict.update(
            self.get_rewards(
                self.policy_cfg["reward"]["weights"],
                modifier=self.env.dt,
                mask=~terminated,
            )
        )

    def set_up_logger(self):
        logger.register_rewards(list(self.policy_cfg["reward"]["weights"].keys()))
        logger.register_rewards(
            list(self.policy_cfg["reward"]["termination_weight"].keys())
        )
        logger.register_rewards(["total_rewards"])
        logger.register_category(
            "algorithm", self.alg, ["mean_value_loss", "mean_surrogate_loss"]
        )

        logger.attach_torch_obj_to_wandb(
            (self.alg.actor_critic.actor, self.alg.actor_critic.critic)
        )
