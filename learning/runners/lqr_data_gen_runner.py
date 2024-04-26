import os
import time
import torch
import numpy as np
from tensordict import TensorDict

from gym import LEGGED_GYM_ROOT_DIR
from learning.utils import Logger
from .custom_critic_runner import OnPolicyRunner
from learning.storage import DictStorage

logger = Logger()
storage = DictStorage()


class LQRDataGenRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )
        self.total_steps = self.num_learning_iterations*self.num_steps_per_env
        self.data_dict = {"observations": np.zeros((self.total_steps, env.num_envs, env.dof_state.shape[0])),
                          "actions": np.zeros((self.total_steps, env.num_envs, env.num_actuators)),
                          "total_rewards": np.zeros((self.total_steps, env.num_envs, 1))}

    def learn(self):
        self.set_up_logger()

        rewards_dict = {}

        self.alg.switch_to_train()
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        tot_iter = self.it + self.num_learning_iterations
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
                    self.data_dict["observations"][self.it*i, ...] = actor_obs
                    self.data_dict["actions"][self.it*i, ...] = self.torques
                    self.data_dict["total_rewards"][self.it*i, ...] = total_rewards
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
        self.save()

        time_str = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "lqrc", "lqr_data", f"lqr_{time_str}.npz")
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.savez(save_path, *self.data_dict)
        print(f"Saved data from LQR Pendulum run to {save_path}")
