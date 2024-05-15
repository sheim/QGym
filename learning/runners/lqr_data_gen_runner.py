import os
import time
import torch
from tensordict import TensorDict

from gym import LEGGED_GYM_ROOT_DIR
from learning.utils import Logger
from .custom_critic_runner import OnPolicyRunner
from learning.storage import DictStorage
from learning.utils import export_to_numpy

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
        self.total_steps = self.num_learning_iterations * self.num_steps_per_env

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

        dict_save_path = self.create_data_save_path_and_name()

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
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update(storage.data)
            torch.save(storage.data.cpu(), dict_save_path + str(self.it) + ".pt")
            export_to_numpy(storage.data, dict_save_path + str(self.it))
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

        print(f"Saved data from LQR Pendulum run to {dict_save_path}")

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

    def create_data_save_path_and_name(self):
        time_str = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "lqrc",
            "lqr_data",
            f"run_{time_str}",
            "iteration_",
        )
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return save_path
