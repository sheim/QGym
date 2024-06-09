import torch
from tensordict import TensorDict

from learning.utils import Logger
from .on_policy_runner import OnPolicyRunner
from learning.storage import DictStorage
from learning.algorithms import PPO2  # noqa F401
from learning.modules.actor import Actor
from learning.modules.lqrc import *  # noqa F401

logger = Logger()
storage = DictStorage()


class MyRunner(OnPolicyRunner):
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
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])
        actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)
        # critic = DenseSpectralLatent(num_critic_obs, **self.critic_cfg)
        critic = eval(self.critic_cfg["critic_class_name"])(
            num_critic_obs, **self.critic_cfg
        )
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

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
                "next_actor_obs": actor_obs,
                "actions": self.alg.act(actor_obs),
                "critic_obs": critic_obs,
                "next_critic_obs": critic_obs,
                "rewards": self.get_rewards({"termination": 0.0})["termination"],
                "timed_out": self.get_timed_out(),
                "terminated": self.get_terminated(),
                "dones": self.get_timed_out() | self.get_terminated(),
            }
        )
        storage.initialize(
            transition,
            self.env.num_envs,
            self.env.num_envs * self.num_steps_per_env,
            device=self.device,
        )

        # burn in observation normalization.
        self.burn_in_normalization()

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
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
                            "terminated": terminated,
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
        self.save()

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

    @torch.no_grad
    def burn_in_normalization(self, n_iterations=100):
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
            rewards = self.alg.critic.evaluate(critic_obs)
        self.alg.critic.value_offset.copy_(rewards.mean())
        print(f"Value offset: {self.alg.critic.value_offset.item()}")
        self.env.reset()
