import os
import torch
from tensordict import TensorDict

from learning.utils import Logger

from .BaseRunner import BaseRunner
from learning.modules import Critic, ChimeraActor
from learning.storage import ReplayBuffer
from learning.algorithms import SAC

logger = Logger()
storage = ReplayBuffer()


class OffPolicyRunner(BaseRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)

    def _set_up_alg(self):
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])
        actor = ChimeraActor(num_actor_obs, num_actions, **self.actor_cfg)
        critic_1 = Critic(num_critic_obs + num_actions, **self.critic_cfg)
        critic_2 = Critic(num_critic_obs + num_actions, **self.critic_cfg)
        target_critic_1 = Critic(num_critic_obs + num_actions, **self.critic_cfg)
        target_critic_2 = Critic(num_critic_obs + num_actions, **self.critic_cfg)

        print(actor)

        self.alg = SAC(
            actor,
            critic_1,
            critic_2,
            target_critic_1,
            target_critic_2,
            device=self.device,
            **self.alg_cfg,
        )

    def learn(self):
        n_policy_steps = int((1 / self.env.dt) / self.actor_cfg["frequency"])
        assert n_policy_steps > 0, "actor frequency should be less than ctrl_freq"
        self.set_up_logger(dt=self.env.dt * n_policy_steps)

        rewards_dict = self.initialize_rewards_dict(n_policy_steps)

        self.alg.switch_to_train()
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        actions = self.alg.act(actor_obs)
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
                "dones": self.get_timed_out(),
            }
        )
        storage.initialize(
            transition,
            self.env.num_envs,
            self.alg_cfg["storage_size"],
            device=self.device,
        )

        # fill buffer
        for _ in range(self.alg_cfg["initial_fill"]):
            with torch.inference_mode():
                actions = torch.rand_like(actions) * 2 - 1
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

                for step in range(n_policy_steps):
                    self.env.step()
                    # put reward integration here
                    self.update_rewards_dict(rewards_dict, step)
                else:
                    # catch and reset failed envs
                    to_be_reset = self.env.timed_out | self.env.terminated
                    env_ids = (to_be_reset).nonzero(as_tuple=False).flatten()
                    self.env._reset_idx(env_ids)

                total_rewards = torch.stack(
                    tuple(rewards_dict.sum(dim=0).values())
                ).sum(dim=(0))

                actor_obs = self.get_noisy_obs(
                    self.actor_cfg["obs"], self.actor_cfg["noise"]
                )
                critic_obs = self.get_obs(self.critic_cfg["obs"])

                transition.update(
                    {
                        "next_actor_obs": actor_obs,
                        "next_critic_obs": critic_obs,
                        "rewards": total_rewards,
                        "timed_out": self.env.timed_out,
                        "dones": self.env.timed_out | self.env.terminated,
                    }
                )
                storage.add_transitions(transition)
                # print every 10% of initial fill
                if (self.alg_cfg["initial_fill"] > 10) and (
                    _ % (self.alg_cfg["initial_fill"] // 10) == 0
                ):
                    print(f"Filled {100 * _ / self.alg_cfg['initial_fill']}%")

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

                    for step in range(n_policy_steps):
                        self.env.step()
                        # put reward integration here
                        self.update_rewards_dict(rewards_dict, step)
                    else:
                        # catch and reset failed envs
                        to_be_reset = self.env.timed_out | self.env.terminated
                        env_ids = (to_be_reset).nonzero(as_tuple=False).flatten()
                        self.env._reset_idx(env_ids)

                    total_rewards = torch.stack(
                        tuple(rewards_dict.sum(dim=0).values())
                    ).sum(dim=(0))

                    actor_obs = self.get_noisy_obs(
                        self.actor_cfg["obs"], self.actor_cfg["noise"]
                    )
                    critic_obs = self.get_obs(self.critic_cfg["obs"])

                    transition.update(
                        {
                            "next_actor_obs": actor_obs,
                            "next_critic_obs": critic_obs,
                            "rewards": total_rewards,
                            "timed_out": self.env.timed_out,
                            "dones": self.env.timed_out | self.env.terminated,
                        }
                    )
                    storage.add_transitions(transition)

                    logger.log_rewards(rewards_dict.sum(dim=0))
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(self.env.timed_out | self.env.terminated)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update(storage.get_data())
            logger.toc("learning")
            logger.log_all_categories()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.save()

    def update_rewards_dict(self, rewards_dict, step):
        # sum existing rewards with new rewards
        rewards_dict[step].update(
            self.get_rewards(
                self.critic_cfg["reward"]["termination_weight"],
                modifier=self.env.dt,
                mask=self.env.terminated,
            ),
            inplace=True,
        )
        rewards_dict[step].update(
            self.get_rewards(
                self.critic_cfg["reward"]["weights"],
                modifier=self.env.dt,
                mask=~self.env.terminated,
            ),
            inplace=True,
        )

    def initialize_rewards_dict(self, n_steps):
        # sum existing rewards with new rewards
        rewards_dict = TensorDict(
            {}, batch_size=(n_steps, self.env.num_envs), device=self.device
        )
        for key in self.critic_cfg["reward"]["termination_weight"]:
            rewards_dict.update(
                {key: torch.zeros(n_steps, self.env.num_envs, device=self.device)}
            )
        for key in self.critic_cfg["reward"]["weights"]:
            rewards_dict.update(
                {key: torch.zeros(n_steps, self.env.num_envs, device=self.device)}
            )
        return rewards_dict

    def set_up_logger(self, dt=None):
        if dt is None:
            dt = self.env.dt
        logger.initialize(
            self.env.num_envs, dt, self.cfg["max_iterations"], self.device
        )

        logger.register_rewards(list(self.critic_cfg["reward"]["weights"].keys()))
        logger.register_rewards(
            list(self.critic_cfg["reward"]["termination_weight"].keys())
        )
        logger.register_rewards(["total_rewards"])
        logger.register_category(
            "algorithm",
            self.alg,
            [
                "mean_critic_1_loss",
                "mean_critic_2_loss",
                "mean_actor_loss",
                "mean_alpha_loss",
                "alpha",
                "alpha_lr",
                "critic_1_lr",
                "critic_2_lr",
                "actor_lr",
            ],
        )
        # logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])

        logger.attach_torch_obj_to_wandb(
            (self.alg.actor, self.alg.critic_1, self.alg.critic_2)
        )

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        save_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_1_state_dict": self.alg.critic_1.state_dict(),
            "critic_2_state_dict": self.alg.critic_2.state_dict(),
            "log_alpha": self.alg.log_alpha,
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_1_optimizer_state_dict": self.alg.critic_1_optimizer.state_dict(),
            "critic_2_optimizer_state_dict": self.alg.critic_2_optimizer.state_dict(),
            "log_alpha_optimizer_state_dict": self.alg.log_alpha_optimizer.state_dict(),
            "iter": self.it,
        }
        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.critic_1.load_state_dict(loaded_dict["critic_1_state_dict"])
        self.alg.critic_2.load_state_dict(loaded_dict["critic_2_state_dict"])
        self.log_alpha = loaded_dict["log_alpha"]
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(
                loaded_dict["actor_optimizer_state_dict"]
            )
            self.alg.critic_1_optimizer.load_state_dict(
                loaded_dict["critic_1_optimizer_state_dict"]
            )
            self.alg.critic_2_optimizer.load_state_dict(
                loaded_dict["critic_2_optimizer_state_dict"]
            )
            self.alg.log_alpha_optimizer.load_state_dict(
                loaded_dict["log_alpha_optimizer_state_dict"]
            )
        self.it = loaded_dict["iter"]

    def switch_to_eval(self):
        self.alg.actor.eval()
        self.alg.critic_1.eval()
        self.alg.critic_2.eval()

    def get_inference_actions(self):
        obs = self.get_noisy_obs(self.actor_cfg["obs"], self.actor_cfg["noise"])
        mean = self.alg.actor.forward(obs)
        actions = torch.tanh(mean)
        actions = (actions * self.alg.action_delta + self.alg.action_offset).clamp(
            self.alg.action_min, self.alg.action_max
        )
        return actions

    def export(self, path):
        self.alg.actor.export(path)
