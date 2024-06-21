import os
import torch
from tensordict import TensorDict

from learning.utils import Logger

from .BaseRunner import BaseRunner
from learning.modules import Actor, Critic
from learning.storage import ReplayBuffer
from learning.algorithms import GePPO

logger = Logger()
storage = ReplayBuffer()


class HybridPolicyRunner(BaseRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

        # TODO: Weights hardcoded for 4 policies
        self.num_old_policies = self.cfg["num_old_policies"]
        self.weights = torch.tensor([0.4, 0.3, 0.2, 0.1]).to(self.device)

    def _set_up_alg(self):
        alg_class = eval(self.cfg["algorithm_class_name"])
        if alg_class != GePPO:
            raise ValueError("HybridPolicyRunner only supports GePPO")

        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])
        # Store pik for the actor
        actor = Actor(num_actor_obs, num_actions, store_pik=True, **self.actor_cfg)
        critic = Critic(num_critic_obs, **self.critic_cfg)
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
                "dones": self.get_timed_out(),
            }
        )
        max_storage = self.env.num_envs * self.num_steps_per_env * self.num_old_policies
        storage.initialize(
            dummy_dict=transition,
            num_envs=self.env.num_envs,
            max_storage=max_storage,
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
                    actions = self.alg.act(actor_obs)
                    self.set_actions(
                        self.actor_cfg["actions"],
                        actions,
                        self.actor_cfg["disable_actions"],
                    )
                    # Store additional data for GePPO
                    log_prob = self.alg.actor.get_actions_log_prob(actions)
                    action_mean = self.alg.actor.action_mean
                    action_std = self.alg.actor.action_std

                    transition.update(
                        {
                            "actor_obs": actor_obs,
                            "actions": actions,
                            "critic_obs": critic_obs,
                            "log_prob": log_prob,
                            "action_mean": action_mean,
                            "action_std": action_std,
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
                    storage.add_transitions(transition)

                    logger.log_rewards(rewards_dict)
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(dones)
            logger.toc("collection")

            # Compute GePPO weights
            n_policies = min(self.it, self.num_old_policies)
            weights_active = self.weights[:n_policies]
            weights_active = weights_active * n_policies / weights_active.sum()
            idx_newest = (self.it - 1) % self.num_old_policies
            indices_all = [
                i % self.num_old_policies
                for i in range(idx_newest, idx_newest - n_policies, -1)
            ]
            weights_all = weights_active[indices_all]
            weights_all = weights_all.repeat_interleave(self.num_steps_per_env)
            weights_all = weights_all.unsqueeze(-1).repeat(1, self.env.num_envs)

            # Update GePPO with weights
            logger.tic("learning")
            self.alg.update(storage.get_data(), weights=weights_all)
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
            self.alg.critic.evaluate(critic_obs)
        self.env.reset()

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
            ["learning_rate", "mean_value_loss", "mean_surrogate_loss"],
        )
        logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])

        # GePPO specific logging
        logger.register_category(
            "GePPO",
            self.alg,
            ["eps", "tv", "adv_mean", "ret_mean", "adv_vtrace_mean", "ret_vtrace_mean"],
        )

        logger.attach_torch_obj_to_wandb((self.alg.actor, self.alg.critic))

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_state_dict": self.alg.critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(
                loaded_dict["critic_optimizer_state_dict"]
            )
        self.it = loaded_dict["iter"]

    def switch_to_eval(self):
        self.alg.actor.eval()
        self.alg.critic.eval()

    def get_inference_actions(self):
        obs = self.get_noisy_obs(self.actor_cfg["obs"], self.actor_cfg["noise"])
        return self.alg.actor.act_inference(obs)

    def export(self, path):
        self.alg.actor.export(path)
