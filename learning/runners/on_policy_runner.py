import os
import torch
from tensordict import TensorDict

from learning.utils import Logger

from .BaseRunner import BaseRunner
from learning.storage import DictStorage

logger = Logger()
storage = DictStorage()


class OnPolicyRunner(BaseRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def learn(self, states_to_log_dict=None):
        self.set_up_logger()

        rewards_dict = {}

        self.alg.switch_to_train()
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        tot_iter = self.it + self.num_learning_iterations
        self.save()

        # * Initialize smooth exploration matrices
        if self.actor_cfg["exploration"]["type"] == "smooth":
            self.alg.actor.sample_weights(batch_size=self.env.num_envs)

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

        # burn in observation normalization.
        with torch.inference_mode():
            for _ in range(200):
                actions = self.alg.act(actor_obs, critic_obs)
                self.set_actions(
                    self.actor_cfg["actions"],
                    actions,
                    self.actor_cfg["disable_actions"],
                )
                self.env.step()
                actor_obs = self.get_noisy_obs(
                    self.actor_cfg["obs"], self.actor_cfg["noise"]
                )

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")

            # * Simulate environment and log states
            if states_to_log_dict is not None:
                it_idx = self.it - 1
                if it_idx % 50 == 0:
                    self.sim_and_log_states(states_to_log_dict, it_idx)

            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # * Re-sample noise matrix for smooth exploration
                    sample_freq = self.actor_cfg["exploration"]["sample_freq"]
                    if (
                        self.actor_cfg["exploration"]["type"] == "smooth"
                        and i % sample_freq == 0
                    ):
                        self.alg.actor.sample_weights(batch_size=self.env.num_envs)

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
            ["mean_value_loss", "mean_surrogate_loss", "learning_rate"],
        )
        logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])

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

    def sim_and_log_states(self, states_to_log_dict, it_idx):
        # Simulate environment for as many steps as expected in the dict.
        # Log states to the dict, as well as whether the env terminated.
        steps = states_to_log_dict["terminated"].shape[2]
        actor_obs = self.get_obs(self.actor_cfg["obs"])
        critic_obs = self.get_obs(self.critic_cfg["obs"])

        with torch.inference_mode():
            for i in range(steps):
                sample_freq = self.actor_cfg["exploration"]["sample_freq"]
                if (
                    self.actor_cfg["exploration"]["type"] == "smooth"
                    and i % sample_freq == 0
                ):
                    self.alg.actor.sample_weights(batch_size=self.env.num_envs)

                actions = self.alg.act(actor_obs, critic_obs)
                self.set_actions(
                    self.actor_cfg["actions"],
                    actions,
                    self.actor_cfg["disable_actions"],
                )

                self.env.step()

                actor_obs = self.get_noisy_obs(
                    self.actor_cfg["obs"], self.actor_cfg["noise"]
                )
                critic_obs = self.get_obs(self.critic_cfg["obs"])

                # Log states (just for the first env)
                terminated = self.get_terminated()[0]
                for state in states_to_log_dict:
                    if state == "terminated":
                        states_to_log_dict[state][0, it_idx, i, :] = terminated
                    else:
                        states_to_log_dict[state][0, it_idx, i, :] = getattr(
                            self.env, state
                        )[0, :]
