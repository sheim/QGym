import torch
from tensordict import TensorDict

from learning.utils import Logger
from .on_policy_runner import OnPolicyRunner
from learning.storage import DictStorage
from learning.algorithms import PPO2  # noqa F401
from learning.modules.actor import Actor
from learning.modules.critic import Critic  # noqa F401
from learning.modules.QRCritics import *  # noqa F401

logger = Logger()
storage = DictStorage()


class MyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)

    def _set_up_alg(self):
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])  # noqa: F841
        actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)
        critic_class_name = self.critic_cfg["critic_class_name"]
        critic = eval(f"{critic_class_name}(num_critic_obs, **self.critic_cfg)")
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

    def learn(self, states_to_log_dict=None):
        n_policy_steps = int((1 / self.env.dt) / self.actor_cfg["frequency"])
        assert n_policy_steps > 0, "actor frequency should be less than ctrl_freq"
        self.set_up_logger(dt=self.env.dt * n_policy_steps)

        rewards_dict = self.initialize_rewards_dict(n_policy_steps)

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

            # * Simulate environment and log states
            if states_to_log_dict is not None:
                it_idx = self.it - 1
                if it_idx % 10 == 0:
                    self.sim_and_log_states(states_to_log_dict, it_idx)

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
            ["mean_value_loss", "mean_surrogate_loss", "learning_rate"],
        )
        logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])
        logger.attach_torch_obj_to_wandb((self.alg.actor, self.alg.critic))

    def update_rewards(self, rewards_dict, terminated):
        # sum existing rewards with new rewards

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

    def sim_and_log_states(self, states_to_log_dict, it_idx):
        # Simulate environment for as many steps as expected in the dict.
        # Log states to the dict, as well as whether the env terminated.
        steps = states_to_log_dict["terminated"].shape[2]
        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])

        with torch.inference_mode():
            for i in range(steps):
                actions = self.alg.act(actor_obs)
                self.set_actions(
                    self.policy_cfg["actions"],
                    actions,
                    self.policy_cfg["disable_actions"],
                )

                self.env.step()

                actor_obs = self.get_noisy_obs(
                    self.policy_cfg["actor_obs"], self.policy_cfg["noise"]
                )

                # Log states (just for the first env)
                terminated = self.get_terminated()[0]
                for state in states_to_log_dict:
                    if state == "terminated":
                        states_to_log_dict[state][0, it_idx, i, :] = terminated
                    else:
                        states_to_log_dict[state][0, it_idx, i, :] = getattr(
                            self.env, state
                        )[0, :]
