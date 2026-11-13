import os
import torch
from tensordict import TensorDict

from learning.utils import Logger

from .BaseRunner import BaseRunner
from learning.storage import DictStorage
from learning.storage.transition_recorder import TransitionRecorder
from gym.utils.helpers import class_to_dict

from learning.offline.replay_fields import (
    NEXT_FEATURE_ATTRS,
    REWARD,
    STEP_FEATURE_ATTRS,
    next_feature_names,
    resolve_replay_fields,
    step_feature_names,
)

logger = Logger()
storage = DictStorage()


class OnPolicyRunner(BaseRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        self.num_steps_per_env = max(1, self.alg_cfg["rollout_size"] // env.num_envs)
        self.recorder = None
        self._skip_next_replay = False
        print(
            f"[OnPolicyRunner] num_steps_per_env={self.num_steps_per_env}"
            f" (batch_size={self.alg_cfg['batch_size']}, num_envs={env.num_envs})"
        )

    def _make_recorder(self, n_policy_steps):
        replay_cfg = self.cfg.get("replay")
        if replay_cfg is None or not replay_cfg["enabled"]:
            return None
        output_dir = replay_cfg["output_dir"]
        if not output_dir:
            if self.log_dir is None:
                raise ValueError(
                    "replay output_dir is required when log_dir is unavailable"
                )
            output_dir = os.path.join(self.log_dir, "replay")
        obs_dims = tuple(self.get_obs_size([name]) for name in self.actor_cfg["obs"])
        action_dims = tuple(
            self.get_action_size([name]) for name in self.actor_cfg["actions"]
        )
        # "fields" is an optional selection; an older saved runner config that
        # predates selectable features records the core transition fields only.
        extra_fields = resolve_replay_fields(replay_cfg.get("fields", ()))
        reward_term_names = tuple(
            self.critic_cfg["reward"]["termination_weight"]
        ) + tuple(self.critic_cfg["reward"]["weights"])
        # Index-to-name maps: raw state rows are ordered by the robot layout, so
        # a dataset reader needs the names to interpret body/joint columns.
        field_metadata = {}
        if extra_fields:
            field_metadata["body_names"] = list(self.env.robot_layout.body_names)
            field_metadata["dof_names"] = list(self.env.actuated_dof_names)
        return TransitionRecorder(
            num_envs=self.env.num_envs,
            obs_names=tuple(self.actor_cfg["obs"]),
            obs_dims=obs_dims,
            action_names=tuple(self.actor_cfg["actions"]),
            action_dims=action_dims,
            noise=self.actor_cfg["noise"],
            scales=self.env.scales,
            device=self.device,
            output_dir=output_dir,
            capacity=int(replay_cfg["capacity"]),
            snapshot_interval=int(replay_cfg["snapshot_interval"]),
            seed=int(replay_cfg["seed"]),
            task_name=str(replay_cfg["task_name"]),
            backend=str(replay_cfg["backend"]),
            source_revision=str(replay_cfg["source_revision"]),
            log_dir=self.log_dir,
            actor_cfg=self.actor_cfg,
            env_cfg=class_to_dict(self.env.cfg),
            runner_cfg=self.cfg,
            episode_length_s=float(self.env.max_episode_length_s),
            n_policy_steps=n_policy_steps,
            control_dt=float(self.env.dt),
            extra_fields=extra_fields,
            reward_term_names=(reward_term_names if REWARD in extra_fields else ()),
            field_metadata=field_metadata,
        )

    def _replay_features(self, names, *, rewards_dict=None):
        """Clone selected raw states at one time point.

        Environment buffers are reused in place, so the clone must happen before
        the next physics step or reset overwrites the source.
        """
        return {
            name: self._replay_feature(name, rewards_dict).clone() for name in names
        }

    def _replay_feature(self, name, rewards_dict):
        if name in ("critic_obs", "next_critic_obs"):
            return self.get_obs(self.critic_cfg["obs"])
        if name == REWARD:
            if rewards_dict is None:
                raise ValueError("reward_terms requires the iteration reward dict")
            # Same order as initialize_rewards_dict: termination terms first.
            return torch.stack(tuple(rewards_dict.sum(dim=0).values()), dim=1)
        if name in STEP_FEATURE_ATTRS:
            attribute = STEP_FEATURE_ATTRS[name]
        elif name in NEXT_FEATURE_ATTRS:
            attribute = NEXT_FEATURE_ATTRS[name]
        else:
            raise ValueError(f"unsupported replay feature {name!r}")
        if not hasattr(self.env, attribute):
            raise AttributeError(
                f"replay field {name!r} needs env attribute {attribute!r}, which "
                f"{type(self.env).__name__} does not provide"
            )
        value = getattr(self.env, attribute)
        if attribute == "_rigid_body_state":
            # The backend stores per-body rows flattened across environments;
            # restore one row per environment for the [N, width] contract.
            return value.view(self.env.num_envs, -1)
        return value

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
        burn_in_ran = False
        if self.actor_cfg["normalize_obs"] or self.critic_cfg["normalize_obs"]:
            self.burn_in_normalization()
            burn_in_ran = True

        self.recorder = self._make_recorder(n_policy_steps)
        # burn_in_normalization resets the environments, so actor_obs above is
        # stale for the first post-burn-in action. Exclude that transition from
        # replay instead of changing PPO's established behavior.
        self._skip_next_replay = burn_in_ran

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
                    policy_actor_obs = actor_obs
                    actions = self.alg.act(actor_obs)
                    policy_actions = actions
                    if self.recorder is not None:
                        pre_action_features = self._replay_features(
                            step_feature_names(self.recorder.extra_fields)
                        )
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

                    if self.recorder is not None:
                        pre_reset_clean_obs = self.get_obs(self.actor_cfg["obs"])
                        pre_reset_height = self.env.base_height.clone()
                        terminated = self.get_terminated()
                        timed_out = self.get_timed_out()
                        post_action_features = self._replay_features(
                            next_feature_names(self.recorder.extra_fields),
                            rewards_dict=rewards_dict,
                        )

                    self.reset_envs()

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

                    if self.recorder is not None:
                        if self._skip_next_replay:
                            self._skip_next_replay = False
                        else:
                            self.recorder.append(
                                actor_obs=policy_actor_obs,
                                action=policy_actions,
                                post_reset_obs=actor_obs,
                                pre_reset_clean_obs=pre_reset_clean_obs,
                                successor_height_m=pre_reset_height,
                                terminated=terminated,
                                timed_out=timed_out,
                                behavior_iteration=self.it,
                                extra_features={
                                    **pre_action_features,
                                    **post_action_features,
                                },
                            )

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

            if (
                self.recorder is not None
                and self.recorder.snapshot_interval > 0
                and self.it % self.recorder.snapshot_interval == 0
            ):
                self.recorder.snapshot(self.it)

        if self.recorder is not None and self.recorder.valid_count > 0:
            self.recorder.snapshot(self.it)
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
            self.env.num_envs,
            dt,
            self.cfg["max_iterations"],
            self.device,
            log_dir=self.log_dir,
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

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        temporary = path + ".tmp"
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_state_dict": self.alg.critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "iter": self.it,
            },
            temporary,
        )
        os.replace(temporary, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True, map_location=self.device)
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
        # Inference is deterministic: observation noise is a training-time
        # augmentation, and device-specific RNG streams otherwise make the
        # same checkpoint take different actions on CPU and GPU.
        obs = self.get_obs(self.actor_cfg["obs"])
        return self.alg.actor.act_inference(obs)

    def export(self, path):
        self.alg.actor.export(path)

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
