"""Disposable prediction adapter over Q2's MuJoCo Warp Go2Trot task.

A ``PredictionTask`` is an independent Q2 ``Go2Trot`` instance whose Warp world
count equals the number of candidates. Every candidate world is a branch of one
recorded evaluation state: physics, action scaling, PD control, torque clipping,
the gait phase clock and the actor-observation construction are Q2's own code.

Everything in the hot path is a device-to-device tensor write followed by
``env.step()``; no value is ever read back to the host.
"""

from __future__ import annotations

import copy

import mujoco_warp as mjw
import torch
import warp as wp

from gym.utils.task_registry import task_registry

import gym.envs  # noqa: F401  (fail-fast task registration)

TASK = "go2trot"

# Q2 task-owned buffers that define one controller step boundary. Physics state
# (dof_pos/dof_vel/root_states) is committed through backend.reset_state; the
# rest are task buffers consumed by step() and observation construction.
STATE_FIELDS = (
    "dof_pos",
    "dof_vel",
    "root_states",
    "dof_pos_target",
    "dof_pos_history",
    "commands",
    "phase",
    "phase_frequency",
    "episode_length_buf",
    "p_gains",
    "d_gains",
)


def build_cfgs(num_envs: int, seed: int = 0):
    """Deep-copy the registered go2trot configs and resolve derived timings."""
    env_cfg, train_cfg = task_registry.get_cfgs(TASK)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)
    env_cfg.env.num_envs = int(num_envs)
    env_cfg.seed = int(seed)
    train_cfg.seed = int(seed)
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return env_cfg, train_cfg


def make_env(num_envs: int, seed: int = 0, device: str = "cuda:0"):
    env_cfg, train_cfg = build_cfgs(num_envs, seed)
    env = task_registry.make_env(TASK, env_cfg, device=device, headless=True)
    return env, env_cfg, train_cfg


def disable_stochastic_task_evolution(env):
    """Freeze commands and disturbances for prediction / fixed-command control.

    Both are instance-local overrides: the environment config keeps its
    registered values, so the artifact config fingerprint still validates.
    Command resampling and pushes draw from the global CUDA RNG, which a
    counterfactual branch cannot know, so a deterministic predictor must be told
    to hold them fixed instead of inventing a future.
    """
    env._resample_commands = lambda _mask: None
    env._push_robots = lambda: None


def capture_state(env) -> dict:
    """Clone the task buffers that determine the next controller step.

    The constraint/acceleration warm-start arrays are included because they are
    the only solver history that survives a step. Without them, candidate
    worlds would start from whatever leftovers the previous decision left
    behind, so identical candidates would not score identically.
    """
    state = {name: getattr(env, name).clone() for name in STATE_FIELDS}
    state["common_step_counter"] = int(env.common_step_counter)
    data = env._backend._d
    state["qacc_warmstart"] = wp.to_torch(data.qacc_warmstart).clone()
    constraint = wp.to_torch(data.cqacc_warmstart)
    if constraint.numel():
        state["cqacc_warmstart"] = constraint.clone()
    return state


class PredictionTask:
    """One batched Q2 Go2Trot world set used only for counterfactual branches."""

    def __init__(self, num_envs: int, seed: int = 0, device: str = "cuda:0"):
        self.num_envs = int(num_envs)
        self.device = device
        self.env, self.env_cfg, self.train_cfg = make_env(num_envs, seed, device)
        disable_stochastic_task_evolution(self.env)
        self.obs_names = tuple(self.train_cfg.actor.obs)
        self.action_names = tuple(self.train_cfg.actor.actions)
        # Policy action units -> radians of joint-position offset.
        self.action_scale = self.env.scales[self.action_names[0]].clone()
        self.backend = self.env._backend
        self._all = torch.ones(self.num_envs, dtype=torch.bool, device=device)

    # ── physical identity ───────────────────────────────────────────────────
    def copy_physics_from(self, source: "PredictionTask") -> None:
        """Copy per-world mass/inertia/friction so candidates share one robot.

        Startup domain randomisation is sampled per environment, so a predictor
        with a different world count necessarily holds different physical
        parameters. They are model tensors: rewriting them needs one
        ``set_const`` to rebuild subtreemass and inverse-inertia constants.
        """
        src = source.backend
        dst = self.backend
        if src._randomize_link_mass != dst._randomize_link_mass:
            raise RuntimeError("link-mass randomisation flags differ")
        if src._randomize_contact_friction != dst._randomize_contact_friction:
            raise RuntimeError("contact-friction randomisation flags differ")
        dst._link_mass_t.copy_(src._link_mass_t)
        dst._link_inertia_t.copy_(src._link_inertia_t)
        dst._body_mass_native_t.copy_(src._body_mass_native_t)
        dst._body_inertia_native_t.copy_(src._body_inertia_native_t)
        dst._contact_friction_t.copy_(src._contact_friction_t)
        if dst._geom_friction_t is not None:
            dst._geom_friction_t.copy_(src._geom_friction_t)
        if dst._pair_friction_t is not None:
            dst._pair_friction_t.copy_(src._pair_friction_t)
        with dst._wp_ctx:
            mjw.set_const(dst._m, dst._d)
        dst._sync_assembled_states()

    # ── state branching ─────────────────────────────────────────────────────
    def scatter(self, state: dict, sync_free: bool = False) -> None:
        """Broadcast one captured state into every candidate world.

        ``sync_free`` commits an all-worlds state write with plain ``copy_``
        instead of ``SimBackend.reset_state``. ``reset_state`` writes a
        permuted quaternion through ``torch.where(..., out=<strided view>)``,
        which makes PyTorch synchronise the stream on every call; that is
        invisible in normal training and only matters for a controller that
        must stay resident. Both paths commit the same state (verified in
        ``verify_parity.py``).
        """
        env = self.env
        for name in STATE_FIELDS:
            getattr(env, name).copy_(state[name])
        env.common_step_counter = state["common_step_counter"]
        data = self.backend._d
        wp.to_torch(data.qacc_warmstart).copy_(state["qacc_warmstart"])
        if "cqacc_warmstart" in state:
            constraint = wp.to_torch(data.cqacc_warmstart)
            if constraint.numel():
                constraint.copy_(state["cqacc_warmstart"])
        if sync_free:
            self._commit_all_worlds()
        else:
            # One atomic write-then-commit, exactly like a task reset:
            # reset_state clamps joint positions, scatters canonical writes into
            # native qpos/qvel, and runs a single forward.
            self.backend.reset_state(self._all)
        env._refresh_base_observations(self._all)
        env.dof_pos_obs[:] = env.dof_pos - env.default_dof_pos
        env._update_phase_observation()
        env._update_gait_reference()
        env._reset_buffers()

    def _commit_all_worlds(self) -> None:
        """``reset_state`` for an all-true mask, without the stream sync.

        Writes go through contiguous scratch tensors: ``copy_`` into a
        non-contiguous column view of a Warp-backed array makes PyTorch issue a
        synchronising copy in this combination, which is harmless in training
        and fatal for a resident controller.
        """
        from gym.utils.torch_quat import quat_rotate_inverse

        env, backend = self.env, self.backend
        canonical_pos = backend._clamp_dof_positions(env.dof_pos)
        native_pos = canonical_pos.index_select(1, backend._native_to_canonical_dof)
        native_vel = env.dof_vel.index_select(1, backend._native_to_canonical_dof)
        dof_state = torch.stack((canonical_pos, env.dof_vel), dim=-1)
        backend._dof_state_t.copy_(dof_state)

        if backend._has_free_joint:
            root = env.root_states
            # Slice-only quaternion reorder: a Python-list index on a
            # non-contiguous view makes PyTorch synchronise the stream.
            quat_wxyz = torch.cat((root[:, 6:7], root[:, 3:6]), dim=1)
            qpos = torch.cat((root[:, :3], quat_wxyz, native_pos), dim=1)
            qvel = torch.cat(
                (
                    root[:, 7:10],
                    quat_rotate_inverse(root[:, 3:7], root[:, 10:13]),
                    native_vel,
                ),
                dim=1,
            )
            backend._qpos_t.copy_(qpos)
            backend._qvel_t.copy_(qvel)
        else:
            backend._qpos_t.copy_(native_pos)
            backend._qvel_t.copy_(native_vel)
        with backend._wp_ctx:
            mjw.forward(backend._m, backend._d)
        backend._sync_assembled_states()
        env._refresh_base_observations(self._all)
        env.dof_pos_obs[:] = env.dof_pos - env.default_dof_pos
        env._update_phase_observation()
        env._update_gait_reference()
        env._reset_buffers()

    def set_action_targets(self, actions: torch.Tensor) -> None:
        """Write policy-unit actions into the PD targets (in place, no alloc)."""
        torch.mul(actions, self.action_scale, out=self.env.dof_pos_target)

    def step(self) -> None:
        """Advance one full control interval (decimation substeps + refresh)."""
        self.env.step()

    def observation(self) -> torch.Tensor:
        return self.env.get_states(self.obs_names)

    def terminated(self) -> torch.Tensor:
        return self.env.terminated

    def base_height(self) -> torch.Tensor:
        return self.env.root_states[:, 2]
