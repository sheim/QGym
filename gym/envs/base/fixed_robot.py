import numpy as np
import torch

from gym.envs.base.base_task import BaseTask
from gym.utils import random_sample
from gym.utils.torch_quat import get_axis_params, to_torch


class FixedRobot(BaseTask):
    def __init__(self, cfg, device, headless, backend):
        self.cfg = cfg
        self.init_done = False

        super().__init__(backend, cfg, device, headless)
        self._parse_cfg(self.cfg)

        if not self.headless:
            self._backend.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._initialize_sim()
        self._init_buffers()
        self.init_done = True
        self.reset()

    def step(self):
        self._reset_buffers()
        self._pre_decimation_step()
        self._render()
        for _ in range(self.cfg.control.decimation):
            self._pre_compute_torques()
            self.torques = self._compute_torques()
            self._post_compute_torques()
            self._step_backend()
            self._post_physics_step()
        self._post_decimation_step()
        self._check_terminations_and_timeouts()

        env_ids = self.to_be_reset.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)

    def _pre_decimation_step(self):
        return None

    def _pre_compute_torques(self):
        return None

    def _post_compute_torques(self):
        if self.cfg.asset.disable_motors:
            self.torques[:] = 0.0

    def _step_backend(self):
        # Map actuated-only torques onto the full DOF space, then hand off to
        # the backend (which owns all gym/sim API calls).
        torques_full_dof = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        torques_full_dof[:, self.actuated_dof_indices] = self.torques
        self._backend.step(torques_full_dof)

    def _post_physics_step(self):
        # backend.step() refreshes all state tensors; nothing to do here.
        pass

    def _post_decimation_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        n = self.num_actuators
        self.dof_pos_history[:, 2 * n :] = self.dof_pos_history[:, n : 2 * n]
        self.dof_pos_history[:, n : 2 * n] = self.dof_pos_history[:, :n]
        self.dof_pos_history[:, :n] = self.dof_pos_target

        self.dof_pos_obs = self.dof_pos - self.default_dof_pos

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_system(env_ids)
        self.dof_pos_history[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

    def _initialize_sim(self):
        """Delegates world-building to the backend, then reads back metadata."""
        self.up_axis_idx = 2  # z-up
        self._backend.setup(self.cfg, self.num_envs, self.device, task=self)

        # Pull metadata that task-level code (rewards, resets) needs.
        self.num_dof = self._backend.num_dof
        self.num_bodies = self._backend.num_bodies
        self.dof_names = self._backend.dof_names
        self.penalised_contact_indices = self._backend.penalised_contact_indices
        self.termination_contact_indices = self._backend.termination_contact_indices
        self.robot_layout = self._backend.robot_layout

        actuated_names = list(self.robot_layout.actuated_dof_names)
        unknown = sorted(set(actuated_names) - set(self.dof_names))
        if unknown:
            raise ValueError(f"unknown actuated_joint_names: {unknown}")
        if len(actuated_names) != self.num_actuators:
            raise ValueError(
                f"cfg.env.num_actuators={self.num_actuators}, but control defines "
                f"{len(actuated_names)} actuated joints: {actuated_names}"
            )
        self.actuated_dof_names = actuated_names
        self.actuated_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in actuated_names],
            dtype=torch.long,
            device=self.device,
        )
        self.torque_limits = self.full_dof_torque_limits.index_select(
            0, self.actuated_dof_indices
        )

    # ------------- Callbacks (called by backend during setup) --------------

    def _process_dof_props(self, props, env_id):
        """Store joint limits from asset DOF properties."""
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )
            self.full_dof_torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )
            for i in range(self.num_dof):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.full_dof_torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
                )
        return props

    # ----------------------------------------
    def _init_buffers(self):
        """Bind live tensor views from the backend, then set up RL buffers."""
        n_envs = self.num_envs

        # Live canonical tensors owned and updated in place by the backend.
        self.root_states = self._backend.root_states
        self.dof_state = self._backend.dof_state
        self.dof_pos = self._backend.dof_pos  # [num_envs, num_dof]
        self.dof_vel = self._backend.dof_vel  # [num_envs, num_dof]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = self._backend.contact_forces  # [N, bodies, 3]

        # ── Non-physics RL buffers (unchanged from original) ───────────────
        self.common_step_counter = 0
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((n_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (n_envs, 1)
        )
        self.torques = torch.zeros(
            n_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.p_gains = torch.zeros(
            self.num_actuators, dtype=torch.float, device=self.device
        )
        self.d_gains = torch.zeros(
            self.num_actuators, dtype=torch.float, device=self.device
        )
        self.actions = torch.zeros(
            n_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.dof_pos_target = torch.zeros(
            n_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.dof_vel_target = torch.zeros(
            n_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.tau_ff = torch.zeros(
            n_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.dof_pos_history = torch.zeros(
            n_envs, self.num_actuators * 3, dtype=torch.float, device=self.device
        )

        # Joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device
        )
        self.default_act_pos = torch.zeros(
            self.num_actuators, dtype=torch.float, device=self.device
        )
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angles = self.cfg.init_state.default_joint_angles
            found = False
            for dof_name in angles.keys():
                if dof_name in name:
                    self.default_dof_pos[i] = angles[dof_name]
                    found = True
            if not found:
                self.default_dof_pos[i] = 0.0
                print(
                    f"Default dof pos of joint {name} was not defined, "
                    + "setting to zero"
                )

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        for actuator_index, name in enumerate(self.actuated_dof_names):
            matching_gains = [
                gain_name
                for gain_name in self.cfg.control.stiffness
                if gain_name in name
            ]
            if len(matching_gains) != 1:
                raise ValueError(
                    f"expected exactly one PD gain match for {name!r}, "
                    f"got {matching_gains}"
                )
            gain_name = matching_gains[0]
            self.p_gains[actuator_index] = self.cfg.control.stiffness[gain_name]
            self.d_gains[actuator_index] = self.cfg.control.damping[gain_name]
            dof_index = self.dof_names.index(name)
            self.default_act_pos[actuator_index] = self.default_dof_pos[0, dof_index]
        self.default_act_pos = self.default_act_pos.unsqueeze(0)
        self.act_idx = self.actuated_dof_indices
        self.initialize_ranges_for_initial_conditions()

    def initialize_ranges_for_initial_conditions(self):
        self.dof_pos_range = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.device
        )
        self.dof_vel_range = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.device
        )
        for joint, vals in self.cfg.init_state.dof_pos_range.items():
            for i in range(self.num_dof):
                if joint in self.dof_names[i]:
                    self.dof_pos_range[i, :] = to_torch(vals, device=self.device)
        for joint, vals in self.cfg.init_state.dof_vel_range.items():
            for i in range(self.num_dof):
                if joint in self.dof_names[i]:
                    self.dof_vel_range[i, :] = to_torch(vals, device=self.device)

    def _get_env_origins(self):
        """Grid of robot spawn origins (called by the backend during setup)."""
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(
            torch.arange(num_rows), torch.arange(num_cols), indexing="ij"
        )
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = self.cfg.env.root_height

    def _reset_system(self, env_ids):
        reset = getattr(self, self.cfg.init_state.reset_mode, None)
        if reset is None:
            raise NameError(f"Unknown default setup: {self.cfg.init_state.reset_mode}")
        reset(env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._backend.reset_dof_state(env_ids_int32)

    # ── Reset modes ─────────────────────────────────────────────────────────

    def reset_to_basic(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0

    def reset_to_range(self, env_ids):
        self.dof_pos[env_ids] = random_sample(
            env_ids,
            self.dof_pos_range[:, 0],
            self.dof_pos_range[:, 1],
            device=self.device,
        )
        self.dof_vel[env_ids] = random_sample(
            env_ids,
            self.dof_vel_range[:, 0],
            self.dof_vel_range[:, 1],
            device=self.device,
        )

    # ── Reward helpers ───────────────────────────────────────────────────────

    def _sqrdexp(self, x, sigma=None):
        if sigma is None:
            return torch.exp(-torch.square(x) / self.cfg.reward_settings.tracking_sigma)
        else:
            return torch.exp(-torch.square(x) / sigma)

    def _reward_torques(self):
        return -torch.mean(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return -torch.mean(torch.square(self.dof_vel), dim=1)

    def _reward_action_rate(self):
        nact = self.num_actuators
        error = torch.square(
            self.dof_pos_history[:, :nact] - self.dof_pos_history[:, 2 * nact :]
        )
        return -torch.mean(error, dim=1)

    def _reward_action_rate2(self):
        nact = self.num_actuators
        error = torch.square(
            self.dof_pos_history[:, :nact]
            - 2 * self.dof_pos_history[:, nact : 2 * nact]
            + self.dof_pos_history[:, 2 * nact :]
        )
        return -torch.mean(error, dim=1)

    def _reward_collision(self):
        return -torch.mean(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        return -self.terminated.float()

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return -torch.mean(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        limit = self.cfg.reward_settings.soft_dof_vel_limit
        error = self.dof_vel.abs() - self.dof_vel_limits * limit
        return -torch.mean(error.clip(min=0.0, max=1.0), dim=1)

    def _compute_torques(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)

        torques = (
            self.p_gains * (self.dof_pos_target + self.default_act_pos - pos)
            + self.d_gains * (self.dof_vel_target - vel)
            + self.tau_ff
        )
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques.view(self.torques.shape)
