import numpy as np

try:
    from isaacgym.torch_utils import get_axis_params, to_torch
except ImportError:
    get_axis_params = None
    to_torch = None

import torch

from gym.envs.base.base_task import BaseTask
from gym.envs.base.isaac_gym_backend import IsaacGymBackend
from gym.utils import random_sample


class FixedRobot(BaseTask):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.init_done = False
        # Temporary device assignment so _parse_cfg can run; corrected below.
        self.device = sim_device
        self._parse_cfg(self.cfg)

        backend = IsaacGymBackend(gym, sim, sim_params, sim_device, headless)
        super().__init__(backend, cfg, backend.device, headless)

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
            self._step_physx_sim()
            self._post_physx_step()
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

    def _step_physx_sim(self):
        # Map actuated-only torques onto the full DOF space, then hand off to
        # the backend (which owns all gym/sim API calls).
        torques_full_dof = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        next_actuated_idx = 0
        for dof_idx in range(self.num_dof):
            if self.cfg.control.actuated_joints_mask[dof_idx]:
                torques_full_dof[:, dof_idx] = self.torques[:, next_actuated_idx]
                next_actuated_idx += 1
        self._backend.step(torques_full_dof)

    def _post_physx_step(self):
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

    # ------------- Callbacks (called by backend during setup) --------------

    def _process_rigid_shape_props(self, props, env_id):
        """Store / randomize rigid shape properties per env."""
        return props

    def _process_dof_props(self, props, env_id):
        """Store joint limits from asset DOF properties."""
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )
            self.torque_limits = torch.zeros(
                self.num_actuators, dtype=torch.float, device=self.device
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                try:
                    self.torque_limits[i] = props["effort"][i].item()
                except Exception:
                    print("WARNING: your system has unactuated joints")
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

    def _process_rigid_body_props(self, props, env_id):
        return props

    # ----------------------------------------
    def _init_buffers(self):
        """Bind live tensor views from the backend, then set up RL buffers."""
        n_envs = self.num_envs

        # Live views into the backend's physics state (zero-copy on GPU).
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
        actuated_idx = []
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

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    self.default_act_pos[i] = angles[dof_name]
                    found = True
                    actuated_idx.append(i)
            if not found:
                try:
                    self.p_gains[i] = 0.0
                    self.d_gains[i] = 0.0
                    print("This should not happen anymore")
                    if self.cfg.control.control_type in ["P", "V"]:
                        print(f"PD gain of joint {name} not defined, set to zero")
                except Exception:
                    pass

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_act_pos = self.default_act_pos.unsqueeze(0)
        self.act_idx = to_torch(actuated_idx, dtype=torch.long, device=self.device)
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
        if hasattr(self, self.cfg.init_state.reset_mode):
            eval(f"self.{self.cfg.init_state.reset_mode}(env_ids)")
        else:
            raise NameError(f"Unknown default setup: {self.cfg.init_state.reset_mode}")
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
        actuated_dof_pos = torch.zeros(
            self.num_envs, self.num_actuators, device=self.device
        )
        actuated_dof_vel = torch.zeros(
            self.num_envs, self.num_actuators, device=self.device
        )
        idx = 0
        for dof_idx in range(self.num_dof):
            if self.cfg.control.actuated_joints_mask[dof_idx]:
                actuated_dof_pos[:, idx] = self.dof_pos[:, dof_idx]
                actuated_dof_vel[:, idx] = self.dof_vel[:, dof_idx]
                idx += 1

        torques = (
            self.p_gains
            * (self.dof_pos_target + self.default_act_pos - actuated_dof_pos)
            + self.d_gains * (self.dof_vel_target - actuated_dof_vel)
            + self.tau_ff
        )
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques.view(self.torques.shape)
