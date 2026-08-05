import numpy as np
import torch

from gym.envs.base.base_task import BaseTask
from gym.utils import random_sample
from gym.utils.gym_math_wrappers import torch_rand_float
from gym.utils.helpers import class_to_dict
from gym.utils.torch_quat import (
    get_axis_params,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    to_torch,
)


class LeggedRobot(BaseTask):
    def __init__(self, cfg, device, headless, backend):
        self.cfg = cfg
        self.init_done = False
        self._physics_step_observers = []

        super().__init__(backend, cfg, device, headless)
        self._parse_cfg(self.cfg)

        if not self.headless:
            self._set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._initialize_sim()
        self._init_buffers()
        self.init_done = True
        self.reset()

    def step(self):
        self._reset_buffers()
        self._pre_decimation_step()
        # * step physics and render each frame
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

    def add_physics_step_observer(self, observer):
        """Register opt-in evaluation instrumentation at the physics rate."""
        self._physics_step_observers.append(observer)

    def remove_physics_step_observer(self, observer):
        self._physics_step_observers.remove(observer)

    def _pre_decimation_step(self):
        return None

    def _pre_compute_torques(self):
        return None

    def _post_compute_torques(self):
        if self.cfg.asset.disable_motors:
            self.torques[:] = 0.0

    def _step_backend(self):
        if self.num_actuators == self.num_dof:
            self._backend.step(self.torques)
            return
        torques_full_dof = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        torques_full_dof[:, self.actuated_dof_indices] = self.torques
        self._backend.step(torques_full_dof)

    def _post_physics_step(self):
        # backend.step() already refreshed all tensors; compute derived quantities.
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )

    def _post_decimation_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        self.base_height = self.root_states[:, 2:3]

        self.dof_pos_obs = self.dof_pos - self.default_dof_pos

        self.dof_pos_history = self.dof_pos_history.roll(self.num_actuators)
        self.dof_pos_history[:, : self.num_actuators] = self.dof_pos_target

        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)
            == 0
        )
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())
        if self.cfg.push_robots.toggle:
            if self.common_step_counter % self.cfg.push_interval == 0:
                self._push_robots()

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # * reset robot states
        self._reset_system(env_ids)
        self._resample_commands(env_ids)
        # * reset buffers
        self.dof_pos_obs[env_ids] = self.dof_pos[env_ids] - self.default_dof_pos
        self.dof_pos_target[env_ids] = self.default_dof_pos
        self.dof_pos_history[env_ids] = self.dof_pos_target[env_ids].tile(3)
        self.episode_length_buf[env_ids] = 0

    def _initialize_sim(self):
        """Delegate flat-world construction to the selected backend."""
        self.up_axis_idx = 2
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type not in (None, "plane"):
            raise ValueError(
                "supported backends currently require flat terrain "
                f"(mesh_type None or 'plane'), got {mesh_type!r}"
            )

        self._backend.setup(self.cfg, self.num_envs, self.device, task=self)
        self.num_dof = self._backend.num_dof
        self.num_bodies = self._backend.num_bodies
        self.dof_names = self._backend.dof_names
        self.penalised_contact_indices = self._backend.penalised_contact_indices
        self.termination_contact_indices = self._backend.termination_contact_indices

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)

        self.robot_layout = self._backend.robot_layout
        self.actuated_dof_names = list(self.robot_layout.actuated_dof_names)
        self.actuated_dof_indices = torch.tensor(
            self.robot_layout.dof_indices(self.actuated_dof_names),
            dtype=torch.long,
            device=self.device,
        )
        self.feet_indices = torch.tensor(
            self.robot_layout.body_group_indices("feet"),
            dtype=torch.long,
            device=self.device,
        )
        if len(self.actuated_dof_names) != self.num_actuators:
            raise ValueError(
                f"cfg.env.num_actuators={self.num_actuators}, but layout defines "
                f"{len(self.actuated_dof_names)} actuated DOFs: "
                f"{self.actuated_dof_names}"
            )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]):
            Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            -self.command_ranges["lin_vel_y"],
            self.command_ranges["lin_vel_y"],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        max_yaw_vel = self.command_ranges["yaw_vel"]
        self.commands[env_ids, 2] = torch_rand_float(
            -max_yaw_vel, max_yaw_vel, (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def _set_camera(self, position, lookat):
        """Set camera position and direction"""
        self._backend.set_camera(position, lookat)

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of
            each environment. Called During environment creation.
            Base behavior: stores position, velocity and torques limits
                defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device
            )

            # props is a dictionary, instead of being an nd array.
            # Slice-assign to preserve the device/dtype of the pre-allocated
            # tensors — torch.from_numpy alone returns a CPU float64 tensor.
            self.dof_pos_limits[:, 0] = torch.from_numpy(props["lower"])
            self.dof_pos_limits[:, 1] = torch.from_numpy(props["upper"])
            self.dof_vel_limits[:] = torch.from_numpy(props["velocity"])
            self.torque_limits[:] = torch.from_numpy(props["effort"])

            # for i in range(len(props)):
            # * soft limits
            # remove? Put into penalty instead
            # m = (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2
            # r = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
            # self.dof_pos_limits[:, 0] = (
            #     m - 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
            # )
            # self.dof_pos_limits[:, 1] = (
            #     m + 0.5 * r * self.cfg.reward_settings.soft_dof_pos_limit
            # )
        return props

    def _compute_torques(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        default_pos = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        torques = (
            self.p_gains * (self.dof_pos_target + default_pos - pos)
            + self.d_gains * (self.dof_vel_target - vel)
            + self.tau_ff
        )
        torques = torch.clip(
            torques, -self.actuated_torque_limits, self.actuated_torque_limits
        )
        return torques.view(self.torques.shape)

    def _reset_system(self, env_ids):
        """Resets selected environmments
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # todo: move getattr to initialization (also in fixed robot)
        reset = getattr(self, self.cfg.init_state.reset_mode, None)
        if reset is None:
            raise NameError(f"Unknown default setup: {self.cfg.init_state.reset_mode}")
        reset(env_ids)

        # * start base position shifted in X-Y plane
        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self._backend.reset_dof_state(env_ids)
        self._backend.reset_root_state(env_ids)

    # * implement reset methods
    def reset_to_basic(self, env_ids):
        """
        Reset to a single initial state
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0
        self.root_states[env_ids] = self.base_init_state

    def reset_to_range(self, env_ids):
        """
        Reset to a uniformly random distribution of states, sampled from a
        range for each state
        """
        # * dof states
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

        # * base states
        random_com_pos = random_sample(
            env_ids,
            self.root_pos_range[:, 0],
            self.root_pos_range[:, 1],
            device=self.device,
        )

        self.root_states[env_ids, 0:7] = torch.cat(
            (
                random_com_pos[:, 0:3],
                quat_from_euler_xyz(
                    random_com_pos[:, 3],
                    random_com_pos[:, 4],
                    random_com_pos[:, 5],
                ),
            ),
            1,
        )
        self.root_states[env_ids, 7:13] = random_sample(
            env_ids,
            self.root_vel_range[:, 0],
            self.root_vel_range[:, 1],
            device=self.device,
        )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a
        randomized base velocity.
        """
        max_vel = self.cfg.push_robots.max_push_vel_xy
        box_dims = (
            torch.tensor(self.cfg.push_robots.push_box_dims, device=self.device) / 2.0
        )
        r_vec = torch.cat(
            (
                torch_rand_float(
                    -box_dims[0],
                    box_dims[0],
                    (self.num_envs, 1),
                    device=self.device,
                ),
                torch_rand_float(
                    -box_dims[1],
                    box_dims[1],
                    (self.num_envs, 1),
                    device=self.device,
                ),
                torch_rand_float(
                    -box_dims[2],
                    box_dims[2],
                    (self.num_envs, 1),
                    device=self.device,
                ),
            ),
            dim=1,
        )
        vel_vec = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 3), device=self.device
        )
        vel_vec[:, 2] = 0  # no z velocity
        self.root_states[:, 7:10] += vel_vec
        self.root_states[:, 10:13] += torch.cross(r_vec, vel_vec, dim=1)
        self._backend.set_all_root_states()

    # ----------------------------------------
    def _init_buffers(self):
        """Bind backend state tensors and initialize processed quantities."""
        self.root_states = self._backend.root_states
        self.dof_state = self._backend.dof_state
        self.dof_pos = self._backend.dof_pos
        self.dof_vel = self._backend.dof_vel
        self.contact_forces = self._backend.contact_forces
        self._rigid_body_state = self._backend.rigid_body_states

        self.base_quat = self.root_states[:, 3:7]
        rbs = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        self._rigid_body_pos = rbs[..., 0:3]
        self._rigid_body_quat = rbs[..., 3:7]
        self._rigid_body_lin_vel = rbs[..., 7:10]
        self._rigid_body_ang_vel = rbs[..., 10:13]

        # * initialize some data used later on
        self.common_step_counter = 0

        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.p_gains = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.d_gains = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.dof_pos_target = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.dof_vel_target = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )
        self.tau_ff = torch.zeros(
            self.num_envs, self.num_actuators, dtype=torch.float, device=self.device
        )

        self.dof_pos_history = torch.zeros(
            self.num_envs, self.num_actuators * 3, dtype=torch.float, device=self.device
        )
        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.dof_pos_obs = torch.zeros_like(self.dof_pos)
        self.base_height = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )

        # Joint position offsets in canonical full-DOF order.
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device
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
            self.p_gains[:, actuator_index] = self.cfg.control.stiffness[gain_name]
            self.d_gains[:, actuator_index] = self.cfg.control.damping[gain_name]
        self.actuated_torque_limits = self.torque_limits.index_select(
            0, self.actuated_dof_indices
        )

        # * check that init range highs and lows are consistent
        # * and repopulate to match
        if self.cfg.init_state.reset_mode == "reset_to_range":
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

        self.root_pos_range = torch.tensor(
            self.cfg.init_state.root_pos_range,
            dtype=torch.float,
            device=self.device,
        )
        self.root_vel_range = torch.tensor(
            self.cfg.init_state.root_vel_range,
            dtype=torch.float,
            device=self.device,
        )

    def _get_env_origins(self):
        """Create the flat-world grid used to offset parallel environments."""
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

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.cfg.push_interval = np.ceil(self.cfg.push_robots.interval_s / self.dt)

    def _sqrdexp(self, x, sigma=None):
        """shorthand helper for squared exponential"""
        if sigma is None:
            return torch.exp(-torch.square(x) / self.cfg.reward_settings.tracking_sigma)
        else:
            return torch.exp(-torch.square(x) / sigma)

    # ------------ reward functions----------------

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity"""
        return -torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        return -torch.mean(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non flat base orientation"""
        return -torch.mean(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize base height away from target"""
        target = self.cfg.reward_settings.base_height_target
        return -torch.square(self.root_states[:, 2] - target)

    def _reward_torques(self):
        """Penalize torques"""
        return -torch.mean(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return -torch.mean(torch.square(self.dof_vel), dim=1)

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        n = self.num_actuators
        error = torch.square(
            self.dof_pos_history[:, :n] - self.dof_pos_history[:, 2 * n :]
        )
        return -torch.mean(error, dim=1)

    def _reward_action_rate2(self):
        """Penalize changes in actions"""
        n = self.num_actuators
        error = torch.square(
            self.dof_pos_history[:, :n]
            - 2 * self.dof_pos_history[:, n : 2 * n]
            + self.dof_pos_history[:, 2 * n :]
        )
        return -torch.mean(error, dim=1)

    def _reward_collision(self):
        """Penalize collisions on selected bodies"""
        return -torch.sum(
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
        # Terminal reward / penalty
        return -self.terminated.float()

    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to the limit"""
        # * lower limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return -torch.mean(out_of_limits, dim=1)

    def _reward_dof_pos_target_limits(self):
        """Penalize commanded positions outside the configured soft limits."""
        limits = self.dof_pos_limits.index_select(0, self.actuated_dof_indices)
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        full_range = limits[:, 1] - limits[:, 0]
        soft_half_range = 0.5 * full_range * self.cfg.reward_settings.soft_dof_pos_limit
        lower = center - soft_half_range
        upper = center + soft_half_range
        target = (
            self.default_dof_pos.index_select(1, self.actuated_dof_indices)
            + self.dof_pos_target
        )
        violation = (lower - target).clip(min=0.0)
        violation += (target - upper).clip(min=0.0)
        return -torch.mean(violation / full_range, dim=1)

    def _reward_dof_vel_limits(self):
        """Penalize dof velocities too close to the limit"""
        # * clip to max error = 1 rad/s per joint to avoid huge penalties
        limit = self.cfg.reward_settings.soft_dof_vel_limit
        error = self.dof_vel.abs() - self.dof_vel_limits * limit
        return -torch.mean(error.clip(min=0.0, max=1.0), dim=1)

    def _reward_torque_limits(self):
        """penalize torques too close to the limit"""
        limit = self.cfg.reward_settings.soft_torque_limit
        error = self.torques.abs() - self.actuated_torque_limits * limit
        return -torch.mean(error.clip(min=0.0, max=1.0), dim=1)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        error = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        error = torch.exp(-error / self.cfg.reward_settings.tracking_sigma)
        return torch.mean(error, dim=1)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.reward_settings.tracking_sigma)

    def _reward_feet_contact_forces(self):
        """penalize high contact forces"""
        return -torch.mean(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.reward_settings.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )
