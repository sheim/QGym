import os
import sys

import numpy as np

try:
    from isaacgym import gymapi, gymtorch, gymutil
    from isaacgym.torch_utils import get_axis_params, to_torch
except ImportError:
    gymapi = None
    gymtorch = None
    gymutil = None
    get_axis_params = None
    to_torch = None

import torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.sim_backend import SimBackend


class IsaacGymBackend(SimBackend):
    """IsaacGym/PhysX implementation of SimBackend.

    Owns all isaacgym API calls so that FixedRobot (and eventually
    LeggedRobot) are free of direct engine imports.
    """

    def __init__(
        self,
        gym: "gymapi.Gym",
        sim: "gymapi.Sim",
        sim_params: "gymapi.SimParams",
        sim_device: str,
        headless: bool = True,
    ) -> None:
        self._gym = gym
        self._sim = sim
        self._headless = headless

        # ── Device detection ───────────────────────────────────────────────
        sim_device_type, self._sim_device_id = gymutil.parse_device_str(sim_device)
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self._device = sim_device
        else:
            self._device = "cpu"

        # ── Viewer ────────────────────────────────────────────────────────
        self._viewer = None
        self._enable_viewer_sync = True
        if not headless:
            self._viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_ESCAPE, "QUIT")
            gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

        # ── State tensors (initialised by setup → _acquire_tensors) ───────
        self._dof_state_raw = None
        self._dof_pos = None
        self._dof_vel = None
        self._root_states_raw = None
        self._contact_forces_raw = None
        self._rigid_body_states_raw = None

        # ── Metadata (set by setup → _create_envs) ────────────────────────
        self._num_dof: int = None
        self._num_bodies: int = None
        self._dof_names: list = None
        self._body_names: list = None
        self._envs: list = []
        self._actor_handles: list = []
        self._penalised_contact_indices: torch.Tensor = None
        self._termination_contact_indices: torch.Tensor = None

    # ── SimBackend.device ───────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    # ── Backward-compat shims (for LeggedRobot during migration) ───────────
    # TODO Phase 3: remove once LeggedRobot is fully migrated.

    @property
    def gym(self):
        return self._gym

    @property
    def sim(self):
        return self._sim

    @property
    def viewer(self):
        return self._viewer

    # ── World building ──────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        """Create ground plane, build N parallel envs, prepare sim, acquire
        tensors.  *task* provides per-env property callbacks if supplied.
        """
        self._cfg = cfg
        self._num_envs = num_envs
        self._create_ground_plane(cfg)
        self._create_envs(cfg, num_envs, device, task)
        self._gym.prepare_sim(self._sim)
        self._acquire_tensors(num_envs)

    def _create_ground_plane(self, cfg) -> None:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = cfg.terrain.static_friction
        plane_params.dynamic_friction = cfg.terrain.dynamic_friction
        plane_params.restitution = cfg.terrain.restitution
        self._gym.add_ground(self._sim, plane_params)

    def _create_envs(self, cfg, num_envs: int, device: str, task=None) -> None:
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = cfg.asset.fix_base_link
        asset_options.density = cfg.asset.density
        asset_options.angular_damping = cfg.asset.angular_damping
        asset_options.linear_damping = cfg.asset.linear_damping
        asset_options.max_linear_velocity = cfg.asset.max_linear_velocity
        asset_options.armature = cfg.asset.armature
        asset_options.thickness = cfg.asset.thickness
        asset_options.disable_gravity = cfg.asset.disable_gravity
        asset_options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = cfg.asset.flip_visual_attachments
        asset_options.max_angular_velocity = cfg.asset.max_angular_velocity

        robot_asset = self._gym.load_asset(
            self._sim, asset_root, asset_file, asset_options
        )
        self._num_dof = self._gym.get_asset_dof_count(robot_asset)
        self._num_bodies = self._gym.get_asset_rigid_body_count(robot_asset)

        dof_props_asset = self._gym.get_asset_dof_properties(robot_asset)
        dof_props_asset["armature"] = cfg.asset.rotor_inertia
        dof_props_asset["damping"] = cfg.asset.joint_damping
        rigid_shape_props_asset = self._gym.get_asset_rigid_shape_properties(
            robot_asset
        )

        body_names = self._gym.get_asset_rigid_body_names(robot_asset)
        self._dof_names = self._gym.get_asset_dof_names(robot_asset)
        self._body_names = body_names
        self._num_bodies = len(body_names)

        penalized_contact_names = []
        for name in cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        start_pose = gymapi.Transform()

        # Env origins — task computes these (may depend on terrain type).
        if task is not None and hasattr(task, "_get_env_origins"):
            task._get_env_origins()
            env_origins = task.env_origins
        else:
            env_origins = self._default_env_origins(cfg, num_envs, device)

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self._envs = []
        self._actor_handles = []

        for i in range(num_envs):
            env_handle = self._gym.create_env(
                self._sim, env_lower, env_upper, int(np.sqrt(num_envs))
            )
            pos = env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = (
                task._process_rigid_shape_props(rigid_shape_props_asset, i)
                if task is not None
                else rigid_shape_props_asset
            )
            self._gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_name = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            robot_handle = self._gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                actor_name,
                i,
                cfg.asset.self_collisions,
                0,
            )

            dof_props = (
                task._process_dof_props(dof_props_asset, i)
                if task is not None
                else dof_props_asset
            )
            self._gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

            body_props = self._gym.get_actor_rigid_body_properties(
                env_handle, robot_handle
            )
            if task is not None:
                body_props = task._process_rigid_body_props(body_props, i)
            self._gym.set_actor_rigid_body_properties(
                env_handle, robot_handle, body_props, recomputeInertia=True
            )

            self._envs.append(env_handle)
            self._actor_handles.append(robot_handle)

        # Contact index lookup
        self._penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=device
        )
        for i, name in enumerate(penalized_contact_names):
            self._penalised_contact_indices[i] = self._gym.find_actor_rigid_body_handle(
                self._envs[0], self._actor_handles[0], name
            )

        self._termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=device
        )
        for i, name in enumerate(termination_contact_names):
            self._termination_contact_indices[i] = (
                self._gym.find_actor_rigid_body_handle(
                    self._envs[0], self._actor_handles[0], name
                )
            )

    def _default_env_origins(self, cfg, num_envs: int, device: str) -> torch.Tensor:
        """Simple grid of origins used when task has no _get_env_origins."""
        origins = torch.zeros(num_envs, 3, device=device)
        num_cols = np.floor(np.sqrt(num_envs))
        num_rows = np.ceil(num_envs / num_cols)
        xx, yy = torch.meshgrid(
            torch.arange(num_rows), torch.arange(num_cols), indexing="ij"
        )
        spacing = cfg.env.env_spacing
        origins[:, 0] = spacing * xx.flatten()[:num_envs]
        origins[:, 1] = spacing * yy.flatten()[:num_envs]
        origins[:, 2] = cfg.env.root_height
        return origins

    def _acquire_tensors(self, num_envs: int) -> None:
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)

        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        self._root_states_raw = gymtorch.wrap_tensor(actor_root_state)
        self._dof_state_raw = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_states_raw = gymtorch.wrap_tensor(rigid_body_state)

        n = num_envs
        self._dof_pos = self._dof_state_raw.view(n, self._num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state_raw.view(n, self._num_dof, 2)[..., 1]
        self._contact_forces_raw = gymtorch.wrap_tensor(net_contact_forces).view(
            n, -1, 3
        )

    # ── Metadata properties ─────────────────────────────────────────────────

    @property
    def num_dof(self) -> int:
        return self._num_dof

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def dof_names(self) -> list:
        return self._dof_names

    @property
    def body_names(self) -> list:
        return self._body_names

    @property
    def penalised_contact_indices(self) -> torch.Tensor:
        return self._penalised_contact_indices

    @property
    def termination_contact_indices(self) -> torch.Tensor:
        return self._termination_contact_indices

    def find_body_index(self, name: str) -> int:
        return self._gym.find_actor_rigid_body_handle(
            self._envs[0], self._actor_handles[0], name
        )

    # ── State tensor properties ─────────────────────────────────────────────

    def _require_setup(self, attr: str) -> None:
        if self._dof_pos is None:
            raise RuntimeError(
                f"IsaacGymBackend.{attr} accessed before setup() was called."
            )

    @property
    def dof_state(self) -> torch.Tensor:
        self._require_setup("dof_state")
        return self._dof_state_raw

    @property
    def dof_pos(self) -> torch.Tensor:
        self._require_setup("dof_pos")
        return self._dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        self._require_setup("dof_vel")
        return self._dof_vel

    @property
    def root_states(self) -> torch.Tensor:
        self._require_setup("root_states")
        return self._root_states_raw

    @property
    def contact_forces(self) -> torch.Tensor:
        self._require_setup("contact_forces")
        return self._contact_forces_raw

    @property
    def rigid_body_states(self) -> torch.Tensor:
        self._require_setup("rigid_body_states")
        return self._rigid_body_states_raw

    # ── Per-step ────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        """Apply torques and step PhysX, then refresh all state tensors."""
        self._gym.set_dof_actuation_force_tensor(
            self._sim, gymtorch.unwrap_tensor(torques.contiguous())
        )
        self._gym.simulate(self._sim)
        if self._device == "cpu":
            self._gym.fetch_results(self._sim, True)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._gym.set_dof_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._dof_state_raw),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states_raw),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def set_all_root_states(self) -> None:
        self._gym.set_actor_root_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._root_states_raw)
        )

    # ── Camera / rendering ──────────────────────────────────────────────────

    def set_camera(self, position, lookat) -> None:
        if self._viewer is None:
            return
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

    def render(self, sync_frame_time: bool = True) -> None:
        if self._viewer is None:
            return
        if self._gym.query_viewer_has_closed(self._viewer):
            sys.exit()
        for evt in self._gym.query_viewer_action_events(self._viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self._enable_viewer_sync = not self._enable_viewer_sync
        if self._device != "cpu":
            self._gym.fetch_results(self._sim, True)
        if self._enable_viewer_sync:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
            if sync_frame_time:
                self._gym.sync_frame_time(self._sim)
        else:
            self._gym.poll_viewer_events(self._viewer)

    def register_dof_state(self, dof_state_raw: torch.Tensor, num_envs: int) -> None:
        """Register a pre-acquired dof_state tensor with this backend.

        Used by LeggedRobot, which acquires tensors itself (Phase 0 shim)
        so that reset_dof_state() has the right tensor reference without
        duplicating the num_projs / projection slicing logic.

        TODO Phase 3: remove once LeggedRobot calls backend.setup().
        """
        self._num_envs = num_envs
        self._dof_state_raw = dof_state_raw
        self._dof_pos = dof_state_raw.view(num_envs, self._num_dof, 2)[..., 0]
        self._dof_vel = dof_state_raw.view(num_envs, self._num_dof, 2)[..., 1]

    def close(self) -> None:
        if self._viewer is not None:
            self._gym.destroy_viewer(self._viewer)
            self._viewer = None
