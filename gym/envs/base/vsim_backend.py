"""vsim (vlearn) backend — CUDA-only batched physics engine.

Convention sheet (all spike-verified 2026-07-12, see q2-backend-integration):
- Quaternions scalar-last [x,y,z,w] like the task layer; transform buffers
  are [qx,qy,qz,qw, px,py,pz] (quaternion FIRST) — reorder only, no swizzle.
- Velocity buffers are [ang(3), lin(3)] (measured; the vendor stub is
  misleading).  Constants VEL_ANG / VEL_LIN below are the only place this
  knowledge lives.
- Contact forces come from per-link `flags="contact"` force sensors injected
  by vsim_asset; sensor buffers are [force(3), torque(3)], world frame,
  +z ≈ m·g reaction on a resting body.
- Motors are injected gear=1.0 per movable joint → set_motor_forces takes
  raw Nm.  Torques arrive in DOF order; a name-derived permutation maps them
  to motor order.
- World is Z-up (create_gym up_axis); the default plane is Y-up and must be
  rotated.  Terrain friction is NOT yet applied to the plane (default
  material) — tracked in q2-backend-integration.
- create_gym/delete_gym is a process singleton; create→delete→create works,
  so close() must be called between backends in one process.
- All assembled tensors are refreshed in step()/resets, never in getters
  (SimBackend contract; see MuJocoWarpBackend for the scar this encodes).

Runtime env: LD_LIBRARY_PATH=<site-packages>/vlearn/lib and
VL_WORKING_DIRECTORY=<dir with License.key> (see scripts/run_vsim_tests.sh).
"""

import xml.etree.ElementTree as ET

import torch

from gym.envs.base.sim_backend import SimBackend
from gym.envs.base.urdf_limits import parse_urdf_limits
from gym.envs.base.vsim_asset import ensure_vsim_asset
from gym import LEGGED_GYM_ROOT_DIR

import numpy as np

# Spatial-vector slot layout (spike-verified): angular first, linear second.
VEL_ANG = slice(0, 3)
VEL_LIN = slice(3, 6)
UNLIMITED = 1.0e6


class VSimBackend(SimBackend):
    """SimBackend backed by the vlearn batched GPU engine."""

    def __init__(self) -> None:
        self._v = None  # vlearn module
        self._gym = None
        self._device: str = "cuda:0"
        self._num_envs: int = 0
        self._num_dof: int = 0
        self._num_bodies: int = 0
        self._dof_names: list = []
        self._body_names: list = []
        self._has_free_joint: bool = False
        self._penalised_contact_indices = None
        self._termination_contact_indices = None
        self._pending_camera = None
        self._render_initialized = False

    # ── Metadata ──────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

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

    def find_body_index(self, name: str) -> int:
        return self._body_names.index(name)

    @property
    def penalised_contact_indices(self) -> torch.Tensor:
        return self._penalised_contact_indices

    @property
    def termination_contact_indices(self) -> torch.Tensor:
        return self._termination_contact_indices

    # ── State tensors (plain returns; refreshed by step/resets) ──────────

    @property
    def dof_state(self) -> torch.Tensor:
        return self._dof_state_t.view(self._num_envs * self._num_dof, 2)

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos_view

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel_view

    @property
    def root_states(self) -> torch.Tensor:
        return self._root_states_t

    @property
    def rigid_body_states(self) -> torch.Tensor:
        return self._rigid_body_states_t.view(self._num_envs * self._num_bodies, 13)

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    # ── World building ────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        if not device.startswith("cuda"):
            raise RuntimeError(f"VSimBackend is CUDA-only, got device={device!r}")
        import vlearn as v

        self._v = v
        self._device = device
        self._num_envs = num_envs
        headless = task.headless if task is not None else True

        # Self-heal the engine's cache structure (vsim::Path::findCachePath
        # requires a cache/ dir with its marker file under the working dir).
        import os

        workdir = os.environ.get("VL_WORKING_DIRECTORY", "vendor/vlearn")
        os.makedirs(os.path.join(workdir, "cache", "tmp"), exist_ok=True)
        marker = os.path.join(workdir, "cache", "donotremove.txt")
        if not os.path.exists(marker):
            open(marker, "w").close()

        knobs = getattr(cfg, "vsim_attributes", None)
        pairs_pe = getattr(knobs, "max_contact_pairs_per_env", 64)
        patches_pe = getattr(knobs, "max_contact_patches_per_env", 64)
        pts_pp = getattr(knobs, "max_contact_points_per_patch", 8)
        self._gym = v.create_gym(
            with_render=not headless,
            with_window=not headless,
            up_axis=v.Vec3(0, 0, 1),
            cuda_device=int(device.split(":")[1]) if ":" in device else 0,
            seed=getattr(cfg, "seed", None),
            max_contact_pairs=pairs_pe * num_envs,
            max_contact_patches=patches_pe * num_envs,
            max_contact_points=pts_pp * patches_pe * num_envs,
            enable_graph_captures=getattr(knobs, "enable_graph_captures", True),
            enable_enhanced_determinism=getattr(
                knobs, "enable_enhanced_determinism", False
            ),
            verbose=False,
        )

        vsim_path = ensure_vsim_asset(cfg, self._gym)
        fixed = bool(getattr(cfg.asset, "fix_base_link", True))
        self._has_free_joint = not fixed

        env_def_h = self._gym.create_environment_def()
        env_def = self._gym.get_environment_def(env_def_h)
        env_def.import_definitions(vsim_path, fixed, merge_fixed_joints=False)

        # Articulation-def name: robot XML name for some assets, root link
        # name for others (both observed) — try both, fail loudly otherwise.
        xroot = ET.parse(vsim_path).getroot()
        art_def = None
        for name in (xroot.get("name"), xroot.find("link").get("name")):
            h = env_def.get_articulation_def_handle_by_name(name)
            d = env_def.get_articulation_def(h)
            if d is not None:
                art_def_h, art_def = h, d
                break
        if art_def is None:
            raise RuntimeError(f"no articulation def found in {vsim_path}")

        art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
        for i in range(art_def.get_num_force_sensor_defs()):
            art_def.get_force_sensor_def(i).max_num_transform_handles = 8

        init = getattr(cfg, "init_state", None)
        pos = list(getattr(init, "pos", [0.0, 0.0, 0.0])) if init else [0.0, 0.0, 0.0]
        rot = list(getattr(init, "rot", [0, 0, 0, 1])) if init else [0, 0, 0, 1]
        self._art_h = env_def.create_articulation(
            art_def_h, v.Transform(v.Quat(*rot), v.Vec3(*pos))
        )
        env_def.finalize()
        self._grp = self._gym.create_environment_group(env_def_h, [num_envs])
        # Articulation instance (sensor handles live on it, valid post-group)
        self._art_instance = env_def.get_articulation(self._art_h)

        terrain = getattr(cfg, "terrain", None)
        if terrain is not None and getattr(terrain, "mesh_type", None) == "plane":
            plane_rot = v.shortest_rotation(v.Vec3(0, 1, 0), v.Vec3(0, 0, 1))
            self._gym.create_plane(v.Transform(plane_rot, v.Vec3(0, 0, 0)))
        self._gym.finalize()

        sim_dt = getattr(cfg, "sim_dt", None)
        if sim_dt is None:
            sim_dt = getattr(cfg.sim, "dt", 0.005) if hasattr(cfg, "sim") else 0.005
        self._gym.set_timestep(sim_dt)
        sim_cfg = getattr(cfg, "sim", None)
        gravity = (
            list(sim_cfg.gravity)
            if sim_cfg is not None and hasattr(sim_cfg, "gravity")
            else [0.0, 0.0, -9.81]
        )
        if getattr(cfg.asset, "disable_gravity", False):
            gravity = [0.0, 0.0, 0.0]
        self._gym.set_gravity(v.Vec3(*gravity))

        # Metadata
        self._num_dof = art_def.get_num_joint_dof_defs()
        self._dof_names = art_def.get_joint_dof_def_names()
        self._body_names = art_def.get_link_def_names()
        self._num_bodies = art_def.get_num_link_defs()
        n_motors = art_def.get_num_motor_defs()
        if n_motors != self._num_dof:
            raise RuntimeError(
                f"motor/dof mismatch: {n_motors} motors vs {self._num_dof} dofs "
                f"(asset post-processing should inject one motor per movable joint)"
            )
        motor_joints = [
            art_def.get_motor_def(i).name.removesuffix("_motor")
            for i in range(n_motors)
        ]
        if sorted(motor_joints) != sorted(self._dof_names):
            raise RuntimeError(
                f"motor joints {motor_joints} != dof names {self._dof_names}"
            )
        # motor i is driven by the torque of dof index _motor_src[i]
        self._motor_src = torch.tensor(
            [self._dof_names.index(j) for j in motor_joints],
            dtype=torch.long,
            device=device,
        )

        self._penalised_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "penalize_contacts_on", []), device
        )
        self._termination_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "terminate_after_contacts_on", []), device
        )

        self._allocate_tensors_and_commands()
        self._run_task_callbacks(vsim_path, cfg, task)
        self._refresh_state()
        if not headless:
            self._apply_camera()

    def _allocate_tensors_and_commands(self) -> None:
        """All buffers allocated once (graph captures need stable addresses)."""
        v, grp, gym = self._v, self._grp, self._gym
        N, nd, L = self._num_envs, self._num_dof, self._num_bodies
        dev = self._device

        # Contract tensors
        self._dof_state_t = torch.zeros(N, nd, 2, device=dev)
        self._dof_pos_view = self._dof_state_t[..., 0]
        self._dof_vel_view = self._dof_state_t[..., 1]
        self._root_states_t = torch.zeros(N, 13, device=dev)
        self._root_states_t[:, 6] = 1.0
        self._rigid_body_states_t = torch.zeros(N, L, 13, device=dev)
        self._rigid_body_states_t[:, :, 6] = 1.0
        self._contact_forces_t = torch.zeros(N, L, 3, device=dev)

        # GET command buffers (contiguous; wrap_gpu_buffer rejects views)
        self._jp_get = torch.zeros(N, nd, device=dev)
        self._jv_get = torch.zeros(N, nd, device=dev)
        self._root_tf_get = torch.zeros(N, 1, 7, device=dev)
        self._root_vel_get = torch.zeros(N, 1, 6, device=dev)
        aks_get = grp.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self._jp_get),
            v.wrap_gpu_buffer(self._jv_get),
            v.wrap_gpu_buffer(self._root_tf_get),
            v.wrap_gpu_buffer(self._root_vel_get),
            self._art_h,
            (0, nd),
            (0, 1),
        )
        self._aks_get_arr = gym.create_gpu_array([aks_get])

        self._link_tf = torch.zeros(N, L, 7, device=dev)
        self._link_vel = torch.zeros(N, L, 6, device=dev)
        lt = grp.create_link_transform_command(
            v.wrap_gpu_buffer(self._link_tf), self._art_h, (0, L)
        )
        lv = grp.create_link_velocity_command(
            v.wrap_gpu_buffer(self._link_vel), self._art_h, (0, L)
        )
        self._lt_arr = gym.create_gpu_array([lt])
        self._lv_arr = gym.create_gpu_array([lv])

        # Per-link contact force sensors: one big buffer, slice per sensor
        # (dim-0 slices are contiguous).  Sensor order == link order (the
        # asset post-processor injects them in link order).
        art = self._art_instance
        self._sensor_big = torch.zeros(L, N, 6, device=dev)
        cmds = []
        for i in range(L):
            cmds.append(
                grp.create_force_sensor_command(
                    v.wrap_gpu_buffer(self._sensor_big[i]),
                    art.get_force_sensor_handle(i),
                )
            )
        self._fs_arr = gym.create_gpu_array(cmds)

        # SET commands (masked; persistent mask buffer)
        self._mask = torch.zeros(N, dtype=torch.bool, device=dev)
        self._jp_set = torch.zeros(N, nd, device=dev)
        self._jv_set = torch.zeros(N, nd, device=dev)
        jp_set_cmd = grp.create_joint_state_command(
            v.wrap_gpu_buffer(self._jp_set),
            self._art_h,
            (0, nd),
            masks_buffer=v.wrap_gpu_buffer(self._mask),
        )
        jv_set_cmd = grp.create_joint_state_command(
            v.wrap_gpu_buffer(self._jv_set),
            self._art_h,
            (0, nd),
            masks_buffer=v.wrap_gpu_buffer(self._mask),
        )
        self._jp_set_arr = gym.create_gpu_array([jp_set_cmd])
        self._jv_set_arr = gym.create_gpu_array([jv_set_cmd])

        if self._has_free_joint:
            self._root_tf_set = torch.zeros(N, 1, 7, device=dev)
            self._root_vel_set = torch.zeros(N, 1, 6, device=dev)
            aks_set = grp.create_articulation_kinematic_state_command(
                v.wrap_gpu_buffer(self._jp_set),
                v.wrap_gpu_buffer(self._jv_set),
                v.wrap_gpu_buffer(self._root_tf_set),
                v.wrap_gpu_buffer(self._root_vel_set),
                self._art_h,
                (0, nd),
                (0, 1),
                masks_buffer=v.wrap_gpu_buffer(self._mask),
            )
            self._aks_set_arr = gym.create_gpu_array([aks_set])

        # Motor buffer
        self._motor_buf = torch.zeros(N, nd, device=dev)
        mot = grp.create_motor_control_command(
            v.wrap_gpu_buffer(self._motor_buf), self._art_h, (0, nd)
        )
        self._motor_arr = gym.create_gpu_array([mot])

    def _run_task_callbacks(self, vsim_path: str, cfg, task) -> None:
        if task is not None:
            task.num_dof = self._num_dof
        if task is not None and hasattr(task, "_get_env_origins"):
            task._get_env_origins()
        if task is not None and hasattr(task, "_process_dof_props"):
            task._process_dof_props(self._make_dof_props(vsim_path, cfg), env_id=0)

    def _make_dof_props(self, vsim_path: str, cfg) -> dict:
        """lower/upper from the .vsim XML (lower>upper == unlimited),
        effort/velocity re-parsed from the source URDF."""
        root = ET.parse(vsim_path).getroot()
        limits = {}
        for j in root.findall("joint"):
            if j.get("type") == "fixed":
                continue
            lim = j.find("limit")
            lo = float(lim.get("lower", "1")) if lim is not None else 1.0
            hi = float(lim.get("upper", "-1")) if lim is not None else -1.0
            if lo > hi:
                lo, hi = -UNLIMITED, UNLIMITED
            limits[j.get("name")] = (lo, hi)
        urdf_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        urdf_ev = parse_urdf_limits(urdf_path)
        n = self._num_dof
        lower = np.zeros(n)
        upper = np.zeros(n)
        effort = np.full(n, UNLIMITED)
        velocity = np.full(n, UNLIMITED)
        for i, name in enumerate(self._dof_names):
            lower[i], upper[i] = limits[name]
            if name in urdf_ev:
                effort[i], velocity[i] = urdf_ev[name]
        return {"lower": lower, "upper": upper, "velocity": velocity, "effort": effort}

    def _build_contact_indices(self, name_patterns: list, device: str) -> torch.Tensor:
        indices = []
        for pattern in name_patterns:
            for i, bname in enumerate(self._body_names):
                if pattern in bname:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ── Per-step ──────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        self._motor_buf.copy_(torques[:, self._motor_src])
        self._gym.set_motor_forces(self._motor_arr)
        self._gym.step()
        self._refresh_state()

    def _refresh_state(self) -> None:
        g = self._gym
        g.compute_kinematics()
        g.get_articulation_kinematic_states(self._aks_get_arr)
        g.get_link_transforms(self._lt_arr)
        g.get_link_velocities(self._lv_arr)
        g.get_sensor_forces(self._fs_arr)
        self._sync_assembled_states()

    def _sync_assembled_states(self) -> None:
        """Refresh contract tensors in place (all tensors live after step)."""
        self._dof_pos_view.copy_(self._jp_get)
        self._dof_vel_view.copy_(self._jv_get)
        rs = self._root_states_t
        rtf = self._root_tf_get[:, 0]
        rvl = self._root_vel_get[:, 0]
        rs[:, 0:3] = rtf[:, 4:7]
        rs[:, 3:7] = rtf[:, 0:4]
        rs[:, 7:10] = rvl[:, VEL_LIN]
        rs[:, 10:13] = rvl[:, VEL_ANG]
        rbs = self._rigid_body_states_t
        rbs[..., 0:3] = self._link_tf[..., 4:7]
        rbs[..., 3:7] = self._link_tf[..., 0:4]
        rbs[..., 7:10] = self._link_vel[..., VEL_LIN]
        rbs[..., 10:13] = self._link_vel[..., VEL_ANG]
        # sensor buffers are [force, torque]; big buffer is [L, N, 6]
        self._contact_forces_t.copy_(self._sensor_big[:, :, 0:3].permute(1, 0, 2))

    # ── Resets (write-then-commit; task wrote desired values into views) ──

    def _commit_state(self, env_ids: torch.Tensor) -> None:
        self._mask.zero_()
        self._mask[env_ids] = True
        self._jp_set.copy_(self._dof_pos_view)
        self._jv_set.copy_(self._dof_vel_view)
        if self._has_free_joint:
            rs = self._root_states_t
            self._root_tf_set[:, 0, 0:4] = rs[:, 3:7]
            self._root_tf_set[:, 0, 4:7] = rs[:, 0:3]
            self._root_vel_set[:, 0, VEL_ANG] = rs[:, 10:13]
            self._root_vel_set[:, 0, VEL_LIN] = rs[:, 7:10]
            self._gym.set_articulation_kinematic_states(self._aks_set_arr)
        else:
            self._gym.set_joint_positions(self._jp_set_arr)
            self._gym.set_joint_velocities(self._jv_set_arr)
        self._refresh_state()

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        self._commit_state(env_ids)

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        if not self._has_free_joint:
            return
        self._commit_state(env_ids)

    def set_all_root_states(self) -> None:
        self.reset_root_state(torch.arange(self._num_envs, device=self._device))

    # ── Rendering / lifecycle ─────────────────────────────────────────────

    def render(self, sync_frame_time: bool = True) -> None:
        r = self._gym.get_render()
        if not self._render_initialized:
            # Vendor demos do this before their loops; the viewer starts
            # paused otherwise.  capped_step syncs sim rate to the display.
            r.capped_step = bool(sync_frame_time)
            r.set_paused(False)
            self._render_initialized = True
        r.render()

    def set_camera(self, position, lookat) -> None:
        """Stash-and-apply: BaseTask calls this BEFORE setup() (init order),
        when the engine isn't up yet.  The camera is applied at the end of
        setup() once the render exists."""
        self._pending_camera = (list(position), list(lookat))
        if self._gym is not None:
            self._apply_camera()

    def _apply_camera(self) -> None:
        if self._pending_camera is None:
            return
        position, lookat = self._pending_camera
        v = self._v
        d = [lookat[i] - position[i] for i in range(3)]
        n = max(sum(x * x for x in d) ** 0.5, 1e-9)
        self._gym.get_render().reset_camera(
            v.Vec3(*position), v.Vec3(*(x / n for x in d))
        )
        self._pending_camera = None

    def close(self) -> None:
        if self._v is not None:
            self._v.delete_gym()
            self._gym = None
