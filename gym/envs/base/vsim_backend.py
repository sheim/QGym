"""vsim (vlearn) backend — CUDA-only batched physics engine.

Convention sheet (all spike-verified 2026-07-12, see q2-backend-integration):
- Quaternions scalar-last [x,y,z,w] like the task layer; transform buffers
  are [qx,qy,qz,qw, px,py,pz] (quaternion FIRST) — reorder only, no swizzle.
- Velocity buffers are [ang(3), lin(3)] (measured; the vendor stub is
  misleading).  Constants VEL_ANG / VEL_LIN below are the only place this
  knowledge lives.
- Contact forces come from per-link `flags="contact"` force sensors injected
    by vsim_asset; sensor buffers are [force(3), torque(3)] in sensor/link
    axes despite requesting the environment frame, so refresh rotates them to
    world axes (+z sums to m·g on a resting body).
- Motors are injected gear=1.0 per movable joint → set_motor_forces takes
  raw Nm. Torques arrive in canonical DOF order; `MotorDef.dof_index` maps
  them to native motor order.
- World is Z-up (create_gym up_axis); the default plane is Y-up and must be
  rotated. Plane tasks assign one explicit material to the robot and terrain
  so the task friction/restitution config is not replaced by engine defaults.
- create_gym/delete_gym is a process singleton; create→delete→create works,
  so close() must be called between backends in one process.
- All assembled tensors are refreshed in step()/resets, never in getters
  (SimBackend contract; see MuJocoWarpBackend for the scar this encodes).

Runtime env: LD_LIBRARY_PATH=<site-packages>/vlearn/lib and
VL_WORKING_DIRECTORY=<dir with License.key> (see scripts/run_vsim_tests.sh).
"""

import os
import xml.etree.ElementTree as ET

import torch

from gym.envs.base.robot_layout import RobotLayout
from gym.envs.base.sim_backend import SimBackend
from gym.envs.base.urdf_limits import parse_urdf_limits
from gym.envs.base.vsim_asset import ensure_vsim_asset
from gym import LEGGED_GYM_ROOT_DIR

import numpy as np

# Spatial-vector slot layout (spike-verified): angular first, linear second.
VEL_ANG = slice(0, 3)
VEL_LIN = slice(3, 6)
UNLIMITED = 1.0e6


def _motor_sources(
    native_to_canonical_dof: list[int], motor_native_dofs: list[int]
) -> list[int]:
    expected = list(range(len(native_to_canonical_dof)))
    if sorted(motor_native_dofs) != expected:
        raise RuntimeError(
            f"motors do not map one-to-one onto articulation DOFs: {motor_native_dofs}"
        )
    return [native_to_canonical_dof[dof_index] for dof_index in motor_native_dofs]


def _canonical_body_sensor_indices(
    canonical_to_native_body: list[int], sensor_native_links: list[int]
) -> list[int]:
    if len(set(sensor_native_links)) != len(sensor_native_links):
        raise RuntimeError(
            f"multiple contact sensors target the same link: {sensor_native_links}"
        )
    missing = sorted(set(canonical_to_native_body) - set(sensor_native_links))
    if missing:
        raise RuntimeError(
            "every canonical body must have exactly one contact sensor: "
            f"sensor native links={sensor_native_links}, missing native links={missing}"
        )
    return [
        sensor_native_links.index(link_index) for link_index in canonical_to_native_body
    ]


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
        self._native_dof_names: list = []
        self._native_body_names: list = []
        self._num_native_bodies: int = 0
        self._num_sensors: int = 0
        self._has_free_joint: bool = False
        self._penalised_contact_indices = None
        self._termination_contact_indices = None
        self._pending_camera = None
        self._render_initialized = False
        self._render_hooks = []  # each called with the render, per frame
        self._window_closed = False

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

    @staticmethod
    def _set_contact_offsets(model_def, knobs) -> None:
        contact_offset = getattr(knobs, "contact_offset", None)
        rest_offset = getattr(knobs, "rest_offset", None)
        if contact_offset is not None:
            model_def.contact_offset = float(contact_offset)
        if rest_offset is not None:
            model_def.rest_offset = float(rest_offset)

    def _create_contact_material(self, env_def, art_def_h, art_def, cfg):
        """Create one explicit material shared by the robot and terrain."""
        knobs = getattr(cfg, "vsim_attributes", None)
        stiffness = getattr(knobs, "contact_stiffness", None)
        terrain = getattr(cfg, "terrain", None)
        if terrain is None and stiffness is None:
            return None

        material = self._v.RigidMaterial()
        material.static_friction = float(getattr(terrain, "static_friction", 1.0))
        material.dynamic_friction = float(getattr(terrain, "dynamic_friction", 1.0))
        material.restitution = float(getattr(terrain, "restitution", 0.0))
        if stiffness is not None:
            material.restitution = -abs(float(stiffness))
            material.damping = float(getattr(knobs, "contact_damping", 0.0))

        handle = env_def.create_rigid_material(material)
        for link_index in range(art_def.get_num_link_defs()):
            env_def.assign_rigid_material_to_articulation_link(
                art_def_h, handle, link_index
            )
        return handle

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
        workdir = os.environ.get("VL_WORKING_DIRECTORY", "thirdparty/vlearn")
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
        solver_iterations = getattr(knobs, "solver_iterations", None)
        if solver_iterations is not None:
            self._gym.set_num_solver_iterations(int(solver_iterations))

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

        self._set_contact_offsets(art_def, knobs)
        contact_material = self._create_contact_material(
            env_def, art_def_h, art_def, cfg
        )
        art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
        for i in range(art_def.get_num_force_sensor_defs()):
            art_def.get_force_sensor_def(i).max_num_transform_handles = 8

        init = getattr(cfg, "init_state", None)
        pos = list(getattr(init, "pos", [0.0, 0.0, 0.0]))
        rot = list(getattr(init, "rot", [0, 0, 0, 1]))
        self._art_h = env_def.create_articulation(
            art_def_h, v.Transform(v.Quat(*rot), v.Vec3(*pos))
        )

        terrain = getattr(cfg, "terrain", None)
        has_plane = (
            terrain is not None and getattr(terrain, "mesh_type", None) == "plane"
        )
        use_local_plane = has_plane and (
            contact_material is not None
            or getattr(knobs, "contact_offset", None) is not None
            or getattr(knobs, "rest_offset", None) is not None
        )
        if use_local_plane:
            plane_def_h = env_def.create_plane_def(
                rigid_material_handle=contact_material
            )
            plane_def = env_def.get_rigid_body_def(plane_def_h)
            self._set_contact_offsets(plane_def, knobs)
            plane_rot = v.shortest_rotation(v.Vec3(0, 1, 0), v.Vec3(0, 0, 1))
            env_def.create_rigid_body(
                plane_def_h,
                v.Transform(plane_rot, v.Vec3(0, 0, 0)),
                name="ground",
            )

        env_def.finalize()
        self._grp = self._gym.create_environment_group(env_def_h, [num_envs])
        # Articulation instance (sensor handles live on it, valid post-group)
        self._art_instance = env_def.get_articulation(self._art_h)

        if has_plane and not use_local_plane:
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

        # Native engine metadata and stable task-facing layout.
        self._native_dof_names = art_def.get_joint_dof_def_names()
        self._native_body_names = art_def.get_link_def_names()
        self._robot_layout = RobotLayout.from_cfg(cfg)
        self._robot_layout.validate_native(
            self._native_dof_names, self._native_body_names
        )
        self._dof_names = list(self._robot_layout.dof_names)
        self._body_names = list(self._robot_layout.body_names)
        self._num_dof = len(self._dof_names)
        self._num_bodies = len(self._body_names)
        self._num_native_bodies = len(self._native_body_names)

        canonical_to_native_dof = self._robot_layout.canonical_to_native_dof(
            self._native_dof_names
        )
        native_to_canonical_dof = self._robot_layout.native_to_canonical_dof(
            self._native_dof_names
        )
        canonical_to_native_body = self._robot_layout.canonical_to_native_body(
            self._native_body_names
        )
        self._canonical_to_native_dof = torch.tensor(
            canonical_to_native_dof, dtype=torch.long, device=device
        )
        self._native_to_canonical_dof = torch.tensor(
            native_to_canonical_dof, dtype=torch.long, device=device
        )
        self._canonical_to_native_body = torch.tensor(
            canonical_to_native_body, dtype=torch.long, device=device
        )

        n_motors = art_def.get_num_motor_defs()
        if n_motors != self._num_dof:
            raise RuntimeError(
                f"motor/dof mismatch: {n_motors} motors vs {self._num_dof} dofs "
                f"(asset post-processing should inject one motor per movable joint)"
            )
        motor_native_dofs = [
            art_def.get_motor_def(i).dof_index for i in range(n_motors)
        ]
        # Motor i is driven by its documented native DOF, converted to the
        # canonical task-facing torque slot.
        self._motor_src = torch.tensor(
            _motor_sources(native_to_canonical_dof, motor_native_dofs),
            dtype=torch.long,
            device=device,
        )

        self._num_sensors = art_def.get_num_force_sensor_defs()
        sensor_native_links = [
            art_def.get_force_sensor_def(i).link_index for i in range(self._num_sensors)
        ]
        self._canonical_body_to_sensor = torch.tensor(
            _canonical_body_sensor_indices(
                canonical_to_native_body, sensor_native_links
            ),
            dtype=torch.long,
            device=device,
        )

        self._penalised_contact_indices = self.build_contact_indices(
            getattr(cfg.asset, "penalize_contacts_on", []), device
        )
        self._termination_contact_indices = self.build_contact_indices(
            getattr(cfg.asset, "terminate_after_contacts_on", []), device
        )

        self._allocate_tensors_and_commands()
        self._run_task_callbacks(xroot, cfg, task)
        self._refresh_state()
        if not headless:
            self._apply_camera()

    def _allocate_tensors_and_commands(self) -> None:
        """All buffers allocated once (graph captures need stable addresses)."""
        v, grp, gym = self._v, self._grp, self._gym
        N, nd = self._num_envs, self._num_dof
        L = self._num_bodies
        native_L = self._num_native_bodies
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

        self._link_tf = torch.zeros(N, native_L, 7, device=dev)
        self._link_vel = torch.zeros(N, native_L, 6, device=dev)
        lt = grp.create_link_transform_command(
            v.wrap_gpu_buffer(self._link_tf), self._art_h, (0, native_L)
        )
        lv = grp.create_link_velocity_command(
            v.wrap_gpu_buffer(self._link_vel), self._art_h, (0, native_L)
        )
        self._lt_arr = gym.create_gpu_array([lt])
        self._lv_arr = gym.create_gpu_array([lv])

        # Per-link contact force sensors: one big buffer, slice per sensor.
        # Sensor order is independent of native link and canonical body order.
        art = self._art_instance
        self._sensor_big = torch.zeros(self._num_sensors, N, 6, device=dev)
        cmds = []
        for i in range(self._num_sensors):
            cmds.append(
                grp.create_force_sensor_command(
                    v.wrap_gpu_buffer(self._sensor_big[i]),
                    art.get_force_sensor_handle(i),
                    frame_type=v.FrameType.ENVIRONMENT,
                )
            )
        self._fs_arr = gym.create_gpu_array(cmds)

        # SET commands (masked; persistent mask buffer)
        self._mask = torch.zeros(N, dtype=torch.bool, device=dev)
        self._jp_set = torch.zeros(N, nd, device=dev)
        self._jv_set = torch.zeros(N, nd, device=dev)
        if not self._has_free_joint:
            # Fixed base: joints are committed directly.
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
        else:
            # Floating base: one kinematic-state command carries root + joints.
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
        self._motor_buf = torch.zeros(N, len(self._motor_src), device=dev)
        mot = grp.create_motor_control_command(
            v.wrap_gpu_buffer(self._motor_buf),
            self._art_h,
            (0, len(self._motor_src)),
        )
        self._motor_arr = gym.create_gpu_array([mot])

    def _run_task_callbacks(self, xroot, cfg, task) -> None:
        if task is None:
            return
        task.num_dof = self._num_dof
        if hasattr(task, "_get_env_origins"):
            task._get_env_origins()
        if hasattr(task, "_process_dof_props"):
            task._process_dof_props(self._make_dof_props(xroot, cfg), env_id=0)

    def _make_dof_props(self, root, cfg) -> dict:
        """lower/upper from the .vsim XML (lower>upper == unlimited),
        effort/velocity re-parsed from the source URDF."""
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
        self._dof_pos_view.copy_(
            self._jp_get.index_select(1, self._canonical_to_native_dof)
        )
        self._dof_vel_view.copy_(
            self._jv_get.index_select(1, self._canonical_to_native_dof)
        )
        rs = self._root_states_t
        rtf = self._root_tf_get[:, 0]
        rvl = self._root_vel_get[:, 0]
        rs[:, 0:3] = rtf[:, 4:7]
        rs[:, 3:7] = rtf[:, 0:4]
        rs[:, 7:10] = rvl[:, VEL_LIN]
        rs[:, 10:13] = rvl[:, VEL_ANG]
        rbs = self._rigid_body_states_t
        link_tf = self._link_tf.index_select(1, self._canonical_to_native_body)
        link_vel = self._link_vel.index_select(1, self._canonical_to_native_body)
        rbs[..., 0:3] = link_tf[..., 4:7]
        rbs[..., 3:7] = link_tf[..., 0:4]
        rbs[..., 7:10] = link_vel[..., VEL_LIN]
        rbs[..., 10:13] = link_vel[..., VEL_ANG]
        # Sensor buffers are [force, torque] in the attached sensor/link frame,
        # even when the command requests ENVIRONMENT (verified against static
        # weight). Rotate the canonical link-local forces into world axes to
        # satisfy the SimBackend contact tensor contract.
        sensor_forces = self._sensor_big.index_select(0, self._canonical_body_to_sensor)
        force_local = sensor_forces[:, :, 0:3].permute(1, 0, 2)
        link_quat = link_tf[..., 0:4]
        q_xyz = link_quat[..., 0:3]
        twice_cross = 2.0 * torch.cross(q_xyz, force_local, dim=-1)
        force_world = (
            force_local
            + link_quat[..., 3:4] * twice_cross
            + torch.cross(q_xyz, twice_cross, dim=-1)
        )
        self._contact_forces_t.copy_(force_world)

    # ── Resets (write-then-commit; task wrote desired values into views) ──

    def _commit_state(self, env_ids: torch.Tensor) -> None:
        self._mask.zero_()
        self._mask[env_ids] = True
        self._jp_set.copy_(
            self._dof_pos_view.index_select(1, self._native_to_canonical_dof)
        )
        self._jv_set.copy_(
            self._dof_vel_view.index_select(1, self._native_to_canonical_dof)
        )
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

    def add_render_hook(self, fn) -> None:
        """Register a callable(render) invoked once per rendered frame."""
        self._render_hooks.append(fn)

    # ── Debug line drawing ────────────────────────────────────────────────
    # vsim has no arrow/capsule debug geoms — only polylines — so callers
    # build shapes from point lists.  Engine types (Vec3, UserLine) stay
    # inside the backend; callers pass plain (x, y, z) tuples.

    def create_debug_line(self, points, color=(1.0, 1.0, 1.0), width: float = 2.0):
        v, r = self._v, self._gym.get_render()
        line = r.create_user_line(
            [v.Vec3(*p) for p in points], v.Vec3(*color), line_width=width
        )
        r.register_line_shape(line)
        return line

    def update_debug_line(self, line, points, visible: bool = True) -> None:
        line.set_visible(visible)
        if visible and points:
            line.set_points([self._v.Vec3(*p) for p in points])

    @property
    def escape_key(self):
        """UserKey.Escape — lets interfaces poll specials without importing
        vlearn themselves (see VsimKeyboardInterface)."""
        return self._v.UserKey.Escape

    @property
    def window_closed(self) -> bool:
        """True once the user closes the viewer window (render() reports it)."""
        return self._window_closed

    def render(self, sync_frame_time: bool = True) -> None:
        r = self._gym.get_render()
        if not self._render_initialized:
            # Vendor demos do this before their loops; the viewer starts
            # paused otherwise.  capped_step syncs sim rate to the display.
            r.capped_step = bool(sync_frame_time)
            r.set_paused(False)
            self._render_initialized = True
        # Per-frame hooks (vlearn input is polled, not event-driven — see
        # VsimKeyboardInterface; the command visualiser also redraws here).
        # They run before render() so a change lands on the same frame.
        for hook in self._render_hooks:
            hook(r)
        self._window_closed = bool(r.render())

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
