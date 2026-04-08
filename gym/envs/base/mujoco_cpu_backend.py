"""Plain MuJoCo (mj_step) backend — no Warp dependency.

Works on any platform (Linux CPU, Mac Apple Silicon) and is the simplest
concrete SimBackend implementation beyond MockBackend.  One shared MjModel,
one MjData per environment; physics runs in a Python loop and state is copied
numpy→torch after each step.

Supports both fixed-base robots (nq == nv, e.g. pendulum) and floating-base
robots (nq == nv + 1, e.g. mini_cheetah).
"""

import numpy as np
import torch
import mujoco

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.sim_backend import SimBackend


def _mj_quat_to_xyzw(q):
    """MuJoCo [w,x,y,z] → task-layer [x,y,z,w]."""
    return q[..., [1, 2, 3, 0]]


def _xyzw_to_mj_quat(q):
    """Task-layer [x,y,z,w] → MuJoCo [w,x,y,z]."""
    return q[..., [3, 0, 1, 2]]


def _set_balanceinertia(spec):
    """Set balanceinertia on MjSpec, handling API differences across versions."""
    if "compiler" in dir(spec):
        spec.compiler.balanceinertia = True  # mujoco >= 3.6
    else:
        spec.balanceinertia = True  # mujoco < 3.6


class MuJocoCPUBackend(SimBackend):
    """SimBackend backed by plain mujoco.mj_step.

    Supports fixed-base (nq == nv) and floating-base (nq == nv + 1) robots.
    """

    def __init__(self) -> None:
        self._mjm: mujoco.MjModel = None
        self._datas: list = []
        self._device: str = "cpu"
        self._num_envs: int = 0
        self._num_dof: int = 0
        self._num_bodies: int = 0
        self._dof_names: list = []
        self._body_names: list = []

        # Floating-base offsets (0 for fixed-base)
        self._has_free_joint: bool = False
        self._qpos_offset: int = 0
        self._qvel_offset: int = 0

        # State tensors (allocated in setup)
        self._dof_state_t: torch.Tensor = None  # [N, num_dof, 2]
        self._dof_pos_view: torch.Tensor = None  # [N, num_dof] view into above
        self._dof_vel_view: torch.Tensor = None  # [N, num_dof] view into above
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, num_bodies, 13]
        self._contact_forces_t: torch.Tensor = None  # [N, num_bodies, 3]

        # Contact body indices (set by setup)
        self._penalised_contact_indices: torch.Tensor = None
        self._termination_contact_indices: torch.Tensor = None

    # ── SimBackend.device ─────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    # ── Metadata ──────────────────────────────────────────────────────────────

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

    # ── Contact indices ────────────────────────────────────────────────────────

    @property
    def penalised_contact_indices(self) -> torch.Tensor:
        return self._penalised_contact_indices

    @property
    def termination_contact_indices(self) -> torch.Tensor:
        return self._termination_contact_indices

    # ── State tensors ──────────────────────────────────────────────────────────

    @property
    def dof_state(self) -> torch.Tensor:
        """[num_envs * num_dof, 2] view — dof_pos / dof_vel are views into this."""
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
        """[num_envs * num_bodies, 13] — pos(3) quat_xyzw(4) linvel(3) angvel(3)."""
        return self._rigid_body_states_t.view(self._num_envs * self._num_bodies, 13)

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        """Load URDF, build N parallel environments, acquire state tensors."""
        self._device = device
        self._num_envs = num_envs

        # 1. Load URDF via MuJoCo's spec API (allows compiler flags + ground plane)
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        spec = mujoco.MjSpec()
        spec.from_file(asset_path)
        _set_balanceinertia(spec)

        # Add free joint for floating-base robots
        if not getattr(cfg.asset, "fix_base_link", True):
            root_body = spec.worldbody.first_body()
            freejoint = root_body.add_freejoint()
            freejoint.name = "root"

        # Add ground plane if terrain config requests it
        terrain_cfg = getattr(cfg, "terrain", None)
        if (
            terrain_cfg is not None
            and getattr(terrain_cfg, "mesh_type", None) == "plane"
        ):
            ground = spec.worldbody.add_geom()
            ground.type = mujoco.mjtGeom.mjGEOM_PLANE
            ground.size = [100, 100, 0.1]
            sf = getattr(terrain_cfg, "static_friction", 1.0)
            df = getattr(terrain_cfg, "dynamic_friction", 1.0)
            ground.friction = [sf, df, 0.0001]

        mjm = spec.compile()
        self._mjm = mjm

        # 2. Apply physics parameters from cfg
        sim_dt = getattr(cfg, "sim_dt", None)
        if sim_dt is None:
            sim_dt = getattr(cfg.sim, "dt", 0.005) if hasattr(cfg, "sim") else 0.005
        mjm.opt.timestep = sim_dt
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None and hasattr(sim_cfg, "gravity"):
            mjm.opt.gravity[:] = np.array(sim_cfg.gravity, dtype=np.float64)
        if getattr(cfg.asset, "disable_gravity", False):
            mjm.opt.gravity[:] = 0.0

        # 3. Detect floating-base (free joint adds 1 extra qpos for quaternion)
        self._has_free_joint = mjm.nq == mjm.nv + 1
        if self._has_free_joint:
            self._qpos_offset = 7  # skip pos(3) + quat(4)
            self._qvel_offset = 6  # skip linvel(3) + angvel(3)
            self._num_dof = mjm.nv - 6
        else:
            assert mjm.nq == mjm.nv, (
                f"Unexpected nq/nv: nq={mjm.nq}, nv={mjm.nv}. "
                f"Expected nq==nv (fixed-base) or nq==nv+1 (free joint)."
            )
            self._qpos_offset = 0
            self._qvel_offset = 0
            self._num_dof = mjm.nv

        # 4. Apply damping/armature to actuated DOFs only
        mjm.dof_damping[self._qvel_offset :] = cfg.asset.joint_damping
        mjm.dof_armature[self._qvel_offset :] = getattr(cfg.asset, "rotor_inertia", 0.0)

        # 5. Contacts: disable for fixed-base (no ground), keep for floating-base
        if not self._has_free_joint:
            mjm.geom_contype[:] = 0
            mjm.geom_conaffinity[:] = 0

        # 6. Extract metadata
        self._body_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(mjm.nbody)
        ]
        self._num_bodies = mjm.nbody

        # DOF names: skip the free joint (index 0) for floating-base
        jnt_start = 1 if self._has_free_joint else 0
        self._dof_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(jnt_start, mjm.njnt)
        ]

        # 7. Build contact index tensors from cfg body-name patterns
        self._penalised_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "penalize_contacts_on", []), device
        )
        self._termination_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "terminate_after_contacts_on", []), device
        )

        # 8. Task callbacks
        if task is not None:
            task.num_dof = self._num_dof
        if task is not None and hasattr(task, "_get_env_origins"):
            task._get_env_origins()
        if task is not None and hasattr(task, "_process_dof_props"):
            task._process_dof_props(self._make_dof_props(mjm), env_id=0)

        # 9. Create one MjData per environment
        self._datas = [mujoco.MjData(mjm) for _ in range(num_envs)]

        # 10. Allocate PyTorch state tensors
        self._dof_state_t = torch.zeros(num_envs, self._num_dof, 2, device=device)
        self._dof_pos_view = self._dof_state_t[..., 0]  # [N, num_dof]
        self._dof_vel_view = self._dof_state_t[..., 1]  # [N, num_dof]
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (scalar-last, w=1)
        self._rigid_body_states_t = torch.zeros(num_envs, mjm.nbody, 13, device=device)
        self._rigid_body_states_t[:, :, 6] = 1.0  # identity quaternion
        self._contact_forces_t = torch.zeros(num_envs, mjm.nbody, 3, device=device)

    def _build_contact_indices(self, name_patterns: list, device: str) -> torch.Tensor:
        indices = []
        for pattern in name_patterns:
            for i, bname in enumerate(self._body_names):
                if pattern in bname:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _make_dof_props(self, mjm: mujoco.MjModel) -> dict:
        """Build DOF-properties dict expected by task._process_dof_props.

        Keys match the IsaacGym dof_props_asset dict: lower, upper, velocity, effort.
        Only includes actuated joints (skips free joint for floating-base).
        """
        jnt_start = 1 if self._has_free_joint else 0
        n = mjm.njnt - jnt_start
        limited = mjm.jnt_limited[jnt_start : mjm.njnt].astype(bool)
        lower = np.where(limited, mjm.jnt_range[jnt_start : mjm.njnt, 0], -1e6)
        upper = np.where(limited, mjm.jnt_range[jnt_start : mjm.njnt, 1], 1e6)
        velocity = np.full(n, 1e6, dtype=np.float64)
        effort = np.full(n, 1e6, dtype=np.float64)
        return {"lower": lower, "upper": upper, "velocity": velocity, "effort": effort}

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        """Apply torques and advance all environments by one timestep."""
        torques_np = torques.cpu().numpy()  # [N, num_dof]
        off = self._qvel_offset
        for i, d in enumerate(self._datas):
            d.qfrc_applied[off:] = torques_np[i]
            mujoco.mj_step(self._mjm, d)
        self._sync_state_from_mujoco()

    def _sync_state_from_mujoco(self) -> None:
        """Copy MuJoCo arrays → PyTorch tensors (all environments)."""
        qoff = self._qpos_offset
        voff = self._qvel_offset
        for i, d in enumerate(self._datas):
            self._dof_pos_view[i] = torch.from_numpy(d.qpos[qoff:].copy())
            self._dof_vel_view[i] = torch.from_numpy(d.qvel[voff:].copy())
            self._contact_forces_t[i] = torch.from_numpy(d.cfrc_ext[:, 3:6].copy())
            # Rigid body states: xpos[nbody,3], xquat[nbody,4](wxyz),
            # cvel[nbody,6](ang,lin)
            rbs = self._rigid_body_states_t[i]
            rbs[:, 0:3] = torch.from_numpy(d.xpos.copy())
            mj_quat = torch.from_numpy(d.xquat.copy())  # [nbody, 4] wxyz
            rbs[:, 3:7] = mj_quat[:, [1, 2, 3, 0]]  # → xyzw
            rbs[:, 7:10] = torch.from_numpy(d.cvel[:, 3:6].copy())  # linear vel
            rbs[:, 10:13] = torch.from_numpy(d.cvel[:, 0:3].copy())  # angular vel
        if self._has_free_joint:
            for i, d in enumerate(self._datas):
                self._root_states_t[i, :3] = torch.from_numpy(d.qpos[:3].copy())
                mj_quat = torch.from_numpy(d.qpos[3:7].copy())
                self._root_states_t[i, 3:7] = mj_quat[[1, 2, 3, 0]]
                self._root_states_t[i, 7:10] = torch.from_numpy(d.qvel[:3].copy())
                self._root_states_t[i, 10:13] = torch.from_numpy(d.qvel[3:6].copy())

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        """Commit dof_pos[env_ids] / dof_vel[env_ids] back to MuJoCo."""
        qoff = self._qpos_offset
        voff = self._qvel_offset
        for i in env_ids.tolist():
            self._datas[i].qpos[qoff:] = self._dof_pos_view[i].cpu().numpy()
            self._datas[i].qvel[voff:] = self._dof_vel_view[i].cpu().numpy()
            mujoco.mj_forward(self._mjm, self._datas[i])

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        """Commit root_states[env_ids] back to MuJoCo (floating-base only)."""
        if not self._has_free_joint:
            return
        for i in env_ids.tolist():
            rs = self._root_states_t[i].cpu()
            self._datas[i].qpos[:3] = rs[:3].numpy()
            # quat: task-layer [x,y,z,w] → MuJoCo [w,x,y,z]
            self._datas[i].qpos[3:7] = rs[3:7][[3, 0, 1, 2]].numpy()
            self._datas[i].qvel[:3] = rs[7:10].numpy()
            self._datas[i].qvel[3:6] = rs[10:13].numpy()
            mujoco.mj_forward(self._mjm, self._datas[i])

    def set_all_root_states(self) -> None:
        """Commit root_states for all envs (used by push_robots)."""
        self.reset_root_state(torch.arange(self._num_envs))
