"""MuJoCo Warp backend — fully vectorised GPU/CPU execution via mujoco_warp.

Requires mujoco >= 3.6 and mujoco_warp >= 3.6.

Step pipeline (mirrors mj_step):
    qfrc_applied[:, offset:] = torques
    mjw.forward(m, d)   # position + velocity + actuation + acceleration
    mjw.euler(m, d)     # semi-implicit Euler integration

State tensors are zero-copy torch views into Warp arrays (via wp.to_torch),
so writes to dof_pos / dof_vel are immediately visible in the Warp sim and
vice versa.

Supports both fixed-base robots (nq == nv, e.g. pendulum) and floating-base
robots (nq == nv + 1, e.g. mini_cheetah).
"""

import numpy as np
import torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.sim_backend import SimBackend


class MuJocoWarpBackend(SimBackend):
    """SimBackend backed by mujoco_warp for vectorised GPU physics."""

    def __init__(self) -> None:
        self._m = None  # mjw.Model
        self._d = None  # mjw.Data
        self._mjm = None  # mujoco.MjModel (kept for metadata queries)
        self._device: str = "cuda:0"
        self._num_envs: int = 0
        self._num_dof: int = 0
        self._num_bodies: int = 0
        self._dof_names: list = []
        self._body_names: list = []

        # Floating-base offsets (0 for fixed-base)
        self._has_free_joint: bool = False
        self._qpos_offset: int = 0
        self._qvel_offset: int = 0

        # Zero-copy torch views into Warp arrays (set in setup)
        self._qpos_t: torch.Tensor = None  # [N, nq] — full qpos
        self._qvel_t: torch.Tensor = None  # [N, nv] — full qvel
        self._qfrc_t: torch.Tensor = None  # [N, nv] — full qfrc_applied
        self._cfrc_t: torch.Tensor = None  # [N, nbody, 6]
        self._xpos_t: torch.Tensor = None  # [N, nbody, 3]
        self._xquat_t: torch.Tensor = None  # [N, nbody, 4]
        self._cvel_t: torch.Tensor = None  # [N, nbody, 6]
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, nbody, 13]

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
    def dof_pos(self) -> torch.Tensor:
        """[num_envs, num_dof] — view into d.qpos (after free-joint offset)."""
        return self._qpos_t[:, self._qpos_offset :]

    @property
    def dof_vel(self) -> torch.Tensor:
        """[num_envs, num_dof] — view into d.qvel (after free-joint offset)."""
        return self._qvel_t[:, self._qvel_offset :]

    @property
    def dof_state(self) -> torch.Tensor:
        """[num_envs * num_dof, 2] — assembled on demand from qpos/qvel views."""
        return torch.stack([self.dof_pos, self.dof_vel], dim=-1).view(
            self._num_envs * self._num_dof, 2
        )

    @property
    def root_states(self) -> torch.Tensor:
        """[num_envs, 13] — assembled from qpos/qvel for floating-base."""
        if not self._has_free_joint:
            return self._root_states_t
        # Assemble from zero-copy qpos/qvel views
        rs = self._root_states_t
        rs[:, :3] = self._qpos_t[:, :3]
        # quat: MuJoCo [w,x,y,z] → task-layer [x,y,z,w]
        rs[:, 3:7] = self._qpos_t[:, 3:7][:, [1, 2, 3, 0]]
        rs[:, 7:10] = self._qvel_t[:, :3]
        rs[:, 10:13] = self._qvel_t[:, 3:6]
        return rs

    @property
    def rigid_body_states(self) -> torch.Tensor:
        """[num_envs * num_bodies, 13] — pos(3) quat_xyzw(4) linvel(3) angvel(3)."""
        rbs = self._rigid_body_states_t
        rbs[:, :, 0:3] = self._xpos_t
        rbs[:, :, 3:7] = self._xquat_t[:, :, [1, 2, 3, 0]]  # wxyz → xyzw
        rbs[:, :, 7:10] = self._cvel_t[:, :, 3:6]  # linear vel
        rbs[:, :, 10:13] = self._cvel_t[:, :, 0:3]  # angular vel
        return rbs.view(self._num_envs * self._num_bodies, 13)

    @property
    def contact_forces(self) -> torch.Tensor:
        """[num_envs, num_bodies, 3] — force part of cfrc_ext."""
        return self._cfrc_t[..., 3:6]

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        """Load asset, build N parallel Warp worlds, acquire state tensors."""
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        self._device = device
        self._num_envs = num_envs

        wp.init()
        self._wp_ctx = wp.ScopedDevice(device)

        # 1. Load URDF via MuJoCo's spec API (allows compiler flags + ground plane)
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        spec = mujoco.MjSpec.from_file(asset_path)
        spec.compiler.balanceinertia = True

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

        # 3. Detect floating-base
        self._has_free_joint = mjm.nq == mjm.nv + 1
        if self._has_free_joint:
            self._qpos_offset = 7
            self._qvel_offset = 6
            self._num_dof = mjm.nv - 6
        else:
            assert mjm.nq == mjm.nv, f"Unexpected nq/nv: nq={mjm.nq}, nv={mjm.nv}."
            self._qpos_offset = 0
            self._qvel_offset = 0
            self._num_dof = mjm.nv

        # 4. Apply damping/armature to actuated DOFs only
        mjm.dof_damping[self._qvel_offset :] = cfg.asset.joint_damping
        mjm.dof_armature[self._qvel_offset :] = getattr(cfg.asset, "rotor_inertia", 0.0)

        # 5. Contacts: disable for fixed-base, keep for floating-base
        if not self._has_free_joint:
            mjm.geom_contype[:] = 0
            mjm.geom_conaffinity[:] = 0

        # 6. Extract metadata
        self._body_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(mjm.nbody)
        ]
        self._num_bodies = mjm.nbody
        jnt_start = 1 if self._has_free_joint else 0
        self._dof_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(jnt_start, mjm.njnt)
        ]

        # 7. Contact index tensors
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

        # 9. Build Warp model and batched data inside the device scope
        with self._wp_ctx:
            self._m = mjw.put_model(mjm)
            mjd = mujoco.MjData(mjm)
            self._d = mjw.put_data(mjm, mjd, nworld=num_envs)

            # 10. Zero-copy torch views (full qpos/qvel arrays)
            self._qpos_t = wp.to_torch(self._d.qpos)  # [N, nq]
            self._qvel_t = wp.to_torch(self._d.qvel)  # [N, nv]
            self._qfrc_t = wp.to_torch(self._d.qfrc_applied)  # [N, nv]
            self._cfrc_t = wp.to_torch(self._d.cfrc_ext)  # [N, nbody, 6]
            self._xpos_t = wp.to_torch(self._d.xpos)  # [N, nbody, 3]
            self._xquat_t = wp.to_torch(self._d.xquat)  # [N, nbody, 4]
            self._cvel_t = wp.to_torch(self._d.cvel)  # [N, nbody, 6]

        # 11. Root states + rigid body states scratch tensors
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (w=1)
        self._rigid_body_states_t = torch.zeros(num_envs, mjm.nbody, 13, device=device)
        self._rigid_body_states_t[:, :, 6] = 1.0

    def _build_contact_indices(self, name_patterns: list, device: str) -> torch.Tensor:
        indices = []
        for pattern in name_patterns:
            for i, bname in enumerate(self._body_names):
                if pattern in bname:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _make_dof_props(self, mjm) -> dict:
        """Build DOF-properties dict matching the IsaacGym interface."""

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
        """Apply torques and advance all worlds by one timestep."""
        import mujoco_warp as mjw

        with self._wp_ctx:
            off = self._qvel_offset
            if off > 0:
                self._qfrc_t[:, off:].copy_(torques)
            else:
                self._qfrc_t.copy_(torques)
            mjw.forward(self._m, self._d)
            mjw.euler(self._m, self._d)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        """Commit dof_pos[env_ids] / dof_vel[env_ids] to the Warp sim."""
        import mujoco_warp as mjw

        # Writes already in qpos_t / qvel_t (zero-copy views); call forward
        # to update all derived kinematic quantities.
        with self._wp_ctx:
            mjw.forward(self._m, self._d)

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        """Commit root_states[env_ids] back to qpos/qvel (floating-base only)."""
        if not self._has_free_joint:
            return
        rs = self._root_states_t[env_ids]
        self._qpos_t[env_ids, :3] = rs[:, :3]
        # quat: task-layer [x,y,z,w] → MuJoCo [w,x,y,z]
        self._qpos_t[env_ids, 3:7] = rs[:, 3:7][:, [3, 0, 1, 2]]
        self._qvel_t[env_ids, :3] = rs[:, 7:10]
        self._qvel_t[env_ids, 3:6] = rs[:, 10:13]

        import mujoco_warp as mjw

        with self._wp_ctx:
            mjw.forward(self._m, self._d)

    def set_all_root_states(self) -> None:
        """Commit root_states for all envs (used by push_robots)."""
        self.reset_root_state(torch.arange(self._num_envs, device=self._device))
