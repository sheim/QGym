"""MuJoCo Warp backend — fully vectorised GPU/CPU execution via mujoco_warp.

Requires Python ≥ 3.10 and mujoco_warp ≥ 3.6.0 (installed separately from
the IsaacGym Python-3.8 environment).  This file is imported lazily inside
select_backend() so the project loads cleanly on Python 3.8.

Step pipeline (mirrors mj_step):
    qfrc_applied[:] = torques
    mjw.forward(m, d)   # position + velocity + actuation + acceleration
    mjw.euler(m, d)     # semi-implicit Euler integration

State tensors are zero-copy torch views into Warp arrays (via wp.to_torch),
so writes to dof_pos / dof_vel are immediately visible in the Warp sim and
vice versa.
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

        # Zero-copy torch views into Warp arrays (set in setup)
        self._qpos_t: torch.Tensor = None  # [N, num_dof]
        self._qvel_t: torch.Tensor = None  # [N, num_dof]
        self._qfrc_t: torch.Tensor = None  # [N, num_dof]
        self._cfrc_t: torch.Tensor = None  # [N, nbody, 6]
        self._root_states_t: torch.Tensor = None  # [N, 13] — zeros for fixed-base

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
        """[num_envs, num_dof] — zero-copy view of d.qpos."""
        return self._qpos_t

    @property
    def dof_vel(self) -> torch.Tensor:
        """[num_envs, num_dof] — zero-copy view of d.qvel."""
        return self._qvel_t

    @property
    def dof_state(self) -> torch.Tensor:
        """[num_envs * num_dof, 2] — assembled on demand from qpos/qvel views.

        Writing into dof_pos or dof_vel is reflected here because both read
        from the same underlying Warp arrays.
        """
        return torch.stack([self._qpos_t, self._qvel_t], dim=-1).view(
            self._num_envs * self._num_dof, 2
        )

    @property
    def root_states(self) -> torch.Tensor:
        """[num_envs, 13] — zeros for fixed-base robots (Phase 1)."""
        return self._root_states_t

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
        # All Warp allocations below use this device context so that tensors
        # end up on the same device as requested (cpu or cuda:N).
        self._wp_ctx = wp.ScopedDevice(device)

        # 1. Load URDF via MuJoCo
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        mjm = mujoco.MjModel.from_xml_path(asset_path)
        self._mjm = mjm

        # 2. Apply physics parameters from cfg
        sim_dt = getattr(cfg, "sim_dt", None) or cfg.sim.dt
        mjm.opt.timestep = sim_dt
        mjm.opt.gravity[:] = np.array(cfg.sim.gravity, dtype=np.float64)
        mjm.dof_damping[:] = cfg.asset.joint_damping
        mjm.dof_armature[:] = getattr(cfg.asset, "rotor_inertia", 0.0)
        if getattr(cfg.asset, "disable_gravity", False):
            mjm.opt.gravity[:] = 0.0

        # 3. Disable contacts (same rationale as MuJocoCPUBackend)
        mjm.geom_contype[:] = 0
        mjm.geom_conaffinity[:] = 0

        # 4. Validate fixed-base assumption
        assert mjm.nq == mjm.nv, (
            f"MuJocoWarpBackend requires a fixed-base robot (nq==nv). "
            f"Got nq={mjm.nq}, nv={mjm.nv}."
        )
        self._num_dof = mjm.nv

        # 5. Extract metadata
        self._body_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(mjm.nbody)
        ]
        self._num_bodies = mjm.nbody
        self._dof_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(mjm.njnt)
        ]

        # 6. Contact index tensors
        self._penalised_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "penalize_contacts_on", []), device
        )
        self._termination_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "terminate_after_contacts_on", []), device
        )

        # 7. Task callbacks
        if task is not None:
            task.num_dof = self._num_dof
        if task is not None and hasattr(task, "_get_env_origins"):
            task._get_env_origins()
        if task is not None and hasattr(task, "_process_dof_props"):
            task._process_dof_props(self._make_dof_props(mjm), env_id=0)

        # 8. Build Warp model and batched data inside the device scope
        with self._wp_ctx:
            self._m = mjw.put_model(mjm)
            mjd = mujoco.MjData(mjm)  # default (zeros) initial state
            self._d = mjw.put_data(mjm, mjd, nworld=num_envs)

            # 9. Zero-copy torch views (created inside same scope → correct device)
            self._qpos_t = wp.to_torch(self._d.qpos)  # [N, nq]
            self._qvel_t = wp.to_torch(self._d.qvel)  # [N, nv]
            self._qfrc_t = wp.to_torch(self._d.qfrc_applied)  # [N, nv]
            self._cfrc_t = wp.to_torch(self._d.cfrc_ext)  # [N, nbody, 6]

        # 10. Fixed-base root_states — always zero (not used by FixedRobot)
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (w=1)

    def _build_contact_indices(self, name_patterns: list, device: str) -> torch.Tensor:
        indices = []
        for pattern in name_patterns:
            for i, bname in enumerate(self._body_names):
                if pattern in bname:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _make_dof_props(self, mjm) -> dict:
        """Build DOF-properties dict matching the IsaacGym interface."""
        n = mjm.njnt
        limited = mjm.jnt_limited[:n].astype(bool)
        lower = np.where(limited, mjm.jnt_range[:n, 0], -1e6)
        upper = np.where(limited, mjm.jnt_range[:n, 1], 1e6)
        velocity = np.full(n, 1e6, dtype=np.float64)
        effort = np.full(n, 1e6, dtype=np.float64)
        return {"lower": lower, "upper": upper, "velocity": velocity, "effort": effort}

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        """Apply torques and advance all worlds by one timestep."""
        import mujoco_warp as mjw

        with self._wp_ctx:
            self._qfrc_t.copy_(torques)
            mjw.forward(self._m, self._d)
            mjw.euler(self._m, self._d)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        """Commit dof_pos[env_ids] / dof_vel[env_ids] to the Warp sim."""
        import mujoco_warp as mjw

        # Writes already in qpos_t / qvel_t (zero-copy views); call forward
        # to update all derived kinematic quantities for every world.
        with self._wp_ctx:
            mjw.forward(self._m, self._d)
