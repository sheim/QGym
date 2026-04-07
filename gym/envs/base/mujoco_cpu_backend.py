"""Plain MuJoCo (mj_step) backend — no Warp dependency.

Works on any platform (Linux CPU, Mac Apple Silicon) and is the simplest
concrete SimBackend implementation beyond MockBackend.  One shared MjModel,
one MjData per environment; physics runs in a Python loop and state is copied
numpy→torch after each step.
"""

import numpy as np
import torch
import mujoco

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.base.sim_backend import SimBackend


class MuJocoCPUBackend(SimBackend):
    """SimBackend backed by plain mujoco.mj_step.

    Suitable for fixed-base robots (nq == nv).  LeggedRobot (floating-base)
    support is added in Phase 3.
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

        # State tensors (allocated in setup)
        self._dof_state_t: torch.Tensor = None  # [N, num_dof, 2]
        self._dof_pos_view: torch.Tensor = None  # [N, num_dof] view into above
        self._dof_vel_view: torch.Tensor = None  # [N, num_dof] view into above
        self._root_states_t: torch.Tensor = None  # [N, 13]
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
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        """Load URDF, build N parallel environments, acquire state tensors."""
        self._device = device
        self._num_envs = num_envs

        # 1. Load URDF via MuJoCo's built-in importer
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        mjm = mujoco.MjModel.from_xml_path(asset_path)
        self._mjm = mjm

        # 2. Apply physics parameters from cfg
        # sim_dt is set by task_registry.set_control_and_sim_dt(); fall back to sim.dt
        sim_dt = getattr(cfg, "sim_dt", None) or cfg.sim.dt
        mjm.opt.timestep = sim_dt
        mjm.opt.gravity[:] = np.array(cfg.sim.gravity, dtype=np.float64)
        mjm.dof_damping[:] = cfg.asset.joint_damping
        mjm.dof_armature[:] = getattr(cfg.asset, "rotor_inertia", 0.0)
        if getattr(cfg.asset, "disable_gravity", False):
            mjm.opt.gravity[:] = 0.0

        # 3. Disable contacts — MuJocoCPUBackend does not add a ground plane;
        # self-collisions between fixed-base robot parts cause unphysical
        # constraint forces.  Contact forces are zeroed (not needed for
        # fixed-base FixedRobot reward functions in Phase 1).
        mjm.geom_contype[:] = 0
        mjm.geom_conaffinity[:] = 0

        # Validate fixed-base assumption (nq == nv for 1-DOF joints)
        assert mjm.nq == mjm.nv, (
            f"MuJocoCPUBackend requires a fixed-base robot (nq==nv). "
            f"Got nq={mjm.nq}, nv={mjm.nv}. Use LeggedRobot backend for floating-base."
        )
        self._num_dof = mjm.nv

        # 4. Extract metadata
        # Bodies include the world body at index 0
        self._body_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(mjm.nbody)
        ]
        self._num_bodies = mjm.nbody
        # DOF names come from 1-DOF joints (njnt == nv for fixed-base)
        self._dof_names = [
            mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(mjm.njnt)
        ]

        # 5. Build contact index tensors from cfg body-name patterns
        self._penalised_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "penalize_contacts_on", []), device
        )
        self._termination_contact_indices = self._build_contact_indices(
            getattr(cfg.asset, "terminate_after_contacts_on", []), device
        )

        # 6. Task callbacks (called once; properties are identical across envs).
        # Provide num_dof early so _process_dof_props can allocate limit tensors.
        if task is not None:
            task.num_dof = self._num_dof
        if task is not None and hasattr(task, "_get_env_origins"):
            task._get_env_origins()
        if task is not None and hasattr(task, "_process_dof_props"):
            task._process_dof_props(self._make_dof_props(mjm), env_id=0)

        # 7. Create one MjData per environment
        self._datas = [mujoco.MjData(mjm) for _ in range(num_envs)]

        # 8. Allocate PyTorch state tensors
        # _dof_state_t is [N, num_dof, 2]; dof_pos/dof_vel are views into it,
        # so writing into dof_pos is automatically reflected in dof_state.
        self._dof_state_t = torch.zeros(num_envs, self._num_dof, 2, device=device)
        self._dof_pos_view = self._dof_state_t[..., 0]  # [N, num_dof]
        self._dof_vel_view = self._dof_state_t[..., 1]  # [N, num_dof]
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (scalar-last, w=1)
        self._contact_forces_t = torch.zeros(num_envs, mjm.nbody, 3, device=device)

    def _build_contact_indices(self, name_patterns: list, device: str) -> torch.Tensor:
        indices = []
        for pattern in name_patterns:
            for i, bname in enumerate(self._body_names):
                if pattern in bname:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _make_dof_props(self, mjm: mujoco.MjModel) -> dict:
        """Build the DOF-properties dict expected by task._process_dof_props.

        Keys match the IsaacGym dof_props_asset dict: lower, upper, velocity, effort.
        When MuJoCo reports a joint as unlimited, use ±1e6 so reward/termination
        logic doesn't incorrectly clamp at zero.
        """
        n = mjm.njnt
        limited = mjm.jnt_limited[:n].astype(bool)
        lower = np.where(limited, mjm.jnt_range[:n, 0], -1e6)
        upper = np.where(limited, mjm.jnt_range[:n, 1], 1e6)
        # MuJoCo's base model doesn't store URDF velocity/effort limits as
        # standalone fields; use large defaults so task code doesn't over-clip.
        velocity = np.full(n, 1e6, dtype=np.float64)
        effort = np.full(n, 1e6, dtype=np.float64)
        return {"lower": lower, "upper": upper, "velocity": velocity, "effort": effort}

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        """Apply torques and advance all environments by one timestep."""
        torques_np = torques.cpu().numpy()  # [N, num_dof]
        for i, d in enumerate(self._datas):
            d.qfrc_applied[:] = torques_np[i]
            mujoco.mj_step(self._mjm, d)
        self._sync_state_from_mujoco()

    def _sync_state_from_mujoco(self) -> None:
        """Copy MuJoCo arrays → PyTorch tensors (all environments)."""
        for i, d in enumerate(self._datas):
            self._dof_pos_view[i] = torch.from_numpy(d.qpos.copy())
            self._dof_vel_view[i] = torch.from_numpy(d.qvel.copy())
            self._contact_forces_t[i] = torch.from_numpy(d.cfrc_ext[:, 3:6].copy())

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        """Commit dof_pos[env_ids] / dof_vel[env_ids] back to MuJoCo."""
        for i in env_ids.tolist():
            self._datas[i].qpos[:] = self._dof_pos_view[i].cpu().numpy()
            self._datas[i].qvel[:] = self._dof_vel_view[i].cpu().numpy()
            mujoco.mj_forward(self._mjm, self._datas[i])
