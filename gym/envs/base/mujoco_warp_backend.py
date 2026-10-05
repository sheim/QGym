"""MuJoCo Warp backend — fully vectorised GPU/CPU execution via mujoco_warp.

Requires mujoco >= 3.6 and mujoco_warp >= 3.6.

Step pipeline (mirrors mj_step):
    qfrc_applied[:, offset:] = torques
    mjw.forward(m, d)             # position + velocity + actuation + acceleration
    mjw.euler(m, d)               # semi-implicit Euler integration
    mjw.rne_postconstraint(m, d)  # populate cfrc_ext (contact forces)
    mjw.kinematics(m, d)          # refresh post-integration body poses
    mjw.com_pos(m, d)
    mjw.com_vel(m, d)             # refresh post-integration body velocities
    _sync_assembled_states()      # refresh root_states / rigid_body_states

Native state tensors are zero-copy torch views into Warp arrays (via
wp.to_torch). Public DOF and body tensors are assembled in canonical robot
order; resets scatter canonical writes back into the native arrays.
All assembled tensors are re-synced in step() and reset methods, never lazily
in their getters — the task layer caches these tensors once at init and expects
in-place updates (SimBackend contract: all tensors live after step() returns).
"""

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from gym.envs.base.domain_randomization import (
    contact_friction_range,
    link_mass_scale_range,
)
from gym.envs.base.mujoco_backend_base import (
    MuJocoBackendBase,
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
)
from gym.utils.torch_quat import quat_apply, quat_rotate_inverse


class MuJocoWarpBackend(MuJocoBackendBase):
    """SimBackend backed by mujoco_warp for vectorised GPU physics."""

    def __init__(self) -> None:
        super().__init__()
        self._m = None  # mjw.Model
        self._d = None  # mjw.Data

        # Zero-copy torch views into Warp arrays (set in setup)
        self._qpos_t: torch.Tensor = None  # [N, nq]
        self._qvel_t: torch.Tensor = None  # [N, nv]
        self._qfrc_t: torch.Tensor = None  # [N, nv]
        self._cfrc_t: torch.Tensor = None  # [N, nbody, 6]
        self._xpos_t: torch.Tensor = None  # [N, nbody, 3]
        self._xquat_t: torch.Tensor = None  # [N, nbody, 4]
        self._cvel_t: torch.Tensor = None  # [N, nbody, 6]
        self._subtree_com_t: torch.Tensor = None  # [N, nbody, 3]
        self._body_rootid_t: torch.Tensor = None  # [nbody]
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, nbody, 13]
        self._dof_state_t: torch.Tensor = None  # [N, num_dof, 2]
        self._dof_pos_view: torch.Tensor = None
        self._dof_vel_view: torch.Tensor = None
        self._contact_forces_t: torch.Tensor = None
        self._geom_friction_t: torch.Tensor = None
        self._pair_friction_t: torch.Tensor = None
        self._randomize_contact_friction = False
        self._randomize_link_mass = False
        self._body_mass_native_t: torch.Tensor = None
        self._body_inertia_native_t: torch.Tensor = None

    # ── State tensors ──────────────────────────────────────────────────────────

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos_view

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel_view

    @property
    def dof_state(self) -> torch.Tensor:
        # Live view into a canonical assembled buffer refreshed every step/reset by
        # _sync_assembled_states.  qpos/qvel live in separate Warp arrays so a
        # true zero-copy interleaved view is impossible — but the task caches
        # this reference once at init (fixed_robot._init_buffers), so returning
        # a per-call torch.stack copy left self.dof_state frozen at the init
        # zeros, and any reward/obs reading dof_state directly (e.g. pendulum's
        # _reward_equilibrium) saw a fabricated near-zero error.  Resets still
        # scatter writes from these public views into native qpos/qvel.
        return self._dof_state_t.view(self._num_envs * self._num_dof, 2)

    @property
    def root_states(self) -> torch.Tensor:
        return self._root_states_t

    @property
    def rigid_body_states(self) -> torch.Tensor:
        return self._rigid_body_states_t.view(self._num_envs * self._num_bodies, 13)

    def _sync_assembled_states(self) -> None:
        """Refresh the assembled scratch tensors from the zero-copy views.

        Must be called whenever the sim state changes (step, resets): the
        task layer caches root_states / rigid_body_states once at init, so
        a lazy getter-side refresh leaves training on frozen observations.

        """
        qpos_native = self._qpos_t[:, self._qpos_offset :]
        qvel_native = self._qvel_t[:, self._qvel_offset :]
        self._dof_pos_view.copy_(
            qpos_native.index_select(1, self._canonical_to_native_dof)
        )
        self._dof_vel_view.copy_(
            qvel_native.index_select(1, self._canonical_to_native_dof)
        )
        if self._has_free_joint:
            rs = self._root_states_t
            rs[:, :3] = self._qpos_t[:, :3]
            rs[:, 3:7] = self._qpos_t[:, 3:7][:, WXYZ_TO_XYZW]
            rs[:, 7:10] = self._qvel_t[:, :3]
            # Free-joint angular qvel is body-local; public velocity is world-frame.
            rs[:, 10:13] = quat_apply(rs[:, 3:7], self._qvel_t[:, 3:6])
        rbs = self._rigid_body_states_t
        xpos = self._xpos_t.index_select(1, self._canonical_to_native_body)
        xquat = self._xquat_t.index_select(1, self._canonical_to_native_body)
        cvel = self._cvel_t.index_select(1, self._canonical_to_native_body)
        root_ids = self._body_rootid_t.index_select(
            0, self._canonical_to_native_body
        ).long()
        root_com = self._subtree_com_t.index_select(1, root_ids)
        angular_velocity = cvel[:, :, 0:3]
        linear_velocity = cvel[:, :, 3:6] - torch.cross(
            xpos - root_com, angular_velocity, dim=-1
        )
        rbs[:, :, 0:3] = xpos
        rbs[:, :, 3:7] = xquat[:, :, WXYZ_TO_XYZW]
        rbs[:, :, 7:10] = linear_velocity
        rbs[:, :, 10:13] = angular_velocity
        cfrc = self._cfrc_t.index_select(1, self._canonical_to_native_body)
        self._contact_forces_t.copy_(cfrc[..., 3:6])

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    @property
    def contact_friction(self) -> torch.Tensor:
        return self._contact_friction_t

    def set_contact_friction(
        self,
        env_ids: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        ids, values = self._prepare_contact_friction_update(env_ids, coefficients)
        if ids.numel() == 0:
            return
        if not self._randomize_contact_friction:
            raise RuntimeError(
                "contact-friction randomization was not enabled before setup"
            )
        self._contact_friction_t[ids] = values
        self._geom_friction_t[ids, :, 0] = values[:, None]
        if self._pair_friction_t is not None:
            self._pair_friction_t[ids, :, 0:2] = values[:, None, None]

    def set_link_mass_scale(
        self,
        env_ids: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        ids, values = self._prepare_link_mass_scale_update(env_ids, scales)
        if ids.numel() == 0:
            return
        if not self._randomize_link_mass:
            raise RuntimeError("link-mass randomization was not enabled before setup")

        masses = self._nominal_link_mass * values
        inertias = self._nominal_link_inertia * values.unsqueeze(-1)
        self._link_mass_t[ids] = masses
        self._link_inertia_t[ids] = inertias
        body_ids = self._canonical_to_native_body
        self._body_mass_native_t[ids[:, None], body_ids[None, :]] = masses
        self._body_inertia_native_t[ids[:, None], body_ids[None, :]] = inertias

        with self._wp_ctx:
            mjw.set_const(self._m, self._d)
        self._sync_assembled_states()

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        self._device = device
        self._num_envs = num_envs
        self._randomize_contact_friction = contact_friction_range(cfg) is not None
        self._randomize_link_mass = link_mass_scale_range(cfg) is not None

        wp.init()
        self._wp_ctx = wp.ScopedDevice(device)

        mjm = self._load_model(cfg)
        self._configure_model(mjm, cfg, device)
        self._run_task_callbacks(mjm, task)
        self._initialize_link_properties(mjm, num_envs, device)

        # Build Warp model and batched data inside the device scope
        with self._wp_ctx:
            batch_sizes = {}
            if self._randomize_contact_friction:
                batch_sizes["geom_friction"] = num_envs
                if mjm.npair:
                    batch_sizes["pair_friction"] = num_envs
            if self._randomize_link_mass:
                for name in (
                    "body_mass",
                    "body_inertia",
                    "body_subtreemass",
                    "body_invweight0",
                    "dof_invweight0",
                ):
                    batch_sizes[name] = num_envs
            self._m = mjw.put_model(mjm, batch_sizes=batch_sizes or None)
            mjd = mujoco.MjData(mjm)
            # mujoco-warp ignores the legacy mjModel.njmax field; forward it
            # (cfg.mujoco.njmax → spec → mjm → put_data).  -1
            # means unset → let warp use its own heuristic.
            njmax = mjm.njmax if mjm.njmax > 0 else None
            self._d = mjw.put_data(mjm, mjd, nworld=num_envs, njmax=njmax)

            # Zero-copy torch views
            self._qpos_t = wp.to_torch(self._d.qpos)
            self._qvel_t = wp.to_torch(self._d.qvel)
            self._qfrc_t = wp.to_torch(self._d.qfrc_applied)
            self._cfrc_t = wp.to_torch(self._d.cfrc_ext)
            self._xpos_t = wp.to_torch(self._d.xpos)
            self._xquat_t = wp.to_torch(self._d.xquat)
            self._cvel_t = wp.to_torch(self._d.cvel)
            self._subtree_com_t = wp.to_torch(self._d.subtree_com)
            self._body_rootid_t = wp.to_torch(self._m.body_rootid)
            if self._randomize_contact_friction:
                self._geom_friction_t = wp.to_torch(self._m.geom_friction)
                if mjm.npair:
                    self._pair_friction_t = wp.to_torch(self._m.pair_friction)
            if self._randomize_link_mass:
                self._body_mass_native_t = wp.to_torch(self._m.body_mass)
                self._body_inertia_native_t = wp.to_torch(self._m.body_inertia)

        # Scratch tensors for assembled state
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0
        nb = self._num_bodies
        self._rigid_body_states_t = torch.zeros(num_envs, nb, 13, device=device)
        self._rigid_body_states_t[:, :, 6] = 1.0
        # Assembled [N, num_dof, 2] dof_state buffer — qpos/qvel are separate
        # Warp arrays, so this is the only way dof_state can stay a live view.
        self._dof_state_t = torch.zeros(num_envs, self._num_dof, 2, device=device)
        self._dof_pos_view = self._dof_state_t[..., 0]
        self._dof_vel_view = self._dof_state_t[..., 1]
        self._contact_forces_t = torch.zeros(
            num_envs, self._num_bodies, 3, device=device
        )
        self._contact_friction_t = torch.full(
            (num_envs,), self._nominal_contact_friction, device=device
        )

        # Tensors must be valid immediately after setup() (tasks cache them
        # during _init_buffers, before the first step).
        self._sync_assembled_states()

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        with self._wp_ctx:
            off = self._qvel_offset
            native_torques = torques.index_select(1, self._native_to_canonical_dof)
            if off > 0:
                self._qfrc_t[:, off:].copy_(native_torques)
            else:
                self._qfrc_t.copy_(native_torques)
            mjw.forward(self._m, self._d)
            mjw.euler(self._m, self._d)
            # cfrc_ext is only populated with constraint/contact forces by
            # rne_postconstraint; forward+euler alone leave it at zero.
            mjw.rne_postconstraint(self._m, self._d)
            # Euler updates qpos/qvel after forward has assembled xpos/xquat
            # and cvel. Refresh those derived fields without recomputing the
            # collision, constraint, or acceleration stages.
            mjw.kinematics(self._m, self._d)
            mjw.com_pos(self._m, self._d)
            mjw.com_vel(self._m, self._d)
        self._sync_assembled_states()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_state(self, reset_mask: torch.Tensor) -> None:
        mask = reset_mask.unsqueeze(1)
        clamped_pos = self._clamp_dof_positions(self._dof_pos_view)
        torch.where(
            mask,
            clamped_pos,
            self._dof_pos_view,
            out=self._dof_pos_view,
        )
        native_pos = self._dof_pos_view.index_select(
            1,
            self._native_to_canonical_dof,
        )
        native_vel = self._dof_vel_view.index_select(
            1,
            self._native_to_canonical_dof,
        )
        torch.where(
            mask,
            native_pos,
            self._qpos_t[:, self._qpos_offset :],
            out=self._qpos_t[:, self._qpos_offset :],
        )
        torch.where(
            mask,
            native_vel,
            self._qvel_t[:, self._qvel_offset :],
            out=self._qvel_t[:, self._qvel_offset :],
        )
        if self._has_free_joint:
            rs = self._root_states_t
            torch.where(mask, rs[:, :3], self._qpos_t[:, :3], out=self._qpos_t[:, :3])
            torch.where(
                mask,
                rs[:, 3:7][:, XYZW_TO_WXYZ],
                self._qpos_t[:, 3:7],
                out=self._qpos_t[:, 3:7],
            )
            torch.where(mask, rs[:, 7:10], self._qvel_t[:, :3], out=self._qvel_t[:, :3])
            torch.where(
                mask,
                # The requested orientation may change in this same reset.
                quat_rotate_inverse(rs[:, 3:7], rs[:, 10:13]),
                self._qvel_t[:, 3:6],
                out=self._qvel_t[:, 3:6],
            )

        with self._wp_ctx:
            mjw.forward(self._m, self._d)
        self._sync_assembled_states()

    def set_all_root_states(self) -> None:
        if not self._has_free_joint:
            return
        rs = self._root_states_t
        self._qpos_t[:, :3].copy_(rs[:, :3])
        self._qpos_t[:, 3:7].copy_(rs[:, 3:7][:, XYZW_TO_WXYZ])
        self._qvel_t[:, :3].copy_(rs[:, 7:10])
        self._qvel_t[:, 3:6].copy_(quat_rotate_inverse(rs[:, 3:7], rs[:, 10:13]))
        with self._wp_ctx:
            mjw.forward(self._m, self._d)
        self._sync_assembled_states()
