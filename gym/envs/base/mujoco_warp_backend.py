"""MuJoCo Warp backend — fully vectorised GPU/CPU execution via mujoco_warp.

Requires mujoco >= 3.6 and mujoco_warp >= 3.6.

Step pipeline (mirrors mj_step):
    qfrc_applied[:, offset:] = torques
    mjw.forward(m, d)             # position + velocity + actuation + acceleration
    mjw.euler(m, d)               # semi-implicit Euler integration
    mjw.rne_postconstraint(m, d)  # populate cfrc_ext (contact forces)
    _sync_assembled_states()      # refresh root_states / rigid_body_states

State tensors are zero-copy torch views into Warp arrays (via wp.to_torch),
so writes to dof_pos / dof_vel are immediately visible in the Warp sim.
root_states and rigid_body_states are assembled (swizzled) copies: they are
re-synced in step() and the reset methods, never lazily in their getters —
the task layer caches these tensors once at init and expects in-place
updates (SimBackend contract: all tensors live after step() returns).
"""

import torch

from gym.envs.base.mujoco_backend_base import (
    MuJocoBackendBase,
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
)


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
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, nbody, 13]

    # ── State tensors ──────────────────────────────────────────────────────────

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._qpos_t[:, self._qpos_offset :]

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._qvel_t[:, self._qvel_offset :]

    @property
    def dof_state(self) -> torch.Tensor:
        # Per-call copy (qpos/qvel live in separate Warp arrays, so an
        # interleaved view is impossible).  Read-only convenience — resets
        # must go through the dof_pos / dof_vel zero-copy views.
        return torch.stack([self.dof_pos, self.dof_vel], dim=-1).view(
            self._num_envs * self._num_dof, 2
        )

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
        if self._has_free_joint:
            rs = self._root_states_t
            rs[:, :3] = self._qpos_t[:, :3]
            rs[:, 3:7] = self._qpos_t[:, 3:7][:, WXYZ_TO_XYZW]
            rs[:, 7:10] = self._qvel_t[:, :3]
            rs[:, 10:13] = self._qvel_t[:, 3:6]
        rbs = self._rigid_body_states_t
        rbs[:, :, 0:3] = self._xpos_t
        rbs[:, :, 3:7] = self._xquat_t[:, :, WXYZ_TO_XYZW]
        rbs[:, :, 7:10] = self._cvel_t[:, :, 3:6]
        rbs[:, :, 10:13] = self._cvel_t[:, :, 0:3]

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._cfrc_t[..., 3:6]

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        self._device = device
        self._num_envs = num_envs

        wp.init()
        self._wp_ctx = wp.ScopedDevice(device)

        mjm = self._load_model(cfg)
        self._configure_model(mjm, cfg, device)
        self._run_task_callbacks(mjm, task)

        # Build Warp model and batched data inside the device scope
        with self._wp_ctx:
            self._m = mjw.put_model(mjm)
            mjd = mujoco.MjData(mjm)
            # mujoco-warp ignores the legacy mjModel.njmax field; forward it
            # (cfg.mjspec_attributes.njmax → spec → mjm → put_data).  -1
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

        # Scratch tensors for assembled state
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0
        nb = mjm.nbody
        self._rigid_body_states_t = torch.zeros(num_envs, nb, 13, device=device)
        self._rigid_body_states_t[:, :, 6] = 1.0

        # Tensors must be valid immediately after setup() (tasks cache them
        # during _init_buffers, before the first step).
        self._sync_assembled_states()

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        import mujoco_warp as mjw

        with self._wp_ctx:
            off = self._qvel_offset
            if off > 0:
                self._qfrc_t[:, off:].copy_(torques)
            else:
                self._qfrc_t.copy_(torques)
            mjw.forward(self._m, self._d)
            mjw.euler(self._m, self._d)
            # cfrc_ext is only populated with constraint/contact forces by
            # rne_postconstraint; forward+euler alone leave it at zero.
            mjw.rne_postconstraint(self._m, self._d)
        self._sync_assembled_states()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        import mujoco_warp as mjw

        with self._wp_ctx:
            mjw.forward(self._m, self._d)
        self._sync_assembled_states()

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        if not self._has_free_joint:
            return
        rs = self._root_states_t[env_ids]
        self._qpos_t[env_ids, :3] = rs[:, :3]
        self._qpos_t[env_ids, 3:7] = rs[:, 3:7][:, XYZW_TO_WXYZ]
        self._qvel_t[env_ids, :3] = rs[:, 7:10]
        self._qvel_t[env_ids, 3:6] = rs[:, 10:13]

        import mujoco_warp as mjw

        with self._wp_ctx:
            mjw.forward(self._m, self._d)
        self._sync_assembled_states()

    def set_all_root_states(self) -> None:
        self.reset_root_state(torch.arange(self._num_envs, device=self._device))
