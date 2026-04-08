"""MuJoCo Warp backend — fully vectorised GPU/CPU execution via mujoco_warp.

Requires mujoco >= 3.6 and mujoco_warp >= 3.6.

Step pipeline (mirrors mj_step):
    qfrc_applied[:, offset:] = torques
    mjw.forward(m, d)   # position + velocity + actuation + acceleration
    mjw.euler(m, d)     # semi-implicit Euler integration

State tensors are zero-copy torch views into Warp arrays (via wp.to_torch),
so writes to dof_pos / dof_vel are immediately visible in the Warp sim.
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
        return torch.stack([self.dof_pos, self.dof_vel], dim=-1).view(
            self._num_envs * self._num_dof, 2
        )

    @property
    def root_states(self) -> torch.Tensor:
        if not self._has_free_joint:
            return self._root_states_t
        rs = self._root_states_t
        rs[:, :3] = self._qpos_t[:, :3]
        rs[:, 3:7] = self._qpos_t[:, 3:7][:, WXYZ_TO_XYZW]
        rs[:, 7:10] = self._qvel_t[:, :3]
        rs[:, 10:13] = self._qvel_t[:, 3:6]
        return rs

    @property
    def rigid_body_states(self) -> torch.Tensor:
        rbs = self._rigid_body_states_t
        rbs[:, :, 0:3] = self._xpos_t
        rbs[:, :, 3:7] = self._xquat_t[:, :, WXYZ_TO_XYZW]
        rbs[:, :, 7:10] = self._cvel_t[:, :, 3:6]
        rbs[:, :, 10:13] = self._cvel_t[:, :, 0:3]
        return rbs.view(self._num_envs * self._num_bodies, 13)

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
            self._d = mjw.put_data(mjm, mjd, nworld=num_envs)

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

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        import mujoco_warp as mjw

        with self._wp_ctx:
            mjw.forward(self._m, self._d)

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

    def set_all_root_states(self) -> None:
        self.reset_root_state(torch.arange(self._num_envs, device=self._device))
