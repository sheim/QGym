"""Isolated-env patch: make Q2's Warp assembled-state sync stream-resident.

``MuJocoWarpBackend._sync_assembled_states`` reorders MuJoCo's scalar-first
quaternions with a Python-list index on a non-contiguous view of a Warp-backed
tensor:

    rs[:, 3:7] = self._qpos_t[:, 3:7][:, WXYZ_TO_XYZW]
    rbs[:, 3:7] = xquat[:, :, WXYZ_TO_XYZW]

Under torch 2.13 that advanced index makes PyTorch synchronise the stream, so
every ``step()`` blocks the host once (twice with the body quaternions). The
equivalent slice-only form below does not. This module is an experiment-scope
patch applied in the isolated checkout only; it is not a product change.
"""

from __future__ import annotations

import torch

from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend
from gym.utils.torch_quat import quat_apply


def _sync_assembled_states_slice_only(self) -> None:
    qpos_native = self._qpos_t[:, self._qpos_offset :]
    qvel_native = self._qvel_t[:, self._qvel_offset :]
    self._dof_pos_view.copy_(qpos_native.index_select(1, self._canonical_to_native_dof))
    self._dof_vel_view.copy_(qvel_native.index_select(1, self._canonical_to_native_dof))
    if self._has_free_joint:
        rs = self._root_states_t
        quat_wxyz = self._qpos_t[:, 3:7]
        rs[:, 0:3].copy_(self._qpos_t[:, 0:3])
        # (w, x, y, z) -> scalar-last (x, y, z, w)
        rs[:, 3:6].copy_(quat_wxyz[:, 1:4])
        rs[:, 6:7].copy_(quat_wxyz[:, 0:1])
        rs[:, 7:10].copy_(self._qvel_t[:, 0:3])
        rs[:, 10:13].copy_(quat_apply(rs[:, 3:7], self._qvel_t[:, 3:6]))
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
    rbs[:, :, 0:3].copy_(xpos)
    rbs[:, :, 3:6].copy_(xquat[:, :, 1:4])
    rbs[:, :, 6:7].copy_(xquat[:, :, 0:1])
    rbs[:, :, 7:10].copy_(linear_velocity)
    rbs[:, :, 10:13].copy_(angular_velocity)
    cfrc = self._cfrc_t.index_select(1, self._canonical_to_native_body)
    self._contact_forces_t.copy_(cfrc[..., 3:6])


_ORIGINAL = MuJocoWarpBackend._sync_assembled_states


def install():
    """Replace the assembled-state refresh with the slice-only equivalent."""
    MuJocoWarpBackend._sync_assembled_states = _sync_assembled_states_slice_only


def uninstall():
    MuJocoWarpBackend._sync_assembled_states = _ORIGINAL
