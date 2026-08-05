"""Plain MuJoCo (mj_step) backend — no Warp dependency.

Works on any platform (Linux CPU, Mac Apple Silicon).  One shared MjModel,
one MjData per environment; physics runs in a Python loop and state is copied
numpy→torch after each step.
"""

import torch
import mujoco

from gym.envs.base.mujoco_backend_base import (
    MuJocoBackendBase,
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
)


class MuJocoCPUBackend(MuJocoBackendBase):
    """SimBackend backed by plain mujoco.mj_step."""

    def __init__(self) -> None:
        super().__init__()
        self._datas: list = []

        # State tensors (allocated in setup)
        self._dof_state_t: torch.Tensor = None  # [N, num_dof, 2]
        self._dof_pos_view: torch.Tensor = None  # [N, num_dof] view into above
        self._dof_vel_view: torch.Tensor = None  # [N, num_dof] view into above
        self._root_states_t: torch.Tensor = None  # [N, 13]
        self._rigid_body_states_t: torch.Tensor = None  # [N, num_bodies, 13]
        self._contact_forces_t: torch.Tensor = None  # [N, num_bodies, 3]

        # Viewer (created lazily on first render call)
        self._viewer = None
        self._show_ui = False  # overridden from cfg.viewer.show_ui in setup()
        self._viewer_key_callback = None
        self._viewer_overlay_fn = None  # called each render() before sync()

    # ── State tensors ──────────────────────────────────────────────────────────

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

    # ── World building ─────────────────────────────────────────────────────────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        self._device = device
        self._num_envs = num_envs

        mjm = self._load_model(cfg)
        self._configure_model(mjm, cfg, device)
        self._run_task_callbacks(mjm, task)

        viewer_cfg = getattr(cfg, "viewer", None)
        self._show_ui = bool(getattr(viewer_cfg, "show_ui", False))

        # Create one MjData per environment
        self._datas = [mujoco.MjData(mjm) for _ in range(num_envs)]

        # Allocate PyTorch state tensors
        self._dof_state_t = torch.zeros(num_envs, self._num_dof, 2, device=device)
        self._dof_pos_view = self._dof_state_t[..., 0]
        self._dof_vel_view = self._dof_state_t[..., 1]
        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        self._root_states_t[:, 6] = 1.0  # identity quaternion (scalar-last, w=1)
        self._rigid_body_states_t = torch.zeros(
            num_envs, self._num_bodies, 13, device=device
        )
        self._rigid_body_states_t[:, :, 6] = 1.0
        self._contact_forces_t = torch.zeros(
            num_envs, self._num_bodies, 3, device=device
        )

    # ── Per-step ───────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        torques_np = torques.cpu().numpy()[:, self._native_to_canonical_dof_np]
        off = self._qvel_offset
        for i, d in enumerate(self._datas):
            d.qfrc_applied[off:] = torques_np[i]
            mujoco.mj_step(self._mjm, d)
        self._sync_state_from_mujoco()

    def _sync_state_from_mujoco(self) -> None:
        qoff = self._qpos_offset
        voff = self._qvel_offset
        dof_order = self._canonical_to_native_dof_np
        body_order = self._canonical_to_native_body_np
        for i, d in enumerate(self._datas):
            # cfrc_ext is only populated with constraint/contact forces by
            # mj_rnePostConstraint; mj_step alone leaves it at zero.
            mujoco.mj_rnePostConstraint(self._mjm, d)
            self._dof_pos_view[i] = torch.from_numpy(d.qpos[qoff:][dof_order].copy())
            self._dof_vel_view[i] = torch.from_numpy(d.qvel[voff:][dof_order].copy())
            self._contact_forces_t[i] = torch.from_numpy(
                d.cfrc_ext[body_order, 3:6].copy()
            )
            # Rigid body states
            rbs = self._rigid_body_states_t[i]
            rbs[:, 0:3] = torch.from_numpy(d.xpos[body_order].copy())
            mj_quat = torch.from_numpy(d.xquat[body_order].copy())
            rbs[:, 3:7] = mj_quat[:, WXYZ_TO_XYZW]
            rbs[:, 7:10] = torch.from_numpy(d.cvel[body_order, 3:6].copy())
            rbs[:, 10:13] = torch.from_numpy(d.cvel[body_order, 0:3].copy())
        if self._has_free_joint:
            for i, d in enumerate(self._datas):
                self._root_states_t[i, :3] = torch.from_numpy(d.qpos[:3].copy())
                mj_quat = torch.from_numpy(d.qpos[3:7].copy())
                self._root_states_t[i, 3:7] = mj_quat[WXYZ_TO_XYZW]
                self._root_states_t[i, 7:10] = torch.from_numpy(d.qvel[:3].copy())
                self._root_states_t[i, 10:13] = torch.from_numpy(d.qvel[3:6].copy())

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        qoff = self._qpos_offset
        voff = self._qvel_offset
        for i in env_ids.tolist():
            self._datas[i].qpos[qoff:] = (
                self._dof_pos_view[i].cpu().numpy()[self._native_to_canonical_dof_np]
            )
            self._datas[i].qvel[voff:] = (
                self._dof_vel_view[i].cpu().numpy()[self._native_to_canonical_dof_np]
            )
            mujoco.mj_forward(self._mjm, self._datas[i])

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        if not self._has_free_joint:
            return
        for i in env_ids.tolist():
            rs = self._root_states_t[i].cpu()
            self._datas[i].qpos[:3] = rs[:3].numpy()
            self._datas[i].qpos[3:7] = rs[3:7][XYZW_TO_WXYZ].numpy()
            self._datas[i].qvel[:3] = rs[7:10].numpy()
            self._datas[i].qvel[3:6] = rs[10:13].numpy()
            mujoco.mj_forward(self._mjm, self._datas[i])

    def set_all_root_states(self) -> None:
        self.reset_root_state(torch.arange(self._num_envs))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, sync_frame_time: bool = True) -> None:
        import platform
        import mujoco.viewer

        if self._viewer is None:
            if platform.system() == "Darwin":
                import mujoco.viewer as _mjv

                if _mjv._MJPYTHON is None:
                    raise RuntimeError(
                        "MuJoCo passive viewer on macOS requires mjpython.\n"
                        "Run with: .venv/bin/mjpython scripts/train.py ...\n"
                        "Or use --headless to disable the viewer."
                    )

            # Indirect through a wrapper so callers can set _viewer_key_callback
            # after the viewer is up (LeggedRobot.__init__ calls reset()→step()
            # →_render() during construction, before any user interface installs).
            def _key_dispatch(keycode, _self=self):
                if _self._viewer_key_callback is not None:
                    _self._viewer_key_callback(keycode)

            # Side panels are hidden by default: their shortcuts bind most
            # letters to visualisation toggles, which fire alongside keyboard
            # teleop (see gym/utils/interfaces/teleop_bindings.py).
            # cfg.viewer.show_ui / play.py --viewer_ui restores them.
            self._viewer = mujoco.viewer.launch_passive(
                self._mjm,
                self._datas[0],
                key_callback=_key_dispatch,
                show_left_ui=self._show_ui,
                show_right_ui=self._show_ui,
            )
        if self._viewer.is_running():
            if self._viewer_overlay_fn is not None:
                self._viewer_overlay_fn(self._viewer)
            self._viewer.sync()
