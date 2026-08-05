"""Keyboard teleop: shared bindings + both viewers' dispatch logic.

vlearn polls key STATE ("is I held?") rather than delivering key EVENTS, so
VsimKeyboardInterface must act on rising edges only — otherwise a held key
ramps the command every frame instead of stepping it once per press.  The
MuJoCo interface gets real events and dispatches directly.

All of this is pure Python, so it is tested here with fakes: no GPU, no
license, no window (runs in the default suite).
"""

import pytest
import torch

from gym.utils.interfaces.teleop_bindings import (
    BINDINGS,
    KEY_TO_ACTION,
    TeleopCommands,
)
from gym.utils.interfaces.VsimKeyboardInterface import VsimKeyboardInterface
from gym.utils.interfaces.MujocoKeyboardInterface import (
    KEYCODE_TO_ACTION,
    MujocoKeyboardInterface,
)

# Keys the vsim viewer reserves: WASD fly the camera, P pauses, O steps.
VSIM_RESERVED = set("WASDPO")


class FakeRender:
    """Stands in for GymRender: reports whichever keys are 'held'."""

    def __init__(self):
        self.held = set()

    def is_key_down(self, key):
        return key in self.held


class FakeBackend:
    def __init__(self):
        self._render_hooks = []
        self._viewer_key_callback = None
        self.window_closed = False
        self.escape_key = "<ESC>"  # stands in for vlearn's UserKey.Escape

    def add_render_hook(self, fn):
        self._render_hooks.append(fn)


class FakeEnv:
    """Minimal env surface the interfaces touch."""

    def __init__(self, num_envs=2):
        self.commands = torch.zeros(num_envs, 3)
        self.timed_out = torch.zeros(num_envs, dtype=torch.bool)
        self.max_episode_length_s = 10.0
        self.cfg = type("cfg", (), {})()
        self._backend = FakeBackend()
        self.exit = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def _make_vsim():
    env = FakeEnv()
    return env, VsimKeyboardInterface(env), FakeRender()


# ── Shared bindings ──────────────────────────────────────────────────────


def test_bindings_avoid_vsim_reserved_keys():
    """The whole reason for the IJKL/NM layout: WASD flies vsim's camera."""
    clashes = VSIM_RESERVED & set(BINDINGS.values())
    assert not clashes, f"bindings collide with vsim-reserved keys: {clashes}"


def test_both_interfaces_use_the_same_bindings():
    """MuJoCo keycodes must be exactly ord() of the shared letters."""
    assert KEYCODE_TO_ACTION == {ord(k): a for k, a in KEY_TO_ACTION.items()}
    assert set(KEYCODE_TO_ACTION.values()) == set(BINDINGS.keys())


def test_keys_are_unique_and_alphanumeric():
    # vsim's is_key_down only accepts alphanumerics as strings.
    assert len(set(BINDINGS.values())) == len(BINDINGS)
    assert all(k.isalnum() and k.isupper() for k in BINDINGS.values())


# ── vsim: polling + edge detection ───────────────────────────────────────


def test_hook_is_registered_on_backend():
    env, ui, _ = _make_vsim()
    assert ui.poll in env._backend._render_hooks


def test_held_key_steps_command_once_per_press():
    """The whole point of edge detection: holding I for many frames = ONE step."""
    env, ui, render = _make_vsim()
    start = env.commands[0, 0].item()  # seeded to 1.0
    step = ui.commands.increment_x

    render.held = {BINDINGS["forward"]}
    for _ in range(10):  # held down for 10 frames
        ui.poll(render)
    assert env.commands[0, 0].item() == start + step

    render.held = set()  # release
    ui.poll(render)
    render.held = {BINDINGS["forward"]}  # press again
    ui.poll(render)
    assert env.commands[0, 0].item() == start + 2 * step


def test_r_resets_once_per_press():
    env, ui, render = _make_vsim()
    render.held = {BINDINGS["reset"]}
    for _ in range(5):
        ui.poll(render)
    assert env.reset_count == 1
    assert env.timed_out.all()


def test_escape_and_window_close_set_exit():
    env, ui, render = _make_vsim()
    ui.poll(render)
    assert env.exit is False

    render.held = {ui._escape_key}
    ui.poll(render)
    assert env.exit is True

    env2, ui2, render2 = _make_vsim()
    env2._backend.window_closed = True
    ui2.poll(render2)
    assert env2.exit is True


# ── MuJoCo: event dispatch ───────────────────────────────────────────────


def test_mujoco_callback_registered_and_dispatches():
    env = FakeEnv()
    ui = MujocoKeyboardInterface(env)
    assert env._backend._viewer_key_callback == ui._on_key

    start = env.commands[0, 0].item()
    ui._on_key(ord(BINDINGS["forward"]))
    assert env.commands[0, 0].item() == start + ui.commands.increment_x


def test_mujoco_ignores_unbound_keys():
    env = FakeEnv()
    ui = MujocoKeyboardInterface(env)
    before = env.commands.clone()
    ui._on_key(ord("Z"))  # not bound
    assert torch.equal(env.commands, before)


# ── Command semantics (shared by both) ───────────────────────────────────


def test_direction_actions_move_expected_axes():
    env = FakeEnv()
    cmds = TeleopCommands(env)
    for action, col, sign in (
        ("forward", 0, +1),
        ("back", 0, -1),
        ("strafe_left", 1, +1),
        ("strafe_right", 1, -1),
        ("yaw_left", 2, +1),
        ("yaw_right", 2, -1),
    ):
        env.commands[:] = 0.0
        cmds.apply(action)
        value = env.commands[0, col].item()
        assert value * sign > 0, f"{action} moved commands[:, {col}] the wrong way"


def test_commands_clamp_at_limits():
    env = FakeEnv()
    cmds = TeleopCommands(env)
    for _ in range(200):
        cmds.apply("forward")
    assert env.commands[0, 0].item() <= cmds.max_vel_forward + 1e-6
    for _ in range(400):
        cmds.apply("back")
    assert env.commands[0, 0].item() >= cmds.max_vel_backward - 1e-6


# ── Viewer UI panels (MuJoCo) ────────────────────────────────────────────


def _pendulum_cfg(show_ui):
    import os
    import types

    from gym import GYM_ROOT_DIR

    return types.SimpleNamespace(
        asset=types.SimpleNamespace(
            file=os.path.join(
                GYM_ROOT_DIR,
                "resources",
                "robots",
                "pendulum",
                "urdf",
                "pendulum.urdf",
            ),
            joint_damping=0.1,
            rotor_inertia=0.0,
            disable_gravity=False,
            penalize_contacts_on=[],
            terminate_after_contacts_on=[],
        ),
        sim=types.SimpleNamespace(gravity=[0.0, 0.0, -9.81]),
        sim_dt=0.005,
        viewer=types.SimpleNamespace(show_ui=show_ui),
    )


def test_setup_reads_show_ui_from_cfg():
    pytest.importorskip("mujoco")
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    for show_ui in (False, True):
        b = MuJocoCPUBackend()
        b.setup(_pendulum_cfg(show_ui), num_envs=1, device="cpu", task=None)
        assert b._show_ui is show_ui


def test_viewer_launched_with_panels_hidden(monkeypatch):
    """The panels must actually be suppressed at launch_passive."""
    pytest.importorskip("mujoco")
    import mujoco.viewer

    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    captured = {}

    class _StubViewer:
        def is_running(self):
            return True

        def sync(self):
            pass

    def fake_launch(model, data, **kwargs):
        captured.update(kwargs)
        return _StubViewer()

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_launch)

    b = MuJocoCPUBackend()
    b.setup(_pendulum_cfg(show_ui=False), num_envs=1, device="cpu", task=None)
    b.render()
    assert captured["show_left_ui"] is False
    assert captured["show_right_ui"] is False

    captured.clear()
    b2 = MuJocoCPUBackend()
    b2.setup(_pendulum_cfg(show_ui=True), num_envs=1, device="cpu", task=None)
    b2.render()
    assert captured["show_left_ui"] is True
    assert captured["show_right_ui"] is True


def test_viewer_configs_default_show_ui_false():
    from gym.envs.base.legged_robot_config import LeggedRobotCfg
    from gym.envs.base.fixed_robot_config import FixedRobotCfg

    assert LeggedRobotCfg.viewer.show_ui is False
    assert FixedRobotCfg.viewer.show_ui is False
