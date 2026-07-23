"""Geometry of the vsim command indicators.

vsim only draws polylines, so arrows are point lists — pure Python, tested
here with a fake backend (no GPU, license, or window; runs in the default
suite).  What matters: the arrows point where the command says, flip when
the command is negative, and hide when the command is ~zero.
"""

import math

import torch

from gym.utils.interfaces.VsimCommandVisualizer import (
    VsimCommandVisualizer,
    arrow_points,
    yaw_arc_points,
    yaw_from_quat_xyzw,
)


class FakeLine:
    def __init__(self, points):
        self.points = list(points)
        self.visible = True


class FakeBackend:
    def __init__(self):
        self._render_hooks = []
        self.lines = []

    def add_render_hook(self, fn):
        self._render_hooks.append(fn)

    def create_debug_line(self, points, color=(1, 1, 1), width=2.0):
        line = FakeLine(points)
        line.color = color
        self.lines.append(line)
        return line

    def update_debug_line(self, line, points, visible=True):
        line.visible = visible
        if visible and points:
            line.points = list(points)


class FakeEnv:
    def __init__(self):
        self.commands = torch.zeros(1, 3)
        # pos (0,0,0.3), identity quat scalar-last, zero velocities
        self.root_states = torch.tensor(
            [[0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0]]
        )
        self._backend = FakeBackend()


# ── Pure geometry ────────────────────────────────────────────────────────


def test_yaw_from_identity_quat_is_zero():
    assert abs(yaw_from_quat_xyzw(0.0, 0.0, 0.0, 1.0)) < 1e-9


def test_yaw_from_quarter_turn_quat():
    # 90° about +Z, scalar-last
    h = math.sqrt(0.5)
    assert abs(yaw_from_quat_xyzw(0.0, 0.0, h, h) - math.pi / 2) < 1e-6


def test_arrow_tip_follows_direction_and_length():
    pts = arrow_points((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 2.0)
    tail, tip = pts[0], pts[1]
    assert tail == (0.0, 0.0, 1.0)
    assert abs(tip[0] - 2.0) < 1e-9 and abs(tip[1]) < 1e-9
    assert len(pts) == 5  # tail, tip, barb, tip, barb


def test_negative_length_points_backwards():
    fwd = arrow_points((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
    back = arrow_points((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.0)
    assert fwd[1][0] > 0 and back[1][0] < 0


def test_arrow_barbs_sit_behind_the_tip():
    """Barbs must trail the tip, or the head renders inside-out."""
    pts = arrow_points((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
    tip = pts[1]
    for barb in (pts[2], pts[4]):
        assert barb[0] < tip[0]
    assert abs(pts[2][1] + pts[4][1]) < 1e-9  # symmetric about the shaft


def test_yaw_arc_spans_expected_angle():
    center, radius = (0.0, 0.0, 0.0), 1.0
    pts = yaw_arc_points(center, 0.0, 2.0, radius, arc_scale=0.5)  # 1.0 rad
    start_angle = math.atan2(pts[0][1], pts[0][0])
    assert abs(start_angle) < 1e-9
    # the arc body (before the arrowhead) ends near 1.0 rad
    angles = [math.atan2(p[1], p[0]) for p in pts[:-2]]
    assert abs(max(angles) - 1.0) < 0.05


def test_yaw_arc_reverses_for_negative_rate():
    pos = yaw_arc_points((0.0, 0.0, 0.0), 0.0, 1.0, 1.0, 0.5)
    neg = yaw_arc_points((0.0, 0.0, 0.0), 0.0, -1.0, 1.0, 0.5)
    assert math.atan2(pos[2][1], pos[2][0]) > 0
    assert math.atan2(neg[2][1], neg[2][0]) < 0


# ── Wiring ───────────────────────────────────────────────────────────────


def test_creates_three_lines_and_registers_hook():
    env = FakeEnv()
    viz = VsimCommandVisualizer(env)
    assert len(env._backend.lines) == 3
    assert viz.draw in env._backend._render_hooks


def test_indicators_hidden_when_commands_zero():
    env = FakeEnv()
    viz = VsimCommandVisualizer(env)
    env.commands[:] = 0.0
    viz.draw()
    assert not any(line.visible for line in env._backend.lines)


def test_each_command_axis_shows_its_own_indicator():
    env = FakeEnv()
    viz = VsimCommandVisualizer(env)
    forward, strafe, yaw = env._backend.lines

    for col, shown in ((0, forward), (1, strafe), (2, yaw)):
        env.commands[:] = 0.0
        env.commands[0, col] = 1.0
        viz.draw()
        assert shown.visible, f"commands[:, {col}] did not show its indicator"
        others = [ln for ln in env._backend.lines if ln is not shown]
        assert not any(ln.visible for ln in others)


def test_arrows_follow_robot_heading():
    """Rotating the base must rotate the forward arrow with it."""
    env = FakeEnv()
    viz = VsimCommandVisualizer(env)
    env.commands[:] = 0.0
    env.commands[0, 0] = 1.0

    viz.draw()
    tip_facing_x = env._backend.lines[0].points[1]
    assert tip_facing_x[0] > 0 and abs(tip_facing_x[1]) < 1e-6

    h = math.sqrt(0.5)  # yaw 90° about +Z
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, h, h])
    viz.draw()
    tip_facing_y = env._backend.lines[0].points[1]
    assert tip_facing_y[1] > 0 and abs(tip_facing_y[0]) < 1e-6


def test_anchor_sits_above_the_base():
    env = FakeEnv()
    viz = VsimCommandVisualizer(env)
    env.commands[0, 0] = 1.0
    viz.draw()
    tail = env._backend.lines[0].points[0]
    base_z = float(env.root_states[0, 2])
    assert abs(tail[2] - (base_z + viz.height_offset)) < 1e-9
