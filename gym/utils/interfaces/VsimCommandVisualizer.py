"""Draws env.commands above the robot in the vsim viewer.

Mirrors CommandVisualizer (MuJoCo) in meaning:

    * forward arrow (red)   — vel_x along the robot's body-x axis
    * strafe arrow (green)  — vel_y along the robot's body-y (left) axis
    * yaw arc (blue)        — arc around the vertical axis, spanning the
      heading change after `yaw_arc_scale` seconds at the current yaw rate

vsim has no arrow debug geom (only polylines), so each arrow is one
polyline that draws the shaft, then backtracks from the tip to sketch both
barbs: [tail, tip, barb_l, tip, barb_r].

Pose comes from `env.root_states` (position + scalar-last quaternion), which
is backend-agnostic, and the engine's Vec3/UserLine types stay behind
VSimBackend.create_debug_line / update_debug_line.
"""

import math

_FORWARD_RGB = (1.0, 0.30, 0.30)
_STRAFE_RGB = (0.30, 0.85, 0.35)
_YAW_RGB = (0.25, 0.55, 1.0)

_EPS = 1e-4


def yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Heading about +Z from a scalar-last quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def arrow_points(tail, direction, length, head_frac=0.25, head_angle=0.5):
    """Polyline tracing an arrow: shaft out, then both barbs from the tip.

    `direction` must be a unit vector; a negative `length` points the arrow
    backwards along it (so a reverse command reads as a backward arrow).
    """
    dx, dy, dz = direction
    tip = (tail[0] + dx * length, tail[1] + dy * length, tail[2] + dz * length)

    # Barbs sit in the horizontal plane, swept ±head_angle from the shaft.
    head_len = abs(length) * head_frac
    back = -math.copysign(1.0, length)
    ca, sa = math.cos(head_angle), math.sin(head_angle)
    barbs = []
    for sign in (+1.0, -1.0):
        # rotate the (backwards) shaft direction by ±head_angle about +Z
        bx = back * (dx * ca - sign * dy * sa)
        by = back * (dx * sign * sa + dy * ca)
        barbs.append((tip[0] + bx * head_len, tip[1] + by * head_len, tip[2]))

    return [tail, tip, barbs[0], tip, barbs[1]]


def yaw_arc_points(center, robot_yaw, yaw_rate, radius, arc_scale, segments_per_rad=12):
    """Arc sweeping the heading change after `arc_scale` seconds, plus a barb.

    Clamped to ±pi so absurd yaw rates stay readable.
    """
    arc_angle = max(-math.pi, min(math.pi, yaw_rate * arc_scale))
    n_seg = max(2, int(abs(arc_angle) * segments_per_rad))

    pts = []
    for i in range(n_seg + 1):
        theta = robot_yaw + arc_angle * i / n_seg
        pts.append(
            (
                center[0] + radius * math.cos(theta),
                center[1] + radius * math.sin(theta),
                center[2],
            )
        )

    # Arrowhead at the far end, tangent to the arc.
    end, prev = pts[-1], pts[-2]
    tangent = (end[0] - prev[0], end[1] - prev[1], 0.0)
    norm = math.hypot(tangent[0], tangent[1])
    if norm > _EPS:
        unit = (tangent[0] / norm, tangent[1] / norm, 0.0)
        head = arrow_points(prev, unit, norm, head_frac=1.5)
        pts.extend(head[2:])  # tip already present; append the barbs
    return pts


class VsimCommandVisualizer:
    def __init__(
        self,
        env,
        scale=0.3,
        height_offset=0.4,
        yaw_radius=0.18,
        yaw_arc_scale=0.5,
        line_width=3.0,
    ):
        self.env = env
        self.backend = env._backend
        self.scale = scale
        self.height_offset = height_offset
        self.yaw_radius = yaw_radius
        self.yaw_arc_scale = yaw_arc_scale

        placeholder = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        self._forward = self.backend.create_debug_line(
            placeholder, _FORWARD_RGB, line_width
        )
        self._strafe = self.backend.create_debug_line(
            placeholder, _STRAFE_RGB, line_width
        )
        self._yaw = self.backend.create_debug_line(placeholder, _YAW_RGB, line_width)

        self.backend.add_render_hook(self.draw)

    def draw(self, render=None) -> None:
        """Recompute the three indicators for env 0 (called per frame)."""
        cmd = self.env.commands[0].detach().cpu().tolist()
        vx = float(cmd[0])
        vy = float(cmd[1]) if len(cmd) > 1 else 0.0
        yaw_rate = float(cmd[2]) if len(cmd) > 2 else 0.0

        root = self.env.root_states[0].detach().cpu().tolist()
        base_pos = root[0:3]
        yaw = yaw_from_quat_xyzw(root[3], root[4], root[5], root[6])
        c, s = math.cos(yaw), math.sin(yaw)

        anchor = (base_pos[0], base_pos[1], base_pos[2] + self.height_offset)

        self.backend.update_debug_line(
            self._forward,
            arrow_points(anchor, (c, s, 0.0), vx * self.scale),
            visible=abs(vx) > _EPS,
        )
        self.backend.update_debug_line(
            self._strafe,
            arrow_points(anchor, (-s, c, 0.0), vy * self.scale),
            visible=abs(vy) > _EPS,
        )
        arc_center = (anchor[0], anchor[1], anchor[2] + 0.05)
        self.backend.update_debug_line(
            self._yaw,
            yaw_arc_points(
                arc_center, yaw, yaw_rate, self.yaw_radius, self.yaw_arc_scale
            ),
            visible=abs(yaw_rate) > _EPS,
        )
