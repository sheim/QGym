"""Draws env.commands as three indicators above the robot's base:

    * forward arrow (red) — vel_x along the robot's body-x axis
    * strafe arrow (green) — vel_y along the robot's body-y (left) axis
    * yaw arc (blue) — circular arrow around the vertical axis, length set by
      the heading change after `yaw_arc_scale` seconds at current yaw_rate

Anchor is `height_offset` above the base. Arrows scale with `scale` (m per
m/s); the arc radius is fixed and its span is the angle that would be swept
in `yaw_arc_scale` seconds.
"""

import numpy as np
import mujoco


_FORWARD_RGBA = np.array([1.0, 0.30, 0.30, 1.0], dtype=np.float32)
_STRAFE_RGBA = np.array([0.30, 0.85, 0.35, 1.0], dtype=np.float32)
_YAW_RGBA = np.array([0.25, 0.55, 1.0, 1.0], dtype=np.float32)


def _add_connector(viewer, geom_type, width, start, end, rgba):
    if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom, int(geom_type), np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba
    )
    mujoco.mjv_connector(
        geom,
        int(geom_type),
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    viewer.user_scn.ngeom += 1


class CommandVisualizer:
    def __init__(
        self,
        env,
        scale=0.3,
        height_offset=0.4,
        shaft_radius=0.015,
        yaw_radius=0.18,
        yaw_arc_scale=0.5,
        arc_segments_per_rad=12,
    ):
        self.env = env
        self.mjd = env._backend._datas[0]
        self.scale = scale
        self.height_offset = height_offset
        self.shaft_radius = shaft_radius
        self.yaw_radius = yaw_radius
        self.yaw_arc_scale = yaw_arc_scale
        self.arc_segments_per_rad = arc_segments_per_rad
        env._backend._viewer_overlay_fn = self._draw

    def _draw(self, viewer):
        viewer.user_scn.ngeom = 0
        cmd = self.env.commands[0].detach().cpu().numpy()
        vx = float(cmd[0])
        vy = float(cmd[1]) if cmd.shape[0] > 1 else 0.0
        yaw_rate = float(cmd[2]) if cmd.shape[0] > 2 else 0.0

        base_pos = np.array(self.mjd.qpos[0:3], dtype=np.float64)
        qw, qx, qy, qz = (float(self.mjd.qpos[i]) for i in (3, 4, 5, 6))
        yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        )
        c, s = np.cos(yaw), np.sin(yaw)
        fwd_world = np.array([c, s, 0.0])
        left_world = np.array([-s, c, 0.0])

        anchor = base_pos + np.array([0.0, 0.0, self.height_offset])

        if abs(vx) > 1e-4:
            end = anchor + fwd_world * vx * self.scale
            _add_connector(
                viewer,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self.shaft_radius,
                anchor,
                end,
                _FORWARD_RGBA,
            )

        if abs(vy) > 1e-4:
            end = anchor + left_world * vy * self.scale
            _add_connector(
                viewer,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self.shaft_radius,
                anchor,
                end,
                _STRAFE_RGBA,
            )

        if abs(yaw_rate) > 1e-4:
            self._draw_yaw_arc(viewer, anchor, yaw, yaw_rate)

    def _draw_yaw_arc(self, viewer, center, robot_yaw, yaw_rate):
        # Arc sweeps the heading change after yaw_arc_scale seconds, clamped
        # to ±π so absurd yaw_rate values stay readable.
        arc_angle = max(-np.pi, min(np.pi, yaw_rate * self.yaw_arc_scale))
        n_seg = max(2, int(abs(arc_angle) * self.arc_segments_per_rad))

        thetas = np.linspace(0.0, arc_angle, n_seg + 1) + robot_yaw
        z = center[2] + 0.05  # lift slightly above the planar arrows
        pts = np.stack(
            [
                center[0] + self.yaw_radius * np.cos(thetas),
                center[1] + self.yaw_radius * np.sin(thetas),
                np.full_like(thetas, z),
            ],
            axis=1,
        )

        for i in range(n_seg - 1):
            _add_connector(
                viewer,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                0.008,
                pts[i],
                pts[i + 1],
                _YAW_RGBA,
            )
        _add_connector(
            viewer,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.012,
            pts[n_seg - 1],
            pts[n_seg],
            _YAW_RGBA,
        )
