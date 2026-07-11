"""Keyboard teleop for the MuJoCo passive viewer.

Mirrors the IsaacGym KeyboardInterface: discrete increments per keypress
driving env.commands (vel_x, vel_y, yaw_rate). Wires into the viewer via
mujoco.viewer.launch_passive's key_callback, set on the backend before
the first render() call.

Key layout:
  Up / Down   forward / back   (vel_x)
  , / .       strafe left/right (vel_y)
  Left/Right  yaw left/right   (yaw_rate)
  R           reset all envs
  Esc / close window   quit (handled by viewer)
"""

import torch


# GLFW key codes — hardcoded to avoid a glfw dep. Values are stable across versions.
KEY_R = 82
KEY_COMMA = 44
KEY_PERIOD = 46
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265


class MujocoKeyboardInterface:
    def __init__(self, env):
        self.env = env

        self.max_vel_forward = 4.0
        self.max_vel_backward = -1.0
        self.increment_x = (self.max_vel_forward - self.max_vel_backward) * 0.1

        self.max_vel_sideways = 1.0
        self.increment_y = self.max_vel_sideways * 0.2

        self.max_vel_yaw = 2.0
        self.increment_yaw = self.max_vel_yaw * 0.2

        env.commands[:] = 0.0
        env.commands[:, 0] = 1.0  # seed with forward velocity so arrow is visible
        if hasattr(env.cfg, "commands"):
            env.cfg.commands.resampling_time = env.max_episode_length_s + 1

        env._backend._viewer_key_callback = self._on_key

        print("______________________________________________________________")
        print("Keyboard teleop (MuJoCo viewer)")
        print("  Up/Down       forward / back")
        print("  Left/Right    yaw left / yaw right")
        print("  , / .         strafe left / strafe right")
        print("  R             reset envs")
        print("  Esc / window  quit")
        print("  commands step in 1/5 increments of max")
        print("______________________________________________________________")

    def _on_key(self, keycode: int) -> None:
        c = self.env.commands
        if keycode == KEY_UP:
            c[:, 0] = torch.clamp(c[:, 0] + self.increment_x, max=self.max_vel_forward)
        elif keycode == KEY_DOWN:
            c[:, 0] = torch.clamp(c[:, 0] - self.increment_x, min=self.max_vel_backward)
        elif keycode == KEY_COMMA:
            c[:, 1] = torch.clamp(c[:, 1] + self.increment_y, max=self.max_vel_sideways)
        elif keycode == KEY_PERIOD:
            c[:, 1] = torch.clamp(
                c[:, 1] - self.increment_y, min=-self.max_vel_sideways
            )
        elif keycode == KEY_LEFT:
            c[:, 2] = torch.clamp(c[:, 2] + self.increment_yaw, max=self.max_vel_yaw)
        elif keycode == KEY_RIGHT:
            c[:, 2] = torch.clamp(c[:, 2] - self.increment_yaw, min=-self.max_vel_yaw)
        elif keycode == KEY_R:
            self.env.timed_out[:] = True
            self.env.reset()
