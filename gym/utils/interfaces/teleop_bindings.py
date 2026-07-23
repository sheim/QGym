"""Shared keyboard teleop bindings and command logic for every viewer.

One table, one set of limits — so the MuJoCo and vsim interfaces cannot
drift apart.  Each interface only supplies the plumbing for its viewer's
input model (MuJoCo: key-event callback; vsim: polled key state).

Key choice (2026-07-23) — the constraint is what the viewers reserve:

* vsim hard-reserves **W A S D** (fly the camera), **P** (pause) and
  **O** (single-step).  Those are actively disruptive, which is why the
  original WASD scheme had to go.  It also only accepts ALPHANUMERIC keys
  as strings, so punctuation (the old MuJoCo `,`/`.` strafe) is impossible.
* MuJoCo's viewer binds nearly every letter to a visualisation toggle
  (I=inertia, J=joint, K=skybox, L=additive, M=CoM, N=constraint, …), so
  no letter is truly free there; a teleop press may also flip a vis flag.
  Launching the viewer with the side panels hidden disables those
  shortcuts if it ever becomes annoying.

That leaves the IJKL diamond, with yaw dropped onto N/M because the
natural U/O pair collides with vsim's step toggle:

      I              I / K   forward / back
    J K L            J / L   strafe left / right
     N M             N / M   yaw left / right
                     R       reset envs
                     Esc     quit (or close the window)

GLFW (MuJoCo) uses ASCII codes for letter keys, so `ord(letter)` is the
keycode — letting both engines share this table verbatim.
"""

import torch

# action name → key letter (uppercase; ord() gives the GLFW keycode)
BINDINGS = {
    "forward": "I",
    "back": "K",
    "strafe_left": "J",
    "strafe_right": "L",
    "yaw_left": "N",
    "yaw_right": "M",
    "reset": "R",
}

# key letter → action name (what the interfaces poll/dispatch on)
KEY_TO_ACTION = {key: action for action, key in BINDINGS.items()}

HELP_LINES = (
    "  I/K           forward / back",
    "  J/L           strafe left / strafe right",
    "  N/M           yaw left / yaw right",
    "  R             reset envs",
    "  Esc / window  quit",
    "  commands step in 1/5 increments of max",
)


class TeleopCommands:
    """Velocity-command limits, increments, and the effect of each action.

    Engine-agnostic: it only touches ``env.commands`` and the reset path.
    """

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
        env.commands[:, 0] = 1.0  # seed forward velocity so motion is visible
        if hasattr(env.cfg, "commands"):
            env.cfg.commands.resampling_time = env.max_episode_length_s + 1

    def apply(self, action: str) -> None:
        """Apply one discrete press of ``action`` (see BINDINGS)."""
        c = self.env.commands
        if action == "forward":
            c[:, 0] = torch.clamp(c[:, 0] + self.increment_x, max=self.max_vel_forward)
        elif action == "back":
            c[:, 0] = torch.clamp(c[:, 0] - self.increment_x, min=self.max_vel_backward)
        elif action == "strafe_left":
            c[:, 1] = torch.clamp(c[:, 1] + self.increment_y, max=self.max_vel_sideways)
        elif action == "strafe_right":
            c[:, 1] = torch.clamp(
                c[:, 1] - self.increment_y, min=-self.max_vel_sideways
            )
        elif action == "yaw_left":
            c[:, 2] = torch.clamp(c[:, 2] + self.increment_yaw, max=self.max_vel_yaw)
        elif action == "yaw_right":
            c[:, 2] = torch.clamp(c[:, 2] - self.increment_yaw, min=-self.max_vel_yaw)
        elif action == "reset":
            self.env.timed_out[:] = True
            self.env.reset()
        else:
            raise ValueError(f"unknown teleop action {action!r}")

    def print_help(self, viewer_name: str) -> None:
        print("______________________________________________________________")
        print(f"Keyboard teleop ({viewer_name})")
        for line in HELP_LINES:
            print(line)
        print("______________________________________________________________")
