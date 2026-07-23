"""Keyboard teleop for the MuJoCo passive viewer.

Wires into mujoco.viewer.launch_passive's key_callback, set on the backend
before the first render() call.  MuJoCo delivers discrete key EVENTS, so a
press maps straight to one command increment (no edge detection needed —
contrast VsimKeyboardInterface, which polls).

Bindings and command logic are shared with the vsim interface — see
teleop_bindings.py for the key layout and why it is IJKL/NM.
"""

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands

# GLFW uses ASCII codes for letter keys, so the shared letter table gives
# the keycodes directly.  Hardcoded to avoid a glfw dependency.
KEYCODE_TO_ACTION = {ord(key): action for key, action in KEY_TO_ACTION.items()}


class MujocoKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.commands = TeleopCommands(env)

        env._backend._viewer_key_callback = self._on_key
        self.commands.print_help("MuJoCo viewer")

    def _on_key(self, keycode: int) -> None:
        action = KEYCODE_TO_ACTION.get(keycode)
        if action is not None:
            self.commands.apply(action)
