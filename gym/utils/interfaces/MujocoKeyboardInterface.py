"""Keyboard teleop for the MuJoCo passive viewer.

Wires into mujoco.viewer.launch_passive's key_callback, set on the backend
before the first render() call.  MuJoCo delivers discrete key EVENTS, so a
press maps straight to one command increment (no edge detection needed —
contrast VsimKeyboardInterface, which polls).

Bindings and command logic are shared with the vsim interface — see
teleop_bindings.py for the key layout and why it is IJKL/NM.
"""

from queue import SimpleQueue

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands

# GLFW uses ASCII codes for letters and separate codes for special keys.
# Hardcoded to avoid a glfw dependency.
SPECIAL_KEYCODES = {"UP": 265, "DOWN": 264}
KEYCODE_TO_ACTION = {
    ord(key) if len(key) == 1 else SPECIAL_KEYCODES[key]: action
    for key, action in KEY_TO_ACTION.items()
}


class MujocoKeyboardInterface:
    """MuJoCo key-event adapter.

    The passive viewer invokes ``key_callback`` on its own UI thread, while the
    task loop owns all ``env.commands`` and reset state. ``deferred=True`` lets
    playback enqueue key actions on that thread and drain them on the task
    thread; the default applies immediately for existing callers.
    """

    def __init__(self, env, commands=None, *, deferred=False):
        self.env = env
        self.commands = TeleopCommands(env) if commands is None else commands
        self.deferred = deferred
        self._pending = SimpleQueue() if deferred else None

        env._backend._viewer_key_callback = self._on_key
        self.commands.print_help("MuJoCo viewer")

    def _on_key(self, keycode: int) -> None:
        action = KEYCODE_TO_ACTION.get(keycode)
        if action is None:
            return
        if self.deferred:
            self._pending.put(action)
            return
        self.commands.apply(action)

    def drain(self) -> tuple[str, ...]:
        """Return queued actions in event order (task-thread only)."""
        if self._pending is None:
            return ()
        actions = []
        while not self._pending.empty():
            actions.append(self._pending.get())
        return tuple(actions)
