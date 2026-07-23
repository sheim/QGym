"""Keyboard teleop for the vsim (vlearn) viewer.

vlearn exposes POLLING input (`GymRender.is_key_down`) rather than the event
callbacks MuJoCo and IsaacGym provide, so this interface remembers the
previous key state and acts on rising edges — giving the same discrete
"1/5 of max per press" feel as the MuJoCo interface instead of ramping the
command every frame a key is held.

Bindings and command logic are shared with the MuJoCo interface — see
teleop_bindings.py (which also explains why the keys are IJKL/NM and not
WASD: vsim flies its camera with WASD).

Polling is driven by VSimBackend.render() via a per-frame render hook, so
the play loop needs no per-step update() call.
"""

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands


class VsimKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.commands = TeleopCommands(env)
        # Specials come from the engine's UserKey enum; the backend owns the
        # vlearn handle, so this module never imports the engine (keeping it
        # testable without a license).
        self._escape_key = env._backend.escape_key
        self._was_down = {key: False for key in KEY_TO_ACTION}

        env._backend.add_render_hook(self.poll)
        self.commands.print_help("vsim viewer")

    def poll(self, render) -> None:
        """Called once per rendered frame by VSimBackend.render()."""
        for key, action in KEY_TO_ACTION.items():
            is_down = render.is_key_down(key)
            if is_down and not self._was_down[key]:
                self.commands.apply(action)
            self._was_down[key] = is_down

        if render.is_key_down(self._escape_key) or self.env._backend.window_closed:
            self.env.exit = True
