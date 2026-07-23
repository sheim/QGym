# thirdparty/vlearn — vsim engine drop zone

The vsim backend's engine (`vlearn`) is closed-source and licensed; its
binaries cannot be committed (100 KB file cap + licensing). This directory
holds the machine-local pieces, all gitignored:

- `vlearn-<version>-cp311-cp311-linux_x86_64.whl` — the engine wheel
  (from the vendor / the vlearn SDK repo). `[tool.uv.sources]` in
  pyproject.toml points here; install with `uv sync --extra vsim`.
- `License.key` — your node-locked license key (one line); found via
  `VL_LICENSE_KEY_PATH` (set in `.env.vsim`).
- `TurboActivate.dat` — vendor-shipped product-definition file (copied from
  the vlearn SDK repo; version-independent); found via
  `VL_TURBO_ACTIVATE_PATH`.
- `cache/` — the engine's cache tree (`cache/tmp/` + marker file
  `cache/donotremove.txt`); required by `vsim::Path::findCachePath`.
  `VSimBackend.setup()` self-heals it if missing.
- `shaders/` + `assets/{VsimTile,Skybox}.png` — renderer files, needed only
  for the viewer (copied from the vlearn SDK repo root). Headless runs work
  without them.

On a new machine: drop in the wheel and `License.key`,
`uv sync --extra vsim`, then run any vsim command once with internet to
node-lock the activation (all lookup paths are wired in `.env.vsim`).
