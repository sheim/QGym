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

On a new machine, drop in the wheel, `License.key`, and `TurboActivate.dat`,
then install and activate:

```bash
# From the repository root:
uv sync --extra vsim

# The native SDK's first activation must start in the license directory.
cd thirdparty/vlearn
LD_LIBRARY_PATH=../../.venv/lib/python3.11/site-packages/vlearn/lib \
VL_WORKING_DIRECTORY="$PWD" \
VL_TURBO_ACTIVATE_PATH="$PWD/TurboActivate.dat" \
VL_LICENSE_KEY_PATH="$PWD/License.key" \
../../.venv/bin/python -c \
  'import vlearn as v; v.create_gym(with_render=False, with_window=False); v.delete_gym(); print("vsim activated")'
cd ../..
```

The first activation needs internet access and writes a node-locked activation
record outside the repository. After it succeeds, normal commands should run
from the repository root with `uv run --env-file .env.vsim ...`.
