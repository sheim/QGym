# thirdparty/vlearn — vsim engine drop zone

The vsim backend's engine (`vlearn`) is closed-source and licensed; its
binaries cannot be committed (100 KB file cap + licensing). The `vlearn/`
subdirectory holds the machine-local pieces, all gitignored:

- `vlearn-0.3.14+cu130-cp311-cp311-linux_x86_64.whl` — the selected engine wheel
  (from the vendor / the vlearn SDK repo). `[tool.uv.sources]` in
  pyproject.toml points here; install with `uv sync --frozen --extra vsim`.
  The CUDA 13.0 variant matches the locked Linux PyTorch build.
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
- `assets/go2/assets/*.obj` + `assets/licenses/go2-LICENSE.txt` — Go2 visual
  meshes from the SDK repository. Go2/Go2Trot VSim conversion requires these
  instead of the unavailable `package://go2_description/dae/*.dae` resources.
  Only visuals are substituted; our URDF remains the physical model.

## Install and activate

From the repository root:

```bash
uv sync --frozen --extra vsim
```

Activate after the first install and whenever `License.key` is replaced or
renewed. Internet access is required during activation:

```bash
# Run from the repository root.
(
  cd thirdparty/vlearn
  LD_LIBRARY_PATH=../../.venv/lib/python3.11/site-packages/vlearn/lib \
  VL_WORKING_DIRECTORY="$PWD" \
  VL_TURBO_ACTIVATE_PATH="$PWD/TurboActivate.dat" \
  VL_LICENSE_KEY_PATH="$PWD/License.key" \
  ../../.venv/bin/python -c \
    'import vlearn as v; v.create_gym(with_render=False, with_window=False); v.delete_gym(); print("vsim activation probe succeeded")'
)
```

The final line should be `vsim activation probe succeeded`. Verify the setup:

```bash
bash scripts/run_vsim_tests.sh
```

After the tests pass, run normal commands from the repository root:

```bash
uv run --frozen --extra vsim --env-file .env.vsim scripts/train.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
```

`--frozen` installs the selected locked dependencies without requiring the
unrelated optional Unitree checkout. The Go2 hardware SDK has its own gitignored
directory, `thirdparty/unitree_sdk2_python/`, populated by
`uv run --frozen python scripts/fetch_unitree_sdk.py`. See
[`README_DEPLOY.md`](../README_DEPLOY.md) for its native DDS prerequisite and
editable installation.
