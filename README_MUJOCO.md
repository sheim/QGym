# Q2 — MuJoCo / vsim RL Training

RL training framework for legged robots. Physics backends:

- **MuJoCo CPU** — all platforms, including macOS
- **MuJoCo Warp** — NVIDIA GPU on Linux
- **vsim** — licensed NVIDIA GPU backend (`--backend vsim`)

## Quick Start

### Fresh setup

The repository uses [uv](https://docs.astral.sh/uv/) exclusively for Python
and dependency management. Do not use `pip install`, `requirements.txt`, or
`setup.py`.

Install uv on Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
uv --version
```

If uv is already installed, update it first:

```bash
uv self update
```

`uv self update` works for uv's standalone installer. If uv was installed by
Homebrew or another package manager, update it through that package manager
instead.

Clone the repository and create the Python 3.11 environment:

```bash
git clone git@github.com:LampLighterLab/QGym.git
cd QGym
uv python install 3.11
uv venv --python 3.11
uv sync --frozen
```

Python 3.11 is used because the optional vsim wheel is built for CPython 3.11.
MuJoCo-only development supports Python 3.11–3.13, but 3.11 is the common,
tested setup.

`--frozen` is intentional for a MuJoCo-only clone. The universal lockfile also
records the optional, gitignored vsim wheel; asking uv to re-resolve or validate
that local source fails until the licensed wheel has been supplied. Use
`--frozen` with `uv sync` and `uv run` when vsim is not installed.

Verify the installation:

```bash
uv run --frozen python -c "import mujoco, torch; print(mujoco.__version__, torch.__version__)"
uv run --frozen python -m pytest tests/unit_tests/ -q
```

Do not run `pytest tests/`: the legacy integration tests require IsaacGym and
its separate Python 3.8 environment.

### Linux GPU setup

First verify that the NVIDIA driver is visible:

```bash
nvidia-smi
```

Then install the declared GPU extra and verify CUDA:

```bash
uv sync --frozen --extra gpu
uv run --frozen python -c "import torch, mujoco_warp; print(torch.cuda.is_available())"
```

The final command must print `True`. MuJoCo Warp cannot work around a missing
or inaccessible NVIDIA driver. GPU training is headless:

```bash
uv run --frozen scripts/train_mujoco.py --task mini_cheetah --device cuda:0 \
    --num_envs 4096 --headless
```

### Optional vsim setup

vsim is a closed-source, node-locked backend. It additionally requires Linux,
an NVIDIA GPU, the system `libczmq4` package, and vendor files that cannot be
committed to this repository.

Place these files under `thirdparty/vlearn/` as described in
[`thirdparty/vlearn/README.md`](thirdparty/vlearn/README.md):

- `vlearn-0.3.12-cp311-cp311-linux_x86_64.whl`
- `License.key`
- `TurboActivate.dat`

Then run:

```bash
uv sync --locked --extra vsim

# One-time node activation (internet required):
cd thirdparty/vlearn
LD_LIBRARY_PATH=../../.venv/lib/python3.11/site-packages/vlearn/lib \
VL_WORKING_DIRECTORY="$PWD" \
VL_TURBO_ACTIVATE_PATH="$PWD/TurboActivate.dat" \
VL_LICENSE_KEY_PATH="$PWD/License.key" \
../../.venv/bin/python -c \
  'import vlearn as v; v.create_gym(with_render=False, with_window=False); v.delete_gym(); print("vsim activated")'
cd ../..

uv run --env-file .env.vsim scripts/train_mujoco.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
bash scripts/run_vsim_tests.sh
```

### Train

```bash
# Pendulum (fixed-base, 1 DOF) — with GUI viewer
uv run --frozen scripts/train_mujoco.py --task pendulum --device cpu --num_envs 256

# Mini Cheetah (floating-base, 12 DOFs) — with GUI viewer
uv run --frozen scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64

# Headless (no viewer window)
uv run --frozen scripts/train_mujoco.py --task pendulum --device cpu --num_envs 256 --headless

# GPU training (Linux only, requires mujoco-warp)
uv run --frozen scripts/train_mujoco.py --task mini_cheetah --device cuda:0 --num_envs 4096 --headless
```

### Test

```bash
uv run --frozen python -m pytest tests/unit_tests/ -v
```

## Notes

**NaN rewards at startup:** reward logging averages over completed episodes.
Until the first episode finishes, all reward values show `nan` — this is
expected and not an error.

## CLI Reference

```
uv run scripts/train_mujoco.py [OPTIONS]

  --task TEXT          Task name (required). See "Available Tasks" below.
  --device TEXT        "cpu" or "cuda:0" (default: cpu)
  --num_envs INT      Number of parallel environments (default: from task config)
  --max_iterations INT Training iterations (default: from task config)
  --seed INT          Random seed
  --headless          Disable GUI viewer
  --disable_wandb     Disable Weights & Biases logging (default: on)
```

## Available Tasks

| Task | Robot | Type | DOFs |
|------|-------|------|------|
| `pendulum` | Pendulum | Fixed-base | 1 |
| `cartpole` | Cart-pole | Fixed-base | 2 |
| `mini_cheetah` | MIT Mini Cheetah | Floating-base | 12 |
| `mini_cheetah_ref` | Mini Cheetah (ref tracking) | Floating-base | 12 |
| `mini_cheetah_osc` | Mini Cheetah (oscillator) | Floating-base | 12 |
| `humanoid` | MIT Humanoid | Floating-base | 30 |
| `humanoid_running` | Humanoid (running) | Floating-base | 30 |

## Architecture

```
SimBackend (ABC)
    ├── IsaacGymBackend     ← legacy, will be removed
    ├── MuJocoWarpBackend   ← GPU (Linux + CUDA)
    └── MuJocoCPUBackend    ← CPU (all platforms)

BaseTask
    ├── FixedRobot          ← pendulum, cartpole
    └── LeggedRobot         ← mini_cheetah, humanoid
         └── ConcreteTask   ← rewards, observations
```

Backend is selected automatically based on `--device`:
- `cpu` → MuJocoCPUBackend (macOS always uses this)
- `cuda:0` → MuJocoWarpBackend

## Platform Notes

### macOS (Apple Silicon / Intel)

Works out of the box with `--device cpu`. The CPU backend uses plain
`mujoco.mj_step` with one MjData per environment (Python loop).
Performance scales linearly with `--num_envs`.

GPU training (`--device cuda:0`) is not available on macOS.

**GUI viewer on macOS:** MuJoCo's passive viewer requires `mjpython` (bundled
with the `mujoco` pip package) instead of the standard Python interpreter.
`mjpython` needs to dlopen `libpython3.13.dylib`, which uv's bundled Python
does not ship. The fix is to create the venv using Homebrew's Python instead:

```bash
brew install python@3.13   # if not already installed
uv venv --python /opt/homebrew/opt/python@3.13/bin/python3.13
uv sync --frozen
.venv/bin/mjpython scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64
```

Alternatively, use `--headless` to skip the viewer entirely (works with uv's default Python).

### Linux (CPU)

Same as macOS. Use `--device cpu`.

### Linux (GPU)

Requires the NVIDIA driver and the `gpu` extra (`uv sync --frozen --extra gpu`).
Use `--device cuda:0` for GPU-accelerated vectorized physics.
Typically 10-15x faster than CPU for large `--num_envs` (4096+).

The GUI viewer is not available with the Warp backend. Use `--headless`.

## Project Structure

```
Q2/
├── gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── sim_backend.py           ← abstract backend interface
│   │   │   ├── mujoco_backend_base.py   ← shared MuJoCo setup logic
│   │   │   ├── mujoco_cpu_backend.py    ← CPU backend (mj_step loop)
│   │   │   ├── mujoco_warp_backend.py   ← GPU backend (mujoco_warp)
│   │   │   ├── base_task.py             ← base task class
│   │   │   ├── fixed_robot.py           ← fixed-base robot task
│   │   │   └── legged_robot.py          ← floating-base robot task
│   │   ├── mini_cheetah/                ← mini cheetah tasks + configs
│   │   └── mit_humanoid/                ← humanoid tasks + configs
│   └── utils/
│       ├── task_registry.py             ← task registration + backend selection
│       └── torch_quat.py               ← pure-torch quaternion math
├── learning/                            ← RL algorithms, runners, storage
├── resources/robots/                    ← URDF files
├── scripts/
│   └── train_mujoco.py                  ← main training script
├── tests/unit_tests/                    ← 147 unit tests
└── pyproject.toml
```
