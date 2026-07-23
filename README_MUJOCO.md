# Q2 — MuJoCo / vsim RL Training

RL training framework for legged robots. Physics backends:
- **MuJoCo CPU** (all platforms incl. macOS) and **mujoco-warp** GPU (Linux+CUDA)
- **vsim** (`--backend vsim`): closed-source licensed GPU engine, ~10× faster
  than warp at 4096 envs. Self-contained under `thirdparty/vlearn/` (wheel +
  license drop zone — see its README); needs system `libczmq4` and process
  env from `.env.vsim`:

```bash
uv sync --extra vsim
uv run --env-file .env.vsim scripts/train_mujoco.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
bash scripts/run_vsim_tests.sh        # vsim test suite (local-only)
```

## Quick Start

### Setup

Requires Python 3.11 (pinned — the vsim wheels are cp311; MuJoCo-only work
also runs on 3.12/3.13) and [uv](https://docs.astral.sh/uv/).

```bash
cd Q2
uv sync          # installs all dependencies, creates .venv
```

On Linux with an NVIDIA GPU, also install the GPU backend:

```bash
uv pip install mujoco-warp
```

### Train

```bash
# Pendulum (fixed-base, 1 DOF) — with GUI viewer
uv run scripts/train_mujoco.py --task pendulum --device cpu --num_envs 256

# Mini Cheetah (floating-base, 12 DOFs) — with GUI viewer
uv run scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64

# Headless (no viewer window)
uv run scripts/train_mujoco.py --task pendulum --device cpu --num_envs 256 --headless

# GPU training (Linux only, requires mujoco-warp)
uv run scripts/train_mujoco.py --task mini_cheetah --device cuda:0 --num_envs 4096 --headless
```

### Test

```bash
uv run python -m pytest tests/unit_tests/ -v
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
uv sync
.venv/bin/mjpython scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64
```

Alternatively, use `--headless` to skip the viewer entirely (works with uv's default Python).

### Linux (CPU)

Same as macOS. Use `--device cpu`.

### Linux (GPU)

Requires `mujoco-warp` (`uv pip install mujoco-warp`).
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
