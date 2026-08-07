# Q2 — MuJoCo / vsim RL Training

RL training framework for legged robots. Physics backends:

- **MuJoCo CPU** — all platforms, including macOS
- **MuJoCo Warp** — NVIDIA GPU on Linux
- **vsim** — licensed NVIDIA GPU backend (`--backend vsim`)

## Quick Start

### Fresh setup

The repository uses [uv](https://docs.astral.sh/uv/) exclusively for Python
and dependency management. `pyproject.toml` and `uv.lock` are the only package
and environment sources of truth.

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

Bare pytest targets the supported unit and MuJoCo CPU gate configured in
`pyproject.toml`; Warp and VSim groups are selected explicitly.

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
uv run --frozen scripts/train.py --task mini_cheetah --device cuda:0 \
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

Install the VSim extra from the repository root:

```bash
uv sync --locked --extra vsim
```

Activate the license after the first install. Repeat this step whenever you
replace or renew `License.key`. The activation requires internet access:

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

The final line should be `vsim activation probe succeeded`. Next, verify the
license and backend integration:

```bash
bash scripts/run_vsim_tests.sh
```

After the tests pass, run VSim commands from the repository root with
`.env.vsim`:

```bash
uv run --env-file .env.vsim scripts/train.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
```

Never commit the wheel, license files, or activation data.

### Train

```bash
# Pendulum (fixed-base, 1 DOF) — with GUI viewer
uv run --frozen scripts/train.py --task pendulum --device cpu --num_envs 256

# Mini Cheetah (floating-base, 12 DOFs) — with GUI viewer
uv run --frozen scripts/train.py --task mini_cheetah --device cpu --num_envs 64

# Headless (no viewer window)
uv run --frozen scripts/train.py --task pendulum --device cpu --num_envs 256 --headless

# GPU training (Linux only, requires mujoco-warp)
uv run --frozen scripts/train.py --task mini_cheetah --device cuda:0 --num_envs 4096 --headless
```

### Resume or play with the saved configuration

Each training run stores its Python configuration sources below
`logs/<experiment>/<run>/files/`. Pass `--original_cfg` to reconstruct the
environment and runner configs from that snapshot instead of using the current
task configs:

```bash
# Continue a run with the configuration that created it.
uv run --frozen scripts/train.py --task mini_cheetah_ref --resume \
    --load_run Jul28_21-18-57_ --checkpoint 500 --original_cfg --headless

# Play a checkpoint with its saved configuration.
uv run --frozen scripts/play.py --task mini_cheetah_ref \
    --load_run Jul28_21-18-57_ --checkpoint 500 --original_cfg
```

The experiment defaults to the task's current `experiment_name`; use
`--experiment_name` when the run is under a different log root. Omitting
`--load_run` selects the latest run in that experiment. Explicit CLI settings,
such as `--num_envs`, `--seed`, or `--max_iterations`, are applied after the
saved configs are loaded.

Only the saved config modules and their saved config dependencies are loaded.
The current task implementation, learning stack, and selected physics backend
remain active. This makes the option suitable for recent compatible runs while
allowing old snapshots to fail clearly when their config contract is no longer
supported. Saved configs are Python code, so use `--original_cfg` only with
trusted run directories. Without `--resume`, `train.py --original_cfg` starts a
new run from the saved configuration rather than loading model state. New
training runs carry the original config snapshot forward under
`files/original_cfg/`, so another resume does not silently substitute the
then-current task config.

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
uv run scripts/train.py [OPTIONS]

  --task TEXT          Task name (required). See "Available Tasks" below.
  --device TEXT        "cpu" or "cuda:0" (default: cpu)
  --num_envs INT      Number of parallel environments (default: from task config)
  --max_iterations INT Training iterations (default: from task config)
  --seed INT          Random seed
  --resume            Resume model and optimizer state
  --load_run TEXT     Run directory below the selected experiment
  --checkpoint INT    Checkpoint iteration (default: latest)
  --original_cfg      Load environment and runner configs from the selected run
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
    ├── MuJocoWarpBackend   ← GPU (Linux + CUDA)
    ├── MuJocoCPUBackend    ← CPU (all platforms)
    └── VSimBackend         ← optional licensed GPU backend

BaseTask
    ├── FixedRobot          ← pendulum, cartpole
    └── LeggedRobot         ← mini_cheetah, humanoid
         └── ConcreteTask   ← rewards, observations
```

Backend is selected automatically based on `--device`:

- `cpu` → MuJocoCPUBackend (macOS always uses this)
- `cuda:0` → MuJocoWarpBackend

## Canonical Robot Layout

Physics engines may compile a URDF's joints, links, motors, and sensors in
different orders. `RobotLayout` gives tasks one stable, engine-independent
order instead. Every supported backend validates its native model against the
layout during setup and builds the permutations needed at the backend
boundary. Tasks consequently always see:

- `dof_names`, DOF state tensors, reset values, and input torques in canonical
  full-DOF order;
- `body_names`, rigid-body states, and contact forces in canonical robot-body
  order, without a simulator world body; and
- actions routed through the canonical `actuated_dof_names` subset.

The layout also assigns exact names to semantic groups such as `feet`,
`front_left_leg`, or `wheel_joints`. Tasks resolve these groups to tensor
indices during initialization instead of relying on backend order, positional
slices, or naming substrings. See
[`gym/envs/base/robot_layout.py`](gym/envs/base/robot_layout.py) for the
implementation and
[`gym/envs/mini_cheetah/mini_cheetah_config.py`](gym/envs/mini_cheetah/mini_cheetah_config.py)
for a complete example.

### DOF and body groups

`dof_groups` and `body_groups` are dictionaries from a task-level concept to
an ordered list of exact canonical names:

- A DOF group collects joints that a task operates on together, such as one
  leg, all hip-abduction joints, an arm, or a set of wheels.
- A body group collects links used together for state, contact, reward, or
  termination logic, such as feet, hands, or other end effectors.

Groups are labels over the canonical lists, not another engine-specific
ordering. They do not need to cover every DOF or body, and groups may overlap.
The member order is preserved and is therefore significant when, for example,
matching a three-column reference trajectory to the hip, knee, and ankle of a
leg.

`RobotLayout` converts a group to indices in the canonical full-DOF or body
tensor:

```python
left_leg = layout.dof_group_indices("left_leg")  # (0, 1, 2)
feet = layout.body_group_indices("feet")         # (3, 6)

left_leg_position = dof_pos[:, left_leg]
foot_forces = contact_forces[:, feet, :]
```

Use `dof_group_indices()` for full-DOF tensors. If a task is indexing an
action or another actuator-only tensor, translate the group's names through
`actuated_dof_names` instead; the actuated subset may omit or reorder entries
from the full DOF list.

`LeggedRobot` specifically resolves `body_groups["feet"]` during
initialization and stores it as `feet_indices`. Foot contact rewards, foot
state, and evaluation metrics use those indices. Other group names have no
automatic behavior: a concrete task chooses how to consume them. Likewise,
body groups do not configure contact penalties or termination by themselves;
`asset.penalize_contacts_on` and `asset.terminate_after_contacts_on` remain
separate substring-matching settings.

### Defining a layout for a new robot

Place the robot URDF below `resources/robots/<robot>/`, using unique joint and
link names. Use `LeggedRobotCfg` for a floating base with ground contacts, or
`FixedRobotCfg` for a world-attached mechanism. Then define the policy-facing
order and semantics in the task's environment config:

```python
class MyRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actuators = 6

    class asset(LeggedRobotCfg.asset):
        file = "{GYM_ROOT_DIR}/resources/robots/my_robot/my_robot.urdf"
        penalize_contacts_on = ["lower_leg"]
        terminate_after_contacts_on = ["base"]

        class robot_layout:
            version = "my_robot_v1"
            dof_names = [
                "left_hip",
                "left_knee",
                "left_ankle",
                "right_hip",
                "right_knee",
                "right_ankle",
            ]
            actuated_dof_names = dof_names
            body_names = [
                "base",
                "left_upper_leg",
                "left_lower_leg",
                "left_foot",
                "right_upper_leg",
                "right_lower_leg",
                "right_foot",
            ]
            dof_groups = {
                "left_leg": dof_names[0:3],
                "right_leg": dof_names[3:6],
            }
            body_groups = {
                "feet": ["left_foot", "right_foot"],
            }
```

Follow these rules when defining it:

1. `dof_names` must contain every non-fixed URDF joint exactly once, and
   `body_names` must contain every URDF link exactly once. Their listed order
   becomes the public tensor order and may differ from the URDF or engine
   order.
2. `actuated_dof_names` is an ordered subset of `dof_names`; its length must
   equal `env.num_actuators`. Unactuated DOFs remain present in state and
   full-DOF torque tensors. When an explicit `robot_layout` exists, declare
   the actuated subset here rather than in `control.actuated_joint_names`.
3. Define task semantics as groups of exact names. Groups may overlap, but
   every member must exist in the corresponding canonical list. A
   `LeggedRobot` needs a `feet` body group; a `FixedRobot` does not.
4. Treat the canonical order as part of the policy/checkpoint interface. Keep
   it stable across asset changes and update `version` when deliberately
   changing its order or meaning.

For a simple robot without an explicit layout, Q2 falls back to URDF
declaration order, uses `control.actuated_joint_names` if supplied, and can
derive `feet` by matching `asset.foot_name`. Prefer an explicit layout once
ordering stability or named semantics matter.

No backend-specific layout code should be added for a new robot. After adding
the task class and configs, register all four declarations in
[`gym/envs/__init__.py`](gym/envs/__init__.py): the task class, environment
config, runner config, and public task name. Add focused tests that:

- construct `RobotLayout.from_cfg()` and assert the canonical names and group
  indices;
- construct the task on MuJoCo CPU and assert the backend exposes that same
  canonical layout; and
- exercise any named actuator, reference-trajectory, end-effector, or contact
  routing used by the task.

Then run the normal gate and a short CPU smoke train before checking optional
GPU backends:

```bash
uv run --frozen python -m pytest -q
uv run --frozen scripts/train.py --task my_robot --backend mujoco \
    --device cpu --num_envs 8 --max_iterations 2 --headless --disable_wandb
```

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
.venv/bin/mjpython scripts/train.py --task mini_cheetah --device cpu --num_envs 64
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
│   │   │   ├── robot_layout.py          ← canonical robot ordering and groups
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
│   └── train.py                         ← main training script
├── tests/unit_tests/                    ← supported local test gate
└── pyproject.toml
```
