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
git clone https://github.com/LampLighterLab/BigRedGym.git
cd BigRedGym
uv python install 3.11
uv venv --python 3.11
uv sync --frozen
```

The package currently requires Python 3.11, matching the optional vsim wheel
and the deployment environment.

`--frozen` is intentional for a MuJoCo-only clone. The universal lockfile also
records the optional, gitignored VSim wheel and Unitree SDK checkout; asking uv
to re-resolve or validate local sources requires those files. Use `--frozen`
with `uv sync` and `uv run` when optional sources are not installed.

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

### Go2 CPU visualization

Go2 and Go2Trot's MuJoCo CPU viewer renders collision shapes by default
(`asset.mujoco_visual_model = None`). No VSim checkout or external mesh assets
are needed.

Detailed meshes remain optional: set `asset.mujoco_visual_model` to
`"{GYM_ROOT_DIR}/thirdparty/vlearn/assets/go2/go2.xml"` in the Go2 config if
you have that XML and its sibling `assets/*.obj` files. Only meshes and
material colors are imported; physics still comes from our URDF. Collision
shapes are hidden in viewer geometry group 3 when detailed meshes are enabled.
Headless MuJoCo CPU and Warp do not load these optional visuals.

When using an older saved config with `--original_cfg`, set its
`asset.mujoco_visual_model` to `None` to render collision shapes without the
external assets.

### Optional vsim setup

vsim is a closed-source, node-locked backend. Its wheel requires Linux x86_64
and Python 3.11, plus an NVIDIA GPU, the system `libczmq4` package, and vendor
files that cannot be committed to this repository.

Place these files under `thirdparty/vlearn/` as described in
[`thirdparty/README.md`](../thirdparty/README.md):

- `vlearn-0.3.15+cu130-cp311-cp311-linux_x86_64.whl`
- `License.key`
- `TurboActivate.dat`

The selected wheel targets CUDA 13.0, matching the locked Linux PyTorch build.
Keep the vendor wheel local; its path and hash are recorded in `pyproject.toml`
and `uv.lock`.

Install the VSim extra from the repository root:

```bash
uv sync --frozen --extra vsim
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

The final line should be `vsim activation probe succeeded`. This checks engine
activation only. VSim backend code, its optional extra, and marked tests remain;
the local test launcher and support workflow are deferred. See
[limitations and validation scope](#limitations-and-validation-scope).

Run VSim commands from the repository root with `.env.vsim`:

```bash
uv run --frozen --extra vsim --env-file .env.vsim scripts/train.py --task mini_cheetah \
    --backend vsim --device cuda:0 --num_envs 4096 --headless
```

Never commit the wheel, license files, or activation data.

Go2 and Go2Trot's VSim backend uses the SDK repository's visual meshes from
`thirdparty/vlearn/assets/go2/assets/` (the `*.obj` files). Keep that directory
from the SDK checkout, along with `assets/licenses/go2-LICENSE.txt`. If your
SDK is elsewhere, set `asset.vsim_visual_mesh_dir` in the Go2 config to its
`assets/go2/assets` directory. The wheel alone does not include these meshes.
VSim substitutes these visuals before converting our URDF; joints, collisions,
and inertias still come from our robot, not the vendor's `go2_og.vsim`.

### Optional Unitree Go2 deployment setup

Training and simulation do not require the Unitree SDK. On Linux, hardware
deployment uses the `unitree_sdk` extra and a pinned editable checkout of the
official SDK. The checkout preserves native CRC libraries missing from upstream
wheels; its Python dependencies are locked normally.

Fetch the SDK, then build native Cyclone DDS 0.10.2 and export
`CYCLONEDDS_HOME` using [`README_DEPLOY.md`](../README_DEPLOY.md):

```bash
uv run --frozen python scripts/fetch_unitree_sdk.py
# Build Cyclone DDS and export CYCLONEDDS_HOME as described in README_DEPLOY.md.
uv sync --frozen --extra unitree_sdk
uv run --frozen --extra unitree_sdk python -m pytest -q -m unitree
```

The tests do not connect to a robot. Use `--extra unitree_sdk` on deployment
commands as well; a default sync removes optional packages. `--frozen` also
allows deployment-only machines to install without the local VSim wheel.

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

Training follows the task's domain-randomization config exactly. There is no
training CLI override: physical DR changes native backend topology and should
be an explicit, reviewable part of the environment definition. Historical
campaign drivers have been retired.

The config makes each axis's sampling cadence explicit:

```python
class domain_randomization:
    class startup:
        # One fixed physical identity per parallel environment.
        contact_friction_range = [0.5, 1.0]
        link_mass_scale_range = [0.9, 1.1]

    class episode:
        # Bound task tensors, resampled whenever an environment resets.
        scale_ranges = {
            "p_gains": [0.9, 1.1],
            "d_gains": [0.9, 1.1],
        }
```

Startup axes still span the configured distribution across all parallel
environments; they simply stay fixed for the lifetime of the task. This avoids
reapplying physical model properties during the frequent asynchronous resets.
Set a startup range to `None`, or omit an episodic target from `scale_ranges`,
to disable it. The task explicitly binds the allowed episodic tensors; the DR
object then owns their nominal copies and current scales. This keeps nominal
DR state out of the environment and prevents config strings from reaching
arbitrary task attributes.

The current startup API still accepts environment indices. Its proposed
full-batch simplification is unimplemented and separate from episodic mask-based
resets; see [limitations and validation scope](#limitations-and-validation-scope).

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

### Intervened height-Q playback

`scripts/play_height_q.py` plays one frozen PPO actor with the original
unprivileged height-Q artifact on MuJoCo CPU. It takes eight stochastic policy
draws per control step and executes either the highest-Q draw (`--mode maxq`)
or the first draw with `Q >= c`, falling back to max-Q (`--mode rejection
--threshold c`). The policy mean is not a candidate. The checkpoint hash, task
config fingerprint, actor observation/action contract, noise, scales, and
normalization flags must match the artifact before the simulator is built.

```bash
# Default max-Q intervention, one CPU environment.
uv run --frozen python scripts/play_height_q.py \
  --task go2trot \
  --artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt

# Threshold rejection with a finite threshold.
uv run --frozen python scripts/play_height_q.py \
  --task go2trot --artifact <artifact> --mode rejection --threshold -1.776

# Finite CPU headless wiring smoke.
uv run --frozen python scripts/play_height_q.py \
  --task go2trot --artifact <artifact> --mode maxq \
  --device cpu --headless --steps 5 --seed 0
```

Playback starts at zero commands. `I/K` are forward/back, `J/L` strafe,
`N/M` yaw, and `R` performs the normal task reset. Commands are held until the
next key event; task command resampling is disabled for this process only.
`R` and every enabled auto-reset zero commands before the next policy action.
`--auto_reset off` is the default and leaves a fallen or timed-out robot in the
simulator; `termination`, `timeout`, and `both` optionally use the normal task
reset path. The task timeout, push disturbances, observation noise, and camera
come from the selected task config. `--no-push_robots` is a playback-only
escape hatch. `--device cuda`, `--backend vsim`, and `--num_envs` other than one
are rejected; Warp/VSim/hardware playback is not implemented.

The headless command proves wiring and artifact identity only. It does not
establish Q accuracy, policy improvement, or manual viewer behavior. Validate
the viewer, keys, zero command hold, resets, and terminal interaction by hand.

The MuJoCo CPU viewer paces simulation to real time by default: one simulated
second takes at least one wall-clock second, including policy and rendering
work. It accounts for all physics substeps between frames and resumes without
fast catch-up after a slow frame. No playback flag is needed. Headless runs
remain unrestricted; backend callers can opt out with
`render(sync_frame_time=False)`. This changes wall-clock pacing, not the physics
timestep or control frequency. Workloads slower than real time remain slower.

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
uv run --frozen python -m pytest -q
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
```

The default suite covers portable correctness contracts; it does not contain a
PPO update/checkpoint or pendulum-control smoke test. Warp, licensed VSim, and
Unitree SDK tests are outside the default gate. See the
[test catalog](../tests/TEST_CATALOG.md) for retained coverage and
[limitations and validation scope](#limitations-and-validation-scope) for
remaining gaps and optional-backend support.

### W&B logging

Use `--run_name tracking_trial` to label an individual training run. With
`--task go2trot`, the log directory is
`logs/go2trot/<timestamp>_tracking_trial/` and the W&B display name is
`<timestamp>_tracking_trial`. `--experiment_name` changes the parent experiment
folder under `logs/`; `--run_name` changes the new run's label, including when
resuming with `--original_cfg`. Omitting either option keeps its config value.

Ordinary W&B logging remains supported. Copy the template
[`user/wandb_config_default.json`](../user/wandb_config_default.json) to the local
`user/wandb_config.json` and set `entity` and `project`, or provide
`--wandb_entity` and `--wandb_project` to `scripts/train.py`. The CLI values
override the local configuration. W&B is enabled only when both names are set
and `--disable_wandb` is absent; authentication uses the normal local W&B setup.
Do not put credentials in the template. Orphaned sweep examples and historical
campaign launchers are no longer maintained workflows.

## Limitations and validation scope

The historical `evaluate_height_q_batched.py --insert_step` results do not
establish fixed-state variance/ranking precision: prefix states vary between
repetitions, candidates are reset-centered, and task restore clamps qpos and
omits history/domain state. Its prior noise-ceiling and averaged-first-reward
claims are superseded. `evaluate_height_q_variance.py` is the isolated replacement
experiment, with native integration/task/domain snapshots, paired continuation
noise, raw binary rewards and empirical repeated-batch statistics. Its native
one-step restoration/replication gates must pass before scientific execution;
neither buffer equality nor one-step agreement proves bitwise long-rollout fidelity.

The corrected 128-state × 8-candidate experiment passed native one-step gates
and completed eight independent 16-return batches. Pair-ranking repeatability
between batches was only 0.518, so 16-return estimates remain unstable even
with fixed states and paired noise. Captured native joint-limit penetration
(six positions across five states, max .210443 rad) was preserved without
clamping. See `docs/offline_height_q.md` and the archived replay/relabel plan for
raw artifacts, commands, variance definitions and the separate tail uncertainty.

`scripts/extend_height_q_variance.py` reindexes that saved 1024-world snapshot
to eight seed lanes × sixteen randomly selected original states × eight original
candidates. It gates complete inventory mapping, one-step replication/restoration
and native command/push pairing before collecting more independent 64-return
batches. This remains an experimental nonsleeping Go2Trot/Warp adapter; selected
state coverage and approximate batch-bootstrap uncertainty are distinct from
the original full-state summary. See the offline-height-Q documentation for its
benchmark/resume CLI and matched regrouping analysis.

`scripts/evaluate_height_q_persistent.py` applies both frozen critics at every
decision to choose among eight fresh samples of the same frozen policy, and is
the first positive intervention result recorded for this plan: paired H500
height-band return improved by +1.72 (original critic) and +1.11 (eval-distribution
critic) over 128 fixed k=100 states × 1024 executions, with better band occupancy
but slightly higher fall rate and slightly worse velocity/yaw tracking. Treat this
as a **policy intervention** at those states only, not policy improvement in the
theorem sense: Q was fit for the nominal policy, persistent deviations invalidate
its Bellman targets, and no `Q^pi` or transfer claim follows. Selected indices are
near-uniform, which is the exchangeability prediction and does not show that the
executed actions follow the nominal distribution; no same-bias/no-information
control was run. Execution-level intervals, state-cluster intervals, the
asymmetric survivor tail bound and the paired truncation interval are reported
separately; see the plan's persistent-intervention section and report.

Floating-base tasks use a flat ground plane. Heightfields, triangle-mesh terrain,
and projectile support are not implemented; MuJoCo fixed-base tasks disable
contacts. Startup friction and mass/inertia randomization remain fixed for each
environment's lifetime. The startup API still takes environment indices:
**full-batch startup DR remains unimplemented**. This is separate from the
implemented episodic reset-mask and atomic state-reset behavior.

Native backend, state/reset, task, and focused learning-code tests remain. Hosted
CI runs the portable and colocated suites, Ruff, and a package build; it has no
PPO update/checkpoint or pendulum-control smoke test. Passing native physics or
configuration tests does not establish policy learning quality or checkpoint
resume correctness. Independent storage/GAE, timeout handling, and normalizer
update/freeze/save/load coverage remain areas for review. Deployment utility
tests use synthetic messages and check static-posture offsets, unrestricted
position targets, and the active gait-frequency observation. The
simulation/deployment observation-parity test has been removed; these utility
checks do not establish hardware behavior or checkpoint compatibility.

VSim backend code, its optional dependency extra, and marked tests remain, but
the local test launcher and support workflow are deferred. Hosted CI does not
run VSim, MuJoCo Warp, or Unitree integration. Optional vendor installation and
activation instructions describe prerequisites; they do not establish current
backend validation. Record which optional checks actually executed.

The Go2 visual-mesh replacement has portable XML tests for preserved physical
properties and link-local visual frames, including preprocessing before conversion.
Those tests use a stand-in converter; they do not establish native VSim import
or rendering correctness.

Go2Trot has a focused performance gate in `scripts/evaluate_go2trot.py`:

```bash
uv run --frozen --extra vsim --env-file .env.vsim scripts/evaluate_go2trot.py \
  --backend vsim --device cuda:0 --load_run RUN --checkpoint 550 \
  --original_cfg --output logs/go2trot-performance.json --check
```

It evaluates deterministic actions on flat ground with no pushes or domain
randomization, using ten fixed velocity commands (m/s and rad/s), each at gait
frequencies of 1, 2, and 3 Hz. Four replicas per case run for eight seconds; the
first two seconds are excluded from steady-state metrics. Failures remain in
the survival denominator and are never reset into successful episodes. JSON
records checkpoint hash, source revision/worktree status, resolved configs,
commands, gait frequencies, bounds, and every case's result. `--check` fails
if any case misses a bound. This is an explicit local learning-quality check;
hosted CI tests its metric definitions with synthetic counterexamples and does
not train or download a checkpoint.

The gate measures planar velocity RMSE, yaw-rate RMSE, target/velocity jitter
above 10 Hz (beyond the first three gait harmonics), target clipping, swing-phase
load, total foot support relative to body weight, contact duty, and each foot's
load share over complete oscillator strides.
Consecutive target changes and joint acceleration are also reported; they
include legitimate stride motion, so they are not themselves jitter gates. Equal
whole-run averages cannot hide alternating overloaded strides. Bounds are
initial engineering acceptance criteria, not guarantees for hardware or other
backends. Old policies need their saved observation/action scaling; changing
action scales at playback changes the controller. Config snapshots predating
the current backend config schema may require migration before evaluation.

Go2Trot now uses its sinusoidal gait reference only in `reference_traj`, a
joint-position tracking reward with weight 3.0 and the same squared-exponential
plus absolute-error form as MiniCheetahRef. PD targets are policy offsets from
the static nominal posture (hip 0, thigh 0.96, calf -1.36 radians). Reference
tracking stays active at zero commands, consistent with the existing trot
contact rewards. Phase observations and frequency remain available to the
policy. This changes action and joint-observation semantics: train a fresh
policy. Loading an old config alone does not restore the previous controller's
additive gait reference. Learning quality for this formulation is unvalidated.
The deployment utilities use the same static-posture action convention, without
an additive gait reference or position-target clipping. Their frequency
observation reads the active controller frequency used by the phase clock.
Hardware validation of this formulation remains outstanding.

The earlier additive-reference tuning checkpoint is provisional: reduced
residual action scales,
revised action-rate and yaw-tracking rewards, and revised command sampling
improve some fixed-command cases, but
turning, fast tracking, and jitter still miss acceptance bounds. Observation
scaling and the phase observations remain unchanged. The action-scale increase
dates to commit `23c5465`; the later config refactors preserved those values.

VSim contact sensors now explicitly request `LOCAL` before rotating forces by
the link orientation. Applying that rotation to `ENVIRONMENT` results gave
incorrect world forces, corrupting gait rewards and load metrics. The native
standing-Go2 regression checks gravity balance at four yaw angles and fails
with the previous frame selection. Policies from the tuning trials were trained
before this fix; their corrected-force evaluations do not establish successful
learning with the repaired reward signal. Retraining remains necessary. Older
reports from before the fix are invalid evidence for contact-load quality.

General-purpose policy comparison reports, benchmark/profile tools, and
their analysis helpers are no longer included. Training, playback,
`--original_cfg`, and ordinary W&B logging remain available. The retired pendulum
calibration failed its learning threshold; its removal supplies no passing
learning-quality evidence. Historical reports were generated, ignored artifacts
and are not guaranteed to exist in a checkout.

Offline replay and fixed-policy height Q are opt-in and independent of PPO
storage. Snapshot, relabel, FQE-target, artifact, and finite-rollout-tail tests
are analytic or fixture-based. The one-iteration CPU smoke verifies the software
path, not Q accuracy or policy quality. Observation-only Q can be biased by
unobserved physical state, collecting behavior policy coverage is finite, and
the validated-domain report only establishes what was actually tested. MPPI,
repeated intervention, other reward channels, bound conditioning, and
finite-horizon Q remain unvalidated. See
[offline replay and fixed-policy height Q](../docs/offline_height_q.md).

## Notes

**NaN rewards at startup:** reward logging averages over completed episodes.
Until the first episode finishes, all reward values show `nan` — this is
expected and not an error.

## CLI Reference

```
uv run --frozen scripts/train.py [OPTIONS]

  --task TEXT          Task name (required). See "Available Tasks" below.
  --device TEXT        "cpu" or "cuda:0" (default: cpu)
  --num_envs INT      Number of parallel environments (default: from task config)
  --max_iterations INT Training iterations (default: from task config)
  --seed INT          Random seed
  --experiment_name TEXT Experiment folder under logs/ (default: task config)
  --run_name TEXT      Run label appended to the timestamp in logs and W&B
  --resume            Resume model and optimizer state
  --load_run TEXT     Run directory below the selected experiment
  --checkpoint INT    Checkpoint iteration (default: latest)
  --original_cfg      Load environment and runner configs from the selected run
  --headless          Disable GUI viewer
  --disable_wandb     Disable configured Weights & Biases logging
```

## Available Tasks

| Task | Robot | Type | DOFs |
|------|-------|------|------|
| `pendulum` | Pendulum | Fixed-base | 1 |
| `cartpole` | Cart-pole | Fixed-base | 2 |
| `mini_cheetah` | MIT Mini Cheetah | Floating-base | 12 |
| `mini_cheetah_ref` | Mini Cheetah (ref tracking) | Floating-base | 12 |
| `mini_cheetah_osc` | Mini Cheetah (oscillator) | Floating-base | 12 |
| `go2` | Unitree Go2 | Floating-base | 12 |
| `go2trot` | Unitree Go2 (trot based on clock) | Floating-base | 12 |
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

With `--backend mujoco` (the default), the backend is selected by `--device`:

- `cpu` → MuJocoCPUBackend (macOS always uses this)
- `cuda:0` → MuJocoWarpBackend

Use `--backend vsim --device cuda:0` to select the licensed VSim backend.

### MuJoCo configuration

Both MuJoCo backends read one `cfg.mujoco` section. Override individual fields
by inheriting the task's section:

```python
class MyGo2Cfg(Go2Cfg):
    class mujoco(Go2Cfg.mujoco):
        njmax = 300
        ccd_iterations = 75
```

The supported settings are `njmax` (Warp constraint-row capacity per world),
`ccd_iterations` (collision-detection iterations), `disableflags` (MuJoCo's
disable-bit mask), and `solref` (contact solver reference for all geometries).
Robot-specific defaults preserve each task's contact tuning. Simulation timing
still comes from `control.desired_sim_frequency`, not this section. VSim does
not consume these MuJoCo settings.

This replaces `mjspec_attributes`, `mjspec_option_attributes`, and
`mjspec_geom_attributes`. Configs must use the unified section; older saved
configs are not converted automatically.

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
[`gym/envs/base/robot_layout.py`](../gym/envs/base/robot_layout.py) for the
implementation and
[`gym/envs/mini_cheetah/mini_cheetah_config.py`](../gym/envs/mini_cheetah/mini_cheetah_config.py)
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
feet = layout.body_group_indices("feet")  # (3, 6)

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
[`gym/envs/__init__.py`](../gym/envs/__init__.py): the task class, environment
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
Worlds are stepped serially; measure throughput for the task and environment
count you intend to use.

GPU training (`--device cuda:0`) is not available on macOS.

**GUI viewer on macOS:** MuJoCo's passive viewer requires `mjpython` (bundled
with the `mujoco` pip package) instead of the standard Python interpreter.
If `mjpython` cannot load the matching `libpython` dylib from uv's bundled
interpreter, create the venv using Homebrew's Python 3.11 instead:

```bash
brew install python@3.11   # if not already installed
uv venv --python "$(brew --prefix python@3.11)/bin/python3.11"
uv sync --frozen
.venv/bin/mjpython scripts/train.py --task mini_cheetah --device cpu --num_envs 64
```

Alternatively, use `--headless` to skip the viewer entirely (works with uv's default Python).

### Linux (CPU)

Same as macOS. Use `--device cpu`.

### Linux (GPU)

Requires the NVIDIA driver and the `gpu` extra (`uv sync --frozen --extra gpu`).
Use `--device cuda:0` for GPU-accelerated vectorized physics.
Measure throughput for the actual task and workload; performance depends on
environment count, contacts, solver configuration, and hardware.

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
│   ├── train.py                         ← public training
│   ├── play.py                          ← interactive playback
│   └── fetch_unitree_sdk.py             ← optional hardware SDK setup
├── tests/unit_tests/                    ← automated regression gate
├── genAI_skills/                        ← developer guidance and validation scope
└── pyproject.toml
```
