# Backend Progression and Parity Plan

## Current scope

Q2 supports three execution targets behind one task-facing contract:

- MuJoCo CPU for portable development, CI, and interactive playback.
- MuJoCo Warp for batched CUDA training.
- Optional licensed VSim for CUDA training and cross-engine evaluation.

The supported world is currently a flat ground plane. Periodic `push_robots`
disturbances remain part of legged-task behavior. Heightfields, trimeshes, and
projectiles are outside the current scope.

Domain randomization is intentionally deferred. It will return as a new
backend-neutral feature with one sampling contract and explicit effects in the
MuJoCo and VSim engine families. Do not reintroduce engine-specific friction or
mass callbacks in the meantime.

## Architecture

```text
SimBackend
├── MuJocoCPUBackend
├── MuJocoWarpBackend
└── VSimBackend

TaskSkeleton
└── BaseTask
    ├── FixedRobot
    └── LeggedRobot
        └── concrete tasks
```

`gym/envs/base/sim_backend.py` and its contract tests are the executable
specification. Tasks consume canonical tensors and named robot semantics;
engine-native ordering and representation stay behind each backend boundary.

Core invariants:

- Public state tensors are valid after setup and refreshed in place after
  every step and reset.
- `dof_pos` and `dof_vel` are writable views into persistent `dof_state`.
- Root and DOF resets follow write-then-commit ordering without clobbering a
  pending root-state write.
- Task-facing quaternions are scalar-last `[x, y, z, w]`.
- Public DOFs, bodies, torques, contacts, and state tensors use canonical
  `RobotLayout` order.
- Contact forces are robot-body net collision forces in world coordinates and
  Newtons.

## Correctness gates

### Portable gate

The default suite combines pure unit behavior, MuJoCo CPU backend contracts,
task registration, and a one-action-step construction test for every declared
task:

```bash
uv run --frozen python -m pytest -q
```

Small deterministic implementation tests remain beside their code and run
explicitly:

```bash
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
```

### CUDA and licensed gates

```bash
uv run --frozen python -m pytest -q -m warp
bash scripts/run_vsim_tests.sh
```

A deselected or skipped hardware group is not passing evidence. Record the
device, backend, seed, environment count, and exact executed tests.

### Repository gate

```bash
uv run --frozen ruff check .
uv build --no-sources
```

GitHub CI runs the portable and colocated suites, Ruff, and the package build.
Smoke training and hardware-specific groups remain explicit local gates.

## Physics and parity evidence

### Pendulum

The analytic energy-pump plus LQR probe exercises a deterministic grid of
initial angles and velocities. MuJoCo CPU, MuJoCo Warp, and VSim all reached a
100% catch rate with a mean catch time of approximately 1.08 seconds in the
recorded 1024-environment campaign. Angular divergence RMS against MuJoCo CPU
was approximately `9e-7 rad` for Warp and `5e-4 rad` for VSim.

Use `scripts/pendulum_fidelity.py` to rerun the probe. Treat these figures as
recorded evidence, not permanent thresholds detached from the current code.

### Mini Cheetah reference tracking

MuJoCo CPU and Warp now agree closely through free flight and contact after the
root-reset ordering and contact refresh fixes. VSim agrees before contact and
has bounded solver-specific differences during contact.

Important invalid evidence remains visible:

- Warp checkpoints produced before the floating-base reset ordering fix are
  invalid. A DOF reset refreshed assembled root state before the pending root
  reset was committed, spawning robots at the wrong height.
- Early contact probes included an unintended initialization step before
  recording. Only probes that restore the requested state immediately before
  capture are comparable.
- A prior VSim contact path returned link-frame vector components despite an
  environment-frame request. The backend now rotates those components into
  world axes. A settled Mini Cheetah supports approximately its `81.3 N`
  weight along world `+z`.
- The last recorded cross-engine transfer campaign remained incomplete: a
  VSim-trained checkpoint failed on both MuJoCo targets even though the other
  tested transfer directions passed. Do not present that campaign as full
  policy parity.
- A later VSim training run ended because the host GPU fell off the PCIe bus,
  not because the policy diverged. Its partial checkpoint is not promotion
  evidence.

Use `scripts/mini_cheetah_fidelity.py`, `scripts/eval_policy.py`, and the
checked-in benchmark wrappers for new evidence. Keep checkpoint selection,
commands, reset distribution, episode duration, and rollout geometry fixed
across transfer cells.

## Promotion criteria

A backend or policy is ready for promotion only when all applicable checks
pass:

1. Shared contract, liveness, reset, routing, contact, and task tests pass.
2. A policy-free invariant or fidelity probe explains physics agreement and
   any accepted cross-engine tolerance.
3. Short training remains finite and produces a valid checkpoint.
4. Deterministic evaluation passes the declared command and reset suite.
5. Cross-backend transfer results are reported for every intended direction;
   failed and invalid cells remain visible.
6. Environment count, timestep, rollout horizon, optimizer batch geometry,
   device, seed, warnings, and checkpoint-selection rule are recorded.

Aggregate reward or a viewer impression alone is not promotion evidence.

## Planned learning-stack pruning

The learning package currently mixes the supported PPO path with configured
research paths and code that no registered task can select. Pruning must follow
runtime reachability and an explicit support decision; an import or historical
checkpoint alone does not make a path supported.

### Inventory

| Status | Runtime chain | Current consumer |
|---|---|---|
| Core | `OnPolicyRunner` → `PPO2` → `Actor`/`SmoothActor` + `Critic` → `DictStorage` | All standard PPO tasks |
| Configured research | `OffPolicyRunner` → `SAC` → `ChimeraActor` + `Critic` → `ReplayBuffer` | `sac_pendulum`, `sac_mini_cheetah` |
| Configured research | `PSACRunner` → `SAC` → `ChimeraActor` + `DenseSpectralLatent` → `ReplayBuffer` | `psd_pendulum` |
| Unconfigured | `CustomCriticRunner`, `MyRunner`, `DataLoggingRunner` | Exported by `learning.runners`, but selected by no registered task |
| Legacy | `OldPolicyRunner` → deprecated `PPO` → `ActorCritic` → `RolloutStorage` | No registered task |
| Orphaned | `StateEstimator` → `StateEstimatorNN` → `SERolloutStorage` | No registered task |
| Partially reachable | Other classes in `QRCritics.py` | Only `DenseSpectralLatent` is selected by a registered task |

Environment-agnostic utilities are not removal candidates merely because a
runner does not call them. Keep normalization, logging, dictionary utilities,
and the colocated PBRS implementation/tutorial unless their own behavior or
tests show that they are obsolete.

### Pruning sequence

1. Keep the core PPO chain as the supported baseline. Add no compatibility
   aliases for removed selectable class names; registry/config failures should
   remain explicit.
2. Decide whether SAC and PSD-SAC are supported research features. For each of
   `sac_pendulum`, `sac_mini_cheetah`, and `psd_pendulum`, run a reduced CPU
   smoke that crosses replay-buffer initialization, performs updates, saves a
   checkpoint, resumes it, and produces finite deterministic inference. Keep
   and test a chain only if that evidence passes and the feature is still
   wanted; otherwise unregister its tasks and configs before deleting it.
3. Remove the definitely unreachable chains in separate reviewable changes:
   first `OldPolicyRunner`/`PPO`/`ActorCritic`/`RolloutStorage`, then the state
   estimator chain, then the three unconfigured runner variants. Update package
   exports and delete tests that exist only for a removed path in the same
   change.
4. After the SAC/PSD decision, reduce `QRCritics.py` to retained classes and
   their demonstrated dependencies. If PSD-SAC stays, give its selected critic
   focused math and checkpoint tests rather than preserving every historical
   critic architecture.
5. Normalize module names only after reachability is settled: for example,
   rename `BaseRunner.py` to `base_runner.py` and any retained custom-critic
   module to snake case. This avoids renaming code immediately before deleting
   it.
6. For every retained runner family, require a registered task, focused unit
   coverage, a tiny train/save/resume/inference smoke, and explicit inclusion
   in developer documentation. Finish each pruning slice with the portable,
   colocated, lint, and package-build gates.

Pruning is complete when every selectable runner and algorithm has a declared
consumer and evidence, every registered learning task uses a supported chain,
and the package no longer exports unreachable implementations.

## Planned domain randomization

Add domain randomization later as a standalone feature, in this order:

1. Define a backend-neutral configuration and sampling API, including seed and
   per-environment reset semantics.
2. Start with friction and body-mass perturbations. State exactly which model
   quantities change and whether changes require engine rebuild, model copies,
   or runtime buffers.
3. Implement MuJoCo CPU first with deterministic sampling and narrow tests.
4. Implement the same observable contract for MuJoCo Warp and VSim without
   silently dropping unsupported axes.
5. Add distribution, reset-isolation, reproducibility, and backend-consumption
   tests before robustness training.
6. Compare nominal and randomized policies with identical rollout and
   evaluation geometry.

Until that work starts, keep training physics deterministic apart from
task-level initial-state, command, and `push_robots` sampling.

## Common commands

```bash
# CPU smoke train
uv run --frozen scripts/train.py \
  --task pendulum \
  --backend mujoco \
  --device cpu \
  --num_envs 8 \
  --max_iterations 2 \
  --headless \
  --disable_wandb

# CUDA training
uv run --frozen scripts/train.py \
  --task mini_cheetah \
  --backend mujoco \
  --device cuda:0 \
  --headless

# Interactive CPU playback
uv run --frozen scripts/play.py --task mini_cheetah --device cpu
```
