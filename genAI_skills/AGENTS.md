# Q2 Developer Guide

This file applies to the entire repository. Q2 is a Python robotics and
reinforcement-learning codebase descended from `legged_gym`/pkGym. Its active
architecture separates task and learning logic from interchangeable physics
backends: MuJoCo CPU, MuJoCo Warp, and optional licensed VSim.

## Start Here

Before changing code:

1. Run `git status --short --branch` and preserve all unrelated work. Dirty
   worktrees are normal in this active research repository.
2. Read the nearest implementation, its config, and its tests. Do not treat an
   old comment, benchmark, pass count, or branch status as current without
   checking it.
3. Use [README_MUJOCO.md](README_MUJOCO.md) for setup and CLI guidance and its
   [limitations and validation scope](README_MUJOCO.md#limitations-and-validation-scope)
   for evidence boundaries. Prefer current code and tests when guidance has drifted.
4. Load the relevant repository skill from `.agents/skills/` for specialized
   workflows.

Do not overwrite, revert, reformat, or incorporate unrelated user changes.
Keep a change focused; research tuning, physics changes, refactors, and bug
fixes should remain separately reviewable.

## Repository Map

- `gym/envs/base/`: backend contract, canonical `RobotLayout`, task lifecycle,
  common robot logic, MuJoCo CPU/Warp, and VSim.
- `gym/envs/<task>/`: concrete robot/task implementations and their nested
  environment and runner configs.
- `gym/utils/task_registry.py`: task construction, config derivation, backend
  selection, runner creation, and log/checkpoint resolution.
- `learning/`: actors, critics, algorithms, runners, storage, normalization,
  and logging. `PPO2` plus `OnPolicyRunner` is the main on-policy path; other
  runners and critics include legacy and research code.
- `scripts/`: public training, playback, and optional SDK setup. `train.py` and
  `play.py` serve every supported backend.
- `genAI_skills/`: developer guidance, setup, and validation limitations.
- `tests/unit_tests/`: cross-cutting correctness gate, including backend
  contracts, physics invariants, state liveness, canonical routing, task
  behavior, and focused learning-code contracts.
- Colocated `test_*.py` files under `gym/` and `learning/`: fast,
  deterministic tests for small environment-agnostic implementations. Keep
  these beside the code when they also provide a useful local usage example.
- `resources/robots/`: URDFs, meshes, trajectories, and robot assets.
- `notebooks/`: cloud training examples; excluded from Ruff.
- `thirdparty/vlearn/`: local-only licensed VSim wheel, license data, and SDK
  support files. Never commit vendor binaries or credentials.
- `logs/`: generated checkpoints, source snapshots, metrics, and experiment
  artifacts. It is not source code and is gitignored.

## Environment

CORL is an optional source submodule at `thirdparty/CORL`, with its own Git
history and development branch matching BRG's branch. See
[`docs/thirdparty/CORL/install.md`](../docs/thirdparty/CORL/install.md) for setup
and supported scope. The adapted IQL artifact smoke path uses existing Torch
and NumPy; do not install upstream CORL requirements into BRG indiscriminately.
BRG consumes weights and metadata without importing CORL or OpenAI Gym. Keep
the current package names; namespace migration is not required for this path.
Commit/publish submodule changes separately before recording their revision in
BRG; a submodule pointer does not capture uncommitted adaptations.

Use uv to manage the checked-in `.venv` and lockfile. `pyproject.toml` is the
only dependency and package source of truth; do not add parallel pip setup or
requirements files.

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
```

Using `.venv/bin/python` directly is appropriate for tools that must bypass uv
resolution, but normal repository commands should use `uv run --frozen`.

For MuJoCo Warp on Linux/NVIDIA:

```bash
uv sync --frozen --extra gpu
uv run --frozen python -c "import torch, mujoco_warp; print(torch.cuda.is_available())"
```

VSim is optional, CUDA-only, licensed, and machine-local. Follow
`thirdparty/README.md`, install with `uv sync --frozen --extra vsim`,
and launch processes with `uv run --frozen --extra vsim --env-file .env.vsim ...`.
Backend code, the optional extra, and marked tests remain; the local test
launcher and support workflow are deferred. The default test gate deselects VSim.

On macOS, headless MuJoCo CPU works normally. The passive viewer must run
under `mjpython` with a compatible dylib-bearing Python; use the recipe in
`genAI_skills/README_MUJOCO.md` or pass `--headless`.

## Architectural Contracts

Treat `gym/envs/base/sim_backend.py` and its contract tests as the executable
specification.

- `setup()` must leave every public state tensor valid. `step()` must update
  every cached public tensor in place before returning. Never hide refresh
  work in a property getter: tasks cache these tensors during initialization.
- `dof_state` is `[num_envs * num_dof, 2]`; `dof_pos` and `dof_vel` are live,
  writable views into persistent public state. An engine may use private native
  buffers, but it must gather/scatter through persistent canonical buffers.
- Resets use one atomic write-then-commit operation. Task code writes public
  DOF and root state, then calls `reset_state(reset_mask)` once. Each backend
  commits the complete selected state and performs one native forward/refresh.
- The task-facing quaternion convention is scalar-last `[x, y, z, w]`.
  Engine-specific ordering conversions belong only at the backend boundary.
- Public DOFs, bodies, actions, contacts, and state tensors use
  `RobotLayout`'s canonical order. Native joint, motor, body, and sensor orders
  are private. Resolve semantic groups by exact names once at setup; do not
  encode leg or joint meaning with positional slices.
- `contact_forces` means the net collision-contact force for each canonical
  robot body in world coordinates and Newtons. Body order excludes a simulator
  world body.
- Torques enter a backend in canonical full-DOF order. The backend handles
  free-joint offsets and native ordering.
- Fixed-base robots have no free joint and MuJoCo disables contacts for them.
  Floating-base robots expose `root_states` and `rigid_body_states` and use the
  configured flat ground plane.
- MuJoCo CPU and Warp share model loading/configuration in
  `mujoco_backend_base.py`; common physics behavior belongs there, not in only
  one implementation.

Any change to setup, step, reset, state assembly, canonical mapping,
quaternion conversion, or contact routing requires a focused regression test.

## Tasks, Configs, and Learning

- Configs are nested `BaseConfig` classes that become instances recursively.
  Physics/world settings belong in the environment config; algorithm, network,
  reward, and logging settings belong in the runner config.
- Control frequency and desired simulation frequency are the source of truth.
  Let `task_registry.convert_frequencies_to_params()` derive decimation,
  `ctrl_dt`, `sim_dt`, and horizon-based discounts.
- Add a task to `class_dict`, `config_dict`, `runner_config_dict`, and
  `task_dict` in `gym/envs/__init__.py`. Task imports are fail-fast, and the
  registry manifest test verifies that every declaration registers exactly.
- Runner observation lists contain environment attribute names.
  `TaskSkeleton.get_state()` divides named quantities by their configured
  scale; action assignment multiplies them back. Every named scaled quantity
  needs a matching `cfg.scaling` entry.
- Reward methods are named `_reward_<weight_name>`, return one value per
  environment, and are selected from nonzero runner-config weights. Supply an
  error whose ideal value is zero to `_sqrdexp`; reduce comparable per-joint
  penalties with a mean unless the intended magnitude genuinely scales with
  robot DOF count.
- Keep rollout collection and optimization geometry distinct.
  `algorithm.rollout_size` controls collected samples/temporal horizon;
  `batch_size` controls PPO optimizer minibatches. Backend comparisons are not
  controlled if either rollout horizon or sample count differs unintentionally.
- Inference must use clean observations and evaluation mode. Observation
  normalizers update on fresh rollout data, not repeatedly on every optimizer
  pass, and their state must survive checkpoint resume.

## Style and Change Discipline

The repository contains modern backend code alongside older research code;
match the local module while improving touched code deliberately.

- Target Python 3.11, four spaces, double quotes, and Ruff's 88-column format.
- Prefer explicit failures to silent fallback behavior. Do not add broad
  `try/except`, default-valued config reads that mask a missing required field,
  or automatic backend substitution. Optional dependency boundaries and
  cleanup that preserves the original exception are legitimate exceptions.
- Prefer querying attributes directly instead of using `getattr()`, e.g. `getattr(cfg, "seed", None)` should be `cfg.seed`, and it should be allowed to fail: if there is supposed to be a default value, it should be set in the inherited default config. Other than that, the user should be forced to be responsible for setting things. This forces the user to be more aware of the implementation.
- Preserve established public names, including historical capitalization such
  as `MuJocoCPUBackend`, unless the task is an explicit API migration.
- Avoid wildcard imports and `eval` in new code even where legacy code uses
  them. Use typed, direct imports and small helpers.
- Add comments for non-obvious engine semantics, tensor aliasing/liveness, or
  experiment rationale. Do not narrate obvious Python.
- Never commit generated checkpoints, logs, W&B data, vendor license/activation
  files, local wheels, or caches. New or modified files must stay within 100
  KiB, except for `uv.lock`, which records the dependency graph. Hooks and CI
  enforce the limit.
- Format only touched Python files in a dirty worktree, then run the full lint
  check:

```bash
uv run --frozen ruff format <touched-python-files>
uv run --frozen ruff check .
```

## Validation

Run the narrowest diagnostic test while iterating, then the complete local
gate before handoff:

```bash
uv run --frozen python -m pytest tests/unit_tests/path_or_test.py -q
uv run --frozen python -m pytest -q
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
```

`pyproject.toml` sets `testpaths = ["tests/unit_tests"]`, so bare pytest targets
the cross-cutting CPU suite. Explicit `pytest gym` and `pytest learning` runs
also collect colocated implementation tests. Colocate a test only when it is
small, deterministic, independent of simulator/full-stack setup, and useful as
an example of the implementation's interface. Put backend fixtures,
cross-package contracts, task construction, and integration behavior under
`tests/unit_tests/`. MuJoCo Warp tests use the `warp` marker and licensed VSim
tests use the `vsim` marker; optional SDK tests use `unitree`. Request these
groups explicitly.

Additional evidence by change type:

- Backend/state/reset/contact change: relevant contract, liveness, routing,
  physics, and CPU/Warp comparison tests. Run `uv run --frozen python -m pytest
  -q -m warp` on CUDA and licensed VSim tests if touched.
- New robot/task: registry proof, reward/observation shape tests, asset limit
  and inertia checks as relevant, then a tiny CPU smoke train.
- Reward/algorithm/normalization change: focused math/unit tests plus a
  controlled same-seed before/after experiment with per-term rewards and
  rollout geometry recorded.
- Physics or parity claim: a predicted invariant or fidelity probe, not only a
  viewer impression or final aggregate reward.

GitHub CI uses uv and runs the portable default suite, explicit colocated
suites, Ruff, and a package build. It contains no PPO update/checkpoint or
pendulum-control smoke test and does not establish learning quality. MuJoCo
Warp, licensed VSim, and Unitree integration are outside hosted CI; record
applicable local evidence and any unavailable validation explicitly.

## Documentation and Skills

- Update `genAI_skills/README_MUJOCO.md` when setup, platform support, or public CLI changes.
- Update [limitations and validation scope](README_MUJOCO.md#limitations-and-validation-scope)
  when supported limitations or evidence boundaries change. Keep consequential
  failed or invalid evidence visible without restoring a campaign chronology.
- Keep `genAI_skills/AGENTS.md` limited to durable repository-wide rules.
- Keep `.agents/skills/` procedural and source-linked. Avoid hard-coded branch
  heads, pass counts, performance numbers, or open/closed statuses that can be
  discovered from current sources.
- Use git history for failure archaeology. Convert a historical lesson into a
  test or invariant instead of maintaining another volatile chronology.

Repository skills:

- `q2-development-environment`: uv, `.venv`, platforms, GPU/VSim setup.
- `q2-backend-development`: backend contracts, canonical mapping, state/reset.
- `q2-task-authoring`: assets, configs, tasks, registration, smoke validation.
- `q2-rl-development`: observations, rewards, PPO/storage/normalization changes.
- `q2-train-and-evaluate`: training, resume, playback, controlled evaluation.
- `q2-testing-and-debugging`: test selection, discriminating probes, regressions.
