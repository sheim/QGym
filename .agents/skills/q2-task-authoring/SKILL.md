---
name: q2-task-authoring
description: Add or modify a Q2 robot, environment, task variant, asset, config, or registry entry end to end. Use when introducing a URDF, creating fixed-base or legged tasks, adding semantic robot groups, changing task observations/actions/rewards, registering a task, or making an existing task work across MuJoCo CPU, Warp, and optional VSim.
---

# Q2 Task Authoring

Read `AGENTS.md`, the nearest existing task and config, the appropriate base
robot class/config, `gym/envs/__init__.py`, `robot_layout.py`, and task-focused
tests. Copy the nearest semantic example, not merely the newest file.

## Choose the task shape

- Use `FixedRobot` for a base attached to the world. Its MuJoCo path disables
  contacts and has no `root_states` commit.
- Use `LeggedRobot` for a free base, ground contact, canonical body state, and
  contact-driven termination/rewards.
- Prefer a config/task variant when physics and robot layout are shared. Do
  not fork backend logic into a concrete task.

## Add the asset

- Place URDFs below `resources/robots/<robot>/`. Give every actuated joint
  finite lower/upper, effort, and velocity limits.
- Keep source inertias physically valid: positive principal moments satisfying
  the triangle inequality. Add a parser-level regression for repaired assets.
- Prefer supported `.stl`/`.obj` visual meshes; MuJoCo strips unsupported DAE
  visuals. Keep collision geometry independent of visual availability.
- Keep files below the repository's 100 KB limit; never solve this by
  force-adding a binary.
- Define an explicit `asset.robot_layout` when policy-facing order or semantic
  groups matter. Canonical names must exactly cover movable URDF joints and
  robot links. Define groups such as legs, feet, or end effectors by exact
  names; avoid semantic slices and substring order assumptions.

## Add configuration

Create environment and runner config subclasses in the task package.

- Set `env.num_envs`, actuator count, episode length, asset path/base mode,
  initial state, control frequencies, terrain/contact lists, scaling, and
  robot layout explicitly.
- Set actor/critic observation attribute lists, action state names, algorithm
  sampling/optimization parameters, reward weights, and experiment name.
- Treat `ctrl_frequency` and `desired_sim_frequency` as inputs. Never assign
  derived dt/decimation values in the task config.
- Make required config axes required. If a backend does not implement a
  configured feature such as per-env domain randomization, disable or document
  it explicitly rather than implying that it works.

## Implement the task

- Keep engine imports and native indices out of concrete task code.
- Preserve the constructor's legacy positional parameters and forward
  `backend=` to the base class.
- Build observations from stable environment attributes; add matching scales
  for every scaled state. Keep action tensors and actuator routing in
  canonical order.
- Name rewards `_reward_<config_weight>`, return `[num_envs]`, and use exact
  semantic group indices cached at setup.
- Use zero-error inputs with `_sqrdexp`; reduce per-joint terms with `mean`
  unless scale-by-DOF is explicitly intended.

## Register and package

Add all four mappings in `gym/envs/__init__.py`: `class_dict`, `config_dict`,
`runner_config_dict`, and `task_dict`. Add the package to
`[tool.setuptools].packages` in `pyproject.toml`. Imports are fail-fast; inspect
the live registry after adding the declarations:

```bash
uv run --frozen python -c "import gym.envs; from gym.utils.task_registry import task_registry; print(sorted(task_registry.task_classes))"
```

## Prove the task

1. Add registry, layout/routing, asset-limit/inertia, and reward/observation
   shape tests appropriate to the task.
2. Run the full unit gate:

```bash
uv run --frozen python -m pytest -q
```

3. Smoke train the CPU path with few environments and iterations, headless and
   without W&B. Use the new task name:

```bash
uv run --frozen scripts/train_mujoco.py --task TASK --backend mujoco --device cpu --num_envs 8 --max_iterations 2 --headless --disable_wandb
```

4. Exercise Warp and VSim only after CPU correctness, with their real contract
   tests and exact-name routing checks. A skipped optional-backend test is not
   a pass. Update `README_MUJOCO.md` if the public task list changes.
