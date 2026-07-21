---
name: q2-config-system
description: Q2's configuration system — BaseConfig class trees (env cfg + runner/train cfg), task registration dicts, frequency→dt derivation, CLI overrides, which fields the MuJoCo backends actually consume vs silently ignore (terrain, projectiles, IsaacGym asset options), and domain-randomization axes. Load when reading/adding/overriding any config field, when a config change has no effect, or when adding a config axis. NOT for reward-function design (legged-rl-reference) or task creation end-to-end (q2-task-authoring).
---

# Q2 Config System

## Shape of a configuration

Every task has TWO config objects, both `BaseConfig` subclasses
(`gym/envs/base/base_config.py:36-60` — nested classes are auto-instantiated
recursively at construction):

- **env cfg** (`LeggedRobotCfg` or `FixedRobotCfg`): `env` (num_envs,
  num_actuators, episode_length_s), `terrain`, `asset` (file, fix_base_link,
  joint_damping, rotor_inertia, penalize_contacts_on,
  terminate_after_contacts_on, disable_gravity), `init_state`, `control`
  (ctrl_frequency, desired_sim_frequency), `commands`, `push_robots`,
  `domain_rand`-style fields, `scaling` (per-quantity obs/action scales).
- **runner cfg** (`*RunnerCfg`): `runner` (experiment_name, run_name,
  max_iterations, save_interval, algorithm_class_name, device, resume,
  load_run, checkpoint), `algorithm` (batch_size, learning rates, PPO/SAC
  hyperparameters), `actor`/`critic` (obs lists, network shape, normalize_obs,
  `critic.reward.weights` + `termination_weight`), `logging`
  (enable_local_saving).

Configs are plain nested classes — change by subclassing (new task) or by
attribute assignment before env construction (scripts do CLI overrides this
way). There is no YAML/CLI framework; `scripts/train_mujoco.py:52-67` is the
pattern to copy.

## Registration (gym/envs/__init__.py)

Three dicts map names → module paths: `class_dict`, `config_dict`,
`runner_config_dict`; `task_dict` binds a task name to a
`[TaskClass, EnvCfg, RunnerCfg]` triple. Each import is individually guarded so
isaacgym-free tasks register even when isaacgym-dependent ones can't (lines
80-122). A task silently disappears from the registry if ANY of its three
imports fails — if `task_registry.get_cfgs(name)` KeyErrors, check imports of
that task's module first.

## Frequencies → timesteps (never set dt by hand)

`task_registry.convert_frequencies_to_params(env_cfg, train_cfg)`
(`gym/utils/task_registry.py:159-178`) computes:

```
decimation = int(desired_sim_frequency / ctrl_frequency)
ctrl_dt    = 1 / ctrl_frequency          → env_cfg.control.ctrl_dt
sim_dt     = ctrl_dt / decimation        → env_cfg.sim_dt  (consumed by backends)
```

Current values: pendulum 25 Hz ctrl / 200 Hz sim (decimation 8); mini_cheetah
100/500 (5); mini_cheetah_ref 50/500 (10); humanoid 100/500 (5). Discount
factors are also derived from horizons here (`set_discount_rates`), see
`legged-rl-reference`.

## What the MuJoCo backends actually read

From `mujoco_backend_base.py` `_load_model`/`_configure_model` (verified lines
88-221): `asset.file` (with `{LEGGED_GYM_ROOT_DIR}` substitution),
`asset.fix_base_link` (adds free joint when False), `sim_dt`, `sim.gravity`,
`asset.disable_gravity`, `asset.joint_damping`, `asset.rotor_inertia`
(→ armature), `asset.penalize_contacts_on`, `asset.terminate_after_contacts_on`
(substring match against body names), `terrain.mesh_type` (**only `"plane"`
does anything**), `terrain.static_friction`, `terrain.dynamic_friction`.

**Silently ignored under MuJoCo** (IsaacGym-era fields — editing them changes
nothing): `env.env_spacing`, `env.num_projectiles`, all terrain machinery except
the flat plane (heightfield/trimesh/curriculum/measure_heights),
`asset.collapse_fixed_joints`, `replace_cylinder_with_capsule`,
`flip_visual_attachments`, `density`, `angular/linear_damping`,
`max_*_velocity`, `thickness`, `default_dof_drive_mode`, `self_collisions`.
When a config edit "does nothing", check this list first.

## Domain randomization axes (status: UNVERIFIED under MuJoCo)

- `randomize_friction = True`, `friction_range = [0.5, 1.0]` (mini_cheetah,
  `mini_cheetah_config.py:83-84`); humanoid `[0.5, 1.25]` +
  `randomize_base_mass = True`, `added_mass_range = [-1.0, 1.0]`.
- These are applied via `_process_rigid_shape_props`/`_process_rigid_body_props`
  callbacks that the **IsaacGym** backend invokes per-env. The MuJoCo backends
  call only `_get_env_origins` and `_process_dof_props`
  (`mujoco_backend_base.py:223-230`) — **friction/mass randomization almost
  certainly does not reach the MuJoCo sim**. This is exactly the two unchecked
  boxes at the bottom of MIGRATION_PLAN.md. Verification experiment lives in
  `q2-phase4-parity-campaign` Phase 3. Do not claim DR works until it's proven.
- `push_robots` (periodic base-velocity kicks) IS backend-agnostic
  (`set_all_root_states`, re-enabled by `2c5d64d`).

## Key per-task numbers (2026-07-10, for orientation not gospel)

| task | num_envs | ep len (s) | ctrl/sim Hz | notable |
|---|---|---|---|---|
| pendulum | 4096 | 10 | 25/200 | weights: equilibrium 1.0, energy 0.5, theta/omega 0.1 |
| mini_cheetah | 4096 | 3 | 100/500 | terminate on `base` contact; tracking_lin_vel 4.0 |
| mini_cheetah_ref | 4096 | 5 | 50/500 | + `reference_traj` reward; terminate on `base`+`thigh` |
| humanoid | 4096 | 5 | 100/500 | 18 actuators; mass DR configured |

Reward weights live in `<Task>RunnerCfg.critic.reward.weights`; a weight of 0
removes the function from the computed set entirely
(`remove_zero_weighted_rewards`).

## Adding a config axis — checklist

1. Add the field on the appropriate cfg class (env vs runner — physics/world →
   env cfg; learning/logging → runner cfg).
2. Consume it explicitly; **no `getattr(..., default)` fallbacks in new code**
   (fail-fast house rule; the existing `getattr`s in `mujoco_backend_base.py`
   predate it).
3. If a backend must consume it, implement in `MuJocoBackendBase` (shared) or
   both backends — never CPU-only (that's how warp-only bugs are born).
4. Engine-model knobs go through the generic mechanism (pulled from jt/port
   into `vsim` 2026-07-11): every attribute of `cfg.mjspec_attributes` /
   `cfg.mjspec_option_attributes` is setattr'd onto the MjSpec / spec.option
   pre-compile (`mujoco_backend_base.py`). CAVEAT learned the hard way:
   spec/model fields that mujoco-warp does not read (e.g. legacy
   `mjModel.njmax`) must ALSO be forwarded explicitly — the warp backend
   passes `mjm.njmax` to `mjw.put_data(njmax=...)`. Current mini_cheetah
   values: `njmax = 200` (warp demanded 160 post-fusestatic; overflow =
   silently dropped contacts), `ccd_iterations = 50` (100 halves throughput).
5. Update the relevant task configs + this skill's table if load-bearing.
6. Re-run `uv run python -m pytest tests/unit_tests/ -q`.

## Known config bugs/smells (open 2026-07-10)

- `cfg.control.control_type` read at `legged_robot.py:588`, defined nowhere.
- `FixedRobotCfgPPO` defines `batch_size` twice (`fixed_robot_config.py:171,175`
  — the second wins).
- `mini_cheetah_ref_config.py:52` `flip_visual_attachments  # deprecated?` —
  inert under MuJoCo either way.

## When NOT to use this skill

- End-to-end new robot/task → `q2-task-authoring` (it sequences this skill).
- Reward math and obs scaling semantics → `legged-rl-reference`.
- Field is set correctly but sim misbehaves → `q2-debugging-playbook`.

## Provenance and maintenance

Verified 2026-07-10 against `port` @ `bc2bd96`. Re-verify:

```bash
grep -n "cfg\." gym/envs/base/mujoco_backend_base.py | grep -v "self\." | head -30   # consumed fields
grep -n "randomize_friction\|randomize_base_mass" gym/envs/*/*config*.py
grep -rn "mjspec_attributes" gym/envs/base/mujoco_backend_base.py gym/envs/mini_cheetah/mini_cheetah_config.py
grep -n "njmax" gym/envs/base/mujoco_warp_backend.py    # put_data forwarding still present?
sed -n "159,178p" gym/utils/task_registry.py                                          # frequency derivation
```
