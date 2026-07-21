---
name: q2-task-authoring
description: End-to-end runbook for adding a new robot or task to Q2 — URDF requirements (limit tags, mesh formats, fusing), config classes, task class with rewards/observations, registration dicts, packaging, smoke tests, and the definition of done. Load when asked to add a robot, create a task variant, or port an environment. Sequences q2-config-system and legged-rl-reference; for base-class changes use q2-architecture-contract instead.
---

# Q2 Task Authoring

Pick the base: **FixedRobot** (base bolted to the world; contacts disabled
entirely) or **LeggedRobot** (free joint + ground plane + contact machinery).
Copy the nearest existing task: `pendulum` (fixed), `mini_cheetah` (legged),
`mini_cheetah_ref` (legged + reference trajectories).

## 1. Asset (`resources/robots/<name>/urdf/<name>.urdf`)

Requirements learned from real failures:
- Every actuated joint MUST carry `<limit effort="..." velocity="..."
  lower="..." upper="..."/>` — MuJoCo's importer drops effort/velocity, Q2
  re-parses them from the URDF text (`2bc709b`); missing tags → silent 1e6
  defaults → `test_urdf_limits`-style test won't protect you unless you write
  one for your robot.
- Visual meshes: `.dae`/`.collada` visuals are STRIPPED at load (MuJoCo can't
  decode them); use `.stl`/`.obj` if you want visuals. Collision geometry is
  unaffected. Relative mesh paths resolve relative to the URDF's directory.
- Bodies connected by rigid/fixed joints get FUSED by MuJoCo (mini_cheetah's
  `foot` vanished into `shank`). If you name-match such bodies
  (`foot_name`, contact lists), you need `fusestatic=False` (on jt/port,
  unmerged 2026-07-10) or name the surviving body.
- Keep every file < 100 KB (CI hard gate) — decimate meshes.

## 2. Env config class (`gym/envs/<name>/<name>_config.py`)

Subclass `LeggedRobotCfg` or `FixedRobotCfg`. Minimum to get physics right:

```python
class MyRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = <N>
        episode_length_s = 5
    class control(LeggedRobotCfg.control):
        ctrl_frequency = 100          # Hz — dts/decimation are DERIVED, never set directly
        desired_sim_frequency = 500
    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/<name>/urdf/<name>.urdf"
        fix_base_link = False          # True → FixedRobot semantics
        joint_damping = 0.01
        rotor_inertia = 0.0
        penalize_contacts_on = ["thigh"]        # substring match on body names
        terminate_after_contacts_on = ["base"]
    class init_state(LeggedRobotCfg.init_state): ...   # default joint angles, root pose
    class scaling(LeggedRobotCfg.scaling): ...          # per-obs/action scales
```

Check the actual body names the backend sees before writing contact lists
(verified command; note MuJoCo may have FUSED bodies away — on `port`,
mini_cheetah has no `foot` bodies, only `*_shank`):

```bash
uv run python -c "
from gym.utils.task_registry import task_registry; import gym.envs
env_cfg, train_cfg = task_registry.get_cfgs('mini_cheetah')   # swap in your task
env_cfg.env.num_envs = 2                                       # default is 4096 MjData allocs!
env_cfg.seed = 0
task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
env = task_registry.make_env_mujoco('mini_cheetah', env_cfg, device='cpu', headless=True)
print(env._backend.body_names); print(env._backend.dof_names)"
```

## 3. Runner config class (same file)

Subclass the matching `*RunnerCfg`: set `runner.experiment_name` (log dir
name), `actor.obs` / `critic.obs` (lists of env attribute names — each must
exist on the env and have a `scaling` entry), `actor.actions`,
`algorithm.batch_size`, and `critic.reward.weights` (only nonzero weights run;
see legged-rl-reference for reward-design rules: errors into `_sqrdexp`,
per-joint means, shape `[num_envs]`).

## 4. Task class (`gym/envs/<name>/<name>.py`)

```python
from gym.envs.base.legged_robot import LeggedRobot

class MyRobot(LeggedRobot):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless, backend=None):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless, backend=backend)

    def _reward_my_term(self):        # discovered by name from weights dict
        return self._sqrdexp(<error_expr>)          # shape [num_envs]
```

Keep the legacy positional signature and pass `backend=` through — that's how
`make_env_mujoco` injects the MuJoCo backend. No engine imports in this file,
ever.

## 5. Register (`gym/envs/__init__.py`)

Add entries to ALL THREE dicts + `task_dict`:

```python
class_dict["MyRobot"] = ".<name>.<name>"
config_dict["MyRobotCfg"] = ".<name>.<name>_config"
runner_config_dict["MyRobotRunnerCfg"] = ".<name>.<name>_config"
task_dict["my_robot"] = ["MyRobot", "MyRobotCfg", "MyRobotRunnerCfg"]
```

Trap: imports are individually try/except-guarded — a typo makes your task
silently vanish from the registry instead of erroring. Live proof: `lander` is
in `task_dict` but absent at runtime (its module imports isaacgym). Confirm
registration (verified command):

```bash
uv run python -c "import gym.envs; from gym.utils.task_registry import task_registry; print(sorted(task_registry.task_classes.keys()))"
```

## 6. Packaging

Add `gym.envs.<name>` to `[tool.setuptools] packages` in `pyproject.toml`.
Known debt: pendulum/cartpole are missing there today — don't propagate the
mistake to your task.

## 7. Definition of done (in order)

1. `uv run python -m pytest tests/unit_tests/ -q` still green.
2. Smoke train, CPU, tiny:
   `uv run scripts/train_mujoco.py --task my_robot --device cpu --num_envs 16 --max_iterations 5 --headless --disable_wandb`
   — no crash, rewards become non-nan after first episodes, sane steps/s.
3. Reward shape check: every `_reward_*` returns `[num_envs]` (crashes here are
   usually missing `dim=` in a reduction).
4. A minimal test for your robot: fixture in `tests/unit_tests/conftest.py`
   (copy `legged_cpu_backend`) + at minimum a "doesn't fall through the
   ground / limits parsed" pair. If your robot has termination contacts, copy
   `test_legged_termination.py`'s pattern.
5. Short real training on CPU; inspect with
   `uv run scripts/play_mujoco.py --task my_robot`.
6. GPU: only after reading `q2-phase4-parity-campaign` (warp `root_states`
   staleness makes GPU training untrustworthy until fixed) — and expect to need
   `njmax`/`ccd_iterations` tuning for contact-rich robots.
7. `uv run ruff format . && uv run ruff check .`

## When NOT to use this skill

- Changing FixedRobot/LeggedRobot/backends themselves → `q2-architecture-contract`.
- Only tuning an existing task's rewards → `legged-rl-reference`.
- The new task misbehaves → `q2-debugging-playbook`.

## Provenance and maintenance

Verified 2026-07-10 against `port` @ `bc2bd96`. The step-5 registry-inspection
command is intentionally marked "verify, don't guess". Re-verify drift:

```bash
sed -n "1,50p" gym/envs/__init__.py            # dict names unchanged?
grep -n "packages" pyproject.toml
grep -n "backend=None" gym/envs/mini_cheetah/mini_cheetah.py   # constructor pattern intact?
```
