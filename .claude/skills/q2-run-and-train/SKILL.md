---
name: q2-run-and-train
description: How to run, watch, and play back training in Q2 — train_mujoco.py/play_mujoco.py command anatomy, task list, log directory and checkpoint conventions, wandb setup, keyboard teleop, sweeps. Load when asked to train a policy, resume/evaluate a run, find checkpoints or logs, hook up wandb, or drive the robot interactively. NOT for environment install (q2-build-and-env), config field semantics (q2-config-system), or interpreting bad training results (q2-debugging-playbook).
---

# Q2 Run & Train

## Train

```bash
# CPU, with viewer
uv run scripts/train_mujoco.py --task pendulum --device cpu --num_envs 256
# CPU headless, no wandb
uv run scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64 --headless --disable_wandb
# GPU (Linux + CUDA; first run JIT-compiles warp kernels — be patient)
uv run scripts/train_mujoco.py --task mini_cheetah --device cuda:0 --num_envs 4096 --headless
# vsim backend (licensed engine; needs `uv sync --extra vsim` + .env.vsim):
uv run --env-file .env.vsim scripts/train_mujoco.py --task mini_cheetah --backend vsim --device cuda:0 --num_envs 4096 --headless
```

`--backend {mujoco,vsim}` selects the engine (default mujoco). vsim is
CUDA-only, needs the process started with `.env.vsim` (LD_LIBRARY_PATH must
exist before the loader runs), and delivered **~317k steps/s on mini_cheetah
@ 4096 envs vs warp's ~29.3k — 10.8×** (2026-07-12, RTX 4080). Keyboard
teleop is MuJoCo-viewer-only (`--no-keyboard` with vsim).

⚠️ Before trusting GPU training results, read the warp `root_states` staleness
warning in `q2-phase4-parity-campaign` (open bug as of 2026-07-10).

Full CLI (`scripts/train_mujoco.py`, verified): `--task` (required), `--device`
(default `cpu`), `--num_envs`, `--max_iterations`, `--seed`, `--batch_size`,
`--headless`, `--disable_wandb`, `--wandb_project`, `--wandb_entity`.
Unset `--seed` draws a random one (0–10000) and stores it in both cfgs.

Tasks: `gym/envs/__init__.py` `task_dict` lists 11, but **10 actually register
in a MuJoCo-only env** (verified 2026-07-10): `pendulum`, `sac_pendulum`,
`psd_pendulum`, `cartpole`, `mini_cheetah`, `mini_cheetah_ref`,
`mini_cheetah_osc`, `sac_mini_cheetah`, `humanoid`, `humanoid_running`.
`lander` silently drops out — its module imports isaacgym, and registration
imports are individually try/except-guarded, so failing tasks vanish instead
of erroring. (README_MUJOCO.md lists only the 7 mainline ones.) Only pendulum
and mini_cheetah/mini_cheetah_ref have logged MuJoCo training runs so far;
humanoid on MuJoCo is unproven as of 2026-07-10.

What the script does (in order): registers tasks → applies CLI overrides →
`convert_frequencies_to_params` (frequencies → decimation/dts) →
`set_log_dir_name` → seeds → wandb setup → `make_env_mujoco` (backend selected by
device) → `randomize_episode_counters` (desynchronizes resets across envs) →
runner → snapshots source into the log dir → `runner.learn()`.

## Where output lands

```
logs/<experiment_name>/<MonDD_HH-MM-SS_><run_name>/
    model_0.pt, model_<save_interval>.pt, …, model_<final>.pt
    files/gym/**.py, files/learning/**.py     ← source snapshot (.py/.json only)
logs/wandb/run-<timestamp>-<id>/              ← wandb offline copies, incl. files/output.log
```

- `experiment_name`/`run_name` come from `train_cfg.runner`; dir name pattern is
  set in `task_registry.set_log_dir_name` (`gym/utils/task_registry.py:139-152`).
- Checkpoints are `model_<iteration>.pt`, saved every
  `train_cfg.runner.save_interval` iterations plus at the end. On-policy
  checkpoint dict: `actor_state_dict`, `critic_state_dict`, both optimizer
  states, `iter` (`learning/runners/on_policy_runner.py:212-222`).
- A run directory containing only `model_0.pt` means training died/was killed in
  the first iterations — check `logs/wandb/run-*/files/output.log` for the
  traceback.
- Everything under `logs/` is gitignored.

## Weights & Biases

- Default is ON but silently disabled unless `user/wandb_config.json` exists
  (copy `user/wandb_config_default.json`, fill `entity` + `project`) or
  `--wandb_entity`/`--wandb_project` are passed
  (`gym/utils/logging_and_saving/wandb_singleton.py:19-49`).
- `--disable_wandb` forces off. The run name in wandb = log dir basename.

## Console output

Per-iteration block printed by `learning/utils/logger/Logger.py`. Key readings:
- **Mean rewards are `nan` until the first episodes complete** — expected, not a
  bug (episode-windowed averages, window = 100 episodes).
- `steps/s` = `step_counter * num_envs / iteration_counter / collection_time`.
  Historical reference points (dated, RTX 4080 / 2026-04): pendulum ~16,600
  steps/s CPU @ 256 envs; ~255,000 steps/s GPU @ 4096 envs.
- `num_steps_per_env` is derived, not configured:
  `max(1, batch_size // num_envs)` (`on_policy_runner.py`, commit `936a4cc`) —
  changing `--num_envs` automatically rescales rollout length to hold batch size.

## Play back a trained policy

```bash
uv run scripts/play_mujoco.py --task mini_cheetah                      # latest run, latest checkpoint
uv run scripts/play_mujoco.py --task mini_cheetah --load_run May08_12-34-56_ --checkpoint 1500
uv run scripts/play_mujoco.py --task pendulum --no-keyboard --headless
```

- Resolution: `--load_run` defaults to the newest dir under
  `logs/<experiment_name>/` (by mtime); `--checkpoint -1` picks the
  highest-numbered `model_*.pt` (`gym/utils/helpers.py:132-168`).
- Play-time overrides applied automatically: `episode_length_s = 50`, command
  resampling off, `push_robots` off, `reset_to_range` init
  (`scripts/play_mujoco.py:61-69`).
- **Keyboard teleop is ON by default** (`--no-keyboard` to disable), CPU-backend
  passive viewer only. Bindings
  (`gym/utils/interfaces/MujocoKeyboardInterface.py`): Up/Down = vel_x (max
  +4.0/−1.0), `,`/`.` = strafe ±1.0, Left/Right = yaw ±2.0, R = reset all envs,
  Esc/close = quit. Steps of 1/5 max; commands seeded with vel_x = 1.0. A
  `CommandVisualizer` overlay draws the commanded vs actual velocity.
- `scripts/play_pendulum.py` renders pendulum diagnostics (phase portrait,
  torque, KE/PE/total energy panels).

## Sweeps and legacy scripts

- `scripts/sweep.py` + `scripts/sweep_configs/*.json` = wandb sweep harness —
  **IsaacGym-era; unverified under MuJoCo** (uses the legacy arg path). Treat as
  a porting task, not a tool, until proven.
- `scripts/train.py`, `scripts/play.py`, `scripts/export_policy.py` are
  IsaacGym-only. There is currently **no policy-export path for MuJoCo-trained
  policies** (open gap; `export_network` exists in
  `learning/modules/utils/neural_net.py` and exports TorchScript + ONNX, but no
  MuJoCo-side script wires it up).

## When NOT to use this skill

- Install/venv problems → `q2-build-and-env`.
- What a config field means / how to change one → `q2-config-system`.
- Run trains but behaves wrongly → `q2-debugging-playbook`, then
  `q2-phase4-parity-campaign` if on GPU.

## Provenance and maintenance

Verified 2026-07-10 against `port` @ `bc2bd96`. Re-verify:

```bash
uv run python - <<'EOF'
import ast, pathlib
tree = ast.parse(pathlib.Path("scripts/train_mujoco.py").read_text())
print([n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
       and getattr(n.func, "attr", "") == "add_argument"])
EOF
uv run python -c "import gym.envs; from gym.utils.task_registry import task_registry; print(sorted(task_registry.task_classes.keys()))"
grep -n "task_dict = {" -A 30 gym/envs/__init__.py        # task list drift
grep -n "save_interval\|model_" learning/runners/on_policy_runner.py | head
```
