# go2trot — training

How to train the `go2trot` task (Unitree Go2, clock-based trot) with the
repository's locked environment.

## Prerequisites

`pyproject.toml` + `uv.lock` are the only dependency sources; everything is
installed into the repository-local `.venv`. Run all commands from the repo
root.

```bash
uv python install 3.11        # local interpreter, used by .venv
uv sync --frozen              # build/refresh .venv from uv.lock
uv run --frozen python -c "import torch, mujoco, mujoco_warp; \
print(torch.__version__, torch.cuda.is_available(), mujoco.__version__)"
```

`--frozen` is required: the lockfile also records optional, gitignored local
sources (`thirdparty/vlearn` wheel, Unitree SDK checkout). Never re-resolve for
a MuJoCo-only clone.

The final check must print `True` for CUDA before using `--device cuda:0`.
`mujoco-warp` comes from the default `dev` group; no extra is needed.

## Command

```bash
uv run --frozen scripts/train.py \
  --task=go2trot \
  --backend=mujoco \
  --device=cuda:0 \
  --num_envs=4096 \
  --max_iterations=600 \
  --headless \
  --run_name=gpu-600-trot-<topic>-<date>
```

- `--headless` is mandatory: the MuJoCo Warp backend has no viewer.
- Always set a unique `--run_name`. It is appended to the timestamped log
  directory, so concurrent runs cannot collide.
- The task and reward definitions come from
  `gym/envs/go2/go2trot_config.py`; there is no training-time override for
  physics or domain randomization.

### Flags

| Flag | Default | Effect |
|---|---|---|
| `--task` | required | Task name; `go2trot` resolves env, runner config, and log root. |
| `--backend` | `mujoco` | `mujoco` selects CPU or Warp from `--device`; `vsim` is licensed/CUDA-only. |
| `--device` | `cpu` | `cuda:0` → `MuJocoWarpBackend`; `cpu` → `MuJocoCPUBackend`. |
| `--num_envs` | config `4096` | Parallel environments. |
| `--max_iterations` | config `500` | Total iterations for a fresh run. |
| `--save_interval` | config `50` | Checkpoint period in iterations. |
| `--seed` | config `-1` | `-1` means an unseeded/random run. |
| `--batch_size` | config `2**15` | PPO optimizer minibatch. |
| `--experiment_name` | `go2trot` | Parent folder under `logs/`. |
| `--run_name` | `""` | Label appended to the timestamped run directory. |
| `--headless` | off | Disable the viewer. Required for Warp. |
| `--disable_wandb` | off | W&B is only active when an entity and project are configured. |
| `--resume` | off | Restore model + optimizer state. |
| `--load_run` / `--checkpoint` | latest | Run directory / iteration to load. |
| `--original_cfg` | off | Rebuild configs from the run's saved snapshot. |

### Effective configuration

Derived from `Go2TrotCfg` / `Go2TrotRunnerCfg`; do not duplicate these numbers
in scripts — read the config.

| Setting | Value |
|---|---|
| episode length | 5 s |
| control / sim frequency | 50 Hz / 50 Hz |
| rollout geometry | `rollout_size = 2**16`, so `num_steps_per_env = 65536 / num_envs` (16 at 4096 envs) |
| PPO | `batch_size = 2**15`, `max_gradient_steps = 24`, adaptive LR, `desired_kl = 0.01` |
| startup DR | contact friction, link mass scale `[0.9, 1.2]` |
| episodic DR | `p_gains [0.9, 1.1]`, `d_gains [0.8, 1.2]` |
| gait | oscillator 1–3 Hz, trot phase offsets (FL/RR, FR/RL) |

## Outputs

```
logs/go2trot/<MonDD_HH-MM-SS>_<run_name>/
├── model_0.pt … model_<N>.pt     # every save_interval iterations
├── vitals.jsonl                  # one JSON row per iteration
├── files/                        # config snapshot for --original_cfg
└── progress.png                  # only if you generate one yourself
```

`vitals.jsonl` is the only metrics record when W&B is unconfigured. Reward means
are `nan` until the first episode completes — that is expected, not a failure.

## Reference run

Measured on this machine (RTX 5090 32 GB, driver 580.x, CUDA 13.0,
`torch 2.13.0+cu130`, 4096 envs, 600 iterations, no W&B):

- wall clock ≈ 4.5 min, ≈ 176k steps/s (~39.3M samples);
- 13 checkpoints (`model_0` … `model_600`);
- episode time reached the full 5 s horizon, total reward 8.4 → 13.5.

CPU comparison at 1024 envs: ≈ 5k steps/s, i.e. ~35× slower.

A rising reward is a training signal, not a quality verdict. The acceptance gate
is `scripts/evaluate_go2trot.py`; a reward curve alone does not establish gait
quality, stability, or transfer.

## CPU fallback

Use MuJoCo CPU only for the viewer or backend debugging; it builds one MuJoCo
world per environment, and `go2trot` enables link-mass randomization, so every
env holds a private model copy (~0.9 MB/env beyond a ~1.5 GB baseline).

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
  uv run --frozen scripts/train.py --task=go2trot --device=cpu --headless \
  --num_envs=4096 --max_iterations=300 --run_name=cpu-300-trot-<topic>-<date>
```

Do not use `ulimit -v`: it caps address space, and MuJoCo reserves a large
virtual arena, aborting with `mujoco.FatalError: Could not allocate memory`.
Without `--headless`, each CPU env also keeps ~5 MB of skybox/ground textures
(~20 GB at 4096 envs) and the run is OOM-killed.

## Resume

```bash
uv run --frozen scripts/train.py --task=go2trot --device=cuda:0 --headless \
  --resume --load_run=<run-directory> --checkpoint=600 \
  --max_iterations=1200 --run_name=<new-label>
```

`--max_iterations` is an **additional** budget on resume: the runner trains
`it + max_iterations` iterations, so resuming at 600 with `--max_iterations=600`
stops at 1200. A resumed run writes a new timestamped directory, seeded with
`model_600.pt` and then saving every `save_interval`. Add `--original_cfg` when
the run predates current config changes.

Note: checkpoint filenames continue the global count (`model_600` … `model_1200`),
but `vitals.jsonl` in the new directory starts again at `iteration: 1` and the
first rows show a metrics warm-up (reward ~7.7, episode time ramping from ~0.3 s)
until the first 5 s episode completes. Offset the iteration column by the resumed
count when stitching curves; the warm-up is a logging window, not a regression.

## Playback and evaluation

```bash
# Viewer (CPU only; Warp is headless) — watch the trained policy
uv run --frozen scripts/play.py --task=go2trot --device=cpu --num_envs=1 \
  --load_run=<run-directory> --checkpoint=600

# Fixed-command acceptance gate
uv run --frozen scripts/evaluate_go2trot.py --load_run=<run-directory> \
  --checkpoint=600 --backend=mujoco --device=cuda:0 \
  --output logs/go2trot/perf-600.json --check
```

## W&B

Logging activates only when both an entity and a project are configured, via
`user/wandb_config.json` (copy `user/wandb_config_default.json`) or
`--wandb_entity` / `--wandb_project`. Use `--disable_wandb` for local
discriminators. Never commit credentials.
