---
name: q2-train-and-evaluate
description: Run, resume, inspect, and play back Q2 policies with MuJoCo CPU or MuJoCo Warp, and plan controlled policy validation. Use for training commands, CLI overrides, W&B, log and checkpoint discovery, deterministic inference, or diagnosing whether two runs are comparable. Optional VSim code remains, but its machine-local setup is deferred.
---

# Q2 Train and Evaluate

Read `genAI_skills/AGENTS.md`, `scripts/train.py --help`, the selected task config,
and `scripts/play.py --help`. Use current CLI help rather than a copied flag
list. Recorded evaluation, comparison, benchmark, and profiler commands are
absent; see the [limitations and validation scope](../../../genAI_skills/README_MUJOCO.md#limitations-and-validation-scope).

## Prepare a controlled run

Record before execution:

- commit and dirty-worktree state;
- task, backend, device, seed, environment count, control/sim frequency;
- `algorithm.rollout_size`, optimizer `batch_size`, gradient steps, and iterations;
- config/CLI overrides and whether W&B is enabled;
- the predicted outcome or acceptance threshold.

Do not compare results with unequal temporal rollout horizons unless rollout
horizon is the independent variable.

## Train

Start with a CPU smoke run:

```bash
uv run --frozen scripts/train.py --task TASK --backend mujoco --device cpu --num_envs 8 --max_iterations 2 --headless --disable_wandb
```

Use MuJoCo Warp only after CPU/unit correctness:

```bash
uv run --frozen scripts/train.py --task TASK --backend mujoco --device cuda:0 --headless
```

The first Warp run may compile kernels. Treat constraint-capacity warnings as
physics failures, not log noise.

VSim backend code and tests remain, but machine-local installation and the
test launcher are deferred. Consult `thirdparty/README.md` and the validation
scope linked above before planning a licensed run.

Use `--disable_wandb` for local discriminators. Otherwise configure credentials
through local W&B authentication. Set project/entity in `user/wandb_config.json`
using `user/wandb_config_default.json`, or pass `--wandb_project` and
`--wandb_entity`. Do not commit credentials. Ordinary W&B logging remains
supported; orphaned sweep configurations were retired.

## Resume and locate artifacts

- Runs land below `logs/<experiment_name>/<timestamp>_<run_name>/` with
  `model_<iteration>.pt`, source snapshots, and metrics.
- Resolve a checkpoint through the same helpers the script uses; do not assume
  lexicographic order equals the highest iteration.
- Resume with explicit experiment/run/checkpoint inputs when reproducibility
  matters. Confirm whether `--max_iterations` means a total or additional
  budget in the current runner/script before scheduling a long run.
- Loading for inference should restore model and normalization state, switch
  to evaluation mode, and avoid optimizer loading unless resuming training.

## Play and evaluate

- Use `scripts/play.py` for interactive playback. MuJoCo Warp is
  headless; use MuJoCo CPU for its passive viewer. VSim's viewer/keyboard code
  remains, with local execution deferred as described above.
- No maintained recorded-evaluation CLI, policy report, benchmark, or profiler
  is present. When a task calls for new measurements, define a focused protocol
  and its implementation within that task's scope. Keep commands, reset
  distribution, seed, episode duration, and environment count fixed across
  comparisons.
- For a physics question, begin with native contract/physics tests and a focused
  policy-free probe with a predicted invariant. Historical campaign and manual
  fidelity programs are retired; recover them from git only for a specific need.

## Offline replay and height Q

- For fixed mid-gait state variance, use
  `scripts/evaluate_height_q_variance.py`: validate native snapshot fidelity,
  benchmark with `--batches 1`, then retain batch0 using `--resume --batches 8`.
  `scripts/summarize_height_q_variance.py DIRECTORY` audits raw batches and
  writes per-state/action statistics without more simulation. The historical
  batched insertion evaluator does not establish fixed-state variance; see
  `docs/offline_height_q.md` for its superseded evidence and current limits.

- `scripts/train.py --replay_record ...` writes bounded replay snapshots under
  the run's `replay/` directory. Recording is opt-in; snapshots are immutable
  and are never deleted automatically. A last-N ring does not retain early data
  after wrap; use `--replay_snapshot_interval` when early/late comparisons matter.
- `scripts/train_height_q.py` loads snapshots, relabels a fixed height reward,
  forms nested data fractions or early/late/mixed datasets, and fits a
  fixed-policy FQE critic. `--subset_group`, height reference/bound, and gamma
  are required. It does not optimize the policy.
- `scripts/evaluate_height_q.py` compares Q against fresh simulator
  continuations from reproducible initial states, with candidate first actions
  and repeated stochastic continuations. Threshold reports include false
  acceptance/rejection and ambiguity; ranking is reported separately from
  aggregate error.
- `scripts/play_height_q.py` is the single-environment MuJoCo CPU playback path
  for a frozen actor plus height Q. `--mode maxq` executes the best of eight
  stochastic draws; `--mode rejection --threshold c` executes the first draw
  with `Q >= c`, falling back to max-Q. The mean is not a candidate. Playback
  validates the checkpoint hash, config fingerprint, actor/noise/scale contract,
  and unprivileged Q inputs before building the simulator. It starts and resets
  at zero commands, holds keyboard commands (no task resampling), and leaves
  auto-reset off by default. A `--headless --steps N` run is a wiring smoke, not
  Q/policy evidence; viewer/keyboard behavior needs manual CPU validation.
- See `docs/offline_height_q.md` for snapshot/artifact schemas, terminal versus
  timeout semantics, absorbing-failure returns, and tail uncertainty. These
  artifacts use actor observations only and do not import CORL.

## Interpret results

- Startup reward means may be NaN until episodes complete; distinguish that
  from tensor/optimizer NaNs.
- Report per-term reward, survival/episode duration, observation/action
  statistics, KL/losses, throughput, and task-specific physical metrics.
- Compare CPU and Warp tightly because they share a MuJoCo model; allow only
  predeclared coarse bounds for cross-formulation VSim contact behavior.
- Treat checkpoint selection as part of the experiment. Do not pick a target
  backend checkpoint after viewing its transfer score unless validation-based
  selection was declared in advance.
- Do not claim correctness from a gait video, aggregate reward, or throughput
  alone. Link claims to tests, saved artifacts, and predicted acceptance
  criteria. Update [limitations and validation scope](../../../genAI_skills/README_MUJOCO.md#limitations-and-validation-scope) when consequential
  evidence or supported limitations change.
