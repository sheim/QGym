# Offline replay and fixed-policy height Q

This pipeline records transitions during normal on-policy training, relabels them
with an observation-derived height reward, fits a fixed-policy Q by fitted Q
evaluation (FQE), and validates that Q against fresh simulator continuations.

It is **BRG-owned code** under `learning/offline/` and `learning/algorithms/fqe.py`.
It does not import or modify the CORL submodule and does not use CORL's benchmark
CLI. CORL remains a reference for single-file offline algorithms; the supported
runtime here is plain PyTorch inside BRG's uv environment.

## Scope

- One reward channel: height relative to a fixed reference and bound.
- One frozen stochastic PPO checkpoint as the continuation policy.
- Recording currently requires one policy step per control step
  (`n_policy_steps == 1`, the `go2trot` configuration). Other actor/control
  frequency ratios need a separate interval contract before recording.
- Discounted continuing return; true termination is an absorbing failure.
- Q inputs are the policy-facing actor observation and the candidate action.
  Ground-truth height is used for labels and diagnostics, never as a Q input.
- Early, late, and mixed behavior data can be compared against the same target
  policy.
- MPPI, intervention execution, other reward channels, bound-conditioned Q, and
  finite-horizon Q are not implemented here.

## Record training transitions

Recording is opt-in. It keeps a bounded host-memory ring and writes immutable
snapshots. It does not replace PPO's `DictStorage` or change the optimizer path.

```bash
uv run --frozen scripts/train.py --task=go2trot --device=cpu --headless \
  --num_envs=1024 --max_iterations=600 --run_name=trot-replay \
  --replay_record --replay_capacity=1000000 \
  --replay_snapshot_interval=50 --replay_seed=0
```

Snapshots are written below `logs/<experiment>/<run>/replay/` unless
`--replay_output_dir` is supplied. Each snapshot is a directory containing
`data.pt` and `manifest.json`:

| field | meaning |
|---|---|
| `actor_obs` | policy-facing observation used to choose the action |
| `action` | sampled policy action in BRG policy units |
| `next_actor_obs` | successor observation; pre-reset and recording-noise-sampled for ended rows |
| `successor_height_m` | clean post-action, pre-reset height in metres |
| `terminated`, `timed_out` | independent episode flags |
| `episode_id`, `step_id`, `env_id` | provenance and episode grouping |
| `behavior_iteration` | PPO iteration that produced the transition |

`--replay_snapshot_interval=0` snapshots only at the end of training. A positive
interval retains early snapshots as well as the last-N ring contents. Snapshots
are never deleted automatically; disk retention is an explicit user decision.
A last-N ring does not preserve early data after the ring wraps.

`manifest.json` records observation/action names, dimensions, scales, noise,
actor config, environment config fingerprint, source revision, retention
capacity, valid count, task/backend/device, seed, and run identity. Combining
snapshots rejects incompatible actor contracts, environment fingerprints, or
source revisions.

The recorder copies the pre-reset clean observation and height before the
runner's reset. It samples recording-only successor noise from a dedicated
generator, so training RNG draws are unchanged. When observation-normalization
burn-in resets the environments, the first post-burn-in transition is excluded
from the replay because its action-selection observation is stale. PPO's
existing first transition is left unchanged.

For `go2trot` the actor-observation replay record is about 470 bytes per
transition (109 float32 observation/action/height fields, two bool flags, and
four int64 provenance fields), so 1M transitions are about 470 MB of snapshot
data. Critic observations are not stored; add them only if a later asymmetric
critic requires them.

## Relabel, select, and fit Q

The reward is:

\[
r'_t =
\begin{cases}
0 & |h_t - h_{\mathrm{ref}}| \le b_h,\\
-1 & \text{otherwise},
\end{cases}
\]

where `h_t` is `successor_height_m` measured after the action and before any
reset. Height exactly on the bound is inside. Change the bound and reference
only by explicit CLI arguments; do not rely on hidden defaults.

```bash
uv run --frozen scripts/train_height_q.py \
  --snapshot logs/go2trot/<run>/replay/replay_iter_0000050 \
  --checkpoint logs/go2trot/<run>/model_50.pt \
  --output /tmp/opencode/height-q \
  --height_reference 0.3 --height_bound 0.05 \
  --gamma 0.99 \
  --subset_group episode \
  --fractions 0.1 0.5 1.0 \
  --updates 100000 --batch_size 256 \
  --device cuda:0 --seed 0
```

`--subset_group` is required. `episode` preserves complete trajectory groups;
`transition` is a transition-level shuffle. Fractions are strictly increasing
and must end at `1.0`, so the three training sets are nested.

For early/late/mixed behavior-data comparisons against the same frozen target
policy:

```bash
uv run --frozen scripts/train_height_q.py \
  --early_snapshot <early-snapshot> --late_snapshot <late-snapshot> \
  --checkpoint <fixed-target-checkpoint> \
  --output /tmp/opencode/height-q-comparison \
  --height_reference 0.3 --height_bound 0.05 --gamma 0.99 \
  --subset_group episode --fractions 0.1 0.5 1.0 \
  --balance_per_snapshot 1000000 --updates 100000 --device cuda:0
```

The early and late snapshots are truncated to the same transition count before
mixing. The mixed set uses equal early/late contributions and is deduplicated by
`(run_id, env_id, episode_id, step_id)`. All three critics use the same target
checkpoint, reward, discount, and evaluation distribution. Changing the target
checkpoint changes what Q means.

FQE fits Q only; it does not optimize the actor. The target is:

\[
y_t = r'_t + \gamma (1-\mathbb{1}[\text{terminated}_t])\,
\mathbb{E}_{a'\sim\pi_{\text{frozen}}(\cdot\mid o_{t+1})}
[Q_{\text{target}}(o_{t+1},a')]
\]

plus the absorbing-failure value for true termination:

\[
y_t = r'_t + \gamma \frac{r'_{\text{outside}}}{1-\gamma}
\quad\text{when terminated}.
\]

A timeout alone bootstraps from the recorded pre-reset successor. If both flags
are true, terminal takes precedence. Observation normalization is fit on each
training subset and stored in the artifact; the frozen actor's own normalization
is separate.

Each artifact directory contains `q.pt` and `manifest.json`. The manifest stores
network shape, preprocessing, reward/discount/failure semantics, policy
checkpoint identity and hash, dataset/subset identity, training seed, source
revision, and a null `validation_report_reference`. Evaluation reports are
separate JSON files and do not mutate the artifact.

## Validate against fresh rollouts

Evaluation uses independent simulator initializations. For each trial seed and
candidate first action, a fresh environment is created with the same spawn seed,
the candidate action is applied, and the frozen stochastic policy controls the
rest of the continuation. If reproducible initialization produces different
initial observations, evaluation fails rather than silently comparing different
states.

```bash
uv run --frozen scripts/evaluate_height_q.py \
  --artifact /tmp/opencode/height-q/height_q/data/100pct \
  --experiment_name go2trot --load_run <run> --checkpoint <same-checkpoint> \
  --device cpu --trials 20 --repetitions 3 --horizon 100 \
  --thresholds -1.0 -0.5 0.0 \
  --candidate_actions candidates.pt \
  --output /tmp/opencode/height-q-eval.json
```

`candidates.pt`/`.npy` has shape `[num_candidates, action_dim]` and uses BRG
policy action units. The checkpoint hash must match the Q artifact, and the
current environment configuration fingerprint must match the recorded dataset
unless the artifact predates fingerprinting. Evaluation fails instead of
silently validating a different task configuration.

For large sweeps, use the batched evaluator. It creates one 1024-world
environment per candidate batch, reproducing the same trial states across
candidate batches and repetitions:

```bash
uv run --frozen python scripts/evaluate_height_q_batched.py \
  --artifact /tmp/opencode/height-q/height_q/data/100pct \
  --load_run <run> --checkpoint <same-checkpoint> \
  --trials 1024 --repetitions 4 --horizon 500 --num_candidates 8 \
  --seed 10000 --output /tmp/opencode/height-q-batched.json
```

It defaults to 1024 trials, 4 repetitions, 8 local candidates, H=500. The
paired-state check tolerates only floating-point-level differences between
fresh batched environments. On the 1024-trial batch, the original 1M critic had
MAE 2.534, bias −0.349, ranking 0.503, Q range 0.387, and return range 2.807;
the eval-distribution critic had MAE 2.972, bias +0.963, ranking 0.495, Q range
0.618, and return range 2.771. The earlier 5-trial eval-distribution advantage
did not hold.

### Insertion step

**Historical evaluator limitation:** the results below do not establish
fixed-state variance or ranking precision. Prefix states changed per repetition,
candidates were reset-centered, predictions were overwritten, prefix survival
was lost, and task restore omitted controller history/domain state while clamping
qpos. Prior noise-ceiling and first-reward-independence claims are superseded.
Use raw binary rewards for rates and keep repetition pairing intact.

The corrected experimental entry point is
`scripts/evaluate_height_q_variance.py --artifacts ORIGINAL EVALDIST
--checkpoint CHECKPOINT --load_run RUN --output NEW_DIR --batches 1`.
After its restoration gates and benchmark pass, resume using the same arguments
with `--resume --batches 8`; saved batch0 is retained. It freezes 128 surviving
k=100 states, eight local insertion-centered candidates (candidate0 is nominal),
and scores both critics on the same 1024-world continuations. Commands, pushes,
policy and observation noise share draws within each state; continuation seeds
are independent. The experiment preserves the snapshot's task flags, counters,
controller history, gains, model mass/inertia/friction and native integration
inputs, bypassing the clamping reset API. One-step numerical fidelity is gated;
this is not a claim of bitwise long-horizon trajectory identity.

`insertion.pt` contains snapshot inventories, fixed noisy observations, candidates,
prefix survivor mask, selected source IDs and direct predictions. `batch_N.pt`
retains 16 individual returns, first rewards, survival and seeds;
`raw_returns.pt` explicitly uses `[batch, repeat, state, candidate]` axes.
`statistics.pt` retains per-state/action batch estimator variance, action spread,
paired candidate-minus-nominal variance, candidate covariance and batch-ranking
agreement. Surviving H=500 returns omit a tail in `[-0.6570483,0]`; terminated
returns include their exact absorbing tail. Rankings against Monte Carlo means
remain comparisons against noisy estimates, not against known true Q values.

Run `uv run --frozen python scripts/summarize_height_q_variance.py OUTPUT_DIR`
after all eight batches to audit seed uniqueness, tensor axes, binary reward
values, bounded returns and the covariance identity. It writes
`detailed_report.json`, `per_state_action.csv`, and raw lower/upper tail bounds.
Only eight independent batches estimate the variance of a 16-return estimator;
that estimate has seven degrees of freedom and can itself be imprecise.

The corrected 2026-11-10 experiment completed all eight batches in 459.10 s,
retaining its 56.82 s benchmark batch. Mean fixed-action batch estimator variance
was 1.2042, within-batch action variance 0.7671, and paired-contrast estimator
variance 1.2091. Batch-to-batch pair-ranking agreement was only 0.51798.
Original/eval-distribution critic ranking against the pooled noisy estimates was
0.51618/0.50781, with MAE 1.6904/2.1407. This does not establish a noise ceiling
or isolate the critic as the sole limiting factor. Full raw data, per-state/action
CSV, exact commands and fidelity caveats are under
`/tmp/opencode/height-q-fixed-variance-20261110-v1/REPORT.md`.

For more independent 64-return batches at the same total retained rollout count,
`scripts/extend_height_q_variance.py --source ORIGINAL_DIR --output NEW_DIR
--checkpoint CHECKPOINT --load_run RUN --blocks 1` selects a reproducible random
16-state subset (default selection seed 2026111064), without consulting returns.
The original 128 continuations supply the first two 64-return batches. Eight
parallel seed lanes reuse exactly the saved physical states/observations/candidates;
all native model constants, history and DR fields follow the same index map.
After native gates and the first-block runtime gate, `--blocks 14 --resume`
collects the remaining 896 continuations per state/action, retaining the benchmark.
New RNG identities are (call seed, lane), with distinct vectorized subsequences
across lanes and candidate-shared draws within each state/lane. There is no prefix
rerun. Output axes are `[batch64,repeat64,state16,candidate8]`, shape [16,64,16,8].

`scripts/summarize_height_q_extension.py NEW_DIR` compares those estimates to
the same returns regrouped [64,16,16,8], with whole-64-batch cluster bootstrap
intervals. Fifteen variance df improve replication precision but only sixteen
states remain: do not equate this subset comparison to the earlier 128-state
pooled summary. Numerical restore and continuing-tail caveats still apply.
This protocol does not select actions with Q or evaluate a top-Q policy.

The 2026-11-10 extension collected 114688 new continuations in 399.99 s. On the
matched sixteen-state dataset, mean fixed-action estimator variance decreased
from 1.271764 at16 to .313614 at64 (ratio .246597; approximate batch-bootstrap
95% interval [.200368,.269332]). Pair-order repeatability remained weak:
51.08% at16 versus 52.78% at64. Mean action-spread variance at64 was .183222,
paired-contrast estimator variance .365503. These are conditional subset results;
more batches do not resolve state coverage or the surviving [-.6570483,0] tail.
Full report: `/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/REPORT.md`.

`--insert_step k` (default 0) moves where the candidate actions are inserted.
At `k = 0` the Q is scored at the reset state, which is the original protocol.
At `k > 0` the frozen policy runs for `k` steps first, then the candidate actions
are applied and the H-step return is measured from that mid-gait state:

```bash
uv run --frozen python scripts/evaluate_height_q_batched.py \
  --artifact /tmp/opencode/height-q/height_q/data/100pct \
  --load_run <run> --checkpoint <same-checkpoint> \
  --trials 1024 --repetitions 16 --horizon 500 --num_candidates 8 \
  --insert_step 100 --seed 10000 --output /tmp/opencode/height-q-midgait.json
```

The first candidate of each repetition runs the prefix and snapshots the state via
`Go2Trot.capture_state()`; every other candidate restores that snapshot with
`Go2Trot.restore_state()`. Re-running the prefix is *not* equivalent: MuJoCo Warp
solver noise is amplified by the closed policy loop, so two replays of a 100-step
prefix diverge by ~2.5 cm in height. The script raises rather than scoring
unpaired states. Worlds that terminate during the prefix have no insertion state
and are excluded. The report records `insertion.height_mean`,
`insertion.inside_band_fraction`, and `insertion.excluded_trials_per_repetition`
so the insertion point can be checked rather than assumed. Prefix rewards are
not part of the measured return.

Restoring clamps joint positions to the asset limits (the backend's `reset_state`
contract) and does not restore contact warm start, so the insertion state is
self-consistent and in-distribution but not the uninterrupted trajectory. Do not
compare absolute returns between `insert_step` values; compare ranking and error
against each run's own measured return.

This matters because the reset pose sits above the reward band: at `k = 0` the
first reward is `-1` for every candidate, and the robot needs two to three steps
to reach `0.32 ± 0.05 m`. Scoring at the reset state therefore cannot
discriminate candidates at the insertion step, and it queries the Q in a region
that is only ~1.2% of the training data.

The batched report also includes one-step and two-step lookahead scores:
`r'(s,a) + gamma * Q(s', pi_mean(s'))` and
`r'(s,a) + gamma * r'(s') + gamma^2 * Q(s'', pi_mean(s''))`, with the exact
absorbing-failure tail on immediate termination. On the same 1024 trials:

| Critic | Score | MAE | Ranking |
|---|---|---:|---:|
| Original | direct Q(s,a) | 2.534 | 0.5040 |
| Original | one-step | 2.524 | 0.5083 |
| Original | two-step | 2.495 | 0.5337 |
| Eval-distribution | direct Q(s,a) | 2.975 | 0.4950 |
| Eval-distribution | one-step | 2.840 | 0.5060 |
| Eval-distribution | two-step | 2.779 | 0.4983 |

Two-step lookahead is the best tested score for the original critic, but 0.534
is still weak; the eval-distribution critic improves calibration without
recovering ranking. Lookahead is a diagnostic, not a fix for candidate
discrimination.

For each `(trial, candidate)` the report includes Q prediction, repeated-rollout
mean return, sampling variance, and a return interval. A surviving finite
rollout omits a continuing tail bounded by
`gamma**H * [min_reward, max_reward] / (1-gamma)`; a true failure appends the
exact absorbing tail. Threshold reports count false acceptance, false rejection,
and ambiguous cases. Pairwise candidate-ranking accuracy is reported per trial.
Aggregate Q error alone is not sufficient evidence for a future threshold filter;
near-threshold and candidate-action accuracy matter more.

## Intervened playback

`scripts/play_height_q.py` loads the artifact's frozen actor and applies one
per-decision intervention on MuJoCo CPU with exactly one environment:

- `maxq`: eight stochastic policy draws, execute the highest-Q draw;
- `rejection --threshold c`: eight stochastic draws in fixed order, execute the
  first draw with `Q >= c`, otherwise execute the highest-Q draw.

The exact policy mean is never a candidate in either mode. Rejection delegates
to `sample_rejection_candidates` and `select_rejection_action`; max-Q delegates
to `sample_candidates` and `select_action`. Do not substitute the deterministic
mean-plus-eight closest-mean controller (`rollout_threshold`) here.

```bash
uv run --frozen python scripts/play_height_q.py \
  --task go2trot \
  --artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --mode rejection --threshold -1.776 --device cpu --auto_reset off
```

Startup and every reset end at zero commands. The task constructor runs its
normal reset before the first policy action, then playback zeroes commands.
`R` calls the normal `env.reset()` and zeroes commands. Auto-reset is off by
default; `termination`, `timeout`, and `both` call `_reset_idx` on the selected
mask and then zero commands. No direct backend state writes or bypassed task
transforms are used. Commands are held until a key event: playback disables
task command resampling with an instance-local override and leaves the shared
task/teleop defaults unchanged. The default episode horizon is the task's 5 s,
not `play.py`'s 50 s. Push disturbances, observation noise, and the camera come
from the task config; `--no-push_robots` disables pushes only for playback.

Before simulator construction, playback requires:

- the checkpoint file hash to equal `policy_spec.checkpoint_sha256`;
- the recorded dataset `config_fingerprint` to equal the current task config
  after `convert_frequencies_to_params` and before playback overrides;
- matching observation/action names and dimensions;
- matching actor noise, task scaling, control dt, and `normalize_obs == false`;
- `n_policy_steps == 1`, standard stochastic actor, and `brg_policy_units`;
- no privileged Q inputs such as `base_height`, `base_lin_vel`, contacts, or
  root/body state.

Failures are explicit; there is no fallback artifact, threshold, or backend.
After construction, the live task tensor widths are checked again. Startup and
final summaries report the selected mode/threshold, fingerprint prefix, seed,
push state, checkpoint, resets, fallbacks, and final commands.

`scripts/play_height_q.py --headless --steps N` is a finite wiring smoke. It
does not establish Q accuracy or policy quality, and it cannot validate the
interactive viewer, keyboard, or thread behavior. Those require manual CPU
viewer validation.

## Evidence boundaries

- The Q uses deployable actor observations. It can be biased when hidden physical
  state (including height, unobserved dynamics, and randomized parameters) is
  not identifiable from those observations. More data does not guarantee removal
  of that ambiguity.
- Training data comes from earlier PPO policies; only the continuation target is
  fixed. Coverage of states and actions is finite. Candidate actions outside the
  tested domain are not validated.
- The absorbing failure convention intentionally changes future reward to -1
  after a fall. The artifact estimates height compliance with a failure penalty,
  not purely observed-height violations.
- Evaluation validates a candidate first action followed by the frozen policy.
  It does not validate repeated MPPI/filter intervention or provide a safety
  guarantee.
- The portable tests check schema, reward labels, FQE targets, artifact
  round-trip, and rollout-tail math. They do not establish robot Q accuracy or
  policy quality. Record the actual simulator experiment separately.

## Smoke record

On 2026-11-10, a one-iteration CPU `go2trot` run with 32 environments and a
128-transition end snapshot produced a loadable replay. Training three nested
subsets produced loadable Q artifacts. A one-trial, three-candidate, five-step
CPU evaluation produced a JSON report. These results verify the software path
only; they are not a Q-accuracy or policy-quality result.

On 2026-11-11, `scripts/play_height_q.py` ran the original fair-100 artifact and
`model_600` on MuJoCo CPU for five headless max-Q steps (`steps=5`, `resets=0`,
zero final command) and five rejection steps (`threshold=-1.776`, `fallbacks=2`).
An `--original_cfg` run matched the saved run config, and a 260-step
`--auto_reset both` run reset once at the 250-step/5 s timeout. These are wiring
and identity smokes, not Q-accuracy, policy-quality, or viewer-validation
evidence; manual CPU viewer validation remains outstanding.
