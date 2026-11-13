# Offline replay and fixed-policy yaw angular-velocity Q

This is the second concrete offline reward path in `learning/offline/`, next to the
[height path](offline_height_q.md). It relabels the same recorded replay
transitions with go2trot's yaw tracking term, fits a fixed-policy Q by fitted Q
evaluation (FQE), and measures plain versus Q-intervened rollouts from fixed
simulator states.

It is a parallel, minimal path, not a generic reward framework. Only two things
differ from the height pipeline: the reward shape (an `[0, 1]` squared
exponential instead of a binary band) and the terminal convention (the return
ends at a true termination instead of adding an absorbing failure tail). The
shared primitives are `TransitionDataset`, `FQE`, `QNetwork`, `FrozenTrotPolicy`,
`PolicySpec`, `TrainingSpec`, candidate sampling, and `WarpInsertionSnapshot`.

## Reward

`Go2Trot._reward_tracking_ang_vel` computes

\[
r'_t = \exp\left(-\frac{1}{\sigma}\left(\frac{c_t - \omega_t}{2.5}\right)^2\right),
\qquad \sigma = 0.25,
\]

where `c_t` is the yaw command and `ω_t` is the measured base yaw rate. Both the
`tracking_sigma = 0.25` and the hard-coded `2.5 rad/s` error divisor matter: an
implementation with only `sigma` is a different, much sharper reward. Prefer
`cfg.reward_settings.tracking_sigma` for the CLI and treat `2.5` as the task's
fixed `error_scale`. `AngularVelocityRewardSpec` records both.

This is the raw term: no `dt`, no reward weight, no termination penalty. Its
range is `[0, 1]`, so it is **not** interchangeable with the height reward's
`0 / -1` band.

### Terminal convention

The label is exactly zero on a true termination
(`terminal_continuation = "zero_after_terminal"`), the FQE terminal target is the
observed reward with `failure_value == 0.0`, and there is no absorbing tail. A
timeout alone is not a failure: the recorded pre-reset successor bootstraps, as
in the height path.

`accumulate_return_zero_after_terminal` applies `gamma**step * reward` while the
world is alive and marks a terminal world dead; it never adds a tail. A surviving
`H = 500` rollout does truncate a nonnegative tail in `[0, gamma**H/(1-gamma)]`
(about `0.657`), so a finite return is a lower bound on the infinite-horizon
return.

### Command timing

The task resamples commands inside `_post_decimation_step`, and the runner
computes rewards after `env.step()`. The reward that governed a recorded
transition therefore uses the **post-resample** command:

- relabeling reads `commands[:, 2]` and `base_ang_vel[:, 2]` from
  `next_actor_obs`, locating both blocks by the manifest's `obs_names` /
  `obs_dims` (never by hard-coded indices);
- live evaluation reads `env.commands[:, 2]` and `env.base_ang_vel[:, 2]`
  **after** `env.step()`, not the pre-step command clone;
- the pre-step clone is kept only for the interval's tracking-error metric.

The evaluation script asserts `angular_velocity_reward(...)` equals
`env._reward_tracking_ang_vel()` on live states; the measured maximum absolute
error is `0.0`.

### Label noise

`next_actor_obs` is `clean successor + one policy-marginal observation-noise
draw` (recording-only for ended rows). For `go2trot`, `base_ang_vel` noise is
`0.01 / 0.3 = 0.0333 rad/s` and `commands` carry no noise, so relabeled `r'` is
slightly noisier than the live reward. Evaluating 50 steps x 1024 worlds gave
mean `0.9624` clean versus `0.9618` noisy, bias `-0.0006`, RMSE `0.0084`
(relative `0.86%`), max `0.077`, and only `0.07%` of rows differing by more than
`0.05`. That noise is accepted, not corrected: the Q input has its own
independent draw, so the label noise is small but irreducible from recorded data.

## Train the Q

```bash
uv run --frozen python scripts/train_angular_velocity_q.py \
  --snapshot logs/go2trot/<run>/replay/replay_iter_0000600 \
  --checkpoint logs/go2trot/<run>/model_600.pt \
  --output /tmp/opencode/angular-vel-q \
  --sigma 0.25 --gamma 0.99 \
  --subset_group episode --fractions 0.1 1.0 \
  --updates 100000 --batch_size 256 \
  --device cuda:0 --seed 0
```

`--snapshot`, `--checkpoint`, `--output`, `--sigma`, `--gamma`, and
`--subset_group` are required. `--error_scale` defaults to the task's `2.5` and
is recorded in the artifact. Fractions must be strictly increasing and end at
`1.0`, so `--fractions 0.1` alone is rejected by the shared subset contract; use
`0.1 1.0` or `1.0`. The script refuses a `--sigma` that differs from the
snapshot's recorded `env_cfg.reward_settings.tracking_sigma`.

Q inputs are the 48-wide actor observation plus the 12-wide action, and the
network is the same `QNetwork` as the height path. `FQE.compute_target` is
unchanged:

\[
y_t = \begin{cases}
r'_t & \text{terminated},\\
r'_t + \gamma\,\mathbb{E}_{a'\sim\pi_r}[Q_{\text{target}}(o_{t+1},a')] & \text{otherwise},
\end{cases}
\]

because `failure_value = 0.0`.

### Convergence and held-out diagnostics

A minibatch loss is not convergence evidence. With `gamma = 0.99` a
zero-initialized FQE target grows towards the `1/(1-gamma) = 100` scale, so the
displayed MSE can shrink while the held-out Bellman residual is still large.
`--validation_fraction 0.1 --split_seed 0` splits **complete episodes** into
disjoint train/validation sets (`episode_split()`); normalization stays fitted on
the train side and the held-out rows are never optimized on. The per-fraction
report (`<fraction>_report.json` next to the artifact) contains:

- the minibatch loss curve (first/last/mean/min/max, first- and last-1000 means,
  fraction of the last 1000 updates below 1.0);
- the one-step Bellman residual on a random 8192-row train sample and on the
  whole held-out split at update 0 and every `--validation_every` updates:
  MSE, RMSE, MAE, bias, prediction mean, and target mean.

Select a configuration by the final held-out residual, not by the training loss.
The residual is a fit diagnostic for this reward's own bootstrap; it is not an
off-policy value estimate and not evidence about states outside the replay.

Each artifact directory has `q.pt` and `manifest.json` with
`format = brg-angular-vel-q-v1`, `return_convention =
discounted_episodic_zero_after_terminal`, `failure_value = 0.0`, `gamma`,
`reward_spec` (including `tracking_sigma` and `error_scale`), `policy_spec`
(checkpoint and sha256), `training_spec`, dataset identity and
`config_fingerprint`, and a null validation reference. `AngularVelocityQ`
rejects any other format, schema, return convention, or nonzero failure value,
and `HeightQ` rejects this format, so one reward's Q cannot be read as another's.

## Candidate-ranking diagnostic

`scripts/diagnose_angular_velocity_candidate_ranking.py` is the decisive
usefulness test. It restores the insertion snapshot in state-major/candidate-minor
order for all 128 stored states, shares commands within each state's eight
candidate worlds (`install_paired_environment_noise`), applies each stored
candidate as the first action, then lets the frozen policy control `H` steps, and
measures the realized yaw return with the same post-step reward and zero terminal
value. In this stored set candidate 0 is the policy mean and candidates 1–7 are
stochastic draws (the persistent rollout instead samples eight IID draws).

For every critic it reports per-state Spearman rank correlation between
`Q(s, a_i)` and realized `r'_i`, argmax agreement (does max-Q pick the realized
best?), top-1 regret (and regret normalized by the realized range), the mean
Q and realized max-advantage over candidate 0, and the correlation between Q
advantage and realized advantage (across candidate pairs and across state
means). Realized returns are reported with their within-state spread and their
across-seed stability, because a spread near the replay floor means no ranker can
help.

```bash
uv run --frozen python scripts/diagnose_angular_velocity_candidate_ranking.py \
  --source /tmp/opencode/height-q-fixed-variance-20261110-v1 \
  --output /tmp/opencode/angular-vel-q-20261112/qbetter/ranking \
  --checkpoint logs/go2trot/<run>/model_600.pt --load_run <run> \
  --artifacts <old-q> <new-q> --artifact_names old new \
  --horizons 100 500 --seeds 7400000 7400100
```

## Paired fixed-state intervention

`scripts/evaluate_angular_velocity_q_persistent.py` restores the saved insertion
snapshot (128 states x 8 lanes, 1024 Warp worlds), validates lane/replication
fidelity, one-step CPU/GPU agreement, the live-reward parity above, the label
noise, and an identical-controller negative control, then runs controllers:

- `plain`: candidate 0 of eight stochastic draws (`critic = None`);
- `<artifact name>`: max-Q over the same eight draws;
- `rejection_c<c>`: first draw with `Q >= c`, else the max-Q draw.

```bash
uv run --frozen python scripts/evaluate_angular_velocity_q_persistent.py \
  --source /tmp/opencode/height-q-fixed-variance-20261110-v1 \
  --output /tmp/opencode/angular-vel-q-20261112/eval \
  --checkpoint logs/go2trot/<run>/model_600.pt --load_run <run> \
  --artifacts /tmp/opencode/angular-vel-q-20261112/q100k/angular_vel_q/data/100pct \
  --artifact_names angular_q --thresholds 5.0 20.0 \
  --blocks 1 --trace_steps 600 --resume
```

It refuses a gamma other than `0.99`, a `tracking_sigma` other than `0.25`, a
nonzero failure value, a checkpoint-hash mismatch, or a current
`config_fingerprint` that differs from the artifact's. Thresholds must be chosen
on the positive return scale; height thresholds such as `-1.776` are meaningless
here. `--blocks` is capped at 16 (eight calls per block), and a first block whose
projected runtime exceeds 12 minutes per controller writes `runtime_blocker.json`
and raises instead of continuing.

Summarize and plot:

```bash
uv run --frozen python scripts/summarize_angular_velocity_q_persistent.py \
  /tmp/opencode/angular-vel-q-20261112/eval
uv run --frozen python scripts/plot_angular_velocity_tracking.py \
  /tmp/opencode/angular-vel-q-20261112/eval
```

The summarizer reports per-controller mean `r'`, survivor fraction, yaw RMSE,
selection frequencies, the paired mean gain against plain with a state-cluster
bootstrap interval and a paired Student-t test, and wins/ties/losses. The plot
reads `trace_<controller>.pt` files, averages yaw rate across lanes and states,
overlays the post-resample reference, and marks the detected command-resample
boundaries (3.0 s = 150 control steps at `dt = 0.02`). The first traced interval
can be shorter than 3.0 s because the insertion state sits mid-command.

## Measured runs (2026-11-12)

Setup for the first fit: final-1M snapshot of
`Nov10_10-36-22_gpu-final-600-trot-hist-20261110`, frozen `model_600`, gravity
gamma `0.99`, Q trained for 100k updates on the 100% subset, one 64-execution
block per state (1024 worlds), H500.

### First fit (100k updates, 100% of the replay)

| Controller | mean `r'` | survivor | yaw RMSE | paired gain vs plain | state-cluster 95% CI | paired t (df 127) | p | W/T/L |
|---|---:|---:|---:|---:|---|---:|---:|---|
| plain | 96.224 | 0.9812 | 0.2200 | - | - | - | - | - |
| max-Q | 95.534 | 0.9762 | 0.2472 | -0.690 | [-0.779, -0.605] | -15.44 | 6.4e-31 | 4/0/124 |
| rejection `c=5` | 96.210 | 0.9807 | 0.2206 | -0.014 | [-0.028, -0.003] | -2.21 | 0.029 | 59/0/69 |
| rejection `c=20` | 95.815 | 0.9795 | 0.2348 | -0.409 | [-0.488, -0.338] | -10.60 | 3.4e-19 | 7/0/121 |

Re-running the identical command into a second directory reproduces the max-Q
gain (`-0.687`) and the `c=20` gain (`-0.400`) to about `0.01` return units, but
the `c=5` gain changes sign (`+0.006`, p `0.42`). The `c=5` difference is smaller
than the ~`0.02` mean-absolute replay difference between two identical plain
rollouts, so it is **not** a detectable effect; only max-Q and `c=20` are.

### Converged refit and ranking ceiling (option-a follow-up)

Four variants were trained on an episode-grouped 90% split (898,702 rows) with a
101,298-row held-out split:

| Variant | updates | lr | batch | target samples | hidden | last-1k loss | held-out MSE | held-out RMSE |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `v0` (= original recipe, **best**) | 100k | 3e-4 | 256 | 1 | 256,256 | 5.47 | **5.63** | 2.37 |
| `v1` | 300k | 3e-4 | 256 | 1 | 256,256 | 5.05 | 5.75 | 2.40 |
| `v3` | 300k | 3e-4 | 512 | 8 | 512,256 | 5.04 | 6.17 | 2.48 |
| `v2` | 500k | 1e-4 | 512 | 8 | 256,256,256 | 5.11 | 6.20 | 2.49 |

Held-out MSE is the 32-successor-sample Bellman residual on 101,298 episode-disjoint
rows from an independent recomputation; the training reports' own one-sample
target-network values (`5.84 / 5.80 / 6.26 / 6.37`) agree to 0.2.

Held-out MSE is flat from 100k to 300k updates while the training loss keeps
falling, and every stability knob makes it worse. Recomputing the target with 32
successor-action samples changes MSE by `< 0.12` (one-sample target RMS deviation
`0.17–0.31`), so the ~2.4 RMSE residual is the achievable fit, not target
sampling noise. Q advantage over candidate 0 moved only `0.224 → 0.234` (best
held-out) or `0.262` (300k), on a return scale of 96.

On 128 fixed states x 8 stored candidates x 2 seeds:

| Critic | H | Spearman | argmax hit | top-1 regret | adv corr (pairs) |
|---|---|---:|---:|---:|---:|
| old 100k | 100 / 500 | -0.017 / -0.030 | 0.106 / 0.102 | 0.49 / 0.59 | 0.048 / -0.007 |
| best held-out | 100 / 500 | -0.003 / 0.005 | 0.113 / 0.102 | 0.51 / 0.95 | 0.028 / -0.012 |
| 300k | 100 / 500 | -0.026 / -0.016 | 0.129 / 0.109 | 0.46 / 0.51 | 0.053 / 0.032 |

Chance argmax hit is `0.125`. The decisive control is test-retest reliability:
two seeds of the *same* plain controller give a within-state rank Spearman of
`0.007` (H100) and `0.019` (H500) with argmax agreement `0.164`. The realized
within-state spread (1.15 / 2.31) exists, but its ordering does not reproduce, so
the ranker ceiling is ~0 and all critics are at it.

Two-block intervention (128 executions/state; rejections use the best held-out Q
at thresholds `37.0/75.5/76.6`):

| Controller | gain vs plain | 95% CI | p | W/T/L | fallback |
|---|---:|---|---:|---|---:|
| old 100k max-Q | -0.653 | [-0.720, -0.589] | 4e-40 | 3/0/125 | - |
| best held-out max-Q | -0.615 | [-0.683, -0.549] | 6e-37 | 4/0/124 | - |
| 300k max-Q | -0.369 | [-0.424, -0.314] | 2e-25 | 13/0/115 | - |
| rejection `c=37` | -0.445 | [-0.500, -0.391] | 8e-32 | 6/0/122 | 0.466 |
| rejection `c=75.5` | -0.656 | [-0.724, -0.590] | 8e-40 | 1/0/127 | 0.795 |
| rejection `c=76.6` | -0.621 | [-0.689, -0.555] | 4e-37 | 3/0/125 | 0.918 |

`best held-out - old` = `+0.038` (CI `[-0.009, +0.085]`, p `0.11`): the converged
refit is statistically indistinguishable from the original. `300k - old` =
`+0.284` (CI `[+0.224, +0.345]`, p `4e-16`): less harmful but still negative.
The intervention ordering does not follow held-out error, so it is a
selection-behavior difference, not better value accuracy.

**Verdict: fundamental low-signal/off-policy-selection limit, not a Q-quality
artifact.** The earlier negative max-Q result is not fixed by more convergence
because there is no better held-out residual to reach with these knobs, and the
single-action ranking target is not reproducible at all. Plot:
`/tmp/opencode/angular-vel-q-20261112/qbetter/eval/angular_velocity_tracking.png`.

FQE loss was unstable over 100k updates (mean 4.31, spikes to 30), and the mean
max-Q minus candidate-0 Q advantage was only `0.224` on a return scale of ~96,
with candidate 0 selected 12.5% of the time. Both interventions therefore act as
weak, near-random re-ranking: they slightly reduce yaw tracking quality and
survival and do **not** improve the estimated `r'`. This is a wiring and
measurement result for one artifact, not evidence that a yaw Q cannot help.

The 600-step time-domain trace (same Q) shows the same direction: world-step yaw
RMSE 0.2373 (plain) versus 0.2566 (max-Q), with the systematic error component
near zero between resamples and spikes at each command change. The plot is
`/tmp/opencode/angular-vel-q-20261112/eval/angular_velocity_tracking.png`.

Validation numbers for that run: lane-fidelity one-step errors within the fixed
limits, CPU/GPU Q max error `2.3e-5` with zero argmax disagreements, live-reward
parity `0.0`, and an identical-controller H500 replay difference with mean
absolute `0.018` (max `3.8`) out of a mean return of `96.2`. That replay spread is
Warp restore roundoff, not controller effect; the binary height reward hid it
because a `1e-3` height difference rarely flips a band membership. The five-step
negative control keeps exact discrete fields and needs `atol = 1e-2` for the
continuous return.

## Evidence boundaries

- One artifact per claim, 128 mid-gait states, two seeds for ranking, one
  checkpoint. No population, safety, or long-horizon determinism claim.
- The convergence sweep shows a held-out residual plateau, and the ranking
  diagnostic shows the realized within-state ordering is not reproducible across
  seeds. That is evidence about candidate actions at policy-noise scale; it does
  not show that a yaw Q cannot help with larger action changes or other states.
- The Q is fitted to a single reward term. A persistent intervention changes the
  continuation policy, so this does not validate `Q^{\pi}` for the intervened
  policy and is not policy improvement.
- Single-block near-zero gains are not effects: the `c=5` result is below the
  measured replay difference and did not reproduce. Report the max-Q and `c=20`
  direction only; the later two-block re-run supersedes both.
- Thresholds were chosen as scale probes from the Q distribution, not tuned on
  these returns; rejection attainment, fallback, and the paired distribution are
  the reported evidence.
- The reward is positive, so a fall simply truncates the return; a Q that
  under-predicts terminal value can prefer actions that raise the fall rate while
  looking slightly better at tracking.
- Portable tests cover the label formula, terminal zero, field lookup, FQE
  zero-terminal target, artifact identity, the no-tail accumulator, episode-split
  disjointness, and the held-out residual metric. They do not establish robot Q
  accuracy or ranking quality.

## Tests

```bash
uv run --frozen python -m pytest tests/unit_tests/test_angular_velocity_q.py -q
```

The durable experiment report is
[angular-velocity-tracking-20261112-v1](plans/supplements/replay_relabel_q/angular-velocity-tracking-20261112-v1/REPORT.md).
