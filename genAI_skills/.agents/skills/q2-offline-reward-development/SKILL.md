---
name: q2-offline-reward-development
description: Add or change a Q2 offline reward and train its corresponding fixed-policy Q from recorded transitions. Use when extending replay/relabel/FQE beyond the three implemented reward paths (height, yaw angular velocity, foot contact force), adding raw reward features, or defining a new artifact, evaluation, or playback contract; not for an arbitrary reward plug-in.
---

# Q2 Offline Reward Development

Read `BigRedGym/genAI_skills/AGENTS.md`, `README_MUJOCO.md`, the durable
[offline height-Q documentation](../../../../docs/offline_height_q.md), the
[offline angular-velocity-Q documentation](../../../../docs/offline_angular_velocity_q.md),
the [offline contact-force-Q documentation](../../../../docs/offline_contact_q.md),
and the
[replay/relabel supplement index](../../../../docs/plans/supplements/replay_relabel_q/README.md).
Inspect the implementation and tests before editing. Three concrete offline
reward paths exist; the tree still has no generic reward framework.

## Three concrete reward paths

All paths share `TransitionDataset`, `FQE`, `QNetwork`, `FrozenTrotPolicy`,
`PolicySpec`, `TrainingSpec`, candidate sampling, and `WarpInsertionSnapshot`.
They differ in the reward shape, the terminal convention, and the artifact
identity, so their artifacts are deliberately not interchangeable.

### Height (binary band)

- `learning/offline/contracts.py`: `HeightRewardSpec` has `reference_height_m`,
  `bound_m`, `inside_reward=0.0`, `outside_reward=-1.0`,
  `timing=post_action_pre_reset`, and `terminal_continuation=absorbing_failure`.
  It requires `bound_m >= 0` and `outside_reward < inside_reward`; other timing
  or continuation values fail.
- `learning/offline/rewards.py`: `relabel_height()` reads only
  `successor_height_m` and writes `reward` plus `reward_spec`; the bound is
  inclusive.
- `learning/offline/q_artifact.py`: `HeightQ` requires format `brg-height-q-v1`,
  schema `1`, and return convention
  `discounted_continuing_absorbing_failure`; `save_height_q()` writes the
  height manifest.
- `scripts/train_height_q.py` requires `--height_reference`, `--height_bound`,
  `--gamma`, and `--subset_group`, and defaults the output label to `height_q`.
  It is FQE, not IQL/CQL, and does not optimize a policy.
- `learning/offline/evaluation.py` and `scripts/play_height_q.py` construct
  height rewards and validate height-Q playback; the playback Q input allow-list
  rejects `base_height`, `base_lin_vel`, `contact_forces`, `contacts`,
  `rigid_body_states`, and `root_states`.

### Yaw angular velocity (squared exponential)

- `AngularVelocityRewardSpec` carries `tracking_sigma`, `error_scale`,
  `timing=post_action_pre_reset`, and
  `terminal_continuation=zero_after_terminal`; `failure_value(gamma) == 0.0` for
  any valid gamma. `error_scale` is the task's hard-coded yaw-error divisor, not
  a config field: `Go2Trot._reward_tracking_ang_vel` computes
  `exp(-((cmd - ang_vel) / error_scale)**2 / tracking_sigma)`. Relabeling with
  only `tracking_sigma` silently labels a sharper, different reward.
- `relabel_angular_velocity()` and `angular_velocity_reward()` live in
  `learning/offline/rewards.py`; the label uses `commands[:, 2]` and
  `base_ang_vel[:, 2]` from `next_actor_obs`, located by manifest
  `obs_names`/`obs_dims` through `observation_field_slice()`, never by index.
- `AngularVelocityQ`/`save_angular_velocity_q()` require format
  `brg-angular-vel-q-v1`, return convention
  `discounted_episodic_zero_after_terminal`, and `failure_value == 0.0`.
- `scripts/train_angular_velocity_q.py` requires `--snapshot`, `--checkpoint`,
  `--output`, `--sigma`, `--gamma`, and `--subset_group`; it refuses a `--sigma`
  that differs from the snapshot's recorded
  `env_cfg.reward_settings.tracking_sigma`. `--validation_fraction` adds an
  episode-grouped held-out split and writes a per-fraction report with the loss
  curve, train-sample Bellman residual, and held-out Bellman-residual curve.
- `scripts/evaluate_angular_velocity_q_persistent.py`,
  `scripts/summarize_angular_velocity_q_persistent.py`, and
  `scripts/plot_angular_velocity_tracking.py` are the evaluation, summary, and
  time-domain entry points. `scripts/diagnose_angular_velocity_candidate_ranking.py`
  scores stored candidate actions against realized finite returns (Spearman,
  argmax hit, regret, advantage correlation). The reward is positive, so
  rejection thresholds must be chosen on that scale; height thresholds are
  meaningless here.

### Foot contact force (per-foot peak-load cap)

- `ContactForceRewardSpec` (`learning/offline/contracts.py`) carries
  `foot_names`, `thresholds_bw`, `percentile`, `timing=post_action_pre_reset`,
  and `terminal_continuation=zero_after_terminal`. It fails closed on mismatched
  lengths, non-positive thresholds, a percentile outside `(0, 100)`, and any
  other timing or continuation value.
- The label is `-1` when any foot's post-action vertical load exceeds its own
  threshold, otherwise `0`; exactly zero on true termination
  (`failure_value == 0.0`, no absorbing tail).
- Inputs are `next_contact_forces` (`[N, bodies*3]`), `bodyweight` (`[N, 1]`,
  from the randomised link masses) and `field_metadata.body_names`. All three
  must be recorded (`--replay_fields=contacts`) or the labeller fails; a
  core-only snapshot cannot be relabelled, and replaying recorded actions cannot
  rebuild the state.
- `derive_contact_thresholds()` takes each foot's percentile of the
  per-touchdown peak distribution (>= 5 control steps above 5 % bodyweight,
  peak >= 30 % bodyweight, inside one episode, terminated rows excluded) and is
  applied to the training pool only. Thresholds are **behaviour-relative**: they
  move 5-8 % between deterministic playback and the stochastic training policy,
  so freeze them from the recording in the artifact.
- `ContactQ`/`save_contact_q()` require format `brg-contact-q-v1`, return
  convention `discounted_episodic_zero_after_terminal`, `failure_value == 0.0`,
  and a parseable contact reward spec. `bodyweight` and `contact_forces` are
  privileged and are rejected as Q inputs.
- `scripts/train_contact_q.py` derives (or accepts) the thresholds, prints the
  label rate, fits with `--validation_fraction` for an episode-grouped held-out
  residual curve, and writes `<fraction>_report.json`.
- Label rates are imbalanced (~2.8 % of steps at the 90th-percentile
  thresholds), so report Bellman residuals and value error by label class, and
  note that the label is a landing-impact label: most flagged steps are in the
  first quarter of a stance, so the Q has to anticipate touchdown.

## Generalize with an explicit schema evolution

A new reward needs a new, named contract, not a changed height-artifact label.
Use a reward-specific spec/format or a versioned generalization, and fail closed:

1. Add a reward spec with units, bounds, timing, terminal convention, source
   fields, and reward name. Validate required values in `__post_init__`; do not
   silently default a new required field.
2. Bump or add the replay/Q format identifiers and `SCHEMA_VERSION` (or keep a
   distinct new format) and update every reader. `load_snapshot()` checks exact
   field-set equality, so a new field invalidates old snapshots; migrate
   explicitly or reject them with a clear error.
3. Give the artifact its own format/loader or a versioned generalized loader.
   `HeightQ`/`save_height_q()` must not accept the new reward by overwriting
   `reward_spec`; never relabel a height-trained Q as a new reward.
4. Record provenance: config fingerprint, actor checkpoint and sha256, dataset
   IDs, split and seed, normalization statistics, gamma, failure value, source
   revision, and `validated_domain`. Keep `validation_report_reference` null
   until a real report exists.
5. Update tests, CLI help, and documentation together. A new flag exists only
   after its parser contains it; use current `--help`, not a copied list.

## Reward timing, labels, and deployment inputs

- The label is measured after the action and before reset. For ended rows,
  `next_actor_obs` is the clean pre-reset successor plus recording-only noise;
  it is never the post-reset observation. Preserve that distinction.
- A label may use privileged simulator measurements (clean height, contacts,
  tracking errors) if recorded, but the Q input must remain the actor
  observation available at deployment. A privileged Q input creates a
  non-deployable claim; report that limitation.
- A new reward must be a scalar per transition and use one explicit control
  interval. To match the original PPO reward, reuse
  `learning/offline/training_reward.py`: `TrainingRewardModel.step_terms()`
  masks ordinary terms with `~terminated`, termination terms with `terminated`,
  and multiplies by `env.dt` in the runner's `mask * ((weight * dt) * f())`
  order; `total()` uses the recorded order. Do not reconstruct it from actor
  observations or compare it to a differently scaled label.
- Derive a label from recorded observation fields only when the exact training
  term is reconstructable from those fields. The angular-velocity path is such a
  case: both `commands` and `base_ang_vel` are recorded actor observations, so
  the yaw term is recovered with `observation_field_slice()` and the task's
  `error_scale`/`tracking_sigma`. Do not infer a reward from an observation that
  the task computed with different scaling, smoothing, or masking.
- The task resamples commands inside `_post_decimation_step` and the runner
  computes rewards after `env.step()`, so a recorded transition's label uses the
  **post-resample** command. Read `next_actor_obs` for the label and
  `env.commands` after `env.step()` for live evaluation; a pre-step command
  clone describes the interval that just ended, not the recorded label.
- `successor_height_m` is one recorded scalar. Do not overload it. If the source
  quantity, contact history, controller history, or domain state was not
  recorded, recollect it; a reward cannot be reconstructed from missing data.
- For FQE targets, query the successor observation produced after the action.
  Do not query the reset observation/state. For insertion experiments, query Q
  at the insertion state and use the fresh post-action observation, not a
  reset-centered state.

## Terminal values, bounds, and tails

- `terminated` and `timed_out` are independent. True termination takes
  precedence and uses the explicit terminal convention; timeout alone
  bootstraps from the recorded pre-reset successor. Keep flags separate.
- The height failure value is `outside_reward / (1 - gamma)` because that reward
  is bounded, and the terminal return absorbs `-1` forever. Do not copy the
  `-100.0` value from a test fixture or hard-code a sentinel for a new reward.
  Compute failure/tail values from the new reward's actual bounds and units, and
  reject invalid `gamma`.
- A bounded positive reward usually wants the zero-after-terminal convention
  instead: the label is exactly `0` on a terminal step, `failure_value == 0.0`,
  and `accumulate_return_zero_after_terminal()` adds `gamma**step * reward` while
  alive without a tail. Do not reuse the height accumulator for such a reward.
  Live label computation must mask with `~terminated` before accumulating.
- A finite rollout omits a tail. Compute its lower/upper bound from the actual
  min/max reward; if the reward is continuous and unbounded, there is no valid
  finite tail bound. State truncation or an explicit terminal value instead of
  inventing a bound. Report the omitted tail alongside any surviving return.
  For a `[0, 1]` reward the surviving tail is bounded above by
  `gamma**H / (1 - gamma)` and below by zero.

## Replay collection and leakage
Recording is opt-in and must not change PPO storage or consume training RNG.
The recorder owns a separate generator; keep it separate. Observation
normalization burn-in resets environments, and the first post-burn-in PPO
transition is excluded from replay (`_skip_next_replay`); preserve that order.

`TRANSITION_FIELDS` currently contains `actor_obs`, `action`,
`next_actor_obs`, `successor_height_m`, `terminated`, `timed_out`, `episode_id`,
`step_id`, `env_id`, and `behavior_iteration`. `TransitionRecorder.append()`
records one policy step per control step and rejects `n_policy_steps != 1`.
Adding raw features requires a
recorder buffer, append arguments, manifest fields, dataset validation, and a
format/schema change. Snapshots are immutable, chronological, bounded by
capacity, and never deleted automatically; a wrapped last-N ring loses early
data, so use `--replay_snapshot_interval` when early/late comparisons matter.

Use collection/snapshot seeds explicitly. `nested_subsets(..., group_by="episode")`
groups by the `(env_id, episode_id)` pair, not `episode_id` alone. For held-out
or train/validation splits, split complete groups and assert disjoint
`(run_id, env_id, episode_id)` sets. `concat_datasets()` deduplicates by
`(run_id, env_id, episode_id, step_id)`. Never create leakage with a random
row split of adjacent transitions, and do not choose a threshold on the same
returns used to report it.

## Fit and compare Q fairly

`FQE.compute_target()` is
`r + gamma * (1 - terminated) * mean_a' Q_target(o', a')`, with the explicit
failure value on termination. Freeze the target policy explicitly: checkpoint,
sha256, actor config, observation noise/scales, action convention, control dt,
`n_policy_steps`, and stochastic action sampling. These belong in `PolicySpec`;
changing any changes what Q means.

- Use `--subset_group episode` for nested fractions when temporal groups matter;
  fractions are strictly increasing and end at `1.0`.
- `updates` is the number of optimizer updates, not an epoch count. Report both
  updates and effective passes over each dataset, since equal fractions do not
  imply equal sample counts. Do not compare runs with different rollout or data
  geometry as if only the algorithm changed.
- Observation normalization is fitted on each training subset and stored in the
  artifact. Freeze it during training, evaluation, and inference; do not
  recompute it from evaluation data. Reward scale/units are not silently
  normalized: their scale changes the target and the meaning of thresholds.
- Use `uv run --frozen` and record CPU/GPU device. A training loss is a fit
  diagnostic, not evidence of Q quality; validate on held-out groups or fresh
  simulators separately.
- FQE from a zero-initialized network with a large discount starts far from its
  fixed point: the target scale grows as `1/(1-gamma)` and the reported minibatch
  loss can fall while the held-out residual stays large. Always add an
  episode-disjoint validation split (`episode_split()`,
  `--validation_fraction`) and report the held-out one-step Bellman residual
  (`_bellman_metrics()`, written to the per-fraction training report), not just
  the training loss mean. Select a variant by held-out error, not by the smaller
  displayed loss.
- A candidate-ranking diagnostic is the decisive usefulness test for an offline
  critic: score stored candidate actions with Q, realize their finite returns
  from the same states, and report Spearman, argmax hit rate, top-1 regret, and
  the Q-advantage-versus-realized-advantage correlation. Also report the realized
  within-state spread and its across-seed stability, and treat the seed-to-seed
  within-state rank correlation of the *same* controller as the ceiling a ranker
  could reach. If that reliability is near zero, no ranker can help and the
  negative result is a data/signal limit, not a training bug.

## Artifact, evaluation, and playback

- Export atomically (temporary directory then rename), refuse an existing
  destination, and write an exact manifest. Validate shape, positive
  observation standard deviations, finite predictions, format, and schema on
  load. Do not use silent defaults for a missing new field.
- Evaluation must use independently initialized fixed states, the same
  candidate actions, repeated stochastic continuations, and preserved trial/
  repetition IDs. Do not sort or re-pair seeds after collection. A candidate
  first action is followed by the frozen policy; predictions must refer to the
  state actually queried.
- Threshold reports need false acceptance, false rejection, ambiguous cases,
  and candidate ranking. Do not reduce a threshold decision to average binary
  success or select it by aggregate error. Keep paired candidate differences,
  covariance across candidates, and effect size (mean difference, variance/SD);
  use a p-value as secondary evidence, not a substitute for the paired estimate.
  Do not claim a noise ceiling from a finite batch.
- A persistent policy intervention is not `Q^pi` validation: executing Q every
  decision changes the continuation policy and invalidates the nominal FQE
  targets. Compare the original PPO return against the height/new-reward return
  with `training_reward.py` / `reward_tradeoff.py`; report the tradeoff and
  missing-feature limitations explicitly.
- `maxq` selects the highest Q among eight stochastic draws. `rejection`
  selects the first draw with `Q >= threshold`, otherwise falls back to max-Q.
  The policy mean is not a candidate in either mode; fallback is a behavior, not
  a safety guarantee. Do not confuse rejection's eight stochastic draws with
  `rollout_threshold`'s mean-plus-eight controller.
- `WarpInsertionSnapshot` and the k-th-prefix protocol are restricted
  Go2Trot/Warp experimental adapters, not a generic restore API. Restoring a
  snapshot does not prove deterministic long-horizon replay; a repeated prefix
  can diverge. Keep the insertion-state device, task flags, history, and domain
  state checks, and compare each `insert_step` only against its own measured
  return. Headless playback is a wiring smoke, not viewer or Q-quality evidence;
  manual CPU viewer validation remains separate.

## Current reference commands
All three reward paths have implemented entry points; do not invent flags. For a new
reward, add its own parser first.

```bash
uv run --frozen python scripts/train.py --help
uv run --frozen python scripts/train_height_q.py --help
uv run --frozen python scripts/evaluate_height_q.py --help
uv run --frozen python scripts/play_height_q.py --help
uv run --frozen python scripts/train_angular_velocity_q.py --help
uv run --frozen python scripts/evaluate_angular_velocity_q_persistent.py --help
uv run --frozen python scripts/summarize_angular_velocity_q_persistent.py --help
uv run --frozen python scripts/plot_angular_velocity_tracking.py --help
uv run --frozen python scripts/train_contact_q.py --help
uv run --frozen python -m pytest tests/unit_tests/test_offline_replay.py \
  tests/unit_tests/test_height_fqe.py tests/unit_tests/test_height_q_playback.py \
  tests/unit_tests/test_angular_velocity_q.py tests/unit_tests/test_contact_q.py -q
uv run --frozen python -m pytest -q
```

Add focused tests for zero/sign/scale, inclusive boundaries,
terminal-versus-timeout target, schema rejection, split disjointness, artifact
round-trip, cross-reward artifact rejection, and the no-tail accumulator. Do not
commit logs, artifacts, or any file over 100 KiB.

Simulator evidence for a new reward needs the same identity gates as height:
checkpoint hash, artifact spec, environment `config_fingerprint`, lane fidelity,
a one-step CPU/GPU Q check, a live reward-parity check against the task method,
and an identical-controller negative control. Warp restore is not bitwise: a
continuous reward exposes roundoff that a binary reward hides, so pin discrete
fields exactly and give the continuous return an explicit small bound (and
report the H500 duplicate-rollout difference as a replay noise floor).
