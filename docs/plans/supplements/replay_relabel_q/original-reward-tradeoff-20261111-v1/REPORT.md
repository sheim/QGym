> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/REPORT.md`](/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/REPORT.md).
> Original sha256: `01f64239ea4767a4cb42c2fcffe1ccfec599c23cfd33a8c58cd1ace1fd766712`.
> Raw artifacts, blocks, checkpoints, logs and CSVs remain in
> [`/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/`](/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for this note. See the
> [supplement index](../README.md).

# Original PPO reward versus original height-Q gain

**Question.** At the same 128 fixed insertion states and the same 1024
paired executions per state, does the original fair-100 height-Q max-of-8
controller improve or sacrifice the original PPO training return relative
to the plain candidate-0 stochastic `model_600` controller? Original return
is the primary objective; height-band return is measured on the same
executions as a secondary objective.

## Bottom line

- Plain original-return mean: **26.977067**; original height-Q mean: **26.950118**; paired gain: **-0.026949** (sacrifice).
- Height-band paired gain on the same executions: **+1.718288**.
- Paired state-level t-test (df=127): mean **-0.026949**, 95% CI **[-0.039288, -0.014610]**, two-sided p=**3.095e-05**. State-cluster bootstrap CI: **[-0.038765, -0.014482]**.
- Execution win/tie/loss: **55741/0/75331** of 131072.

This is the measured trade-off at fixed states, not a population claim,
not `Q^pi` validation, not policy improvement, and not a safety result.

## Identity and protocol

- Source insertion: `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`
  SHA-256 `dcfca737603a5e238feecd5fceae643b426c3d99af5c8ab79df3acf6e9dbf712`.
- Checkpoint: `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt`
  SHA-256 `f88b1f9b829675c97ed0c420202cf54456dc414641def34e379e39407ee3adf6`.
- Original height-Q: `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct`
  (`manifest.json` `db267188c3f8cfe5cfe9d54638e318c387537d3320c799374d557fcbb90de0e9`,
  `q.pt` `e59d738a606979edeb021c86651e78f632bd4b78a99a9a24e80ebecccdad2236`).
- Training reward config: `{"action_rate": 0.1, "action_rate2": 0.05, "ang_vel_xy": 0.01, "min_base_height": 0.5, "orientation": 1.0, "reference_traj": 3.0, "swing_contact": 3.0, "torques": 5e-06, "tracking_ang_vel": 4.0, "tracking_lin_vel": 4.0, "trot_support": 3.0}`
  plus termination `{"termination": 0.01}`.
- Gamma **0.99** read from the loaded training config (`train.algorithm.gamma`), dt **0.02**, control **50.0 Hz**, decimation **1**, `n_policy_steps=1`.
- 128 fixed states x 8 lanes = 1024 Warp worlds; 16 blocks x 8 calls;
  1024 executions per state per controller; seeds
  `7100000 + 10*call`, `call=0..127`.
- H500, gamma **0.99** for both objectives,
  `episode_length_s=13.02` (no admin timeout).
- Native inventory **82** fields restored. The short same-controller
  negative control reproduced discrete selections, live-step counts and
  height returns exactly; continuous original returns and actions repeated
  within the predeclared numeric tolerances (solver scratch is not part of
  the snapshot). H5 height parity was exact for returns, steps and
  selection.

## Original PPO reward results

Original return is `sum_t gamma^t * r_t`, where `r_t` is the runner's
post-step `mask * ((weight * dt) * reward_function())` term sum. Ordinary
terms use `~terminated`; the termination term uses `terminated`. Rewards
are exactly zero after a true termination; there is no absorbing `-100`
tail. The finite H500 return omits a continuing tail that is **not**
bounded by the height interval.

| Controller | Mean | Pooled SD | Min | Max | 128-state mean SD |
|---|---:|---:|---:|---:|---:|
| plain | 26.977067 | 1.595573 | 6.718256 | 29.328643 | 1.244362 |
| original | 26.950118 | 1.551451 | 7.197974 | 29.328767 | 1.207611 |
Pooled SD is the sample SD over 131072 executions per controller.


Terminal fraction: plain **0.021370**, original **0.022362**. Mean live steps: **496.098** vs **495.882**.
Raw blocks retain per-execution original return, per-term discounted sums,
final alive mask, live-step count, termination step, first reward and
tracking aggregates. No full per-step state tensors are stored.

### Behavior aggregates

| Controller | Survivor fraction | Mean live steps | XY RMSE | Yaw RMSE | Speed / command | First-step termination |
|---|---:|---:|---:|---:|---:|---:|
| plain | 0.978630 | 496.098 | 0.449373 | 0.771272 | 0.742534 | 0.000000 |
| original | 0.977638 | 495.882 | 0.456089 | 0.768642 | 0.736196 | 0.000000 |

### Per-term paired contributions

All entries are scalar after time aggregation, in original-return units.
`delta = original - plain`; positive means the original controller's
trajectory produced a larger weighted term. Negative terms are penalties,
so a positive delta on a penalty means less cost. The total gain is
negative here, so `delta / total` has the opposite sign of a positive
delta.

| Term | Weight | Plain | Original | Delta | Delta / total | Penalty |
|---|---:|---:|---:|---:|---:|---|
| action_rate | 0.1 | -0.012354 | -0.012103 | +0.000251 | -0.93% | yes |
| action_rate2 | 0.05 | -0.008572 | -0.008230 | +0.000341 | -1.27% | yes |
| ang_vel_xy | 0.01 | 0.005163 | 0.005314 | +0.000151 | -0.56% | no |
| min_base_height | 0.5 | 0.732220 | 0.737267 | +0.005047 | -18.73% | no |
| orientation | 1 | 3.911147 | 3.913303 | +0.002156 | -8.00% | no |
| reference_traj | 3 | 5.135251 | 5.147015 | +0.011764 | -43.65% | no |
| swing_contact | 3 | -0.089403 | -0.082005 | +0.007398 | -27.45% | yes |
| torques | 5e-06 | -0.000425 | -0.000420 | +0.000004 | -0.02% | yes |
| tracking_ang_vel | 4 | 7.696689 | 7.685943 | -0.010746 | +39.88% | no |
| tracking_lin_vel | 4 | 6.710590 | 6.649529 | -0.061062 | +226.58% | no |
| trot_support | 3 | 2.896762 | 2.914507 | +0.017745 | -65.85% | no |
| termination | 0.01 | -0.000000 | -0.000000 | -0.000000 | +0.00% | yes |

Per-term deltas sum to **-0.026949**; the reported total gain is **-0.026949** (exact identity checked).

## Paired gain and precision

| Objective | Mean gain | Pooled SD | State mean SD | State-cluster CI95 | Execution win/tie/loss |
|---|---:|---:|---:|---|---|
| original | -0.026949 | 0.854399 | 0.070546 | [-0.038765, -0.014482] | 55741/0/75331 |
| height | +1.718288 | 5.456443 | 2.079561 | [+1.376687, +2.090214] | 74525/27786/28761 |

Execution win/tie/loss fractions for the original objective: **0.425270/0.000000/0.574730**.

The state-cluster bootstrap resamples the 128 fixed state rows and is a
descriptive across-state interval. The paired t-test treats the 128
state mean gains as independent working units; executions within a state
share seeds, lanes and the fixed initial state. Neither is a random
state-population or safety statement. The original objective was the
primary hypothesis; height is secondary.

Because the plain original-return baseline is positive, the paired
gain is **-0.100%** of that
baseline.

## 16-execution batches

| Objective | Positive / tie / negative batches | Positive fraction | Mean per-state batch SD | All-positive states | Majority-positive states |
|---|---:|---:|---:|---:|---:|
| original | 2848/0/5344 | 0.347656 | 0.166716 | 0/128 | 31/128 |
| height | 7023/0/1169 | 0.857300 | 1.044088 | 22/128 | 128/128 |

Batches are 64 contiguous groups of 16 executions per state (two calls
times all eight lanes). A positive batch fraction is not expected utility.

## Height-band results

The height objective is unchanged from the existing height-Q experiments:
`0` inside `0.32 +/- 0.05 m`, `-1` outside, with the discounted absorbing
failure tail. It is reported on the same trajectories but is a different
objective with a different terminal convention.

| Controller | Mean | Pooled SD | Min | Max | 128-state mean SD |
|---|---:|---:|---:|---:|---:|
| plain | -5.283413 | 8.115789 | -100.000002 | 0.000000 | 6.735841 |
| original | -3.565125 | 6.323508 | -100.000002 | 0.000000 | 5.040793 |

The height `[-0.657048, 0]` survivor-tail interval applies only to this objective. It is not applied to the original PPO return.

## Reproducibility against the prior persistent run

| Controller | Prior height mean | Current height mean | Mean drift | State-mean correlation | Mean abs state diff | Max abs state diff |
|---|---:|---:|---:|---:|---:|---:|
| plain | -5.285775 | -5.283413 | +0.002361 | 0.999996 | 0.012508 | 0.091919 |
| original | -3.566637 | -3.565125 | +0.001512 | 0.999992 | 0.014835 | 0.094034 |

This is an aggregate reproducibility check, not a bitwise match claim:
Warp closed-loop noise and possible source drift can change long-horizon
returns.

## Scope limitations

- The 128 states are fixed `k=100` insertion snapshots from one frozen
  policy and one source lane; they are not a random state population.
- H500 is twice the 250-step training episode. `episode_length_s` was
  extended so no administrative timeout occurs; this is an horizon
  extension, not a training-episode replay.
- Zeroing all rewards after true termination is deliberate for the
  original objective. It differs from the height absorbing failure tail
  and from any finite-horizon Q target.
- The original H500 return omits a continuing future tail. Unlike height,
  original rewards are not bounded in `[-1, 0]`, so the `[-0.657, 0]`
  height bound is not a bound on original return.
- Per-term values are scalar after time aggregation; no per-joint or
  per-body reward dimensions were stored.
- The height Q was trained on nominal-policy data. The intervened branch
  is off that distribution; selected Q is not `Q^pi` truth.
- Execution-level intervals are conditional on the fixed states. The
  state-cluster and paired t intervals are descriptive and depend on the
  working independence assumption across fixed configurations.

## Artifacts

- Raw blocks: `/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/plain_00.pt` ... `original_15.pt` (32 files).
- Audit and summaries: `/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/results.json`, `per_config.csv`, `terms.csv`, `batch_means_original.pt`, `batch_means_height.pt`.
- Inputs, code hashes, commands and exits: `/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/provenance.json`, `code.json`, `code_resume.json`, `commands.txt`, `commands.log`, `validation.json`, `validation_resume.json`, `completion.json`, `timings.jsonl`, `dirty.txt`, `dirty_resume.txt`.
