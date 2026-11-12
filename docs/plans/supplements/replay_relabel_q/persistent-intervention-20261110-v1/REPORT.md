# Persistent sample-eight intervention — 2026-11-10/11

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-persistent-intervention-20261110-v1/REPORT.md`](/tmp/opencode/height-q-persistent-intervention-20261110-v1/REPORT.md).
> Original sha256: `bf7501425e817f06f0ec05c8e2fa702c2c78df675c897c6296b8827916cb227c`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-persistent-intervention-20261110-v1/`](/tmp/opencode/height-q-persistent-intervention-20261110-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../README.md).

## Question and approved protocol

Does a frozen height Q, applied at **every** decision to pick the best of eight
fresh stochastic samples of the same frozen policy, change the H500 height-band
return relative to plain stochastic `model_600`? This is a **policy-intervention**
probe: a measured change under a different policy, not a policy-improvement
theorem and not an evaluation of Q as `Q^pi`. The Q targets were fit for the
nominal policy, and persistent deviations leave that distribution.

- H500, gamma 0.99, reward 0 inside 0.32 ± 0.05 m, -1 outside, measured after the
  action and before reset. True termination adds the exact absorbing -1 tail.
  Administrative timeouts are suppressed and audited (zero observed).
- Three controllers: `baseline` (candidate0 only), `original` (max of eight
  samples by the original fair-100 critic), `evaldist` (same with the
  eval-distribution-500 critic).
- Both Q controllers use the frozen checkpoint
  `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt`
  and artifacts
  `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct`,
  `/tmp/opencode/height-q-evaldist-500-v1/height_q_evaldist_500/eval500/100pct`.
  Q inputs are the 48 actor-observation fields that exist at deployment; height
  is never an input.
- 128 original k=100 physical snapshots and their originally retained noisy
  insertion observations come from
  `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`; no prefix
  rerun. 1024 worlds = lane8 × state128. The candidate axis is neural only: the
  eight candidates are scored offline and only the chosen action enters physics.
- 16 blocks × 8 calls × 8 lanes = **1024 stochastic executions per state per
  controller**, i.e. 131072 rollouts/controller and 393216 total. Actual executed
  decisions: 65,016,291 / 64,994,521 / 64,980,358.
- Randomness identity is `(call seed, lane, state)` with one seed per execution:
  `torch.manual_seed(seed)` (environment commands/pushes), `seed+1` (observation
  noise), `seed+2` (candidate Gaussian draws). Seeds `7100000 + 10·call`, so all
  three controllers share the identical draw streams for a given execution; the
  states diverge once the actions differ. Lanes are distinct vectorized
  subsequences, not separately seeded generators.
- At each step each branch samples eight candidates from its **own** current
  observation; baseline executes index 0, so its marginal is exactly the frozen
  policy's action distribution. The Q branches execute the Q-argmax index.

## Validation gates (all passed before full collection)

| Gate | Result |
|---|---|
| Source snapshot fidelity (uninterrupted vs restored next step) | qpos 2.15e-6 rad, qvel 1.03e-4 rad/s, contact 1.50e-3 N |
| state128/lane8 replication and lane-restore | qpos 1.43e-6, qvel 7.16e-5, contact 5.65e-4 N |
| 82-entry inventory mapping (native integration, model constants, controller history, DR) | exact tensor equality |
| CPU vs GPU Q predictions | max abs error 5.7e-6 / 7.6e-6; split-batch error 9.5e-6 / 1.1e-5 |
| Sampled-argmax agreement, CPU vs GPU | 0 disagreements (131072 rows per critic) |
| Negative control, identical plain controllers | exact equality of rewards, returns, selection counts and masks over 5 steps |
| Timeouts, non-binary first rewards, non-finite returns | none |
| Selection counts vs live decisions | exact match for all controllers and states |

No task reset, no joint-limit clamp and no administrative reset was applied.
Native model/DR/controller/history state is restored per execution and asserted
before stepping. Five-step exactness for identical controllers is established;
a full-length duplicate plain controller was not run, and Warp round-off can
diverge over 500 closed-loop steps.

## Returns

Per-state means over 1024 paired executions. Execution CIs use within-state
standard errors only; state-cluster CIs resample the 128 states and are the
population-level statement. Tail intervals apply the survivor bound
`[-0.6570483, 0]` and exclude sampling uncertainty.

| Quantity | baseline | original-Q | evaldist-Q |
|---|---:|---:|---:|
| Mean H500 return | -5.2858 | **-3.5666** | **-4.1758** |
| State-mean SD | 6.739 | 5.048 | 5.663 |
| State-mean q05 | -19.564 | -15.662 | -15.591 |
| Execution CI (fixed states) | [-5.3105, -5.2610] | [-3.5875, -3.5458] | [-4.1994, -4.1521] |
| State-cluster CI | [-6.4766, -4.1763] | [-4.4693, -2.7341] | [-5.1939, -3.2461] |
| Tail-only interval | [-5.9287, -5.2858] | [-4.2090, -3.5666] | [-4.8179, -4.1758] |
| Inside-band step fraction | 95.460% | **97.177%** | **96.553%** |
| Fall fraction | 2.155% | 2.242% | 2.275% |
| Mean steps | 496.03 | 495.87 | 495.76 |
| First reward = -1 | 7.031% | 7.031% | 7.031% |
| First-step termination | 0% | 0% | 0% |
| XY tracking RMSE (preterminal) | 0.4493 | 0.4561 | 0.4549 |
| Yaw tracking RMSE (preterminal) | 0.2208 | 0.2259 | 0.2254 |
| Speed / command speed | 0.7426 | 0.7362 | 0.7383 |

First-reward and fall metrics use individual executions, never thresholded
averages. The identical 7.031% first reward reflects the shared retained initial
observation and shared first-step draw.

## Paired gains versus baseline

Paired by execution (identical seed stream), then averaged per state.

| Quantity | original-Q | evaldist-Q |
|---|---:|---:|
| Mean paired gain | **+1.7191** | **+1.1100** |
| Per-state gain SD | 2.0784 | 1.5488 |
| Per-state gain median | +0.6595 | +0.4477 |
| Per-state gain q05 | +0.0905 | +0.0263 |
| Per-state gain range | +0.0055 … +11.978 | -2.160 … +6.713 |
| State wins / ties / losses | 128 / 0 / 0 | 123 / 0 / 5 |
| Execution-level win probability | 0.5691 | 0.5154 |
| Paired execution CI | [+1.6917, +1.7466] | [+1.0817, +1.1384] |
| Paired state-cluster CI | [+1.3781, +2.0850] | [+0.8564, +1.3840] |
| Paired tail interval | [+1.0763, +2.3615] | [+0.4671, +1.7521] |
| Blocks 0–7 / 8–15 gain | +1.7006 / +1.7376 | +1.0963 / +1.1238 |

Gain deciles (original-Q): +0.006, +0.151, +0.250, +0.360, +0.515, +0.661,
+1.356, +1.930, +2.998, +4.819, +11.978. The distribution is right-skewed: half
the states gain at least 0.66, a few near-failure states gain up to 12.
Split-half agreement (1.70 vs 1.74; 1.10 vs 1.12) shows the effect is not
carried by one block.

## Selection, Q advantage and behavior changes

- Selected **indices** are nearly uniform (original 12.4982–12.5031%, evaldist
  12.4931–12.5059%). This is expected under exchangeability and is not evidence
  about the mechanism: eight IID candidates scored by any exchangeable critic,
  including a perfect one, give a uniform marginal index distribution. What the
  controller executes is not the nominal action distribution but a
  Q-selected order statistic, so an effect of selection bias alone cannot be
  excluded by this statistic. No randomized-score control was run; a random
  index choice would reproduce the baseline marginal exactly and therefore could
  not separate the two explanations either.
- The critic's mean predicted **per-decision** advantage is 0.1226 (original)
  and 0.2056 (evaldist). This is a single-state advantage under the nominal
  observation/action distribution and is **not** a forecast of the realised
  cumulative 500-step gain (+1.719 / +1.110); the two quantities differ by
  construction (per-decision vs accumulated, nominal vs intervened state
  distribution, learned target vs measured return). Their ratio is not an
  error or an underforecast.
- Outside-band step rate falls by **1.73 pp** (original; state CI [−2.03, −1.45] pp)
  and **1.08 pp** (evaldist; [−1.28, −0.90] pp). Negative means fewer violations.
- Fall rate rises by **+0.086 pp** (original; state CI [−0.038, +0.212] pp) and
  **+0.120 pp** (evaldist; [0.000, +0.244] pp). Positive means more falls; the
  original estimate is unresolved and the evaldist interval only touches zero.
- Velocity and yaw tracking degrade modestly (XY RMSE +0.007 m/s, yaw +0.005
  rad/s, speed/command −0.006 to −0.004): the critic trades ~1.5% of tracking
  quality for ~1.7 pp more time in the height band. The height reward does not
  pay for tracking, so this trade is the expected direction; it rules out a
  catastrophic gait failure, not all forms of height gaming.

Sign conventions above are explicit: `delta_oob = controller − baseline` outside-band
step rate, `delta_fall = controller − baseline` fall rate.

## Interpretation and limits

- **A realised persistent-intervention gain exists at these states and for this
  checkpoint.** Both critics improve mean H500 height-band return, with all-state
  wins for the original critic and a stable block split.
- **This is a policy intervention, not policy improvement in the theorem sense.**
  Q was trained for the nominal policy; the intervened branch leaves that
  distribution, so its targets are off-policy and no improvement or optimality
  guarantee follows from these measurements, even though the measured mean rose.
  It is also not a validation of approximate `Q^pi`.
- Because selected indices are uniform but selected actions are a Q-order
  statistic, the gain cannot be decomposed here into "Q information" versus
  "selection/order-statistic bias". Separating them needs a control that selects
  with the same bias but no action-value information (not run).
- Finite-sample and truncation uncertainty are separate. Paired truncation uses
  the asymmetric survivor bound (baseline fraction on the lower side,
  controller fraction on the upper side), giving widths ≈1.285 rather than the
  erroneous ≈2.57 from double-counting both fractions. State-cluster intervals
  are ~3× wider than execution intervals; quoting execution CIs alone would
  overstate precision.
- The 128 states are k=100 states from one frozen policy and one seed lane, not a
  state-population sample.
- Restoration is numerically validated one step, not bit-identical over long
  closed-loop rollouts; identical-controller equality holds exactly for 5 steps
  only.
- The eval-distribution critic's narrower gain and 5 losing states are reported
  as measured; training on the evaluation distribution did not dominate here.

## Audit corrections (2026-11-11)

Two analysis errors were found after the first draft; raw blocks were not rerun
and all corrected values come from the same 48 chunks:

1. **Paired truncation interval was doubled.** Each surviving execution's omitted
   tail is `[-0.6570483, 0]`, so a paired difference is bounded by the baseline
   survivor fraction on the low side and the controller's on the high side. The
   first draft added both fractions to both sides. An analytic two-execution
   oracle gives a true width of 0.657 and an old-formula width of 1.314 (ratio
   exactly 2.0). Corrected: original `[+1.0763, +2.3615]` (was
   `[+0.4339, +3.0043]`), evaldist `[+0.4671, +1.7521]` (was
   `[-0.1750, +2.3950]`). Single-controller tail intervals were already correct.
2. **Occupancy/fall statements mixed sign conventions.** CIs had been quoted in
   the opposite-difference units of their point estimates. Both are now stated
   as controller − baseline with the direction named: outside-band rate −1.73 pp
   / −1.08 pp (improvement), fall rate +0.086 pp / +0.120 pp (degradation).

`audit_corrected.json` contains the recomputation, seed-identity check, zero
timeouts, selection/steps equality, and the analytic oracle; `results.json` was
regenerated by the corrected summarizer.

## Timing, provenance and commands

Benchmark block (retained, not re-run): baseline 29.0368 s, original 29.7233 s,
evaldist 29.6375 s; ×16 projection 464.59 / 475.57 / 474.20 s, below the 720 s
gate, so no approval pause was needed. Full collection totals: 465.461 s,
476.240 s, 476.804 s. The resume spanned 23:31:14–23:52:55 local for 45 blocks
(15 blocks × 3 controllers) after the benchmark. Timings exclude environment
setup, native validation, and offline analysis. Peak allocated Torch memory
during collection was 103 MB beyond the environment.

Commands (run from `BigRedGym/`):

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python \
  scripts/evaluate_height_q_persistent.py \
  --source /tmp/opencode/height-q-fixed-variance-20261110-v1 \
  --output /tmp/opencode/height-q-persistent-intervention-20261110-v1 \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --load_run Nov10_10-36-22_gpu-final-600-trot-hist-20261110 \
  --artifacts \
    /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct \
    /tmp/opencode/height-q-evaldist-500-v1/height_q_evaldist_500/eval500/100pct \
  --blocks 1                 # benchmark; then --blocks 16 --resume
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python \
  scripts/summarize_height_q_persistent.py \
  /tmp/opencode/height-q-persistent-intervention-20261110-v1
```

Logs: `/tmp/opencode/height-q-persistent-{benchmark,full,analysis,diagnostics}.log`,
plus gate logs `height-q-persistent-{portable,gym,learning,warp}.log`.

Artifacts in this directory: `baseline_00..15.pt`, `original_00..15.pt`,
`evaldist_00..15.pt` (raw `[state128, execution64]` plus per-decision selection
counts, 19.35 MB per controller), `validation.json`, `validation_resume.json`,
`provenance.json`, `code.json`, `code_resume.json`, `commands.txt`, `dirty.txt`,
`dirty_resume.txt`, `results.json`, `per_state.csv`, `diagnostics.json`,
`diagnostics.py`.

## Validation record

- Focused `tests/unit_tests/test_persistent_intervention.py` +
  `test_fixed_state_variance.py`: 11 passed (4 new, including the asymmetric
  survivor-bound regression test).
- Portable gate: 233 passed, 97 deselected.
- Colocated gym / learning: 28 / 7 passed.
- Warp: 32 passed, 298 deselected; 18 pre-existing MULTICCD fixture warnings.
- Ruff and `git diff --check`: clean.

No commits, todos, plan-completion or archival actions. Prior CORL/config dirt
preserved.
