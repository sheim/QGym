# Fixed-state mid-gait variance — 2026-11-10

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-fixed-variance-20261110-v1/REPORT.md`](/tmp/opencode/height-q-fixed-variance-20261110-v1/REPORT.md).
> Original sha256: `86815efa98c48a18ca5e816eddf77c2ca9909991173c2a6ae9005e8ce9ec3e2b`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-fixed-variance-20261110-v1/`](/tmp/opencode/height-q-fixed-variance-20261110-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../README.md).

## Protocol and artifacts

Frozen `model_600.pt` from `Nov10_10-36-22_gpu-final-600-trot-hist-20261110`.
Seed 240000; 100 frozen-policy steps once; all 1024 prefix worlds survived;
the first 128 were captured and replicated into 8 candidate worlds each.
Candidate0 is the insertion-observation policy mean; seven candidates add
independent Gaussian perturbations at the learned policy standard deviation.
Candidates and initial noisy observations are fixed for every repetition.

Eight independent batches of 16 continuations, H500, gamma .99, reward 0
inside 0.32±.05 m and -1 outside. True termination contributes the observed
terminal reward followed by the analytic absorbing -1 tail. No timeout reset.
Policy/observation noise, command resampling and pushes are paired within a
state across candidates; each continuation has a distinct seed. Both critics
were scored once on the same insertion observations/candidates and returns.

- Exact commands: `commands.txt`.
- Complete per-state/action values: `per_state_action.csv` (1024 rows).
- Raw tensors: `raw_returns.pt`, axes **[batch, repeat, state, candidate]**,
  shape **[8,16,128,8]**, including individual first rewards and tail bounds.
- Original batches: `batch_0.pt` ... `batch_7.pt`; unique seeds retained.
- State/model/controller/DR snapshot, observations, candidates and direct
  predictions: `insertion.pt`; 82-entry snapshot inventory.
- Full statistics: `statistics.pt`, `detailed_report.json`.
- Native gates and audits: `validation.json`, `insertion_audit.json`,
  `clamping_audit.json`; code/resource provenance: `provenance.json`.

## Measurements

Variance rows average the per-state/action quantities, not pooled state variance.
Within-batch action spread is over the eight **16-return means** for each state.
Paired contrasts exclude the nominal-versus-itself zero column.

| Quantity | Mean | Median |
|---|---:|---:|
| Variance of a fixed-action 16-return batch estimator | 1.2042 | 0.5638 |
| SD of that estimator | 0.9144 | 0.7509 |
| Within-batch across-action variance | 0.7671 | 0.1667 |
| Within-batch across-action SD | 0.6207 | 0.4083 |
| Within-batch across-action range | 1.8178 | 1.2010 |
| Paired candidate-minus-nominal estimated mean | -0.0654 | -0.0215 |
| Absolute paired estimated mean | 0.3221 | 0.1619 |
| Individual paired-return variance | 17.8964 | 5.1882 |
| Paired 16-return batch-estimator variance | 1.2091 | 0.2840 |

Independent batch pairs agree on candidate pair ordering **51.798%** of the
time; winner agreement is **14.509%**. Pairwise ties count one-half; winner
ties use the first index. The batch pair tie fraction is 0.00349%.
The saved covariance matrix satisfies Var(A−N)=Var(A)+Var(N)−2Cov(A,N).
No independent-variance subtraction or estimated noise ceiling is used.

| Critic | MAE to pooled 128-return finite estimate | Bias | Pair accuracy to that noisy estimate | Mean predicted action range |
|---|---:|---:|---:|---:|
| Original fair 100% | 1.6904 | +0.5048 | 0.51618 | 0.25033 |
| Eval-distribution 100% | 2.1407 | +0.2482 | 0.50781 | 0.44792 |

The first reward is -1 in **7.03125%** of individual cases (9/128 states),
the same for all eight candidates and all repetitions in this sample. This
is computed from individual binary values, not thresholded averaged rewards;
it does not prove action independence beyond the sampled candidates/states.
Termination by H500 is **2.12021%**. Mean finite/absorbing return is **-5.27575**.
For survivors the omitted tail is in **[-0.6570483,0]**; the mean return interval
from tail bounds alone is **[-5.91887,-5.27575]**, excluding sampling uncertainty.
Paired contrasts also have tail uncertainty (mean interval width about 1.2868).

**Interpretation:** 16-return action estimates have poor repeated-batch ranking
stability. Neither critic has demonstrated useful ranking here, but these data
do not justify saying the critic alone is the limiting factor. Empirical mean
differences remain noisy; eight batches estimate variance with only seven
degrees of freedom. These measurements supersede the earlier noise ceilings,
sorted-repetition analyses and repetition-varying mid-gait comparisons.

## Fidelity, checks, timing and caveats

Snapshot restore bypasses the task reset/clamp and restores native integration
inputs (including warmstart), model mass/inertia/friction and set_const outputs,
task history/gains/flags/counters, and DR state. Every repetition checks exact
pre-step buffer equality. Native next-step scratch is recomputed by forward.

Uninterrupted-versus-restored max next-step discrepancies: joint position
**1.4314e-6 rad**, joint velocity **7.1555e-5 rad/s**, base angular observation
**6.3926e-5 rad/s**, contact force **0.001160 N**. Replication gates also pass
(max contact discrepancy 0.000877 N). These are numerical one-step fidelity
tests, not bitwise long-horizon trajectory equivalence. Warp numerical variation
can be amplified by closed-loop dynamics.

Five captured states contain six joint positions beyond asset limits, up to
**0.210443 rad**. They are preserved as simulated, not projected/clamped.
Insertion heights average 0.317965 m; range 0.227442–0.373954 m; 93.75% inside
the band. The adapter is experimental and restricted to this 1024-world
Go2Trot nonsleeping Warp configuration, not a general backend checkpoint API.

- Focused tests: **5 passed**.
- Portable suite: **227 passed, 97 deselected**.
- Colocated gym/learning: **28 / 7 passed**.
- Warp suite: **32 passed, 292 deselected**; 18 existing MULTICCD support warnings
  for cylinder pairs in contract fixtures. Experiment log has no such warning.
- Ruff: clean.
- Batch0 benchmark: **56.8191 s**, reused as batch0.
- Remaining batches: 57.5226, 57.3381, 57.8430, 57.2717, 57.4348, 57.6173,
  57.2523 s. Total **459.0989 s (7 min 39.10 s)**, 131072 continuations.
- One in-run GPU sample: 2010/32607 MiB, 26% utilization; not a peak measurement.
- Resume initially failed before physics on NumPy-valued config metadata.
  Converted metadata to JSON primitives; retained original metadata file,
  identical snapshot tensors and all benchmark returns. No benchmark rerun.

Logs: `/tmp/opencode/height-q-fixed-variance-benchmark-v1.log`,
`height-q-fixed-variance-full-v1.log` (metadata failure),
`height-q-fixed-variance-full-v2.log` (successful continuation),
`height-q-fixed-variance-analysis.log`, and matching `portable-final`, `gym`,
`learning`, `warp` logs under `/tmp/opencode/`.

Prior unrelated dirty-tree work is preserved. No commits, todos, plan completion
or archival actions were taken.
