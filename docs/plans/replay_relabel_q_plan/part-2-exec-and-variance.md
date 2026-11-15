# Part 2 — execution log, full variance report, 64-rollout reanalysis and sixteen-batch report (S10-S13)

> Part of the replay/relabel Q plan. Index: [README.md](README.md).

## Exec — 2026-11-10 — phase edit (V1..V3)

### Active execution — fixed-state batch variance (2026-11-10)

User authorized EDIT and dirty-tree execution, preserving prior CORL/config work.
Implement isolated `scripts/evaluate_height_q_variance.py` plus
`learning/offline/fixed_state_variance.py` and focused tests. Capture once at
k=100; select first 128 prefix survivors, replicate each across 8 GPU worlds,
generate nominal plus 7 local learned-std candidates at insertion, and score both
existing critics once. Use 8 independent batches of 16 paired continuations,
H=500, gamma=.99, height 0.32±.05, absorbing terminal failure. Benchmark batch0
before remaining batches and retain it. Save raw [batch,repeat,state,candidate]
returns/rewards, source survivor flags, native/task/DR state and predictions.

Gate: exact restored buffer equality, unclamped qpos, uninterrupted/restored and
source/replica one-step fidelity (root/angvel 2e-3, qpos 1e-4, qvel 1e-2,
contacts 1 N). Report numerical errors, not bitwise trajectory identity.
Statistics retain shared-seed covariance and report empirical batch estimator
variance, action spread, paired contrasts and between-batch ranking agreement.
No independent-variance subtraction or noise-ceiling claim.

Initial verification: focused analytic/fake-env tests `4 passed`; full Ruff clean.
Benchmark command/log and final gate results will be recorded after execution.

#### Fixed-state verification and benchmark

- Expanded focused tests: `5 passed` (includes rebound controller history,
  flags/counters and DR RNG restoration). Portable gate: `227 passed, 97 deselected`;
  colocated `gym`: `28 passed`; `learning`: `7 passed`; Ruff clean.
- Native snapshot inventory: 82 entries, including integration warmstart, task
  controller/history/flags, batched model properties and DR state. Every restore
  checks exact buffer equality; native reset/forward is not invoked by restore.
- Uninterrupted versus restored one-step max error: qpos `1.4314e-6 rad`,
  qvel `7.1555e-5 rad/s`, root/body angular observation `6.3926e-5`,
  contact force `0.001160 N`. Source-to-replicated-world max contact difference
  `0.000877 N`; other maxima match. All predeclared gates pass.
- Captured prefix: all 1024 worlds survived; first 128 selected once. Their
  insertion height mean `0.317965 m`, range `[0.227442,0.373954]`, 93.75% inside
  the band. Episode counters are 100; common counter is 101 (factory reset step).
- **Clamping audit:** 6 joint positions across 5 captured states exceed asset
  limits, maximum `0.210443 rad`. The native snapshot preserves these observed
  positions exactly; it does not project them through the task reset clamp.
  Native constraint penetration remains a property of the simulated source states.
- Batch0: `56.8191 s` for 16 continuations × 1024 worlds × H500; retained for
  the eight-batch experiment. One in-run GPU sample: `2010/32607 MiB`, 26%
  utilization (sample, not peak memory).
- Initial resume failed before simulation because config metadata contained a
  NumPy scalar rejected by weights-only loading. Converted config metadata to
  JSON primitives; retained `insertion-original-metadata.pt` and all snapshot
  tensors/batch0. Source now emits portable metadata. Resume log v2 records the
  successful remaining batches; repaired snapshot tensor equality was audited.
- Output root: `/tmp/opencode/height-q-fixed-variance-20261110-v1/`.
  Exact commands: `commands.txt`; native gates: `validation.json`; state/limit
  audits: `insertion_audit.json`, `clamping_audit.json`; source/resource record:
  `provenance.json`. Logs are `/tmp/opencode/height-q-fixed-variance-{benchmark-v1,full-v1,full-v2,portable-final,gym,learning}.log`.

#### Fixed-state experiment result

All eight batches executed; benchmark batch0 retained. Raw shape is
`[8,16,128,8]` = [batch,repeat,state,candidate], 131072 continuations.
Batch times: 56.8191, 57.5226, 57.3381, 57.8430, 57.2717, 57.4348, 57.6173,
57.2523 s; total **459.0989 s**. Native Warp suite additionally passed:
`32 passed, 292 deselected`, 18 fixture MULTICCD support warnings; experiment
log has no warning/error/overflow. Postprocessing audits axes, unique seeds,
binary rewards, return bounds and the covariance identity.

| Quantity (mean over fixed state/actions) | Result |
|---|---:|
| Variance of repeated 16-return estimator | 1.20421 |
| Within-batch across-action variance | 0.76708 |
| Within-batch across-action range | 1.81783 |
| Paired nonnominal-minus-nominal estimated mean | -0.06542 |
| Mean absolute paired estimated difference | 0.32212 |
| Paired individual-return variance | 17.89643 |
| Paired 16-return estimator variance | 1.20912 |
| Between-batch pair-ranking agreement | 0.51798 |
| Between-batch winner agreement | 0.14509 |

Original/eval-distribution critic MAE to the pooled noisy 128-return finite
estimates: **1.69043 / 2.14065**; pair-ranking accuracy **0.51618 / 0.50781**.
Direct predictions are frozen once at the shared insertion states. First-reward
negative rate is 7.03125% from raw binary values; no candidate differences were
observed in these first rewards. Termination by H500: 2.12021%. Mean return
tail-only interval: [-5.91887,-5.27575]; surviving omitted tail [-0.6570483,0].

Interpretation: 16-return candidate estimates have poor batch-to-batch ranking
stability; no useful critic ranking is established. The experiment does not
identify the critic as the sole limiting factor, and supplies no noise ceiling.
Eight independent batches give only seven degrees of freedom for estimator
variance. Tail intervals exclude sampling uncertainty. Native restoration is
numerically validated one step, not bit-identical over long closed-loop rollouts.

Full report:
[durable copy](../supplements/replay_relabel_q/fixed-variance-20261110-v1/REPORT.md);
per-state/action values: `per_state_action.csv`; complete statistics:
`detailed_report.json`, `statistics.pt`; raw returns/first rewards/tail bounds:
`raw_returns.pt`. Analysis command is in `commands.txt` and does not simulate.
Plan stays in EDIT; no commit, todo changes, completion or archival action.

Covers the three verification cases in §3.d: replay/relabel integrity, FQE semantics/artifact round-trip, and the robot pipeline/scientific experiment. Experiment settings in §4.e remain unresolved; no scientific GPU experiment was launched. A one-iteration CPU software smoke was run to verify wiring.

### Verification result — V1
Command: `uv run --frozen python -m pytest tests/unit_tests/test_offline_replay.py tests/unit_tests/test_height_fqe.py -q`
Expectation: ring/snapshot/relabel/subset integrity plus analytic FQE targets, artifact round-trip, and finite-rollout tail bounds.
Actual: `17 passed`.
Log: terminal output in the execution session.
Pass.

### Verification result — V2
Command: `uv run --frozen python -m pytest -q`
Expectation: existing portable gate remains green after runner/recorder integration.
Actual: `221 passed, 97 deselected`.
Log: terminal output in the execution session.
Pass.

### Verification result — V3
Commands: `uv run --frozen python -m pytest gym -q` and `uv run --frozen python -m pytest learning -q`.
Expectation: colocated suites remain green.
Actual: `27 passed` and `7 passed`.
Log: terminal output in the execution session.
Pass.

### Verification result — static checks
Commands: `uv run --frozen ruff format <touched paths>` and `uv run --frozen ruff check .`.
Expectation: touched files are formatted and the repository lint gate is clean.
Actual: `All checks passed!` for the full repository.
Log: terminal output in the execution session.
Pass.

### Verification result — smoke
Commands:
- `scripts/train.py --task=go2trot --device=cpu --headless --num_envs=32 --max_iterations=1 --batch_size=4096 --replay_record --replay_capacity=128 --replay_snapshot_interval=0 --run_name=replay-smoke-end-20261110`
- `scripts/train_height_q.py` on the resulting 128-transition snapshot with 10/50/100% fractions.
- `scripts/evaluate_height_q.py` on the 100% artifact with one trial, one repetition, five-step horizon, and three candidate actions.
Expectation: recording, snapshot loading, height relabeling, nested subsets, FQE fitting, artifact export, and fresh-rollout evaluation work end-to-end.
Actual: snapshot published with `valid_count=128`; three Q artifacts produced; evaluation JSON produced. The initial 100% smoke loss showed terminal-target spikes, which is expected at 20 updates with the absorbing-failure value; this is a wiring smoke, not a convergence or Q-quality result.
Pass for wiring; scientific evidence remains open.

### Verification result — CUDA recorder smoke
Command: one-iteration `go2trot` on `cuda:0` with 256 environments, `--replay_record --replay_capacity=256`.
Expectation: the recorder works on CUDA tensors and writes a loadable snapshot.
Actual: first attempt failed because episode/step counters were CPU while `dones` was CUDA; moved `dones` to CPU for counter updates. Retest passed at ~18k steps/s for 256 environments. This exposed a device bug not covered by the CPU-only portable tests.
Pass after fix.

### Data collection — 2026-11-10
Early run: `logs/go2trot/Nov10_10-35-07_gpu-early-50-trot-hist-20261110`, iterations 0–50, 3,276,800 transitions, 1.5 GiB snapshot.
Final run: `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110`, resumed from the early `model_50.pt` to iteration 600, last 1,000,000 transitions, 449 MiB snapshot.
Mix: `logs/go2trot/hist-mix-20261110/replay/replay_iter_0000600`, equal-size early/final transition sample (2,000,000 transitions).
Analysis outputs: `/tmp/opencode/height-distribution-20261110.png` and `.json`. Phase/height analysis: `/tmp/opencode/phase-height-20261110.png`, `.json`, and `/tmp/opencode/phase-bound-sensitivity-20261110.json`. These are distribution-selection evidence, not Q accuracy or policy-quality evidence.
Key phase finding: phase is uniform over [0, 2π); final-policy frequency is ~1–3 Hz; phase-locked mean height has ~1.8 cm amplitude. For a 0.32 ± 0.05 m bound, 5.3% of final transitions fall outside, varying from 3.2% at 95° to 8.4% at 205°. Tighter bounds amplify the phase dependence (0.32 ± 0.02 m: 38.3% overall, 31.0–47.7% by phase).

### Deviation
`TransitionRecorder` snapshot publication had an end-of-run collision with a periodic snapshot at the same iteration. Fixed by treating an unchanged ring as a no-op before the existing-path check; the immutable-path check still rejects a different snapshot at the same iteration. Tests updated.

The recorder skips the first post-burn-in transition when observation normalization burn-in resets the environments, because the runner's `actor_obs` captured before burn-in is stale. PPO's established first transition is unchanged; only the replay record is excluded.

Recording currently rejects `n_policy_steps != 1` because the height reward is measured once after the policy interval. Other actor/control frequency ratios need a separate interval contract.

## 10. Full fixed-state variance report — merged 2026-11-10

Full content of [the original REPORT.md](../supplements/replay_relabel_q/fixed-variance-20261110-v1/REPORT.md)
follows, with heading levels adjusted. The earlier execution summary and historical
results remain above. Artifact names in this section resolve under
[/tmp/opencode/height-q-fixed-variance-20261110-v1/](/tmp/opencode/height-q-fixed-variance-20261110-v1/).
Source implementations: `BigRedGym/scripts/evaluate_height_q_variance.py`,
`BigRedGym/scripts/summarize_height_q_variance.py`, and
`BigRedGym/learning/offline/fixed_state_variance.py`.

### Fixed-state mid-gait variance — 2026-11-10

#### Protocol and artifacts

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

#### Measurements

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

#### Fidelity, checks, timing and caveats

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

Direct log links: [benchmark](/tmp/opencode/height-q-fixed-variance-benchmark-v1.log),
[failed resume](/tmp/opencode/height-q-fixed-variance-full-v1.log),
[completed collection](/tmp/opencode/height-q-fixed-variance-full-v2.log),
[analysis](/tmp/opencode/height-q-fixed-variance-analysis.log),
[portable](/tmp/opencode/height-q-fixed-variance-portable-final.log),
[gym](/tmp/opencode/height-q-fixed-variance-gym.log),
[learning](/tmp/opencode/height-q-fixed-variance-learning.log),
[Warp](/tmp/opencode/height-q-fixed-variance-warp.log).

## 11. Variance at 64 rollouts — two disjoint groups, 2026-11-10

Reused the existing 128 continuations per fixed state/action: original batches
0–3 versus 4–7, reshaping [8,16,128,8] to [2,64,128,8]. Identical grouping for
all states/actions preserves shared-noise pairing. No new simulator collection
or random regrouping. Full [64-rollout report](../supplements/replay_relabel_q/fixed-variance-64-20261110-v1/REPORT.md),
[script](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/analyze.py),
[results](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/results.json),
[per-state/action tensors](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/statistics.pt),
and [log](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/analysis.log).

### 11.a Observed variance and ordering stability

For each state/action, let M_g be the mean of its 64 returns in group g.
Sample Var_g(M_g), with correction=1, is (M_0−M_1)^2/2: variance of a
64-return estimator, not the pooled 128-return mean. Across-action variance
uses correction=1 over eight action means within each group/state. Paired
contrasts subtract nominal within the same group before computing variance;
the nominal self-contrast is excluded. Medians follow the original report's
torch.median lower-middle convention. These are descriptive summaries across
fixed states/actions, not pooled-state variances or confidence intervals.

| Quantity | At16 mean | At16 median | Observed at64 mean | Observed at64 median |
|---|---:|---:|---:|---:|
| Fixed-action estimator variance | 1.204215 | 0.563815 | **0.343431** | **0.060939** |
| Fixed-action estimator SD | 0.914370 | 0.750876 | 0.382869 | 0.246858 |
| Within-group across-action variance | 0.767078 | 0.166738 | **0.316077** | **0.042201** |
| Within-group across-action SD | 0.620655 | 0.408336 | 0.372277 | 0.205428 |
| Within-group across-action range | 1.817829 | 1.200961 | 1.107343 | 0.600737 |
| Paired estimator variance | 1.209124 | 0.284019 | **0.280147** | **0.040696** |

Across-action variance by group, mean (median): group0 0.305306 (0.047770),
group1 0.326847 (0.040662). This spread includes estimation noise and true
candidate differences; it is not a noise-corrected action signal.
Average fixed-action SD at64 is **0.382869**, whereas sqrt(mean variance) is
**0.586030** (at16: 0.914370 versus 1.097367).

Pair ordering agreement: **0.5546875 (55.46875%)**, versus 0.5179817044 at16.
Winner agreement: **0.1953125 (19.53125%)**, versus 0.1450892857 at16.
Pair ties count half; winner ties use the first index; at64 pair-tie fraction
is zero. Repeatability is not accuracy against true Q, and the reused-data
comparison is not an independent significance test or a noise ceiling.

Pooled diagnostics are unchanged: paired mean −0.065416 (median −0.021472),
absolute paired mean 0.322115 (0.161948), individual paired-return variance
17.896427 (5.188242). Regrouping leaves pooled critic targets unchanged.

### 11.b Projection under independent continuations

Var64 = Var16/4 assumes independent, identically distributed continuations
at each fixed state/action. It does not assume independent actions; apply
the scaling to paired contrasts with their covariance retained.

| Projected at64 quantity | Mean | Median |
|---|---:|---:|
| Fixed-action estimator variance | **0.301054** | **0.140954** |
| Fixed-action estimator SD | 0.457185 | 0.375438 |
| Paired estimator variance | **0.302281** | **0.071005** |

Projected sqrt(mean variance) is 0.548684. These projections use the at16
variance estimates (seven df); they are distinct from the observed two-group
estimates. Across-action spread cannot simply be divided by four because it
contains true action differences as well as noise.

### 11.c Precision, checks and approval boundary

**Two groups give only one degree of freedom per variance estimate: weak
precision.** Correlated actions, states and candidate pairs do not supply
additional independent 64-rollout batch replicates. The lower observed median
is not evidence of better-than-1/n scaling; one-df variances are highly skewed
and variable. Observed mean variance 0.343431 versus projection 0.301054 is a
preliminary result; ranking repeatability remains weak at 55.47%.

No new costly collection is needed for this preliminary answer. More independent
64-rollout batches would be needed for precise empirical variance or strong
ordering-stability conclusions; none were launched. No plain-policy versus
top-Q sampled-action selection experiment was implemented or run; that second
experiment requires further user approval after these first results.

Original restoration, state-selection and truncation caveats in §10 apply.
The survivor omitted tail remains [-0.6570483,0]; mean tail-only interval
[-5.91887,-5.27575] and paired tail uncertainty exclude sampling uncertainty.
More rollouts do not remove truncation bias or establish filter suitability.

Offline checks passed: float64 formulas, axes/shape, 128 unique seeds,
grouping/pooling identities, Var_two=(difference)^2/2, covariance identity
Var(A−N)=Var(A)+Var(N)−2Cov(A,N), and original at16 variance reproduction.
Reproduction command is in the linked report. Historical suite counts in §10
refer to the original implementation; this reanalysis changes no product code.

## 12. Approved EDIT: sixteen independent 64-return batches — 2026-11-10

Select 16 of the original 128 states by CPU `torch.randperm`, seed 2026111064,
independently of returns; retain exact original observations/eight candidates.
Reuse all 128 old continuations as batches 0–1. Collect fourteen additional
64-return batches using 1024 worlds ordered [lane8,state16,candidate8]. This
adds 114688 rollouts, for 131072 total retained rollouts. No prefix rerun.

New call seeds start at 6400000, stride 10; the identity of a continuation is
(call seed, lane). Each lane consumes distinct vectorized RNG draws, while the
eight candidate worlds share observation/action/command/push randomness. Native
model/DR/controller/history snapshots use the same saved-world index mapping.
Run exact inventory and native next-step/replication/pairing gates first; retain
the first new block as the benchmark and stop if projected new collection exceeds
720 seconds. Expected simulation time remains comparable to prior 459 seconds.

Analyze raw [16,64,16,8] returns and the same data regrouped [64,16,16,8]. Report
mean/median estimator variance, within-batch action spread, paired contrasts,
rank/winner repeatability, and batch-cluster bootstrap uncertainty. Fifteen df
per at64 variance; actions and states are not independent bootstrap replicates.
Tail and experimental restore limitations persist. Top-Q policy selection is
explicitly deferred. No commits, todos, archival, or plan-completion actions.

### 12.a Implementation and benchmark gate

Selected original indices: `[24,12,116,80,50,82,59,92,4,97,9,63,90,87,13,23]`.
`WarpInsertionSnapshot.load_export` applies one saved-world mapping to every
native integration/model/controller/task/DR entry, including flattened views.
The full 82-entry equality audit passes. Native source-to-lane next-step maximum
errors: qpos 6.5565e-7 rad, qvel 3.2663e-5 rad/s, contact 0.0008621 N; repeated
lane restore maxima: qpos 7.1526e-7, qvel 3.3855e-5, contact 0.0006256 N.
Actual native commands/pushes are candidate-shared and lane-distinct. A separate
CUDA audit with the real policy's 48 observation/12 action dimensions also
confirms candidate sharing and distinct draws for all 28 lane pairs.

First retained new block: **29.411492 s**, projected new collection **411.760892 s**;
runtime gate passed. Collection resumes with `--blocks 14 --resume` under
`/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/`, logs
`/tmp/opencode/height-q-16x64-benchmark.log` and `height-q-16x64-full.log`.
Focused tests 7 passed; portable 229 passed/97 deselected; colocated 35 passed;
Ruff and `git diff --check` clean. Final collection/statistics follow below.

### 12.b Completed collection and matched-data results

All fourteen new blocks completed: **399.989556 s (6m39.99s)** additional
collection, including the retained 29.411492 s benchmark; subsequent blocks
28.2630–28.6392 s. This meets the comparable-runtime goal (original 459.0989 s).
Times exclude setup/native validation/offline analysis. Combined raw data are
**[16,64,16,8]**, 131072 retained returns, of which 114688 are new. Both critics
reuse their exact original predictions on the same selected observations/actions.

Full [report](../supplements/replay_relabel_q/fixed-variance-16x64-20261110-v1/REPORT.md),
[JSON](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/results.json),
[raw returns](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/raw_returns.pt),
[per-state/action CSV](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/per_state_action.csv),
[commands](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/commands.txt).
`insertion.pt`, `provenance.json`, `code_provenance.json`, native validation and
CUDA noise audits reside in that same directory. Analysis log:
`/tmp/opencode/height-q-16x64-analysis.log`.

At16 below regroups the **same combined dataset** as [64,16,16,8]. At64 has
16 independent batches (15 df); at16 has 64 batches (63 df). Summary means and
lower-middle medians are across per-state/action quantities, not pooled-state
variance. Paired contrasts preserve covariance and exclude nominal self-contrast.

| Quantity | At16 mean | At16 median | At64 mean | At64 median |
|---|---:|---:|---:|---:|
| Fixed-action estimator variance | 1.271764 | 0.277224 | **0.313614** | **0.077920** |
| Fixed-action estimator SD | 0.932266 | 0.526521 | 0.458830 | 0.279141 |
| Within-batch action variance | 0.644969 | 0.140738 | **0.183222** | **0.033911** |
| Within-batch action SD | 0.566918 | 0.375150 | 0.308394 | 0.184148 |
| Within-batch action range | 1.652487 | 1.094448 | 0.900168 | 0.536622 |
| Paired-contrast estimator variance | 1.302996 | 0.285495 | **0.365503** | **0.064551** |

Pair-order agreement: **51.0830% at16 → 52.7772% at64**. Winner agreement:
**13.4487% → 18.0729%**. Pair ties count half; winner ties select the first index.
The variance reduction is close to IID 1/n scaling, while ranking repeatability
remains weak; this is not accuracy against known true Q or a critic noise ceiling.

Whole-64-batch bootstrap (4000 resamples, seed 2026111065) keeps all states/actions
and nested four 16-return subgroups together. Approximate 95% percentile intervals:

- At64 mean fixed-action variance **[0.247670,0.334420]**; median variance
  **[0.057594,0.105392]**.
- At64 mean within-batch action variance **[0.165520,0.202223]**.
- At64 mean paired estimator variance **[0.255878,0.444997]**.
- Mean variance64/variance16 **0.246597**, interval **[0.200368,0.269332]**.

These are conditional on sixteen fixed states/candidate sets, not intervals for
the state population. With only sixteen clusters and skewed returns, ordinary
percentile-bootstrap variance coverage is approximate and may be biased; action
columns are never treated as independent replication. The original 128-state
pooled summary is not the matched comparison above. Original-only at16 variance
on this subset was 1.157721 (median .452057); using only fourteen new at64 batches
gives .315808 (median .088595), pair agreement .529337 and winner .182005.

### 12.c Diagnostics, caveats and execution boundary

Original/eval-distribution critic MAE to pooled finite estimates: **1.495400 /
2.741345**; biases **+.641948 / +.189999**; pair accuracy **.542411 / .459821**;
predicted action ranges **.195765 / .383914**. These remain noisy finite-target
diagnostics, not validated top-Q policy performance.

Individual first rewards are all zero in this randomly selected subset; no binary
reward was inferred from an averaged value. Termination is **2.670288%**. Mean
finite/absorbing return **-5.312794**; mean tail-only interval
**[-5.952297,-5.312794]**. Survivor omitted tail remains [-.6570483,0]; mean
paired-tail interval width **1.278371**. Tail bounds exclude sampling uncertainty.

Selected states have no out-of-limit joint positions, but the original full-source
restore gate still preserves all six such positions. No clamp/reset/prefix rerun
was introduced. Native source-to-lane and lane-restoration checks passed again on
resume (maximum contact discrepancies .000969 N and .000626 N). All 82 fields,
including native model constants, history and DR, are mapped identically. Actual
CUDA observation/action noise and native command/push checks confirm candidate
sharing and lane distinction. Vectorized lane subsequences are distinct RNG draws;
they are not separately seeded generators. Reused/new batches differ in native
world placement, so one-step fidelity does not imply bitwise long-rollout identity.

Analysis checks pass for original-data reuse, exact observations/candidates/critic
predictions, 896 unique new (seed,lane) identities, raw axes/bounds/binary rewards,
matched pooling, rank formulas and paired covariance identity. No claims of a
general snapshot API or bitwise trajectory equivalence. **Top-Q policy selection
remains explicitly deferred.** No commits, todos, plan completion or archival.

Final validation: **7 focused**, **229 portable / 97 deselected**, **35 colocated
(28 gym + 7 learning)**, **32 Warp / 294 deselected** passed. Warp emitted the
same 18 MULTICCD cylinder-pair support warnings in existing contract fixtures;
collection emitted no physics warnings (the benchmark's tensor-copy advisory
was removed before resume). Ruff, format checks and `git diff --check` pass.
Logs: `/tmp/opencode/height-q-16x64-{portable,colocated,warp,analysis}.log`.

## 13. Full sixteen-batch report — merged 2026-11-10

Verbatim report below from
[/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/REPORT.md](../supplements/replay_relabel_q/fixed-variance-16x64-20261110-v1/REPORT.md).
Artifact-relative paths resolve in the original artifact directory. Provenance: [code record](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/code_provenance.json), [commands](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/commands.txt), [benchmark log](/tmp/opencode/height-q-16x64-benchmark.log), [collection log](/tmp/opencode/height-q-16x64-full.log), [analysis log](/tmp/opencode/height-q-16x64-analysis.log).

```markdown
# Sixteen independent 64-return batches — 2026-11-10

## Protocol

Random selection seed 2026111064; original state indices `[24, 12, 116, 80, 50, 82, 59, 92, 4, 97, 9, 63, 90, 87, 13, 23]`. Selection precedes reading returns.
Exact original physical snapshots, noisy insertion observations, eight candidate actions and both critics' predictions are retained; no prefix rerun.
Two batches reuse all 128 original continuations; fourteen new batches add 896 continuations per state/action: 114688 new, 131072 retained rollouts.
1024 worlds = lane8 × state16 × candidate8. Each new 64-return batch contains eight calls × eight lanes. RNG identity is (call seed,lane): distinct vectorized random subsequences across lanes, not eight separately seeded generators. Candidate noise is shared within each state/lane. New call seeds 6400000..6401110 stride10 do not collide with old streams.
Frozen policy, H500, gamma .99, height reward 0 inside .32±.05m and -1 outside; observed terminal reward plus exact absorbing failure tail.

## Matched-data results

All rows summarize per-state/action quantities, not pooled-state variance. At64 uses 16 batches (15 df); at16 regroups the SAME returns into 64 batches (63 df). Medians use torch's lower-middle convention. Paired contrasts exclude nominal-minus-itself; covariance is retained.

| Quantity | At16 mean | At16 median | At64 mean | At64 median |
|---|---:|---:|---:|---:|
| fixed_action_variance | 1.271764 | 0.277224 | 0.313614 | 0.077920 |
| fixed_action_sd | 0.932266 | 0.526521 | 0.458830 | 0.279141 |
| within_batch_action_variance | 0.644969 | 0.140738 | 0.183222 | 0.033911 |
| within_batch_action_sd | 0.566918 | 0.375150 | 0.308394 | 0.184148 |
| within_batch_action_range | 1.652487 | 1.094448 | 0.900168 | 0.536622 |
| paired_estimator_variance | 1.302996 | 0.285495 | 0.365503 | 0.064551 |

rank_agreement: at16 **51.083021%**, at64 **52.777158%**.

winner_agreement: at16 **13.448661%**, at64 **18.072917%**.
Pair ties count half; winner ties select the first candidate. Repeatability is not accuracy against true Q. Action spread contains both signal and noise.

## Uncertainty

4000 whole-64-batch resamples, seed 2026111065; fixed states/candidates; paired regrouping; approximate percentile intervals with only 16 clusters, not exact variance coverage or state-population uncertainty. Actions/states stay together in every resample. Ordinary percentile bootstrap variance intervals can be biased with only 16 batches and skewed returns; they are conditional, approximate intervals.
- At64 fixed_action_variance.mean: 95% bootstrap interval **[0.247670, 0.334420]**.
- At64 fixed_action_variance.median: 95% bootstrap interval **[0.057594, 0.105392]**.
- At64 within_batch_action_variance.mean: 95% bootstrap interval **[0.165520, 0.202223]**.
- At64 paired_estimator_variance.mean: 95% bootstrap interval **[0.255878, 0.444997]**.
- Mean variance ratio64/16: **0.246597**, 95% paired cluster-bootstrap interval [0.20036818087100983, 0.2693323493003845]; 1/4 is the IID scaling reference, not a fitted noise ceiling.

The old 128-state summary and this random 16-state subset have different coverage; do not directly interpret their differences as rollout-count effects. `original_subset_at16` and `new_only_at64_sensitivity` are retained in JSON. More batches reduce variance-estimation uncertainty, not hidden-state critic bias, finite-horizon truncation or state-population uncertainty.

## Pooled diagnostics and critics

First reward -1 rate **0.000000%**, termination **2.670288%**. Rates use individual binary rewards, never thresholded averages.
Mean finite/absorbing return **-5.312794**. Mean tail-only interval **[-5.952297287757066, -5.31279406610262]**; surviving omitted tail [-0.6570483,0]. Paired tail interval mean width 1.278371. These exclude sampling uncertainty.

| Critic | MAE to pooled finite estimate | Bias | Pair accuracy to noisy estimate | Predicted action range |
|---|---:|---:|---:|---:|
| Original fair100 | 1.495400 | 0.641948 | 0.542411 | 0.195765 |
| Eval-distribution100 | 2.741345 | 0.189999 | 0.459821 | 0.383914 |

## Fidelity, timing and artifacts

All 82 native integration/model/task/controller/DR entries use the same index mapping; every continuation asserts exact restoration. Native one-step source/replica and lane restore gates, plus command/push candidate pairing and lane distinction pass (`validation.json`, `validation_resume.json`). This proves numerical one-step fidelity, not bitwise long-horizon equivalence. World reordering can amplify numerical solver differences in closed loop; reused versus new batches also differ in native world placement.
Selected-state joint-limit audit: {'selected_out_of_limit_positions': 0, 'selected_states_with_excess': 0, 'max_excess_rad': 0.0, 'all_insertion_observations_candidates_predictions_exact': True}. Native positions are preserved without reset clamping. Adapter remains restricted to this 1024-world nonsleeping Go2Trot/Warp configuration.
New collection **399.989556s** (6.666 min), first block **29.411492s**, retained. Original full collection was 459.0989s. Timings exclude environment setup, native validation and offline analysis; no extra critic rollout is used.

- `raw_returns.pt`: explicit [batch64,repeat64,state16,candidate8] axes, returns/first rewards/survival, old seeds and new (seed,lane) identities.
- `new_block_00.pt`..`new_block_13.pt`: retained new blocks and timings.
- `insertion.pt`, `insertion_audit.json`: exact selected snapshot, observations, candidates and both original predictions.
- `statistics.pt`, `per_state_action.csv`, `results.json`: full statistics, tail bounds, matched grouping, bootstrap intervals and sensitivity summaries.
- `provenance.json`, `code_provenance.json`, `commands.txt`: selection, stream identities, source hashes, code hashes/revision/dirty status and CLI.

Reproduce analysis from BigRedGym: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python scripts/summarize_height_q_extension.py /tmp/opencode/height-q-fixed-variance-16x64-20261110-v1`.

**Top-Q sampled-action policy selection remains explicitly deferred.** No noise-ceiling claim, commits, todos, plan completion or archival.

## Validation record

- Focused tests: 7 passed.
- Portable gate: 229 passed, 97 deselected.
- Colocated gym/learning: 35 passed (28 + 7).
- Warp: 32 passed, 294 deselected; 18 existing MULTICCD cylinder-pair support
  warnings in contract fixtures. Collection has no physics warnings.
- Ruff, touched-file format check and `git diff --check`: clean.
- `cuda_noise_audit.json`: real-policy CUDA observation/action draws share all
  candidate noise and differ across all 28 lane pairs. Separate diagnostic seeds
  99000001/99000002 do not enter collected returns.
- Analysis assertions: exact original data reuse and insertion observations/actions/
  predictions, 896 unique new seed/lane identities, axes/finite bounds/binary
  rewards, matched pooling, rank formulas and covariance identity pass.

Logs: `/tmp/opencode/height-q-16x64-{benchmark,full,analysis,portable,colocated,warp}.log`.
The benchmark printed a harmless tensor-copy advisory; `torch.as_tensor` removed
it before resume, with no change to seeds or retained benchmark data.

Interpretation: variance decreased approximately fourfold with fourfold rollout
count; ranking repeatability is still weak. Neither this nor critic agreement
against noisy finite estimates establishes useful policy selection.
```

