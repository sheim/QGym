# Part 4 — extra tests: threshold pilot, rejection sweep, PPO-reward tradeoff

> Part of the replay/relabel Q plan. Index: [README.md](README.md).

## Extra test — threshold-constrained persistent action selection — 2026-11-11

Status: **EXTRA TEST — pilot complete; full sweep NOT launched; awaiting
numerical threshold approval.** This is an appended extra-evidence section, not
a reopened defect and not a new plan. The §17 completion record and the archive
location `docs/plans/replay_relabel_q_plan/README.md` are
unchanged. No commit, todo, or archival movement was made.

### Extra scope and protocol

- Frozen `model_600.pt` from
  `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/` and the
  original fair-100 PPO-data critic at
  `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct`.
- Same 128 fixed `k = 100` native snapshots and initial noisy observations from
  `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`; no prefix
  rerun, no state clamping, no administrative timeout reset.
- H = 500, gamma = 0.99, reward 0 inside `0.32 ± 0.05 m` and -1 outside,
  observed terminal reward plus the exact absorbing-failure tail.
- Nine candidates at every decision: index 0 is the exact policy mean; indices
  1..8 are the same eight Gaussian draws used by the prior sample-eight
  protocol. The historical baseline candidate0 is now index 1. The prior max-Q
  controller semantics remain over indices 1..8 and are not conflated with a
  max over nine.
- Threshold controller: score all nine with the original Q; feasible means
  `Q >= c`; select the feasible candidate minimizing
  `sum(((a - mu) / sigma) ** 2)`; if none feasible, fall back to argmax Q over
  all nine. Mean control always selects index 0. Ties select the lowest index.
  `sigma` must be finite and strictly positive.
- The exact mean has normalized distance zero, so any feasible mean is always
  selected. The controller therefore decomposes into: mean (feasible mean),
  closest feasible stochastic candidate (mean infeasible but some stochastic
  candidate feasible), or fallback max-Q (none feasible).

### Implementation and tests

- `learning/offline/persistent_intervention.py`: added
  `sample_candidates_with_mean`, `score_candidates`,
  `threshold_action_indices`, `select_threshold_action`, `rollout_threshold`,
  `rollout_mean`, and a shared rollout loop. The existing `sample_candidates`,
  `select_action`, and `rollout` behavior is preserved; the stochastic block is
  identical for a paired generator seed.
- `tests/unit_tests/test_persistent_intervention.py`: six new tests cover the
  mean-plus-eight construction, byte-identical stochastic draws, finite
  positive `sigma`, feasibility/distance/fallback/ties/negative thresholds,
  selection of the scored action, and every-decision threshold/mean rollouts
  with absorbing-tail masking.
- `scripts/measure_height_q_threshold_pilot.py`: nominal-trajectory Q
  measurement. `scripts/summarize_height_q_threshold_pilot.py`: simulation-free
  distribution summary and threshold proposal.

### Pilot protocol, validation and benchmark

- 8 calls × 8 lanes = **64 executions/state**, seeds `7100000 + 10 * call`
  (the same protocol as persistent block 0, calls 0..7). 4,068,657 active
  decisions were scored.
- The executed action is new index 1, the first of the same eight draws, i.e.
  the historical baseline candidate0. All nine candidates are scored, but no
  candidate intervention changes the nominal action.
- Q is recorded before each step for alive worlds only; terminated worlds make
  no later decisions. Weighting is explicit: decision-weighted (each active
  call/lane/state/step equally, the primary per-decision rate) and
  state-step-weighted (average over calls/lanes for each state/step, then each
  state-step equally); state-weighted summaries average over steps/executions
  per state, then over states.
- Native validation: 82-entry snapshot inventory; every restore exact; source
  replication limits inherited from `insertion.pt`; CPU/GPU Q max error
  `3.81e-6`, split-batch `9.54e-6`, zero argmax disagreements; zero timeouts.
- Benchmark: 3.702–3.844 s/call, total **29.9 s**, peak allocated
  `124,339,200` bytes (~119 MiB) on the RTX 5090.
- Nominal cross-check against the existing persistent baseline block 0 for the
  same states and seeds: return correlation **0.99652**, mean difference
  **-0.00560**, 6504/8192 exact returns, first rewards exact 8192/8192,
  survival exact 8149/8192. Residual differences are Warp closed-loop numerical
  noise; the action/index and first-reward streams match.
- Pilot outcomes: mean H500 return **-5.2297** (SD 7.8577); termination
  **1.8188%**; inside-band step fraction **95.4497%**; first-reward -1 rate
  **7.03125%**; mean active steps **496.66**; zero timeouts.

### Q distribution along the nominal trajectory

Decision-weighted, active only, 4,068,657 decisions. The critic is strongly
left-skewed; the mean is not a good center for threshold placement.

| Statistic | mean | SD | min | q01 | q05 | q25 | q50 | q75 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q(exact mean) | -4.4326 | 7.6599 | -171.5235 | -27.5555 | -16.7558 | -4.8658 | -1.7761 | -0.9563 | -0.6259 | -0.5101 | 5.8575 |
| Q(max over 9) | -4.3082 | 7.5046 | -169.9485 | -26.8260 | -16.3255 | -4.7147 | -1.7214 | -0.9250 | -0.6027 | -0.4855 | 6.5850 |
| Q(min over 9) | -4.5750 | 7.8160 | -175.7362 | -28.3166 | -17.2009 | -5.0501 | -1.8487 | -0.9983 | -0.6574 | -0.5433 | 5.2537 |
| Q(max) - Q(min) | 0.2668 | 0.5286 | 0.0058 | 0.0262 | 0.0374 | 0.0651 | 0.1135 | 0.2806 | 0.9159 | 1.9262 | 27.0047 |

Supporting distributions:

| Distribution | mean | SD | q05 | q50 | q95 | n |
|---|---:|---:|---:|---:|---:|---:|
| State-step Q(mean) | -4.4495 | 4.0915 | -12.7230 | -2.6433 | -0.9565 | 64,000 |
| State Q(mean) | -4.4507 | 3.8066 | -11.9554 | -2.4563 | -1.1307 | 128 |
| Pooled candidate Q | -4.4381 | 7.6588 | -16.7577 | -1.7825 | -0.6286 | 36,617,913 |

Raw-range diagnostic (not clipped): Q(mean) is below the known continuous lower
bound -100 in **0.1104%** of decisions and above 0 in **0.00157%**; raw min
-171.52, max +5.86. Pooled candidate Q is below -100 in 0.1107%, above 0 in
0.00177%, min -175.74, max +6.59. This is a critic diagnostic, not a threshold
filter.

### Five proposed fixed thresholds

These are quantiles of the decision-weighted Q(exact mean) distribution:
`[-16.756, -4.866, -1.776, -0.956, -0.626]`. They were fixed from Q values
before any intervention outcome was observed; no threshold is tuned to realised
gains. Rates are on the nominal pilot distribution.

| c | source | mean pass | any stochastic feasible | any candidate feasible | fallback | mean selected | closest stochastic selected | state-step mean pass | state-step fallback |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **-16.756** | Q(mean) q05 | 95.000% | 95.275% | 95.275% | 4.725% | 95.000% | 0.275% | 99.080% | 0.900% |
| **-4.866** | Q(mean) q25 | 75.001% | 75.607% | 75.608% | 24.392% | 75.001% | 0.607% | 68.103% | 31.753% |
| **-1.776** | Q(mean) q50 | 49.999% | 50.995% | 50.997% | 49.003% | 49.999% | 0.998% | 36.202% | 63.516% |
| **-0.956** | Q(mean) q75 | 24.984% | 26.513% | 26.517% | 73.483% | 24.984% | 1.533% | 4.980% | 94.914% |
| **-0.626** | Q(mean) q95 | 5.003% | 6.403% | 6.408% | 93.592% | 5.003% | 1.405% | 0.556% | 99.441% |

### Interpretation and warnings

- Q is a discounted **cost-like return**: 0 inside the band, -1 outside, with an
  absorbing failure tail. Its theoretical continuous range is `[-100, 0]`.
  A threshold `c` is a minimum acceptable expected return, not a probability.
  More negative `c` is permissive; `c` closer to 0 is restrictive. The median
  Q(mean) is -1.776, so the q05 value -16.756 is far out in the left tail; the
  five thresholds are intentionally non-uniformly spaced in Q.
- Because the mean has distance zero, `mean selected = mean pass`. The
  closest-feasible-stochastic rescue is rare (0.28–1.53%). At permissive
  thresholds the controller is mostly the mean; at restrictive thresholds it is
  mostly the fallback max-Q over all nine.
- Decision-weighted and state-step-weighted rates differ substantially (for
  example, at `c = -0.956`, 24.98% vs 4.98% mean pass). Decision weighting is
  the expected per-decision selection rate; the state-step view shows that
  high-Q decisions are concentrated. The full sweep must report per-state
  outcomes rather than only aggregate rates.
- These rates describe the **nominal** trajectory. Persistent intervention
  changes the state distribution, so actual feasibility, fallback and mean
  rates can differ, and the frozen Q's targets are off-policy once the branch
  deviates. No safety, `Q^pi`, policy-improvement, or generalization claim is
  made.
- The pilot uses the same 128 states and protocol seeds as the confirmation
  runs, so it is explicitly an **exploratory same-state calibration**, not an
  independent outcome sample. Thresholds were chosen from Q only.
- Raw out-of-range critic outputs exist at a low rate (~0.11%); they are
  reported un-clipped and may contribute to the left tail.

### Full-sweep budget and gate

Five thresholds plus the mean control = **6 controllers**. Each controller uses
128 calls × 8 lanes = 1,024 executions/state × 128 states = **131,072
rollouts**, H500. At the existing persistent-controller rate of ~470 s
(~7.8 min) per controller, the sweep is roughly **48 min of simulation**,
plus setup/validation. No full sweep was launched. The real-environment
threshold controller has not been executed against Warp: its selection logic is
unit-tested with analytic/fake-environment cases, and it shares the rollout loop
validated by the nominal pilot. The hard gate is user approval of the five
numerical thresholds.

### Verification result

| Gate | Command | Result |
|---|---|---|
| Focused | `pytest tests/unit_tests/test_persistent_intervention.py tests/unit_tests/test_fixed_state_variance.py -q` | **17 passed** |
| Portable | `pytest -q` | **239 passed, 97 deselected** |
| Colocated gym | `pytest gym -q` | **28 passed** |
| Colocated learning | `pytest learning -q` | **7 passed** |
| Static | `ruff format` on touched files; `ruff check .` | **clean** |
| Pilot measurement | `scripts/measure_height_q_threshold_pilot.py --calls 8 --horizon 500` | **exit 0**, 29.9 s |
| Pilot summary | `scripts/summarize_height_q_threshold_pilot.py --directory ...` | **exit 0** |

Artifacts: `/tmp/opencode/height-q-threshold-pilot-20261111-v1/` contains
`call_00.pt`…`call_07.pt`, `summary.json`, `thresholds.json`,
`distribution.pt`, `per_step.csv`, `provenance.json`, `validation.json`,
`benchmark.json`, `commands.txt`, `dirty.txt`, and `code.json`. Preflight and
pilot logs are under
`/tmp/opencode/height-q-threshold-preflight-20261111-v1/`.

Status: **awaiting user approval of the five numerical thresholds.** No full
sweep, commit, todo update, plan completion, or archival movement was made.

## Extra test — rejection and closest-mean threshold sweep — 2026-11-11

Status: **EXTRA TEST COMPLETED 2026-11-11.** This appended section records the
approved follow-up to the pilot above. The §17 completion record, the archive
location, and the pilot section are unchanged. No commit, todo update, plan
completion, or archival movement was made.

### Extra scope and protocol

- Frozen `model_600.pt` and the original fair-100 PPO-data height Q only
  (`/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct`). No critic
  was retrained, selected, or tuned against outcomes.
- Same 128 fixed `k = 100` native snapshots and initial noisy observations from
  `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`; 1024 Warp
  worlds = `lane8 × state128`; native 82-field restore per call; no clamp, no
  prefix rerun, no administrative timeout reset.
- H = 500, gamma = 0.99, reward 0 inside `0.32 ± 0.05 m` and -1 outside, observed
  terminal reward plus the exact absorbing-failure tail. Seeds
  `7100000 + 10 * call`, identical to the original persistent sequence.
- Five **stochastic rejection** controllers at
  `c = [-16, -4.866, -1.776, -0.956, -0.626]`. At every decision, draw the same
  eight stochastic actions in fixed order, score all eight, execute the first
  with `Q >= c`; if none is feasible, execute the max-Q action among the eight.
  The policy mean is **not** a candidate. The rule is bounded sequential
  rejection evaluated by parallel scoring, not “highest Q among feasible” and
  not “closest to the mean”. Ties in the fallback use the lowest index.
  `c = -16` is the main-session threshold; the pilot's `-16.756` is not used.
- Two **deterministic closest-mean** controllers at `c = [-4.866, -1.776]`.
  Candidate 0 is the exact policy mean; indices 1..8 are the same eight
  stochastic draws. Among feasible `Q >= c`, select the minimum
  `sum(((a - mu) / sigma) ** 2)`; if none is feasible, take the max-Q action
  among all nine. `sigma` must be finite and strictly positive. Ties use the
  lowest index.
- No additional mean-only controller was run and no extra controller was added.
  The plain baseline and the historical eight-way max-Q best (`original`) are
  reused from `/tmp/opencode/height-q-persistent-intervention-20261110-v1/`.
- 7 controllers × 16 blocks (eight calls/block) × 128 states × 8 lanes =
  131,072 executions/controller. The documented 7-controller budget is about
  56 min; measured simulation time was 57.33 min.

### Implementation

- `learning/offline/persistent_intervention.py`: added
  `sample_rejection_candidates`, `rejection_action_indices`,
  `select_rejection_action`, and `rollout_rejection`; retained existing
  `sample_candidates`, `select_action`, `rollout`, `sample_candidates_with_mean`,
  `threshold_action_indices`, `select_threshold_action`, `rollout_threshold`,
  and `rollout_mean` behavior. `threshold_action_indices` now also returns the
  candidate distance matrix, and rollouts record selected Q, selected distance,
  effective-attempt histograms, fallback sums, and the termination step.
- `scripts/evaluate_height_q_rejection.py`: full-sweep evaluator with native
  gates, exact short negative controls, an actual eight-step CUDA controller
  smoke for every selected controller, retained per-controller block 0, the
  12-minute hard gate, resumable blocks, timestamps, code/source hashes, and
  provenance.
- `scripts/summarize_height_q_rejection.py`: simulation-free audit and summary
  producing raw returns, per-config return/threshold CSVs, pair-gain CSV,
  `results.json`, and a generated `REPORT.md`.
- `tests/unit_tests/test_persistent_intervention.py`: added five focused tests
  for rejection ordering, fallback, ties, attempt accounting, the shared
  stochastic block, and exact short negative controls (permissive rejection =
  plain, restrictive rejection = eight-way best, permissive distance = mean).
- No product training path changed; pre-existing CORL/config dirt was preserved.

### Validation and execution

- Native gates: 82-entry inventory, source/replication/lane fidelity within
  existing limits (source qpos 1.67e-6 rad, contact 7.55e-4 N; replication
  contact 5.65e-4 N), CPU/GPU Q max error 5.72e-6, split-batch 9.54e-6, zero
  argmax disagreements.
- Exact five-step negative controls: permissive rejection equals the plain
  rollout, restrictive rejection equals the historical eight-way best, and
  permissive distance selection equals the mean control. All compared tensors
  were exactly equal.
- Actual CUDA controller smoke (not the nominal pilot path): all seven
  controllers executed eight real steps over 8,192 decisions each, with finite
  returns, zero timeouts, and selection/attempt accounting equal to live
  decisions.
- Benchmarks: block 0 per controller was retained and reused by the full run.
  Projections were 489–505 s/controller, inside the 720 s gate. The initial
  benchmark harness raised `FileExistsError` because the block writer was called
  once per controller while also iterating all controllers; it was refactored to
  `write_controller_block`, the failed directory was removed, and the retained
  benchmark was rerun once. No result from the failed attempt is used.
- Full run: 112 block files, seven controllers × 16 blocks, EXIT 0. No
  `evaluate_height_q_rejection` process remained after completion. Per-controller
  simulation totals: 488.01, 488.77, 487.68, 488.42, 488.02 s (rejection) and
  499.25, 499.67 s (distance); **3439.83 s = 57.33 min**.
- Analysis exit 0. Seeds, source/checkpoint/artifact hashes, shapes, binary first
  rewards, finite returns, zero timeouts, and selection accounting were audited
  against the reused references.

### Results

Reference-return table. `Pooled SD` is ddof=1 over 131,072 individual returns
(outcome spread, not the standard error of the mean); min/max are observed values,
not bounds.

| Controller | Kind | c | Mean H500 | Pooled SD | Min | Max | Inside-band | Fall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reject_16 | rejection | -16 | **-4.6491** | 7.0608 | -100 | 0 | 96.025% | 1.880% |
| reject_4p866 | rejection | -4.866 | **-3.8862** | 6.4381 | -100 | 0 | 96.844% | 2.132% |
| reject_1p776 | rejection | -1.776 | **-3.6243** | 6.3197 | -100 | 0 | 97.123% | 2.119% |
| reject_0p956 | rejection | -0.956 | **-3.5779** | 6.3291 | -100 | 0 | 97.171% | 2.220% |
| reject_0p626 | rejection | -0.626 | **-3.5606** | 6.3168 | -100 | 0 | 97.193% | 2.200% |
| distance_4p866 | distance | -4.866 | **-3.7754** | 6.5290 | -100 | 0 | 97.051% | 1.988% |
| distance_1p776 | distance | -1.776 | **-3.6129** | 6.3856 | -100 | 0 | 97.167% | 2.060% |
| plain | reference | — | -5.2858 | 8.1216 | -100 | 0 | 95.460% | 2.155% |
| best8 | reference | — | -3.5666 | 6.3288 | -100 | 0 | 97.177% | 2.242% |

Threshold attainment per configuration (one configuration = one fixed state,
1024 executions; t interval df=1023). “CI above/below” uses the t interval;
“tail” replaces sampling uncertainty with the survivor omitted interval
`[-0.657048, 0]`. `Individual >= c` is shown only to separate individual outcomes
from expectation calibration.

| Controller | c | Mean >= c | Wilson 95% | CI above | CI below | CI uncertain | Tail above | Tail below | Tail uncertain | Individual >= c |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| reject_16 | -16 | 118/128 | [0.862, 0.957] | 118 | 10 | 0 | 118 | 10 | 0 | 91.30% |
| reject_4p866 | -4.866 | 95/128 | [0.660, 0.810] | 91 | 29 | 8 | 91 | 33 | 4 | 74.85% |
| reject_1p776 | -1.776 | 74/128 | [0.492, 0.660] | 72 | 52 | 4 | 58 | 54 | 16 | 61.92% |
| reject_0p956 | -0.956 | 58/128 | [0.370, 0.539] | 52 | 69 | 7 | 16 | 70 | 42 | 56.05% |
| reject_0p626 | -0.626 | 42/128 | [0.253, 0.413] | 34 | 78 | 16 | 0 | 86 | 42 | 52.62% |
| distance_4p866 | -4.866 | 93/128 | [0.644, 0.796] | 93 | 33 | 2 | 91 | 35 | 2 | 75.47% |
| distance_1p776 | -1.776 | 74/128 | [0.492, 0.660] | 71 | 53 | 4 | 60 | 54 | 14 | 62.76% |

Reference state means against the same candidate thresholds (context only; the
references were not assigned a threshold): plain 116, 85, 61, 39, 22 out of 128
for `-16, -4.866, -1.776, -0.956, -0.626`; best8 121, 100, 73, 59, 44.

#### Consolidated threshold attainment versus plain PPO

Each cell is the number of fixed configurations (of 128) whose 1024-rollout mean
finite H500 plus absorbing return is at least `c`. The delta is the net count
difference for rejection minus plain, not a count of rescued paired states; net
counts can hide per-configuration regressions.

| Threshold `c` | Plain PPO | Rejection | Rejection − plain | Closest-mean |
|---:|---:|---:|---:|---:|
| -16 | 116/128 | 118/128 | +2 | — |
| -4.866 | 85/128 | 95/128 | +10 | 93/128 |
| -1.776 | 61/128 | 74/128 | +13 | 74/128 |
| -0.956 | 39/128 | 58/128 | +19 | — |
| -0.626 | 22/128 | 42/128 | +20 | — |

This table has no sampling-uncertainty, multiplicity, or survivor-tail
correction; the per-configuration Student-t and survivor-tail columns above are
the corresponding uncertainty statements.

Paired gains use identical states, execution order, and seeds. The paired-tail
interval is the tested asymmetric survivor bound (reference survivor weight on
the lower side, controller survivor weight on the upper side).

| Controller | vs | Mean gain | State CI | Paired tail | Exec win | Batch16 positive |
|---|---|---:|---|---:|---:|---:|
| reject_16 | plain | +0.6367 | [0.4132, 0.9008] | [-0.0062, 1.2814] | 25.22% | 61.29% |
| reject_16 | best8 | -1.0824 | [-1.2918, -0.8839] | [-1.7248, -0.4377] | 24.57% | 17.65% |
| reject_4p866 | plain | +1.3995 | [1.0570, 1.7704] | [0.7567, 2.0426] | 44.23% | 77.58% |
| reject_4p866 | best8 | -0.3196 | [-0.4115, -0.2402] | [-0.9619, 0.3234] | 29.26% | 30.70% |
| reject_1p776 | plain | +1.6615 | [1.3150, 2.0405] | [1.0186, 2.3046] | 54.18% | 84.03% |
| reject_1p776 | best8 | -0.0577 | [-0.1114, -0.0195] | [-0.7000, 0.5854] | 27.24% | 44.79% |
| reject_0p956 | plain | +1.7079 | [1.3660, 2.0818] | [1.0650, 2.3504] | 56.51% | 85.45% |
| reject_0p956 | best8 | -0.0112 | [-0.0272, 0.0016] | [-0.6536, 0.6312] | 23.38% | 48.40% |
| reject_0p626 | plain | +1.7252 | [1.3837, 2.0979] | [1.0823, 2.3678] | 56.86% | 85.78% |
| reject_0p626 | best8 | **+0.0060** | [0.0005, 0.0118] | [-0.6363, 0.6486] | 17.84% | 49.85% |
| distance_4p866 | plain | +1.5103 | [1.1582, 1.8868] | [0.8675, 2.1543] | 54.63% | 83.42% |
| distance_4p866 | best8 | -0.2088 | [-0.3701, -0.0708] | [-0.8511, 0.4352] | 32.92% | 44.92% |
| distance_1p776 | plain | +1.6729 | [1.3222, 2.0502] | [1.0300, 2.3164] | 56.49% | 85.99% |
| distance_1p776 | best8 | -0.0463 | [-0.1532, 0.0175] | [-0.6886, 0.5972] | 34.16% | 49.72% |
| best8 | plain | +1.7191 | [1.3779, 2.0907] | [1.0763, 2.3615] | 56.91% | 85.49% |

Selection diagnostics. Effective attempt 1..8 is the first feasible draw index;
an all-infeasible fallback is counted as attempt 8. For rejection, the mean is
not sampled and mean feasibility is not defined. For distance, the mean-feasible
rate equals the mean-selected rate.

| Controller | Fallback | Mean feasible | Mean selected | First draw | Attempt mean | Selected Q |
|---|---:|---:|---:|---:|---:|---:|
| reject_16 | 4.608% | — | — | 95.633% | 1.331 | -4.2570 |
| reject_4p866 | 23.452% | — | — | 78.743% | 2.661 | -4.1565 |
| reject_1p776 | 48.289% | — | — | 56.483% | 4.413 | -4.1051 |
| reject_0p956 | 72.683% | — | — | 34.474% | 6.138 | -4.1054 |
| reject_0p626 | 92.907% | — | — | 16.898% | 7.552 | -4.0836 |
| distance_4p866 | 23.122% | 76.264% | 76.264% | — | — | -4.1004 |
| distance_1p776 | 47.918% | 51.058% | 51.058% | — | — | -4.0964 |

Effective-attempt histograms (attempts 1..8; the last bucket includes fallback):

- reject_16: `[.951, .001, .001, 0, 0, 0, 0, .046]`
- reject_4p866: `[.758, .003, .001, .001, .001, .001, 0, .235]`
- reject_1p776: `[.504, .005, .002, .002, .001, .001, .001, .483]`
- reject_0p956: `[.254, .008, .004, .002, .002, .001, .001, .728]`
- reject_0p626: `[.053, .007, .004, .002, .002, .001, .001, .930]`

### Interpretation

- **Actual expected return versus the fixed threshold.** In aggregate, the
  expected H500/absorbing return meets its threshold at `c = -16`
  (-4.6491 >= -16) and `c = -4.866` (-3.8862 >= -4.866). It does **not** meet
  the fixed threshold at `c = -1.776`, `-0.956`, or `-0.626` (-3.6243, -3.5779,
  -3.5606 are all below their c). Per configuration, the counts are 118, 95,
  74, 58, and 42 of 128. A fixed per-decision `Q >= c` therefore does not imply
  that the realized expected total reaches `c`; this is an empirical
  intervention result, not a maintained budget.
- **`c = -16` is near plain empirically, not identically.** Rejection keeps the
  first draw in 95.6% of decisions, falls back in 4.6%, and improves the mean over
  plain by +0.637. The matched tails include zero, and 10/128 configurations are
  confidently below -16. This is not a guarantee that the permissive
  rejection rule reproduces the plain policy.
- **`c = -0.626` is mostly the eight-way best empirically, but not the same
  rule.** It falls back in 92.9% of decisions and selects the first feasible draw
  in 16.9% (rather than the highest-Q feasible draw). Its mean is -3.5606 versus
  -3.5666 for best8, a +0.0060 difference whose state CI barely excludes zero
  while the paired tail and 49.85% batch16-positive fraction do not resolve a
  difference. The feasible-first difference is real at the decision level even
  where the aggregate is nearly tied, and the actual rates above should be used
  rather than saying the controller “is” best8.
- **Restrictive rejection does not improve threshold attainment over best8.**
  At the same `c`, best8 clears the state-mean threshold 100/128, 73/128, 59/128,
  and 44/128 for `-4.866, -1.776, -0.956, -0.626`; rejection clears 95, 74, 58,
  and 42. The aggregate mean is slightly higher at the tightest threshold, but
  the per-configuration attainment is not uniformly better, so attainment need
  not be monotone in `c` or implied by the aggregate gain.
- **Closest-mean versus rejection.** The two deterministic controllers are not
  clearly superior: at `c = -4.866`, distance has the better mean (-3.7754 versus
  -3.8862) but slightly lower state-mean attainment (93 versus 95 of 128); at
  `c = -1.776` the means are close (-3.6129 versus -3.6243) and attainment ties
  at 74/128. The exact mean is selected whenever feasible (76.3% and 51.1% of
  decisions).
- **Secondary behavior.** Inside-band occupancy is 96.0–97.2% versus 95.46%
  plain; fall fractions are 1.88–2.22% versus 2.16% plain and 2.24% best8. XY
  tracking RMSE is 0.436–0.456 and speed/command 0.736–0.766 across controllers,
  close to the previously reported height-only trade-off. These are simulation
  summaries at 128 fixed states, not a safety or hardware claim.
- **Scope of the statistics.** The threshold intervals are per-configuration
  t intervals conditional on the fixed state and are not independent
  state-population replication. Multiple thresholds and controllers were
  examined; the Wilson intervals are descriptive and not multiplicity-corrected
  guarantees. Min/max and q05 are observed sample values, not bounds. The
  surviving-rollout tail is separate from sampling uncertainty. The critic was
  trained on nominal-policy data; persistent deviations and fallback violations
  move it off-policy, so selected Q is conditional on the executed policy, not
  `Q^pi` truth.

### Artifacts, commands and verification

Output root: `/tmp/opencode/height-q-rejection-threshold-20261111-v1/`
(112 block files, `raw_returns.pt`, `per_config_returns.csv`,
`per_config_threshold.csv`, `pair_gains.csv`, `results.json`,
`benchmark.json`, `timings.jsonl`, `validation.json`, `provenance.json`,
`code.json`, `commands.txt`, `run_audit.json`). The generated report moved to
the [durable copy](../supplements/replay_relabel_q/rejection-threshold-20261111-v1/REPORT.md).

Inputs: source insertion SHA-256
`dcfca737603a5e238feecd5fceae643b426c3d99af5c8ab79df3acf6e9dbf712`;
checkpoint SHA-256
`f88b1f9b829675c97ed0c420202cf54456dc414641def34e379e39407ee3adf6`;
artifact `manifest.json`/`q.pt` SHA-256
`db267188c3f8cfe5cfe9d54638e318c387537d3320c799374d557fcbb90de0e9` /
`e59d738a606979edeb021c86651e78f632bd4b78a99a9a24e80ebecccdad2236`;
code revision `dfc506e449eedad5f57bb208e38b4dca69977e3e`.

Commands (all foreground, output redirected, true exit captured):
benchmark `... evaluate_height_q_rejection.py ... --blocks 1` EXIT 0; full
`... --blocks 16 --resume` EXIT 0; summary
`... summarize_height_q_rejection.py /tmp/opencode/height-q-rejection-threshold-20261111-v1`
EXIT 0. Logs:
`/tmp/opencode/height-q-rejection-benchmark-20261111.log`,
`height-q-rejection-full-20261111.log`,
`height-q-rejection-analysis-20261111.log`.

| Gate | Command | Result |
|---|---|---|
| Focused | `pytest tests/unit_tests/test_persistent_intervention.py tests/unit_tests/test_fixed_state_variance.py -q` | 21 passed, EXIT 0 |
| Portable | `pytest -q` | 243 passed, 97 deselected, EXIT 0 |
| Colocated gym | `pytest gym -q` | 28 passed, EXIT 0 |
| Colocated learning | `pytest learning -q` | 7 passed, EXIT 0 |
| Warp | `pytest -q -m warp` | 32 passed, 308 deselected, 18 pre-existing warnings, EXIT 0 |
| Static | `ruff check .`; `ruff format --check` touched files; `git diff --check` | clean, EXIT 0 |

No threshold was selected or tuned from realized gains. No commit, todo,
plan-completion, or archival movement was made; unrelated dirty work is
preserved.

## Extra test — original PPO reward sacrifice versus height-Q gain — 2026-11-11

### Extra scope and protocol

User-approved EDIT, run in the foreground. Question: at the same 128 fixed k=100
insertion states, does replacing the plain `model_600` stochastic draw
(candidate 0 of 8) with the original fair-100 height-Q max-of-8 controller
reduce the **original PPO training reward**, and how does that compare with its
height-band gain? The original reward is the primary hypothesis; height is the
same-execution secondary objective. The output must state whether there is a
sacrifice or an improvement without presuming either.

- **Controllers (new, paired):** `plain` executes candidate 0 of the same eight
  stochastic draws every step; `original` executes the original height-Q argmax
  of the same eight draws. No reused baseline blocks: both controllers are
  recollected on the same paired seeds so both objectives come from identical
  executions.
- **States and executions:** the 128 fixed insertion states from
  `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`
  (SHA-256 `dcfca737603a5e238feecd5fceae643b426c3d99af5c8ab79df3acf6e9dbf712`),
  replicated to 1024 Warp worlds as 128 states x 8 lanes (source-world ids
  `(torch.arange(128) * 8).repeat(8)`).
- **Seeds:** `7100000 + 10 * call`, `call = 0..127`; per call `env=seed`,
  observation noise `seed+1`, action noise `seed+2`. 16 blocks x 8 calls, 64
  executions per state per block, 1024 executions per state per controller.
- **Horizon and controls:** H500, gamma from the loaded training config
  (`algorithm.gamma`, expected 0.99), `dt = env.dt` (expected 0.02), no admin
  timeout: `episode_length_s = max(episode_length_s, 601 * ctrl_dt + 1)`, and
  any `timed_out` is a hard failure. All 82 native snapshot entries (physics
  state, controller history, DR generators, buffers) are restored; the short
  same-controller negative control must be exact and the height return must
  match the existing persistent-intervention rollout on a 5-step paired probe.
- **Original objective:** per transition, exactly the runner dispatch
  `mask * ((weight * dt) * reward_function())` with ordinary weights masked by
  `~env.terminated` and `termination_weight` masked by `env.terminated`; return
  `sum_t gamma^t * r_t`, with every reward zero after true termination. No
  artificial absorbing `-100` tail. Per-term discounted sums, their sum, the
  final live mask, live-step count and termination step are recorded per
  execution. A terminal step may contribute only the termination term.
- **Height objective:** unchanged binary `0` inside `0.32 +/- 0.05 m`, `-1`
  outside, with the existing discounted absorbing-failure tail; same trajectory
  and alive mask as the original objective.
- **Reward dispatch:** one shared non-mutating helper used by the rollout, with a
  parity unit that calls the real `OnPolicyRunner.update_rewards_dict` on a real
  environment step. No hand-written duplicate reward formula left untested.
  The actual loaded config, nonzero weights, gamma, dt, control/sim frequencies
  and decimation are recorded and hashed; no guessed defaults.
- **Pilot and runtime gate:** retain block 0 for each controller, report its
  seconds and projected 16-block controller time. If a projected controller
  time exceeds 12 minutes (720 s), write a runtime blocker and stop rather than
  launch the remaining blocks.
- **Analysis:** per controller original mean +/- pooled execution SD and
  min/max, 128-state mean SD; paired `original - plain` gain for original and
  height, state-cluster bootstrap CI (fixed-state working independence caveat),
  paired Student-t on the 128 per-state mean gains (df 127, original objective
  primary); percent original gain relative to the plain baseline where that
  baseline is positive; per-term paired contributions verifying exact sum;
  execution win/tie/loss and SD; 64 contiguous 16-execution batches per state
  with positivity and batch SD; per-config CSV for both objectives. The state
  bootstrap and t-test treat the 128 fixed configurations as independent
  working units, not a random state population.
- **Reporting boundaries:** the original finite-horizon return omits a
  continuing tail and is **not** bounded by the height `[-0.657, 0]` interval;
  that bound applies only to the height objective. H500 is twice the 250-step
  training episode and is an extension; zeroing rewards after true termination
  is deliberate. No `Q^pi`, policy-improvement, safety, hardware, or population
  claim. H150 is not approved and is not run.
- **Artifacts:** new unique directory
  `/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/`; 32 raw block
  files (16 per controller), `provenance.json`, `code.json`, `commands.log`
  with true exits and start/end times, `validation.json`, `timings.jsonl`,
  summaries and CSVs. No full per-step state tensors; optional diagnostics stay
  small. The generated report is relocated to
  `BigRedGym/docs/plans/supplements/replay_relabel_q/original-reward-tradeoff-20261111-v1/REPORT.md`
  with a compatibility symlink at the `/tmp` path.
- **Constraints:** no commits, no todos, no plan archive movement, no edits to
  other reports; preserve all unrelated dirty playback work. At the end audit
  complete block counts, commands/EXIT statuses, and live processes.

### Extra implementation and unit tests

Add `learning/offline/reward_tradeoff.py` with a non-mutating training-reward
dispatcher/model and a dual-objective rollout that reuses the existing
sample-eight selection and height-tail helper. Refactor
`BaseRunner.setup_reward_functions` to use the shared reward-function map so the
experiment and runner cannot drift. Add
`tests/unit_tests/test_reward_tradeoff.py` covering weight/termination masks,
dt scaling, terminal-only termination term, zero after terminal, per-term sum
equals total, malformed dispatch, and exact parity with the real
`OnPolicyRunner.update_rewards_dict`. Add
`scripts/evaluate_height_q_reward_tradeoff.py` and
`scripts/summarize_height_q_reward_tradeoff.py`. Preserve existing public
rollout APIs and their tests.

### Extra execution and verification — 2026-11-11

Foreground runs, true exits captured in
`/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/commands.log`.

- Pilot (block 0 only, retained): `plain 33.0978 s` and `original 33.9174 s`;
  projected controller times `529.564 s` and `542.678 s`, each below the
  `720 s` gate. No runtime blocker was written.
- Full `--blocks 16 --resume`: 32/32 blocks complete, EXIT 0. Wall timing sum
  `1083.072 s`; plain 16 blocks `536.628 s`, original 16 blocks `546.444 s`.
- No live evaluator/simulator process remained after the run.

Protocol gates in `validation.json` (pilot) and `validation_resume.json` (full):

- 82 native snapshot entries restored; source-fidelity maxima
  `root 8.52e-5`, `dof_pos 2.03e-6`, `dof_vel 1.03e-4`,
  `base_ang_vel 8.52e-5`, `contact 7.71e-4`, all below the existing limits.
- Lane replication and 5-step lane fidelity passed with the same limits.
- CPU/GPU Q maximum error `5.72e-6`, batched split error `9.54e-6`, zero
  argmax disagreements.
- One real post-step comparison of all 12 configured terms against the actual
  `OnPolicyRunner.update_rewards_dict` was **bitwise exact**; reward totals
  finite (mean `0.272688`, range `0.184132..0.317855`).
- H5 height parity against the existing persistent rollout was exact for
  height return, steps, survival and selection for both controllers. Repeated
  same-controller runs matched discrete selections/steps/height exactly;
  continuous maximum differences were original return `1.54e-4`, original
  terms `1.15e-4`, actions `0.01894`, Q-gain sum `1.29e-3`. Solver scratch is
  not part of the snapshot, so these are numeric tolerances, not a bitwise
  trajectory claim.
- Every stored block has 128 states x 64 executions, the exact paired seed
  list, zero timeouts, candidate0-only plain selection, and per-term sums
  bitwise equal to the stored original return.

### Extra results — 2026-11-11

| Quantity | Plain | Original height-Q | Paired gain |
|---|---:|---:|---:|
| Original PPO return mean | 26.977067 | 26.950118 | **-0.026949** |
| Original pooled execution SD | 1.595573 | 1.551451 | 0.854399 |
| Original 128-state mean SD | 1.244362 | 1.207611 | 0.070546 |
| Height return mean | -5.283413 | -3.565125 | **+1.718288** |
| Height pooled execution SD | 8.115789 | 6.323508 | 5.456443 |

- Original paired state-cluster bootstrap CI95: `[-0.038765, -0.014482]`;
  paired Student t(127) = `-4.321855`, p = `3.10e-5`,
  t CI95 `[-0.039288, -0.014610]`. Height paired state-cluster CI
  `[+1.376687, +2.090214]`; t(127) = `9.348230`.
- Execution original win/tie/loss = `55741 / 0 / 75331`
  (`0.425270 / 0 / 0.574730`). 16-execution batches positive/tie/negative
  = `2848 / 0 / 5344` (`0.347656` positive); 0/128 states all-positive,
  31/128 majority-positive. Height batch positive fraction `0.857300`,
  22/128 all-positive.
- The plain original baseline is positive, so the paired original sacrifice is
  `-0.0999%` of that baseline.
- Per-term original deltas (exact sum `-0.026949`): tracking_lin_vel
  `-0.061062`, tracking_ang_vel `-0.010746`, trot_support `+0.017745`,
  reference_traj `+0.011764`, swing_contact `+0.007398`, min_base_height
  `+0.005047`, orientation `+0.002156`, action_rate2 `+0.000341`,
  action_rate `+0.000251`, ang_vel_xy `+0.000151`, torques `+0.0000043`,
  termination `-1.00e-8`. The negative delta is dominated by linear-velocity
  tracking; several penalties and gait-shaping terms improve slightly.
- Terminal fraction plain `0.021370`, original `0.022362`; mean live steps
  `496.098` vs `495.882`. XY RMSE `0.449373` vs `0.456089`; yaw RMSE
  `0.771272` vs `0.768642`; speed/command `0.742534` vs `0.736196`.
- Prior persistent height aggregate drift: plain current - prior
  `+0.002361`, state-mean correlation `0.999996`; original `+0.001512`,
  correlation `0.999992`. This is a reproducibility check, not a bitwise
  match claim.

Interpretation: at these fixed states the original height-Q controller buys a
**+1.7183** height-band gain at a small but detectable **-0.0269** original
PPO-return cost (-0.10% of a positive baseline). The original loss is not a
uniform per-execution loss (42.5% of executions win) but 57.5% lose and only
34.8% of 16-execution batches are positive. This is fixed-state, finite-H500,
paired evidence; it is not a state population, policy-improvement theorem,
`Q^pi` validation, safety result, or training-horizon claim. H500 is twice the
250-step training episode; all original rewards are zero after true
termination and no absorbing `-100` tail is used. The height `[-0.657, 0]`
bound is not applied to original reward.

### Extra verification gates

| Gate | Command | Result |
|---|---|---|
| Focused portable | `pytest tests/unit_tests/test_reward_tradeoff.py tests/unit_tests/test_persistent_intervention.py -q` | 22 passed, EXIT 0 |
| Portable | `pytest -q` | 300 passed, 97 deselected, EXIT 0 |
| Colocated gym | `pytest gym -q` | 28 passed, EXIT 0 |
| Colocated learning | `pytest learning -q` | 7 passed, EXIT 0 |
| Warp | `pytest -q -m warp` | 32 passed, 365 deselected, 18 warnings, EXIT 0 |
| Static | `ruff check .`; `ruff format --check` touched files; `git diff --check` | clean, EXIT 0 |

Durable report:
[`docs/plans/supplements/replay_relabel_q/original-reward-tradeoff-20261111-v1/REPORT.md`](../supplements/replay_relabel_q/original-reward-tradeoff-20261111-v1/REPORT.md)
(moved report with compatibility symlink at the original `/tmp` path; raw data
stays in `/tmp`). No commit, todo, plan-completion, or archive movement; prior
completion record and unrelated dirty playback work were not modified.

Additional sign detail: of the 128 per-state original-return mean gains, 35 are
positive and 93 are negative (median `-0.043125`, q05 `-0.115500`, q95
`+0.103518`, min `-0.178172`, max `+0.279470`). The negative aggregate is
therefore not carried by a few outlier states; it is the majority sign at the
state level. Per-state execution SD has mean `0.667161` (range
`0.173981..3.043307`).

End audit (`/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/commands.log`,
EXIT 0): 16 plain and 16 original blocks, exact paired seed lists, zero
timeouts, 12-term sums bitwise equal to original returns, no runtime blocker,
no live evaluator/process, and the durable report symlink resolves to
`BigRedGym/docs/plans/supplements/replay_relabel_q/original-reward-tradeoff-20261111-v1/REPORT.md`.
