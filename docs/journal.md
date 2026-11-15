# Dev Journal

Development log.
Each entry records a plan-action cycle.

## [2026-11-10] Offline replay and height Q

### Context
Implemented the approved plan for bounded PPO replay recording, height reward relabeling, fixed-policy FQE, artifact export, and fresh-rollout validation. The first experiment is one height reward with a fixed bound; pitch/roll, joint velocity, bound conditioning, MPPI, intervention, and finite-horizon Q remain deferred.

### Plan
1. Add offline contracts, bounded recorder, snapshot loader, height relabeler, and nested/early-late dataset selection.
2. Load the frozen stochastic PPO actor and fit a continuing-return FQE critic with absorbing failure and timeout bootstrapping.
3. Export a runtime Q artifact using actor observations and candidate actions only.
4. Evaluate against fresh simulator continuations with reproducible initial states, candidate actions, threshold reports, and truncated-tail intervals.
5. Run focused tests, full gates, and a CPU wiring smoke.

#### Files to edit
| File | Change |
|------|--------|
| `learning/offline/`, `learning/algorithms/fqe.py`, `learning/storage/transition_recorder.py` | New offline contracts, recorder, dataset, reward, policy, FQE, artifact, evaluation modules |
| `learning/runners/on_policy_runner.py`, `scripts/train.py` | Opt-in recording, pre-reset capture, snapshot cadence |
| `scripts/train_height_q.py`, `scripts/evaluate_height_q.py` | Offline fitting and validation CLIs |
| `tests/unit_tests/test_offline_replay.py`, `tests/unit_tests/test_height_fqe.py` | Portable integrity and analytic tests |
| Docs and skills | `docs/offline_height_q.md`, README, CORL install, test catalog, MuJoCo limitations, Q2 skills |

### Result
✅

Implemented the pipeline and documented it. Focused tests pass (17), full portable gate passes (221 passed, 97 deselected), and colocated suites pass (27 + 7). A one-iteration CPU smoke recorded a 128-transition snapshot, trained 10/50/100% Q artifacts, and produced a fresh-rollout evaluation JSON.

#### Files changed
| File | Change |
|------|--------|
| `learning/offline/__init__.py`, `contracts.py`, `dataset.py`, `rewards.py`, `nominal_policy.py`, `training.py`, `q_artifact.py`, `evaluation.py` | New offline pipeline |
| `learning/storage/transition_recorder.py` | Bounded recorder and atomic snapshots |
| `learning/algorithms/fqe.py` | Q network and fixed-policy FQE |
| `learning/runners/on_policy_runner.py` | Opt-in recording hook, pre-reset successor/height capture, burn-in first-transition exclusion |
| `scripts/train.py`, `scripts/train_height_q.py`, `scripts/evaluate_height_q.py` | CLI entry points |
| `tests/unit_tests/test_offline_replay.py`, `tests/unit_tests/test_height_fqe.py` | Regression tests |
| `docs/offline_height_q.md`, `README.md`, `docs/thirdparty/CORL/install.md`, `tests/TEST_CATALOG.md`, `genAI_skills/README_MUJOCO.md`, Q2 skill files | Documentation and skill updates |

### Notes
- Snapshots are immutable and never deleted automatically; a last-N ring does not preserve early data after wrap.
- Q uses actor observations only; hidden physical state can bias it.
- The smoke is software wiring evidence, not Q accuracy or policy quality.
- A CUDA recording smoke exposed a CPU/CUDA device mismatch in episode-counter updates; `dones` is now moved to CPU for those counters. The retest passed.
- Distribution collection: early run `Nov10_10-35-07_gpu-early-50-trot-hist-20261110` (3,276,800 transitions, 1.5 GiB), final resumed run `Nov10_10-36-22_gpu-final-600-trot-hist-20261110` (1,000,000 transitions, 449 MiB), and equal-size 2M mix at `logs/go2trot/hist-mix-20261110/replay/replay_iter_0000600`. Plot and quantiles: `/tmp/opencode/height-distribution-20261110.{png,json}`.
- Phase analysis: phase is uniform; final-policy height has a phase-locked mean amplitude of ~1.8 cm. A 0.32 ± 0.05 m bound has 5.3% outside, 3.2–8.4% by phase; a ±0.02 m bound has 38.3% outside, 31.0–47.7% by phase. Outputs: `/tmp/opencode/phase-height-20261110.{png,json}` and `/tmp/opencode/phase-bound-sensitivity-20261110.json`.
- ±0.05 experiment: trained 10/50/100% episode-nested critics on the final package with reference 0.32 m, bound 0.05 m. Equal-epoch (40k/200k/400k) evaluation: MAE 3.424/3.579/3.204, candidate-ranking accuracy 0.507/0.543/0.464. Q candidate range shrank 1.10→0.59→0.41 versus measured return range ~4.1. The data-efficiency hypothesis is not established, and the critic is nearly action-insensitive; do not use it for MPPI/filter integration yet.
- Evaluation exposed a CPU/CUDA candidate-action device mismatch; fixed by moving the candidate to the environment device. Full portable gate still passes (221 passed, 97 deselected).
- Horizon validation on the fair 100% critic: aggregate MAE 3.216/3.331/3.308/3.322 at H=250/500/750/1000, which is stable within 3.3%. But late terminations appear only after 250 steps (0% → 3.1% → 4.4%); paired late-failure MAE worsens from 5.08 to 9.45, and Q is pessimistic on survivors but optimistic on late failures. H=250 is not sufficient for continuing-return validation; use H≥500. Q-tail containment falls from 61.9% to 0% as H grows.
- Evaluation-distribution Q experiment: collected 254,927 H=500 transitions from 512 evaluation rollouts (16 trials, 4 reps, 8 local candidates), then trained a new FQE critic with the same policy/reward/gamma (100k updates). On held-out trials 100–104 at H=500, MAE fell from 2.653 (original) to 1.300 (eval-distribution), and bias fell from −1.83 to −0.36. Candidate ranking stayed near chance (0.550 → 0.486), and Q range across candidates (0.58) remained far below return range (1.96). Distribution shift explains calibration, not action discrimination.
- The eval-distribution training data had only 6 terminal transitions; the new evaluation had none. Late-failure coverage remains unresolved.
- Added `scripts/evaluate_height_q_batched.py`: one 1024-world env per candidate batch, same trial states across candidate batches/repetitions, default 1024 trials × 4 reps × 8 candidates at H=500.
- Batched 1024-trial evaluation: original Q MAE 2.534, bias −0.349, ranking 0.503, Q range 0.387, return range 2.807; eval-distribution Q MAE 2.972, bias +0.963, ranking 0.495, Q range 0.618, return range 2.771. The earlier 5-trial eval-distribution advantage (MAE 1.30 vs 2.65) did not hold; the eval-distribution critic overfit its 16 training trials. Both critics are at chance on candidate ranking; the original is better calibrated on the fresh 1024-trial batch. Threshold false accepts at −5: 323 (original) vs 857 (eval-distribution).
- One-step lookahead: original MAE 2.525, bias −0.451, ranking 0.5095, score range 0.54; eval-distribution MAE 2.841, bias +0.518, ranking 0.5066, score range 0.87. Lookahead improves calibration modestly (especially eval-distribution, false accepts at −2 drop 31→0) but ranking remains chance. Not a fix for action discrimination.
- Two-step lookahead: original MAE 2.495, bias −0.334, ranking 0.5337, score range 0.596; eval-distribution MAE 2.779, bias +0.485, ranking 0.4983, score range 0.663. Two-step is the best tested score for the original critic (ranking 0.504→0.508→0.534), but still weak; eval-distribution calibration improves without ranking recovery. Threshold false accepts at −5: original 323 direct → 283 two-step; eval-distribution 862 → 576.
- Reset transient: the reset pose (`root z = 0.40`, partially extended legs) is above the `[0.27, 0.37]` band. The first recorded transition is 0.3939 m, and the height enters the band only at `step_id` 3 (99.97%). So every episode starts with 2–3 unconditional `-1` rewards (≈ −2.97 discounted) that no candidate action can change; the first two steps are ballistic (spread < 1.2 cm across 4096 envs). About 1.2 of the 5.3 percentage points of out-of-bound transitions are this transient; steady-state is ~4.1%.
- Q input confirmed to hold no height channel (48-dim actor obs). Height is nevertheless highly recoverable from it: an MLP on the 48 obs predicts post-action height with MAE 0.68 cm and R² 0.874 (test height std 2.57 cm); gravity alone is weak (R² 0.09), joint angles carry the most (0.378 linear).
- Evaluation states are in support but under-covered: standardized 48-dim NN distance median 0.55 versus 2.62 for held-out training rows, 0% beyond the training p99, but 100% of nearest neighbours are `step_id ≤ 2` reset-transient rows, which are 1.19% of the 1M snapshot.
- State-wide Q check on 512 eval-distribution rollouts (254,927 transitions): R² 0.511, corr 0.733, MAE 3.17, bias −0.79. At the reset states (`step_id` 0–2) bias is only −0.28, better than steady state (−0.98), so the Q is not broken specifically at the reset cluster. A first version of this analysis had a return-recursion bug (tail multiplied instead of accumulated); the numbers above are from the corrected version, cross-checked against the 4.68% out-of-bound rate.
- The ranking measurement is noise-dominated: repetition std 2.15 versus within-trial candidate-mean std 0.95; within-trial split-half reliability 0.555 (Spearman–Brown 0.714 for 4 reps); variance shares 78% between-trial / 4.2% between-candidate / 18% repetition. A perfect predictor scored against the 4-rep measured means reaches only 0.678 pairwise accuracy; ceilings are 0.717/0.757/0.792/0.824 at 8/16/32/64 reps. The two-step score of 0.534 is below that floor but the test cannot separate a good critic from a mediocre one at 4 reps.
- Added `--insert_step k` to `scripts/evaluate_height_q_batched.py`: the frozen policy runs `k` steps from reset, then the candidate actions are inserted and the H-step return is measured from that mid-gait state. Prefix seeds depend only on the repetition, so all candidates of a repetition share one paired state; the script raises if the insertion observation is not reproduced (`rtol=1e-3, atol=1e-4`, because MuJoCo Warp reproduces a 20-step contact trajectory only to ~2e-4 in velocity). Worlds terminating during the prefix are excluded and counted. `episode_length_s` now covers `k + H`. `insert_step = 0` reproduces the previous smoke metrics exactly (0.5223 ranking, 4.2254 MAE).
- Prefix determinism check: two independent 20-step prefixes differ by 8.6e-7 m in height. At `k = 20` the insertion state has height 0.3149 m, 96.9% inside the band, and the insertion-step reward is −1 in 1.6% of cases versus 100% at reset.
- Added `Go2Trot.capture_state()` / `Go2Trot.restore_state()` (DOF, root, command, phase, phase frequency, target, episode counter, common step counter; backend commit; clears per-episode flags). Unit test in `gym/envs/go2/test_go2trot.py`.
- Prefix replay does not reproduce the state: two identical replays diverge by 1.6e-6 m height at k=20, 2.6e-2 m at k=50, and 2.5e-2 m at k=100, with `dof_vel` differing by 47 at k=100. Isolation showed the policy network and env construction are bit-reproducible; with a constant action physics drifts only 2.4e-5 over 100 steps. The divergence is Warp solver noise amplified by the closed policy loop, so the evaluator snapshots once per repetition and restores instead of replaying.
- Restore is exact and repeatable: two restores plus one step differ by 2e-5 in observation and 3e-8 m in height. Restore clamps joints to asset limits (up to 8e-3 rad); teleport fidelity vs a true continuation after one step is height 2.3e-5 m, `dof_pos` 8e-3, `dof_vel` 1.2 rad/s.
- 16-repetition control at `insert_step = 0`: direct 0.4985 / one-step 0.5212 / two-step 0.5395, MAE 2.326/2.317/2.297. The 16-rep ceiling is 0.686, so the reset-state ranking is genuinely poor, not measurement-limited.
- Mid-gait results (`insert_step = 100`, 16 reps, 1024 trials): original critic direct 0.5179 / one-step 0.5220 / two-step 0.5348 (MAE 2.045/2.016/1.904); eval-distribution critic direct 0.5053 / one-step 0.5192 / two-step 0.5234 (MAE 2.494/2.384/2.276). Ceiling at this insertion point is 0.655, lower than at reset because the mid-gait continuation noise is larger (repetition sd 5.02 vs 3.00). Trial-level bootstrap: all two-step results exclude 0.5, all are far below the ceiling.
- Mid-gait insertion did not fix ranking: two-step 0.5395 → 0.5348. The critic recovers only 11–22% of the achievable margin over chance at both insertion points. One-step lookahead at k=100 is the best-calibrated configuration tested (bias +0.029 vs +0.714 direct) but does not improve ranking.
- First-step reward distribution: 100% −1 at k=0 (insertion height 0.3961 m); 1.17% −1 at k=100 (insertion height 0.3203 m), identical for all eight candidates in every trial. The immediate reward carries no candidate information at either insertion point.
- k=100 insertion states are a representative sample, not a pocket: per-dimension obs std matches the training set to within 5% on 7 of 8 channels (`projected_gravity` 0.67 because training contains falls); nearest training neighbour median 2.52 versus 2.62 for held-out training rows and 0.55 for the reset cluster; pairwise spread among the 1024 states 9.10, identical to 400 random training states. Height std 2.16 cm over a 0.193–0.402 m range.
- Full gates pass: 222 portable, 28 `gym`, 7 `learning`; Ruff clean.
- No todos were created or updated. No commit was made.

## [2026-11-10] Fixed mid-gait state batch variance

### Context
User approved EDIT on dirty trees and the full 128-state × 8-candidate ×
8-batch × 16-continuation experiment after restoration validation and batch0
benchmark. Prior journal claims of exact restore, noise ceilings, resolved
mid-gait ranking, and first-reward independence are superseded: the evaluator
changed insertion states across repetitions and restored incomplete/clamped
state; estimated means/sorted repetitions do not identify a noise ceiling.

### Plan
1. Add isolated native Warp snapshot and fixed-state evaluator, preserving prior work.
2. Validate pairing/statistics and native one-step restoration/replication fidelity.
3. Benchmark one 16-continuation batch; reuse it as batch0 of eight.
4. Save raw axes, full insertion provenance, paired statistics and both critic scores.

#### Files to edit
| File | Change |
|---|---|
| `BigRedGym/learning/offline/fixed_state_variance.py` | Snapshot/paired noise/statistics |
| `BigRedGym/scripts/evaluate_height_q_variance.py` | Fixed protocol and checkpointed batches |
| `BigRedGym/tests/unit_tests/test_fixed_state_variance.py` | Analytic and fake-env regressions |
| Plan, offline docs, test catalog, MuJoCo limitations | Correct evidence boundaries and results |

### Result
✅ Implemented and ran all eight batches after native fidelity gates; batch0
benchmark retained (56.8191 s), total 459.0989 s. Output:
`/tmp/opencode/height-q-fixed-variance-20261110-v1/REPORT.md` with exact commands
in `commands.txt`, full raw `[8,16,128,8]` returns, first rewards and tail bounds,
snapshot/controller/domain inventory, direct scores for both critics and
per-state/action CSV/statistics.

Mean repeated-16-return estimator variance 1.2042; within-batch action variance
0.7671; paired nonnominal contrast mean -0.0654 and estimator variance 1.2091
(covariance retained). Independent-batch pair-ranking agreement 0.51798; winner
agreement 0.14509. Original/eval-dist accuracy against noisy pooled estimates
0.51618/0.50781; MAE 1.6904/2.1407. No noise ceiling or critic-only bottleneck
claim. Raw first-reward negative rate 7.03125%; terminal fraction 2.12021%;
surviving omitted tail [-0.6570483,0].

Validation: 5 focused, 227 portable, 28 gym, 7 learning, 32 Warp tests pass;
Ruff clean. Warp contract fixtures emitted 18 MULTICCD support warnings; the
experiment did not. Exact pre-step buffer restoration and native one-step
source/replica gates pass (qpos max 1.43e-6 rad, qvel 7.16e-5 rad/s).

#### Files changed
| File | Change |
|---|---|
| `BigRedGym/learning/offline/fixed_state_variance.py` | Isolated 1024-world native snapshot, paired noise, covariance-aware statistics |
| `BigRedGym/scripts/evaluate_height_q_variance.py` | Fixed insertion/candidates, benchmark/resume batches, both critics |
| `BigRedGym/scripts/summarize_height_q_variance.py` | Offline audits, full tail bounds, per-state/action CSV/report |
| `BigRedGym/tests/unit_tests/test_fixed_state_variance.py` | Indexing, pairing, history/flags/counters/RNG restoration and analytic return tests |
| Plan, offline docs, test catalog, MuJoCo limitations, testing/train skills | Superseded claims and corrected workflow/results |

### Notes
- Native snapshot preserves six out-of-limit joint positions across five states
  (maximum excess .210443 rad); no restore clamping. This is source simulator
  constraint penetration, retained faithfully. No bitwise long-horizon claim.
- Initial resume failed on NumPy scalar config metadata; converted metadata only,
  retained original file/tensors/batch0, then resumed successfully.
- Eight batches give only seven degrees of freedom for variance estimates.
- Prior unrelated changes preserved; no commits or todos; plan remains active
  in EDIT, without completion/archive.

## [2026-11-10] Full variance report merge and 64-rollout reanalysis

### Context
User approved merging the full fixed-state report and determining variance at
64 rollouts. Existing 128 paired continuations support two disjoint groups;
the plain-policy versus top-Q sampled-action experiment requires further approval.

### Plan
1. Append the complete original report to the active plan, retaining history.
2. Analyze original batches 0–3 versus 4–7 as two 64-return estimates with
   identical grouping across states/actions; report one-df observed variance
   separately from independent-continuation variance-at-16 divided by four.
3. Audit formulas/axes and record results without new simulator collection.

#### Files to edit
| File | Change |
|---|---|
| `docs/plans/replay_relabel_q_plan/README.md` | Full report and 64-rollout findings |
| `/tmp/opencode/height-q-fixed-variance-64-20261110-v1/` | Analysis script, tensors, JSON, report and log |

### Result
✅ Full original report merged into plan §10; verbatim-content check passed
after adjusting heading depth. Plan §11 records the two disjoint 64-return
groups and projection. Observed fixed-action variance mean/median
**0.343431/0.060939**, versus at16 **1.204215/0.563815**; projected at64
**0.301054/0.140954** under independent continuations. Observed within-group
action variance **0.316077/0.042201**; paired estimator variance
**0.280147/0.040696**, projected **0.302281/0.071005**. Pair-order agreement
**55.46875%**, winner agreement **19.53125%**.

Checks passed for shape/axes, unique seeds, pooling, two-group variance formula,
paired covariance and reproduction of original at16 statistics. Artifacts and
exact command are in `/tmp/opencode/height-q-fixed-variance-64-20261110-v1/REPORT.md`.

#### Files changed
| File | Change |
|---|---|
| `docs/plans/replay_relabel_q_plan/README.md` | Full original report (§10), observed/projection at64 (§11) |
| `/tmp/opencode/height-q-fixed-variance-64-20261110-v1/analyze.py` | Float64 offline analysis and formula audits |
| Same artifact directory: `results.json`, `statistics.pt`, `REPORT.md`, `analysis.log` | Reproducible numerical results and limitations |

### Notes
- Only one df per observed at64 variance: preliminary, weak precision. Average
  SD 0.382869 differs from sqrt(mean variance) 0.586030. No noise-ceiling claim.
- Existing data suffices for this preliminary answer; precise empirical variance
  needs more independent 64-return batches. No additional physics collection.
- Plain-policy versus top-Q sampled-action experiment remains unimplemented/unrun
   pending further approval. No commits, todos, completion or archival actions.

## [2026-11-10] More independent 64-return batches at matched rollout budget

### Context
Approved EDIT continuation: trade state coverage for independent batch replication,
using a return-independent random 16-state subset and eight original candidates.

### Plan
1. Reuse original 128 continuations and collect 896 new ones per state/action,
   mapping 1024 worlds to eight seed lanes × sixteen states × eight candidates.
2. Validate full native snapshot selection/replication and shared candidate noise;
   retain first block as benchmark and stop above a 12-minute projection.
3. Analyze sixteen 64-return batches and matched regrouped 16-return estimates,
   preserving covariance and batch-cluster uncertainty; keep top-Q selection deferred.

#### Files to edit
| File | Change |
|---|---|
| `BigRedGym/learning/offline/fixed_state_variance.py` | Export selection/replication indexing |
| `BigRedGym/scripts/extend_height_q_variance.py` | Reused-state lane collector and native gates |
| `BigRedGym/scripts/summarize_height_q_extension.py` | Matched grouping and uncertainty |
| Focused tests, active plan, offline docs/test catalog | Validation and evidence |

### Result
✅ Collected 114688 additional continuations in **399.989556 s (6m39.99s)**,
retaining the 29.411492 s benchmark. Combined original/new returns have explicit
[16,64,16,8] axes and 15 df per at64 variance. Return-independent selection seed
2026111064 chose original indices `[24,12,116,80,50,82,59,92,4,97,9,63,90,87,13,23]`.
No prefix rerun; all 82 snapshot fields and exact observations/actions/predictions
are retained. Native replication/restore, commands/pushes and real-policy CUDA
noise-pairing audits pass.

Matched same-data at16→at64 mean estimator variance **1.271764→.313614**;
median **.277224→.077920**. At64 action-spread variance mean/median
**.183222/.033911**; paired estimator variance **.365503/.064551**. Rank
agreement **51.0830%→52.7772%**, winner agreement **13.4487%→18.0729%**.
Whole-batch 4000-resample bootstrap: at64 mean variance CI [.247670,.334420];
ratio64/16 .246597, CI [.200368,.269332]. Conditional approximate intervals
preserve state/action covariance; sixteen-state coverage differs from prior128.

Report and complete artifacts:
`/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/REPORT.md`.
Active plan §12 contains protocol, results, provenance, native gates and caveats.

Validation: 7 focused, 229 portable (97 deselected), 35 colocated and 32 Warp
(294 deselected) pass; Ruff/format/diff checks clean. Eighteen existing MULTICCD
warnings arose in Warp contract fixtures, not collection physics. Offline analysis
assertions pass for reused data, identities, axes, binary rewards, exact insertion
observations/actions/predictions, pooling and paired covariance.

#### Files changed
| File | Change |
|---|---|
| `BigRedGym/learning/offline/fixed_state_variance.py` | Complete exported snapshot selection/replication |
| `BigRedGym/scripts/extend_height_q_variance.py` | Benchmark/resume lane collection and native gates |
| `BigRedGym/scripts/summarize_height_q_extension.py` | Matched grouping, covariance checks, bootstrap, full report |
| Focused tests, offline docs, MuJoCo limitations, test catalog, plan | Coverage and documented evidence |

### Notes
The 4× rollout count reduces variance approximately 4×, yet action ranking remains
weak. Surviving omitted tail remains [-.6570483,0]; mean tail-only interval
[-5.952297,-5.312794] excludes sampling error. Native one-step fidelity is not
bitwise long-horizon equivalence. Top-Q policy selection remains deferred; no
commits, todos, plan-completion or archival actions.

## [2026-11-10] Persistent sample-eight intervention at every decision

### Context
User approved extending the plan's earlier exclusion to run a frozen-policy
intervention: does max-Q selection among eight fresh stochastic samples improve
H500 height-band return relative to plain `model_600`, and does that survive
occupancy/fall/tracking checks? Also merged the full 16×64 report into the plan.

### Plan
1. Merge the complete sixteen-batch REPORT.md as a new plan section (no duplicate).
2. Implement a persistent controller and simulator collector reusing the original
   128 k=100 snapshots and noisy insertion observations, with shared per-execution
   RNG seeds and neural-only candidates.
3. Benchmark one block per controller; continue only under a 12-minute/controller
   projection; validate CPU/GPU Q batching and a short identical-controller control.
4. Collect 16 blocks × 8 lanes × 8 calls = 1024 executions/state per controller,
   then analyze paired returns, occupancy, falls, tracking and tails.

#### Files to edit
| File | Change |
|---|---|
| `BigRedGym/learning/offline/persistent_intervention.py` | Sample-eight controller and rollout |
| `BigRedGym/scripts/evaluate_height_q_persistent.py` | Benchmark/resume collector and gates |
| `BigRedGym/scripts/summarize_height_q_persistent.py` | Simulation-free audit and paired summary |
| `BigRedGym/tests/unit_tests/test_persistent_intervention.py` | Marginal/pairing/selection/tail contracts |
| Plan, journal, MuJoCo limitations, test catalog | Protocol, evidence, provenance |

### Result
✅ Benchmark projections 464.59 / 475.57 / 474.20 s per controller were inside the
720 s gate, so full collection ran: 465.46 / 476.24 / 476.80 s, 23:31:14–23:52:55
for the 45 resumed blocks. 48 block files retained; counts audited at
[128 states, 1024 executions] each with zero timeouts.

Mean H500 return: baseline **-5.2858**, original-Q **-3.5666**, evaldist-Q
**-4.1758**. Paired gains **+1.7191** (state SD 2.078, median +0.660, q05 +0.090,
128/0/0 wins, state CI [+1.378,+2.085], tail interval [+1.076,+2.361]) and
**+1.1100** (median +0.448, q05 +0.026, 123/0/5, state CI [+0.856,+1.384],
tail interval [+0.467,+1.752]). Outside-band step rate −1.73 / −1.08 pp; fall
rate +0.086 / +0.120 pp; XY tracking RMSE .4493 → .4561 / .4549. Selected indices
are near-uniform, which exchangeability predicts regardless of critic quality, so
the executed actions are a Q-selected order statistic, not the nominal marginal;
no same-bias control was run. The per-decision Q advantage (0.123/0.206) is not
comparable to the cumulative intervention gain and is not an underforecast.

Gates: native restore qpos 2.15e-6, qvel 1.03e-4, contact 1.50e-3; replication
qpos 1.43e-6, contact 5.65e-4; 82-field inventory exact; CPU/GPU Q 5.7e-6 / 7.6e-6
with 0 argmax disagreements; identical plain controllers exactly equal over five
steps. Validation: 11 focused, 233 portable (97 deselected), 28 gym + 7 learning,
32 Warp (298 deselected, 18 pre-existing warnings); Ruff and `git diff --check`
clean. One process audit at 23:53 found no live collector and all artifacts on disk.

Report: `/tmp/opencode/height-q-persistent-intervention-20261110-v1/REPORT.md`;
plan §13–§14 records protocol, timings, results, verification and provenance.
Runtime model metadata is not exposed by the session tools; the only provenance is
the stated `opencode-go/deepseek-v4.1-flash` configuration.

#### Files changed
| File | Change |
|---|---|
| `docs/plans/replay_relabel_q_plan/README.md` | Full 16×64 report (§13), approved protocol and results (§14) |
| `BigRedGym/learning/offline/persistent_intervention.py` | New controller/rollout helpers |
| `BigRedGym/scripts/evaluate_height_q_persistent.py` | New benchmark/resume experiment entry point |
| `BigRedGym/scripts/summarize_height_q_persistent.py` | New audit/summary entry point |
| `BigRedGym/tests/unit_tests/test_persistent_intervention.py` | New focused contracts |
| `BigRedGym/genAI_skills/README_MUJOCO.md`, `tests/TEST_CATALOG.md` | Evidence boundary and coverage |

### Notes
This is a **policy intervention** at 128 fixed k=100 states, not policy
improvement in the theorem sense and not `Q^pi` validation: persistent deviations
invalidate the frozen Q's target distribution. The gain comes with slightly more
falls (+0.09/+0.12 pp, CI touching zero) and ~1.5% worse tracking, so it is not
an unalloyed improvement. Selected indices are near-uniform, which exchangeability
predicts even for a perfect critic, so the executed actions are a Q-selected order
statistic rather than the nominal distribution; no same-bias control was run.
Fixed-state execution CIs and state-cluster CIs differ ~3×. Restoration is
one-step numerical, not bitwise long-horizon. No commits, todos, plan completion
or archival; prior CORL/config dirt preserved.

### Post-hoc audit — 2026-11-11
Two report errors were found and corrected from the same raw blocks (no rerun):
1. Paired truncation interval had both survivor fractions added to both sides,
   doubling its width. Corrected (asymmetric bound, baseline fraction low /
   controller fraction high): original [+1.0763,+2.3615] (was [+0.4339,+3.0043]),
   evaldist [+0.4671,+1.7521] (was [-0.1750,+2.3950]). Analytic oracle confirms
   old width = 2.0× true width; single-controller tail intervals were correct.
2. Occupancy/fall CIs used opposite-difference units. Now both are
   controller − baseline: outside-band rate −1.73/−1.08 pp, fall rate
   +0.086/+0.120 pp.
Also corrected interpretation: near-uniform selected indices do not imply nominal
action marginal; per-decision Q advantage (0.12/0.21) is not comparable to the
cumulative intervention gain and is not an "underforecast"; the experiment is a
policy intervention, not "one-step policy improvement". `results.json` regenerated
by the corrected summarizer; `audit_corrected.json` records the audit.

## [2026-11-11] Two additional Q critics: mixed 2M versus fresh 1M

### Context
User requested, after the paired16 comparison, two more fixed-policy height-Q
critics: one on the existing 2M PPO-collected early/final mixture and one on 1M
fresh transitions from unrolling frozen stochastic `model_600`. Equal epochs
(102.4) were explicitly accepted even though data composition and size differ.
Baseline/original/evaldist evaluation returns were to be reused, not rerun.

### Plan
1. Merge paired16 results and caveats into plan §15.
2. Collect 1M fresh nominal-policy transitions on GPU with the normal 5 s
   timeout/reset/pre-reset-successor contract and disjoint seed.
3. Train mixed (800k updates) and fresh (400k updates) critics at equal epochs
   with the original FQE architecture/optimizer, after a timing benchmark.
4. Evaluate both on the same 128 states / 1024 seeds / H500 / eight-action
   persistent controller, reusing the three old controllers.
5. Analyze individual and batch16 paired gains with the corrected tail helper;
   run focused, portable, colocated, Warp and Ruff gates; write report/journal.

#### Files to edit
| File | Change |
|---|---|
| `BigRedGym/scripts/collect_height_q_frozen_policy.py` | New frozen-policy replay collector |
| `BigRedGym/scripts/evaluate_height_q_persistent.py` | Optional artifact names and baseline skip |
| `BigRedGym/scripts/summarize_height_q_critic_datasets.py` | New simulation-free paired analysis |
| Plan and journal | Protocol, results, provenance |

### Result
✅ Fresh collection: 4096 worlds, seed 20261111, 611 policy steps =
2,502,656 appended, last 1M retained, 8,192 unique env/episode pairs, 15 true
terminations, 4,082 timeouts, 20.49 s. Mixed 2M snapshot audited read-only
(fingerprint `6412a878…`, pre-reset labels, finite fields).

Training: mixed 800k updates / 745.76 s / mean loss 6.554596; fresh 400k /
371.89 s / 0.337882. Policy specs match original fair100 exactly.

Evaluation: benchmark 29.43 / 29.52 s per first block (projected 470.8 / 472.3 s
per controller), full 895.74 s; only mixed/fresh simulated, 32 new blocks.

| Controller | Mean H500 | Paired gain | Individual W/T/L | Batch16 positive | Tail interval |
|---|---:|---:|---|---:|---|
| original (reused) | -3.5666 | 1.7191 | 56.91/21.22/21.87% | 85.49% | [1.0763,2.3615] |
| evaldist (reused) | -4.1758 | 1.1100 | 51.54/19.90/28.56% | 75.52% | [0.4671,1.7521] |
| mixed (new) | -4.3765 | 0.9093 | 48.15/19.01/32.84% | 72.88% | [0.2664,1.5574] |
| fresh (new) | -3.5946 | 1.6911 | 56.14/20.47/23.39% | 83.06% | [1.0483,2.3351] |

Per-state means positive: 128/123/120/125 of 128; per-state tail lower bound
positive: 67/54/46/66. Inside-band rates 97.18/96.55/96.34/97.20% vs 95.46%
baseline; falls 2.24/2.28/1.36/1.99% vs 2.16%. Selection near-uniform
(candidate0 ≈ 0.125; per-decision Q gain 0.1226/0.2056/0.1709/0.1116).

Gates: focused 22 passed; portable 233 (97 deselected); gym 28; learning 7;
Warp 32 (298 deselected, 18 pre-existing warnings); Ruff and `git diff --check`
clean. Source restore, CPU/GPU Q (max 4.01e-5/9.54e-6, zero argmax
disagreements), identical-controller control, zero timeouts and counts passed.

Report: `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/REPORT.md`;
commands/exit audit and manifest in the same root.

#### Files changed
| File | Change |
|---|---|
| `docs/plans/replay_relabel_q_plan/README.md` | Paired16 merge §15 and new protocol/results §16 |
| `docs/journal.md` | This entry |
| `BigRedGym/scripts/collect_height_q_frozen_policy.py` | New collector |
| `BigRedGym/scripts/evaluate_height_q_persistent.py` | Artifact naming and `--skip_baseline` |
| `BigRedGym/scripts/summarize_height_q_critic_datasets.py` | New analysis |

### Notes
The fresh critic is close to original on these fixed states; mixed is lower.
This is not a clean data-size effect (2M early+final mixture versus 1M nominal
`model_600`) and does not generalize beyond the 128 fixed states. Persistent
Q-selected actions are off the frozen policy distribution, so this is not
`Q^pi` validation, a policy-improvement theorem, or a noise-ceiling result.
The v1 `persistent_intervention.py` hash differs from the current tree (the
post-hoc asymmetric tail/sign change); reused baseline/original/evaldist blocks
are matched by seeds, source snapshot and checkpoint, not bitwise code identity.
No commits, todos, plan completion or archival; unrelated dirty work preserved.

## [2026-11-11] Return SD and paired t-test addendum

### Context
The five-controller mean-return table reported means and cluster/execution CIs
but not the spread of individual rollouts, and the four Q-vs-baseline gains had
no paired significance summary. The user asked for per-controller descriptive
SD and a correct paired Student t-test, without treating the 131,072 correlated
executions as IID and without rerunning any collection.

### Plan
1. Read only the existing raw `[128,64]` return blocks for baseline/original/
   evaldist (v1) and mixed/fresh (new); no simulator or training.
2. Compute pooled individual-rollout SD over all 131,072 returns per controller
   (ddof=1), the separate SD of the 128 state means, and paired t tests on the
   128 per-state mean differences with Holm adjustment for four comparisons.
3. Verify against existing `results.json`, an independent p-value formula and
   system SciPy; add the tables and caveats to both reports, plan §16.h and this
   journal.
4. Audit exact foreground commands and exit statuses; leave product code,
   snapshots and original collections untouched.

#### Files to edit
| File | Change |
|---|---|
| `analysis/return_stats.py`, `analysis/verify_scipy.py` | New stats and SciPy verification scripts |
| `analysis/return_stats.json`, `return_stats.md`, `scipy_verification.json` | New numerical/verification artifacts |
| `REPORT.md`, `analysis/REPORT.md` | Mean ± pooled SD table and paired t-test section |
| Plan, journal, `commands.log` | §16.h, this entry, command audit |

### Result
✅ All five means/state-mean SDs and four gain means/SDs reproduce
`analysis/results.json` exactly. Pooled individual SD is 8.1216 (baseline),
6.3288, 7.1375, 6.8288 and 6.4504; state-mean SD is 6.7391, 5.0477, 5.6627,
5.6668 and 5.1106. Paired tests: original t=9.3579, raw p=3.765e-16, Holm
p=1.506e-15; evaldist t=8.1085, p=3.726e-13, Holm 7.452e-13; mixed t=7.0441,
p=1.050e-10, Holm 1.050e-10; fresh t=9.1288, p=1.355e-15, Holm 4.064e-15; all
df=127. Mean gains 1.7191/1.1100/0.9093/1.6911 with gain SD
2.0784/1.5488/1.4604/2.0959. The tests are paired one-sample tests on 128
fixed configurations, not independent two-sample or execution-level tests.
`return_stats.py` (exit 0, 0.97 s) and system SciPy 1.8.0 `verify_scipy.py`
(exit 0, 0.14 s) passed all cross-checks; Ruff passed on both scripts.

### Notes
Tiny p values are scientific notation, not zero, but they only establish a
nonzero fixed-state mean difference under the configuration-independence
assumption. The 128 states share one seed list and are fixed k=100 snapshots
from one policy/lane, so no state-population, generalization or safety claim is
made; finite H500 means no p value concerns the infinite-horizon return. The
mean gain is only 0.13–0.27 of the pooled individual SD (48–57% individual win,
19–21% ties), while paired `mean gain / gain SD` is 0.62–0.83; the descriptive
inverse state-variance/gain ordering is not a fitted causal law. No product
source, snapshot, Q artifact or original collection was modified; no new
experiment, commits, todos, completion or archival.

#### Files changed
| File | Change |
|---|---|
| `docs/plans/replay_relabel_q_plan/README.md` | Statistics addendum §16.h |
| `docs/journal.md` | This entry |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/return_stats.py` | New descriptive/paired-test script |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/verify_scipy.py` | New SciPy cross-check |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/return_stats.json`, `return_stats.md`, `state_means_for_scipy.npz`, `scipy_verification.json` | New stats/verification artifacts |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/REPORT.md` | Mean ± pooled SD and paired t-test section |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/REPORT.md` | Same addendum for the analysis report |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/commands.log` | Foreground command/exit audit |

## [2026-11-11] Completion, archive and commit of offline replay/height-Q work

### Context
The user confirmed the replay/relabel/FQE/variance/persistent-intervention work
is finished and explicitly authorized archive plus commits in both owning repos.
Scientific limitations and the unrelated dirty CORL/config work had to survive
unchanged. No todos were to be touched.

### Plan
1. Re-run the final portable, colocated, Ruff and diff gates on the current tree.
2. Mark the plan completed, preserve its scientific boundaries, and move it to
   `docs/plans/` without truncation.
3. Repoint journal links, including earlier entries, and update the one
   user-facing "active plan" reference.
4. Commit only session-owned product files and docs with a temporary index, so
   the pre-existing staged `.gitmodules`/`thirdparty/CORL` state and unrelated
   worktree edits remain untouched.

#### Files to edit
| File | Change |
|------|--------|
| `docs/plans/replay_relabel_q_plan/README.md` | Completed status and §17 archive record |
| `docs/journal.md` | Fixed archived-plan links and this entry |
| `BigRedGym/genAI_skills/README_MUJOCO.md` | Archived-plan wording |
| `BigRedGym/` offline replay/FQE/variance/persistent files | Session-only product commit |

### Result
✅ The full portable gate passed (233 passed, 97 deselected), colocated suites
passed (28 gym + 7 learning), `ruff check .` passed, all 30 touched Python files
were already formatted, and `git diff --check` was clean. The plan was archived
with its full history and limitations, including the statement that persistent
Q-selection is not safety or `Q^pi` validation. Product and docs commits used
session-only temporary indexes; unrelated staged and worktree changes were left
dirty. Product commit: `dfc506e449eedad5f57bb208e38b4dca69977e3e`
(`Add offline replay, height FQE, and persistent Q intervention`).

#### Files changed
| File | Change |
|------|--------|
| `docs/plans/replay_relabel_q_plan/README.md` | Removed from active plans (archived) |
| `docs/plans/replay_relabel_q_plan/README.md` | Completed 2026-11-11 archive record |
| `docs/journal.md` | Archive links and completion entry |
| `BigRedGym/genAI_skills/README_MUJOCO.md` | Active → archived plan wording |
| `BigRedGym/learning/offline/`, `learning/algorithms/fqe.py`, `learning/storage/transition_recorder.py` | Offline replay, FQE, variance and intervention implementation |
| `BigRedGym/learning/runners/on_policy_runner.py`, `scripts/train.py`, `gym/envs/go2/go2trot.py` | Recorder integration, CLI and task state capture |
| `BigRedGym/scripts/*height_q*`, `scripts/collect_height_q_frozen_policy.py` | Collect/train/evaluate/summarize entry points |
| `BigRedGym/tests/unit_tests/test_*height*`, `test_offline_replay.py`, `test_persistent_intervention.py`, `gym/envs/go2/test_go2trot.py` | Focused regression tests |
| `BigRedGym/docs/offline_height_q.md`, `README.md`, `tests/TEST_CATALOG.md`, `genAI_skills/README_MUJOCO.md`, Q2 skills | Documentation and evidence boundaries |

### Notes
Excluded from the product commit: the pre-existing staged `.gitmodules` and
`thirdparty/CORL` pointer, CORL docs/artifact files, unrelated `genAI_skills/
AGENTS.md` CORL text, and all existing worktree dirt. Plan figure assets exceed
100 KiB and were not committed. No logs, datasets, checkpoints, vendor files or
todos were added. No amend, force-push or push was performed.

## [2026-11-11] Threshold-constrained persistent action-selection pilot

### Context
Extra experiment appended to the archived height-Q plan: at every decision use
the frozen policy mean (candidate 0) plus the same eight stochastic draws
(indices 1..8), score all nine with the original fair-100 Q, select the closest
feasible candidate with `Q >= c`, and fall back to max-Q over all nine if none
is feasible. The full sweep is gated on user approval of five numerical
thresholds; only the nominal-distribution pilot was run.

### Plan
1. Implement mean-plus-eight candidate construction and the constrained
   selection/rollout primitives, preserving existing sample-eight semantics.
2. Add analytic/fake-environment tests for feasibility, distance, fallback,
   ties, negative thresholds, finite positive std, and mean control.
3. Run a 64-execution/state nominal pilot over the 128 fixed k=100 snapshots;
   score Q at every active decision without intervening.
4. Summarize the Q distribution, propose five fixed thresholds, and append
   extra evidence to the archived plan without moving it.

### Result
✅ Focused tests 17 passed; portable 239 passed/97 deselected; gym 28; learning
7; Ruff clean. The pilot completed in 29.9 s (8 calls, 3.70–3.84 s/call,
~119 MiB peak) with 82-entry native restore, zero timeouts, and CPU/GPU Q max
error 3.81e-6 / zero argmax disagreements. Nominal returns match the existing
baseline block 0 (correlation 0.99652, mean difference -0.00560, first rewards
exact). Proposed thresholds: **-16.756, -4.866, -1.776, -0.956, -0.626**;
decision-weighted mean-pass rates 95.0%, 75.0%, 50.0%, 25.0%, 5.0%. Full sweep
(~6 controllers, ~48 min) was not launched; awaiting threshold approval.

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/persistent_intervention.py` | Nine-candidate construction, threshold selection, threshold/mean rollouts |
| `BigRedGym/tests/unit_tests/test_persistent_intervention.py` | Six new focused tests |
| `BigRedGym/scripts/measure_height_q_threshold_pilot.py` | Nominal Q-distribution pilot |
| `BigRedGym/scripts/summarize_height_q_threshold_pilot.py` | Simulation-free summary and threshold proposal |
| `docs/plans/replay_relabel_q_plan/README.md` | Appended Extra test section; §17 completion unchanged |
| `docs/journal.md` | This entry |

### Notes
The Q distribution is strongly left-skewed (Q(mean) median -1.776, q05 -16.756,
raw min -171.52), so the five quantile thresholds are non-uniformly spaced and
the nearest-feasible-stochastic rescue is rare (0.28–1.53%). Nominal rates will
shift under persistent intervention because the branch leaves the frozen
policy's state distribution. The real-environment threshold controller is not
yet executed; its selection logic is unit-tested and it shares the rollout loop
validated by the nominal pilot. No commit, todo, plan-completion, or archival
movement; unrelated CORL/config dirt preserved. Pilot artifacts:
`/tmp/opencode/height-q-threshold-pilot-20261111-v1/`.

## [2026-11-11] Rejection and closest-mean threshold sweep

### Context
The prior pilot fixed five Q thresholds from the nominal trajectory but did not
execute threshold policies. The user asked for a full sweep measuring actual
H500 threshold attainment, mainly stochastic rejection (sample eight, execute the
first feasible) rather than nearest-mean, plus two deterministic closest-mean
controllers for comparison. The plain baseline and historical eight-way best were
to be reused, with no extra mean-only controller.

### Plan
1. Add rejection selection and rollout primitives while preserving existing
   sample-eight/threshold/mean behavior; add focused analytic and fake-environment
   tests, including exact short negative controls.
2. Add a full-sweep evaluator with native restore gates, a real short CUDA
   controller smoke for every controller, retained benchmarks, resumable blocks,
   and a 12-minute/controller gate.
3. Run five rejection controllers at `c = [-16, -4.866, -1.776, -0.956, -0.626]`
   and two distance controllers at `[-4.866, -1.776]` over 128 states × 1024
   paired executions, reusing the plain/best-eight blocks.
4. Summarize actual threshold attainment, per-config CIs and survivor tails,
   pair gains, batch16 groups, selection/attempt/fallback diagnostics, and append
   the results to the archived plan.

### Result
✅ Focused 21; portable 243/97; gym 28; learning 7; Warp 32/308 (18 pre-existing
warnings); Ruff, format and diff-check clean. All seven controllers passed the
native/CPU-GPU/exact negative-control gates and a real eight-step CUDA smoke.
Block-0 projections were 489–505 s/controller. The full 112-block run exited 0 in
**57.33 min**, with 131,072 executions/controller and zero timeouts.

Actual mean H500 returns: rejection -4.6491, -3.8862, -3.6243, -3.5779, -3.5606;
distance -3.7754, -3.6129; plain -5.2858; best8 -3.5666. Aggregate expectation
meets its own c only at -16 and -4.866; at -1.776/-0.956/-0.626 the means remain
below c. State-mean attainment is 118, 95, 74, 58, 42 of 128 (rejection) and 93,
74 (distance). `c = -16` is empirically near plain (+0.637 gain, 95.6% first draw,
4.6% fallback) but not equivalent. `c = -0.626` is empirically near best8 (+0.006,
paired tail includes zero, 92.9% fallback) but feasible-first selection is a
different rule, and restrictive rejection does not beat best8's state-mean
attainment. Distance is not clearly better than rejection.

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/persistent_intervention.py` | Rejection primitives, selected-Q/distance/attempt/termination diagnostics |
| `BigRedGym/tests/unit_tests/test_persistent_intervention.py` | Five rejection/negative-control tests |
| `BigRedGym/scripts/evaluate_height_q_rejection.py` | Seven-controller full sweep with gates, smoke, benchmarks, resume |
| `BigRedGym/scripts/summarize_height_q_rejection.py` | Simulation-free audit, attainment/pair summaries, CSV/report export |
| `BigRedGym/tests/TEST_CATALOG.md` | Rejection test and evaluator coverage |
| `docs/plans/replay_relabel_q_plan/README.md` | Appended Extra test results; prior completion unchanged |
| `docs/journal.md` | This entry |

### Notes
A first benchmark attempt raised `FileExistsError` because the block writer was
invoked per controller while also iterating all controllers. It was refactored
before any result was used; the failed directory was removed and block 0 was
rerun once and retained. Thresholds were fixed from Q quantiles before outcomes
and were not tuned from gains. Statistics are conditional on 128 fixed k=100
states, one frozen policy and one source lane; they are not a state population,
hardware, safety, `Q^pi`, or policy-improvement claim. Artifacts:
`/tmp/opencode/height-q-rejection-threshold-20261111-v1/`. No commit, todo,
plan-completion or archival movement; unrelated dirt preserved.

## [2026-11-11] Durable report supplements for replay/relabel height-Q

### Context
The replay/relabel/FQE reports existed only under `/tmp/opencode`, while the
archived plan was the sole durable record. The user asked to move the generated
report Markdown into product docs under `docs/plans/supplements/`, preserve the
raw experiment artifacts in `/tmp/opencode`, and keep report links working. The
user later asked for one explicit consolidated table comparing per-configuration
threshold attainment against plain PPO.

### Plan
1. Inventory reports referenced by the archived plan; exclude unrelated
   experiments and the pre-fix rejection scratch duplicate.
2. Record pre-move sha256/size, move nine report documents into unique
   subdirectories under `BigRedGym/docs/plans/supplements/replay_relabel_q/`,
   and leave compatibility symlinks at the original `/tmp/opencode` paths.
3. Add provenance notes, migrate only broken relative artifact/report links, and
   keep artifact paths pointing at the original directories.
4. Add the consolidated `c` versus plain/rejection/closest-mean attainment table
   to the archived plan extra-test results and to the durable rejection report.
5. Repoint archived-plan report links to the durable product-docs paths, write an
   index and move manifest, and verify file existence, hashes, and diff hygiene.

#### Files to edit
| File | Change |
|------|--------|
| `BigRedGym/docs/plans/supplements/replay_relabel_q/` | Nine durable reports, `README.md`, `MOVE_MANIFEST.md` |
| `docs/plans/replay_relabel_q_plan/README.md` | Durable report links and consolidated attainment table |
| `docs/journal.md` | This entry |

### Result
✅ Nine reports moved from the original `/tmp/opencode` experiment directories;
all original report paths now symlink to the durable copies. Raw data,
checkpoints, logs, and other artifacts stayed in place. Removing the added
provenance notes and reversing the documented link/table changes reproduced the
recorded pre-move hashes for all nine files. The consolidated table rows are
`c=-16` 116/118 (+2, distance —), `c=-4.866` 85/95 (+10, 93),
`c=-1.776` 61/74 (+13, 74), `c=-0.956` 39/58 (+19, distance —), and
`c=-0.626` 22/42 (+20, distance —).

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/docs/plans/supplements/replay_relabel_q/README.md` | Chronology, status, caveats, reading order, artifact policy |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/MOVE_MANIFEST.md` | Pre/post hashes, destinations, intentional changes |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/*/REPORT.md`, `analysis/REPORT.md`, `analysis/return_stats.md` | Nine moved report documents |
| `docs/plans/replay_relabel_q_plan/README.md` | Durable report links; consolidated table in the rejection extra-test results |
| `docs/journal.md` | This entry |

### Notes
The new table is a net count comparison of configurations whose 1024-rollout
mean finite H500 plus absorbing return is at least `c`; it is not a count of
rescued paired states, can hide per-configuration regressions, and carries no
sampling-uncertainty, multiplicity, or survivor-tail correction. No summarizer
was rerun, because several would write `REPORT.md` through the new compatibility
symlinks. No commits, todos, plan completion, or unrelated worktree changes.

## [2026-11-11] Plan: Q-intervened height-Q playback

### Context
The user requested a new maintained parallel playback script that loads the
frozen `go2trot` `model_600` actor and the original fair-100 height-Q artifact,
with stochastic max-Q or rejection selection, zero initial commands, manual
reset, and auto-reset disabled by default. Planning only; runtime model
`opencode-go/deepseek-v4.1-flash` (same for subagents).

### Plan
1. Discover current `play.py` CLI, keyboard callback, command resampling, task
   reset, and height-Q helper behavior.
2. Record workspace/branch/dirty/submodule baseline and the fair-100
   checkpoint/artifact identities.
3. Write `docs/plans/q_intervened_playback_plan.md` with CLI,
   reset semantics, validation gates, unit/fake-env tests, CPU headless smoke,
   docs/skills updates, and no-commit/todo constraints.

#### Files to edit
| File | Change |
|------|--------|
| `docs/plans/q_intervened_playback_plan.md` | New plan only |
| `docs/journal.md` | This entry |

### Result
✅ Plan written for user review. No product edits, tests, commits, todos, or
plan archival. Discovered that current `play.py` never auto-resets on fall or
timeout, seeds forward 1.0, and `TeleopCommands` replaces its 9999 s command
resampling time with `max_episode_length_s + 1`.

#### Files changed
| File | Change |
|------|--------|
| `docs/plans/q_intervened_playback_plan.md` | New plan only |
| `docs/journal.md` | This entry |

### Notes
Proposed defaults: `--mode maxq`, `--auto_reset off`, zero initial commands,
disabled command resampling, CPU/1 env, artifact-noise observations, and normal
task reset with zero commands after manual/auto resets. Non-blocking choices:
observation noise on, pushes preserved from task config, and timeout horizon
from task config rather than `play.py`'s 50 s. Manual viewer validation cannot
be claimed from headless execution.

### Result (implementation, 2026-11-11)
✅ Playback implemented and verified through V1-V7. Details, deviations, test
results, and the remaining manual viewer gate are in
[`plans/q_intervened_playback_plan.md`](plans/q_intervened_playback_plan.md)
§11. No commits, todos, or plan archival; V8 remains user-only.

## [2026-11-12] Edit — Playback plan archived at user request

Implementation committed as BigRedGym `6507a13`; automated checks passed.
V8 manual viewer validation remains unverified. No todos changed.
Plan: [Q-intervened playback](plans/q_intervened_playback_plan.md).

## [2026-11-11] Extra test: original PPO reward versus height-Q tradeoff

### Context
User approved a fresh paired H500 run comparing plain candidate-0 stochastic
`model_600` with the original fair-100 height-Q max-of-8 controller at the same
128 fixed insertion states. The primary question was whether the height-Q
intervention sacrifices the original PPO training return; height-band gain was
secondary on the same executions.

### Plan
1. Add a shared exact training-reward dispatcher and dual-objective rollout.
2. Add portable tests plus a real-environment runner-parity gate.
3. Pilot block 0, retain it, gate projected runtime at 12 min/controller.
4. Collect 16 blocks x 2 controllers at paired seeds 7100000+10*call.
5. Summarize original and height objectives, per-term contributions, state
   bootstrap, paired t, batches, CSV, and prior-run drift.
6. Append protocol/results to the archived plan; write a durable report and
   update the supplement README and test catalog.

#### Files to edit
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/training_reward.py` | New shared runner-equivalent dispatch |
| `BigRedGym/learning/offline/reward_tradeoff.py` | New dual-objective rollout |
| `BigRedGym/learning/offline/tradeoff_analysis.py` | New portable statistics |
| `BigRedGym/learning/runners/BaseRunner.py` | Use shared reward-function map |
| `BigRedGym/scripts/evaluate_height_q_reward_tradeoff.py` | New collection script |
| `BigRedGym/scripts/summarize_height_q_reward_tradeoff.py` | New offline summary |
| `BigRedGym/tests/unit_tests/test_reward_tradeoff.py` | New tests |
| `docs/plans/replay_relabel_q_plan/README.md` | Protocol and results |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/` | Durable report and index |
| `BigRedGym/tests/TEST_CATALOG.md` | Scope update |

### Result
✅ 32/32 blocks complete, EXIT 0. Original PPO paired mean gain
**-0.026949** (t(127) = -4.321855, state CI95 [-0.038765, -0.014482],
-0.0999% of a positive 26.977067 baseline). Height paired gain
**+1.718288** (state CI95 [+1.376687, +2.090214]). Execution original
win/tie/loss 55741/0/75331; 16-batch positive fraction 0.347656. The measured
tradeoff is a small original-return sacrifice for a height gain at fixed
states, not a population or safety claim. All gates passed; no commits, todos,
or archive movement.

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/training_reward.py` | New |
| `BigRedGym/learning/offline/reward_tradeoff.py` | New |
| `BigRedGym/learning/offline/tradeoff_analysis.py` | New |
| `BigRedGym/learning/runners/BaseRunner.py` | Shared dispatch import |
| `BigRedGym/scripts/evaluate_height_q_reward_tradeoff.py` | New |
| `BigRedGym/scripts/summarize_height_q_reward_tradeoff.py` | New |
| `BigRedGym/tests/unit_tests/test_reward_tradeoff.py` | New |
| `BigRedGym/tests/TEST_CATALOG.md` | Catalog scope |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/README.md` | Index/scope |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/MOVE_MANIFEST.md` | Report row |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/original-reward-tradeoff-20261111-v1/REPORT.md` | Durable report |
| `docs/plans/replay_relabel_q_plan/README.md` | Extra-test section |
| `docs/journal.md` | This entry |

### Notes
Raw blocks and logs stay in
`/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/`; the report has a
compatibility symlink there. H500 is twice the 250-step training episode, and
the original objective zeroes rewards after true termination with no absorbing
tail. Runtime sum 1083.07 s for 32 blocks. Unrelated dirty playback work was
preserved.

## [2026-11-12] Angular-velocity-tracking Q (second offline reward path)

### Context
Add the go2trot yaw tracking term as a second concrete offline reward path:
relabel the existing final-1M replay, fit a fixed-policy FQE Q, and measure
plain versus max-Q and rejection interventions from the existing 128-state
insertion snapshot. Locked decisions: raw `exp` term in `[0, 1]`, zero after
true termination, `gamma = 0.99`, Q input = actor obs (48) + action (12), final
1M snapshot only, label noise accepted but quantified.

### Plan
1. Add `AngularVelocityRewardSpec` and the angular artifact/return identifiers.
2. Relabel from `next_actor_obs` yaw fields located by manifest names.
3. Add `AngularVelocityQ`/`save_angular_velocity_q` and `train_angular_velocity_q`.
4. Add the no-absorbing-tail return accumulator and angular rollouts (plain,
   max-Q, rejection) with a trace variant.
5. Add train/evaluate/summarize/plot scripts with the same identity gates as
   height plus live reward parity, label-noise, and replay-noise measurements.
6. Add portable tests, docs, durable report, skill and catalog updates, and run
   the V1-V4 gates.

#### Files to edit
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/contracts.py` | Angular reward spec and constants |
| `BigRedGym/learning/offline/rewards.py` | Angular relabel and field lookup |
| `BigRedGym/learning/offline/q_artifact.py` | Angular Q artifact |
| `BigRedGym/learning/offline/training.py` | Angular FQE orchestration |
| `BigRedGym/learning/offline/fixed_state_variance.py` | Zero-after-terminal accumulator |
| `BigRedGym/learning/offline/persistent_intervention.py` | Angular rollouts and traces |
| `BigRedGym/scripts/*angular_velocity*` | Train/evaluate/summarize/plot CLIs |
| `BigRedGym/tests/unit_tests/test_angular_velocity_q.py` | Portable contracts |
| Docs, skill, catalog, report | Record the second path and its evidence |

### Result
❌ for improvement, ✅ for the pipeline. Focused tests 9 passed; portable gate
309 passed/97 deselected; `pytest gym` 28; `pytest learning` 7; Ruff clean.

V2: 100k-update artifact on the 100% subset carries
`format=brg-angular-vel-q-v1`,
`return_convention=discounted_episodic_zero_after_terminal`, `failure_value=0.0`,
`gamma=0.99`, `tracking_sigma=0.25`, obs/action 48/12, checkpoint sha `f88b1f9b…`.
V3 (1 block, 64 executions/state): plain `r'=96.2242`; max-Q `95.5344`, paired
gain `-0.6898` (CI `[-0.7794, -0.6052]`, t `-15.44`, p `6.4e-31`, W/T/L
`4/0/124`); rejection `c=5` gain `-0.0144` (CI `[-0.0283, -0.0032]`, p `0.029`,
fallback 0.1%); rejection `c=20` gain `-0.4088` (p `3.4e-19`, fallback 33%).
An independent rerun of the same command reproduced max-Q `-0.6871` and `c=20`
`-0.4003` but flipped `c=5` to `+0.0059` (p `0.42`), i.e. `c=5` is inside the
~0.02 mean-absolute replay floor and is not a detectable effect.
Label-noise RMSE `0.00835` (relative 0.86%) at `σ_ang_vel=0.0333 rad/s`.
V4 plot written; detected resamples at 0.98/3.98/6.98/9.98 s; world-step yaw
RMSE 0.2373 plain versus 0.2566 max-Q.

Key deviation: the plan's formula omitted the task's hard-coded `/2.5`, so the
implemented spec carries `error_scale=2.5`; a live parity check against
`Go2Trot._reward_tracking_ang_vel` returns max absolute error `0.0`.
Also found: Warp restore is not bitwise, and the continuous reward exposes
H500 duplicate-rollout differences (mean absolute `0.018`, max `3.82` on a mean
return of `96.2`) that the binary height reward hid; the negative control now
pins discrete fields exactly and bounds the continuous return.

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/contracts.py` | `AngularVelocityRewardSpec`, angular format/return constants |
| `BigRedGym/learning/offline/rewards.py` | `angular_velocity_reward`, `observation_field_slice`, `relabel_angular_velocity` |
| `BigRedGym/learning/offline/q_artifact.py` | `AngularVelocityQ`, `save_angular_velocity_q` |
| `BigRedGym/learning/offline/training.py` | `train_angular_velocity_q` |
| `BigRedGym/learning/offline/fixed_state_variance.py` | `accumulate_return_zero_after_terminal` |
| `BigRedGym/learning/offline/persistent_intervention.py` | Angular common rollout, max-Q/rejection, trace |
| `BigRedGym/scripts/train_angular_velocity_q.py` | New CLI |
| `BigRedGym/scripts/evaluate_angular_velocity_q_persistent.py` | New CLI with parity/label-noise/replay-noise gates |
| `BigRedGym/scripts/summarize_angular_velocity_q_persistent.py` | New summary with paired t and state-cluster CI |
| `BigRedGym/scripts/plot_angular_velocity_tracking.py` | New time-domain plot |
| `BigRedGym/tests/unit_tests/test_angular_velocity_q.py` | New portable tests |
| `BigRedGym/docs/offline_angular_velocity_q.md` | New reward-path documentation |
| `BigRedGym/docs/offline_height_q.md` | Cross-link |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/angular-velocity-tracking-20261112-v1/REPORT.md` | Durable report |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/README.md` | Index and scope |
| `BigRedGym/genAI_skills/.agents/skills/q2-offline-reward-development/SKILL.md` | Document the second path |
| `BigRedGym/tests/TEST_CATALOG.md` | New test row |
| `docs/journal.md` | This entry |

### Notes
Raw artifacts and logs stay in `/tmp/opencode/angular-vel-q-20261112/`. The Q is
weakly discriminative (mean max-Q advantage `0.224` on a ~96 return scale) and
the interventions reduce survival, so the negative gain is a fit/diagnostic
result, not evidence that a yaw Q cannot help. Height behavior and tests were
left untouched; no commits, staging, or todos were made. Deferred: threshold
sweep on a converged Q, multi-checkpoint Qs, and any policy-optimizing use.

## [2026-11-12] Angular-velocity Q: converged refit, ranking ceiling, re-run

### Context
Follow-up (option a) to the negative max-Q result: train a better-converged yaw
Q and re-evaluate, to decide whether the earlier result is a Q-quality artifact.
Added held-out convergence evidence (episode-split residual) and a candidate
ranking diagnostic, then a seed-paired two-block intervention with the best Q.

### Plan
1. Add an episode-grouped 90/10 split and per-fraction training reports with the
   loss curve, train-sample and held-out Bellman residual.
2. Sweep 100k/300k/500k, lower lr, batch 512, target samples 8, wider nets; pick
   the best by held-out error.
3. Diagnose candidate ranking against realized returns for old and new Qs,
   including a cross-seed reliability ceiling.
4. Re-run plain + max-Q (old/new) + rejection thresholds for 2 blocks on the same
   seeds; compare against the original numbers.
5. Update the durable report, the reward doc, plan section 6, and this journal.

#### Files to edit
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/dataset.py` | `episode_split` |
| `BigRedGym/learning/offline/training.py` | held-out validation + report |
| `BigRedGym/scripts/train_angular_velocity_q.py` | validation flags |
| `BigRedGym/scripts/diagnose_angular_velocity_candidate_ranking.py` | new diagnostic |
| `BigRedGym/scripts/evaluate_angular_velocity_q_persistent.py` | multi-critic traces |
| `BigRedGym/tests/unit_tests/test_angular_velocity_q.py` | split + residual tests |
| Docs, report, plan, catalog, skill | New evidence and verdict |

### Result
✅ Measured; ❌ for "a converged Q fixes it".

Best held-out variant is `v0` (= the original 100k/lr3e-4/b256/tas1/[256,256]
recipe on the 90% split): held-out MSE 5.626 (RMSE 2.372, MAE 0.555) at the
32-sample target. `v1` 300k: 5.746; `v3` 300k wide/tas8: 6.171; `v2` 500k
lr1e-4/b512/tas8: 6.203. Training loss still falls at 300k (last-1k 5.05 vs
5.47) without held-out gain. Target-sampling noise contributes only RMS
0.17–0.31, so the ~2.4 residual is not sampling noise.

Q advantage over candidate 0: old 0.2238 -> v0 0.2339 (+4%) / v1 0.2624 (+17%)
on the persistent eight-draw protocol; stored candidates 0.2073 -> 0.2105/0.2243.

Ranking (128 states x 8 candidates x 2 seeds): old/new/300k Spearman -0.017/
-0.003/-0.026 at H100 and -0.030/+0.005/-0.016 at H500; argmax hit 0.10–0.13 vs
0.125 chance; top-1 regret 0.46–0.95; Q-advantage correlation <=0.18. Cross-seed
test-retest within-state Spearman is 0.007 (H100) / 0.019 (H500) with argmax
agreement 0.164, i.e. the realized ranking itself is not reproducible.

Two-block intervention (128 execs/state): old -0.653 [CI -0.720,-0.589],
v0 -0.615 [-0.683,-0.549], v1 -0.369 [-0.424,-0.314], rejections
-0.445/-0.656/-0.621; all p <= 1e-24. `v0 - old` = +0.038 (p 0.11, n.s.);
`v1 - old` = +0.284 (p 4e-16) but still negative vs plain.

Verdict: fundamental low-signal/off-policy-selection limit, not a Q-quality
artifact — no variant lowered held-out error and the ranking target has no
cross-seed reliability at policy-noise action scale.

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/dataset.py` | `episode_split` (disjoint episode groups) |
| `BigRedGym/learning/offline/training.py` | optional held-out validation, loss curve, Bellman-residual report |
| `BigRedGym/scripts/train_angular_velocity_q.py` | `--validation_fraction/--split_seed/--validation_every` |
| `BigRedGym/scripts/diagnose_angular_velocity_candidate_ranking.py` | New ranking diagnostic with cross-seed ceiling |
| `BigRedGym/scripts/evaluate_angular_velocity_q_persistent.py` | `--trace_names` |
| `BigRedGym/tests/unit_tests/test_angular_velocity_q.py` | Split and residual tests (11 total) |
| `BigRedGym/docs/offline_angular_velocity_q.md` | Convergence, ranking, verdict |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/angular-velocity-tracking-20261112-v1/REPORT.md` | Follow-up section and verdict |
| `BigRedGym/docs/plans/supplements/replay_relabel_q/README.md` | Index row |
| `BigRedGym/tests/TEST_CATALOG.md` | Test scope |
| `BigRedGym/genAI_skills/.agents/skills/q2-offline-reward-development/SKILL.md` | Held-out/ranking practice |
| `docs/plans/angular_velocity_q_plan.md` | Section 6 |
| `docs/journal.md` | This entry |

### Notes
Artifacts under `/tmp/opencode/angular-vel-q-20261112/qbetter/`. Height path and
its tests untouched; no commits, staging, or todos. Deferred: candidate sets with
larger action perturbations, other insertion states, and any policy-optimizing
use. The 300k Q's smaller intervention loss is a selection-behavior difference,
not evidence of better value accuracy.

## [2026-11-12] Edit — Angular-velocity Q plan completed and archived

Result: neither max-Q nor rejection improves the yaw-tracking reward `r'`; a
converged refit behaves identically, so this is a fundamental low-signal /
off-policy-selection limit at policy-noise-scale actions (not a Q-quality
artifact). Durable docs and the supplement report are in place. No commits,
staging, or todos. Plan: [Angular-velocity Q](plans/angular_velocity_q_plan.md).

## [2026-11-13] Feature — Selectable replay fields; 1M all-signal dataset

### Context
The recorder stored a fixed core field list, so offline reward work could not
label anything that the 48-d actor observation does not carry (contacts, body
kinematics, torques, per-term rewards, critic observation). The user asked to
"add a flag or config input so we can reuse it for a different subset of
fields", then to collect 1M transitions from the trained go2trot policy with
every available signal.

### Plan
1. Add `learning/offline/replay_fields.py`: group/field registry, canonical
   order, `resolve_replay_fields`, step/next partition.
2. Add `RecorderConfig.fields` and `REPLAY_FORMAT_V2`; leave `SCHEMA_VERSION`
   and the legacy v1 reader alone.
3. Extend `TransitionRecorder` with exact-set/dimension/finite validation and a
   manifest that declares `recorded_fields`, `field_dims`, `field_metadata`.
4. Capture pre-action and pre-reset successor features in `OnPolicyRunner`.
5. Load/concat v1 and v2 snapshots in `dataset.py`; add `--replay_fields`.
6. Document, test, then collect 1M from `model_600`.

#### Files to edit
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/replay_fields.py` | New feature registry and resolver |
| `BigRedGym/learning/offline/contracts.py` | `REPLAY_FORMAT_V2`, `RecorderConfig.fields` |
| `BigRedGym/learning/storage/transition_recorder.py` | Selectable extra buffers, v2 manifest |
| `BigRedGym/learning/offline/dataset.py` | v1/v2 field resolution and validation |
| `BigRedGym/learning/runners/on_policy_runner.py` | Feature snapshots and metadata |
| `BigRedGym/scripts/train.py` | `--replay_fields` |
| `BigRedGym/docs/offline_height_q.md` | Field groups, naming rule, size budget |
| `BigRedGym/tests/unit_tests/test_offline_replay.py` | 13 new tests |

### Result
✅ Implemented as planned. Gates: 20 focused / 321 portable / 28 gym / 7
learning tests, Ruff clean. Smoke found that `_rigid_body_state` is flattened
across environments and needed a `[N, -1]` view. Collected 1M transitions
(1,048,576 appended, last 1M retained) from resumed `model_600` on Warp at 4096
envs × 16 iterations: 52 fields, 1856 float32/transition, 7.458 GB, 8,088
`(env_id, episode_id)` groups, 3,997 timeouts, 8 terminations. Committed as
`Add selectable replay field recording`.

#### Files changed
| File | Change |
|------|--------|
| `BigRedGym/learning/offline/replay_fields.py` | New registry (`STEP`/`NEXT` attrs, groups, `resolve_replay_fields`) |
| `BigRedGym/learning/offline/contracts.py` | `REPLAY_FORMAT_V2`; optional validated `fields` |
| `BigRedGym/learning/storage/transition_recorder.py` | Lazy extra buffers, exact-set/width/finite checks, v2 manifest |
| `BigRedGym/learning/offline/dataset.py` | `_resolve_recorded_fields`, `_validate_extra_fields`, concat over declared set |
| `BigRedGym/learning/runners/on_policy_runner.py` | `_replay_features`, reward-term names, body/dof metadata |
| `BigRedGym/scripts/train.py` | `--replay_fields` |
| `BigRedGym/docs/offline_height_q.md` | "Selectable raw-state features" section |
| `BigRedGym/tests/unit_tests/test_offline_replay.py` | Round trip, rejection, legacy, concat tests |
| `docs/plans/replay_field_selection_plan.md` | This plan/report |
| `docs/journal.md` | This entry |

### Notes
Artifact: `/tmp/opencode/replay-1m-all-signals-20261113/replay_iter_0000616`
(run `logs/go2trot/Nov13_15-11-03_gpu-1m-all-signals-20261113b`, source
`146b041`). `contact_forces`/`rigid_body_states` are per body (42 for Go2), so
feet are indices 11/20/29/38 from `field_metadata["body_names"]`. No `dof_acc`
or `feet_air_time` exists in this codebase. Dataset is ~2 episodes per env; more
trajectory diversity needs more steps at fewer envs. No todos added.

## [2026-11-15] Foot contact-force reward r', Q_r', ranking and intervention

### Context
Third concrete offline reward path: a per-foot peak ground-force constraint
encoded as `r' = -1` when any foot exceeds its own threshold, else 0. The user
fixed the two open decisions: thresholds at the 90th percentile of each foot's
per-touchdown peak distribution, in bodyweight units; bodyweight recorded as a
new replay field rather than assumed constant.

### Plan
1. Add `LeggedRobot.bodyweight` and record it as a `contacts` replay field.
2. Add `ContactForceRewardSpec`, the labeller, threshold derivation, the
   `brg-contact-q-v1` artifact and `scripts/train_contact_q.py`.
3. Record a fresh contact-labelled 1M-transition snapshot and verify the label
   rate from the data alone.
4. Fit, then run the decisive tests: within-state spread and candidate ranking,
   and persistent max-Q / Q-threshold intervention with controls.
5. Document, move the BRG plans and journal into the product docs, commit.

### Result
✅ Pipeline and evidence complete; ❌ for one-step candidate ranking (protocol
ceiling), ✅ for persistent intervention.

The recorded dataset (4096 worlds, resumed `model_600`, `--replay_fields=contacts`)
holds 1,048,576 transitions with `contact_forces`, `next_contact_forces` and
`bodyweight` (152.3–184.2 N, constant per environment). Thresholds derived from
the training pool alone: `0.9526 / 0.9891 / 0.8631 / 0.8358` bodyweight, label
rate 2.826 % of steps / 10.00 % of touchdowns.

Fit (60k updates, γ=0.99, episode-grouped 80/20): held-out explained variance
0.978. By class, flagged rows (2.9 %) have MAE 0.611 with the Q under-predicting
the violation cost by ~0.5; unflagged MAE 0.103 — the overall RMSE hides this.

Candidate ranking (128 states x 8 candidates): within-state spread 2.36 (H100) /
2.65 (H500), Q Spearman 0.055 / 0.017 against a same-controller test-retest
ceiling of 0.059 / 0.084. A one-step intervention is noise-dominated, so no
ranker can help and Q is at the ceiling (the yaw path measured 0.007 / 0.019).

Persistent intervention (128 states x 8 lanes, H500, four seed blocks): plain
−2.467, argmin-Q −5.958 (control), random −2.587 (control, ≈ plain), **max-Q
−0.859** (gain +1.607 ± 0.049, violations 2.53 % → 0.89 %), rejection thresholds
interpolate (−1.507 / −1.308 / −1.109). PPO reward per step −0.15 %, xy tracking
+1.5 %, yaw error and survival unchanged: the constraint improvement is
essentially free, and the control ordering rules out a protocol artifact.

Gates: 334 portable, 28 gym, 7 learning, Ruff clean; lane fidelity
root_states 9.5e-5 vs 2e-3 and contact_forces 1.3e-3 vs 1.0, zero timeouts.

#### Files changed
| File | Change |
|---|---|
| `gym/envs/base/legged_robot.py`, `gym/envs/go2/go2trot.py` | `bodyweight` property, shared by the contact term |
| `learning/offline/replay_fields.py`, `learning/offline/height_q_playback.py` | `bodyweight` field in `contacts`; privileged-input guard |
| `learning/offline/contracts.py`, `rewards.py`, `training.py`, `q_artifact.py` | Contact spec, labeller, shared FQE core, `brg-contact-q-v1` |
| `learning/offline/persistent_intervention.py` | Contact rollouts (plain / max-Q / rejection) |
| `scripts/train_contact_q.py`, `diagnose_contact_candidate_ranking.py`, `evaluate_contact_q_persistent.py` | Train, rank and intervene CLIs |
| `tests/unit_tests/test_contact_q.py`, `test_offline_replay.py`, `test_domain_randomization.py` | Contact contracts; `concat_datasets` single/mismatched-width regressions |
| `docs/offline_contact_q.md`, `docs/offline_height_q.md`, skill | Third reward path documentation |
| `docs/plans/foot_contact_force_q_plan.md`, `docs/plans/supplements/foot_contact_study/` | Plan and study report |
| `docs/plans/*`, `docs/journal.md` | BRG plans and journal moved from the workspace docs tree |

### Notes
Artifacts stay in `logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/`
(gitignored). The label is behaviour-relative: playback and the stochastic
recording differ by 5–8 % in thresholds, verified by a matched stochastic replay.
`concat_datasets` crashed on a single snapshot and only shape-checked the last of
several — fixed with two regressions. No todos added. Ranking and intervention
are conditional on 128 fixed insertion states from one prefix at `model_616`;
Warp restore is not bitwise, so returns are paired comparisons.
