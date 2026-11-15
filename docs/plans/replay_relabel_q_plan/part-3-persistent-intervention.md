# Part 3 — persistent intervention, paired regrouping, extra critics and completion (S14-S17)

> Part of the replay/relabel Q plan. Index: [README.md](README.md).

## 14. Approved EDIT: persistent intervention — 2026-11-10

User explicitly extends earlier scope exclusions to permit this frozen-policy
experiment. H500, gamma .99, height .32±.05, absorbing true failures. Compare
plain stochastic model600 (candidate0) with max-original-fair100 and max-evaldist100
among eight fresh stochastic actions from the SAME frozen policy at each branch's
CURRENT observation, at EVERY decision. No retraining or privileged Q inputs.

Reuse original 128 k100 physical snapshots and fixed original noisy observations
from `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`; no prefix
rerun. Physical world layout [lane8,state128]; candidate8 is neural-only. Use
candidate0 source world per state, native exact restoration of every model/task/
controller/history/DR field, no clamp or timeout reset. Initial observations are
retained; later observation noise, candidate Gaussian draws and environment RNG
streams pair across controllers by (call seed,lane,state). Means follow each
branch's own observation. All lanes consume distinct vectorized subsequences.

Target 128 calls × 8 lanes = 1024 executions/state, 131072 rollouts/controller,
196608000 physical world-steps total. Retain reusable first benchmark block for
each controller. Stop for approval if projected collection exceeds 720s/controller;
do not silently reduce counts. Use GPU Q batching only after CPU/GPU normalization
and prediction agreement. Save resumable chunks, raw [controller,state,execution]
returns, first rewards, termination/step/OOB counts, preterminal command-tracking
sums and denominators, selection counts and Q gains. Fixed-state execution CIs
and state-cluster uncertainty are separate from survivor tail [-.6570483,0] and
paired-tail intervals. Include per-state means, win/tie/loss, SD and q05; occupancy,
falls and command tracking guard interpretation. Approximate nominal Q gives no
automatic policy-improvement theorem under persistent state-distribution changes.

Output: `/tmp/opencode/height-q-persistent-intervention-20261110-v1/`.
Validation: focused every-decision selection, candidate0 marginal, batched Q,
RNG pairing/lane distinction, masks/tails/axes, native fidelity and identical-
controller negative control, followed by portable/colocated gates and Ruff.

### 14.a Execution, audit and timings — 2026-11-10/11

Implementation: `BigRedGym/learning/offline/persistent_intervention.py`
(controller), `BigRedGym/scripts/evaluate_height_q_persistent.py` (simulator,
resume chunks), `BigRedGym/scripts/summarize_height_q_persistent.py`
(simulation-free audit/summary), `BigRedGym/tests/unit_tests/test_persistent_intervention.py`.
No product training path changed; all pre-existing CORL/config dirt preserved.

Benchmark command (foreground, output redirected):
`... evaluate_height_q_persistent.py --source /tmp/opencode/height-q-fixed-variance-20261110-v1
--output /tmp/opencode/height-q-persistent-intervention-20261110-v1
--checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt
--load_run Nov10_10-36-22_gpu-final-600-trot-hist-20261110 --artifacts <fair100>
<evaldist100> --blocks 1`.
Log `/tmp/opencode/height-q-persistent-benchmark.log`, final mtime 23:30:30 -0500.
Harness returned completion with no stdout (all output redirected); exit code was
not captured for this invocation. Retained block times: 29.0368 / 29.7233 /
29.6375 s (baseline/original/evaldist); projections 464.59 / 475.57 / 474.20 s,
inside the 720 s gate, so collection continued without an approval pause.

Full command: the same invocation with `--blocks 16 --resume`. Log
`/tmp/opencode/height-q-persistent-full.log`, mtime 23:52:54 -0500; 48 block files
present; terminal entries are block 15 for all three controllers. Foreground
call returned only on completion; exit code not explicitly captured. Resume
window 23:31:14–23:52:55 for 45 controller-blocks.

Post-run audit (one process/state check, 23:53:11): no
`evaluate_height_q_persistent` process alive; log size 13152 bytes with unchanged
mtime; all 16 blocks × 3 controllers on disk. Summary command exit status `0`
(`height-q-persistent-analysis.log`), diagnostics exit status `0`
(`height-q-persistent-diagnostics.log`). Data volume 58,082,078 bytes total,
19,351,696 bytes per controller, not placed in git.

Runtime model provenance: this session's tools expose no runtime model
identifier. The task states the session provider/model as `opencode-go` /
`deepseek-v4.1-flash`; that configuration is the only model provenance available
here and is not independently verifiable from inside the process.

### 14.b Results

All 128 states × 1024 paired executions per controller; executed decisions
65,016,291 / 64,994,521 / 64,980,358. Gates: source restore qpos 2.15e-6 rad,
qvel 1.03e-4 rad/s, contact 1.50e-3 N; replication qpos 1.43e-6, contact 5.65e-4;
82-field inventory exact; CPU/GPU Q max error 5.7e-6/7.6e-6 and split-batch
9.5e-6/1.1e-5 with 0 argmax disagreements; identical plain controllers exactly
equal over 5 steps; zero timeouts; selection counts equal live decisions.

| Quantity | baseline | original-Q | evaldist-Q |
|---|---:|---:|---:|
| Mean H500 return | -5.2858 | **-3.5666** | **-4.1758** |
| Execution CI (fixed states) | [-5.3105,-5.2610] | [-3.5875,-3.5458] | [-4.1994,-4.1521] |
| State-cluster CI | [-6.4766,-4.1763] | [-4.4693,-2.7341] | [-5.1939,-3.2461] |
| Tail-only interval | [-5.9287,-5.2858] | [-4.2090,-3.5666] | [-4.8179,-4.1758] |
| Inside-band step fraction | 95.460% | 97.177% | 96.553% |
| Fall fraction | 2.155% | 2.242% | 2.275% |
| XY / yaw tracking RMSE | .4493 / .2208 | .4561 / .2259 | .4549 / .2254 |
| Speed / command speed | .7426 | .7362 | .7383 |

Paired gains: original +1.7191 (per-state SD 2.0784, median +0.6595, q05 +0.0905,
range +0.0055…+11.978; 128/0/0 win/tie/loss; execution CI [+1.6917,+1.7466],
state CI [+1.3781,+2.0850], tail interval [+1.0763,+2.3615]); evaldist +1.1100
(SD 1.5488, median +0.4477, q05 +0.0263, range -2.160…+6.713; 123/0/5;
execution CI [+1.0817,+1.1384], state CI [+0.8564,+1.3840], tail interval
[+0.4671,+1.7521]). Block halves: +1.7006/+1.7376 and +1.0963/+1.1238.
Signed behaviour changes (controller − baseline): outside-band step rate
**−1.73 pp** (state CI [−2.03,−1.45]) and **−1.08 pp** ([−1.28,−0.90]); fall rate
**+0.086 pp** ([−0.038,+0.212]) and **+0.120 pp** ([0.000,+0.244]). Tracking
degrades ~1.5%, the direction the height-only reward predicts.

Selected **indices** are near-uniform (12.4931–12.5059%). That is the
exchangeability prediction for eight IID candidates under any exchangeable
critic, including a perfect one, so it says nothing about mechanism; the executed
action is a Q-selected order statistic, not a sample from the nominal policy.
Without a same-bias/no-information control (not run, and a random index control
would reproduce the nominal marginal), Q information cannot be separated from
selection bias here. The critic's predicted **per-decision** advantage (0.1226 /
0.2056) is a single-state quantity under the nominal distribution, not a forecast
of the realised cumulative 500-step gain; the numerical ratio is not an error or
underforecast.

Interpretation: this is the plan's first positive **policy-intervention** result,
but it is not policy improvement in the theorem sense — the frozen Q's targets
are off-policy once the branch deviates — and it is not validation of approximate
Q under persistent deviation. No `Q^pi` accuracy, transfer, threshold-filter,
safety or generalization claim. Tail and sampling uncertainty are reported
separately; restoration is one-step numerical, not bitwise long-horizon.

### 14.b.1 Post-hoc audit corrections — 2026-11-11

Raw blocks were not rerun; both corrections recompute from the same 48 chunks
(`audit_corrected.json`, corrected `results.json`):

1. **Paired truncation interval had been doubled.** Each survivor's omitted tail
   is [-0.6570483,0], so the paired mean interval uses the baseline survivor
   fraction on the lower side and the controller fraction on the upper side. The
   first draft added both fractions to both sides. Analytic two-execution oracle:
   true width 0.657 versus old-formula width 1.314, ratio exactly 2.0. Corrected
   original [+1.0763,+2.3615] (was [+0.4339,+3.0043]); evaldist
   [+0.4671,+1.7521] (was [-0.1750,+2.3950]). Single-controller tail intervals
   were already correct.
2. **Occupancy/fall CIs had mixed sign conventions.** Point estimates and CIs
   are now both controller − baseline with the direction named. Consequence: the
   fall increase is a small point-estimate degradation whose CI touches zero for
   evaldist, not a resolved harm.

Re-audited: seeds identical across controllers, zero timeouts, selection sums
equal live decisions, and the analytic bound matches the interval endpoints. The
asymmetric bound is now a tested helper
(`paired_survivor_tail_interval` in `learning/offline/persistent_intervention.py`)
with a corner-enumeration regression test, and the summarizer plus
`audit_corrected.json` reproduce the corrected intervals.

### 14.c Verification record

| Gate | Command | Result |
|---|---|---|
| Focused | `pytest tests/unit_tests/test_persistent_intervention.py tests/unit_tests/test_fixed_state_variance.py -q` | 11 passed |
| Portable | `pytest -q` | 233 passed, 97 deselected (EXIT=0) |
| Colocated | `pytest gym -q`; `pytest learning -q` | 28 / 7 passed (EXIT=0) |
| Warp | `pytest -q -m warp` | 32 passed, 298 deselected, 18 pre-existing MULTICCD warnings (EXIT=0) |
| Static | `ruff format` touched files; `ruff check .`; `git diff --check` | clean |
| Data audit | `scripts/summarize_height_q_persistent.py` | shapes [128,1024], binary rewards, counts match, zero timeouts (EXIT=0) |

Logs: `height-q-persistent-{benchmark,full,analysis,diagnostics,portable,gym,learning,warp}.log`
under `/tmp/opencode/`. Report:
[durable copy](../supplements/replay_relabel_q/persistent-intervention-20261110-v1/REPORT.md).
Plan stays in EDIT; no commits, todos, completion or archival.

## 15. Paired 16-execution regrouping — merged 2026-11-11

Source report:
[durable copy](../supplements/replay_relabel_q/paired16-comparison-20261111-v1/REPORT.md)
(raw blocks, analysis script, manifest, commands and true exit statuses retained
in the original artifact directory). The analysis imported no product code; the 32 `baseline`/`original`
blocks are `[state128, execution64]` with `seed = 7100000 + 10*(block*8 + call)`
and `e = call*8 + lane`. The 16 `evaldist` blocks were present but excluded.
`evaldist` is therefore not part of any statistic in this section.

### 15.a Result

For each of the 128 fixed configurations and 1024 executions, the paired gain is
`G = G_original - G_plain`. The values below restate the report; they do not
replace the report's formulas.

| Quantity | Value |
|---|---:|
| Paired executions | 131072 |
| Mean paired gain | **1.719137** |
| Fixed-state execution CI95 | [1.691714, 1.746561] |
| State-cluster bootstrap CI95 | [1.371995, 2.083445] |
| H500 paired survivor-tail interval | [1.076251, 2.361458] |
| Individual execution win / tie / loss | **0.569130 / 0.212204 / 0.218666** |
| 64 batch16 positive / tie / negative | **0.854858 / 0.000000 / 0.145142** |
| Per-state mean gain positive | **128/128** |
| Per-state tail lower bound positive | 67/128 |

Per-state mean-gain distribution: median 0.661046, q05 0.090459, range
0.005541…11.978000. The 27814 individual ties are exact equalities (27358 both
returns zero, 456 identical nonzero). The primary batch is contiguous
`e//16`; the stratified `(call,lane)` partition gives 0.856079 positive, the
same mean, and one state at exactly 32/64. The mean is invariant to regrouping.
The H500 paired interval is asymmetric: baseline survivor fraction on the lower
side, original survivor fraction on the upper side. It is an omitted-tail
uncertainty interval, not a sampling CI.

### 15.b Mandatory caveats

- The 56.9% individual win rate and the 85.5% batch16 positive fraction are
  different estimands. Batch16 averaging reduces the noise that produces zero
  or negative individual gains; it does **not** raise the mean gain (still
  1.719137), make a typical execution more likely to improve, or estimate
  expected utility. A batch16 positive fraction is not `E[G]`.
- 21.2% of executions tie and 21.9% lose; 48 states have a raw execution win
  probability below 0.5 even though every state-level mean gain is positive.
  The gain is carried by the magnitude of wins and avoided violations.
- These are 128 fixed k=100 states from one frozen policy and one source lane,
  not a random state population. Execution rows share state, seeds and lanes;
  execution-level CIs are conditional precision, not independent replication.
- The primary estimand is a **policy-intervention** measurement at fixed states,
  not `Q^pi` validation, not policy improvement in the theorem sense, and not a
  guarantee that the whole state distribution improves.
- The return stops at H500. The global paired tail interval is strictly positive,
  but 61/128 per-state tail lower bounds include zero.
- Two of four recorded run-time source hashes differ from the current working
  tree; the raw blocks are authoritative. No cross-critic or `evaldist` claim is
  made here.

## 16. Approved EDIT: two additional Q critics and reused-baseline evaluation — 2026-11-11

### 16.a User request and estimand

The user explicitly requested, after the paired16 comparison, two additional
height-Q critics compared on the same fixed states and 1024 execution seeds:

1. **mixed critic**: refit on the existing 2M mixed PPO-collected replay
   (`logs/go2trot/hist-mix-20261110/replay/replay_iter_0000600`);
2. **fresh critic**: refit on 1M fresh transitions obtained by unrolling the
   frozen nominal stochastic PPO policy `model_600` in the normal training
   environment.

This is a data-composition / data-provenance comparison at equal epochs, not an
equal-data-size comparison: mixed has 2M transitions, fresh has 1M. It is a
fixed-policy evaluation exercise, not policy optimization. The four-critic table
will report original, evaldist, mixed and fresh against the same plain baseline;
it will not claim that any critic validates `Q^pi`, that repeated intervention
is an unbiased value estimate, or that noise ceilings make differences
"significant".

### 16.b Frozen controls and equal-epoch budget

- Checkpoint: `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt`,
  sha256 `f88b1f9b829675c97ed0c420202cf54456dc414641def34e379e39407ee3adf6`.
- Reward: binary outside-band `-1`, inside-band `0`, reference 0.32 m, bound
  ±0.05 m, `post_action_pre_reset`, absorbing true failure, gamma 0.99.
- FQE architecture and optimizer: identical `QNetwork` 48→256→256→1, ReLU,
  Adam 3e-4, batch 256, target tau 0.005, one target-action sample, observation
  normalization, seed 0. No architecture or hyperparameter tuning against the
  evaluation set.
- Original `.../fair-v1/100pct` manifest: 1,000,000 transitions, 400,000 updates,
  batch 256 → 102.4 epochs. The mixed 2M critic therefore gets **800,000**
  updates (same 102.4 epochs); the fresh 1M critic gets **400,000** updates.
- The original artifact is fixed and hashed; only its stored returns are reused.

### 16.c Existing mixed dataset validation

Before training, validate the immutable mixed snapshot metadata and arrays:
2,000,000 rows, 48/12 dimensions, all replay fields, finite tensors, bool
flags, int64 provenance, `reward_timing = post_action_pre_reset`,
`observation_noise_convention = recorded_policy_observation`, the recorded
`config_fingerprint = 6412a878…`, the early/final parent snapshots and the
equal-size 1M+1M composition. The snapshot is read-only; no rewrite.

### 16.d Fresh 1M collection protocol

- Build the normal `go2trot` training environment through the task registry with
  the registered config and the existing 5 s administrative timeout
  (`episode_length_s = 5`, 50 Hz control, 250 policy steps). No evaluation-style
  horizon extension.
- 4096 vectorized worlds on `cuda:0` (MuJoCo Warp), nominal stochastic policy
  only: every action is `mean + std * N(0,1)` from the frozen actor. No
  candidate search, no Q selection, no deterministic mean action override.
- Disjoint seed `20261111`, which is outside the 128-state evaluation seeds
  `7100000 + 10*(...)` and the saved-state seed `2400000`. Environment seed and
  dedicated action/observation generators are derived from it.
- Run about 2.5M total env steps and publish the last 1M transitions from the
  immutable ring. This gives roughly 2.4 episodes per world of independent
  DR/command/phase resets while retaining exactly 1M rows.
- Reuse `TransitionRecorder` semantics: actor observation is the policy-facing
  noisy observation; action is in BRG policy units; `successor_height_m` and the
  pre-reset clean successor observation are captured immediately after
  `env.step()` and before reset; ended rows use pre-reset successor plus
  recording-only noise; `terminated` and `timed_out` stay independent. Reset and
  timeout handling therefore match normal training, including timeout
  bootstrapping and absorbing true failure.
- Assert the environment fingerprint equals `6412a878…`, the checkpoint hash
  matches, and the published snapshot has exactly 1,000,000 valid rows and
  validates through `load_snapshot`.
- Benchmark collection time and memory first. If collection is not a minor
  cost, report the measured budget rather than silently shrinking counts.

### 16.e Training, evaluation and analysis

- Train each critic with `scripts/train_height_q.py` on `--fractions 1.0`,
  `--subset_group episode`, seed 0, `device cuda:0`, the budgets in §16.b.
  Record steps, epochs, wall time, loss trace summary (not learned quality), and
  the resulting artifact hashes.
- Benchmark training first with a short update count. If either full critic is
  projected to exceed ~1 hour, stop and report options rather than launch the
  full grid.
- Evaluate each new critic with the persistent sample-eight controller using the
  same 128 saved k=100 configurations, the same 1024 execution seeds
  `7100000 + 10*(block*8 + call)`, 1024 Warp worlds, H500, and eight fresh
  stochastic policy actions at every decision. Reuse the existing
  baseline/original/evaldist raw blocks from
  `/tmp/opencode/height-q-persistent-intervention-20261110-v1/`; do not rerun
  those three. Only `mixed` and `fresh` are simulated in the new directory.
- Verify before full evaluation: identical source restoration/seed identity,
  checkpoint hash, reward/gamma, artifact policy spec, baseline raw-block
  presence and shapes, CPU/GPU Q agreement, and a short identical-controller
  negative control. Do not clamp model parameters; use the existing validated
  native restore adapter and source mapping.
- Analysis uses the same definitions as §15 and the persistent summarizer:
  per-state paired means, individual win/tie/loss over 131072 pairs, primary
  contiguous 64 batch16 means per state, state-cluster bootstrap CIs,
  per-controller survivor-tail intervals, and the corrected asymmetric paired
  tail interval (`paired_survivor_tail_interval`). Report return, inside-band
  occupancy, falls, and command tracking. Raw first states, seed lists and
  execution ordering must be identical across all five controllers.

### 16.f Outputs and boundaries

- Unique output root:
  `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/`, containing the
  fresh snapshot, both Q artifacts, new controller blocks, validation, analysis,
  commands with true exit statuses, provenance and `REPORT.md`.
- Reproducibility manifest distinguishes the old run's recorded code hashes from
  the new collector/trainer/evaluator/analysis hashes; do not rewrite old
  snapshots or artifacts.
- Large data and checkpoints stay under `/tmp/opencode/` and gitignored
  `logs/`; no large committed file. No commits, todos, plan completion or
  archival. Unrelated dirty work is preserved.
- Explicitly report data budgets (worlds, episodes, resets, retained rows),
  FQE steps/epochs/time/loss, policy-selection performance, and any blocker.

### 16.g Execution and results — 2026-11-11

Output root `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/`, report
`REPORT.md`, commands/exit audit `commands.log`, provenance
`experiment_manifest.json`. All runs were foreground; every recorded
`/usr/bin/time` exit status is 0.

**Fresh data.** 4096 Warp worlds, nominal stochastic `model_600` only, seed
20261111, 5 s / 250-step timeout. 611 policy steps = 2,502,656 appended
transitions; ring retained the last 1,000,000, spanning 8,192 unique env/episode
pairs (about 2 episodes/world), 15 true terminations and 4,082 timeouts.
Collection wall 20.49 s (26.47 s including startup). Snapshot and
`collect_provenance.json` hashes are in `experiment_manifest.json`; fingerprint
`6412a878…` and pre-reset labels validated. The mixed 2M snapshot was read-only;
its 2M/48/12 shapes, finite fields, flags, episode/step ids, fingerprint and
`post_action_pre_reset` convention were audited before training.

**Training.** Same architecture/optimizer/reward/gamma/seed; mixed 800,000
updates on 2M (102.4 epochs), fresh 400,000 on 1M (102.4 epochs). Wall time
745.76 s and 371.89 s; mean loss 6.554596 and 0.337882 (loss is a fit
diagnostic, not learned-quality evidence). Policy specs match the original
fair100 artifact exactly; `q.pt` hashes are in the manifest.

**Evaluation.** Baseline/original/evaldist blocks reused from
`/tmp/opencode/height-q-persistent-intervention-20261110-v1/`; only mixed/fresh
were simulated. One-block benchmark 29.43 / 29.52 s → projected 470.8 / 472.3 s
per controller, inside the 720 s gate. Full 16 blocks per new controller:
895.74 s. Source restore (root_states 6.89e-5, dof_pos 1.67e-6, dof_vel
8.24e-5, contact 7.35e-3), replication/lane fidelity, 82-field inventory,
CPU/GPU Q (4.01e-5 / 9.54e-6, zero argmax disagreements), five-step identical
control, zero timeouts and selection-count accounting all passed. Seeds and
source/checkpoint hashes match the reused run; no baseline rerun.

| Controller | Mean H500 | Paired gain | Individual W/T/L | Batch16 positive | State-cluster CI | Paired tail |
|---|---:|---:|---|---:|---|---|
| baseline | -5.2858 | — | — | — | [-6.5552,-4.1916] | [-5.9287,-5.2858] |
| original | -3.5666 | **1.7191** | 56.91/21.22/21.87% | **85.49%** | [1.3720,2.0834] | [1.0763,2.3615] |
| evaldist | -4.1758 | **1.1100** | 51.54/19.90/28.56% | 75.52% | [0.8493,1.3776] | [0.4671,1.7521] |
| mixed | -4.3765 | **0.9093** | 48.15/19.01/32.84% | 72.88% | [0.6708,1.1674] | [0.2664,1.5574] |
| fresh | -3.5946 | **1.6911** | 56.14/20.47/23.39% | 83.06% | [1.3430,2.0582] | [1.0483,2.3351] |

Per-state means are positive in 128/128 (original), 123/128 (evaldist), 120/128
(mixed) and 125/128 (fresh); per-state paired-tail lower bounds are positive in
67, 54, 46 and 66 of 128. Inside-band step fractions are 97.18% (original),
96.55% (evaldist), 96.34% (mixed) and 97.20% (fresh) versus 95.46% baseline;
fall fractions are 2.24%, 2.28%, 1.36% and 1.99% versus 2.16%. XY RMSE is
0.4561/0.4549/0.4619/0.4579 versus 0.4493, and speed/command is
0.7362/0.7383/0.7195/0.7292 versus 0.7426. Selected indices are near-uniform
for every Q controller (candidate0 ≈ 0.125; mean per-decision Q gain 0.1226,
0.2056, 0.1709, 0.1116), which is exchangeability, not mechanism evidence.

The fresh critic is close to original on these fixed states and the mixed critic
is lower. This is **not** a clean data-size/data-quality effect: mixed is 2M of
early+final PPO data, fresh is 1M nominal `model_600`, and their composition
differs. No `Q^pi` validation, policy-improvement theorem, state-population or
noise-ceiling claim is made. The v1 `persistent_intervention.py` hash differs
from the current tree (plan §14.b.1 attributes that to the post-hoc tail/sign
correction; this task did not edit rollout selection), so baseline/original/
evaldist are reused with seed/source/checkpoint identity checks rather than
bitwise code identity. This limitation is recorded in the report.

Validation: focused 22 passed; portable 233 passed / 97 deselected; gym 28;
learning 7; Warp 32 passed / 298 deselected (18 pre-existing MULTICCD warnings);
`ruff check .` and `git diff --check` clean. Plan stays in EDIT; no commits,
todos, completion or archival.


### 16.h Return SD and paired Student t-test addendum — 2026-11-11

User request: add descriptive return SD to the five-controller mean table and a
correct paired Student t-test against the plain baseline for the four Q
controllers, with raw and Holm-adjusted p values. This is a **statistics-only
addendum** to §16.g: no simulator, rollout, training or original collection was
run or modified. The reused baseline/original/evaldist blocks and the new
mixed/fresh blocks were read from their existing paths.

**Method.** The descriptive SD is the pooled SD of all 131,072 individual H500
returns per controller (128 states × 1024 matched seeds, ddof=1); it is outcome
spread, not the standard error of the mean. A separate SD of the 128 state-mean
returns (ddof=1) is reported. The primary test is paired, not independent
two-sample: for each state `s`,
`d_s = (1/1024) Σ_e (R_ctrl[s,e] - R_base[s,e])`, then a one-sample Student t
test on the 128 `d_s` values with `t = mean(d)/(SD(d)/√128)`, df=127,
two-sided. The 131,072 executions are correlated and are never treated as IID;
no execution-level p value is reported. Holm adjustment covers the four
controller-versus-baseline comparisons (m=4). CIs use the two-sided 0.975
Student t quantile at df=127. Tiny p values are printed in scientific notation,
never as zero.

| Controller | Mean H500 ± pooled individual SD | SD of 128 state means | Mean gain | Gain SD | SE | t(127) | Raw p | Holm p | 95% CI gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | -5.2858 ± 8.1216 | 6.7391 | — | — | — | — | — | — | — |
| original | -3.5666 ± 6.3288 | 5.0477 | 1.7191 | 2.0784 | 0.18371 | 9.3579 | 3.765e-16 | 1.506e-15 | [1.3556, 2.0827] |
| evaldist | -4.1758 ± 7.1375 | 5.6627 | 1.1100 | 1.5488 | 0.13690 | 8.1085 | 3.726e-13 | 7.452e-13 | [0.8391, 1.3809] |
| mixed | -4.3765 ± 6.8288 | 5.6668 | 0.9093 | 1.4604 | 0.12908 | 7.0441 | 1.050e-10 | 1.050e-10 | [0.6538, 1.1647] |
| fresh | -3.5946 ± 6.4504 | 5.1106 | 1.6911 | 2.0959 | 0.18525 | 9.1288 | 1.355e-15 | 4.064e-15 | [1.3246, 2.0577] |

**Interpretation.** All four paired mean gains remain nonzero after Holm
adjustment (`p ≤ 1.05e-10`), but the p values are precision statements about the
fixed-state mean difference, not evidence that a typical execution improves.
The mean gains are only 0.13–0.27 of the pooled individual SD, individual win
fractions are 48–57% with 19–21% ties, and the paired differences are much
tighter (gain SD 1.46–2.10; paired `mean gain / gain SD = 0.62–0.83`). The
across-controller ordering is broadly inverse—the two smallest state-mean SDs
(original 5.0477, fresh 5.1106) have the largest gains (1.7191, 1.6911), while
the two larger ones (evaldist 5.6627, mixed 5.6668) have smaller gains (1.1100,
0.9093)—but evaldist and mixed are nearly tied in state-mean SD while differing
in gain, and only four fixed-state summaries are compared. This is therefore a
descriptive "more outcome variance, smaller gain" pattern, not a fitted or
causal variance–gain law, and not a claim that the baseline's larger pooled SD
(8.1216) makes its mean less precise.

**Boundaries.** The 128 k=100 configurations are fixed snapshots from one
policy and one source lane and share the same seed list across states; the t
test assumes independent configurations conditional on this design and has only
128 configuration-level units. It is not a state-population,
checkpoint-generalization or safety claim. The return is truncated at finite
H500, so no p value concerns the infinite-horizon return.

**Verification and audit.** `analysis/return_stats.py` reads only the raw
`[128,64]` blocks, reproduces all five `analysis/results.json` means and
state-mean SDs and all four paired gain means/SDs with exact float equality,
computes p by regularized incomplete beta and by independent quadrature
(agreement), and checks translation/scale/permutation/sign invariance plus the
known `[1,2,3,4]` paired sample (`t=√15`, `p=0.0304663`).
`analysis/verify_scipy.py` with system SciPy 1.8.0 recomputes `ttest_rel`,
`ttest_1samp`, t quantile, CIs and Holm adjustment and agrees. Both commands ran
in the foreground with exit status 0; exact invocations are appended to
`commands.log`. `ruff format --check` / `ruff check` passed on both scripts.
Artifacts: `analysis/return_stats.json`, `return_stats.md`,
`state_means_for_scipy.npz`, `scipy_verification.json`, their stdout/time logs,
and the updated `REPORT.md` and `analysis/REPORT.md`. Durable migrated copies:
[main report](../supplements/replay_relabel_q/two-critic-datasets-20261111-v1/REPORT.md),
[analysis report](../supplements/replay_relabel_q/two-critic-datasets-20261111-v1/analysis/REPORT.md),
and [return-stat addendum](../supplements/replay_relabel_q/two-critic-datasets-20261111-v1/analysis/return_stats.md).
No product source, snapshot, Q artifact or original collection was changed. Plan remains in EDIT;
no commits, todos, completion or archival.

## 17. Completion and archive — 2026-11-11

Status: **completed 2026-11-11**. The historical "plan remains active/EDIT" notes
above describe the state at their writing; this section supersedes them. The plan
is archived at
`docs/plans/replay_relabel_q_plan/README.md`. No experiment data,
checkpoint or generated artifact was added to git.

Delivered implementation: bounded opt-in replay recording and immutable snapshots;
height relabeling and nested/early-late/mixed dataset construction; frozen
stochastic nominal-policy loading; continuing-return FQE with timeout bootstrapping
and absorbing terminal failure; Q artifact export; single- and batched-rollout
evaluation; fixed-state native snapshot variance experiments; persistent
sample-eight intervention; frozen-policy fresh-data collection; and simulation-free
summarizers. Product code, tests, offline documentation, test catalog, MuJoCo
limitations, and Q2 skill guidance were updated. Unrelated CORL/config work in the
dirty checkout was not included.

Final verification on 2026-11-11 (documentation/archive changes only; no product
source was edited in this step):

| Gate | Result |
|---|---|
| Portable `pytest -q` | **233 passed, 97 deselected** |
| Colocated `pytest gym -q` / `pytest learning -q` | **28 / 7 passed** |
| `ruff check .` | **All checks passed** |
| `ruff format --check` on all touched Python files | **30 files already formatted** |
| `git diff --check` | **clean** |

Scientific limitations preserved:

- Q receives only the 48-dim actor observation. Height is reconstructible from
  that observation but the critic remains observation-only; hidden physical state,
  randomized dynamics, and partial observability can bias Bellman targets.
- Variance and intervention results use 128 fixed `k = 100` states from one
  frozen checkpoint and source lane, 1024 executions per state/controller, H500,
  gamma 0.99, and the binary `0.32 ± 0.05 m` height reward. They are not a random
  state population, checkpoint-transfer, or whole-distribution guarantee.
- The persistent sample-eight experiment is a **policy intervention**, not policy
  improvement in the theorem sense and not `Q^pi` validation. Persistent
  deviations move the branch off the frozen policy's distribution, invalidating
  the critic's off-policy targets. It is not a safety, threshold-filter, repeated
  intervention, or generalization claim. Selected indices are near-uniform, which
  is expected under exchangeability and does not identify a mechanism.
- The reported gains come with slightly higher fall rate and slightly worse
  command tracking; execution-level and state-cluster uncertainty differ. The
  per-decision Q advantage is not a forecast of the cumulative H500 gain.
- No noise ceiling is established. Estimated means can be noisy, and the earlier
  noise-ceiling, sorted-repetition, and mid-gait precision claims are superseded.
- Returns stop at finite H500. Surviving-rollout omitted tails are reported as
  intervals, separate from sampling uncertainty; late failures beyond the
  training episode length remain a coverage limitation.
- Native restoration is numerically validated for one step and full buffer
  equality, not bitwise long-horizon trajectory equivalence. Warp solver noise can
  be amplified by closed-loop dynamics.
- Mixed/fresh critic differences are data-composition/provenance comparisons at
  equal epochs, not a clean data-size or data-quality effect.
- Top-Q-based runtime filtering, other reward channels, bound conditioning,
  finite-horizon Q, history/estimators, MPPI, and safety certification remain
  outside the validated scope.

