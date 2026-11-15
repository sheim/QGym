# Foot contact-force Q — offline reward `r'`, `Q_r'`, and intervention

Status: **COMPLETED AND ARCHIVED — 2026-11-15.** The per-foot peak-load
constraint is implemented as the third offline reward path, trained on a fresh
1.05M-transition contact-labelled recording, and validated in the simulator:
candidate ranking is at the protocol's noise ceiling, while a persistent max-Q
intervention reduces the discounted constraint cost by 65 % at no measured cost
to the original PPO reward.

Durable documentation: [`docs/offline_contact_q.md`](../offline_contact_q.md).
Study and measurement report:
[`docs/plans/supplements/foot_contact_study/REPORT.md`](supplements/foot_contact_study/REPORT.md).

## 1. Goal

### 1.a. Goal statements
- Encode a foot ground-reaction-force constraint as an offline reward `r'` that a
  fixed-policy Q can learn from recorded transitions.
- Train `Q_{r'}^{π_r}` for the frozen `go2trot` PPO policy with the actor
  observation as the only policy-facing input.
- Test whether `Q_r'` can (a) rank candidate first actions and (b) intervene
  persistently to reduce constraint violations without wrecking the task reward.

### 1.b. Success criteria
- A labelled dataset exists with the forces and the normaliser the label needs.
- `r'` is defined exactly, bounded, and reproducible from a snapshot.
- The artifact passes the same identity/parity gates as the other reward paths.
- Ranking and intervention are measured against controls, not asserted.

### 1.c. Constraints
- `r'` uses the actor observation only; forces and bodyweight are privileged.
- One reward channel per artifact; no re-labelling of another path's Q.
- No change to PPO training behaviour; recording is opt-in.
- Thresholds are behaviour-relative and must travel with the artifact.

### 1.d. Questions (answered)
- Force source: `env.contact_forces` per body, world frame, Newtons, net contact
  force — 96.8 % of trot samples have no non-foot contact, so the foot vector is
  a pure ground reaction. Answered by decomposition (§4).
- Threshold definition: per-leg 90th percentile of the per-touchdown peak load,
  in bodyweight units. Answered by the user.
- Bodyweight: recorded per transition (`LeggedRobot.bodyweight`), not a nominal
  constant — randomisation spans 152.3–184.2 N. Answered by the user.
- Label form: `-1` if any foot exceeds its threshold, else `0`. Answered by the
  user; a per-leg variant is a one-line change.

## 2. Implementation

| Piece | Location |
|---|---|
| `LeggedRobot.bodyweight` (live, follows randomised masses) | `gym/envs/base/legged_robot.py` |
| `bodyweight` as a replay field inside the `contacts` group | `learning/offline/replay_fields.py` |
| `ContactForceRewardSpec` (`foot_names`, `thresholds_bw`, `percentile`, timing, zero-after-terminal) | `learning/offline/contracts.py` |
| `foot_vertical_load`, `contact_force_reward`, `relabel_contact_force`, `derive_contact_thresholds` | `learning/offline/rewards.py` |
| Shared FQE core + `train_contact_q` | `learning/offline/training.py` |
| `ContactQ`, `save_contact_q`, `brg-contact-q-v1` | `learning/offline/q_artifact.py` |
| Contact rollouts (plain / max-Q / rejection) | `learning/offline/persistent_intervention.py` |
| CLIs: train, rank, intervene | `scripts/train_contact_q.py`, `scripts/diagnose_contact_candidate_ranking.py`, `scripts/evaluate_contact_q_persistent.py` |
| Tests | `tests/unit_tests/test_contact_q.py` (9) |

Two defects found while integrating and fixed with regressions:
`concat_datasets()` crashed on a single snapshot and shape-checked only the last
of several; `contact_force_reward()` built its thresholds on CPU and broke any
CUDA use. `_bellman_metrics()` now also reports target spread, explained
variance and residuals split by label class.

## 3. Data

Recording `logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113`, resumed
`model_600`, 4096 worlds × 16 iterations, snapshots at 604/608/612/616,
`--replay_fields=contacts` (forces + bodyweight):

| item | value |
|---|---|
| transitions | 1,048,576 (3.7 GB with all four snapshots) |
| fields | `contact_forces`, `next_contact_forces` (`[N, 126]`), `bodyweight` (`[N, 1]`) |
| bodyweight | 152.3–184.2 N (mean 167.6, sd 6.4), constant within an environment |
| terminated rows | 0.10 %, reward 0 and excluded from the peak estimate |

Verified from the snapshot (`verify_contact_snapshot.py`): label rate 2.83 % of
steps, 10.00 % of touchdowns, per-leg 0.84/0.84/0.61/0.61 %.

## 4. Why the foot force is only the ground reaction

Decomposing the raw MuJoCo contact list (4 worlds × 200 steps) into per-body
attributions reproduces `contact_forces` with **max abs error 0.0 N**:

| contact kind | count |
|---|---|
| foot ↔ ground plane | 1957 |
| foot ↔ another robot body | 0 |
| calf shell ↔ ground plane | 261 |
| robot body ↔ robot body | 0 |

So the feet only ever touch the ground here, while the calf shells do touch it
(12.2 % of steps) and `go2trot` has no active `collision` weight for them.
`contact_forces` is additive per body and would include self-collision if it
happened.

## 5. Results

### 5.1 Thresholds and label rate

| threshold family | FL | FR | RL | RR | steps flagged | touchdowns flagged |
|---|---|---|---|---|---|---|
| deterministic playback (study) | 0.900 | 0.968 | 0.787 | 0.778 | 2.82 % | 10.0 % |
| stochastic replay (matched) | 0.951 | 0.986 | 0.845 | 0.823 | — | — |
| **recorded dataset (used)** | **0.9526** | **0.9891** | **0.8631** | **0.8358** | **2.826 %** | **10.00 %** |

At mean bodyweight the thresholds are 160 / 166 / 145 / 140 N. The 5–8 % gap
between playback and recording is the stochastic training actions (σ ≈ 0.246),
not pushes, policy drift or reset transients — each was tested separately.

### 5.2 Fit (60k updates, 80/20 episode split, γ = 0.99)

| update | held-out RMSE | MAE | explained variance | target mean |
|---|---|---|---|---|
| 0 | 0.183 | 0.077 | 0.006 | +0.08 |
| 10k | 0.207 | 0.105 | 0.935 | −0.74 |
| 20k | 0.222 | 0.107 | 0.967 | −1.24 |
| 40k | 0.245 | 0.113 | 0.976 | −1.78 |
| 60k | 0.260 | 0.118 | **0.978** | −1.99 |

By label class at 60k (held out): flagged rows (2.9 %) MAE 0.611, prediction
−4.156 vs target −4.649 — the Q **under-predicts a violation by ~0.5**;
unflagged MAE 0.103. Raw RMSE grows while the fit improves because the FQE
target contains the Q's own magnitude.

### 5.3 Candidate ranking and within-state spread

128 states × 8 candidate first actions (mean + 1σ draw, candidate 0 = mean),
then the frozen policy for H steps; return is the contact label with
zero-after-terminal at γ = 0.99.

| H | within-state spread (p50) | Q Spearman (95 % CI) | same-controller ceiling | argmax hit (chance 0.125) |
|---|---|---|---|---|
| 100 | 2.36 (1.87) | 0.055 (−0.007, 0.118) | **0.059** | 0.141 |
| 500 | 2.65 (2.12) | 0.017 (−0.046, 0.082) | **0.084** | 0.125 |

A single first action followed by a stochastic policy is noise-dominated: the
same candidate set re-run under another seed ranks at 0.06–0.08, so no ranker
can beat chance on this protocol and the Q is at that ceiling. The yaw path
measured the same ceiling (0.007/0.019).

### 5.4 Persistent intervention

128 states × 8 lanes, H = 500, four blocks with independent seeds; every
controller of a block consumes the same eight-draw block, so the comparison is
paired per state. Mean ± sd over blocks.

| controller | contact return | violation rate | gain vs plain | PPO reward/step | xy err | survival |
|---|---|---|---|---|---|---|
| plain | −2.467 ± 0.074 | 2.53 % | 0 | 0.2745 | 0.396 | 0.980 |
| argmin-Q (control) | −5.958 ± 0.161 | 6.20 % | −3.491 ± 0.118 | 0.2728 | 0.404 | 0.975 |
| random (control) | −2.587 ± 0.178 | 2.65 % | −0.120 ± 0.245 | 0.2744 | 0.394 | 0.981 |
| **max-Q** | **−0.859 ± 0.041** | **0.89 %** | **+1.607 ± 0.049** | 0.2741 | 0.402 | 0.982 |
| reject −2.5 | −1.507 ± 0.052 | 1.53 % | +0.960 ± 0.040 | 0.2743 | 0.399 | 0.982 |
| reject −2.0 | −1.308 ± 0.060 | 1.35 % | +1.159 ± 0.066 | 0.2743 | 0.400 | 0.981 |
| reject −1.5 | −1.109 ± 0.044 | 1.13 % | +1.358 ± 0.066 | 0.2743 | 0.401 | 0.980 |

- The effect is symmetric and monotone: argmin-Q is far worse than plain,
  random selection is indistinguishable from plain, max-Q is far better.
- Violations fall from 2.53 % to 0.89 % of steps (65 %) and the discounted
  contact cost from −2.47 to −0.86, consistently across four blocks.
- Cost is ≈ zero: PPO reward −0.15 %, xy tracking +1.5 %, yaw error and survival
  unchanged. It is not buying safety by standing still.
- Rejection is a monotone dial but never beats max-Q, because the Q's absolute
  scale under-predicts the violation cost (5.2). Its ordering is good; its
  calibration is not.

This is a **persistent policy intervention, not `Q^π` validation**: executing Q
changes the continuation policy and invalidates the nominal FQE target.

## 6. Evidence boundaries

- 128 insertion states come from one export prefix at `model_616`; the four
  blocks vary continuation seeds, not states.
- Warp insertion restore is not bitwise, so returns are paired comparisons
  against the identical-controller noise floor, not exact values.
- MuJoCo CPU/Warp only; VSim unvalidated.
- The label is a peak-load constraint sampled at 20 ms; a shorter impact spike
  is not observable at this rate.
- Thresholds are relative to the recorded behaviour policy; the artifact stores
  them and they must not be transplanted across datasets.
- No safety, transfer, or policy-improvement claim.

## 7. Reproduce

```bash
cd BigRedGym
uv run --frozen scripts/train_contact_q.py \
  --snapshot logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/replay/replay_iter_0000616 \
  --checkpoint logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/model_616.pt \
  --foot_names FL_foot FR_foot RL_foot RR_foot --gamma 0.99 --subset_group episode \
  --validation_fraction 0.2 --validation_every 10000 --updates 60000 \
  --learning_rate 1e-3 --device cuda:0 \
  --output logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/offline_q_60k
uv run --frozen scripts/diagnose_contact_candidate_ranking.py \
  --output logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/contact_ranking \
  --checkpoint logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/model_616.pt \
  --load_run Nov13_21-02-10_contact-bw-1m-20261113 \
  --artifacts logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/offline_q_60k/contact_q/data/100pct
uv run --frozen scripts/evaluate_contact_q_persistent.py \
  --source logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/contact_ranking \
  --output logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/contact_persistent \
  --checkpoint logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/model_616.pt \
  --load_run Nov13_21-02-10_contact-bw-1m-20261113 \
  --artifacts logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/offline_q_60k/contact_q/data/100pct \
  --thresholds -2.5 -2.0 -1.5 --blocks 4
```

## 8. Verification record

- Portable gate: 334 passed, 97 deselected. Colocated: 28 gym + 7 learning.
- `ruff check .` and `ruff format` clean on every touched file.
- Contact path tests: 9 (spec invariants, label at/above threshold, bodyweight
  normalisation, terminal masking, missing-field rejection, threshold
  derivation, artifact round-trip, cross-reward rejection, non-zero failure
  rejection, legacy snapshot rejection).
- Recording fidelity: bodyweight constant per environment; thresholds and label
  rate reproduced from the training pool alone.
- Intervention gates: source and lane fidelity inside the pre-registered limits
  (root_states 9.5e-5 vs 2e-3, contact_forces 1.3e-3 vs 1.0), zero timeouts.
