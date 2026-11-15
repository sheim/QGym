# Angular-velocity-tracking Q — relabel r' from existing replay and train Q_{r'}^{π_r}

Status: **COMPLETED AND ARCHIVED — 2026-11-12.** Neither max-Q nor rejection
improves the yaw-tracking reward r'; a converged refit is indistinguishable from
the original (see §7), so this is a fundamental low-signal / off-policy-selection
limit at policy-noise-scale actions, not a Q-quality artifact.

## Archive — 2026-11-12

Implemented and verified (V1–V4). Key correction: the task's yaw-tracking term
divides the error by a hard-coded 2.5 (`go2trot.py:263`), so
`r' = exp(-((cmd−ω)/2.5)²/0.25)`, confirmed by a live parity check (max error
`0.0`). Automated gates: 11 focused / 311 portable / 28 gym / 7 learning tests,
Ruff and `git diff --check` clean. Durable docs: `docs/offline_angular_velocity_q.md`,
the supplement report `angular-velocity-tracking-20261112-v1`, and the
`q2-offline-reward-development` skill. No commits, staging, or todos were made.
Open: none.

## 1. Goal

### 1.a. Goal statements
- Define offline reward r' mirroring go2trot's yaw angular-velocity tracking term.
- Train Q_{r'}^{π_r} for the frozen model_600 policy on existing replay transitions.
- Measure whether max-Q and rejection interventions improve estimated r' over plain PPO.

### 1.b. Success criteria
- Relabeled r' matches the training term's formula and terminal semantics exactly.
- Q artifact passes the same identity and parity validations as the height-Q artifact.
- Paired evaluation reports each intervention's r' gain over plain with confidence intervals.

### 1.c. Constraints
- Reuse the existing final-1M replay snapshot; no re-collection or recorder changes.
- r' is the raw exp term ∈ [0,1], zero after termination; no dt or weight scaling.
- Preserve unrelated dirty worktree changes; no commits or pushes, and no new
  todos, without explicit user approval.

### 1.d. Questions
- Reward form = raw exp term (σ=0.25) — answered.
- Terminal semantics = zero after true termination — answered.
- base_ang_vel label noise = accept + quantify — answered.
- FQE dataset size = final 1M snapshot — answered.
- Q input = actor obs (48) + action (12), incl. base_ang_vel — answered.

## 2. High-Level Architecture

Reuse the height pipeline's shared primitives (`TransitionDataset`, `FQE`,
`QNetwork`, `FrozenTrotPolicy`, `PolicySpec`, `TrainingSpec`, `EvaluationSpec`,
candidate/snapshot machinery). Add a parallel minimal angular path; do not
generalize the height modules or touch their tests. Only two semantics differ:
reward shape (exp term vs band) and terminal continuation (zero vs absorbing).

### 2.a. Interfaces

```python
# ========================
# Library: contracts.py  (add)
# ========================
# Section: reward spec — the yaw tracking term.
class AngularVelocityRewardSpec:
  tracking_sigma: float            # 0.25 (cfg.reward_settings.tracking_sigma)
  error_scale: float = 2.5         # task's hard-coded /2.5 yaw divisor (go2trot.py:263)
  timing: str = "post_action_pre_reset"
  terminal_continuation: str = "zero_after_terminal"
  # failure_value(gamma) -> 0.0  (no absorbing tail; terminal value is zero)

ZERO_AFTER_TERMINAL_RETURN_CONVENTION = "discounted_episodic_zero_after_terminal"
ANGULAR_VEL_Q_FORMAT = "brg-angular-vel-q-v1"

# ========================
# Library: rewards.py  (add)
# ========================
# Relabel r' from the recorded successor observation; command and base_ang_vel
# come from next_actor_obs so the post-resample command matches training.
def relabel_angular_velocity(dataset: TransitionDataset,
                             spec: AngularVelocityRewardSpec) -> TransitionDataset:
  # reward = (~terminated) * exp(-((next_commands[2]-next_base_ang_vel[2])/2.5)^2 / sigma)
  # field indices located from dataset.manifest["obs_names"]/["obs_dims"]

# ========================
# Library: q_artifact.py  (add)
# ========================
# Runtime Q interface, same QNetwork(obs_dim=48, action_dim=12) as height.
class AngularVelocityQ:
  def predict(observations, actions) -> Tensor   # normalized obs, [N] scores

def save_angular_velocity_q(path, q_network, *, obs_mean, obs_std, gamma,
                            failure_value=0.0, reward_spec, policy_spec,
                            training_spec, dataset_manifest, subset,
                            source_revision) -> Path

# ========================
# Library: training.py  (add)
# ========================
# Same FQE loop as train_height_q, but failure_value=0.0 and save_angular_velocity_q.
def train_angular_velocity_q(dataset, policy, policy_spec, *, spec,
                             output_dir, subset, source_revision) -> Path

# ========================
# Library: fixed_state_variance.py  (add)
# ========================
# No absorbing -1 tail: add observed reward, mark terminal dead.
def accumulate_return_zero_after_terminal(total, alive, reward, terminated,
                                          step, gamma) -> (total, alive)

# ========================
# Library: persistent_intervention.py  (add rollouts)
# ========================
# Reward is the clean env term, post-step (post-resample command).
def rollout_angular_vel(env, policy, spec, snapshot, initial_obs, critic, seed,
                        horizon=500) -> dict          # maxq vs candidate0
def rollout_rejection_angular_vel(env, policy, spec, snapshot, initial_obs,
                                  critic, seed, threshold, horizon=500) -> dict
```

### 2.b. Data structures

```python
# AngularVelocityRewardSpec — invariants
#   tracking_sigma > 0; terminal_continuation == "zero_after_terminal"
#   failure_value(gamma) == 0.0 for any 0 <= gamma < 1

# Relabeled TransitionDataset — state
#   gains "reward" field: float32 [N], values in [0,1]; exactly 0 where terminated
#   manifest gains reward_spec / reward_field; source snapshot immutable

# AngularVelocityQ artifact — state
#   manifest.format == ANGULAR_VEL_Q_FORMAT
#   manifest.return_convention == ZERO_AFTER_TERMINAL_RETURN_CONVENTION
#   manifest.failure_value == 0.0; obs_dim == 48; action_dim == 12
#   obs_mean/obs_std shapes (48,), all obs_std > 0
```

### 2.c. Pseudo code

```python
# ---- offline relabel ----
dataset = load_snapshot(replay_iter_0000600)          # final 1M, model_600
spec = AngularVelocityRewardSpec(tracking_sigma=0.25)
relabeled = relabel_angular_velocity(dataset, spec)   # reward in [0,1], 0 on terminal

# ---- train Q ----
policy, policy_spec = load_policy(manifest, model_600.pt, device)
artifact = train_angular_velocity_q(relabeled, policy, policy_spec, ...)
#   FQE: target = reward + gamma*next_Q (continuing); = reward (terminal, failure_value=0)

# ---- paired evaluation ----
snapshot.restore(); snapshot.assert_restored()        # fixed 128 states x 8 lanes
for controller in (plain, maxq, rejection):
    for step in range(500):
        action = choose(obs)                          # candidate0 | maxq | rejection
        env.step()
        r = (~terminated) * exp(-((env.commands[:,2] - env.base_ang_vel[:,2]) / 2.5)**2 / 0.25)
        total, alive = accumulate_return_zero_after_terminal(total, alive, r,
                                                             env.terminated, step, 0.99)
# compare: mean/paired r' gain (maxq vs plain), (rejection vs plain), CI + t-test
```

## 3. Refined steps

### setup
- S1. Record baseline: `git -C BigRedGym status --short`, sha256 of `model_600.pt`,
  the replay `manifest.json` + `data.pt`, and the insertion snapshot
  `insertion.pt`. Preserve unrelated CORL/config/skill dirty entries; no staging.
- S2. Confirm identity: snapshot `config_fingerprint` (6412a878…) matches the
  `go2trot` cfg built by `task_registry.get_cfgs(...)` with the same `load_run`;
  confirm `tracking_sigma = 0.25` in the snapshot manifest and in
  `go2_config.py`. Reuse the existing `--load_run` + `model_600.pt` pair.

### edit
- E1. `learning/offline/contracts.py` (modify) — add `AngularVelocityRewardSpec`
  (tracking_sigma, timing, terminal_continuation, `failure_value()==0.0`) and the
  `ZERO_AFTER_TERMINAL_RETURN_CONVENTION` / `ANGULAR_VEL_Q_FORMAT` constants.
  Impact: defines the new reward and artifact identity without touching height
  specs.
- E2. `learning/offline/rewards.py` (modify) — add `relabel_angular_velocity`
  that slices `next_actor_obs` for `commands[2]` and `base_ang_vel[2]` via
  manifest `obs_names`/`obs_dims`, computes the exp term, masks `terminated`→0.
  Impact: produces the `reward` field and `reward_spec` in the dataset manifest.
- E3. `learning/offline/q_artifact.py` (modify) — add `AngularVelocityQ` and
  `save_angular_velocity_q` with the new format/return-convention and
  `failure_value=0.0`. Impact: persists/loads the angular-Q artifact.
- E4. `learning/offline/training.py` (modify) — add `train_angular_velocity_q`
  reusing the FQE loop with `failure_value=0.0` and `save_angular_velocity_q`.
  Impact: fits Q_{r'}^{π_r}; FQE itself is unchanged.
- E5. `learning/offline/fixed_state_variance.py` (modify) — add
  `accumulate_return_zero_after_terminal` (no absorbing tail). Impact: correct
  H500 return for zero-after-termination.
- E6. `learning/offline/persistent_intervention.py` (modify) — add
  `rollout_angular_vel` and `rollout_rejection_angular_vel` that compute the clean
  post-step exp reward and use the zero-after-terminal accumulator. Also add a
  trace-recording variant that returns per-step `base_ang_vel[:,2]` and
  `commands[:,2]` (liveness-masked) for the time-domain plot. Impact: live
  paired evaluation plus time-series diagnostics.
- E7. `scripts/train_angular_velocity_q.py`, `scripts/evaluate_angular_velocity_q_persistent.py`,
  `scripts/summarize_angular_velocity_q_persistent.py`,
  `scripts/plot_angular_velocity_tracking.py` (new) — mirror the height
  CLIs with `--sigma 0.25` and a rejection threshold sweep; the plot script
  renders the time-domain tracking figure. Impact: runnable
  train/evaluate/summarize/plot entry points.
- E8. `tests/unit_tests/test_angular_velocity_q.py` (new) — relabel, spec
  invariants, FQE zero-terminal, artifact round-trip, accumulator. Impact:
  regression coverage independent of the simulator.
- E9. `docs/plans/supplements/replay_relabel_q/README.md` (modify) and a new
  report dir; note the second reward in the
  `q2-offline-reward-development` skill. Impact: durable experiment record.

### build
- B1. `uv sync --frozen` (deps unchanged; no re-resolution). No compilation or
  colcon-style worker pool — this is uv-managed Python, so no OOM-from-build.
- B2. `uv run --frozen ruff format <touched>` then `uv run --frozen ruff check .`
  and `git -C BigRedGym diff --check`. GPU training/eval uses the existing
  cuda:0 + 1024-world Warp path with the 12-min/controller runtime gate.

### verification
- V1. `uv run --frozen python -m pytest tests/unit_tests/test_angular_velocity_q.py -q`
  → relabel matches `exp(-((cmd−ang_vel)/2.5)²/0.25)`, terminal reward is 0, artifact
  round-trips with `failure_value==0.0` and the new format.
- V2. Train smoke: `scripts/train_angular_velocity_q.py` on a 10% subset, few
  updates → artifact exists, manifest `return_convention`/`failure_value`
  correct, `gamma==0.99`.
- V3. Paired eval smoke (1 block, 8 calls) plain+maxq+rejection → summarize
  reports mean/paired r' gain with CI; label-noise check compares relabeled r'
  vs clean `env.base_ang_vel` r' on a short rollout and reports the RMSE.
- V4. Time-domain tracking plot: pick 2–4 insertion configs, run their parallel
  executions, record per-step `base_ang_vel[:,2]` and the reference
  `commands[:,2]`, and plot mean±std vs time for plain vs maxQ. Note the yaw
  command resamples every **3.0 s** (150 steps at dt=0.02), not 5 s; the plot
  overlays the true piecewise-constant reference at those boundaries.

## 4. Review

- Terminal vs timeout: true termination → reward 0 + `failure_value=0` (FQE
  terminal target 0); timeout → bootstrap from pre-reset successor, matching the
  height pipeline. Dead-world reward is harmless (masked by `alive`).
- Command timing: relabel uses `next_actor_obs` (post-resample command); live
  reward reads `env.commands` **after** `env.step()`, not the existing
  pre-step capture — verified consistent with `_reward_tracking_ang_vel`.
- Formula: exact `exp(-((cmd−ang_vel)/2.5)²/0.25)` — the `/2.5` is the task's
  hard-coded yaw divisor (`go2trot.py:263`), not in the base class; `tracking_sigma=0.25`;
  yaw is the `[:,2]` slice of both `commands` and `base_ang_vel`.
- Label noise: relabeled reward uses noisy `base_ang_vel` (σ≈0.033) while live
  eval uses clean; V3 quantifies the RMSE, and the Q input noise is a different
  draw than the label noise, so it is irreducible-but-small — no silent drift.
- Scale: angular Q values are positive (γ=0.99 ⇒ up to ~100), unlike the
  negative height Q; rejection thresholds must be swept on this scale, not
  inherited from height.
- Loader reuse: `evaluation.load_policy_from_artifact`/`policy_spec_from_artifact`
  only read `manifest["policy_spec"]` (HeightQ-typed but AngularVelocityQ-safe);
  widen the annotation rather than duplicate.
- Memory: V4 trace ≈ 4 MB per 1024-world call; record selectively for the 2–4
  chosen states. Full 16-block eval is ~30–40 min GPU; smoke is 1 block.
- Dirty worktree: adds new files and modifies the six library modules plus docs;
  CORL/config/skill entries are preserved (S1 baseline); no commits/pushes/new
  todos without approval.

## 5. Documentation

Work the implementer executes (Scaffold does not write outside the plans dir):

### Journal
- Add a `journal.md` entry for the angular-velocity-Q plan-action cycle, per
  AGENTS.md §Docs. Do **not** create or edit `todos.md` rows, and do not commit,
  without explicit user approval.

### Skill
- Update `BigRedGym/genAI_skills/.agents/skills/q2-offline-reward-development/SKILL.md`:
  document the angular-velocity reward as the second concrete path (shared
  relabel/FQE/artifact primitives; `terminal_continuation="zero_after_terminal"`,
  `failure_value=0`, positive reward scale). Keep the "not a generic framework"
  framing; do not overclaim.

### Documentation
- Add `BigRedGym/docs/offline_angular_velocity_q.md` (reward formula, terminal
  semantics, command timing, label-noise caveat, run commands) and cross-link it
  from `docs/offline_height_q.md` and the supplement README.
- Add the durable report under
  `BigRedGym/docs/plans/supplements/replay_relabel_q/angular-velocity-tracking-20261112-v1/REPORT.md`
  (V1–V4 results, plots, threshold sweep) and index it in the supplement README.

## 6. Execution results — 2026-11-12

Implemented E1–E9. **Formula correction found during execution:** `Go2Trot`
overrides the base reward (`go2trot.py:261-264`) with a hard-coded `/2.5` yaw
divisor, so the term is `exp(-((cmd−ang_vel)/2.5)²/0.25)`, not the base-class
`exp(-(cmd−ang_vel)²/0.25)` this plan initially assumed. Implemented the task's
true formula (`error_scale=2.5`); a live parity check against
`Go2Trot._reward_tracking_ang_vel` returned max error `0.0`.

- V1 — command `pytest tests/unit_tests/test_angular_velocity_q.py -q`; expect
  relabel/terminal/artifact/accumulator correct. Actual: focused `9 passed`;
  portable `309 passed`; `gym 28 passed`; `learning 7 passed`; Ruff + `diff --check`
  clean. PASS.
- V2 — train smoke on real snapshot + `model_600.pt`; expect artifact identity.
  Actual: 200-update 10%/100% + 100k 100% artifacts; `format=brg-angular-vel-q-v1`,
  `return_convention=discounted_episodic_zero_after_terminal`, `failure_value=0.0`,
  `gamma=0.99`, obs/action 48/12, checkpoint `f88b1f9b…`, fingerprint `6412a878…`.
  PASS.
- V3 — paired eval smoke (1 block, 64 exec/state, plain+maxQ+rejection c=5/c=20).
  Actual: plain `r'=96.2242`, **maxQ `95.5344` (gain −0.6898, CI [−0.7794,−0.6052],
  t=−15.44, p=6.4e-31, W/T/L 4/0/124)**, rejection c=5 −0.0144 (flips to +0.0059
  on rerun ⇒ within replay floor), c=20 −0.4088 (p=3.4e-19). Label-noise RMSE
  `0.00835` (0.86%); live reward parity `0.0`. PASS (valid run); **negative result**.
- V4 — plot at `/tmp/opencode/angular-vel-q-20261112/eval/angular_velocity_tracking.png`;
  yaw-command resamples at 0.98/3.98/6.98/9.98 s (150 steps = 3.0 s); world-step
  yaw RMSE 0.2373 plain vs 0.2566 max-Q. PASS.

**Interpretation:** as trained here (100k updates, FQE loss mean 4.31, not
converged; mean Q advantage 0.22 on a ~96 return scale), neither max-Q nor a
strong rejection threshold improves `r'`; both reduce it, plausibly via a higher
fall rate (positive reward ⇒ a fall truncates the remaining return). The c=5
gain is inside the measured Warp replay floor and is not an effect. One block,
128 mid-gait states, one checkpoint — preliminary, not `Q^π`/safety evidence.


## 7. Option-(a) follow-up: better-converged Q, ranking ceiling, re-run

User-approved follow-up after the negative first result: train a better-converged
Q and re-evaluate as the real test of whether the negative max-Q result is a
Q-quality artifact.

### 7.a. Method

- Same snapshot (`replay_iter_0000600`), checkpoint (`model_600`), spec
  (`error_scale=2.5`, `sigma=0.25`, `failure_value=0`, `gamma=0.99`).
- Added an episode-grouped 90/10 train/validation split (`episode_split`) and a
  per-fraction report with the minibatch loss curve plus the train-sample and
  held-out one-step Bellman residual (`_bellman_metrics`, every 25k updates).
- Variants: `v0` 100k/lr3e-4/b256/tas1/[256,256] (fair control);
  `v1` 300k/same; `v3` 300k/lr3e-4/b512/tas8/[512,256];
  `v2` 500k/lr1e-4/b512/tas8/[256,256,256].
- Independent residual decomposition at 1/8/32 successor-action samples to
  separate fit error from target-sampling noise.
- Candidate-ranking diagnostic: 128 states x 8 stored candidates, two seeds,
  candidates as the first action then the frozen policy (H=100 and H=500),
  Spearman / argmax hit / top-1 regret / Q-vs-realized advantage correlation,
  plus within-state cross-seed rank reliability as the achievable ceiling.
- Intervention re-run with 2 blocks and seed-paired controllers: plain,
  old-100k max-Q, best-held-out max-Q, 300k max-Q, and three rejection
  thresholds (`37.0/75.5/76.6`, quantiles of the best Q distribution).

### 7.b. Results

- Held-out MSE (32-sample target): `v0 5.63` (best), `v1 5.75`, `v3 6.17`,
  `v2 6.20`; the original 100k Q (trained on 100%, leakage-optimistic) 5.48.
  Training loss still falls at 300k (last-1k 5.05 vs 5.47) without held-out gain;
  every stability knob is worse. Target-sampling noise contributes only
  RMS 0.17–0.31 to a 2.4 residual.
- Q advantage over candidate 0: old 0.224 → v0 0.234 / v1 0.262 (persistent
  eight-draw protocol); stored-candidate 0.207 → 0.211 / 0.224.
- Ranking: all critics at chance (Spearman −0.03…+0.005, argmax hit 0.10–0.13
  vs 0.125 chance, regret 0.46–0.95, advantage correlation ≤0.18). The same
  plain controller ranked against itself with two seeds gives within-state
  Spearman 0.007 (H100) / 0.019 (H500) and argmax agreement 0.164 — the ceiling.
- Intervention (2 blocks, 128 executions/state): old −0.653, v0 −0.615,
  v1 −0.369, rejections −0.445/−0.656/−0.621; all p ≤ 1e-24. `v0 − old` =
  +0.038 (p 0.11, not significant); `v1 − old` = +0.284 (p 4e-16) but still
  negative vs plain. Survival delta is positive only for v1 (+0.0032).

### 7.c. Verdict

**Fundamental low-signal / off-policy-selection limit, not a Q-quality artifact.**
No variant lowered the held-out residual, and the realized within-state candidate
ordering is not reproducible across seeds, so no ranker can exceed ≈0 on this
target. The converged refit is statistically indistinguishable from the original;
the less-negative 300k run has a worse held-out residual, so its smaller loss is
a selection-behavior difference, not better value accuracy. Scope: policy-noise
scale candidate actions, 128 states, 2 seeds, 1–2 blocks, one checkpoint — not a
claim that yaw Q cannot help at other action scales or states.

Artifacts: `/tmp/opencode/angular-vel-q-20261112/qbetter/` (`v0..v3` training
reports, `ranking/ranking.json`, `eval/` with 2-block results, plot).
