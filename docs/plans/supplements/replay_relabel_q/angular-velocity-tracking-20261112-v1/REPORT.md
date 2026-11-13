# Angular-velocity tracking Q — replay relabel and persistent intervention (2026-11-12)

## Provenance

- Plan: `.opencode/workspace/plans/angular_velocity_q_plan.md`.
- Artifact directory: `/tmp/opencode/angular-vel-q-20261112/` (`q/` = 200-update
  smoke, `q20k/`, `q100k/` = 100k-update fit, `eval/` = paired block, traces,
  plot, `results.json`, `validation.json`; `eval_fresh/` = independent
  reproduction of the same command; `qbetter/` = held-out variant sweep, ranking
  diagnostic, and two-block re-run).
- Replay dataset:
  `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/replay/replay_iter_0000600`
  (final 1M transitions, capacity 1M, obs_dim 48, config fingerprint `6412a878…`).
- Frozen policy: `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt`,
  sha256 `f88b1f9b829675c9…`.
- Insertion states: `/tmp/opencode/height-q-fixed-variance-20261110-v1/insertion.pt`
  (128 states x 8 lanes, seed 240000, same checkpoint).
- Environment: RTX 5090, `cuda:0`, MuJoCo Warp, `num_envs = 1024`, headless.
- Code identity: `eval/code_resume.json` matches the current tree for the
  evaluation script and all four library modules. The four `*_00.pt` block files
  were written by the first invocation (`eval/code.json`), which differs only by
  the added label-noise validation and the trace-horizon episode-length override;
  every rollout module hash is identical, and no timeout occurred in either run.
  `eval/validation.json` and the traces come from the resume invocation.
- Task code at this revision hard-codes the yaw-error divisor `2.5` in
  `_reward_tracking_ang_vel`, so the relabeled reward is
  `exp(-((cmd - ang_vel)/2.5)**2 / 0.25)`, not `exp(-(cmd-ang_vel)**2/0.25)`.
  A live parity check against the task method returned maximum absolute error
  `0.0`.

## What was measured

`r' = exp(-((c_t - ω_t)/2.5)^2 / 0.25)`, zero after a true termination, no
`dt`/weight scaling, `gamma = 0.99`, `H = 500`. Reward range `[0, 1]`, so the
return is positive and a fall truncates it instead of adding a failure tail.
Data: the final-1M snapshot only. Q input: 48-wide actor observation + 12-wide
action. Fit: FQE, 100k updates, batch 256, `gamma = 0.99`, `failure_value = 0.0`,
100% subset.

Fixed-state protocol: 128 mid-gait insertion states x 8 lanes, eight paired calls
per controller block (64 executions per state), seeds
`7100000 + 10*(block*8 + call)`, plain = candidate 0 of eight stochastic draws,
max-Q = argmax of the same eight, rejection = first draw with `Q >= c` else
max-Q.

## Results (one block, 64 executions per state)

| Controller | mean `r'` | survivor | yaw RMSE | paired gain | state-cluster 95% CI | paired t (df 127) | p | W/T/L | fallback |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|
| plain | 96.2242 | 0.9812 | 0.2200 | - | - | - | - | - | - |
| max-Q | 95.5344 | 0.9762 | 0.2472 | -0.6898 | [-0.7794, -0.6052] | -15.44 | 6.4e-31 | 4/0/124 | - |
| rejection `c=5` | 96.2098 | 0.9807 | 0.2206 | -0.0144 | [-0.0283, -0.0032] | -2.21 | 0.029 | 59/0/69 | 0.097% |
| rejection `c=20` | 95.8155 | 0.9795 | 0.2348 | -0.4088 | [-0.4884, -0.3377] | -10.60 | 3.4e-19 | 7/0/121 | 33.0% |

Paired execution intervals (64 executions per state) agree in sign:
max-Q `[-0.753, -0.626]`, `c=5` `[-0.027, -0.002]`, `c=20` `[-0.466, -0.351]`.
`prob_execution_win` is 0.199, 0.503, and 0.202.

### Reproduction check (independent fresh invocation)
The same command was rerun into `eval_fresh/` to separate controller effects from
Warp replay roundoff:

| Controller | mean `r'` (main / fresh) | paired gain (main / fresh) | p (main / fresh) | W/L (main / fresh) |
|---|---|---|---|---|
| plain | 96.2242 / 96.2196 | - | - | - |
| max-Q | 95.5344 / 95.5325 | -0.6898 / -0.6871 | 6.4e-31 / 1.6e-31 | 4/124 vs 3/125 |
| rejection `c=5` | 96.2098 / 96.2255 | -0.0144 / **+0.0059** | 0.029 / **0.416** | 59/69 vs 72/56 |
| rejection `c=20` | 95.8155 / 95.8193 | -0.4088 / -0.4003 | 3.4e-19 / 8.6e-19 | 7/121 vs 8/120 |

The max-Q and `c=20` effects reproduce to ~0.01 return units. The `c=5` effect
does **not**: it is smaller than the ~0.02 mean-absolute replay difference
measured for two identical plain rollouts, so treat `c=5` as indistinguishable
from plain. Single-block near-zero gains in this protocol are not effects.

Supporting diagnostics:

- mean Q advantage of max-Q over candidate 0: `+0.224` on a return scale of
  ~96; candidate 0 selected in 12.5% of max-Q decisions (uniform tie-breaking).
- FQE loss: mean 4.31 over 100k updates with spikes to 30; the target scale is
  O(1e2) and the fit is not converged in any calibrated sense.
- survivor deltas: max-Q `-0.0050`, `c=5` `-0.0005`, `c=20` `-0.0017`; yaw-RMSE
  deltas: `+0.0273`, `+0.0005`, `+0.0150`.
- 600-step trace (seed 7300000, no terminations in this window): world-step yaw
  RMSE 0.2373 plain versus 0.2566 max-Q; the lane/state-mean systematic error is
  near zero between resamples and spikes at each command change. Detected command
  resamples at 0.98, 3.98, 6.98, and 9.98 s, i.e. exactly 150 control steps.

## Follow-up (option a): a better-converged Q and the ranking ceiling

The first fit used 100k updates, `lr 3e-4`, batch 256, one target action sample,
hidden `[256, 256]`, and its minibatch loss was unstable (mean 4.31). This
follow-up asks whether the negative max-Q result is an undertrained-Q artifact.
Artifacts: `/tmp/opencode/angular-vel-q-20261112/qbetter/`.

### Held-out convergence

Every variant trains on an **episode-grouped 90% split** (898,702 rows) and is
evaluated on 101,298 disjoint held-out rows. Normalization is fitted on the
train split. The per-variant report (`*_report.json`) records the loss curve and
the held-out one-step Bellman residual at update 0 and every 25k updates; the
table's residual columns come from an independent recomputation that bootstraps
from the online Q, with 1 and 32 successor-action samples. The report's own
target-network values at one sample are `v0 5.84`, `v1 5.80`, `v3 6.26`,
`v2 6.37` — the same ordering within 0.2, so `v0` and `v1` are tied either way.

| Variant | updates | lr | batch | target samples | hidden | last-1k loss | held-out MSE (1 sample) | held-out MSE (32) | MAE (32) | RMSE (32) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `v0_100k_ref` (best, = original recipe) | 100k | 3e-4 | 256 | 1 | 256,256 | 5.472 | 5.659 | **5.626** | 0.555 | 2.372 |
| `v1_300k_ref` | 300k | 3e-4 | 256 | 1 | 256,256 | 5.048 | 5.789 | 5.746 | 0.554 | 2.397 |
| `v3_300k_wide` | 300k | 3e-4 | 512 | 8 | 512,256 | 5.044 | 6.271 | 6.171 | 0.753 | 2.484 |
| `v2_500k_var` | 500k | 1e-4 | 512 | 8 | 256,256,256 | 5.109 | 6.312 | 6.203 | 0.736 | 2.491 |
| `q100k` (original, trained on 100%) | 100k | 3e-4 | 256 | 1 | 256,256 | n/a | 5.508* | 5.479* | 0.507* | 2.341* |

`*` The original artifact was trained on all 1M rows, including the held-out
episodes, so its residual is optimistic. `v0` is the fair control.

Loss curves (`validation MSE` at 25k-update checkpoints, v0/v1):
`0.37 → 3.9 → 5.5 → 5.6 → 5.8` (v0) and
`0.37 → 3.9 → 5.5 → 5.6 → 5.9 → 5.9 → 5.8 → … → 5.8` (v1, flat from 75k).
The residual rises while the value scale grows to `Q ≈ 45`, then plateaus;
absolute MSE therefore cannot be read as "worse" at 25k.

Conclusions:

- **More updates do not lower held-out error.** 300k is tied with 100k
  (5.75 vs 5.63 at the 32-sample target), while the *training* loss keeps
  falling (5.05 vs 5.47 last-1k). This is mild overfitting, not convergence.
- The stability knobs requested (lower lr 1e-4, batch 512, `target_action_samples
  8`, wider hidden) all made the held-out residual **worse** (6.17–6.20).
- Target-sampling noise is not the explanation: recomputing the target with 32
  successor-action samples changes MSE by <0.12, and the one-sample target's RMS
  deviation from the 32-sample mean is only 0.17–0.31. The ~2.4 RMSE residual is
  the achievable fit for this feature/target set, not sampling noise.
- Q advantage over candidate 0 (persistent eight-draw protocol): old `0.2238`,
  `v0` `0.2339` (+4%), `v1` `0.2624` (+17%). On the stored 8-candidate set:
  `0.207 → 0.211 → 0.224`. The target of beating `0.224` is met only by the 300k
  Q, whose held-out residual is *not* better.

### Candidate-ranking ceiling

All 128 insertion states, eight stored candidates per state (`candidate 0` is the
policy mean, `1–7` are stochastic draws), two independent seeds, paired commands
within each state. Ranking is against the realized finite yaw return.

| Critic | H | Spearman | argmax hit | top-1 regret | Q max adv | realized max adv | adv corr (pairs) | adv corr (state means) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old100k | 100 | −0.017 | 0.106 | 0.490 | 0.207 | 0.501 | 0.048 | 0.170 |
| v0_100k | 100 | −0.003 | 0.113 | 0.514 | 0.211 | 0.501 | 0.028 | 0.142 |
| v1_300k | 100 | −0.026 | 0.129 | 0.463 | 0.224 | 0.501 | 0.053 | 0.179 |
| old100k | 500 | −0.030 | 0.102 | 0.588 | 0.207 | 0.623 | −0.007 | −0.015 |
| v0_100k | 500 | 0.005 | 0.102 | 0.952 | 0.211 | 0.623 | −0.012 | 0.015 |
| v1_300k | 500 | −0.016 | 0.109 | 0.506 | 0.224 | 0.623 | 0.032 | 0.007 |

Chance argmax hit rate is 0.125. Every critic is at chance, and none improves
with more updates: the ranking is flat across the sweep even though the Q
advantage grew.

The decisive control is the **test-retest reliability of the realized ranking**:
two independent seeds of the *same* plain controller produce per-state Spearman
`0.0073` (H100) and `0.0190` (H500), with argmax agreement `0.164` in both cases.
The within-state realized spread is 1.15 (H100) and 2.31 (H500), but its ordering
does not reproduce. At the single-action, policy-noise scale the realized
candidate ordering is dominated by trajectory divergence, so a ranker's ceiling
is ≈0 and the Qs are already there.

### Intervention re-run (2 blocks, 128 executions per state)

All critics share the same block/seed protocol; `old100k`, `v0` and `v1` are
max-Q controllers, rejections use `v0` with thresholds from its Q distribution
(`q50`, `q80`, `q95` = 37.0, 75.5, 76.6).

| Controller | mean `r'` | gain vs plain | state-cluster CI | t(127) | p | W/T/L | fallback |
|---|---:|---:|---|---:|---:|---|---:|
| plain | 96.2245 | – | – | – | – | – | – |
| old100k max-Q | 95.5715 | −0.6530 | [−0.7204, −0.5894] | −19.56 | 4.0e-40 | 3/0/125 | – |
| v0 max-Q (best held-out) | 95.6093 | −0.6152 | [−0.6831, −0.5487] | −18.10 | 5.8e-37 | 4/0/124 | – |
| v1 (300k) max-Q | 95.8559 | −0.3686 | [−0.4240, −0.3144] | −13.10 | 2.5e-25 | 13/0/115 | – |
| rejection `c=37` | 95.7799 | −0.4446 | [−0.5004, −0.3908] | −15.82 | 8.0e-32 | 6/0/122 | 0.466 |
| rejection `c=75.5` | 95.5686 | −0.6559 | [−0.7240, −0.5900] | −19.43 | 7.5e-40 | 1/0/127 | 0.795 |
| rejection `c=76.6` | 95.6036 | −0.6208 | [−0.6891, −0.5545] | −18.18 | 3.8e-37 | 3/0/125 | 0.918 |

Cross-controller paired tests on the same 128 states and 128 executions:

- `v0 − old100k` = **+0.0378** (CI95 [−0.0089, +0.0845], t 1.60, p 0.111,
  W/L 74/54): the best held-out Q is statistically indistinguishable from the
  original.
- `v1 − old100k` = **+0.2844** (CI95 [+0.2242, +0.3446], t 9.35, p 4e-16,
  W/L 114/14): the 300k Q is less harmful, but still negative vs plain.
- `v1 − v0` = **+0.2466** (CI95 [+0.1826, +0.3105], p 5e-12).

Survival deltas: old −0.0034, v0 −0.0059, v1 **+0.0032**; yaw-RMSE deltas
+0.026, +0.024, +0.014. The intervention ordering does not follow held-out error
(v1 has the worse residual but the smaller loss), so the difference is a
selection-behavior difference, not evidence of a better value estimate.

Validation for this run: live reward parity 0.0, label-noise RMSE 0.00836
(0.86%), CPU/GPU Q max error 2.3e-5 with zero argmax disagreements, five-step
negative-control return difference 0.0032, and an H500 identical-plain replay
difference of mean absolute 0.031 (max 13.3, heavy-tailed) on a mean return of
96.26.

### Verdict

**Fundamental low-signal / off-policy-selection limit, not a Q-quality artifact.**
Two independent ceilings agree: (1) held-out residual is flat from 100k to 300k
updates and *worse* for every stability variant, so no better-converged Q was
available; (2) the realized within-state ranking of these candidate actions does
not reproduce across two seeds (Spearman 0.007/0.019), so no ranker could exceed
≈0 on this target. Every critic is at chance ranking, and every intervention is
negative; the best held-out Q is statistically indistinguishable from the
original (+0.038, p 0.11), and the only variant with a smaller loss (300k) is not
better on held-out error and does not reach a nonnegative gain.

Scope: 128 states, 8 stored candidate actions at policy-noise scale, 2 seeds,
one checkpoint, 1–2 blocks. This does not show that yaw Q can never help; it
shows that within this action scale and horizon, candidate selection has no
learnable/reproducible signal in this replay, so a persistent max-Q intervention
adds noise-avoidance behavior rather than value-driven improvement.

## Label noise

`next_actor_obs = clean successor + one policy-marginal noise draw`;
`base_ang_vel` noise is `0.01/0.3 = 0.0333 rad/s` and `commands` carry none.
On 50 live steps x 1024 worlds: clean mean `0.9624`, noisy mean `0.9618`, bias
`-0.00062`, RMSE `0.00835` (relative `0.86%`), mean absolute `0.0052`, max
`0.0773`, `0.074%` of rows differing by more than `0.05`. Accepted as small and
irreducible from the recorded data.

## Validation gates (same run)

- source/lane fidelity one-step errors within the fixed limits
  (`base_ang_vel` 6.4e-5, `dof_pos` 1.4e-6, `contact_forces` 7.2e-4).
- CPU/GPU Q agreement: max error `2.3e-5`, batch-split error `2.3e-5`, zero
  argmax disagreements.
- live reward parity versus `Go2Trot._reward_tracking_ang_vel`: `0.0`.
- identical-controller negative control over five steps: exact selection,
  survival, step, termination-step, timeout, and first-termination fields;
  continuous return difference `0.0033` (bounded at `1e-2`).
- identical-controller replay over H500 (two plain rollouts, same seed): mean
  signed difference `+0.003`, mean absolute `0.018`, SD `0.148`, max `3.82` on a
  mean return of `96.2` (`relative_mean_absolute = 1.9e-4`). This is Warp restore
  roundoff, not an intervention effect. The height path's binary reward hid the
  same roundoff because a `1e-3` height change rarely flips band membership.
- first block runtime: 28.5–30.0 s per controller (`projected_controller_seconds`
  456–481 s), inside the 12-minute/controller gate. Traces took 4.4–4.5 s each.

## Interpretation and limits

- The angular-velocity Q, as trained here, does **not** improve the estimated
  reward. Both interventions reduce mean `r'` and survival; the effect is small
  for a near-inactive rejection threshold and larger as the threshold removes
  more candidate 0 selections. The follow-up sweep shows this is not explained by
  an undertrained Q: the held-out residual does not improve with more updates and
  the realized candidate ranking is not reproducible across seeds.
- Only the max-Q and `c=20` effects are larger than the measured replay floor and
  reproduce across an independent invocation. The `c=5` gain changes sign between
  runs (`-0.0144` then `+0.0059`) and must be reported as no detectable effect,
  not as a real small regression.
- With a mean Q advantage of 0.22 on a return scale of 96, max-Q is close to
  random re-ranking of eight nearly tied draws; the consistent negative gains
  are better explained by the fall-rate increase (and slightly worse tracking)
  than by a calibrated ranking.
- Positive rewards mean a fall costs the remaining return. A Q that under-values
  the post-fall future can prefer actions that look marginally better at tracking
  while raising the fall rate. The `delta_survivor_rate` column is the relevant
  diagnostic; it is negative for every intervention.
- One artifact, one block, 128 mid-gait states, one checkpoint, one machine.
  Persistent intervention changes the continuation policy, so this is not
  `Q^{pi}` validation, not policy improvement, and not safety evidence. Warp
  replay is not bitwise deterministic; every claim here is a paired
  fixed-state statement on these 1024 worlds.
- Thresholds `5` and `20` were scale probes, not tuned on these returns, and no
  multiplicity correction is applied to the three paired tests.

## Reproduction

```bash
uv run --frozen python scripts/train_angular_velocity_q.py \
  --snapshot logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/replay/replay_iter_0000600 \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --output /tmp/opencode/angular-vel-q-20261112/q100k \
  --sigma 0.25 --gamma 0.99 --subset_group episode --fractions 0.1 1.0 \
  --updates 100000 --batch_size 256 --device cuda:0 --seed 0

uv run --frozen python scripts/evaluate_angular_velocity_q_persistent.py \
  --source /tmp/opencode/height-q-fixed-variance-20261110-v1 \
  --output /tmp/opencode/angular-vel-q-20261112/eval \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --load_run Nov10_10-36-22_gpu-final-600-trot-hist-20261110 \
  --artifacts /tmp/opencode/angular-vel-q-20261112/q100k/angular_vel_q/data/100pct \
  --artifact_names angular_q --thresholds 5.0 20.0 --blocks 1 --trace_steps 600

uv run --frozen python scripts/summarize_angular_velocity_q_persistent.py \
  /tmp/opencode/angular-vel-q-20261112/eval
uv run --frozen python scripts/plot_angular_velocity_tracking.py \
  /tmp/opencode/angular-vel-q-20261112/eval
```

Plot: `/tmp/opencode/angular-vel-q-20261112/eval/angular_velocity_tracking.png`.
Raw blocks: `baseline_00.pt`, `angular_q_00.pt`, `rejection_c5_00.pt`,
`rejection_c20_00.pt`; summary `results.json`, `per_state.csv`; traces
`trace_baseline.pt`, `trace_angular_q.pt`.
