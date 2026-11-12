# Two additional height-Q critics at equal epochs — 2026-11-11

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-two-critic-datasets-20261111-v1/REPORT.md`](/tmp/opencode/height-q-two-critic-datasets-20261111-v1/REPORT.md).
> Original sha256: `8ddcecfec8175b45ce86ad6f453abf80f85ff945deefa855277047ceb4c121ec`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-two-critic-datasets-20261111-v1/`](/tmp/opencode/height-q-two-critic-datasets-20261111-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../README.md).

**Question.** On the same 128 fixed k=100 states and the same 1024 execution seeds, how do four frozen height-Q critics compare when each selects the maximum of eight fresh stochastic `model_600` actions at every decision? The two new critics are fitted at equal 102.4 epochs on the existing 2M mixed PPO replay and on 1M fresh transitions from the frozen nominal policy.

**Scope.** This is a fixed-state policy-intervention comparison with H500, gamma 0.99, binary height reward 0.32 ± 0.05 m, and absorbing true failure. It is not policy optimization, not `Q^pi` validation, and not an equal-data-size comparison: mixed has 2M transitions and fresh has 1M.

## Bottom line

- Fresh-data critic: paired mean gain **1.6911**, individual win/tie/loss **56.14%/20.47%/23.39%**, batch16 positive **83.06%**, paired tail interval **[1.0483, 2.3351]**.
- Mixed-data critic: paired mean gain **0.9093**, individual win/tie/loss **48.15%/19.01%/32.84%**, batch16 positive **72.88%**, paired tail interval **[0.2664, 1.5574]**.
- Reused references (no rerun): original-Q gain **1.7191** and evaldist-Q gain **1.1100**. The original row reproduces the paired16 report exactly, including the state-cluster bootstrap interval **[1.3720, 2.0834]**.
- The fresh critic is close to the original critic on these states; the mixed critic is lower. Because the datasets differ in size and policy mixture, that ordering is **not** a clean data-size or data-quality effect and does not generalize beyond these fixed states.
- Paired Student t tests on the 128 per-state mean differences give `t = 7.04–9.36`, df=127, and Holm-adjusted `p ≤ 1.05e-10` for all four controllers. These are fixed-state paired precision statements, not safety, infinite-horizon, or typical-execution claims.

## Frozen controls and budgets

- Checkpoint `model_600.pt`, sha256 `f88b1f9b829675c97ed0c420202cf54456dc414641def34e379e39407ee3adf6`; fixed reward and gamma as above.
- Original fair100 manifest: 1,000,000 transitions, 400,000 updates, batch 256 = 102.4 epochs. Mixed: 800,000 updates = 102.4 epochs. Fresh: 400,000 updates = 102.4 epochs. Network, Adam 3e-4, target tau 0.005, one target-action sample, observation normalization and seed 0 are unchanged.
- Fresh collection: 4,096 MuJoCo Warp worlds, nominal stochastic policy only (mean + std × Gaussian, no candidate search), 5 s / 250-step administrative timeout, seed 20261111 (disjoint from the 7,100,000-series evaluation seeds and 2,400,000 saved-state seed), 611 policy steps = 2,502,656 appended transitions; the last 1,000,000 were retained, spanning 8,192 unique env/episode pairs, 15 true terminations and 4,082 timeouts. Wall time 20.49 s.
- Mixed snapshot: 2,000,000 rows from `hist-mix-20261110/replay/replay_iter_0000600`, immutable, existing early/final mixture; fingerprint `6412a878…` and `post_action_pre_reset` labels validated.
- Training wall time: mixed 745.76 s, fresh 371.89 s. Mean training loss (not learned-quality evidence): mixed 6.554596, fresh 0.337882.

## Per-controller results

| Controller | Mean H500 return ± pooled SD | SD of 128 state means | Execution CI95 | State-cluster CI95 | Tail interval | Inside band | Falls | XY RMSE | Speed/cmd |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| baseline | -5.2858 ± 8.1216 | 6.7391 | [-5.3105, -5.2610] | [-6.5552, -4.1916] | [-5.9287, -5.2858] | 95.46% | 2.16% | 0.4493 | 0.7426 |
| original | -3.5666 ± 6.3288 | 5.0477 | [-3.5875, -3.5458] | [-4.5196, -2.7401] | [-4.2090, -3.5666] | 97.18% | 2.24% | 0.4561 | 0.7362 |
| evaldist | -4.1758 ± 7.1375 | 5.6627 | [-4.1994, -4.1521] | [-5.2380, -3.2342] | [-4.8179, -4.1758] | 96.55% | 2.28% | 0.4549 | 0.7383 |
| mixed | -4.3765 ± 6.8288 | 5.6668 | [-4.3973, -4.3557] | [-5.4428, -3.4450] | [-5.0246, -4.3765] | 96.34% | 1.36% | 0.4619 | 0.7195 |
| fresh | -3.5946 ± 6.4504 | 5.1106 | [-3.6161, -3.5732] | [-4.5561, -2.7570] | [-4.2386, -3.5946] | 97.20% | 1.99% | 0.4579 | 0.7292 |

The first SD is the pooled SD of all 131,072 individual H500 returns for that controller (128 states × 1024 matched seeds, ddof=1). It is outcome spread, **not** the standard error of the mean. The state-mean SD is the separate SD of the 128 per-state means (ddof=1), i.e. between-configuration spread after each mean averages 1,024 seeds.

## Paired gain versus plain baseline

| Controller | Mean gain | Execution CI95 | State-cluster CI95 | Individual win/tie/loss | Batch16 positive | Paired tail interval |
|---|---:|---|---|---|---:|---|
| original | 1.7191 | [1.6917, 1.7466] | [1.3720, 2.0834] | 56.91% / 21.22% / 21.87% | 85.49% | [1.0763, 2.3615] |
| evaldist | 1.1100 | [1.0817, 1.1384] | [0.8493, 1.3776] | 51.54% / 19.90% / 28.56% | 75.52% | [0.4671, 1.7521] |
| mixed | 0.9093 | [0.8825, 0.9360] | [0.6708, 1.1674] | 48.15% / 19.01% / 32.84% | 72.88% | [0.2664, 1.5574] |
| fresh | 1.6911 | [1.6634, 1.7189] | [1.3430, 2.0582] | 56.14% / 20.47% / 23.39% | 83.06% | [1.0483, 2.3351] |

## Paired Student t-test on the 128 fixed configurations

The primary test forms each state's matched mean difference
`d_s = (1/1024) Σ_e (R_ctrl[s,e] - R_base[s,e])`, then applies a one-sample
Student t test to the 128 `d_s` values: `t = mean(d) / (SD(d)/√128)`, df=127,
two-sided. This is a **paired** test, not an independent two-sample test: the
same 128 configurations and the same 1,024 seeds are used for every controller.
The 131,072 executions are correlated and are not treated as IID, so no
execution-level p value is reported. The existing Execution CI95 column is
conditional precision on the fixed states and is separate from this test.

| Controller | Mean gain | Gain SD | SE | t | df | Raw two-sided p | Holm p (m=4) | 95% CI mean gain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| original | 1.7191 | 2.0784 | 0.18371 | 9.3579 | 127 | 3.765e-16 | 1.506e-15 | [1.3556, 2.0827] |
| evaldist | 1.1100 | 1.5488 | 0.13690 | 8.1085 | 127 | 3.726e-13 | 7.452e-13 | [0.8391, 1.3809] |
| mixed | 0.9093 | 1.4604 | 0.12908 | 7.0441 | 127 | 1.050e-10 | 1.050e-10 | [0.6538, 1.1647] |
| fresh | 1.6911 | 2.0959 | 0.18525 | 9.1288 | 127 | 1.355e-15 | 4.064e-15 | [1.3246, 2.0577] |

Gain SD is the SD of the 128 per-state mean differences (ddof=1), SE is
`Gain SD/√128`, and the CI uses the two-sided 0.975 Student t quantile at
df=127 (1.97882). Holm adjustment covers the four controller-versus-baseline
comparisons and is valid under their shared-baseline dependence. All p values
are finite and printed in scientific notation; none is exactly zero.

The p values are precision statements about the fixed-state mean difference,
not evidence that a typical execution improves. The mean gains are only
0.13–0.27 of the pooled individual-rollout SD, and individual win fractions are
48–57% with 19–21% ties. Matching state and seed removes much of that outcome
noise: the SD of the 128 paired differences is 1.46–2.10, giving paired
standardized effects `mean gain / gain SD = 0.62–0.83`. That is why the paired
statistics are sharp even though the raw return distribution is wide.

The descriptive ordering is broadly inverse: the two smallest across-state
state-mean SDs (original 5.0477, fresh 5.1106) have the largest mean gains
(1.7191, 1.6911), while the two larger ones (evaldist 5.6627, mixed 5.6668)
have smaller gains (1.1100, 0.9093). This does not establish a "more variance,
smaller gain" law: evaldist and mixed have nearly identical state-mean SDs but
different gains, and only four fixed-state summaries are compared. The pooled
individual SD is outcome spread, not uncertainty of the mean; the baseline's
largest pooled SD (8.1216) is consistent with its wider low-return and
out-of-band outcomes, not with a less precise mean.

The 128 configurations are fixed k=100 snapshots from one policy and one source
lane, and they share the same seed list across states. The t test therefore
assumes independent configurations conditional on this design and has only 128
configuration-level units; it is not a state-population, checkpoint-
generalization, or safety claim. The return stops at finite H500, so none of
these p values is a statement about the infinite-horizon return. No simulator,
rollout, or training was run for this addendum; the reused
baseline/original/evaldist collection and the new mixed/fresh blocks were read
only, and no original collection was modified or rerun.

## Behaviour and per-state distribution

| Controller | Delta outside-band (pp) | State CI pp | Delta fall (pp) | State CI pp | Per-state mean gain positive | Per-state tail lower bound positive | Batch16 all-positive states |
|---|---:|---|---:|---|---:|---:|---:|
| original | -1.73 | [-2.03, -1.45] | +0.09 | [-0.04, +0.22] | 128/128 | 67/128 | 22/128 |
| evaldist | -1.08 | [-1.28, -0.90] | +0.12 | [+0.00, +0.25] | 123/128 | 54/128 | 8/128 |
| mixed | -0.97 | [-1.14, -0.80] | -0.79 | [-0.97, -0.63] | 120/128 | 46/128 | 5/128 |
| fresh | -1.77 | [-2.08, -1.47] | -0.16 | [-0.29, -0.04] | 125/128 | 66/128 | 22/128 |

A negative delta outside-band rate means fewer out-of-band steps (an improvement); a positive delta fall rate means more falls. Both are controller minus baseline.

## Policy-selection performance

| Controller | Candidate0 fraction | Mean per-decision Q gain | Selection fractions (candidates 0–7) |
|---|---:|---:|---|
| baseline | 1.0000 | 0.0000 | 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 |
| original | 0.1250 | 0.1226 | 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250 |
| evaldist | 0.1250 | 0.2056 | 0.1250, 0.1250, 0.1251, 0.1249, 0.1250, 0.1250, 0.1251, 0.1250 |
| mixed | 0.1250 | 0.1709 | 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250 |
| fresh | 0.1250 | 0.1116 | 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1251, 0.1250, 0.1250 |

Selected indices are near-uniform for every Q controller, which is the exchangeability prediction for eight IID candidates under any exchangeable critic and says nothing about mechanism. The executed action is a Q-selected order statistic, not a sample from the nominal policy. The per-decision Q gain is not a forecast of the cumulative 500-step paired gain.

## Uncertainty and tail reading

- Execution CIs are conditional on the 128 fixed states. The state-cluster bootstrap resamples states and is the descriptive across-state statement. Execution rows share states, seeds and lanes; they are not independent IID draws.
- The paired tail interval is asymmetric: lower side uses the baseline survivor fraction and upper side the controller survivor fraction. It is an omitted-tail interval for the finite H500 return, not a sampling CI and not a noise ceiling.
- The individual win fraction and batch16 positive fraction are different estimands. Batch16 averaging reduces sign noise; it does not change the mean, estimate expected utility, or mean that a typical execution improves.

## Paired16 merge (plan §15)

The completed paired16 comparison of original-Q versus plain reported mean gain **1.719137**, individual win/tie/loss **56.91% / 21.22% / 21.87%**, batch16 positive **85.49%**, and per-state means positive in **128/128** states. Those values are reproduced here for the original row. The caveats remain: 21.2% of executions tie and 21.9% lose; 48 states have raw execution win probability below 0.5; batch16 positive is not expected utility; the 128 states are fixed; the return stops at H500; and this is a policy intervention, not `Q^pi` validation or a state-population claim.

## Validation

- Benchmark before full evaluation: 29.43 s and 29.52 s for the first mixed/fresh block; projections 470.8 s and 472.3 s per controller, below the 720 s gate. Full evaluation wall time 895.74 s.
- Native source restore: root_states 6.89e-5, dof_pos 1.67e-6, dof_vel 8.24e-5, base_ang_vel 6.90e-5, contact 7.35e-3; replication and lane fidelity within limits; 82-field inventory exact.
- CPU/GPU Q: max errors 4.01e-5 (mixed) and 9.54e-6 (fresh), batch errors 3.24e-5 and 1.14e-5, **0 argmax disagreements**; identical-controller 5-step negative control exactly equal; zero timeouts; selection counts equal live decisions.
- Focused 22 passed; portable 233 passed / 97 deselected; gym 28 passed; learning 7 passed; Warp 32 passed / 298 deselected (18 pre-existing MULTICCD warnings); `ruff check .` and `git diff --check` clean.
- Return-stat addendum recomputed from the raw blocks only (no simulation or training): all five means and state-mean SDs, and all four paired gain means/SDs, reproduce `analysis/results.json` with exact float equality; mpmath beta and independent quadrature p values agree; SciPy 1.8.0 `ttest_rel`/`ttest_1samp` reproduces t, p, Holm adjustment and CIs; both stats scripts and Ruff exited 0.
- All raw command invocations, logs, wall times and true exit statuses are in `commands.log`; every recorded exit status is 0.

## Provenance and code identity

- Old run code hashes are recorded in `/tmp/opencode/height-q-persistent-intervention-20261110-v1/code.json` and `code_resume.json`.

| File | Old v1 run sha256 | Current sha256 | Match |
|---|---|---|---|
| `learning/offline/q_artifact.py` | `e37ae8f4296f92d0…` | `e37ae8f4296f92d0…` | yes |
| `learning/offline/fixed_state_variance.py` | `6788d59d899b314b…` | `6788d59d899b314b…` | yes |
| `learning/offline/persistent_intervention.py` | `64490b1546be3297…` | `1e73146d0ff248ad…` | no |
| `scripts/evaluate_height_q_persistent.py` | `a6a2a63c06087c64…` | `faedabb3cc848be5…` | no (naming/skip-baseline edit) |

- Current `persistent_intervention.py` and `evaluate_height_q_persistent.py` hashes differ from the v1 run. Plan §14.b.1 records the post-hoc change to the asymmetric tail helper and sign correction; this task did not edit `rollout`/`select_action` semantics. The evaluator change adds only artifact naming and baseline skipping. Baseline/original/evaldist raw blocks were reused, not rerun; seed, source-snapshot and checkpoint hashes match.
- New collector/trainer/analysis code hashes are in `experiment_manifest.json`; the old and new hashes are kept separate.
- Runtime model identity is the session configuration `opencode-go/deepseek-v4.1-flash`; it is not independently verified inside the process.

## Limitations

- 128 fixed k=100 snapshots from one policy and one source lane; not a random state population.
- Mixed contains early- and final-policy transitions; fresh contains only nominal `model_600`. Their data composition differs by construction, so the critic ordering is not a clean data-size effect.
- Persistent Q-selected actions leave the nominal on-policy distribution; the frozen critics' targets are off-policy once the branch deviates. No claim that repeated intervention validates `Q^pi`, that any critic is correct, or that the measured ordering transfers.
- H500 truncation: per-state tail lower bounds are positive for only a subset of states (see table), so the global interval is stronger than the per-state statement.
- The fresh snapshot is the last-N ring window of a 2.5M-step run, so the retained window can begin mid-episode; this is the same retention behavior as the training recorder and does not alter transition-level reset/timeout labels.
- The paired t test has only 128 configuration-level units and shares one seed list across states; it assumes independent configurations conditional on this fixed design. It is not an execution-level or state-population test, and finite-H500 p values are not infinite-horizon return or safety statements.
- Source-code hash difference between the v1 baseline/original/evaldist run and the new mixed/fresh run is disclosed above; identity checks cover seeds, states, source snapshot and checkpoint, not bitwise code equality.

## Artifacts

- `fresh_1m/replay_iter_0000600/` — fresh 1M snapshot + manifest; `fresh_1m/collect_provenance.json`.
- `q/height_q/{mixed,fresh}/100pct/` — Q artifacts.
- `eval/` — new mixed/fresh blocks, provenance, validation, code hashes.
- `analysis/` — `REPORT.md`, `results.json`, `per_state.csv`, `analysis_manifest.json`.
- `analysis/` return-stat addendum — `return_stats.py`, `return_stats.json`, `return_stats.md`, `state_means_for_scipy.npz`, `verify_scipy.py`, `scipy_verification.json`, and the `return_stats` / `scipy_verify` / `ruff_stats` stdout and time logs.
- `data_audit.json`, `experiment_manifest.json`, `commands.log`, `git_status.txt`, and the command logs in this directory.
- Reused raw blocks: `/tmp/opencode/height-q-persistent-intervention-20261110-v1/`.
- Report timestamp: 2026-11-11T06:51:29Z; return-stat addendum 2026-11-11T16:12:13Z.
