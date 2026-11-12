# Paired 16-execution regrouping: original-Q versus plain

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-paired16-comparison-20261111-v1/REPORT.md`](/tmp/opencode/height-q-paired16-comparison-20261111-v1/REPORT.md).
> Original sha256: `f6bc04e2a09e34196a0e27e3d7fd621ba2be11ff857e9cafda0f47917dcb9006`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-paired16-comparison-20261111-v1/`](/tmp/opencode/height-q-paired16-comparison-20261111-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../README.md).

**Question.** For each of the 128 fixed configurations, does the original-height-Q controller change the H500 height-band return relative to plain stochastic `model_600`? The primary estimand is the paired execution gain `G = G_original - G_plain`, summarized both at the individual-execution level and after regrouping the 1024 executions per state into 64 disjoint batches of 16.

**Model / ownership.** `opencode-go/deepseek-v4.1-flash` was explicitly configured for this analysis. It is recorded from the harness session configuration; runtime identity is not independently verified. Commands, cwd, host and true exit statuses are in `commands.log`.

**Scope.** Original-Q versus plain only. The 16 `evaldist` blocks are present in the raw directory but are excluded from every statistic here, as requested. No simulator, rollout or high-cost computation was run.

## Bottom line

- The paired mean gain is **1.719137** return units, identical at execution and batch16 levels. The 128 per-state mean gains are positive in **128/128** states (min 0.005541, median 0.661046, max 11.978000).
- Individual executions are **win/tie/loss = 0.569130/0.212204/0.218666** (74597/27814/28661 of 131072). Of the 27814 ties, 27358 have both returns exactly zero and 456 have identical nonzero returns; all ties have 500 live steps. They are exact equalities, not numerical near-ties.
- After averaging 16 executions, **0.854858 of the 8192 batch gains are positive**, 0.145142 negative and 0.000000 exactly zero. Per state: **22** states have all 64 batches positive, **128** have a positive majority, and **0** have none.
- The raw execution win rate (0.569130) and the pooled batch16 positive fraction (0.854858) differ by 0.285728 (batch16 minus execution). They are different estimands; the batch16 fraction is **not** expected utility, not `E[G]`, and not the probability that one execution improves.
- The H500 finite-target tail caveat remains: the global paired worst-case interval is **[1.076251, 2.361458]**, and the per-state tail lower bound is strictly positive in only **67/128** states. The raw per-state means are all positive, but the per-state worst-case tail statement is not.

## Data and identity audit

The raw directory contains 48 blocks (16 each for `baseline`, `original`, `evaldist`). This analysis loaded the 32 `baseline` and `original` blocks. Every block stores `[state128, execution64]` plus the candidate-axis `selection_counts[128,64,8]`.

| Object | Audited mapping / value |
|---|---|
| Execution order in a block | `e = call*8 + lane` (call-major, lane-minor) |
| Seed identity | `seed = 7100000 + 10*(block*8 + call)`; `env=seed`, observation=`seed+1`, action=`seed+2` |
| Per-state inventory | 16 blocks x 8 calls x 8 lanes = 1024 executions; 128 unique seeds x 8 lanes |
| Primary batch | `batch = e//16`; contiguous 16 = 2 calls x all 8 lanes; 64 batches |
| Sensitivity batch | `batch = call*8 + lane`; one fixed slot across all 16 blocks; 64 batches |
| Batch comparisons | 128 states x 64 batches = 8192 |

The lane label is the vectorized replication index derived from the documented evaluator layout. The primary batch statistics are invariant to any permutation of lanes within a call because every primary batch contains all eight lanes. The stratified sensitivity batch fixes both call and lane and varies blocks.

### Raw checks that passed

- 364 raw assertions: file presence, axes labels, exact seed lists, shapes, dtypes, finite returns/steps, zero timeouts, selection counts summing to live decisions, baseline candidate0-only selection, and execution-grid coverage.
- 25 synthetic checks: known mean/SD/win/tie/loss, tie tolerance, contiguous and stratified regroup invariance, tail formula, Student-t quantiles, Wilson interval, exact sign test, and seed formula.
- Exact regroup invariant: maximum absolute difference between the per-state direct mean and the mean of the 64 batch means is `8.882e-16`. The pooled mean is unchanged by regrouping.

### Provenance caution

The run-time source hashes recorded in `code.json` do not all match the current working-tree source. This analysis therefore imports no product code and uses only the raw tensor blocks; the stored axes, seeds and shapes are the identity evidence. The concurrent audit's `REPORT.md`, `results.json`, `per_state.csv` and audit files were read-only observations and were not modified.

| Source/observation | Run-time sha256 | Current sha256 | Match |
|---|---|---|---|
| `evaluate_height_q_persistent.py` | `a6a2a63c06087c64...` | `34e87387e238dd5b...` | NO |
| `persistent_intervention.py` | `64490b1546be3297...` | `1e73146d0ff248ad...` | NO |
| `fixed_state_variance.py` | `6788d59d899b314b...` | `6788d59d899b314b...` | yes |
| `q_artifact.py` | `e37ae8f4296f92d0...` | `e37ae8f4296f92d0...` | yes |

## Estimands and formulas

Let `G_s,e = R_original[s,e] - R_plain[s,e]` for state `s` and execution `e=0..1023`. A tie is `|G_s,e| <= 1e-9`; an audit found no nonzero value inside that tolerance, so ties are exact equalities.

- Per-state mean: `mean_e G_s,e`.
- Per-state execution SD: sample SD (`ddof=1`) over 1024 gains.
- Per-state execution win/tie/loss probabilities: fractions of the 1024 gains in `(tol, inf)`, `[-tol, tol]`, `(-inf, -tol)`.
- Primary `gain16_s,b = mean` of the 16 paired gains in batch `b`; `b=0..63`. Its sample SD/variance use `ddof=1` across the 64 batches.
- Tail interval: each survivor omits `[-TAIL_WIDTH, 0]`, with `TAIL_WIDTH = 0.99^500/(1-0.99) = 0.6570483042`. For a paired mean the lower side uses the plain survivor fraction and the upper side the original survivor fraction.

## Global results

| Quantity | Value |
|---|---:|
| Paired executions | 131072 |
| Mean paired gain | **1.719137** |
| Fixed-state execution CI95 (within-state SE) | [1.691714, 1.746561] |
| State-cluster bootstrap CI95 | [1.371995, 2.083445] |
| H500 paired tail interval | [1.076251, 2.361458] |
| Tail width | 1.285207 |
| Pooled execution SD | 5.470063 |
| Mean per-state execution SD | 4.200772 |
| Plain survivor fraction | 0.978447 |
| Original survivor fraction | 0.977585 |

The fixed-state CI is conditional on these 128 states and uses only within-state execution SEs. The state-cluster bootstrap resamples the 128 state rows and is the descriptive across-state statement. Neither treats the 131072 execution rows as independent IID draws.

## Per-state distribution across the 128 fixed configurations

Each row summarizes 128 values, one per state. These are distributions across states, not quantiles of the pooled 131072 executions or the pooled 8192 batch means.

| Per-state metric | Mean | Median | q05 | q25 | q75 | q95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean gain | 1.719137 | 0.661046 | 0.090459 | 0.325399 | 2.440377 | 5.840902 | 0.005541 | 11.978000 |
| Execution SD | 4.200772 | 3.168171 | 1.460305 | 1.991340 | 5.501488 | 10.869355 | 1.135500 | 14.117385 |
| Execution win probability | 0.569130 | 0.561035 | 0.255762 | 0.409912 | 0.724365 | 0.854932 | 0.222656 | 0.933594 |
| Execution tie probability | 0.212204 | 0.138672 | 0.000000 | 0.004883 | 0.395508 | 0.592334 | 0.000000 | 0.661133 |
| Execution loss probability | 0.218666 | 0.205566 | 0.123047 | 0.166016 | 0.255859 | 0.361670 | 0.066406 | 0.468750 |
| Gain16 SD | 1.045817 | 0.769067 | 0.359023 | 0.494030 | 1.369581 | 2.714048 | 0.263976 | 3.695421 |
| Gain16 variance | 1.616494 | 0.591465 | 0.128906 | 0.244066 | 1.875755 | 7.369026 | 0.069683 | 13.656136 |
| Gain16 win fraction | 0.854858 | 0.875000 | 0.614844 | 0.777344 | 0.984375 | 1.000000 | 0.531250 | 1.000000 |
| Gain16 tie fraction | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Gain16 loss fraction | 0.145142 | 0.125000 | 0.000000 | 0.015625 | 0.222656 | 0.385156 | 0.000000 | 0.468750 |
| Positive batches (of 64) | 54.711 | 56.000 | 39.350 | 49.750 | 63.000 | 64.000 | 34.000 | 64.000 |
| Tail lower bound | 1.076251 | 0.018114 | -0.552633 | -0.327792 | 1.796322 | 5.217252 | -0.650866 | 11.365867 |

- The SD across the 128 per-state mean gains is 2.078428; the median is 0.661046.
- Per-state mean gain is positive in **128/128** states; negative in 0.
- Batch-positive counts: all 64 positive in **22** states; a majority (>32) positive in **128**; none positive in **0**. The minimum per-state positive count is **34/64**.
- Raw execution win probability is >0.5 in **79** states, <0.5 in **48**, and exactly 0.5 in **1**. Every state still has a positive mean gain. This is the central reason a win rate is not an expected utility.
- Wilson 95% intervals for the raw execution win probability place the lower bound above 0.5 in 71 states and the upper bound below 0.5 in 39 states; 18 contain 0.5. These intervals are nominal binomial intervals over 1024 executions and ignore seed/lane clustering; the global seed-cluster interval is the honest one.

## Execution level versus batch16 level

| Quantity | Individual executions | 16-execution batches |
|---|---:|---:|
| Comparisons | 131072 | 8192 |
| Win / tie / loss counts | 74597 / 27814 / 28661 | 7003 / 0 / 1189 |
| Win / tie / loss fractions | 0.569130 / 0.212204 / 0.218666 | 0.854858 / 0.000000 / 0.145142 |
| Mean gain | 1.719137 | 1.719137 |
| Pooled gain SD | 5.470063 | 2.424473 |
| Per-state SD, mean [min, max] | 4.200772 [1.135500, 14.117385] | 1.045817 [0.263976, 3.695421] |
| Per-state win fraction, mean [min, max] | 0.569130 [0.222656, 0.933594] | 0.854858 [0.531250, 1.000000] |

The raw execution win rate is **0.569130** (seed-cluster CI95 [0.566589, 0.571625]; state-cluster CI95 [0.535751, 0.601852]). The pooled batch16 positive fraction is **0.854858** (state-cluster CI95 [0.832516, 0.876834]). Batch16 minus execution is 0.285728.

This difference is expected under a positive mean with asymmetric wins and many exact ties: averaging 16 executions reduces the noise that produces zero or negative execution gains, so the sign of the averaged gain is positive much more often. It does **not** mean that the average gain is larger (it is exactly the same number), nor that a typical execution is more likely to improve in states where the raw win probability is below 0.5. A batch16 positive fraction is therefore not an expected-utility statement and not a substitute for the mean.

### Illustrative states

| State | Mean gain | Exec win/tie/loss | Exec SD | Batch16 win | Batch16 SD | Positive batches |
|---:|---:|---:|---:|---:|---:|---:|
| 28 (Lowest raw execution win probability) | 0.030989 | 0.222656 / 0.625977 / 0.151367 | 2.040358 | 0.531250 | 0.542302 | 34/64 |
| 115 (Lowest mean gain) | 0.005541 | 0.228516 / 0.630859 / 0.140625 | 1.235761 | 0.593750 | 0.356900 | 38/64 |
| 111 (Highest mean gain) | 11.978000 | 0.838867 / 0.000000 / 0.161133 | 14.117385 | 1.000000 | 3.695421 | 64/64 |

## Batch-gain quantiles: three distinct objects

The following are deliberately separated. Mixing them is the most likely way to misread this regrouping.

### A. Pooled individual-execution gains (131072 rows)

| q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -84.582171 | -10.600962 | -3.248307 | 0.000000 | 0.167478 | 2.789999 | 10.831566 | 18.972589 | 83.451376 |

### B. Pooled batch16 gains (8192 batch means)

| q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -12.174155 | -1.732846 | -0.467947 | 0.190424 | 0.829807 | 2.574349 | 6.681338 | 9.833175 | 19.129171 |

### C. Within-state batch16 quantiles, summarized across states

For each state, compute the stated quantile of its 64 batch gains; then report the median and range of those 128 state-level quantiles. These are not the pooled quantiles in section B.

| Within-state quantile | Median across states | Min across states | Max across states |
|---|---:|---:|---:|
| q05 | -0.240490 | -4.086307 | 5.009012 |
| q25 | 0.282171 | -1.519623 | 9.619097 |
| q50 | 0.642765 | 0.011650 | 12.508737 |
| q75 | 1.117087 | 0.183541 | 14.123752 |
| q95 | 2.168054 | 0.448291 | 17.102708 |

The mean per-state batch16 variance is 1.616494, versus mean per-state execution variance / 16 = 1.603769. The ratio is 1.007934, consistent on average with near-independent 16-execution batches; this is a descriptive variance check, not an independence proof.

## Precision, signs and the tail caveat

- Statewise paired t CI95 on the 64 batch means: the lower bound is positive in **118/128** states; with a Bonferroni 0.05/128 threshold it is positive in **101/128**. This is a descriptive precision statement with a multiple-testing disclaimer, not 128 independent confirmatory tests.
- Exact sign-test p<0.05 (uncorrected) in 118/128 states; after Bonferroni, 105/128. A state with only 34/64 positive batches is not declared to have no effect: the sign of a 16-mean is a coarse, underpowered statistic, and the mean magnitude can be positive while the raw win probability is below 0.5.
- The global H500 tail interval is [1.076251, 2.361458]. It remains strictly positive. Per-state tail lower bounds are positive in only 67/128 states; the low-gain states cannot exclude a worst-case surviving tail. The finite target is H500, and the omitted survivor tail is an uncertainty interval, not a sampling CI.
- No claim of statistical insignificance is made from a negative or tied 16-execution mean alone. Raw positive frequency is reported separately, with the caveat that its nominal intervals ignore within-state seed/lane dependence.

## Partition sensitivity

Primary batches are contiguous in the stored order (`e//16`): two calls x all eight lanes. The sensitivity partition fixes one `(call, lane)` slot and averages across all 16 blocks. Both are 64 disjoint batches of 16 and both preserve the paired identity.

| Quantity | Primary contiguous | Stratified by block |
|---|---:|---:|
| Mean gain | 1.719137 | 1.719137 |
| Positive / tie / negative batches | 7003 / 0 / 1189 | 7013 / 1 / 1178 |
| Positive fraction | 0.854858 | 0.856079 |
| All 64 positive states | 22 | 25 |
| Majority-positive states | 128 | 127 |
| None-positive states | 0 | 0 |
| Mean per-state batch16 SD | 1.045817 | 1.031595 |
| Mean per-state batch16 variance | 1.616494 | 1.548496 |

The conclusion is not sensitive to this partition choice: the mean is invariant, the positive fraction is 85.5% versus 85.6%, no state has zero positive batches under either partition, and only one state fails a strict majority under the stratified partition (it has exactly 32/64).

## Answer to the comparison

**Yes, original-Q improves the measured H500 height-band return relative to plain stochastic `model_600` at these 128 fixed configurations.** The paired mean gain is +1.7191 return units; it is positive in all 128 per-state means, remains positive after collapsing to 64 disjoint 16-execution batches in 85.5% of batches, and has a strictly positive global worst-case survivor-tail bound [+1.0763, +2.3615]. The improvement is not a uniform per-execution win: 21.2% of executions tie exactly and 21.9% lose, and 48 states have a raw win probability below 0.5. The mean gain is carried by the magnitude of the wins and avoided violations. That is an expected-return statement at fixed states, not a theorem of policy improvement, not a state-population claim, and not evidence that every execution is more likely to improve.

## Limitations

- The 128 states are fixed k=100 snapshots from one frozen policy and one source lane; they are not a random state population.
- Q was trained for the nominal policy. The intervened branch leaves that distribution, so this is a policy-intervention measurement, not `Q^pi` validation and not a policy-improvement guarantee.
- The return is truncated at H500; surviving omitted tails are bounded only by the worst-case interval. The global bound is positive, but 61/128 per-state lower bounds include zero.
- Execution rows within a state share seeds, lanes and the fixed initial state. Batch means reduce variance but do not create independent state replications. CIs that resample states are wider than execution-level CIs; the latter are conditional precision only.
- The primary and stratified batch partitions are both reported; other disjoint pairings would give somewhat different batch SDs and sign counts. The mean is invariant, and the two natural partitions agree closely.
- Two of the four recorded source files differ from the run-time hashes; raw blocks are authoritative and no product code was imported. The concurrently edited shared report/results were not modified.
- `evaldist` is excluded by request; no cross-critic claim is made.

## Artifacts

- `analysis.py`: this independent analysis script.
- `run.sh`: foreground command/exit-status audit wrapper.
- `analysis_manifest.json`: inputs, hashes, environment and formulas.
- `results.json`: complete machine-readable results.
- `per_state.csv`: one row per fixed configuration.
- `batch_means_primary.csv`: 8192 primary batch means.
- `batch_means_stratified_sensitivity.csv`: 8192 stratified sensitivity batch means.
- `REPORT.md`: this report.
- `commands.log`: commands, cwd and true exit statuses.
- `stdout.log`: captured script output.

Analysis timestamp (UTC): `2026-11-11T05:44:19.950710+00:00`.
