# Two additional height-Q critics: equal-epoch data comparison

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/REPORT.md`](/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/REPORT.md).
> Original sha256: `e6dd2477db458bd0d3f72e376e6eced53c911f8cc59a3d32e540dafbaf0977c6`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-two-critic-datasets-20261111-v1/`](/tmp/opencode/height-q-two-critic-datasets-20261111-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../../README.md).

**Question.** On the same 128 fixed k=100 states and the same 1024 execution seeds, how do four frozen height-Q critics compare when each selects the maximum of eight fresh stochastic `model_600` actions at every decision? The new critics are fitted at equal 102.4 epochs on the existing 2M mixed replay and on 1M fresh frozen-policy transitions.

**Scope and estimand.** Fixed-state policy intervention, H500, gamma 0.99, binary height reward 0.32 +/- 0.05 m, absorbing true failure. This is not policy optimization, not `Q^pi` validation, and not an equal-data-size comparison (mixed has 2M transitions, fresh has 1M).

## Data identity

- Baseline/original/evaldist blocks reused from `/tmp/opencode/height-q-persistent-intervention-20261110-v1`; no rerun.
- New mixed/fresh blocks from `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/eval`.
- Every block stores `[state128, execution64]`; execution order is `e = call*8 + lane` and `seed = 7100000 + 10*(block*8 + call)`.
- Primary batch16 is contiguous `e//16` (two calls x all eight lanes), 64 disjoint batches per state.
- Source insertion sha256 matches across reused and new runs; all controllers share the same 128 states and seed lists.

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

Cross-check: the `original` row reproduces the paired16 report's mean 1.719137, individual 56.91/21.22/21.87%, batch16 85.49% and tail [1.0763, 2.3615]; the paired state CI uses the same bootstrap seed.

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

## Behaviour changes and per-state distribution

| Controller | Delta outside-band (pp) | State CI pp | Delta fall (pp) | State CI pp | Per-state mean gain positive | Batch16 all-positive states |
|---|---:|---|---:|---|---:|---:|
| original | -1.73 | [-2.03, -1.45] | +0.09 | [-0.04, +0.22] | 128/128 | 22/128 |
| evaldist | -1.08 | [-1.28, -0.90] | +0.12 | [+0.00, +0.25] | 123/128 | 8/128 |
| mixed | -0.97 | [-1.14, -0.80] | -0.79 | [-0.97, -0.63] | 120/128 | 5/128 |
| fresh | -1.77 | [-2.08, -1.47] | -0.16 | [-0.29, -0.04] | 125/128 | 22/128 |

## Reading the comparison

- **original**: mean paired gain 1.7191; individual win/tie/loss 56.91%/21.22%/21.87%; batch16 positive 85.49%; per-state means positive 128/128; per-state tail lower bound positive 67/128.
- **evaldist**: mean paired gain 1.1100; individual win/tie/loss 51.54%/19.90%/28.56%; batch16 positive 75.52%; per-state means positive 123/128; per-state tail lower bound positive 54/128.
- **mixed**: mean paired gain 0.9093; individual win/tie/loss 48.15%/19.01%/32.84%; batch16 positive 72.88%; per-state means positive 120/128; per-state tail lower bound positive 46/128.
- **fresh**: mean paired gain 1.6911; individual win/tie/loss 56.14%/20.47%/23.39%; batch16 positive 83.06%; per-state means positive 125/128; per-state tail lower bound positive 66/128.

In the behaviour table, a negative delta outside-band rate means fewer out-of-band steps (an improvement); a positive delta fall rate means more falls. Both are controller minus baseline.

The individual win fraction and the batch16 positive fraction are different estimands. Averaging 16 executions reduces sign noise but does not change the mean, estimate expected utility, or imply that a typical execution improves. Execution rows are not independent IID draws: they share fixed states and seed/lane structure. The state-cluster bootstrap is the honest across-state statement; the execution CI is conditional precision.

The paired tail interval is asymmetric: the lower side uses the baseline survivor fraction and the upper side the controller survivor fraction. It is an omitted-tail uncertainty interval for the finite H500 return, not a sampling CI and not a noise ceiling.

## Data and training budgets

- Fresh collection: 611 policy steps x 4096 worlds = 2502656 appended, 1000000 retained, 8192 unique env/episode pairs, 15 true terminations, 4082 timeouts, 20.49 s.
- Mixed dataset: 2000000 transitions; 800,000 updates x batch 256 = 102.4 epochs.
- Fresh dataset: 1000000 transitions; 400,000 updates x batch 256 = 102.4 epochs.
- Training wall time: mixed 745.76 s, fresh 371.89 s.

## Limitations

- Four critics are compared on one frozen checkpoint, one reward, and 128 fixed states from one source lane; this is not a state-population or checkpoint-generalization claim.
- The mixed dataset contains early-policy and final-policy transitions; the fresh dataset is nominal `model_600` only. Their state/return distributions differ by construction, so a critic difference is not a clean data-size effect.
- Persistent Q-selected actions leave the nominal on-policy distribution; the frozen critics' targets are off-policy once the branch deviates. No claim that repeated intervention estimates `Q^pi` or that any critic is 'correct' is made.
- The paired t test treats the 128 fixed configurations as units and shares one seed list across states; it is not an execution-level or state-population test. Finite-H500 p values are not infinite-horizon return or safety statements.
- The return is truncated at H500; the omitted survivor tail is bounded only by the reported interval.
- Runtime model identity is the session configuration `opencode-go/deepseek-v4.1-flash`; it is not independently verified inside the process.
