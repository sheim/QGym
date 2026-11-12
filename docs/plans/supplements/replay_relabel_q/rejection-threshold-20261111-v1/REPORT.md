# Rejection and closest-mean threshold sweep — 2026-11-11

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-rejection-threshold-20261111-v1/REPORT.md`](/tmp/opencode/height-q-rejection-threshold-20261111-v1/REPORT.md).
> Original sha256: `5cc8e9f024bda94b527e1d19c38e97b46cde548e6f5b8e290b188e3b54140d6d`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-rejection-threshold-20261111-v1/`](/tmp/opencode/height-q-rejection-threshold-20261111-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note, plus the explicitly marked
> consolidated attainment table added on 2026-11-11 at user request. See the
> [supplement index](../README.md).

Frozen `model_600`, original fair-100 PPO-data height Q, 128 fixed `k=100` native snapshots, H500, gamma 0.99, 1024 paired executions per state, reward 0 inside `0.32 ± 0.05 m` and -1 outside, absorbing true failure.

## Aggregate returns

| Controller | Kind | c | Mean H500 | Pooled SD | Min | Max | Inside-band step | Fall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reject_16 | rejection | -16 | -4.6491 | 7.0608 | -100.0000 | 0.0000 | 96.025% | 1.880% |
| reject_4p866 | rejection | -4.866 | -3.8862 | 6.4381 | -100.0000 | 0.0000 | 96.844% | 2.132% |
| reject_1p776 | rejection | -1.776 | -3.6243 | 6.3197 | -100.0000 | 0.0000 | 97.123% | 2.119% |
| reject_0p956 | rejection | -0.956 | -3.5779 | 6.3291 | -100.0000 | 0.0000 | 97.171% | 2.220% |
| reject_0p626 | rejection | -0.626 | -3.5606 | 6.3168 | -100.0000 | 0.0000 | 97.193% | 2.200% |
| distance_4p866 | distance | -4.866 | -3.7754 | 6.5290 | -100.0000 | 0.0000 | 97.051% | 1.988% |
| distance_1p776 | distance | -1.776 | -3.6129 | 6.3856 | -100.0000 | 0.0000 | 97.167% | 2.060% |
| plain | reference | — | -5.2858 | 8.1216 | -100.0000 | 0.0000 | 95.460% | 2.155% |
| best8 | reference | — | -3.5666 | 6.3288 | -100.0000 | 0.0000 | 97.177% | 2.242% |

`Pooled SD` is ddof=1 over all 131,072 individual returns; it is outcome spread, not the standard error of the mean. `Min`/`Max` are observed individual returns, not distribution bounds.

## Threshold attainment by configuration

Each configuration is one fixed state. Its 1024 executions give a mean and a Student-t interval (df=1023). `CI above/below` means the whole interval is on one side of the controller's fixed `c`; `tail above/below` uses the omitted survivor interval `[-0.657048, 0]` instead. `Individual >= c` is not expectation calibration.

| Controller | c | Mean >= c | 95% Wilson | CI above | CI below | CI uncertain | Tail above | Tail below | Tail uncertain | Individual >= c |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| reject_16 | -16 | 118/128 | [0.862, 0.957] | 118 | 10 | 0 | 118 | 10 | 0 | 91.30% |
| reject_4p866 | -4.866 | 95/128 | [0.660, 0.810] | 91 | 29 | 8 | 91 | 33 | 4 | 74.85% |
| reject_1p776 | -1.776 | 74/128 | [0.492, 0.660] | 72 | 52 | 4 | 58 | 54 | 16 | 61.92% |
| reject_0p956 | -0.956 | 58/128 | [0.370, 0.539] | 52 | 69 | 7 | 16 | 70 | 42 | 56.05% |
| reject_0p626 | -0.626 | 42/128 | [0.253, 0.413] | 34 | 78 | 16 | 0 | 86 | 42 | 52.62% |
| distance_4p866 | -4.866 | 93/128 | [0.644, 0.796] | 93 | 33 | 2 | 91 | 35 | 2 | 75.47% |
| distance_1p776 | -1.776 | 74/128 | [0.492, 0.660] | 71 | 53 | 4 | 60 | 54 | 14 | 62.76% |

Reference state means against each candidate threshold (no controller was assigned these):

| Reference | c=-16 | c=-4.866 | c=-1.776 | c=-0.956 | c=-0.626 |
|---|---:|---:|---:|---:|---:|
| plain | 116/128 | 85/128 | 61/128 | 39/128 | 22/128 |
| best8 | 121/128 | 100/128 | 73/128 | 59/128 | 44/128 |

### Consolidated threshold attainment versus plain PPO

Added 2026-11-11 from the tables above; this table is not part of the original
report sha256.

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

## Paired gains

| Controller | vs | Mean gain | State CI | Paired tail | Exec win | Batch16 positive |
|---|---|---:|---|---|---:|---:|
| reject_16 | plain | 0.6367 | [0.4132, 0.9008] | [-0.0062, 1.2814] | 25.22% | 61.29% |
| reject_16 | best8 | -1.0824 | [-1.2918, -0.8839] | [-1.7248, -0.4377] | 24.57% | 17.65% |
| reject_4p866 | plain | 1.3995 | [1.0570, 1.7704] | [0.7567, 2.0426] | 44.23% | 77.58% |
| reject_4p866 | best8 | -0.3196 | [-0.4115, -0.2402] | [-0.9619, 0.3234] | 29.26% | 30.70% |
| reject_1p776 | plain | 1.6615 | [1.3150, 2.0405] | [1.0186, 2.3046] | 54.18% | 84.03% |
| reject_1p776 | best8 | -0.0577 | [-0.1114, -0.0195] | [-0.7000, 0.5854] | 27.24% | 44.79% |
| reject_0p956 | plain | 1.7079 | [1.3660, 2.0818] | [1.0650, 2.3504] | 56.51% | 85.45% |
| reject_0p956 | best8 | -0.0112 | [-0.0272, 0.0016] | [-0.6536, 0.6312] | 23.38% | 48.40% |
| reject_0p626 | plain | 1.7252 | [1.3837, 2.0979] | [1.0823, 2.3678] | 56.86% | 85.78% |
| reject_0p626 | best8 | 0.0060 | [0.0005, 0.0118] | [-0.6363, 0.6486] | 17.84% | 49.85% |
| distance_4p866 | plain | 1.5103 | [1.1582, 1.8868] | [0.8675, 2.1543] | 54.63% | 83.42% |
| distance_4p866 | best8 | -0.2088 | [-0.3701, -0.0708] | [-0.8511, 0.4352] | 32.92% | 44.92% |
| distance_1p776 | plain | 1.6729 | [1.3222, 2.0502] | [1.0300, 2.3164] | 56.49% | 85.99% |
| distance_1p776 | best8 | -0.0463 | [-0.1532, 0.0175] | [-0.6886, 0.5972] | 34.16% | 49.72% |
| best8 | plain | 1.7191 | [1.3779, 2.0907] | [1.0763, 2.3615] | 56.91% | 85.49% |

Paired tail intervals use the tested asymmetric survivor helper: the lower side subtracts the reference survivor weight and the upper side adds the controller survivor weight. Batch16 group fractions are over 128 states × 64 contiguous groups of 16 executions; they are not `E[gain]`.

## Selection diagnostics

| Controller | Fallback | Mean feasible | Mean selected | First draw selected | Attempt mean | Mean selected Q |
|---|---:|---:|---:|---:|---:|---:|
| reject_16 | 4.608% | — | — | 95.633% | 1.331 | -4.2570 |
| reject_4p866 | 23.452% | — | — | 78.743% | 2.661 | -4.1565 |
| reject_1p776 | 48.289% | — | — | 56.483% | 4.413 | -4.1051 |
| reject_0p956 | 72.683% | — | — | 34.474% | 6.138 | -4.1054 |
| reject_0p626 | 92.907% | — | — | 16.898% | 7.552 | -4.0836 |
| distance_4p866 | 23.122% | 76.264% | 76.264% | — | — | -4.1004 |
| distance_1p776 | 47.918% | 51.058% | 51.058% | — | — | -4.0964 |

Rejection selection uses eight stochastic candidates only; the mean is not included. `Effective attempt` is the first feasible draw index (1–8), with an all-infeasible fallback counted as eight. The deterministic controller uses the exact mean plus the same eight draws and minimizes the policy-standardized squared distance among feasible candidates, so its mean-feasible rate equals its mean-selected rate.

## Caveats

- These 128 `k=100` states are fixed snapshots from one frozen policy and one source lane. They are not a state population, hardware trial, or safety evidence.
- A fixed `c` is compared with the expected H500 (plus absorbing) return from insertion. `Q(s_t, a_t) >= c` at each decision does not maintain an original budget, and a selected prediction is conditional on the actually executed policy, not `Q^pi` truth.
- The critic was trained on nominal-policy data. Persistent deviations move the branch off its target distribution; fallback violations and hidden physical state can bias selection.
- Threshold attainment need not be monotone in `c` across configurations. Multiple thresholds and controllers are examined; the reported intervals are per-comparison descriptive statements, not multiplicity-corrected guarantees.
- `Min`/`Max` and `q05` are observed sample values, not bounds. Survivor tails are separate from sampling uncertainty.
