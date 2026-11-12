# Replay/relabel height-Q experiment reports

Durable copies of the generated Markdown reports from the archived
[replay/relabel/FQE plan](../../../../../.opencode/workspace/plans/archive/replay_relabel_q_plan.md).
The reports cover fixed-state return variance, a persistent eight-candidate Q
intervention, the paired 16-execution regrouping, two additional critics,
the rejection/closest-mean threshold sweep, and an original-PPO-reward versus
height-Q tradeoff on the same paired executions.

## What was moved

These are the report documents, not the experiment outputs. Raw tensors,
datasets, checkpoints, logs, CSVs, manifests, and run directories remain under
`/tmp/opencode/height-q-*` at the original paths. Each report carries a top
provenance note with the original report path, original sha256, and original
artifact directory. Report text is preserved; only relative links that would
otherwise resolve inside this docs tree were migrated (and the added provenance
notes). The rejection report additionally carries one explicitly marked
consolidated threshold-attainment table added at user request on 2026-11-11.

Compatibility symlinks remain at every original `/tmp/opencode/.../REPORT.md`
(or `analysis/REPORT.md`, `analysis/return_stats.md`) path and point here.
Do not run a summarizer against an original experiment directory if it would
write `REPORT.md`: that write would follow the symlink and replace the durable
copy. The raw scripts, data, and logs are otherwise still addressable at their
original paths.

## Index

| Date | Report | Status and main caveat | Original artifact directory |
|---|---|---|---|
| 2026-11-10 | [Fixed-state mid-gait variance](fixed-variance-20261110-v1/REPORT.md) | Initial eight-batch, 16-return variance and critic comparison. Superseded for variance precision by the 64-return and sixteen-batch reports; remains the source of the fixed insertion states and critic predictions. | `/tmp/opencode/height-q-fixed-variance-20261110-v1/` |
| 2026-11-10 | [Variance at 64 rollouts](fixed-variance-64-20261110-v1/REPORT.md) | Preliminary two-group regrouping of the same 128 returns (one degree of freedom). Superseded for independent 64-return precision by the sixteen-batch report; no new physics was run. | `/tmp/opencode/height-q-fixed-variance-64-20261110-v1/` |
| 2026-11-10 | [Sixteen independent 64-return batches](fixed-variance-16x64-20261110-v1/REPORT.md) | Strongest repeated-batch variance result: 16 independent 64-return batches, 15 df, bootstrap intervals. It uses a 16-state subset, so do not treat its difference from the 128-state summary as a rollout-count effect. | `/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/` |
| 2026-11-10/11 | [Persistent sample-eight intervention](persistent-intervention-20261110-v1/REPORT.md) | Measured H500 policy-intervention gain for two critics. Includes the post-hoc audit corrections: paired survivor-tail arithmetic and occupancy/fall sign conventions. Not policy improvement, `Q^pi` validation, safety, or a state-population claim. | `/tmp/opencode/height-q-persistent-intervention-20261110-v1/` |
| 2026-11-11 | [Paired 16-execution regrouping](paired16-comparison-20261111-v1/REPORT.md) | Independent raw-block reanalysis of original-Q versus plain. Correct asymmetric paired tail; 128/128 per-state means positive but only 67/128 per-state tail lower bounds positive. Two of four recorded code hashes differ from the current tree; raw blocks are authoritative. | `/tmp/opencode/height-q-paired16-comparison-20261111-v1/` |
| 2026-11-11 | [Two additional critics at equal epochs](two-critic-datasets-20261111-v1/REPORT.md) | Consolidated mixed/fresh versus original/evaldist comparison, including the paired t-test, policy-selection diagnostics, and paired16 cross-check. Not an equal-data-size comparison; fixed states and one checkpoint. | `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/` |
| 2026-11-11 | [Two additional critics — analysis report](two-critic-datasets-20261111-v1/analysis/REPORT.md) | Earlier generated analysis report for the same experiment. Preserved for provenance; the consolidated top-level report above contains the later addendum and is the better entry point. | `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/` |
| 2026-11-11 | [Return dispersion and paired t-test addendum](two-critic-datasets-20261111-v1/analysis/return_stats.md) | Descriptive SD and paired Student-t addendum for the two-critic experiment. The documented p-values are fixed-state precision statements, not infinite-horizon or safety claims. | `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/` |
| 2026-11-11 | [Rejection and closest-mean threshold sweep](rejection-threshold-20261111-v1/REPORT.md) | Final seven-controller sweep. Aggregate expectation meets the fixed threshold only at `c = -16` and `c = -4.866`; attainment is not monotone in `c`. Threshold intervals are per-configuration and not multiplicity-corrected guarantees. | `/tmp/opencode/height-q-rejection-threshold-20261111-v1/` |
| 2026-11-11 | [Original PPO reward versus height-Q gain](original-reward-tradeoff-20261111-v1/REPORT.md) | New paired `plain` versus `original` rollouts on the same 1024 executions per state. Measures the original dt-weighted training return (zero after true termination, no absorbing tail) and the existing height return. The original-return paired mean is negative while the height gain is positive; fixed-state, finite-horizon, no population or safety claim. | `/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/` |

## Supersession and reading order

- For **independent-batch variance precision**, read the
  [sixteen-batch report](fixed-variance-16x64-20261110-v1/REPORT.md) after the
  [initial report](fixed-variance-20261110-v1/REPORT.md). The
  [two-group 64-rollout report](fixed-variance-64-20261110-v1/REPORT.md) is a
  preliminary one-df reanalysis, not additional independent collection.
- For **paired original-Q versus plain**, the
  [paired16 report](paired16-comparison-20261111-v1/REPORT.md) is the
  raw-only independent analysis; its asymmetric survivor-tail interval
  supersedes any earlier doubled-tail arithmetic. The
  [persistent report](persistent-intervention-20261110-v1/REPORT.md) carries the
  corresponding audit corrections for its own first draft.
- For the **four-critic comparison**, start with the
  [consolidated report](two-critic-datasets-20261111-v1/REPORT.md); the
  [analysis subreport](two-critic-datasets-20261111-v1/analysis/REPORT.md) and
  [return-stat addendum](two-critic-datasets-20261111-v1/analysis/return_stats.md)
  are supporting artifacts.
- For **threshold attainment**, the
  [rejection sweep](rejection-threshold-20261111-v1/REPORT.md) is the final
  report. The earlier nominal threshold pilot produced no standalone
  `REPORT.md`; its evidence is summarized in the archived plan's extra-test
  section under `/tmp/opencode/height-q-threshold-pilot-20261111-v1/`. A
  pre-fix duplicate `REPORT.md` in
  `/tmp/opencode/height-q-rejection-summary-scratch-20261111/` is not part of
  this record: it repeated one controller's results across every threshold and
  was superseded by the final sweep.
- For the **original PPO reward tradeoff**, read the
  [tradeoff report](original-reward-tradeoff-20261111-v1/REPORT.md). It is a
  fresh paired `plain`/`original` collection, not a reuse of the persistent
  baseline blocks; the height numbers are compared to the persistent run only
  as an aggregate reproducibility check.

## Common scientific boundaries

The height-Q reports use one frozen `model_600` checkpoint, 128 fixed `k=100`
snapshots from one policy and one source lane, H500 with gamma 0.99, a binary
height reward inside `0.32 ± 0.05 m`, and an absorbing true-failure tail. The
original-reward tradeoff report uses the same states, checkpoint, horizon and
height objective but additionally records the dt-weighted original PPO training
return with all rewards zero after true termination; that objective has no
absorbing tail and is not bounded by the height interval. None of the reports
establishes a state population, policy improvement in the theorem sense,
`Q^pi` accuracy, transfer, hardware behavior, or safety. Execution-level
intervals are conditional on the fixed states; the state-cluster intervals are
the descriptive across-state statements.

## Integrity manifest

[MOVE_MANIFEST.md](MOVE_MANIFEST.md) records the pre-move sha256 and size for
each source report, its original path, and its durable destination. The
post-move verification re-inserted the original bytes by removing the added
provenance note and reversing the documented link rewrites; every reconstructed
file matched its recorded sha256.
