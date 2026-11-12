> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/return_stats.md`](/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/return_stats.md).
> Original sha256: `a63c07d99a62209722b0bd1c83ac1170715f7ae85c5e43e949f294cc0d0c8fff`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-two-critic-datasets-20261111-v1/`](/tmp/opencode/height-q-two-critic-datasets-20261111-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../../README.md).

## Return dispersion and paired Student t-test

| Controller | Mean H500 return ± pooled individual-rollout SD | SD of 128 state means |
|---|---:|---:|
| baseline | -5.2858 ± 8.1216 | 6.7391 |
| original | -3.5666 ± 6.3288 | 5.0477 |
| evaldist | -4.1758 ± 7.1375 | 5.6627 |
| mixed | -4.3765 ± 6.8288 | 5.6668 |
| fresh | -3.5946 ± 6.4504 | 5.1106 |

The pooled individual-rollout SD is over all 131,072 returns for that controller (128 states x 1024 matched seeds, ddof=1). It describes outcome spread, **not** the standard error of the mean. The separate SD of the 128 state-mean returns (ddof=1) describes between-state spread after each state mean averages 1024 seeds.

| Controller | Mean gain | Gain SD | SE | t | df | Raw two-sided p | Holm p (m=4) | 95% CI mean gain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| original | 1.7191 | 2.0784 | 0.18371 | 9.3579 | 127 | 3.765e-16 | 1.506e-15 | [1.3556, 2.0827] |
| evaldist | 1.1100 | 1.5488 | 0.13690 | 8.1085 | 127 | 3.726e-13 | 7.452e-13 | [0.8391, 1.3809] |
| mixed | 0.9093 | 1.4604 | 0.12908 | 7.0441 | 127 | 1.050e-10 | 1.050e-10 | [0.6538, 1.1647] |
| fresh | 1.6911 | 2.0959 | 0.18525 | 9.1288 | 127 | 1.355e-15 | 4.064e-15 | [1.3246, 2.0577] |

Primary statistic: for each state `s`, form the matched difference `d_s = (1/1024) sum_e (R_ctrl[s,e] - R_base[s,e])`, then test the 128 `d_s` values with a one-sample Student t test (`t = mean(d) / (SD(d)/sqrt(128))`, df=127). This is a **paired** comparison, not an independent two-sample test: every controller uses the same 128 configurations and the same 1024 seeds. The 131,072 executions are correlated and are not treated as IID; no execution-level p value is reported.

The p value addresses only the null of zero mean paired difference under independent configuration sampling. The 128 configurations are fixed k=100 snapshots from one policy and one source lane, and they share the same seed list, so the test is conditional on this design and does not establish a state-population, checkpoint-generalization or safety claim. Holm adjustment covers the four controller-versus-baseline comparisons and remains valid under their shared-baseline dependence. All p values are finite and printed in scientific notation; none is exactly zero. The return is truncated at H500, so no p value is a statement about the infinite-horizon return.

No simulator, rollout or training was run for this addendum. The reused baseline/original/evaldist collection and the new mixed/fresh blocks were read only; no original collection was modified or rerun.
