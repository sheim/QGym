# Fixed-state variance at 64 rollouts — 2026-11-10

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-fixed-variance-64-20261110-v1/REPORT.md`](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/REPORT.md).
> Original sha256: `dca51a58671226f19d220e889c1a9b0ba800490422e5b27695668ae29d7ba8d3`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-fixed-variance-64-20261110-v1/`](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/);
> artifact names below resolve there, not in this docs tree. Relative artifact
> links were rewritten to that original directory; the cross-report link points
> to the migrated durable report. Content is otherwise preserved apart from
> this note. See the [supplement index](../README.md).

## Protocol and formulas

Reused [original raw returns](/tmp/opencode/height-q-fixed-variance-20261110-v1/raw_returns.pt)
with shape [8,16,128,8] = [batch,repeat,state,candidate]. Original batches
0–3 form group 0 and 4–7 group 1: shape [2,64,128,8]. The grouping is identical
for every state/action and preserves matched continuation noise across actions.
No new physics or random regroupings were run; no additional independent data
are claimed. All original H500, gamma .99, reward, frozen policy, insertion
state, restoration and truncation caveats remain in force; see the
[original report](../fixed-variance-20261110-v1/REPORT.md).

For each fixed state/action, M_g is its mean of 64 returns. Observed estimator
variance is sample Var_g(M_g), correction=1, equal to (M_0−M_1)^2/2.
This estimates the variance of a **64-return estimator**, not of the pooled
128-return mean. Within-group action variance is sample Var_action(M_g),
correction=1 over eight candidates, then summarized across two groups/128 states.
It mixes true action differences and estimation noise, not pure action signal.
Paired variance uses M_g(action)−M_g(nominal), retaining covariance; nominal
self-contrasts are excluded. Summaries use torch.median (lower middle for even
counts), matching the original report. Across-state variance is never pooled
into the estimator variance.

## Observed results

| Quantity | 16-rollout mean | 16-rollout median | 64-rollout mean | 64-rollout median |
|---|---:|---:|---:|---:|
| Fixed-action estimator variance | 1.204215 | 0.563815 | **0.343431** | **0.060939** |
| Fixed-action estimator SD | 0.914370 | 0.750876 | 0.382869 | 0.246858 |
| Within-group across-action variance | 0.767078 | 0.166738 | **0.316077** | **0.042201** |
| Within-group across-action SD | 0.620655 | 0.408336 | 0.372277 | 0.205428 |
| Within-group across-action range | 1.817829 | 1.200961 | 1.107343 | 0.600737 |
| Paired estimator variance | 1.209124 | 0.284019 | **0.280147** | **0.040696** |

Group 0/1 across-action variance mean (median) separately:
0.305306 (0.047770) / 0.326847 (0.040662).
The average fixed-action SD **0.382869** differs from sqrt(average variance)
**0.586030**; the corresponding 16-rollout values are 0.914370 and 1.097367.

Ordering agreement across the two 64-return groups is **0.5546875 (55.46875%)**
versus 0.5179817044 at 16; winner agreement is **0.1953125 (19.53125%)**
versus 0.1450892857. Pair ties score one-half, winner ties use first index;
64-return pair-tie fraction is zero. Agreement measures repeatability, not
correctness against true Q. These comparisons reuse the same data and are
not independent significance tests or noise-ceiling estimates.

Unchanged pooled diagnostics (mean/median): paired estimated difference
−0.065416/−0.021472; absolute paired estimated difference 0.322115/0.161948;
individual paired-return variance 17.896427/5.188242. The pooled 128-return
targets and critic comparisons are unchanged by regrouping.

## Independent-continuation projection — distinct from observation

If continuations are independent and identically distributed for a fixed
state/action, Var(mean of 64) = Var(mean of 16)/4. This assumes independence
**across continuations**, not across actions. Applying the same scaling to
paired contrasts respects their shared-noise covariance.

| Projected quantity at 64 | Mean | Median |
|---|---:|---:|
| Fixed-action estimator variance | **0.301054** | **0.140954** |
| Fixed-action estimator SD | 0.457185 | 0.375438 |
| Paired estimator variance | **0.302281** | **0.071005** |

Projected sqrt(mean variance) is 0.548684. These are projections from the
eight-batch 16-return estimates (seven df), not additional observed batches.
Across-action spread is not projected by dividing by four because it includes
true action differences as well as noise.

## Precision and conclusion

**Only two disjoint 64-return estimates give one degree of freedom per
fixed-state/action or paired variance. Precision is weak.** The many states,
actions and candidate pairs do not create additional independent 64-rollout
batch replicates, and actions are correlated through matched noise. The small
observed median relative to the projected median is not evidence of better
than 1/n scaling: one-df sample variances are highly variable and skewed.
Mean/median are descriptive, not confidence intervals.

Observed mean variance drops from 1.204215 to 0.343431, with a 0.301054
independence projection. Ranking repeatability remains weak at 55.47%.
This preliminary result answers the variance question without new collection;
more independent 64-rollout batches are needed for a precise empirical variance
or a strong ordering-stability conclusion. No new full collection was launched.
The plain-policy versus top-Q sampled-action selection experiment remains
unimplemented/unrun pending further user approval.

Finite/absorbing returns retain survivor omitted tail [-0.6570483,0]; original
mean tail-only interval [-5.91887,-5.27575] and paired tail uncertainty remain.
More repetitions reduce sampling noise, not truncation bias. These data do not
establish critic-only limitation or action-filter suitability.

## Reproduction and checks

From `BigRedGym/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python /tmp/opencode/height-q-fixed-variance-64-20261110-v1/analyze.py > /tmp/opencode/height-q-fixed-variance-64-20261110-v1/analysis.log 2>&1
```

[Script](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/analyze.py),
[full numerical results](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/results.json),
[per-state/action statistics](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/statistics.pt),
[execution log](/tmp/opencode/height-q-fixed-variance-64-20261110-v1/analysis.log).
Checks passed: axes/shape, 128 unique seeds, grouping/pooling identities,
two-batch squared-difference variance identity, paired covariance identity,
and exact agreement with original mean fixed-action/paired variance at 16.
All arithmetic uses float64. No product source changes or simulator retests
were required for this offline statistical reanalysis.
