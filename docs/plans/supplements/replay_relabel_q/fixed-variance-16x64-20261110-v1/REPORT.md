# Sixteen independent 64-return batches — 2026-11-10

> **Provenance.** Durable copy moved on 2026-11-11 from
> [`/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/REPORT.md`](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/REPORT.md).
> Original sha256: `44da6f7c6938a00c186f7307799bb99eb9531c2a42f40218ff0272ef47ffeaf9`.
> Raw artifacts, datasets, checkpoints, and logs remain in
> [`/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/`](/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/);
> artifact names below resolve there, not in this docs tree. Content is
> preserved except for migrated links and this note. See the
> [supplement index](../README.md).

## Protocol

Random selection seed 2026111064; original state indices `[24, 12, 116, 80, 50, 82, 59, 92, 4, 97, 9, 63, 90, 87, 13, 23]`. Selection precedes reading returns.
Exact original physical snapshots, noisy insertion observations, eight candidate actions and both critics' predictions are retained; no prefix rerun.
Two batches reuse all 128 original continuations; fourteen new batches add 896 continuations per state/action: 114688 new, 131072 retained rollouts.
1024 worlds = lane8 × state16 × candidate8. Each new 64-return batch contains eight calls × eight lanes. RNG identity is (call seed,lane): distinct vectorized random subsequences across lanes, not eight separately seeded generators. Candidate noise is shared within each state/lane. New call seeds 6400000..6401110 stride10 do not collide with old streams.
Frozen policy, H500, gamma .99, height reward 0 inside .32±.05m and -1 outside; observed terminal reward plus exact absorbing failure tail.

## Matched-data results

All rows summarize per-state/action quantities, not pooled-state variance. At64 uses 16 batches (15 df); at16 regroups the SAME returns into 64 batches (63 df). Medians use torch's lower-middle convention. Paired contrasts exclude nominal-minus-itself; covariance is retained.

| Quantity | At16 mean | At16 median | At64 mean | At64 median |
|---|---:|---:|---:|---:|
| fixed_action_variance | 1.271764 | 0.277224 | 0.313614 | 0.077920 |
| fixed_action_sd | 0.932266 | 0.526521 | 0.458830 | 0.279141 |
| within_batch_action_variance | 0.644969 | 0.140738 | 0.183222 | 0.033911 |
| within_batch_action_sd | 0.566918 | 0.375150 | 0.308394 | 0.184148 |
| within_batch_action_range | 1.652487 | 1.094448 | 0.900168 | 0.536622 |
| paired_estimator_variance | 1.302996 | 0.285495 | 0.365503 | 0.064551 |

rank_agreement: at16 **51.083021%**, at64 **52.777158%**.

winner_agreement: at16 **13.448661%**, at64 **18.072917%**.
Pair ties count half; winner ties select the first candidate. Repeatability is not accuracy against true Q. Action spread contains both signal and noise.

## Uncertainty

4000 whole-64-batch resamples, seed 2026111065; fixed states/candidates; paired regrouping; approximate percentile intervals with only 16 clusters, not exact variance coverage or state-population uncertainty. Actions/states stay together in every resample. Ordinary percentile bootstrap variance intervals can be biased with only 16 batches and skewed returns; they are conditional, approximate intervals.
- At64 fixed_action_variance.mean: 95% bootstrap interval **[0.247670, 0.334420]**.
- At64 fixed_action_variance.median: 95% bootstrap interval **[0.057594, 0.105392]**.
- At64 within_batch_action_variance.mean: 95% bootstrap interval **[0.165520, 0.202223]**.
- At64 paired_estimator_variance.mean: 95% bootstrap interval **[0.255878, 0.444997]**.
- Mean variance ratio64/16: **0.246597**, 95% paired cluster-bootstrap interval [0.20036818087100983, 0.2693323493003845]; 1/4 is the IID scaling reference, not a fitted noise ceiling.

The old 128-state summary and this random 16-state subset have different coverage; do not directly interpret their differences as rollout-count effects. `original_subset_at16` and `new_only_at64_sensitivity` are retained in JSON. More batches reduce variance-estimation uncertainty, not hidden-state critic bias, finite-horizon truncation or state-population uncertainty.

## Pooled diagnostics and critics

First reward -1 rate **0.000000%**, termination **2.670288%**. Rates use individual binary rewards, never thresholded averages.
Mean finite/absorbing return **-5.312794**. Mean tail-only interval **[-5.952297287757066, -5.31279406610262]**; surviving omitted tail [-0.6570483,0]. Paired tail interval mean width 1.278371. These exclude sampling uncertainty.

| Critic | MAE to pooled finite estimate | Bias | Pair accuracy to noisy estimate | Predicted action range |
|---|---:|---:|---:|---:|
| Original fair100 | 1.495400 | 0.641948 | 0.542411 | 0.195765 |
| Eval-distribution100 | 2.741345 | 0.189999 | 0.459821 | 0.383914 |

## Fidelity, timing and artifacts

All 82 native integration/model/task/controller/DR entries use the same index mapping; every continuation asserts exact restoration. Native one-step source/replica and lane restore gates, plus command/push candidate pairing and lane distinction pass (`validation.json`, `validation_resume.json`). This proves numerical one-step fidelity, not bitwise long-horizon equivalence. World reordering can amplify numerical solver differences in closed loop; reused versus new batches also differ in native world placement.
Selected-state joint-limit audit: {'selected_out_of_limit_positions': 0, 'selected_states_with_excess': 0, 'max_excess_rad': 0.0, 'all_insertion_observations_candidates_predictions_exact': True}. Native positions are preserved without reset clamping. Adapter remains restricted to this 1024-world nonsleeping Go2Trot/Warp configuration.
New collection **399.989556s** (6.666 min), first block **29.411492s**, retained. Original full collection was 459.0989s. Timings exclude environment setup, native validation and offline analysis; no extra critic rollout is used.

- `raw_returns.pt`: explicit [batch64,repeat64,state16,candidate8] axes, returns/first rewards/survival, old seeds and new (seed,lane) identities.
- `new_block_00.pt`..`new_block_13.pt`: retained new blocks and timings.
- `insertion.pt`, `insertion_audit.json`: exact selected snapshot, observations, candidates and both original predictions.
- `statistics.pt`, `per_state_action.csv`, `results.json`: full statistics, tail bounds, matched grouping, bootstrap intervals and sensitivity summaries.
- `provenance.json`, `code_provenance.json`, `commands.txt`: selection, stream identities, source hashes, code hashes/revision/dirty status and CLI.

Reproduce analysis from BigRedGym: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python scripts/summarize_height_q_extension.py /tmp/opencode/height-q-fixed-variance-16x64-20261110-v1`.

**Top-Q sampled-action policy selection remains explicitly deferred.** No noise-ceiling claim, commits, todos, plan completion or archival.

## Validation record

- Focused tests: 7 passed.
- Portable gate: 229 passed, 97 deselected.
- Colocated gym/learning: 35 passed (28 + 7).
- Warp: 32 passed, 294 deselected; 18 existing MULTICCD cylinder-pair support
  warnings in contract fixtures. Collection has no physics warnings.
- Ruff, touched-file format check and `git diff --check`: clean.
- `cuda_noise_audit.json`: real-policy CUDA observation/action draws share all
  candidate noise and differ across all 28 lane pairs. Separate diagnostic seeds
  99000001/99000002 do not enter collected returns.
- Analysis assertions: exact original data reuse and insertion observations/actions/
  predictions, 896 unique new seed/lane identities, axes/finite bounds/binary
  rewards, matched pooling, rank formulas and covariance identity pass.

Logs: `/tmp/opencode/height-q-16x64-{benchmark,full,analysis,portable,colocated,warp}.log`.
The benchmark printed a harmless tensor-copy advisory; `torch.as_tensor` removed
it before resume, with no change to seeds or retained benchmark data.

Interpretation: variance decreased approximately fourfold with fourfold rollout
count; ranking repeatability is still weak. Neither this nor critic agreement
against noisy finite estimates establishes useful policy selection.
