---
name: q2-testing-and-debugging
description: Select, run, extend, and interpret Q2 tests and diagnose crashes, hangs, invalid physics, stale state, wrong rewards, or training anomalies. Use before claiming a change works, when pytest or a simulator fails, when CPU/Warp/VSim disagree, when adding a regression test, or when deciding what evidence is sufficient for backend, task, RL, or parity changes.
---

# Q2 Testing and Debugging

Read `genAI_skills/AGENTS.md`, `pyproject.toml`'s pytest settings, the failing code path,
and the nearest test. Start with the cheapest experiment that distinguishes
plausible causes; do not begin by tuning around the symptom.

## Test entry points

The default pytest suite is intentionally the unit suite:

```bash
uv run --frozen python -m pytest -q
```

Iterate with a focused file/node or expression:

```bash
uv run --frozen python -m pytest tests/unit_tests/test_task_state_liveness.py -q
uv run --frozen python -m pytest -q -k 'routing or reset'
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
```

Put cross-cutting backend, task, and full-stack tests under the configured
`tests/unit_tests/` path. Colocate small deterministic tests with implementations
under `gym/` or `learning/` when they do not require simulator/full-stack setup
and also serve as a concise usage example. Bare pytest does not collect these
colocated tests; run the relevant source root explicitly. VSim backend code
and tests remain, but machine-local setup and its test launcher are deferred.
See the [limitations and validation scope](../../../genAI_skills/README_MUJOCO.md#limitations-and-validation-scope); do not invoke a missing launcher.

A skip for unavailable CUDA, Warp, VSim, or a license is not proof for that
backend. Record what executed.

## Offline replay and fixed-policy Q evidence

- Fixed-state variance uses `scripts/evaluate_height_q_variance.py` and
  `test_fixed_state_variance.py`. Require saved insertion state/candidates,
  native restore/replication fidelity gates, controller/domain history, and
  raw `[batch, repeat, state, candidate]` returns. Keep shared-seed covariance
  when estimating candidate contrasts. Do not derive noise ceilings from noisy
  estimated means, sort away pairing, or threshold averaged binary rewards.

- `tests/unit_tests/test_offline_replay.py` protects ring ordering and overwrite,
  pre-reset successors, independent flags, recording-RNG isolation, immutable
  snapshots, schema validation, height relabeling, nested subsets, and mixture
  deduplication.
- `tests/unit_tests/test_height_fqe.py` protects continuing-return targets,
  timeout bootstrapping, terminal absorbing failure and flag precedence, Q
  artifact round-trip, frozen actor loading, and finite-rollout tail bounds.
  These are analytic/fixture oracles, not robot Q-quality evidence.
- A real snapshot/FQE/evaluation smoke verifies wiring only. For a learning
  claim, run independent simulator continuations, candidate first actions,
  nested fractions, multiple seeds, and threshold false-acceptance analysis.
  Report observation-only hidden-state limitations and the omitted continuing
  tail; do not make decreasing error a hard assertion.
- Recording must not advance training RNG. Snapshot publication must be atomic;
  an interrupted temporary directory is not a dataset. Do not silently delete
  prior experiment snapshots.

## Match evidence to the change

- Backend metadata/state/reset: shared contract tests plus backend-specific
  variants.
- Cached/assembled state: task-level liveness tests, including partial and
  floating-base reset placement.
- Canonical routing: permutation sentinels, named one-hot actuation, and named
  contact tests.
- Physics: energy/invariant tests, analytic probes, and CPU/Warp lockstep.
- Contacts: force frame/direction, body routing, weight support, impact
  impulse, and termination tests; do not compare only instantaneous peaks
  across solver formulations.
- Task/config: registry, shapes, scale round trips, asset limits/inertias, and
  a tiny CPU smoke train.
- RL/normalization/storage: deterministic unit tests, clean inference, save/
  load state, then controlled curves and per-term metrics.

## Triage known failure families

- Startup reward means are NaN but tensors are finite: wait for completed
  episodes before diagnosing the logger.
- First Warp run is silent: inspect traceback/GPU activity for JIT compilation
  before treating it as a hang.
- Task registration fails: import `gym.envs`; the fail-fast traceback identifies
  the broken declared class or config. Then run the registry manifest test.
- State/reward frozen on Warp: inspect cached tensor identity and values across
  steps; assembled public tensors must refresh in `step()`, not getters.
- Floating base resets at the wrong pose: verify DOF reset does not clobber a
  pending public root-state write.
- Contacts are zero or termination never fires: verify the post-constraint
  force computation, canonical body routing, force frame, and threshold.
- Body/foot missing: compare URDF links, `RobotLayout`, and engine-native body
  names; fixed-body fusing or a bad mapping is likely.
- Joint limits are huge: verify the URDF limit tags and Q2's explicit parser.
- Reward optimizes the wrong posture: state what input makes each `_sqrdexp`
  argument zero and test that point.
- Config change does nothing: find the actual consumer. Remove stale fields
  rather than preserving configuration that no supported backend implements.
- CPU and Warp match but VSim differs only after contact: use policy-free
  step/drop/impact/slide probes before blaming observations or PPO.

## Debugging discipline

1. Reproduce with exact task/backend/device/seed/config and save the traceback
   or artifact.
2. State competing mechanisms and the observation that would separate them.
3. Inspect current code and git history for the path; do not trust a dated
   failure-chronicle status.
4. Add temporary instrumentation narrowly; do not hide failures with a
   fallback or `try/except` bandage.
5. Fix the source and add a regression that fails without the fix.
6. Run the focused test, full unit gate, relevant optional-backend tests, and
   Ruff. Remove debug flags and temporary prints.
7. Update [limitations and validation scope](../../../genAI_skills/README_MUJOCO.md#limitations-and-validation-scope) if the finding changes supported
   limitations or consequential evidence.

Test meaningful contracts rather than defensive restatements. Before adding a
type, shape, dtype, or device guard, trace where the value was produced. Keep a
guard and focused test at an external/config/backend boundary; if the same
internal path just constructed the value correctly, consume it directly and
let the operation fail naturally if that invariant is broken.

GitHub CI runs the uv-managed portable and colocated suites, Ruff, and a package
build. The portable suite has no maintained end-to-end PPO update/checkpoint
regression and does not establish learning quality. Run applicable task
training, Warp, and Unitree checks locally; VSim execution awaits its deferred
local setup and launcher.
