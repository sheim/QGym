---
name: q2-testing-and-validation
description: Q2's test suites and evidence standards — what each suite covers, exactly how to run them (and which invocations blow up), the contract-test architecture for physics backends, CI reality vs local, what counts as proof for a physics or training change, and how to add tests. Load before claiming anything "works", before/after changing gym/ or learning/ code, when adding tests, or when pytest errors out. NOT a debugging guide (q2-debugging-playbook) and NOT the parity campaign itself (q2-phase4-parity-campaign).
---

# Q2 Testing & Validation

## The one command that matters

```bash
uv run python -m pytest tests/unit_tests/ -q
# expected (2026-07-12, branch vsim): 158 passed, 62 skipped in ~30 s
# (skips = by-design device combos + opt-in vsim tests)
bash scripts/run_vsim_tests.sh
# vsim suite (opt-in: license + CUDA): 46 passed in ~8 s
# (contract 22, physics 2, parity 2, legged 11, termination 1, liveness 1,
#  asset 5, pendulum-fixture extras; sets Q2_VSIM_TESTS/LD_LIBRARY_PATH/
#  VL_WORKING_DIRECTORY — vsim tests NEVER run in CI: node-locked license)
```

The 21 skips are structural, not missing coverage: the CPU-backend contract
tests skip their `cuda:0` device parametrization ("MuJocoCPUBackend only
supports CPU"). On a CUDA machine all warp tests genuinely execute.

**NEVER run bare `pytest` from the repo root.** There is no pytest config or
testpaths; collection reaches `tests/integration_tests/conftest.py`, whose
`pytest_sessionstart` builds every env via IsaacGym and INTERNALERRORs
(`AttributeError: 'NoneType' object has no attribute 'parse_arguments'`) — even
with `--collect-only`. Always target a path explicitly.

## Suite inventory

| Suite | Command | Status (2026-07-10) |
|---|---|---|
| Backend/task unit tests | `uv run python -m pytest tests/unit_tests/ -q` | **the real gate**; 172 collected |
| In-package legacy tests | `uv run python -m pytest gym learning -q` | 13 pass in ~2 s; what CI runs |
| Integration tests | — | DEAD without IsaacGym (INTERNALERROR on collection) |
| Regression tests | — | DOUBLY dead: needs IsaacGym AND reference file `main_output.pt` was never committed (100 KB CI cap likely why) |

## The contract-test architecture (how backends earn trust)

One suite of shape/physics/reset invariants runs against every backend via
fixtures in `tests/unit_tests/conftest.py`:

- `MockBackend` (`mock_backend.py`) — closed-form damped pendulum,
  semi-implicit Euler; validates the tests themselves.
- `mujoco_cpu_backend` / `mujoco_warp_backend` — 4/16-env pendulum fixtures.
- `legged_cpu_backend` / `legged_warp_backend` — 12-DOF mini_cheetah
  floating-base fixtures (sim_dt 0.002, terminate on `base`, penalize `thigh`).

What the invariants pin down (per file):
- `test_backend_contract.py` (44) — tensor shapes (`root_states [N,13]`,
  `dof_state [N*nd,2]`…), metadata, gravity/torque sanity, energy drift < 5%
  over 200 free steps, partial-reset isolation, **dof_pos must be a live view
  of dof_state**.
- `test_mujoco_{cpu,warp}_backend_contract.py` (42 each) — same contract on real
  backends, plus device placement.
- `test_mujoco_{cpu,warp}_physics.py` (2 each) — damped pendulum: energy never
  increases > 1e-4 J/step over 4000 steps; all 32 envs converge to < 0.01 J.
- `test_cross_backend_physics.py` (1) — **CPU and warp in lockstep 2000 steps;
  |Δpos|, |Δvel| < 1e-3 at every step** (pendulum, seed 42).
- `test_legged_backend_contract.py` (22) — mini_cheetah: quat scalar-last,
  doesn't fall through plane, reset persistence, CPU↔warp trajectory match at
  loosened 0.01 tolerance (contact-rich).
- `test_legged_termination.py` (2) — full env via `make_env_mujoco`, drops robot
  upside-down, asserts base contact force > 1 N (regression for the
  `cfrc_ext`/`rne_postconstraint` bug, commit `fba2deb`).
- `test_urdf_limits.py` (2) — URDF `<limit effort/velocity>` round-trip
  (regression for MuJoCo's importer dropping them, commit `2bc709b`).
- `test_task_skeleton.py` (13) — obs get/set scaling round-trips, reset buffers.

- `test_task_state_liveness.py` (2, added 2026-07-11) — steps a full
  mini_cheetah env (cpu/warp) and asserts the **task's cached**
  `root_states`/`_rigid_body_state` update in place after `step()` (regression
  for the warp getter-only-refresh bug; closes blind spot W4). Contract tests
  alone cannot catch this class of bug — they re-call properties per
  assertion; fixture-swap this file for every new backend.

## CI reality (do not assume CI covers you)

- `.github/workflows/unit_tests.yml` runs **only on push/PR to `main`/`dev`**,
  on Python 3.10 with `pip install -r requirements.txt`, command:
  `pytest gym learning` (the 13 legacy tests). It never touches
  `tests/unit_tests/`.
- `.github/workflows/basic_checks.yml` runs on every push: 100 KB file-size cap
  only.
- **Net: pushes to `port` run zero tests in CI.** Local
  `uv run python -m pytest tests/unit_tests/` is the only gate. Self-hosted
  IsaacGym runners were deliberately removed 2025-10 (`7cf29c7`, `0606552`).

## What counts as evidence here

1. **Physics-backend change** → full unit suite green on CPU *and* CUDA machine,
   cross-backend lockstep tests pass at stated tolerances. If you touched
   step/reset/state-sync: add or extend a contract test — the suite is the spec.
2. **Physics correctness claim** → an invariant test (energy monotonicity,
   convergence to equilibrium), not "the viewer looks right". Pattern to copy:
   `test_mujoco_cpu_physics.py` (measure a conserved/dissipated quantity with a
   stated per-step tolerance).
3. **Training change** → reward curve evidence: same task, same seed, before vs
   after, reported with iterations, num_envs, device, steps/s (wandb or the
   printed logger block). Historical baseline: pendulum reaches reward ≈ +1.2
   (CPU, 256 envs) / ≈ +1.5 (GPU, 4096) in 200 iterations. "It looks like it
   walks" is not evidence — see q2-research-methodology.
4. **Determinism claim** → `MuJocoCPUBackend` same-seed same-trajectory (this is
   a Phase-4 checklist item; regression harness in tests/regression_tests is
   dead — see above).

## Adding tests

- New backend → add a fixture in `conftest.py`, parametrize the existing
  contract files over it (that was the design intent; MIGRATION_PLAN "Test
  strategy").
- New task → cheapest meaningful test is a `make_env_mujoco` smoke +
  a couple of `_reward_*` shape checks (`[num_envs]`), mirroring
  `test_legged_termination.py`'s construction pattern.
- Keep tests engine-free where possible (MockBackend), CPU-MuJoCo where not;
  guard warp tests with `pytest.importorskip("mujoco_warp")` + CUDA skip, as the
  existing files do.
- Slowest tests are the warp physics pair (~6 s each); keep new tests in that
  budget or below.

## When NOT to use this skill

- A test is failing and you don't know why → `q2-debugging-playbook`.
- Designing the parity/validation experiments for Phase 4 →
  `q2-phase4-parity-campaign`.
- Wondering whether a historical failure was already fought →
  `q2-failure-archaeology`.

## Provenance and maintenance

Verified by execution 2026-07-10 (`port` @ `bc2bd96`, Linux, RTX 4080).
Re-verify:

```bash
uv run python -m pytest tests/unit_tests/ -q | tail -1
uv run python -m pytest gym learning -q | tail -1
uv run python -m pytest tests/unit_tests/ --collect-only -q | tail -1   # 172 as of 2026-07-10
ls tests/regression_tests/                                              # main_output.pt still missing?
grep -n "branches" .github/workflows/unit_tests.yml                     # CI still main/dev only?
```
