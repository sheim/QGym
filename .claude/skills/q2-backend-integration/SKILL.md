---
name: q2-backend-integration
description: Executable runbook for integrating a NEW physics engine backend into Q2 (e.g. v-sim, Newton, or any future engine) — the SimBackend contract obligations, the proven pendulum→contract-tests→floating-base bring-up ladder distilled from the MuJoCo CPU/Warp integrations, the design axes to decide up front (batching, tensor liveness, device story), non-negotiable lessons from past backend bugs, and the definition of done. Load when adding/porting a physics backend, implementing SimBackend, or evaluating an engine for integration. NOT for using existing backends (q2-run-and-train) or MuJoCo-specific semantics (mujoco-backend-reference).
---

# Q2 Backend Integration Runbook

Q2 has executed this playbook twice (MuJocoCPUBackend, MuJocoWarpBackend; see
MIGRATION_PLAN.md Phases 1–3 for the full worked example) and once before that
in reverse (extracting IsaacGymBackend, Phase 0). A third integration (v-sim)
is planned as of 2026-07-10. The ladder below is the order that worked; the
lessons are the bugs that cost real time. Do not reorder the ladder — every
step gates the next.

## Step 0 — Decide the three design axes (before any code)

Write the answers into your PR description; they shape everything:

1. **Batching model.** One engine world per env (CPU backend: N × `MjData`,
   Python loop) or one batched world (Warp: `nworld=N`, one kernel)? Determines
   step cost scaling and reset granularity.
2. **Tensor story.** Are state tensors honest per-step copies (CPU backend
   syncs everything in `step()` — always fresh, slower) or zero-copy views
   into engine memory (Warp — fast, but every ASSEMBLED tensor needs an
   explicit per-step refresh)? This axis caused the worst bug in the repo
   (lesson L1). Write down, per tensor in the contract: view or copy, and
   where it gets refreshed.
3. **Device story.** Which torch devices are legal? Fail-fast rule: the
   backend either supports the requested device or raises at `setup()` —
   **no silent fallbacks** (a warp→CPU fallback was deliberately removed in
   `ebc2925`; see q2-conventions-and-change-control non-negotiable 1).

Also collect the engine's convention sheet (fill this table in your backend's
module docstring, mirroring `mujoco_backend_base.py:18-20`):

| Convention | Engine | Q2 task layer |
|---|---|---|
| Quaternion order | ? | scalar-last `[x,y,z,w]` |
| Root/DOF state layout | ? | `root_states [N,13]`, `dof_state [N*nd,2]` |
| Body velocity layout | ? | rigid_body_states lin 7:10 / ang 10:13 |
| Contact force frame + activation | ? (does it need a post-step call, like MuJoCo's `rne_postconstraint`?) | world-ish force `[N, bodies, 3]` |
| Applied-force interface | ? | full-DOF torques, backend applies free-joint offset |
| Asset format + known importer losses | ? (MuJoCo drops URDF `<limit>`, fuses rigid-joint bodies) | URDF in `resources/robots/` |

## The bring-up ladder

**1. Skeleton.** `gym/envs/base/<engine>_backend.py`, subclass `SimBackend`
(`gym/envs/base/sim_backend.py` — the ABC docstrings ARE the spec). If you
foresee an engine family (CPU/GPU variants), split a shared base like
`MuJocoBackendBase` (asset loading, metadata, contact indices) from thin
step/reset/tensor subclasses — that refactor paid off (`92cb8e9`).

**2. Fixed-base pendulum only.** Implement `setup()` for
`resources/robots/pendulum/urdf/pendulum.urdf`: load asset, apply
`cfg.asset.joint_damping`/`rotor_inertia`, `cfg` timestep/gravity, extract
`num_dof`/`num_bodies`/names, build contact index tensors, invoke the task
callbacks (`_get_env_origins`, `_process_dof_props` with a
lower/upper/velocity/effort dict — `mujoco_backend_base.py:242-263` is the
reference shape). Disable contacts for fixed-base. Then `step()` +
`reset_dof_state()`.

**3. Contract tests immediately.** Add fixtures in
`tests/unit_tests/conftest.py` (copy the `mujoco_cpu_backend` /
`mujoco_warp_backend` pairs, including importorskip + device-skip patterns) and
a `test_<engine>_backend_contract.py` that is a fixture-swapped copy of
`test_mujoco_cpu_backend_contract.py` (21 tests: shapes, metadata, physics
sanity, reset persistence, partial-reset isolation, dof_state-view liveness).

```bash
uv run python -m pytest tests/unit_tests/ -q -k "<engine>"
```

All green before proceeding. This suite exists precisely so new backends
inherit the spec for free (MIGRATION_PLAN "Test strategy").

**4. Physics sanity.** `test_<engine>_physics.py` = copy of
`test_mujoco_cpu_physics.py`: 32 damped pendulums from random ICs, zero
torque; energy never rises > 1e-4 J/step over 4000 steps; all envs converge
to < 0.01 J. This catches wrong damping/timestep/units cheaply.

**5. Cross-backend lockstep vs `MuJocoCPUBackend`** (the reference
implementation): copy `test_cross_backend_physics.py` — identical ICs, 2000
steps, per-step |Δpos|,|Δvel| < 1e-3 for the pendulum. Different integrators
may need a looser bound — if so, STATE the bound and why in the test
docstring; don't quietly widen it.

**6. RL smoke, pendulum.** Wire the backend into
`task_registry.select_backend()` (`gym/utils/task_registry.py`) — decide the
device-string or cfg trigger, keep selection fail-fast — then:

```bash
uv run scripts/train_mujoco.py --task pendulum --device <dev> --num_envs 256 --max_iterations 200 --headless --disable_wandb
```

Expected: reward climbs to ≥ +1.0 by iteration 200 (historical marks: −0.63 →
+1.22 CPU; 0.01 → +1.54 warp). If physics tests pass but RL fails → plumbing,
not physics; see lesson L1 first.

**7. Floating base.** The engine's analog of MuJoCo's `nq = nv + 1` free
joint: implement offsets, `root_states` assembly with the quaternion swizzle
at the boundary ONLY, `reset_root_state`/`set_all_root_states`,
`rigid_body_states`, enable contacts, ground plane from
`cfg.terrain` (plane only — that's the supported scope today).

**8. Legged contract + termination.** Fixtures `legged_<engine>_backend`
(mini_cheetah, 12 DOF) + fixture-swapped `test_legged_backend_contract.py`
(quat convention, doesn't fall through plane, reset persistence, CPU-lockstep
at 0.01 contact-rich tolerance) + the `test_legged_termination.py` pattern
(drop robot upside-down, base contact force > 1 N — this is the test that
caught the missing post-step contact call, `fba2deb`).

**9. Task-level liveness test (MANDATORY — the root_states lesson).**
Fixture-swap `tests/unit_tests/test_task_state_liveness.py` if it exists (it
is campaign Phase 0 work); otherwise write it now: build the full env via
`task_registry.make_env_mujoco`-equivalent, step ~50 times with torques,
assert the TASK'S CACHED `env.root_states` / `env._rigid_body_state` changed
and share storage with the backend's. The contract tests alone will NOT catch
getter-only refresh (they re-call properties; the task layer doesn't).

**10. End-to-end + benchmark.** mini_cheetah train on the new backend;
inspect gait with `play_mujoco.py`; record steps/s at a stated
num_envs/batch_size next to the existing backends' numbers. Document first-run
warmup behavior (warp JIT compiles for minutes — users WILL Ctrl-C an
undocumented pause; see q2-debugging-playbook row 2).

## Non-negotiable lessons (each one is a scar)

- **L1 — Refresh assembled tensors in `step()`, never in property getters.**
  The warp backend refreshed `root_states` only in its getter; the task caches
  the tensor once; GPU training ran on frozen observations for months and the
  "fix" was an accidental getter invocation ("unknown why this works",
  `2326b71`). The contract line is explicit: *all tensors live after step()
  returns* (`sim_backend.py:60-62`). Views may be lazy; assembled/swizzled
  tensors may not.
- **L2 — `dof_pos`/`dof_vel` must be writable views** the engine actually
  reads back on reset. Verified by the view-consistency contract tests; don't
  satisfy them with a fresh-copy getter (that's how warp's `dof_state` became
  a trap, W3 in q2-architecture-contract).
- **L3 — Conversions live at the backend boundary only.** One swizzle constant
  pair, used in ≤ 4 places (root read/write, body read, reset). A fifth use
  means you're converting in the wrong layer.
- **L4 — Know the engine's contact-force activation.** MuJoCo needed an extra
  post-step call (`rne_postconstraint`) or terminations silently never fire.
  Ask the same question of every engine; keep the termination test.
- **L5 — Asset importers lie.** MuJoCo dropped URDF effort/velocity limits and
  fused rigid-joint bodies (mini_cheetah lost its `foot`). Diff the engine's
  parsed model (names, limits, masses) against the URDF text on first load;
  re-parse what the importer discards (`_parse_urdf_limits` pattern).
- **L6 — Per-robot engine tuning goes in config, not code.** Capacity knobs
  (warp's `njmax`, `ccd_iterations`) went hardcode → isinstance hack →
  generic `cfg.mjspec_attributes` mechanism on jt/port. Give your engine the
  same generic pass-through from day one.
- **L7 — Device/init order.** `backend.device` is meaningful only after
  `setup()`; tasks pass the requested device string (bug `973b8d5`). Viewer or
  UI callbacks must attach via indirection because the env renders during
  construction (`mujoco_cpu_backend.py:162-171`).
- **L8 — State the determinism story** (same seed → same trajectory?) in the
  module docstring. CPU-MuJoCo: yes; warp: not guaranteed. Tests and
  validation claims depend on it.

## Definition of done

- Contract + physics + lockstep + legged + termination + **liveness** tests
  green: `uv run python -m pytest tests/unit_tests/ -q` (update the expected
  counts in q2-testing-and-validation).
- Pendulum AND mini_cheetah train; steps/s recorded; warmup documented.
- `select_backend` wiring is fail-fast; `q2-run-and-train` device table,
  `q2-build-and-env` platform matrix, `q2-architecture-contract` diagram, and
  MIGRATION_PLAN (or its successor doc) updated in the same change.
- Convention sheet filled in the module docstring; new engine gotchas get a
  `<engine>-backend-reference` skill if they run past a page (MuJoCo's did).

## V-sim / vlearn — SPIKE-VERIFIED convention sheet (2026-07-12, wheel 0.3.11a0)

Engine = closed-source `vlearn` package (binary wheels cp310/cp311 only →
repo repinned to Python 3.11). **Integration COMPLETE 2026-07-12** (branch
`vsim`): `gym/envs/base/vsim_backend.py` + `vsim_asset.py`, 46 opt-in tests
green (contract, energy, cross-engine parity vs analytic + MuJoCo envelope,
legged contract incl. static-weight sign check, termination, task liveness),
pendulum + mini_cheetah train end-to-end via `--backend vsim`. Benchmarks
(RTX 4080): mini_cheetah 4096 envs **317k steps/s vs warp 29.3k (10.8×)**;
warmup ~0.05 s. The full bring-up took ~1 day following this ladder — the
spike answered every unknown before backend code was written.

| Convention | vsim (verified) | Q2 task layer |
|---|---|---|
| Quaternion | `[x,y,z,w]` scalar-last — SAME as Q2 | reorder only: transform buffer is `[qx,qy,qz,qw,px,py,pz]` (quat FIRST) |
| Velocity layout | **`[ang(3), lin(3)]`** (measured; vendor stub misleading) | root_states 7:10 = buf 3:6, 10:13 = buf 0:3 |
| Contact forces | per-link force sensors (`flags="contact"` XML), **force in `[0:3]`**, torque `[3:6]`; +z reaction ≈ m·g on resting body; plane contacts ARE seen; no filters needed; set `max_num_transform_handles` on sensor defs pre-finalize | `contact_forces[:, i] = sensor_buf_i[:, 0:3]` |
| Applied forces | MOTOR mode, `gear="1.0"` ⇒ raw Nm (α=τ/I verified ±3%) | motors injected per movable joint (converter emits NONE) |
| Asset | `.vsim` XML (URDF-like); `convert_urdf_to_vsim` preserves names/limits/fixed-joints; `lower>upper` = unlimited; visual `.dae` paths WARN-unresolved (harmless, strip later); single-link assets need a dummy-link+fixed-joint to register as articulations | pipeline in `gym/envs/base/vsim_asset.py` |
| World | Z-up via `create_gym(up_axis=Vec3(0,0,1))` + `set_gravity` WORKS, but **`create_plane()` default is Y-up — rotate its transform (+Y→+Z)** | matches Q2 Z-up |
| Fixed base | `import_definitions(fixed=True)` = fixed base, joint dynamics LIVE; masked per-env joint set works | FixedRobot semantics ✓ |
| Batching / resets | native `create_environment_group([N])`; partial resets via bool `masks_buffer` on set commands | env_ids → mask |
| Lifecycle | `create_gym`/`delete_gym` singleton; **create→delete→create works in-process** → normal test fixtures with `close()` teardown | — |
| Perf | ~3.4M passive sim-steps/s @4096 mini_cheetah envs (RTX 4080); warmup 0.05 s (no JIT pause) | — |

API corrections vs the stubs/first reading: link/joint commands take index
**ranges** `(lo, hi)`, not index lists; `get_articulation` lives on
`EnvironmentDef`; one `JointStateCommand` serves positions AND velocities
(selected by `gym.get/set_joint_{positions,velocities}`);
`TransformType={COM,MODEL}`, `FrameType={ENVIRONMENT,LOCAL,WORLD}`;
articulation-def name may be the robot XML name OR the root link name — look
up with a try-both. `gym.compute_kinematics()` required after `step()` before
link transforms/velocities are valid.

Runtime requirements: system `libczmq4` (new in 0.3.11; 0.3.5 didn't need it);
`LD_LIBRARY_PATH=<site-packages>/vlearn/lib`;
`VL_WORKING_DIRECTORY=<dir containing License.key/TurboActivate.dat>`.
⚠️ License: as of 2026-07-12 the 0.3.11a0 wheel ran on a **1-day trial** —
the 0.3.5 node-lock did not carry over; proper re-activation is a human task.

## When NOT to use this skill

- Working ON an existing backend's bug → `q2-debugging-playbook` +
  `q2-architecture-contract`.
- MuJoCo/warp API details → `mujoco-backend-reference`.
- The warp parity/IsaacGym-removal work → `q2-phase4-parity-campaign` (do that
  campaign's Phase 0 BEFORE starting a new backend — otherwise the new backend
  will copy the stale-tensor pattern from warp).

## Provenance and maintenance

Distilled 2026-07-10 from MIGRATION_PLAN.md Phases 0–3, the two MuJoCo
backends, conftest fixture patterns, and the archaeology of `ebc2925`,
`973b8d5`, `fba2deb`, `2bc709b`, `2326b71`, `135adf2`. Re-verify:

```bash
ls gym/envs/base/*backend*.py                       # backend inventory (v-sim landed yet?)
grep -n "def select_backend" -A 12 gym/utils/task_registry.py
ls tests/unit_tests/ | grep -i "liveness"           # Step-9 test exists yet?
```
