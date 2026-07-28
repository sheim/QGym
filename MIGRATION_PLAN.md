# IsaacGym → MuJoCo Warp Migration Plan

## Goal

Replace the IsaacGym/PhysX physics backend with MuJoCo Warp while keeping the
RL training infrastructure backend-independent. Runners and algorithms remain
unchanged; task logic and rewards may be refactored where they currently attach
robot semantics to backend-native numeric indices. The result will support:

- **Linux GPU** — MuJoCo Warp (`mujoco_warp`), batched across CUDA
- **Linux CPU** — MuJoCo Warp on CPU (Warp supports CPU execution)
- **Mac (Apple Silicon)** — plain `mujoco.mj_step` loop; no CUDA required

IsaacGym remains a fully-functional backend throughout the migration and is
only removed once the MuJoCo Warp backend is validated end-to-end.

---

## Style Guidelines

We use uv for environment management and ruff for style and linting.
Do not `try ... catch` exceptions, things should just fail instead.
This is dev code, we want the code to fail fast and obviously.
Always run ruff on the code at the end of a session to clean up.

## Architecture

```
SimBackend (ABC)  ←  gym/envs/base/sim_backend.py
    ├── IsaacGymBackend   (Phase 0 — existing, extracted)
    ├── MuJocoWarpBackend (Phase 1 — GPU + CPU Linux)
    └── MuJocoCPUBackend  (Phase 2 — Mac / CI fallback)

TaskSkeleton  (unchanged: get_states / set_states / scaling / episode tracking)
    └── BaseTask(TaskSkeleton)   ← holds self._backend, no engine imports
         ├── FixedRobot          ← delegates step/reset/init to backend
         └── LeggedRobot         ← same + terrain; gym/sim shims during migration
              └── ConcreteTask   ← backend-independent rewards and observations
```

`task_registry.py` selects the backend based on config / platform:

```python
def select_backend(cfg, device: str) -> SimBackend:
    if device.startswith("cuda"):
        return MuJocoWarpBackend()   # fails if mujoco_warp not installed
    if platform.system() == "Darwin":
        return MuJocoCPUBackend()
    return MuJocoCPUBackend()
```

### SimBackend contract (summary)

| Method / property | Description |
|---|---|
| `setup(cfg, num_envs, device, task)` | Load asset, build N parallel envs, prepare sim |
| `step(torques)` | Apply forces and advance physics; all tensors live after return |
| `reset_dof_state(env_ids)` | Commit dof_pos/dof_vel for given envs to the simulator |
| `reset_root_state(env_ids)` | Commit root_states for given envs (legged robots) |
| `set_all_root_states()` | Commit root_states for all envs (push_robots) |
| `dof_pos` | `[num_envs, num_dof]` — live tensor view |
| `dof_vel` | `[num_envs, num_dof]` — live tensor view |
| `dof_state` | `[num_envs * num_dof, 2]` — (pos, vel) pairs; dof_pos/dof_vel are views into this |
| `root_states` | `[num_envs, 13]` — pos(3) quat(4) linvel(3) angvel(3) |
| `contact_forces` | `[num_envs, num_bodies, 3]` |
| `num_dof`, `num_bodies`, `dof_names`, `body_names`, `find_body_index()` | Metadata |

---

## Test strategy

Tests live in `tests/unit_tests/` and require no physics engine — they run on
any machine including Mac CI.  The `MockBackend` implements `SimBackend` with
a simple pendulum integrator (semi-implicit Euler) so physics-correctness tests
are fast and deterministic.

When a new backend is implemented, **the same test files are run against it**
by adding a fixture in `conftest.py`.  This is the primary correctness gate
for each new backend.

Run with: `uv run python -m pytest tests/unit_tests/ -v`

---

## Phase 0 — Backend extraction  ✅ COMPLETE

**Goal:** Refactor only.  IsaacGym behaviour is unchanged; the codebase gains
a clean seam for swapping physics engines.

### Changes

| File | Change |
|---|---|
| `gym/envs/base/sim_backend.py` | **New** — `SimBackend` ABC |
| `gym/envs/base/isaac_gym_backend.py` | **New** — all IsaacGym API calls extracted here |
| `gym/envs/base/base_task.py` | Rewritten — holds `self._backend`, no isaacgym imports; `gym`/`sim` shim properties for LeggedRobot compat |
| `gym/envs/base/fixed_robot.py` | `_step_physx_sim` and world-building route through `self._backend` |
| `gym/envs/base/legged_robot.py` | Backend created in constructor; `_step_physx_sim` / `_post_physx_step` route through backend; remaining `self.gym.*` calls flow via BaseTask shims (cleaned up in Phase 3) |
| `gym/utils/helpers.py` et al. | `isaacgym` imports made conditional (`try/except`) so the package loads without IsaacGym installed |
| `gym/envs/__init__.py` | Task registration wrapped in `try/except ImportError` |
| `tests/unit_tests/` | **New** — test infrastructure (see below) |

### Tests

**Unit tests** (`tests/unit_tests/`) — no IsaacGym, no MuJoCo, no CUDA required:

| File | What it tests |
|---|---|
| `mock_backend.py` | `SimBackend` implementation with pendulum physics |
| `conftest.py` | `device` fixture parametrised over `["cpu", "cuda:0"]` |
| `test_backend_contract.py` | 44 tests: tensor shapes, metadata, physics sanity, reset contract |
| `test_task_skeleton.py` | 13 tests: `get_states`, `set_states`, scaling, reset buffers |

**Integration test** (existing IsaacGym):
```
uv run scripts/train.py --task mini_cheetah_ref --headless
```
Expected: training runs to completion, rewards improve.  Learning curves must
be identical to pre-refactor (no behavioural change).

---

## Phase 1 — MuJoCo backends, FixedRobot

**Goal:** `MuJocoCPUBackend` and `MuJocoWarpBackend` both implementing `SimBackend`.
Pendulum trains to convergence using the new backends on CPU and GPU Linux.
CPU backend is implemented first — it has no Warp dependency, validates the interface,
and produces a running contract test suite that the Warp backend can then inherit.

### Sub-tasks

#### 1a — Extend `SimBackend` ABC with contact-index properties  ✅ COMPLETE

`FixedRobot._initialize_sim()` reads `penalised_contact_indices` and
`termination_contact_indices` from the backend immediately after `setup()`.
These are not yet in the ABC.  Add them as concrete default-raise properties
(same pattern as `rigid_body_states`):

```python
@property
def penalised_contact_indices(self) -> torch.Tensor:
    raise NotImplementedError

@property
def termination_contact_indices(self) -> torch.Tensor:
    raise NotImplementedError
```

`IsaacGymBackend` already implements them and is unaffected.

| File | Change |
|---|---|
| `gym/envs/base/sim_backend.py` | Add two default-raise properties |

---

#### 1b — Implement `MuJocoCPUBackend`  ✅ COMPLETE

**New file:** `gym/envs/base/mujoco_cpu_backend.py`

One `mujoco.MjModel` shared across all envs; one `mujoco.MjData` per env.

**`setup(cfg, num_envs, device, task)` sequence:**
1. Resolve asset path from `cfg.asset.file`
2. `mjm = mujoco.MjModel.from_xml_file(asset_path)` — MuJoCo's built-in URDF importer
3. Apply physics params from cfg: `mjm.opt.timestep = cfg.sim_dt`, gravity, `dof_damping`, `dof_armature`
4. Extract metadata — `num_dof`, `num_bodies`, `dof_names`, `body_names`
5. Build contact index tensors from `cfg.asset.penalize_contacts_on` / `terminate_after_contacts_on`
6. Call task callbacks: `_get_env_origins()`, `_process_dof_props(props, 0)` where props dict has
   keys `lower`, `upper`, `velocity`, `effort` (matching the existing IsaacGym interface)
7. Create `self._datas = [mujoco.MjData(mjm) for _ in range(num_envs)]`
8. Allocate PyTorch state tensors:
   ```python
   self._dof_state_t  = torch.zeros(num_envs, num_dof, 2, device=device)
   self._dof_pos_view = self._dof_state_t[..., 0]   # [N, num_dof] view — satisfies dof_state contract
   self._dof_vel_view = self._dof_state_t[..., 1]   # [N, num_dof] view
   self._root_states_t    = torch.zeros(num_envs, 13, device=device)
   self._contact_forces_t = torch.zeros(num_envs, mjm.nbody, 3, device=device)
   ```

**`step(torques)`:** write `qfrc_applied`, call `mj_step` for each env, bulk-copy numpy→torch.

**`reset_dof_state(env_ids)`:** write torch views back into `d.qpos`/`d.qvel`, call `mj_forward`.

**Coordinate note:** for fixed-base robots `nq == nv == num_dof` — no free-joint offset.
Validated by asserting `mjm.nq == mjm.nv` in `setup()`.

| File | Change |
|---|---|
| `gym/envs/base/mujoco_cpu_backend.py` | **New** — `MuJocoCPUBackend(SimBackend)` |

---

#### 1c — Contract tests for `MuJocoCPUBackend`  ✅ COMPLETE

Add fixture in `tests/unit_tests/conftest.py` using the existing `pendulum.urdf`.
Parametrise `test_backend_contract.py` over `MockBackend` and `MuJocoCPUBackend`.
All 44 contract tests must pass before proceeding to 1d.

```
uv run python -m pytest tests/unit_tests/ -v -k "mujoco_cpu"
```

| File | Change |
|---|---|
| `tests/unit_tests/conftest.py` | Add `mujoco_cpu_pendulum_backend` fixture |

---

#### 1d — Backend injection into `FixedRobot` and `task_registry`  ✅ COMPLETE

**`gym/envs/base/fixed_robot.py`:** add `backend=None` kwarg.  If provided, use it directly;
otherwise fall back to `IsaacGymBackend`.  Fully backward-compatible.

**`gym/utils/task_registry.py`:** add `select_backend(cfg, device)` with platform/import detection:

```python
def select_backend(cfg, device: str):
    import platform
    if platform.system() == "Darwin":
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        return MuJocoCPUBackend()
    try:
        import mujoco_warp
        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend
        return MuJocoWarpBackend()
    except ImportError:
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        return MuJocoCPUBackend()
```

Add `make_env_mujoco(name, env_cfg, device)` that constructs the backend and passes it
as `backend=` to the task constructor.

| File | Change |
|---|---|
| `gym/envs/base/fixed_robot.py` | Add `backend=None` kwarg |
| `gym/utils/task_registry.py` | Add `select_backend()` + `make_env_mujoco()` |

---

#### 1e — Physics sanity test (CPU backend, pendulum)  ✅ COMPLETE

Before running any RL, verify the physics are correct.  Add a script / pytest test that:

1. Instantiates `MuJocoCPUBackend` with N environments at random initial conditions
   (positions and velocities drawn uniformly from a wide range)
2. Runs for several thousand steps with **zero torque** — the pendulum has joint damping,
   so it must always dissipate energy
3. Asserts two things:
   - **Monotonic energy decrease:** at every step, mean total mechanical energy ≤ previous step
   - **Convergence:** after sufficient steps all envs reach the same fixed energy
     (bottom equilibrium `E = 0`, within a small tolerance) regardless of IC

```
uv run python -m pytest tests/unit_tests/test_mujoco_cpu_physics.py -v
```

This test is backend-agnostic and will be re-run against `MuJocoWarpBackend` in 1g.

| File | Change |
|---|---|
| `tests/unit_tests/test_mujoco_cpu_physics.py` | **New** — damped pendulum convergence test |

---

#### 1f — RL smoke test (CPU backend, pendulum only)  ✅ COMPLETE

`scripts/train_mujoco.py --task pendulum --num_envs 256 --max_iterations 200 --device cpu`

Result: reward improved from −0.63 → +1.22 over 200 iterations at ~16,600 steps/s.
All 80 unit tests pass (21 GPU tests skipped on CPU-only machine).

---

#### 1g — Implement `MuJocoWarpBackend`  ✅ COMPLETE

**New file:** `gym/envs/base/mujoco_warp_backend.py`

Uses `mjw.make_data(mjm, nworld=num_envs)` for fully vectorised GPU execution.

Key differences from `MuJocoCPUBackend`:

| Concern | CPU Backend | Warp Backend |
|---|---|---|
| World creation | N `MjData` instances | `mjw.make_data(mjm, nworld=N)` |
| `step()` | Python loop + `mj_step` | `mjw.step(mjm, mjd)` (one kernel) |
| State tensors | numpy copy → torch | `wp.to_torch(mjd.qpos)` (zero-copy) |
| Device | always CPU | `cuda` or `cpu` via `wp.init()` |

The `dof_state` view contract is satisfied because `dof_pos` and `dof_state` both
reference the same underlying Warp array.  `reset_dof_state` writes into the torch
view (which is zero-copy into Warp storage) then calls `mjw.mj_forward`.

| File | Change |
|---|---|
| `gym/envs/base/mujoco_warp_backend.py` | **New** — `MuJocoWarpBackend(SimBackend)` |

---

#### 1h — Contract + physics tests for `MuJocoWarpBackend`  ✅ COMPLETE

Add `mujoco_warp_pendulum_backend` fixture in `conftest.py` (skipped if `mujoco_warp`
unavailable).  Run full contract suite and the damped-pendulum physics test from 1e.

```
uv run python -m pytest tests/unit_tests/ -v -k "mujoco_warp"
```

---

#### 1i — GPU convergence + performance benchmark ✅ COMPLETE

`scripts/train_mujoco.py --task pendulum --num_envs 4096 --max_iterations 200 --device cuda:0`

Result: reward 0.01 → 1.54 over 200 iterations at ~255,000 steps/s (RTX 4080).
15× throughput vs CPU backend (16,600 steps/s with 256 envs).

---

### MuJoCo coordinate note

For `FixedRobot` (fixed-base), `qpos` and joint DOFs are the same — no offset.
`LeggedRobot` is handled in Phase 3.

URDF assets are loaded natively via MuJoCo's built-in URDF importer
(`mujoco.MjModel.from_xml_file(urdf_path)`).

---

## Phase 2 — Mac / CPU fallback  ✅ COMPLETE

Implemented during Phase 1.  `MuJocoCPUBackend` exists and passes all contract
tests.  `select_backend()` routes `--device cpu` to it; Mac always gets CPU backend.

---

## Phase 3 — LeggedRobot migration

**Goal:** `LeggedRobot` fully migrated to `SimBackend`.  Remove `gym`/`sim`
shim properties from `BaseTask`.  Mini-cheetah trains with MuJoCo backends
on CPU and GPU.

**MVP scope:** flat ground plane, no heightfield/trimesh terrain, no projectiles.

### Key design decisions

**Quaternion convention:** MuJoCo uses scalar-first `[w,x,y,z]`.  IsaacGym and
the entire task layer use scalar-last `[x,y,z,w]`.  Conversion happens **inside
the backend only** — two swizzle operations (read qpos → root_states, write
root_states → qpos during reset).  Task-layer code is untouched.

**Terrain:** The `Terrain` class (generates numpy heightfield data) stays in the
task layer.  The upload to the physics engine moves into `backend.setup()`.
MVP = ground plane only; heightfield/trimesh added later.

**Projectiles:** Deferred.  Deeply IsaacGym-specific (multi-actor env,
`gym.create_sphere`).  Gated behind `isinstance(backend, IsaacGymBackend)`.
Mini-cheetah default config has `push_robots.toggle = False` and no projectiles.

**`_create_envs()` → `backend.setup()`:** All asset loading, env creation, and
body-index computation moves into the backend, matching the FixedRobot pattern.

### Sub-tasks

#### 3a — Pure-torch quaternion utilities  ✅ COMPLETE

`legged_robot.py` imports `quat_rotate_inverse`, `quat_from_euler_xyz`,
`to_torch`, `get_axis_params` from `isaacgym.torch_utils` with no fallback.

Create `gym/utils/torch_quat.py` with pure-torch implementations.  Update
import fallback blocks in `legged_robot.py`, `gym_math_wrappers.py`,
`fixed_robot.py`.

| File | Change |
|---|---|
| `gym/utils/torch_quat.py` | **New** — `quat_rotate_inverse`, `quat_from_euler_xyz`, `quat_apply`, `normalize` |
| `gym/envs/base/legged_robot.py` | Update `except ImportError` fallback |
| `gym/utils/gym_math_wrappers.py` | Import from `torch_quat` instead of `None` |

---

#### 3b — `backend=` kwarg for LeggedRobot  ✅ COMPLETE

Same pattern as FixedRobot.  Also fix the `_parse_cfg` ordering bug
(`_parse_cfg` runs before `super().__init__`, causing device mismatch).

| File | Change |
|---|---|
| `gym/envs/base/legged_robot.py` | Add `backend=None` kwarg, move `_parse_cfg` after `super().__init__` |
| `gym/envs/mini_cheetah/mini_cheetah.py` | Pass through `backend=` |
| `gym/envs/mini_cheetah/mini_cheetah_osc.py` | Pass through `backend=` |
| `gym/envs/mini_cheetah/mini_cheetah_ref.py` | Pass through `backend=` |
| `gym/envs/mit_humanoid/mit_humanoid.py` | Pass through `backend=` |
| `gym/envs/mit_humanoid/humanoid_running.py` | Pass through `backend=` |
| `gym/envs/mit_humanoid/lander.py` | Pass through `backend=` |

---

#### 3c — Floating-base coordinate handling in MuJoCo backends ✅

Both backends currently assume `nq == nv` (fixed-base).  For floating-base:
`nq = 7 + num_joints`, `nv = 6 + num_joints`.

Detect `has_free_joint = (mjm.nq == mjm.nv + 1)`.  Set `qpos_offset = 7`,
`qvel_offset = 6`.  `dof_pos = qpos[:, 7:]`, `dof_vel = qvel[:, 6:]`.

Assemble `root_states` from `qpos[:, 0:7]` + `qvel[:, 0:6]` with quaternion
swizzle `[w,x,y,z] → [x,y,z,w]`.

Implement `reset_root_state(env_ids)` and `set_all_root_states()` with
reverse swizzle.  Apply torques to `qfrc_applied[:, qvel_offset:]` only.
Enable contacts (remove `geom_contype[:] = 0` for floating-base).

| File | Change |
|---|---|
| `gym/envs/base/mujoco_cpu_backend.py` | Floating-base detection, coordinate slicing, root state assembly, quat swizzle |
| `gym/envs/base/mujoco_warp_backend.py` | Same |

---

#### 3d — Ground plane in MuJoCo backends ✅

Use `mujoco.MjSpec` to add a plane geom after loading the URDF:
```python
spec = mujoco.MjSpec.from_file(urdf_path)
ground = spec.worldbody.add_geom()
ground.type = mujoco.mjtGeom.mjGEOM_PLANE
ground.size = [100, 100, 0.1]
ground.friction = [static_friction, dynamic_friction, 0.0001]
mjm = spec.compile()
```

Only add when `cfg` has terrain config and `mesh_type == "plane"`.

| File | Change |
|---|---|
| `gym/envs/base/mujoco_cpu_backend.py` | Ground plane creation in `setup()` |
| `gym/envs/base/mujoco_warp_backend.py` | Same |

---

#### 3e — Rigid body states in MuJoCo backends ✅

MuJoCo sources: `d.xpos [nbody, 3]`, `d.xquat [nbody, 4]` (scalar-first),
`d.cvel [nbody, 6]` (angular-first then linear).

Assemble `[N * num_bodies, 13]`:
```python
rbs[..., 0:3]   = xpos                    # position
rbs[..., 3:7]   = xquat[..., [1,2,3,0]]   # wxyz → xyzw
rbs[..., 7:10]  = cvel[..., 3:6]          # linear velocity
rbs[..., 10:13] = cvel[..., 0:3]          # angular velocity
```

| File | Change |
|---|---|
| `gym/envs/base/mujoco_cpu_backend.py` | Implement `rigid_body_states` property |
| `gym/envs/base/mujoco_warp_backend.py` | Same |

---

#### 3f — Migrate `_initialize_sim()` and `_create_envs()` ✅

Replace the 137-line `_create_envs()` (47 gym calls) with `backend.setup()`.
Move terrain/env creation logic into `IsaacGymBackend.setup()` for backward
compat.  Move task-level init (base_init_state, feet_indices) into LeggedRobot
using `backend.body_names` / `backend.find_body_index()`.

| File | Change |
|---|---|
| `gym/envs/base/legged_robot.py` | Rewrite `_initialize_sim()`, delete `_create_*` methods |
| `gym/envs/base/isaac_gym_backend.py` | Extend `setup()` to handle terrain + env creation |

---

#### 3g — Migrate `_init_buffers()` ✅

Replace gym tensor acquisition with backend properties.  Remove
`register_dof_state()` hack.

| File | Change |
|---|---|
| `gym/envs/base/legged_robot.py` | Use `self._backend.root_states`, `.dof_pos`, etc. |
| `gym/envs/base/isaac_gym_backend.py` | Remove `register_dof_state()` |

---

#### 3h — Migrate `_reset_system()` and `_push_robots()` ✅

Replace `gym.set_dof_state_tensor_indexed` → `backend.reset_dof_state()`.
Replace `gym.set_actor_root_state_tensor_indexed` → `backend.reset_root_state()`.
Replace `gym.set_actor_root_state_tensor` → `backend.set_all_root_states()`.

| File | Change |
|---|---|
| `gym/envs/base/legged_robot.py` | Route through backend |

---

#### 3i — Remove gym/sim shims, gate projectiles ✅

`prepare_sim` gated behind `isinstance(IsaacGymBackend)`.  Projectiles
only called from `_create_envs()` which is IsaacGym-only.  `gym`/`sim`
shim properties in `base_task.py` kept until Phase 4 (IsaacGym removal).
Commented-out dead code cleaned up throughout `legged_robot.py`.

---

#### 3j — Legged robot contract tests ✅

New `tests/unit_tests/test_legged_backend_contract.py` using mini_cheetah URDF.
Tests: `num_dof == 12`, `root_states` shape + quaternion convention,
`rigid_body_states` shape, robot doesn't fall through ground, reset persists,
cross-backend trajectory comparison.

| File | Change |
|---|---|
| `tests/unit_tests/test_legged_backend_contract.py` | **New** |
| `tests/unit_tests/conftest.py` | Add mini_cheetah fixtures |

---

#### 3k — End-to-end mini_cheetah smoke test ✅

```
uv run scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64 --max_iterations 50
PYTHONPATH=. .venv311/bin/python scripts/train_mujoco.py --task mini_cheetah --device cuda:0 --num_envs 4096 --max_iterations 200
```

Expected: reward improves, no crashes, no NaNs.

---

#### 3l — HumanoidRunning compatibility ✅

Replace `self.gym.get_actor_rigid_body_dict()` with
`self._backend.body_names` lookup.

| File | Change |
|---|---|
| `gym/envs/mit_humanoid/humanoid_running.py` | Use backend body_names |

---

### Ordering

3a, 3b, 3c, 3d, 3e can proceed in parallel.  3f requires all of these.
Then 3g → 3h → 3i sequentially.  3j scaffolded early, finalized after 3i.
3k after 3i.  3l after 3g.

### Gotchas

- **MuJoCo URDF free joint:** auto-created, name may be empty — skip in DOF name extraction
- **Contact forces:** MuJoCo `cfrc_ext` is body-frame `[torque, force]`; IsaacGym is world-frame `[force]` — may need reward threshold tuning
- **URDF mesh paths:** mini_cheetah uses relative paths (`meshes/*.dae`) — MuJoCo resolves relative to URDF location
- **Warp sliced views:** `qpos[:, 7:]` from `wp.to_torch` is non-contiguous — verify writes propagate through `mjw.forward()`
- **Quaternion swizzle:** MuJoCo scalar-first `[w,x,y,z]` ↔ task-layer scalar-last `[x,y,z,w]`

---

## Phase 4 — Validation and IsaacGym removal

**Goal:** Confirm correctness, benchmark performance, remove IsaacGym.

### Validation checklist

| Check | Method |
|---|---|
| Unit tests pass on GPU, CPU, Mac | `uv run python -m pytest tests/unit_tests/` |
| Pendulum converges (both backends) | Same seed, compare final reward and curve shape |
| Mini-cheetah converges (both backends) | Same seed, compare reward and gait |
| Checkpoint portability | Save with IsaacGym backend, load and run with MuJoCo Warp |
| Determinism | Same seed → identical trajectory on `MuJocoCPUBackend` |
| Performance | Wall-clock per iteration, GPU utilisation; target ≥ IsaacGym throughput |

**Pendulum physics-fidelity — ✅ DONE 2026-07-24 (branch `vsim`).** An analytic
energy-pump + LQR swing-up controller (`scripts/pendulum_fidelity.py`) drives the
pendulum identically on every backend over the deterministic `reset_to_uniform`
grid, so any cross-engine divergence is pure physics. Result (1024 envs): CPU,
warp, and vsim all **100% catch, mean 1.08 s**; angular-divergence RMS vs CPU is
9e-7 rad (warp, float noise) and 5e-4 rad (vsim, genuine cross-solver, negligible
and non-growing under the stabilizing LQR). Interactive plots:
`notebooks/pendulum_fidelity.py` (marimo). The IsaacGym leg stays the separate
checkpoint-portability check below. See `q2-phase4-parity-campaign` Phase 2b.

### Mini Cheetah reference-tracking parity — ❌ PARITY NOT YET ACHIEVED

`mini_cheetah_ref` is the Phase 4 contact-rich parity task. It exercises the
floating base, 12 PD-controlled joints, ground contacts, termination logic,
reference-trajectory observations/rewards, and checkpoint transfer across:

- MuJoCo CPU (`mujoco`, `cpu`)
- MuJoCo Warp (`mujoco`, `cuda:0`)
- vsim (`vsim`, `cuda:0`)

**Validity warning (2026-07-27):** the first Warp campaign is invalid. During a
legged reset, `reset_dof_state()` refreshed assembled `root_states` before the
pending root reset was committed. This clobbered the requested base pose and
spawned Warp robots around `z=0.008` instead of the configured `z=0.5` in the
drop probe. The fix preserves pending root state during DOF reset and is covered
by `test_floating_base_reset_placement_{cpu,warp}`. All Warp legged checkpoints
and curves produced before that fix must be discarded and regenerated.

**Post-fix campaign result (2026-07-27, seed 7, 250 iterations):** all three
backends completed the prescribed run without a crash or NaN, but the RL
transfer gate failed. Gate 1 passed (26 CPU/Warp contract tests and 60 vsim
tests). Gate 2 also passed: CPU and Warp were effectively identical in the
contact-free step probe (joint-trajectory RMS `2.25e-7 rad`) and close in the
drop probe (settled-height RMS `1.26e-5 m`). vsim's expected solver difference
was concentrated in contact (`1.32e-2 m` settled-height RMS and `8.34 N`
GRF-time RMS versus CPU); its contact-free joint-trajectory RMS was `2.60e-2
rad`.

**Drop-discrepancy investigation (2026-07-27):** the probe originally inherited
one implicit initialization step from `TaskSkeleton.reset()`, so it did not
start from the requested state. The probe now restores the deterministic state
immediately before recording. With that correction, CPU, Warp, and vsim agree
to numerical precision before ground contact (vsim/CPU maximum error before
`t=0.20 s`: `<4e-7 m` base height and `<1.1e-8 rad` joint position). At the
500 Hz physics rate, vsim responds at `0.200 s`; MuJoCo responds at `0.202 s`.
The URDF conversion preserves the foot spheres and transforms, but the contact
configuration is not equivalent:

- MuJoCo uses Newton, up to 100 iterations, zero margin/gap, and compliant
  `solref=[0.02, 1]` / default `solimp`.
- vsim uses 8 solver iterations, its rigid/default material, and a plane with
  `contact_offset=0.01`. No explicit rigid material is assigned, so the
  configured terrain friction/restitution is not applied.
- MuJoCo's three friction slots are sliding/torsional/rolling; the current
  `[static_friction, dynamic_friction, 0.0001]` assignment therefore does not
  represent separate static/dynamic friction.
- vsim's link order is `base,LH,RH,LF,RF`, while injected force sensors retain
  XML order `base,RF,LF,RH,LH`. The backend currently copies sensor slot `i`
  into contact-body slot `i`, violating the contact tensor/body-name contract.

At 500 Hz, MuJoCo's first total foot force is `192 N` and rises smoothly; vsim
starts at `447 N`, spikes to `1,440 N` on the next substep, then briefly returns
to zero. Raising vsim from 8 to 16 solver iterations removes the zero-force
substep and lowers force RMS, but 32/64 iterations do not improve trajectory
RMS. A diagnostic soft-contact material (`stiffness=5,000`, `damping=100`)
roughly halves short-horizon height and force RMS, confirming contact
compliance as the leading physics parameter. Before another RL run: fix the
sensor/link permutation, apply explicit terrain/robot materials in both
backends, then tune vsim soft contact against the drop curve.

Final training metrics:

| Backend | Total reward | `reference_traj` | Episode time | Steps/s |
|---|---:|---:|---:|---:|
| CPU | 8.040 | 0.872 | 4.63 s | 5,690 |
| Warp | 5.951 | -0.008 | 2.06 s | 74,046 |
| vsim | 6.722 | 0.383 | 2.36 s | 304,597 |

Transfer evaluation (`reset_to_range`; each cell is mean reward / survival,
train backend down, evaluation backend across):

| Train \ Eval | CPU | Warp | vsim |
|---|---:|---:|---:|
| CPU | 8.471 / 95.7% | 8.289 / 96.5% | 4.899 / 0.0% |
| Warp | 5.278 / 15.6% | 5.045 / 12.5% | 4.226 / 0.0% |
| vsim | 5.242 / 0.0% | 5.054 / 0.0% | 6.443 / 75.0% |

The CPU policy transfers well to Warp, confirming useful CPU/Warp runtime
compatibility. However, the Warp-trained policy is weak even in-domain, vsim
cross-transfer collapses immediately in both directions, and deterministic
closed-loop CPU/Warp DOF RMS is `0.265 rad` for the CPU policy (`0.531 rad` for
the Warp policy). Warp also emitted three `opt.ccd_iterations ... needs to be
increased` warnings while evaluating the vsim policy. Those failures block
parity. The low-level probes are close, so the next investigation should focus
on training-distribution differences, early termination/contact amplification,
the CCD warning, and why Warp's `reference_traj` term falls from `0.582` to
approximately zero. Evidence is saved under
`logs/mc_fid`, `logs/mini_cheetah_ref_{cpu,warp,vsim}`, `logs/mc_rl_eval`, and
`logs/mc_rl_sample`.

#### Gate 0 — canonical robot layout and backend mappings

**Goal:** `LeggedRobot`, tasks, policies, and checkpoints see one stable robot
layout regardless of the engine's link, DOF, motor, or sensor order. Native
engine indices are private backend implementation details. Complete this gate
before contact-parameter tuning or another RL campaign.

##### Canonical task-facing contract

Add a `RobotLayout` (name TBD) resolved once during backend setup. It defines:

- ordered canonical DOF names;
- ordered canonical actuated-joint names;
- ordered canonical robot-body names (never MuJoCo's `world` body);
- exact named semantic groups such as feet, individual legs, arms, and end
  effectors; group names are asset-defined and do not assume every robot has
  `RF/LF/RH/LH`;
- a layout version stored with checkpoints.

For mini-cheetah, freeze the existing URDF/legacy policy order as layout v1:
`RF, LF, RH, LH`, with joints ordered `haa, hfe, kfe` within each leg. Keep the
canonical order explicit rather than deriving it from whichever engine loaded
the asset. Resolve names to cached tensor indices at setup; rewards must not do
string lookups every step.

The public `SimBackend` tensors and metadata use canonical order:

- `dof_pos`, `dof_vel`, `dof_state`: canonical DOF order;
- input torques: canonical full-DOF order;
- `body_names`, `rigid_body_states`, `contact_forces`: canonical robot-body
  order;
- `find_body_index()`, penalized-contact indices, and termination-contact
  indices: canonical body indices.

Each backend retains private native buffers and builds explicit mappings:

- MuJoCo: canonical DOFs to `qpos`/`qvel` addresses and canonical bodies to
  compiled body IDs; omit body 0 (`world`) from the public body space.
- vsim: canonical DOFs to native articulation DOFs; motors through
  `MotorDef.dof_index`; force sensors through `ForceSensorDef.link_index` (or
  its exact link name), then native link to canonical body. Never equate motor
  order, sensor order, and articulation order.
- Refresh gathers native state into canonical tensors. Reset and torque
  application scatter canonical tensors into native buffers.

Define `contact_forces` semantically as the net collision-contact force on each
canonical robot body, in world coordinates and Newtons. Verify that MuJoCo's
and vsim's selected force sources satisfy this meaning; do not treat matching
shape and ordering as sufficient.

##### Task-layer changes

- `LeggedRobot`: consume the resolved layout; build feet/contact groups from
  exact canonical names rather than substring/order assumptions. Keep generic
  PD, observation concatenation, reset, and permutation-invariant rewards
  unchanged.
- `FixedRobot`: use named actuated joints instead of positional boolean masks.
- `mini_cheetah_ref`: replace the `0:3`, `3:6`, `6:9`, `9:12` reference slices
  and four-slot GRF phase mask with cached leg/foot group indices and explicit
  per-leg phase offsets.
- `mini_cheetah_osc`: replace semantic strides such as `0:12:3` with a named
  joint group.
- Humanoid tasks: follow up by replacing leg/arm splits and individual semantic
  joint numbers with named groups. This is required before claiming the
  interface is general beyond mini-cheetah.
- `TaskSkeleton` and runner action/observation plumbing remain unchanged:
  canonical ordering makes their positional tensor interface safe.
- Save the layout version, ordered DOF/actuator names, and ordered
  observation/action fields in every new checkpoint. Metadata-free checkpoints
  emit a legacy-layout warning and are assumed to use the current layout.
  Pre-Gate-0 CPU/Warp mini-cheetah checkpoints use layout v1; pre-Gate-0 vsim
  checkpoints used reverse native order and must be regenerated or migrated.
  Loading a checkpoint with known mismatched metadata fails clearly.

##### Implementation checklist

- [x] Verify the historical CPU/Warp/checkpoint action order, then freeze it as
      mini-cheetah layout v1.
- [x] Add the canonical layout/schema and exact-name validation.
- [x] Add native↔canonical DOF/body mappings to MuJoCo CPU and Warp.
- [x] Add native↔canonical DOF/body/motor/sensor mappings to vsim.
- [x] Make all public backend metadata and tensors canonical and robot-only.
- [x] Refactor mini-cheetah semantic rewards and actuator selection to named
      groups.
- [x] Add checkpoint layout metadata and mismatch validation.
- [x] Confirm no task or reward reads a backend-native index.

##### Meaningful tests

Keep this suite small and diagnostic:

1. **Permuted-backend round trip:** a fake backend deliberately reverses DOF,
   body, motor, and sensor order. Unique sentinel values must appear under the
   correct canonical names after refresh, reset, and torque application.
2. **Named actuator routing:** apply a one-hot torque/target to every canonical
   mini-cheetah joint in turn and assert that MuJoCo and vsim drive the same
   named physical joint. This catches a policy action reaching the wrong motor.
3. **Named contact routing:** give each vsim sensor a distinct synthetic force,
   or generate isolated named-foot contacts, and assert that the force appears
   in the matching canonical body/foot slot. This specifically guards the
   current sensor/link permutation bug.
4. **Task permutation invariance:** feed identical canonical joint/contact
   states through `mini_cheetah_ref` using two different native permutations;
   observations, reference targets, swing/stance masks, and rewards must match.
5. **Checkpoint compatibility:** layout v1 loads with the same observation and
   action meanings on MuJoCo and vsim; a changed/missing name or reordered
   unversioned layout fails with a useful error.

Run the engine-independent tests normally, then the real backend routing tests:

```bash
uv run python -m pytest tests/unit_tests/test_robot_layout.py \
  tests/unit_tests/test_legged_backend_contract.py -v
bash scripts/run_vsim_tests.sh
```

**Gate acceptance checks:**

- MuJoCo CPU, Warp, and vsim expose identical ordered mini-cheetah
  `dof_names`, robot-only `body_names`, feet indices, and action meanings.
- All one-hot/sentinel routing tests pass in both read and write directions.
- The same canonical task state produces identical task observations,
  references, contact masks, and rewards before stepping physics.
- Existing layout-v1 mini-cheetah checkpoints retain their policy-facing
  action/observation order.
- Rerun Gate 2 `step` after mapping is fixed. Only then use `drop` differences
  to tune contact materials/solver parameters.

**Implemented and checked 2026-07-28.** The canonical mini-cheetah interface is
`RF, LF, RH, LH` for DOFs and feet on all three backends; MuJoCo's `world` body
is no longer public. The engine-independent suite passes (`199 passed`) and the
licensed vsim suite passes (`62 passed`). MuJoCo Warp-specific routing,
liveness, reset, and physics checks pass (`52 passed`).

The 0.5 s contact-free Gate 2 `step` probe improved vsim/CPU joint-trajectory
RMS from the pre-fix `2.60e-2 rad` to `4.15e-5 rad`; CPU/Warp RMS is
`9.10e-8 rad`. This closes the ordering/actuation part of Gate 0. A 0.6 s
`drop` probe exposes identical DOF/foot labels across all outputs while the
remaining vsim/CPU differences are contact-rich (`2.36e-2 m` settle-height RMS,
`1.61e1 N` GRF-time RMS), so subsequent work can focus on material/contact
semantics rather than permutations.

#### Gate 1 — state/reset regressions

Run these before producing any new training evidence:

```bash
uv run python -m pytest \
  tests/unit_tests/test_task_state_liveness.py \
  tests/unit_tests/test_legged_backend_contract.py -v
bash scripts/run_vsim_tests.sh
```

Required result: CPU, Warp, and vsim state tensors remain live after stepping;
floating-base resets preserve the configured root height; the vsim suite is
green. A skipped Warp test does not close this gate on a GPU parity machine.

#### Gate 2 — policy-free physics probes

`scripts/mini_cheetah_fidelity.py` separates contact behavior from joint/PD
behavior using identical deterministic initial conditions:

- `drop`: floating base released from `z=0.5`, default pose held by PD. Compares
  base height, quaternion, impact timing, and foot ground-reaction forces.
- `step`: fixed base clear of the ground, then every joint receives a target
  step. Compares contact-free actuator, limb, and integration response.

Run all three engines for both probes:

```bash
mkdir -p logs/mc_fid

uv run scripts/mini_cheetah_fidelity.py run --probe drop \
  --backend mujoco --device cpu --out logs/mc_fid/drop_cpu.npz
uv run scripts/mini_cheetah_fidelity.py run --probe drop \
  --backend mujoco --device cuda:0 --out logs/mc_fid/drop_warp.npz
uv run --env-file .env.vsim scripts/mini_cheetah_fidelity.py run --probe drop \
  --backend vsim --device cuda:0 --out logs/mc_fid/drop_vsim.npz
uv run scripts/mini_cheetah_fidelity.py compare logs/mc_fid/drop_*.npz

uv run scripts/mini_cheetah_fidelity.py run --probe step \
  --backend mujoco --device cpu --out logs/mc_fid/step_cpu.npz
uv run scripts/mini_cheetah_fidelity.py run --probe step \
  --backend mujoco --device cuda:0 --out logs/mc_fid/step_warp.npz
uv run --env-file .env.vsim scripts/mini_cheetah_fidelity.py run --probe step \
  --backend vsim --device cuda:0 --out logs/mc_fid/step_vsim.npz
uv run scripts/mini_cheetah_fidelity.py compare logs/mc_fid/step_*.npz
```

Visualize the drop trajectories, impact timing, per-foot forces, and aggregate
divergence interactively:

```bash
uv run marimo edit notebooks/mini_cheetah_drop_fidelity.py
```

Expected result: CPU and Warp remain tightly matched because they use the same
MuJoCo model/solver. vsim may differ more in `drop` because its contact solver
is different, but `step` should remain close if torque mapping, joint ordering,
PD gains, armature, damping, and timestep agree. Any large CPU/Warp difference,
wrong initial base height, missing foot impulse, or large vsim `step` error is a
backend bug and blocks RL comparison.

#### Gate 3 — train and evaluate the 3×3 backend matrix

The campaign script trains one policy per backend with seed 7, then evaluates
every trained policy on every backend:

```bash
ITERS=250 SEED=7 GPU_ENVS=4096 CPU_ENVS=256 EVAL_ENVS=256 T_END=5.0 \
  bash scripts/run_mc_ref_benchmark.sh
```

CPU training is the long pole and runs last. The script creates:

| Artifact | Contents |
|---|---|
| `logs/mini_cheetah_ref_{cpu,warp,vsim}/.../vitals.jsonl` | Training curves, reward terms, losses, action statistics, throughput |
| `logs/mc_rl_eval/<train>__<eval>.npz` | 3×3 `reset_to_range` transfer matrix: aggregate reward, per-term reward, survival, episode length, base height/uprightness |
| `logs/mc_rl_sample/<train>__<eval>.npz` | 3×3 deterministic `reset_to_basic` samples with DOF trajectories |

Inspect partial or complete results with:

```bash
uv run marimo edit notebooks/mini_cheetah_ref.py
# Read-only:
uv run marimo run notebooks/mini_cheetah_ref.py
```

#### Acceptance criteria

- All three training runs complete without crashes, NaNs, dropped-constraint
  warnings, stale reward terms, or frozen state observations.
- Total reward and `reference_traj`/tracking terms improve on every backend;
  compare curve shape and per-term decomposition rather than requiring exact
  final-reward equality.
- CPU and Warp training distributions and transfer cells are close enough to
  be explained by batching/nondeterminism, not a systematic backend offset.
- A policy transferred between CPU and Warp retains comparable survival,
  posture, and reference tracking. vsim cross-transfer may be less exact due to
  contact-model differences, but must not collapse immediately.
- Deterministic DOF samples show CPU≈Warp. vsim divergence should be primarily
  contact-driven and consistent with the policy-free `drop`/`step` results.
- Record final metrics and plots in the Phase 4 PR before marking this check
  complete. Only post-reset-fix checkpoints count as evidence.

### IsaacGym removal

Once all checks pass:
1. Delete `gym/envs/base/isaac_gym_backend.py`
2. Remove `try/except` guards around isaacgym imports (they become unconditional `pass` or are deleted)
3. Remove isaacgym from `requirements.txt` / `pyproject.toml`
4. Remove `gym/sim` shim properties from `BaseTask` (already done in Phase 3)

---

## Python and environment setup

The project targets **Python >= 3.11** with **mujoco >= 3.6** and
**mujoco-warp >= 3.6**.  The default `uv` environment is the primary
development environment for all MuJoCo backends.

```bash
uv sync                           # creates .venv with Python 3.11+
uv run python -m pytest tests/    # run tests
uv run scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64
uv run scripts/train_mujoco.py --task pendulum --device cuda:0 --num_envs 4096
```

**IsaacGym legacy:** IsaacGym requires Python 3.8 and has its own venv.
The IsaacGym backend continues to work when run from that venv (`scripts/train.py`).
It will be removed in Phase 4.

## Notes and known gotchas

- **isaacgym import order:** IsaacGym must be imported before PyTorch.  All
  files that import isaacgym use `try/except ImportError` with the isaacgym
  import placed before `import torch`.
- **MuJoCo quaternion convention:** scalar-first `[qw, qx, qy, qz]` internally,
  converted to scalar-last `[qx, qy, qz, qw]` at the backend boundary.
  The task layer always sees scalar-last (matching IsaacGym convention).
- **`dof_state` view contract:** `dof_pos` and `dof_vel` must be views into
  `dof_state`, not copies.  Writing into `dof_pos[env_ids]` must be reflected
  in `dof_state` automatically.  Verified by `test_dof_state_view_consistent_*`.
- **Phase 3 TODO markers:** Shim code (backend's `gym`/`sim` properties,
  `register_dof_state()`, etc.) is annotated `# TODO Phase 3: remove`.

## Things to check

- [ ] friction domain-randomization
- [ ] mass domain-randomization
