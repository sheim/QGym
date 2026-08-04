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
ground.friction = [dynamic_friction, 0.005, 0.0001]
mjm = spec.compile()
mjm.geom_friction[:] = [dynamic_friction, 0.005, 0.0001]
```

MuJoCo's slots are sliding, torsional, and rolling friction. Only add the plane
when `cfg` has terrain config and `mesh_type == "plane"`, and assign the shared
coefficient to robot and ground geoms so imported robot defaults cannot
override a lower terrain value.

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

### Retirement scope decision — 2026-08-04

IsaacGym removal no longer depends on IsaacGym checkpoint portability,
IsaacGym throughput comparisons, cross-engine policy parity, rough terrain,
or projectiles. Those results remain below as historical engineering evidence,
not as removal gates. The supported engine families are MuJoCo (CPU and Warp
execution modes) and optional licensed VSim, initially on flat ground.

Domain randomization is intentionally deferred. Existing IsaacGym-specific
friction and mass randomization code and configuration may be removed during
retirement; a later feature will define one backend-neutral sampling contract
and implement its effects deliberately for both engine families.

The removal gates are now:

1. A deterministic, skip-free default suite for pure and MuJoCo CPU behavior,
   plus explicit passing Warp and VSim groups on capable machines.
2. Explicit registration and construction of every supported task; broken
   imports must fail rather than silently remove tasks from the registry.
3. No executable Python import or runtime branch for IsaacGym.
4. A clean frozen uv environment, package build, lint pass, task smoke tests,
   and short supported-runner checks.

### Historical validation checklist (superseded as removal gates)

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

**Post-fix campaign status (updated 2026-07-28):** Gates 1 and 2 pass. The
contract suites pass, CPU and Warp are numerically equivalent, and the
remaining Vsim physics differences are concentrated in contact and satisfy the
coarse sim-to-sim bounds below. Gate 3 now passes training convergence and four
of the six cross-engine directions, but remains open because the final
Vsim-trained checkpoint collapses on MuJoCo CPU and Warp.

**Contact-discrepancy investigation (updated 2026-07-28):** the probe originally
inherited one implicit initialization step from `TaskSkeleton.reset()`, so it
did not start from the requested state. The probe now restores the deterministic
state immediately before recording. CPU, Warp, and vsim agree to numerical
precision before ground contact (vsim/CPU maximum error before `t=0.20 s`:
`<4e-7 m` base height and `<1.1e-8 rad` joint position).

The contact tensor routing and frame are now fixed. Vsim sensors are resolved
through their native link indices into canonical body order. Although the
sensor command requests the environment frame, the returned vector components
are attached to the rotating sensor/link frame; the backend now rotates them
into world axes. A settled mini-cheetah reports approximately its `81.3 N`
weight along world `+z`, and the licensed regression suite checks this.

Plane tasks now explicitly assign the configured terrain material to both the
Vsim robot and plane. MuJoCo's terrain mapping was also corrected: its three
friction slots mean sliding/torsional/rolling, not static/dynamic/rolling, so
the shared dynamic coefficient feeds MuJoCo's single Coulomb sliding
coefficient. The coefficient must be assigned to every robot and terrain geom:
MuJoCo combines equal-priority contacts using the larger coefficient, so a
robot geom left at its imported `1.0` default silently defeats terrain values
below one. The current task has static = dynamic = `1.0`; only Vsim can
represent them separately. Nonzero `terrain.restitution` still needs a
deliberate MuJoCo mapping because MuJoCo rebound is primarily controlled by
`solref`/`solimp`, not a rigid-material restitution field.

The remaining impact difference is primarily contact formulation. MuJoCo uses
a compliant constraint (`solref=[0.02, 1]`, default `solimp`) and Newton with
up to 100 iterations. Vsim remains on its rigid/LCP formulation, now with 16
solver iterations for Mini Cheetah. Its initial normal impulse is still sharper
even though the integrated impulse, settled pose, and task contact decisions
are reasonably close. See the direct contact probes below for measured bounds.

**Gate 3 controlled campaign (2026-07-28, seed 7):** the original benchmark
used 256 CPU environments but 4096 GPU environments with a batch size of 4096.
`OnPolicyRunner` derives `num_steps_per_env = batch_size // num_envs`, so that
comparison collected 16-step CPU rollouts but only one step per Warp/Vsim
environment. The resulting weak GPU policies were an inequivalent GAE/PPO
experiment, not backend-parity evidence. The benchmark now enforces 256
environments, batch size 4096, and therefore 16 temporal steps per update on
all three backends.

Two evaluation issues were also removed. Inference previously called
`get_noisy_obs()`, making supposedly deterministic actions depend on each
device's RNG stream. It now uses clean observations. `reset_to_basic` also
retained random commands and gait phases; the evaluator now fixes the command
to `[1, 0, 0]`, phase to zero, and refreshes all derived state after the exact
system reset.

With equal 16-step rollouts, all three policies learn the reference task:

| Backend | Total reward | `reference_traj` | Episode time | Steps/s |
|---|---:|---:|---:|---:|
| CPU | 7.976 | 0.885 | 4.65 s | 5,294 |
| Warp | 8.055 | 0.847 | 4.96 s | 13,599 |
| Vsim | 7.739 | 1.025 | 4.81 s | 36,615 |

Final-checkpoint transfer at iteration 250 (`reset_to_range`; each cell is mean
reward / survival, train backend down and evaluation backend across):

| Train \ Eval | CPU | Warp | Vsim |
|---|---:|---:|---:|
| CPU | 8.144 / 90.6% | 7.852 / 88.7% | 7.192 / 72.7% |
| Warp | 8.226 / 99.6% | 8.013 / 98.4% | 7.839 / 97.3% |
| Vsim | 6.280 / 4.7% | 6.010 / 3.5% | 7.823 / 69.5% |

The failure is late and asymmetric rather than a basic inability to transfer.
At iteration 100, the Vsim policy scores `7.882 / 90.6%` on CPU, `7.672 /
89.8%` on Warp, and `7.631 / 86.3%` on Vsim. By iteration 150 its CPU survival
has fallen to `6.2%` in the checkpoint screen even though the in-domain
training terms continue to improve. The CPU policy is not yet converged at
iteration 100 (`18.0%` native survival), so reducing the common iteration
budget would only move the failure to another row. This is evidence of late
Vsim-specific policy exploitation and requires a training robustness measure
or a predeclared validation/early-stopping rule; selecting checkpoint 100 after
looking at target-backend performance would not close the gate honestly.

The deterministic runtime check passes decisively. For all three final
policies, full five-second CPU/Warp DOF-trajectory RMS is at most
`1.92e-6 rad`. CPU/Vsim divergence begins after `0.04–0.06 s` from the standing
contact state and is consistent with the contact-model difference rather than
an ordering, observation, command, or device-RNG mismatch.

Controlled evidence is under `logs/mini_cheetah_ref_cpu`,
`logs/mini_cheetah_ref_{warp,vsim}_h16`, `logs/mc_rl_eval_h16`,
`logs/mc_rl_eval_h16_i100`, and `logs/mc_rl_sample_h16`. Earlier campaign data
is retained under `logs/archive` but is not parity evidence.

**Large-batch follow-up (2026-07-28):** increasing the PPO batch to `2**15 =
32768` while preserving the 16-step horizon required 2048 environments. This
collects eight times as many transitions per iteration as the controlled
baseline. Warp and Vsim were trained for 250 iterations; the corresponding CPU
run was stopped after two iterations because each update took approximately
`20.7 s` and the projected run time was 86 minutes. The matrix below therefore
reuses the controlled CPU policy and changes only the two GPU-trained rows.

| Backend | Total reward | `reference_traj` | Episode time | Steps/s |
|---|---:|---:|---:|---:|
| Warp, batch 32768 | 8.230 | 0.766 | 4.63 s | 55,632 |
| Vsim, batch 32768 | 8.893 | 1.012 | 4.80 s | 184,910 |

Final-checkpoint transfer:

| Train \ Eval | CPU | Warp | Vsim |
|---|---:|---:|---:|
| CPU, baseline | 8.144 / 90.6% | 7.852 / 88.7% | 7.192 / 72.7% |
| Warp, batch 32768 | 9.243 / 98.0% | 8.966 / 95.7% | 8.479 / 92.2% |
| Vsim, batch 32768 | 7.863 / 8.6% | 7.684 / 9.4% | 9.805 / 96.9% |

The larger batch improves Vsim's native final survival from `69.5%` to `96.9%`
and delays specialization, but does not fix Vsim-to-MuJoCo transfer:

| Vsim checkpoint | Batch 4096: CPU / Vsim survival | Batch 32768: CPU / Vsim survival |
|---:|---:|---:|
| 50 | 92.2% / 28.1% | 98.4% / 100.0% |
| 100 | 90.6% / 84.4% | 82.8% / 98.4% |
| 150 | 6.2% / 60.9% | 57.8% / 95.3% |
| 200 | 10.9% / 93.8% | 20.3% / 100.0% |
| 250 | 4.7% / 69.5% | 8.6% / 96.9% |

Thus batch variance/sample diversity is part of the problem—the transfer
window becomes much wider—but the converged Vsim objective still rewards a
backend-specific strategy. Increasing the batch alone is not a gate-closing
fix. One rare Warp `opt.ccd_iterations ... needs to be increased` warning also
appeared during the 2048-environment training population. Large-batch artifacts
are under `logs/mini_cheetah_ref_{warp,vsim}` runs dated `Jul28_22-25-17_` and
`Jul28_22-29-16_`, plus `logs/mc_rl_eval_b32768`.

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
- a layout version that identifies the canonical interface in configuration
  and tests.

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
- `TaskSkeleton` and the learning runners remain unaware of robot layouts:
  canonical ordering makes their positional tensor interface safe without
  coupling checkpoint I/O to asset metadata.
- Pre-Gate-0 CPU/Warp mini-cheetah checkpoints use layout v1. Pre-Gate-0 vsim
  checkpoints used reverse native order and must be regenerated or migrated;
  checkpoint loading does not attempt to infer or repair that ordering.

##### Implementation checklist

- [x] Verify the historical CPU/Warp/checkpoint action order, then freeze it as
      mini-cheetah layout v1.
- [x] Add the canonical layout/schema and exact-name validation.
- [x] Add native↔canonical DOF/body mappings to MuJoCo CPU and Warp.
- [x] Add native↔canonical DOF/body/motor/sensor mappings to vsim.
- [x] Make all public backend metadata and tensors canonical and robot-only.
- [x] Refactor mini-cheetah semantic rewards and actuator selection to named
      groups.
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
is no longer public. The engine-independent suite passes (`204 passed`) and the
licensed vsim suite passes (`62 passed`). MuJoCo Warp-specific routing,
liveness, reset, and physics checks pass (`52 passed`).

The 0.5 s contact-free Gate 2 `step` probe improved vsim/CPU joint-trajectory
RMS from the pre-fix `2.60e-2 rad` to `4.15e-5 rad`; CPU/Warp RMS is
`9.10e-8 rad`. This closes the ordering/actuation part of Gate 0. The fresh
3.0 s `drop` probe exposes identical DOF/foot labels across all outputs.
CPU/Warp remain close (`1.36e-5 m` base-height trajectory RMS). With repaired
source inertias and 16 Vsim LCP iterations, Vsim contacts 10 ms earlier at the
100 Hz task boundary, settles at `0.2579 m` versus `0.2546 m`, and differs from
CPU by `1.23e-2 m` in base-height trajectory RMS, `6.42e-2 rad` in joint
position RMS, and `5.28e-1 rad/s` in joint velocity RMS. The remaining drop
differences are contact-rich. Ordering, routing, and non-contact inertia are no
longer suspects; the direct probes below quantify the contact-formulation
residual before the next RL campaign.

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
  base height, quaternion, impact timing, foot ground-reaction forces, and
  joint position/velocity phase portraits.
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

Visualize the drop trajectories, impact timing, per-foot forces, joint phase
portraits, and aggregate divergence interactively:

```bash
uv run marimo edit notebooks/mini_cheetah_drop_fidelity.py
```

Expected result: CPU and Warp remain tightly matched because they use the same
MuJoCo model/solver. vsim may differ more in `drop` because its contact solver
is different, but `step` should remain close if torque mapping, joint ordering,
PD gains, armature, damping, and timestep agree. Any large CPU/Warp difference,
wrong initial base height, missing foot impulse, or large vsim `step` error is a
backend bug and blocks RL comparison.

##### Direct non-contact diagnostic probes

Four backend-level probes bypass task PD control and run without gravity or
contact. They use one environment per canonical joint (plus the default pose
for kinematics):

- `torque`: one 2 ms, one-hot `1 Nm` torque step; records the full
  torque-to-joint-acceleration response matrix.
- `damping`: one-hot `1 rad/s` initial joint velocity, then zero applied
  torque for `0.25 s`; records passive joint decay.
- `kinematics`: default pose plus a `0.2 rad` offset on each joint in turn;
  compares every canonical body transform relative to the base.
- `reaction`: floating base, constant one-hot `1 Nm` torque for `0.10 s`;
  records joint motion and the equal-and-opposite base response.

Rerun all four on this machine:

```bash
mkdir -p logs/mc_fid

for PROBE in torque damping kinematics reaction; do
  uv run scripts/mini_cheetah_fidelity.py run --probe "$PROBE" \
    --backend mujoco --device cpu --out "logs/mc_fid/${PROBE}_cpu.npz"
  uv run scripts/mini_cheetah_fidelity.py run --probe "$PROBE" \
    --backend mujoco --device cuda:0 --out "logs/mc_fid/${PROBE}_warp.npz"
  uv run --env-file .env.vsim scripts/mini_cheetah_fidelity.py run \
    --probe "$PROBE" --backend vsim --device cuda:0 \
    --out "logs/mc_fid/${PROBE}_vsim.npz"
  uv run scripts/mini_cheetah_fidelity.py compare \
    "logs/mc_fid/${PROBE}_cpu.npz" \
    "logs/mc_fid/${PROBE}_warp.npz" \
    "logs/mc_fid/${PROBE}_vsim.npz"
done
```

**Result (2026-07-28):**

| Probe | MuJoCo Warp vs CPU | vsim vs CPU | Interpretation |
|---|---:|---:|---|
| One-step torque | `5.92e-6 rad/s²` acceleration RMS | `1.39e-2 rad/s²` (`0.04%`) acceleration RMS | Torque/inertia response now agrees closely |
| Passive damping | `2.90e-8 rad/s` velocity-trajectory RMS | `6.67e-5 rad/s` velocity-trajectory RMS | Passive joint behavior agrees closely |
| Forward kinematics | `8.03e-9 m`, `3.61e-8 rad` RMS | `1.03e-7 m`, `1.43e-7 rad` RMS | Joint axes, frames, and body transforms match |
| Floating reaction | `1.27e-7 rad/s` base-angular-velocity RMS | `3.61e-3 rad/s` base-angular-velocity RMS; `1.29e-3 rad/s` joint-velocity RMS | Small cross-integrator residual remains |

These are the final reruns after repairing every nonphysical source inertia in
both `mini_cheetah_simple.urdf` and `mini_cheetah_rotor.urdf`:

- The base `iyy=0.362030 kg·m²` was a decimal-place typo. It is now
  `0.036203 kg·m²`, matching the upstream MIT dynamics model's principal
  moments `[0.011253, 0.036203, 0.042673] kg·m²`.
- The upstream CAD thigh tensor is slightly nonphysical after decimal
  rounding. Its principal moments were projected to the nearest strictly
  physical tensor in Frobenius norm while preserving its principal axes; the
  corrected Cartesian tensor is explicit in the URDF rather than left to each
  importer.
- Each `0.01 kg` foot had a zero tensor. It is now a uniform sphere matching
  the existing `0.0202 m` collision sphere, with its COM at the sphere center.
- `test_mini_cheetah_inertias.py` parses both source assets and rejects
  non-positive moments or a violated principal-moment triangle inequality.

Against the pre-repair result, Vsim's one-step acceleration RMS shrank about
`120×`; base angular reaction shrank about `19×`; joint-velocity reaction
shrunk about `88×`; and passive-decay velocity RMS shrank about `30×`.
Kinematics stayed at numerical precision, as expected. This closes the
non-contact inertia/import discrepancy; the remaining differences are small
enough to move the investigation to contact.

##### Direct contact diagnostic probes

`impact` records the first second of a `z=0.5 m` drop at the 500 Hz physics
rate, including world-frame force vectors, foot sphere clearance, rebound, and
settled height. `slide` first settles the robot for one second, injects
`1 m/s` into the floating base, and measures whole-system COM deceleration
rather than base-only motion (the latter is contaminated by leg motion).

The production Vsim run uses rigid/LCP contact, 16 solver iterations, zero
restitution, the engine's default contact offset, and an explicit material
carrying the task's nominal friction values. Run all three contact probes on
each backend:

```bash
mkdir -p logs/mc_fid

for PROBE in impact slide drop; do
  uv run scripts/mini_cheetah_fidelity.py run --probe "$PROBE" \
    --backend mujoco --device cpu --label mujoco-cpu \
    --out "logs/mc_fid/${PROBE}_cpu.npz"
  uv run scripts/mini_cheetah_fidelity.py run --probe "$PROBE" \
    --backend mujoco --device cuda:0 --label mujoco-warp \
    --out "logs/mc_fid/${PROBE}_warp.npz"
  uv run --env-file .env.vsim scripts/mini_cheetah_fidelity.py run \
    --probe "$PROBE" --backend vsim --device cuda:0 --label vsim-lcp-16 \
    --out "logs/mc_fid/${PROBE}_vsim.npz"
  uv run scripts/mini_cheetah_fidelity.py compare \
    "logs/mc_fid/${PROBE}_cpu.npz" \
    "logs/mc_fid/${PROBE}_warp.npz" \
    "logs/mc_fid/${PROBE}_vsim.npz"
done
```

**Result (2026-07-28):**

| Impact model | First force | Peak total Fz | 100 ms impulse | COM rebound | Settled base z |
|---|---:|---:|---:|---:|---:|
| MuJoCo CPU | `0.202 s` | `244.2 N` | `23.247 N·s` | `0.633 m/s` | `0.2485 m` |
| MuJoCo Warp | `0.202 s` | `244.2 N` | `23.247 N·s` | `0.633 m/s` | `0.2485 m` |
| Vsim rigid/LCP, 8 iterations (diagnostic) | `0.200 s` | `1303.4 N` | `25.955 N·s` | `0.506 m/s` | `0.2495 m` |
| Vsim rigid/LCP, 16 iterations (production) | `0.200 s` | `845.5 N` | `25.710 N·s` | `0.516 m/s` | `0.2528 m` |

| Slide model | Half-speed time | First stop | COM stopping distance | Mean foot slip |
|---|---:|---:|---:|---:|
| MuJoCo CPU | `0.184 s` | `0.392 s` | `0.1778 m` | `0.0545 m` |
| MuJoCo Warp | `0.184 s` | `0.392 s` | `0.1776 m` | `0.0544 m` |
| Vsim rigid/LCP, 8 iterations (diagnostic) | `0.154 s` | `0.344 s` | `0.1484 m` | `0.0114 m` |
| Vsim rigid/LCP, 16 iterations (production) | `0.150 s` | `0.336 s` | `0.1451 m` | `0.0018 m` |

The 100 Hz task-level `drop` probe applies the exact reward and termination
thresholds:

| Backend | >50 N foot onset | Foot-mask duty | Foot-mask F1 vs CPU | Base/thigh >1 N samples |
|---|---:|---:|---:|---:|
| MuJoCo CPU | `0.210 s` | `3.6%` | `100%` | `0%` |
| MuJoCo Warp | `0.210 s` | `3.6%` | `100%` | `0%` |
| Vsim rigid/LCP, 8 iterations | `0.200 s` | `2.8%` | `77.8%` | `0%` |
| Vsim rigid/LCP, 16 iterations | `0.200 s` | `3.2%` | `83.9%` | `0%` |

Interpretation:

- The `2 ms` physics-rate onset difference is one sample. Vsim reports
  the force generated during a step alongside post-step link transforms, so
  its smaller onset penetration is a detection/readback detail, not a
  ballistic-flight discrepancy.
- Restitution values `0.1–0.5` did not increase whole-system rebound and
  changed the impact peak non-monotonically. Keep the task's physically neutral
  `0.0`; tuning restitution here would only fit this articulated impact.
- Rigid/LCP contact offsets from `0` through the engine default produced the
  same onset and trajectory to measurement precision. Keep the default offset
  for collision-candidate robustness.
- Moving from 8 to 16 iterations lowers the impulse-concentration peak by
  `35%`, removes the zero-force physics substep, and improves task foot-mask F1
  from `77.8%` to `83.9%`. Higher counts were non-monotonic. Sixteen is a
  modest general convergence setting and is now the Mini Cheetah default.
- Vsim's 100 ms impulse is `10.6%` above MuJoCo and its settled base height is
  within `4.3 mm`; its COM rebound is `18.5%` lower. The still-larger
  instantaneous peak is the expected difference between MuJoCo's compliant
  constraint and Vsim's rigid impulse concentration, not a routing or inertia
  error.
- Friction sweeps must compare equal material semantics. Once MuJoCo assigns
  the configured coefficient to robot and ground geoms, lower equal
  coefficients move both engines in the same direction. A common `0.25`
  happens to give the closest synthetic slide, but there is no hardware basis
  for replacing the task's nominal `1.0`; doing so would be overfitting.
  With `1.0`, Vsim stops `14%` earlier and `18%` shorter. Its much lower foot
  slip reflects the engines' different rigid friction constraints and
  articulated-foot motion and is recorded for later hardware calibration.
- The ordinary upright drop correctly produces no base/thigh termination on
  any backend. Separate upside-down base-impact tests pass on CPU, Warp, and
  Vsim, confirming the task's `>1 N` termination path reaches the correct body.

For sim-to-sim validation, do not require force-sample lockstep or an identical
instantaneous peak. Require instead:

- ballistic position/joint agreement until contact and onset within two
  500 Hz samples;
- correct canonical body routing, world-frame force direction, and settled
  vertical force near robot weight;
- 100 ms impulse within `15%`, settled height within `5 mm`, and no
  missing/alternating task-rate contacts;
- slide stopping time and COM distance within `20%`, with foot-slip and rebound
  residuals explicitly recorded rather than hidden by backend-specific
  material coefficients;
- foot-mask F1 at least `0.8` at the task's `50 N` threshold and matching
  termination decisions at `1 N`.

The rigid/LCP production configuration meets these deliberately coarse
sim-to-sim gates. It is not a claim that the contact waveforms are identical:
rebound, foot slip, and the instantaneous impact peak remain model-formulation
differences to validate or randomize against hardware before deployment.

#### Gate 3 — train and evaluate the 3×3 backend matrix

The campaign script trains one policy per backend with seed 7, then evaluates
every trained policy on every backend. PPO rollout geometry is part of the
controlled input: with the current batch size, 256 environments gives 16
consecutive steps per environment on every backend.

```bash
ITERS=250 SEED=7 TRAIN_ENVS=256 BATCH_SIZE=4096 EVAL_ENVS=256 T_END=5.0 \
  bash scripts/run_mc_ref_benchmark.sh
```

Do not increase only the GPU environment count for this gate: 4096 environments
with batch size 4096 changes the rollout horizon from 16 steps to one. Run
scaling/throughput experiments separately. CPU training is the long pole and
runs last. The script creates:

| Artifact | Contents |
|---|---|
| `logs/mini_cheetah_ref_{cpu,warp,vsim}/.../vitals.jsonl` | Training curves, reward terms, losses, action statistics, throughput |
| `logs/mc_rl_eval/<train>__<eval>.npz` | 3×3 `reset_to_range` transfer matrix: aggregate reward, per-term reward, survival, episode length, base height/uprightness |
| `logs/mc_rl_sample/<train>__<eval>.npz` | 3×3 deterministic `reset_to_basic` samples with DOF trajectories |

Inspect partial or complete results with:

```bash
uv run marimo edit notebooks/mini_cheetah_gate3.py  # controlled Gate 3 report
uv run marimo edit notebooks/mini_cheetah_ref.py
# Read-only:
uv run marimo run notebooks/mini_cheetah_gate3.py
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

**Current result:** all training, CPU/Warp equivalence, CPU/Warp-to-Vsim
transfer, and deterministic-trajectory checks pass. The final Vsim-to-MuJoCo
transfer cells fail because of late-training model exploitation, so Gate 3 and
overall mini-cheetah parity remain open.

#### Native-backend policy tuning — hardware-oriented scorecard

The next phase intentionally stops treating one policy as the optimum for both
contact formulations. It will tune two explicit configurations:

- `mini_cheetah_ref_mujoco_config.py`, trained and selected on MuJoCo Warp;
- `mini_cheetah_ref_vsim_config.py`, trained and selected on VSim.

Both must use the same evaluation protocol. Backend-specific training rewards
may differ, but model selection must not use reward totals: those totals change
when reward weights change and can hide noisy or actuator-saturating gaits.

`scripts/eval_policy.py --command_profile hardware` now applies a balanced,
fixed nine-case command suite: stand, two forward speeds, backward, left/right
strafe, left/right yaw, and combined translation/yaw. The dedicated wrapper
runs both a nominal basic-state evaluation and a randomized-initial-state
robustness evaluation:

```bash
# MuJoCo-native policy (use cuda:0 for the Warp runtime used during training)
bash scripts/eval_mc_ref_hardware.sh \
  mujoco cuda:0 PATH/TO/MUJOCO/model_N.pt mujoco-candidate

# VSim-native policy
bash scripts/eval_mc_ref_hardware.sh \
  vsim cuda:0 PATH/TO/VSIM/model_N.pt vsim-candidate
```

The default is 288 environments (32 per command), five seconds, seed zero, and
a 0.5-second settling exclusion. Each run writes an NPZ with per-environment
values and a human-readable `*.summary.json` with overall and per-command
mean/median/p90 summaries.

The scorecard deliberately keeps physical metrics separate instead of
combining them into one arbitrary quality score:

| Group | Primary metrics |
|---|---|
| Robustness | survival, episode duration, unsafe-body contact fraction, phase/direction-staggered velocity-impulse failure |
| Command tracking | forward/lateral velocity RMSE, yaw-rate RMSE |
| Body stability | tilt RMS, base-height mean/std/range/drift, vertical velocity/acceleration, roll/pitch rate |
| Actuation | joint velocity/acceleration, position-target velocity/acceleration, torque RMS/rate/utilization, absolute mechanical power, minimum joint-limit margin, 500 Hz torque/velocity PSD and high-frequency power |
| Gait/contact | gait-reference joint RMSE, touchdown RPD/trot classification, cycle consistency, body-weight-normalized per-leg GRF balance, foot slip, phase/contact agreement |

The gait and disturbance metrics follow the evaluation ideas in
[Zhang et al. (2024)](https://arxiv.org/abs/2402.08662):

- analogous to Fig. 3, average vertical GRF is reported independently for
  RF/LF/RH/LH and normalized by nominal robot weight; ideal balanced use is
  centered at `0.25` per leg;
- touchdown relative phase difference uses RF as reference. Ideal trot is
  `(LF, RH, LH) = (π, π, 0)`, with circular cycle-to-cycle variation and the
  paper's nearest-symmetric-gait/transition classification;
- analogous to Table I, planar velocity impulses are spread over 36 directions
  and 50 control steps spanning `0.5 s`, so the full default is 1,800 trials
  per magnitude after five seconds at a `3 m/s` forward command.

Run the full native disturbance sweep with:

```bash
bash scripts/eval_mc_ref_disturbance.sh \
  mujoco cuda:0 PATH/TO/MUJOCO/model_N.pt mujoco-candidate
bash scripts/eval_mc_ref_disturbance.sh \
  vsim cuda:0 PATH/TO/VSIM/model_N.pt vsim-candidate
```

Torque and joint velocity are sampled after every 500 Hz PD/physics substep,
not aliased from the 100 Hz policy rate. The artifact contains the mean
per-joint PSD, spectral centroid, dominant frequency, gait-band fraction, and
power above the configurable `10 Hz` diagnostic cutoff, including the worst
single joint. The cutoff is not a hardware safety certification; finalize it
from the motor, gearbox, driver, and low-level controller bandwidth.

Provisional selection rules:

- require complete nominal survival and at least 99% robustness survival before
  comparing smoothness;
- reject forbidden contacts, joint-limit crossings, or repeated torque
  saturation;
- compare command RMSE per command case, not just the population average;
- among policies with comparable tracking, prefer lower base vertical
  acceleration, target acceleration, torque rate/utilization, mechanical power,
  and stance-foot slip;
- treat peak contact force and binary phase-contact scores as diagnostic
  within a backend. Their absolute values depend strongly on the contact
  formulation and are not suitable for declaring MuJoCo or VSim “better.”

Absolute smoothness limits should be finalized from the real motor/controller
bandwidth, joint limits, and hardware logs. Until those exist, preserve the
full Pareto table and do not trade a large increase in actuator or impact
metrics for a small reward improvement.

**Initial native baseline (2026-07-28):** the final batch-32768 Warp and VSim
policies were evaluated with 288 environments per run. Both achieved 100%
nominal and randomized-start survival on this moderate command suite, so
survival alone cannot select between them.

| Nominal metric | MuJoCo Warp policy on Warp | VSim policy on VSim |
|---|---:|---:|
| Forward / lateral RMSE | `0.184 / 0.142 m/s` | `0.148 / 0.082 m/s` |
| Yaw-rate RMSE | `0.277 rad/s` | `0.364 rad/s` |
| Base tilt RMS | `3.65 deg` | `4.06 deg` |
| Vertical acceleration RMS | `7.36 m/s²` | `12.49 m/s²` |
| Gait-reference joint RMSE | `0.256 rad` | `0.213 rad` |
| Policy-target acceleration RMS | `5896 rad/s²` | `4429 rad/s²` |
| Torque utilization RMS / saturation samples | `0.211 / 0.169%` | `0.187 / 0.227%` |
| Absolute mechanical power | `134.5 W` | `128.2 W` |
| Actual joint-limit margin, p10 | `14.6%` | `9.3%` |
| Policy-target joint-limit margin, p10 | `-2.1%` | `0.6%` |
| Unsafe-body contact fraction | `0%` | `0%` |
| Foot slip / peak force (diagnostic) | `0.561 m/s / 143 N` | `0.195 m/s / 668 N` |

The expanded gait-quality baseline adds:

| Nominal metric | MuJoCo Warp policy on Warp | VSim policy on VSim |
|---|---:|---:|
| Base-height std / range | `11.8 / 46.6 mm` | `25.3 / 97.4 mm` |
| Torque power above 10 Hz | `72.9%` | `55.8%` |
| Joint-velocity power above 10 Hz | `51.1%` | `38.4%` |
| Moving trials classified trot | `91.8%` | `68.4%` |
| Trot RPD error / cycle variation | `0.719 / 0.800 rad` | `0.849 / 0.279 rad` |
| Overall per-leg GRF CV | `0.178` | `0.320` |
| 1 m/s per-leg GRF (RF/LF/RH/LH), body weight | `0.252 / 0.329 / 0.166 / 0.246` | `0.378 / 0.074 / 0.040 / 0.384` |

Both PSDs peak near the commanded `2.5 Hz` gait frequency, but contain large
higher-harmonic/contact energy. VSim's lower spectral ratios do not make its
gait preferable: at `1 m/s` it relies primarily on the RF/LH diagonal pair,
has only `20%` complete four-foot RF cycles, is not classified as trot, and
has over twice MuJoCo's base-height variation.

A preliminary Table-I-style screen used 360 trials per magnitude: all 36
directions at ten times uniformly spread over the full `0.5 s` phase window.
Use the full 1,800-trial wrapper for final config selection. The table reports
post-impulse failure conditional on the environment surviving until its
scheduled impulse. MuJoCo had `0%` pre-impulse failures; VSim had `8.3%`
pre-impulse failures during the five-second settling period, which is reported
separately instead of being attributed to the impulse.

| Velocity impulse | MuJoCo post-impulse failure | VSim post-impulse failure |
|---:|---:|---:|
| `1.5 m/s` | `28.1%` | `34.5%` |
| `2.0 m/s` | `40.0%` | `56.4%` |
| `2.5 m/s` | `53.6%` | `70.3%` |
| `3.0 m/s` | `64.2%` | `79.1%` |
| `3.5 m/s` | `71.7%` | `83.9%` |

The first tuning targets are therefore concrete:

- add enough target/joint-limit pressure that commanded positions retain a
  hardware margin rather than relying on the PD plant to remain in bounds;
- reduce occasional torque saturation and high target acceleration without
  giving up command tracking;
- reduce VSim vertical base acceleration and improve yaw tracking;
- improve MuJoCo lateral tracking and target smoothness.

The large foot-force and contact-duty differences are consistent with the
known compliant-versus-rigid contact formulations. They remain useful for
within-backend candidate selection, but are not cross-backend ranking metrics.
Baseline artifacts are under `logs/mc_ref_hardware/baseline_b32768/`.
The plots and paper-style tables are in
`notebooks/mini_cheetah_tuning.py`; disturbance artifacts are under
`logs/mc_ref_disturbance/baseline_b32768_360/`.

#### VSim deployment tuning

Policy tuning now treats VSim as a standalone deployment target; MuJoCo is not
part of the VSim checkpoint-selection loop. The first candidate is the
separate `mini_cheetah_ref_vsim` task in
`gym/envs/mini_cheetah/mini_cheetah_ref_vsim_config.py`.

The first tuning hypothesis deliberately leaves VSim physics, the reference
trajectory, PD gains, actor architecture, and 2.5 Hz gait frequency unchanged.
It changes:

- the command distribution to the deployment scorecard envelope
  (`-0.5..1.5 m/s` forward, `±0.4 m/s` lateral, `±0.75 rad/s` yaw), with 60%
  explicitly axis-aligned commands so pure strafe and yaw are trained rather
  than left to extrapolation from almost-always-combined random commands;
- PPO collection to `2**15` samples from 256 environments, giving 128 policy
  steps or 3.2 reference gait cycles per rollout;
- stronger reference/contact rewards to prevent an RF/LH-only gait;
- moderate vertical-motion, action-rate/curvature, joint-speed, torque-limit,
  soft position-target-limit, shank-contact, and stand-still terms;
- lower initial action noise and entropy pressure;
- a meaningful termination penalty.

The first 200-iteration screen also exposed and fixed a task reward bug:
Mini Cheetah pre-squared yaw error before passing it to `_sqrdexp`, which
squared it again. The resulting fourth-power error supplied almost no yaw
tracking gradient near the target. New training uses the intended squared
exponential error. The VSim tuning config also uses a `1.0 rad/s` error scale
instead of the inherited `2.5 rad/s` scale, which made the scorecard's
`±0.75 rad/s` yaw commands too weakly distinguishable from zero yaw.
The retained yaw tracking weight is `4.0`. A bounded iteration-400-to-500
continuation at `10.0` did not solve turning and degraded the more important
deployment metrics, so a larger scalar weight is not the next tuning lever.

Legacy per-environment friction and mass randomization are explicitly disabled
in this config because VSim currently uses a shared material and articulation
definition. Add VSim-native domain randomization as a later robustness phase;
do not assume the inherited flags provide it.

Train the first full candidate with:

```bash
uv run --env-file .env.vsim scripts/train_mujoco.py \
  --task mini_cheetah_ref_vsim \
  --backend vsim \
  --device cuda:0 \
  --headless \
  --disable_wandb
```

Resume a tuning run for another 100 iterations with:

```bash
uv run --env-file .env.vsim scripts/train_mujoco.py \
  --task mini_cheetah_ref_vsim \
  --backend vsim \
  --device cuda:0 \
  --max_iterations 100 \
  --resume \
  --load_run RUN_DIRECTORY \
  --checkpoint 100 \
  --headless \
  --disable_wandb
```

On resume, `--max_iterations` is the number of additional iterations. The
continued run is written to a new timestamped directory while loading model
and optimizer state from the selected source run.

Evaluate checkpoints only on VSim:

```bash
TASK=mini_cheetah_ref_vsim \
OUT_DIR=logs/mc_ref_hardware/vsim_tune \
bash scripts/eval_mc_ref_hardware.sh \
  vsim cuda:0 PATH/TO/model_N.pt vsim-tune-N
```

For the first sweep, evaluate at least checkpoints 100, 200, 300, 400, and
500. Select on the VSim scorecard Pareto front: survival and forbidden contact
first; then 1 m/s trot classification, complete-cycle fraction, per-leg GRF
balance, base-height steadiness, command RMSE, target joint-limit margin, and
per-joint torque/velocity spectra. Reward total is not a selection metric.

The initial curriculum screen produced the following native-VSim nominal
scorecard. Values aggregate the stand, forward/backward, strafe, yaw, and
combined command cells:

| Checkpoint | Survival | vx / vy / yaw RMSE | Height std | Torque / velocity >10 Hz | Trot | GRF CV |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 88.9% | 0.210 / 0.150 / 0.275 | 9.8 mm | 10.6% / 22.6% | 63.1% | 0.369 |
| 200 | 89.2% | 0.116 / 0.103 / 0.303 | 10.3 mm | 10.5% / 19.6% | 82.0% | 0.283 |
| 300 | 100% | 0.103 / 0.096 / 0.273 | 6.4 mm | 16.9% / 24.8% | 69.5% | 0.220 |
| **400** | **100%** | **0.089 / 0.102 / 0.254** | **6.9 mm** | **20.6% / 24.5%** | **97.7%** | **0.133** |
| 500, yaw weight 10 | 99.3% | 0.113 / 0.100 / 0.228 | 7.9 mm | 23.5% / 25.4% | 75.4% | 0.225 |

Checkpoint 400 is the current Pareto candidate. Its robust scorecard also
reached 100% survival, 99.6% trot classification, 7.1 mm height variation,
and 0.138 GRF CV. At 1 m/s it reached 100% survival and trot classification
with 0.077 m/s forward RMSE, 9.2 mm height variation, 14.9% torque and 18.6%
joint-velocity energy above 10 Hz, and 0.082 GRF CV. Pure yaw remained the
clear failure: `0.716 / 0.759 rad/s` left/right RMSE at checkpoint 400. The
higher-weight continuation only changed these to `0.606 / 0.749 rad/s`, while
breaking trot classification in both pure-yaw cells.

These are curriculum checkpoints, not a clean ablation: the yaw reward fix,
axis-aligned command sampler, and final reward settings were introduced during
the sweep. Use
`logs/mini_cheetah_ref_vsim_tune/Jul29_10-32-55_/model_400.pt` only as an
interim behavior candidate.

A fresh run from iteration zero with the checked-in config was started in
`logs/mini_cheetah_ref_vsim_tune/Jul29_10-42-19_`. It completed iteration 142,
but the RTX 5080 fell off its PCIe bus (`NVRM Xid 79`, followed by Xid 154
requiring a node reboot) before the next checkpoint. This was a host
GPU/PCIe/driver failure, not a divergent policy: the last recorded rollout had
finite rewards, full episode duration, and no limit violations. The
iteration-100 checkpoint is intact, finite, and contains actor, critic, and
both optimizer states. Resume it for 400 additional iterations only after the
host is considered stable:

```bash
uv run --env-file .env.vsim scripts/train_mujoco.py \
  --task mini_cheetah_ref_vsim \
  --backend vsim \
  --device cuda:0 \
  --max_iterations 400 \
  --resume \
  --load_run Jul29_10-42-19_ \
  --checkpoint 100 \
  --headless \
  --disable_wandb
```

After the clean run, evaluate the saved checkpoints and then use a targeted
yaw-command curriculum or gait-turning term; do not deploy the interim policy
yet.

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
