# IsaacGym → MuJoCo Warp Migration Plan

## Goal

Replace the IsaacGym/PhysX physics backend with MuJoCo Warp while keeping the
RL training infrastructure (runners, algorithms, task logic, reward functions)
completely unchanged.  The result will support:

- **Linux GPU** — MuJoCo Warp (`mujoco_warp`), batched across CUDA
- **Linux CPU** — MuJoCo Warp on CPU (Warp supports CPU execution)
- **Mac (Apple Silicon)** — plain `mujoco.mj_step` loop; no CUDA required

IsaacGym remains a fully-functional backend throughout the migration and is
only removed once the MuJoCo Warp backend is validated end-to-end.

---

## Style Guidelines

We use uv for environment management and ruff for style and linting.

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
              └── ConcreteTask   ← rewards, observations — never changes
```

`task_registry.py` selects the backend based on config / platform:

```python
def select_backend(device: str) -> SimBackend:
    if platform.system() == "Darwin":
        return MuJocoCPUBackend()
    try:
        import mujoco_warp
        return MuJocoWarpBackend()
    except ImportError:
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

## Phase 2 — Mac / CPU fallback (`MuJocoCPUBackend`)

**Goal:** Plain `mujoco.mj_step` loop with no Warp dependency.  Runs on Mac
(Apple Silicon) and any Linux CPU.  Slower (serial across envs) but
functionally identical.

### Key design decisions

- One `mujoco.MjModel` + `mujoco.MjData` per environment, stored in lists
- `step()` loops over envs in Python, calls `mujoco.mj_step(m, d)`
- After each step, numpy arrays are explicitly copied into PyTorch tensors
  (`self.dof_pos[:] = torch.from_numpy(np.stack([d.qpos for d in datas]))`)
- `dof_pos` / `dof_vel` are regular CPU PyTorch tensors (no zero-copy)
- `reset_dof_state` writes directly into `MjData.qpos` / `MjData.qvel`

### Files to create / modify

| File | Change |
|---|---|
| `gym/envs/base/mujoco_cpu_backend.py` | **New** — `MuJocoCPUBackend(SimBackend)` |
| `gym/utils/task_registry.py` | `select_backend()` returns `MuJocoCPUBackend` on Mac or when Warp unavailable |

### Tests

Add `MuJocoCPUBackend` fixture to `conftest.py` and run the full contract suite.
This suite is the primary gate for Mac CI since it requires no GPU.

**Mac smoke test (to be added to CI):**
```
python -m pytest tests/unit_tests/ -v      # no uv, no GPU
python -m pytest tests/unit_tests/ -k "cpu" -v
```
Expected: all contract tests pass on CPU.

---

## Phase 3 — LeggedRobot migration

**Goal:** `LeggedRobot` fully migrated to `SimBackend`.  Remove `gym`/`sim`
shim properties from `BaseTask`.  Mini-cheetah and humanoid train with
MuJoCo Warp backend.

### Additional complexity vs FixedRobot

**Generalised coordinate indexing:**
MuJoCo `qpos` for a free-floating robot is `[px, py, pz, qx, qy, qz, qw, j0, j1, ...]`
(`nq = 7 + num_joints`), `qvel` is `[vx, vy, vz, wx, wy, wz, dj0, dj1, ...]`
(`nv = 6 + num_joints`).  The backend assembles `root_states` from these slices:

```python
# root_states: [num_envs, 13]  pos(3) quat(4) linvel(3) angvel(3)
root_states[:, :3]  = qpos[:, :3]   # position
root_states[:, 3:7] = qpos[:, 3:7]  # quaternion (MuJoCo: scalar-last [qx,qy,qz,qw])
root_states[:, 7:10] = qvel[:, :3]  # linear velocity
root_states[:, 10:13] = qvel[:, 3:6] # angular velocity
dof_pos = qpos[:, 7:]
dof_vel = qvel[:, 6:]
```

⚠️ Verify quaternion convention matches existing reward functions (e.g.
`quat_rotate_inverse` in `_post_physx_step`).

**Contact forces:**
`d.cfrc_ext` shape is `[nworld, nbody, 6]` (torque + force).  Expose as:
```python
contact_forces = wp.to_torch(d.cfrc_ext)[..., 3:6]  # [N, nbody, 3]
```

**Terrain:**
Replace `gym.add_heightfield()` with a procedurally generated MuJoCo
`<hfield>` asset written at init time.  The height data from `terrain.py` is
reused; only the upload mechanism changes.

**Projectiles / push_robots:**
These call `gym.set_actor_root_state_tensor()` directly via the shim.
Migrate to `backend.set_all_root_states()` / `backend.reset_root_state()`.

### Files to modify

| File | Change |
|---|---|
| `gym/envs/base/legged_robot.py` | Full migration: call `backend.setup()`, remove `_create_envs`, `_create_*terrain`, `_init_buffers` tensor acquisition, `_step_physx_sim`, `_reset_system`; root_states sliced from qpos/qvel |
| `gym/envs/base/mujoco_warp_backend.py` | Add `root_states` property with free-joint slicing |
| `gym/envs/base/base_task.py` | Remove `gym` / `sim` shim properties |
| `gym/utils/terrain.py` | Add `to_mjcf_hfield()` method alongside existing IsaacGym upload |

### Tests

Add legged robot contract tests:
```
tests/unit_tests/test_legged_backend_contract.py
```
Testing root_states shape/content, contact_forces shape, and reset of both
DOF and root state.

**Convergence test:**
```
uv run scripts/train.py --task mini_cheetah_ref --headless
```
Expected: locomotion policy converges.  Gait pattern and reward curves
qualitatively match IsaacGym baseline.

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

### IsaacGym removal

Once all checks pass:
1. Delete `gym/envs/base/isaac_gym_backend.py`
2. Remove `try/except` guards around isaacgym imports (they become unconditional `pass` or are deleted)
3. Remove isaacgym from `requirements.txt` / `pyproject.toml`
4. Remove `gym/sim` shim properties from `BaseTask` (already done in Phase 3)

---

## Notes and known gotchas

- **isaacgym import order:** IsaacGym must be imported before PyTorch.  All
  files that import isaacgym use `try/except ImportError` with the isaacgym
  import placed before `import torch`.
- **`uv run` vs `.venv/bin/python`:** The project venv requires ninja to
  JIT-compile the gymtorch extension.  Always use `uv run` for both training
  and tests.
- **MuJoCo quaternion convention:** scalar-last `[qx, qy, qz, qw]`.  IsaacGym
  also uses scalar-last.  Verify this on a per-function basis before assuming
  they are identical.
- **`dof_state` view contract:** `dof_pos` and `dof_vel` must be views into
  `dof_state`, not copies.  Writing into `dof_pos[env_ids]` must be reflected
  in `dof_state` automatically.  Verified by `test_dof_state_view_consistent_*`.
- **Phase 3 TODO markers:** Shim code (backend's `gym`/`sim` properties,
  `register_dof_state()`, etc.) is annotated `# TODO Phase 3: remove`.
