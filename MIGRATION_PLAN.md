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

## Phase 1 — MuJoCo Warp backend, FixedRobot

**Goal:** `MuJocoWarpBackend` implementing `SimBackend`.  Pendulum and Cartpole
train to convergence using the new backend on GPU and CPU Linux.

### Key design decisions

- `mjw.make_data(mjm, nworld=num_envs)` maps directly to `num_envs`
- `dof_pos` / `dof_vel` exposed as `wp.to_torch(d.qpos)` / `wp.to_torch(d.qvel)` — zero-copy on GPU
- Torques applied via `d.qfrc_applied` (raw generalised forces, matching the PD-computed torques from `_compute_torques`)
- No `refresh_*` calls needed — tensors are live after `mjw.step()`
- `reset_dof_state` writes via `wp.from_torch(...)` indexed slice
- Device selected as `"cuda"` or `"cpu"` and passed to `wp.init()`

### Files to create / modify

| File | Change |
|---|---|
| `gym/envs/base/mujoco_warp_backend.py` | **New** — `MuJocoWarpBackend(SimBackend)` |
| `gym/utils/task_registry.py` | Add backend selection logic; `make_env()` passes backend to task constructor |
| `gym/envs/base/sim_config.py` | Add MuJoCo solver params alongside existing `physx.*` block |

### MuJoCo coordinate note

For `FixedRobot` (fixed-base), `qpos` and joint DOFs are the same — no offset.
`LeggedRobot` is handled in Phase 3.

URDF assets are loaded natively via MuJoCo's built-in URDF importer
(`mujoco.MjModel.from_xml_file(urdf_path)`).

### Tests

Run the existing contract suite against the new backend by adding a fixture:

```python
# conftest.py addition
@pytest.fixture
def mujoco_warp_pendulum_backend(device):
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend
    b = MuJocoWarpBackend()
    b.setup_from_mjcf(PENDULUM_MJCF, num_envs=16, device=device)
    return b
```

Then run `test_backend_contract.py` parametrised over both `MockBackend` and
`MuJocoWarpBackend`.

**Convergence test:**
```
uv run scripts/train.py --task pendulum --headless
```
Expected: policy converges to upright balance within 200 iterations.  Compare
learning curve to IsaacGym baseline (same random seed).

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
