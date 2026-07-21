---
name: q2-architecture-contract
description: Load-bearing design decisions, invariants, and known weak points of Q2's SimBackend architecture (IsaacGym→MuJoCo port). Load BEFORE modifying anything in gym/envs/base/ (backends, base_task, fixed_robot, legged_robot, task_skeleton), before touching state tensors, resets, or quaternion handling, and whenever observations look stale or wrong. NOT a MuJoCo API guide (use mujoco-backend-reference) and NOT a symptom lookup (use q2-debugging-playbook).
---

# Q2 Architecture Contract

## The system in one diagram

```
SimBackend (ABC)                     gym/envs/base/sim_backend.py
    ├── IsaacGymBackend              legacy; removed in Phase 4; needs isaacgym+py3.8
    ├── MuJocoCPUBackend             mj_step loop, one MjData per env; all platforms
    ├── MuJocoWarpBackend            mujoco_warp, batched nworld; Linux+CUDA (or warp-CPU)
    │    both share MuJocoBackendBase (URDF→MjSpec loading, model config, metadata)
    └── VSimBackend                  vlearn engine, CUDA-only, licensed; ~10× warp
         (gym/envs/base/vsim_backend.py + vsim_asset.py URDF→.vsim pipeline)

TaskSkeleton                         gym/envs/base/task_skeleton.py — obs get/set + scaling, episode buffers
    └── BaseTask                     holds self._backend; NO engine imports; gym/sim shims (Phase-4 removal)
         ├── FixedRobot              pendulum, cartpole    (fixed base, contacts disabled)
         └── LeggedRobot             floating base + ground plane
              └── concrete tasks     mini_cheetah*, humanoid*, lander — rewards/obs only

task_registry (gym/utils/task_registry.py)
    make_env_mujoco(name, env_cfg, device, headless) → select_backend(cfg, device) → task(backend=...)
    select_backend: device "cuda*" → MuJocoWarpBackend (unconditional, fails loudly if absent);
                    otherwise (incl. macOS) → MuJocoCPUBackend
```

**Why this design:** the port strategy (MIGRATION_PLAN.md) was to cut a clean seam
(`SimBackend`) so the physics engine can be swapped while runners, algorithms, task
logic, and reward functions stay byte-identical. IsaacGym stays functional until the
MuJoCo path is validated end-to-end (Phase 4). Concrete tasks must never import an
engine. The seam is permanent, not scaffolding: further backends are planned
(v-sim, decision 2026-07-10) — new engines follow `q2-backend-integration`, and
every contract violation below gets copied into them if left unfixed.

## The SimBackend contract

From `gym/envs/base/sim_backend.py` (all shapes verified 2026-07-10):

| Member | Contract |
|---|---|
| `setup(cfg, num_envs, device, task)` | build world; state tensors valid afterwards; calls `task._get_env_origins()` and `task._process_dof_props(props, env_id=0)` |
| `step(torques)` | torques are `[num_envs, num_dof]` **full-DOF generalized forces** (backend applies the free-joint offset itself); ALL state tensors must be live when it returns |
| `reset_dof_state(env_ids)` | caller writes desired values into `dof_pos`/`dof_vel` views FIRST, then calls this to commit |
| `reset_root_state(env_ids)` / `set_all_root_states()` | same write-then-commit pattern for `root_states`; no-op default for fixed-base |
| `dof_pos`, `dof_vel` | `[num_envs, num_dof]`, must be **live writable views** into `dof_state` |
| `dof_state` | `[num_envs*num_dof, 2]` (pos, vel) pairs |
| `root_states` | `[num_envs, 13]` = pos(3), quat(4) **scalar-last**, lin_vel(3), ang_vel(3) |
| `contact_forces` | `[num_envs, num_bodies, 3]` |
| `rigid_body_states` | `[num_envs*num_bodies, 13]`, legged only |
| `penalised_contact_indices`, `termination_contact_indices` | built in setup from `cfg.asset.penalize_contacts_on` / `terminate_after_contacts_on` (substring match on body names) |

## Invariants — do not break these

1. **Quaternions are scalar-last `[x,y,z,w]` everywhere above the backend.**
   MuJoCo is scalar-first `[w,x,y,z]` internally; the swizzle happens ONLY inside
   backends via `WXYZ_TO_XYZW` / `XYZW_TO_WXYZ` (`mujoco_backend_base.py:19-20`).
   If you index a quaternion in task code, `[:, 3]` is w.
2. **Write-then-commit reset protocol.** Resets write into the live tensor views,
   then call `reset_dof_state`/`reset_root_state` to push into the engine. Never
   bypass it: the CPU backend copies views→MjData + `mj_forward`
   (`mujoco_cpu_backend.py:124-141`); warp writes land zero-copy and only
   `mjw.forward` is needed.
3. **`dof_pos`/`dof_vel` must remain views.** Task code holds references captured
   once in `_init_buffers` (`legged_robot.py:496-505`, `fixed_robot.py:153-158`)
   and expects in-place updates forever after.
4. **"All tensors live after `step()` returns"** (`sim_backend.py:60-62`). The CPU
   backend honors this by syncing everything each step (`_sync_state_from_mujoco`,
   `mujoco_cpu_backend.py:97-120`). **The warp backend currently violates it** —
   see weak point W1 below.
5. **Torques are full-DOF.** `FixedRobot`/`LeggedRobot` build
   `[num_envs, num_dof]` tensors; backends slice with `_qvel_offset` for the free
   joint (`mujoco_warp_backend.py:127-131`, `mujoco_cpu_backend.py:91-93`).
6. **Floating-base detection is `nq == nv + 1`** → offsets qpos 7 / qvel 6
   (`mujoco_backend_base.py:179-191`). Fixed-base asserts `nq == nv` and disables
   all contacts (`geom_contype[:] = 0`, lines 198-200).
7. **Frequencies, not dts, are the source of truth.** `cfg.control.ctrl_frequency`
   and `desired_sim_frequency` → `task_registry.set_control_and_sim_dt` computes
   `decimation`, `ctrl_dt`, `sim_dt`. Never hand-set dt fields (see
   q2-config-system).
8. **Reward functions are discovered by name.** The runner builds a dict of bound
   `_reward_<name>` methods from the nonzero weights in
   `train_cfg.critic.reward.weights` (moved out of the env in 2024, commit
   `ea5cdff`). Adding a reward = define `_reward_foo(self)` returning
   `[num_envs]` + add weight `foo` in the runner cfg. Nothing else.
9. **Per-joint rewards take the `mean` over joints, not the sum** (commit
   `c5a598f`), so weights transfer across robots with different DOF counts. Keep
   new per-joint rewards consistent.
10. **Init-order hazards (learned the hard way):**
    - `backend.device` is only valid after `backend.setup()`; pass the requested
      device string through instead (bug fixed in `973b8d5`).
    - `LeggedRobot._parse_cfg` must run after `super().__init__` (Phase 3b fix).
    - The CPU viewer is created lazily on first `render()`, which happens during
      `LeggedRobot.__init__` — key callbacks must attach via the
      `_viewer_key_callback` indirection, not at viewer creation
      (`mujoco_cpu_backend.py:162-171`).

## Known weak points (statuses updated 2026-07-11)

- **W1 — warp `root_states` staleness — FIXED 2026-07-11 (branch `vsim`).**
  Historically, `MuJocoWarpBackend` refreshed `_root_states_t` only in the
  `root_states` property getter while `LeggedRobot._init_buffers` caches the
  tensor once (`legged_robot.py:496`) — so GPU training ran with frozen
  `base_lin_vel`/`base_ang_vel`/`projected_gravity`/`base_height` (and
  inflated rewards: mini_cheetah ~6.2 fake vs ~1.1 real @ iter 30). Fix:
  `_sync_assembled_states()` called from `step()`, `setup()`, and both reset
  methods; getters are plain returns. `origin/jt/port` carries a redundant
  getter-side-effect band-aid — remove it at merge. Any branch without this
  fix, and any pre-fix GPU result, still has the bug.
- **W2 — warp `rigid_body_states` — FIXED with W1** (same `_sync_assembled_states`
  path). `legged_robot.py:502` caches it as `self._rigid_body_state`.
- **W3 — warp `dof_state` getter returns a fresh `torch.stack` COPY** (documented
  in a code comment now), not a view — an interleaved view over separate qpos/qvel
  Warp arrays is impossible. Read-only convenience; resets must go through the
  `dof_pos`/`dof_vel` zero-copy views. Do not read `self.dof_state` in task code.
- **W4 — CLOSED 2026-07-11:** `tests/unit_tests/test_task_state_liveness.py`
  steps a full mini_cheetah env (cpu + warp) and asserts the task's CACHED
  tensors update in place. Fixture-swap it for every new backend
  (q2-backend-integration, ladder step 9).
- **W5 — `BaseTask.gym`/`BaseTask.sim` shims** (`base_task.py:35-41`) and
  `isaac_gym_backend.py` exist until Phase 4. Code gated on
  `isinstance(backend, IsaacGymBackend)` (projectiles, `prepare_sim`) is
  intentionally dead under MuJoCo.
- **W6 — `cfg.control.control_type` is read at `legged_robot.py:588` but defined
  in no config class** — latent AttributeError if that branch executes.
- **W7 — PBRS (`learning/utils/PBRS/`) is broken since `ea5cdff` (2024-09)**: it
  calls the deleted `env.compute_reward`. Its unit test passes (tests the class
  in isolation). Do not wire it up without repairing it.

## When NOT to use this skill

- Chasing a symptom → `q2-debugging-playbook` first.
- MuJoCo/warp API semantics (fusestatic, cfrc_ext, MjSpec) → `mujoco-backend-reference`.
- Reward/observation design → `legged-rl-reference`.
- Adding a task without touching base classes → `q2-task-authoring`.

## Provenance and maintenance

Facts verified against `port` @ `bc2bd96` on 2026-07-10. Re-verify before relying:

```bash
git log -1 --oneline                                             # still bc2bd96?
grep -n "root_states" gym/envs/base/mujoco_warp_backend.py       # W1: getter-only refresh?
grep -n "self.root_states = self._backend" gym/envs/base/legged_robot.py   # cached at :496?
grep -n "control_type" gym/envs/base/legged_robot.py gym/envs/base/*config*.py  # W6 still undefined?
grep -rn "compute_reward" learning/utils/PBRS/                   # W7 still calling deleted API?
grep -n "TODO Phase 3" gym/envs/base/*.py                        # shims still present?
```
