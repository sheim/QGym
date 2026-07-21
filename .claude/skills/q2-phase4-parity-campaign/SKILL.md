---
name: q2-phase4-parity-campaign
description: The executable, decision-gated campaign for Q2's hardest live problem — making GPU (warp) training trustworthy, proving CPU/GPU/legacy parity, and completing MIGRATION_PLAN Phase 4 (IsaacGym removal). Numbered phases with exact commands and expected observations, ranked solution menu for the warp root_states staleness bug, fenced-off wrong paths, and a promotion protocol. Load whenever working on GPU training correctness, backend parity, Phase 4, IsaacGym removal, or before trusting/benchmarking any warp training result.
---

# Campaign: Warp Parity → Phase 4 Close-out

**Mission:** end state where `--device cuda:0` training is provably correct,
CPU/GPU learning curves agree, domain randomization status is known (working or
explicitly de-scoped), IsaacGym code is deleted, and CI runs the real suite.

**Prime blocker (2026-07-10):** on `port`, `MuJocoWarpBackend` refreshes
`root_states` (and `rigid_body_states`) only inside their property getters;
`LeggedRobot._init_buffers` caches those tensors once
(`legged_robot.py:496,502`). Therefore **all GPU training so far ran with
frozen base-state observations** (`base_lin_vel`, `base_ang_vel`,
`projected_gravity`, `base_height`). `origin/jt/port` band-aids it with a line
whose author wrote "unknown why this works" (it works because the assignment
triggers the getter; the copy itself is a no-op). Everything below is ordered
around removing this uncertainty first.

Decision gates — ANSWERED by Steve, 2026-07-10:
**G1:** `jt/port` WILL be merged in; Steve handles the merge himself, later.
⇒ do NOT cherry-pick from it or duplicate its work; write fixes so they
survive the merge (expect conflicts in `mujoco_warp_backend.py`,
mini_cheetah configs, `on_policy_runner.py`), and flag any change that
touches those files.
**G2:** an IsaacGym-capable machine STILL EXISTS ⇒ the original Phase-4
legacy A/B checks (learning-curve comparison, IsaacGym-checkpoint
portability) are feasible; the "redefined" MuJoCo-only checks below remain
the fallback when that machine isn't at hand.

---

## Phase 0 — Make warp state sync trustworthy — ✅ COMPLETE 2026-07-11 (branch `vsim`)

Executed as specified below. Results: 0.1 test written
(`tests/unit_tests/test_task_state_liveness.py`) — red on warp (tensor
byte-identical after 50 control steps), green on cpu, confirming the bug at
task level. 0.2 fixed via **option A**: `_sync_assembled_states()` called from
`step()`, `setup()`, `reset_dof_state()`, `reset_root_state()`; getters plain;
W3 documented in code. 0.3 suite: **153 passed, 21 skipped** (was 151).
0.4 overhead: ~29.3k steps/s after vs ~29.3–31.8k before (mini_cheetah, 4096
envs, 30 iters, RTX 4080) — within noise. Bonus finding: pre-fix GPU
`total_rewards` ≈ 6.2 vs ≈ 1.1 post-fix at iteration 30 — frozen observations
were inflating rewards, confirming pre-fix GPU results were fake. Remaining
tie-off: remove jt/port's band-aid line during the merge (Phase 4.1).
Original phase spec kept below for reference/audit.

**0.1 Write the failing test FIRST** (this is the missing regression, weak
point W4). New file `tests/unit_tests/test_task_state_liveness.py`, pattern
copied from `test_legged_termination.py` (builds a full env via
`task_registry.make_env_mujoco`):

- Build `mini_cheetah`, 2 envs, seed 0, device parametrized cpu / cuda:0
  (skip cuda if unavailable).
- Record `env.root_states.clone()`, drive nonzero torques ~50 env steps,
  assert `not torch.allclose(env.root_states, before)` AND
  `env.root_states.data_ptr() == env._backend.root_states.data_ptr()`
  (same storage, updated in place). Repeat for `env._rigid_body_state`.

**Expected observations at this gate:** cpu variant PASSES on `port` today;
cuda variant FAILS (frozen tensor). If the cuda variant *passes* on
unmodified `port`, your test is wrong (probably re-reading the property) —
fix the test, do not proceed.

**0.2 Fix — solution menu, ranked:**

| Option | What | Verdict |
|---|---|---|
| **A (recommended)** | Move the swizzle/assembly refresh into `MuJocoWarpBackend.step()` (and `reset_root_state`), mirroring the CPU backend's `_sync_state_from_mujoco`; getters become plain returns | Honors the SimBackend contract ("all tensors live after step()", `sim_backend.py:60-62`); fixes root_states AND rigid_body_states; obligation: measure step overhead before/after (expect noise-level — two small tensor copies vs a physics solve; see 0.4) |
| B | Adopt jt/port's `self.root_states[...] = self._root_states_t` line | Works by getter side effect; obscure, leaves `rigid_body_states` stale, invites deletion by a future cleaner. Acceptable only as a stopgap WITH a comment explaining the mechanism, and it does not close this phase |
| C | Make the task layer re-read `self._backend.root_states` each `_post_physx_step` | Violates the migration's core principle (task layer unchanged across backends); touches many call sites |
| D | Rebuild root_states as genuinely zero-copy (custom warp kernel writing a persistent buffer) | Over-engineering until A is shown too slow |

Also resolve **W3** while in the file: warp `dof_state` getter returns a
`torch.stack` copy; either make the cached `self.dof_state` unnecessary or
document it as write-never/read-never on warp (architecture skill W3).

**0.3 Gate:** `uv run python -m pytest tests/unit_tests/ -q` — new tests green
on cpu AND cuda; expected total goes from 172 collected / 151 passed
(2026-07-10 baseline) to strictly more, zero failures.

**0.4 Overhead check:**
`uv run scripts/train_mujoco.py --task mini_cheetah --device cuda:0 --num_envs 4096 --max_iterations 30 --headless --disable_wandb`
before/after the fix; compare steps/s from the logger block. Expected: within
noise (±5%). If >10% regression → the copy is on the wrong stream/device;
investigate before merging, don't accept it.

---

## Phase 1 — Warp warnings and the old GPU-crash ghost

1. Reproduce or retire the never-closed "quadruped on GPU crashes partway
   through" thread (`63b5b3d`, 2026-04-08):
   `uv run scripts/train_mujoco.py --task mini_cheetah --device cuda:0 --num_envs 4096 --max_iterations 1000 --headless --disable_wandb`
   run to completion twice. **If it crashes:** capture the exact error +
   iteration, add an archaeology entry, and check constraint-capacity warnings
   first (below). If clean twice → mark the thread closed in
   q2-failure-archaeology.
2. ✅ RESOLVED 2026-07-11 (done early, as a dependency of the jt/port tuning
   pull): mjspec config mechanism merged into `vsim` WITH the missing piece —
   warp ignores `mjModel.njmax`, so the backend now forwards it to
   `mjw.put_data(njmax=...)` (jt/port's spec-only route was a silent no-op;
   his branch trains with dropped constraints). mini_cheetah: `njmax=200`
   (runtime demanded 160 post-fusestatic), `ccd_iterations=50` (100 halves
   throughput). "nefc overflow" count in a training log must be ZERO — it
   means dropped constraints, not noise. Per-robot values; never copy blindly.

---

## Phase 2 — Pendulum parity (cheap, decisive)

Same seed, both backends:

```bash
uv run scripts/train_mujoco.py --task pendulum --device cpu    --num_envs 256  --max_iterations 200 --seed 7 --headless --disable_wandb
uv run scripts/train_mujoco.py --task pendulum --device cuda:0 --num_envs 4096 --max_iterations 200 --seed 7 --headless --disable_wandb
```

**Expected (historical marks, 2026-04, RTX 4080):** CPU ~16,600 steps/s,
reward reaching ≈ +1.2; GPU ~255,000 steps/s, reward ≈ +1.5. Gate: both
curves rise monotonically-ish to ≥ +1.0 by iteration 200 and the per-term
reward decomposition (printed each iteration) has no term frozen at its
initial value. Exact final-reward equality is NOT expected (different
num_envs/batch trajectories); curve *shape* agreement is.
**If GPU underperforms CPU per-step:** re-run the lockstep physics test
(`uv run python -m pytest tests/unit_tests/test_cross_backend_physics.py -v`
— prints max deviations; tolerance 1e-3). Physics matching + learning
diverging ⇒ obs/plumbing problem → debugging playbook, staleness family.

---

## Phase 3 — Mini_cheetah parity + domain-randomization verdict

1. Convergence both backends (batch equalized):
   `--task mini_cheetah --seed 7`, CPU `--num_envs 64`, GPU `--num_envs 4096`,
   same `--batch_size` (num_steps_per_env auto-adjusts, commit `936a4cc`).
   Gate: tracking rewards climb on both; inspect gait with
   `uv run scripts/play_mujoco.py --task mini_cheetah` (keyboard teleop;
   trotting, no limb dragging, responds to Up/Down/turn commands).
2. **Friction DR experiment (answers MIGRATION_PLAN's first open checkbox).**
   Hypothesis to refute: friction randomization is a no-op under MuJoCo
   (the `_process_rigid_shape_props` callback is IsaacGym-only; MuJoCo backends
   call only `_get_env_origins` + `_process_dof_props`,
   `mujoco_backend_base.py:223-230`).
   Discriminator (no training needed): build two envs with
   `friction_range=[0.05,0.05]` vs `[2.0,2.0]` and print the model's ground/foot
   `geom_friction` — identical values ⇒ confirmed no-op.
   Then either (a) implement: per-env friction is awkward with one shared
   MjModel on CPU (model-level) but per-world-able in warp — design decision,
   record it; or (b) de-scope explicitly: tick the plan checkbox with
   "not applied under MuJoCo, tracked as future work", and remove the dead
   config axes from active task configs. **Silence is the only forbidden
   outcome.**
3. Same for `randomize_base_mass` (humanoid config uses it).

---

## Phase 4 — Reconciliation, removal, CI

1. **jt/port reconciliation (G1: Steve merges it himself, later).** Most of
   jt/port already lives on `vsim` in corrected form (tuning, fusestatic,
   mjspec mechanism + njmax forwarding — pulled 2026-07-11 per Steve). At
   merge time: (a) DELETE the band-aid line
   `self.root_states[...] = self._root_states_t` and its comment
   (superseded by `_sync_assembled_states`); (b) prefer vsim's
   `njmax=200`/comments over jt's 90 (his route was a no-op); (c) the only
   genuinely new content arriving is SaveStates tooling (commented out) and
   `info.txt` (delete).
2. **Checkpoint portability.** Primary (G2: IsaacGym machine exists): the
   original plan check — save a checkpoint under the IsaacGym backend
   (`scripts/train.py` on that machine's py3.8 venv), load and run it under
   MuJoCo warp/CPU here; and compare same-seed learning curves IsaacGym vs
   MuJoCo per reward term. Fallback when that machine isn't available: train
   on CPU backend, `play_mujoco.py --device cuda:0` the same checkpoint (and
   vice versa). Gate either way: loads cleanly, behavior qualitatively
   identical.
3. **Determinism:** two runs, same seed, `--device cpu`, assert identical
   reward at iteration 10 (CPU backend is deterministic; warp is not
   guaranteed).
4. **IsaacGym removal** (MIGRATION_PLAN Phase 4 list, expanded):
   delete `gym/envs/base/isaac_gym_backend.py`; remove isaacgym import guards
   (`grep -rn "isaacgym" gym/ learning/ scripts/ | grep -v Binary` must end
   empty); remove `gym`/`sim` shims in `base_task.py` + `TODO Phase 3`
   markers; delete dead IsaacGym-only code paths (projectiles, terrain
   curriculum, `KeyboardInterface.py`, `VisualizationRecorder.py`,
   `scripts/train.py`, `play.py`, `export_policy.py`, `tests/integration_tests/`,
   `tests/regression_tests/` or their rewrite); retire `requirements.txt` +
   `setup.py`; fix `[tool.setuptools] packages` (add pendulum/cartpole).
   Each deletion: full suite + a pendulum & mini_cheetah smoke train.
5. **CI modernization:** `unit_tests.yml` → uv + Python 3.13 + `uv run python
   -m pytest tests/unit_tests/ gym learning -q`, triggered on `port` (and
   later `main`). This closes the "zero CI on port" hole.
6. Tick MIGRATION_PLAN checkboxes; update README_MUJOCO if CLI changed; add
   archaeology entries for every battle fought.

---

## Fenced-off wrong paths (do not walk them again)

- **Do not tune reward weights to make GPU training "work"** while Phase 0 is
  open — you'd be fitting around frozen observations (jt/port's July tuning
  churn happened in exactly this shadow).
- **Do not benchmark or publish GPU throughput/learning numbers** from
  pre-Phase-0 code; retroactively, the Phase-3k GPU smoke result carries this
  asterisk.
- **Do not add a silent warp→CPU fallback** to dodge GPU issues (removed
  deliberately in `ebc2925`; fail-fast is policy).
- **Do not call the property getter from reward/obs code** as "the fix" —
  same side-effect trap, one layer up.
- **Do not commit binary references** to revive the regression suite (100 KB
  gate; that's how it died).

## Promotion protocol

A phase counts as done only when: its gate command outputs match the stated
expectations (paste them in the PR); new/changed behavior has a test; suite
green cpu+cuda; MIGRATION_PLAN + affected skills updated in the same change;
gates from q2-conventions-and-change-control all pass. Success is measured by
test output and curves — never by eye.

## When NOT to use this skill

- General debugging → `q2-debugging-playbook`. Evidence standards →
  `q2-research-methodology`. The underlying design rules →
  `q2-architecture-contract`.

## Provenance and maintenance

Grounded 2026-07-10 in: `mujoco_warp_backend.py` (getter-only refresh),
`legged_robot.py:496,502` (caching), jt/port commits `e604532`/`2326b71`
(discovery + band-aid), MIGRATION_PLAN Phase 4 + open checkboxes, and the
2026-04 benchmark numbers in MIGRATION_PLAN 1f/1i. Re-verify before executing:

```bash
grep -n "root_states" gym/envs/base/mujoco_warp_backend.py   # staleness fixed already? → skip to Phase 1
git log --oneline port..origin/jt/port | wc -l               # reconciliation still pending?
grep -n "\[ \]" MIGRATION_PLAN.md                            # open checkboxes
```
