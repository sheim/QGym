---
name: q2-failure-archaeology
description: The chronicle of Q2's major investigations, bugs, dead ends, reverts, and their resolutions — symptom → root cause → evidence → status, mined from git history through 2026-07-10 (port branch and jt/port). Load BEFORE re-investigating any suspicious behavior, before "fixing" something that might be settled, and when a commit message or code comment references a past battle. For live triage use q2-debugging-playbook; this file is the historical record behind it.
---

# Q2 Failure Archaeology

Format: **hash date — title** / symptom → root cause → fix / STATUS.
Branch context: `port` @ `bc2bd96` (2026-05-21) is canonical; `origin/jt/port`
(JoshuaTchou, through `15d3138`, 2026-07-09) is 13 commits ahead, unmerged.

## Era 1 — pre-port (settled battles you may still trip over)

- **`ea5cdff` 2024-09-09 — "THIS BREAKS PBRS" (self-documented).** Reward
  evaluation moved from env to runner: deleted `TaskSkeleton.compute_reward()`
  and the `eval("self._reward_"+name)` hack; runner now builds a dict of bound
  `_reward_*` methods. Benefits: per-reward profiling, no eval. Casualty:
  `learning/utils/PBRS/` still calls the deleted `env.compute_reward`. No fix
  ever landed on any branch. STATUS: **open — PBRS is dead code in tree**; its
  unit test passes because it tests the class in isolation.
- **`7296c6c` 2024-09-09 — switch computed once per decimation.** ~10% speedup;
  divisor became `reward_settings.switch_scale`. STATUS: kept.
- **`b1bd4f2` 2024-09-11 — hardcoded humanoid coupling Jacobian.** `_apply_coupling`
  rebuilt a constant J + `torch.inverse` every call; precomputing halved its
  cost. Lesson: profile before optimizing — the two big speedups of that era
  both came from profiling. STATUS: kept. (Commit also smuggled unrelated
  humanoid hyperparameter changes — see change-control skill for why we don't
  do that.)
- **`c0bda90` 2025-10-24 — OSC (oscillator/CPG gait controller) resurrection.**
  Large refactor of `MiniCheetahOsc`: per-leg oscillators with GRF feedback,
  `MINI_CHEETAH_WEIGHT = 8.292*9.81` normalization, removed dead
  swing/stance-velocity rewards, gaussian command noise. Note in commit: osc
  omega/coupling "gets overwritten" by randomization. STATUS: fixed then; MuJoCo
  behavior unvalidated.
- **`7cf29c7` + `0606552` 2025-10-31 — CI teardown.** Self-hosted IsaacGym
  runners removed (PR_tests/push_tests/regression_tests workflows); replaced by
  cheap `pytest gym learning` on hosted runners; regression test made a manual
  standalone. STATUS: deliberate; consequence today = no CI on `port` (see
  q2-testing-and-validation).
- **`c5a598f` 2025-11-11 — per-joint rewards sum→mean** so weights transfer
  across robots; mini_cheetah weights rescaled ~×10 (`torques 5e-7→5e-6`,
  `action_rate 0.01→0.1`, `action_rate2 0.001→0.01`). STATUS: kept — respect it
  when adding per-joint rewards.
- **Horse-branch reverts (`3e92cf7`, `6d4822f` 2026-02; `c7de97c` 2026-03)** —
  tendons/switch-function, descent-reward smoothing, height plots. All on the
  yl/horse lineage, **not in port's ancestry**. STATUS: irrelevant to port;
  don't excavate.

## Era 2 — the port, Phase 0→3 (2026-03-31 → 2026-05-21)

- **`b329655` 2026-03-31 — robot purge.** Deleted a1, anymal_b/c, cassie +
  meshes. Side effect: the `pendulum` task_dict entry was deleted too —
  pendulum silently unregistered until `5d4ba51` restored it. Lesson: task
  registration lives in dicts; deletions can take innocent bystanders. STATUS:
  fixed.
- **`0153705` 2026-04-02 — Phase 0 extraction.** SimBackend ABC +
  IsaacGymBackend (+1349/−487); contract-test scaffolding born
  (MockBackend, 44-test contract suite). STATUS: foundation, done.
- **GPU bring-up saga 2026-04-07/08** (`5d4ba51`→`973b8d5`), the densest bug
  cluster in the repo:
  - `5d4ba51` both MuJoCo backends + train_mujoco.py; CPU worked, warp didn't.
  - `ebc2925` `select_backend` made fail-fast: `cuda*` → warp unconditionally
    (a silent CPU fallback was considered and rejected — house rule).
  - `1bdd0d8` floating-base coordinates (`nq=7+nj`/`nv=6+nj`), quaternion
    boundary convention settled, pure-torch `torch_quat.py` replaces
    `isaacgym.torch_utils`. Blocker found: mujoco-warp needs Python ≥3.11 +
    mujoco ≥3.6, incompatible with the py3.8 IsaacGym venv →
  - `63b5b3d` repo re-founded as standalone `q2` project (uv, py3.11–3.13).
    Commit message records: **"quadruped on GPU crashes partway through"** —
    never explicitly closed. Closest follow-ups are jt/port's warp params
    (below). STATUS: **open thread**.
  - `92cb8e9` dedup into `MuJocoBackendBase` (MjSpec pipeline, swizzles).
  - `2e28f8d` deleted commented-out legacy shim blocks.
  - `973b8d5` device-assignment bug: tasks passed `backend.device` to
    `super().__init__` before `backend.setup()` had set it. Fix: pass the
    requested device string. STATUS: fixed; the trap pattern remains
    (architecture invariant 10).
- **`bb8c981`/`0abf893` 2026-04-08/09 — macOS.** mjpython/libpython dylib trap
  discovered and documented; CPU backend raises instructive RuntimeError.
  STATUS: fixed, documented.
- **`936a4cc` 2026-04-20 — batch-size collapse at small num_envs.** Fixed
  `num_steps_per_env` (24/32, tuned for 4096 IsaacGym envs) starved PPO when
  running 64 CPU envs. Fix: `num_steps_per_env = max(1, batch_size //
  num_envs)`, removed from all task configs. STATUS: fixed.
- **`fba2deb` 2026-05-08 — contact termination never fired.** Root cause:
  MuJoCo leaves `cfrc_ext` at zero after `mj_step`/`forward+euler`; it is only
  filled by `mj_rnePostConstraint` (CPU) / `mjw.rne_postconstraint` (warp).
  Fix: call it post-step in both backends + regression test
  (`test_legged_termination.py`). STATUS: fixed with test.
- **`bfd6b2d` 2026-05-18 — props-dict length bug.** `_process_dof_props`
  iterated `range(len(props))`, but MuJoCo props is a dict → `len` = 4 keys, so
  only 4 of 12 joints got real limits. Fix: vectorized dict reads. Commit also
  left debug flags (`disable_gravity=True`, `disable_actions=True`) — cleaned
  up in `4538b5e`. STATUS: fixed. Lesson: WIP commits with debug flags happen;
  grep configs for `disable_` before trusting a training result.
- **`2bc709b` 2026-05-18 — URDF limits silently discarded.** MuJoCo's URDF
  importer drops `<limit effort velocity>` (expects them on actuators, which we
  don't create). Backend had been serving 1e6/1e3-style fakes. Fix:
  `_parse_urdf_limits()` (ElementTree) + test asserting mini_cheetah's real
  numbers (effort 18/18/28, velocity 41/41/26.8). Left open in-code question:
  soft `dof_pos_limits` computation commented out ("remove? Put into penalty
  instead"). STATUS: fixed with test; soft-limit question **open**.
- **`95b9a6e` 2026-05-18 — pendulum learned to lie horizontal.** `_reward_theta
  = _sqrdexp(cos θ)` peaks at cos θ = 0 → horizontal. Fix: `theta_err = 1 −
  cos θ`. Also added `play_pendulum.py` diagnostics (phase portrait, energy
  panels). STATUS: fixed. Lesson generalized in the debugging playbook: know
  the zero-point of every `_sqrdexp` argument.
- **`83526ff`/`afa8642` 2026-05-18 — menagerie visuals.** Skybox, lighting,
  checker plane; `_load_urdf_spec` injects
  `<mujoco><compiler discardvisual="false"/></mujoco>` (also keeps the spec
  mutable — without it add_texture/add_material get silently pruned at
  compile), strips `.dae` visuals MuJoCo can't decode. STATUS: done.
- **`894cded` 2026-05-21 — keyboard teleop**, CPU/passive-viewer only; viewer
  key callback must attach via `_viewer_key_callback` indirection because the
  viewer is created during env construction. STATUS: done; warp has no viewer.

## Era 3 — jt/port (2026-06-23 → 2026-07-09, UNMERGED as of 2026-07-10)

- **`831074e`→`daa3ef8` — warp model params.** mujoco-warp runtime warnings
  (constraint capacity, CCD iterations) for mini_cheetah silenced by
  `njmax=90`, `ccd_iterations=50`; evolved from hardcode → isinstance hack →
  generic `cfg.mjspec_attributes`/`cfg.mjspec_option_attributes` mechanism.
  STATUS: **the refactor silently regressed the njmax half** (discovered
  2026-07-11): mujoco-warp ignores the legacy `mjModel.njmax` field the spec
  route sets — capacity must be passed to `mjw.put_data(njmax=...)`.
  831074e (direct put_data arg) worked; 513fbcb (spec attr) no-ops → jt/port
  trains with "nefc overflow" (silently dropped constraints). Fixed on
  `vsim`: mechanism pulled + warp backend forwards `mjm.njmax` to put_data;
  njmax retuned 90→200 (runtime demanded 160 post-fusestatic at 4096 envs);
  `ccd_iterations=50` kept as a measured tradeoff (100 halves throughput,
  13.2k→7.1k steps/s, and still warns occasionally).
- **`8e2cfc7`/`e604532` — state logging (SaveStates.py).** Buffers of chosen
  env states, histogram PNGs + `saved_states.pt` post-training. The
  instrumentation that exposed the root_states bug. `15d3138` commented it out
  (left in tree). STATUS: useful debug tool, currently commented out on jt/port
  only.
- **`e604532` + `2326b71` — THE root_states discovery.** Joshua's
  instrumentation showed warp `root_states` frozen during training. The "fix"
  `self.root_states[...] = self._root_states_t` in `step()` works **only
  because the property getter performs the refresh as a side effect** (the
  assignment itself is a self-copy no-op); `2326b71` is a comment-only commit —
  "unknown why this works". Root cause chain: warp backend refreshes
  `_root_states_t` only in the getter; `legged_robot.py:496` caches the tensor
  once at init; sim_backend.py's contract ("all tensors live after step()")
  violated. **Implication: every GPU training result on `port` HEAD has frozen
  base-state observations** — including, retroactively, the Phase-3k GPU smoke
  test. STATUS: **open on port**; band-aid on jt/port; proper fix + regression
  test = campaign Phase 0. Same pattern latent for `rigid_body_states`.
- **`163323c` → `135adf2` — fused-body saga.** mini_cheetah `foot` body fused
  into `shank` by MuJoCo's rigid-joint fusing → `foot_name="foot"` matched
  nothing. First workaround renamed config to `"shank"`; proper fix
  `spec.compiler.fusestatic = False` and rename reverted. STATUS: pulled into
  `vsim` 2026-07-11 (required by the GRF rewards — `feet_indices` needs foot
  bodies). Side effect: shifted the CPU↔warp chaotic divergence curve; the
  lockstep test was restructured into two windows (1e-4 pre-chaos ≤100 steps,
  0.2 blow-up bound ≤200) — stronger against systematic bugs, robust to chaos.
- **`e7d8b0b`, `15d3138` — mini_cheetah_ref weight tuning.** Net deltas:
  orientation 1.0→1.5, min_base_height 1.5→1.0, action_rate2 0.01→0.05,
  reference_traj 1.0→3.0, swing/stance_grf 0→1.5, ang-vel tracking divisor
  5.0→2.5, scaling.base_height 0.3→0.15. STATUS: pulled into `vsim`
  2026-07-11 per Steve ("trains better"); note the tuning was done on top of
  jt/port's root_states band-aid (live obs) but WITH dropped constraints
  (njmax no-op) — on vsim it now runs with correct constraint capacity.

## Standing conclusions (don't relitigate)

1. Fail-fast backend selection (no silent CPU fallback) is deliberate (`ebc2925`).
2. Contact forces require the post-constraint RNE call — regression-tested.
3. URDF `<limit>` parsing is ours, not MuJoCo's — regression-tested.
4. Reward-function discovery lives in the runner; envs just define `_reward_*`.
5. The py3.8-IsaacGym and py3.11+-MuJoCo worlds cannot share a venv; that's why
   the repo is uv-first and IsaacGym paths are import-guarded.

## When NOT to use this skill

- Live triage of a fresh symptom → `q2-debugging-playbook` (it encodes these
  lessons as a lookup table).
- Understanding the intended design → `q2-architecture-contract`.

## Provenance and maintenance

Mined 2026-07-10 from `git log`/`git show` on `port`, `origin/jt/port`, and
pre-port history. All hashes are in this repo. Re-verify drift:

```bash
git log --oneline port..origin/jt/port        # jt/port merged yet? update Era 3 statuses
git log --oneline -5                          # new commits since bc2bd96 → chronicle them
grep -rn "compute_reward" learning/utils/PBRS/   # PBRS repaired yet?
```
