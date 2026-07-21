---
name: q2-debugging-playbook
description: Symptom→triage table for Q2's known failure modes — startup NaNs, warp JIT "hangs", stale GPU observations, isaacgym NoneType errors, pytest INTERNALERROR, macOS viewer crash, missing contact forces, fused bodies, bogus joint limits, reward functions that optimize the wrong thing — each with its discriminating experiment. Load FIRST for any unexplained failure, crash, hang, or "training runs but robot is wrong" report, before reading source. For settled historical investigations see q2-failure-archaeology; for design invariants see q2-architecture-contract.
---

# Q2 Debugging Playbook

Work the table top-down; run the discriminating experiment before theorizing.

## Startup & environment symptoms

| Symptom | Likely cause | Confirm / fix |
|---|---|---|
| Reward columns print `nan` for the first iterations | Expected: episode-windowed averages have no completed episodes yet | Wait one episode length; documented in README_MUJOCO.md "Notes" |
| GPU run prints nothing for minutes; Ctrl-C traceback ends in `warp/_src/context.py … kernel.module.load` | First-run warp JIT kernel compilation, not a hang | Wait it out once; kernel cache makes later runs fast. (This is exactly how the 2026-07-10 `model_0.pt`-only runs died — user interrupt, not a crash) |
| `AttributeError: 'NoneType' object has no attribute 'parse_arguments'` | You ran an IsaacGym-only path (`train.py`, `play.py`, `export_policy.py`, `get_args`, integration tests) — the import guard set `gymutil=None` (`gym/utils/helpers.py:38-45`) | Use `train_mujoco.py`/`play_mujoco.py` instead |
| `pytest` INTERNALERROR before any test runs | Bare `pytest` collected `tests/integration_tests/` (session-start hook builds IsaacGym envs) | Target paths: `pytest tests/unit_tests/` or `pytest gym learning` |
| macOS: RuntimeError "passive viewer requires mjpython" | uv-bundled Python lacks `libpython3.13.dylib` | Homebrew-python venv recipe in `q2-build-and-env` T1, or `--headless` |
| `KeyError` from `task_registry.get_cfgs(name)` | One of the task's three imports failed silently (per-import guards in `gym/envs/__init__.py:87-111`) | `uv run python -c "import gym.envs.<pkg>.<module>"` to surface the real ImportError |
| Warp prints "nefc overflow - please increase njmax to N" | Constraint buffer too small — warp SILENTLY DROPS constraints (wrong physics, not cosmetic) | Set `cfg.mjspec_attributes.njmax ≥ N` (mechanism merged 2026-07-11; the warp backend forwards `mjm.njmax` to `put_data` — spec attr alone is NOT enough). mini_cheetah uses 200 |
| Warp prints "ccd_iterations … needs to be increased" occasionally | CCD hit its iteration cap on some frames — contact accuracy slightly reduced there | Deliberate tradeoff at 50 (100 halves steps/s); bump `cfg.mjspec_option_attributes.ccd_iterations` only with a throughput measurement |

## vsim backend symptoms

| Symptom | Likely cause | Confirm / fix |
|---|---|---|
| `OSError: libTurboActivate.so` on `import vlearn` | Process started without `.env.vsim` (loader reads LD_LIBRARY_PATH at exec; in-process fixes are impossible — libgym's NEEDED entries are SONAME-less) | `uv run --env-file .env.vsim …` or `bash scripts/run_vsim_tests.sh` |
| `libczmq.so.4: cannot open` | System dep new in vlearn 0.3.11 | `sudo apt install libczmq4` |
| `FATAL: License validation failed` | VL_WORKING_DIRECTORY not pointing at the dir with License.key/TurboActivate.dat, or license expired (0.3.11a0 ran on a 1-day trial 2026-07-12) | check `.env.vsim`; re-activate (human) |
| Robot launches skyward at spawn (z ≈ +5 m) | Spawn pose penetrates the ground → `maxDepenetrationVelocity` (10 m/s) kick. Tasks reset before stepping; RAW backend fixtures must spawn clear of leg length | spawn higher (see `legged_vsim_backend` conftest comment) |
| Contact forces read 0 at one instant on a settled robot | Instantaneous totals oscillate through zero during collapse rebounds — not sleeping, not a sensor bug | average over ~50 steps (see `test_contact_forces_carry_weight`) |

## "Training runs but results are wrong"

| Symptom | Likely cause | Confirm / fix |
|---|---|---|
| GPU (`--device cuda:0`) training plateaus / rewards implausibly high and flat / velocity-tracking terms frozen; same run on CPU learns | **Stale `root_states` on warp** — task caches the tensor; the backend refreshed it only in the property getter. FIXED 2026-07-11 (`_sync_assembled_states` in `step()`, branch `vsim`) — but present on `port`, on `jt/port` (band-aid only), and in any pre-fix GPU result/checkpoint | Discriminator: `uv run python -m pytest tests/unit_tests/test_task_state_liveness.py -v` — fails ⇒ the code you're on lacks the fix. Do NOT tune rewards around it |
| Body-pose-dependent logic wrong on GPU only (feet positions etc.) | Same pattern for `rigid_body_states` — fixed together with the above | Same liveness test covers it |
| Robot never terminates on base contact; falls through logic | `cfrc_ext` is only populated by `mj_rnePostConstraint` / `mjw.rne_postconstraint` — if someone removed the post-step call, contacts read zero | Regression test exists: `uv run python -m pytest tests/unit_tests/test_legged_termination.py -v` (story: commit `fba2deb`) |
| `find_body_index`/foot lookup fails, or a body listed in the URDF is missing | MuJoCo fuses bodies connected by rigid joints unless `fusestatic` is disabled; `vsim` sets `spec.compiler.fusestatic = False` (pulled 2026-07-11) — branches without it (incl. `port`) have e.g. mini_cheetah `foot` fused into `shank` | print `backend.body_names`; on vsim expect 18 mini_cheetah bodies incl. `*_foot` |
| Joint limits absurd (1e6) or only some joints limited | Two historical bugs: (a) MuJoCo's URDF importer discards `<limit effort velocity>` — worked around by `_parse_urdf_limits` (commit `2bc709b`); (b) props-dict iterated by `len(props)` = 4 keys not num_joints (fixed `bfd6b2d`) | `uv run python -m pytest tests/unit_tests/test_urdf_limits.py -v`; for a NEW robot check its URDF actually has `<limit>` tags with both attributes |
| Policy optimizes something absurd (e.g. pendulum happiest horizontal) | Reward zero-point wrong: `_sqrdexp(x)` PEAKS at x=0 — passing `cos θ` rewards θ=90° (bug fixed in `95b9a6e`) | For every `_sqrdexp` reward, write down what makes the argument zero; that's what you're maximizing |
| Device mismatch tensor errors during env construction | Init-order: `backend.device` valid only after `setup()`; `_parse_cfg` before `super().__init__` (both fixed once, `973b8d5` / Phase 3b — regressions possible) | Check the traceback's tensor devices; see q2-architecture-contract invariant 10 |
| Config edit has no effect | Field is IsaacGym-only under MuJoCo (terrain variants, projectiles, asset options…) | Check the "silently ignored" list in `q2-config-system` |
| Every env behaves identically / resets synchronized | `randomize_episode_counters` not called (train_mujoco does it; custom scripts must too) | grep your entry script |

## Discriminating experiments (cheap, in order of power)

1. **Suite**: `uv run python -m pytest tests/unit_tests/ -q` — 30 s; catches
   contract/physics regressions but NOT task-level warp staleness (W4).
2. **CPU-vs-GPU A/B**: same task/seed/iterations on `--device cpu` and
   `cuda:0`, compare reward curves per-term. Divergence isolated to
   base-state-dependent terms ⇒ staleness family.
3. **Lockstep cross-backend**: `uv run python -m pytest
   tests/unit_tests/test_cross_backend_physics.py -v` — max per-step deviation
   printed; separates "physics differ" from "task/obs plumbing differs".
4. **Frozen-tensor probe**: instrument the task, print `id()` and a value of the
   suspect cached tensor across 3 steps. Distinguishes stale-reference bugs
   from wrong-value bugs in one minute.
5. **Energy audit** (new physics config): zero torques + damping, assert energy
   decreases — copy `test_mujoco_cpu_physics.py`.

## House rules while debugging

- No `try/except` band-aids; make it crash where the cause is.
- Fix at the source, add the regression test next to the fix (pattern:
  `fba2deb` + `test_legged_termination.py`), then record the battle in
  `q2-failure-archaeology`.
- If your finding contradicts this playbook, update the playbook in the same
  change.

## When NOT to use this skill

- Environment/install failures → `q2-build-and-env`.
- The bug is already chronicled (check before deep-diving) →
  `q2-failure-archaeology`.
- You've confirmed the warp staleness family → go straight to
  `q2-phase4-parity-campaign`.

## Provenance and maintenance

Compiled 2026-07-10 from git archaeology + live verification on `port` @
`bc2bd96`. Re-verify the load-bearing rows:

```bash
grep -n "rne_postconstraint" gym/envs/base/mujoco_*backend*.py    # contact-force call still present?
grep -n "fusestatic" gym/envs/base/mujoco_backend_base.py         # merged from jt/port yet?
grep -n "root_states" gym/envs/base/mujoco_warp_backend.py        # staleness fixed yet? update table if so
```
