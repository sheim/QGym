# Q2 — RL training for legged robots (IsaacGym → MuJoCo port)

Lineage: RSL `legged_gym` → MIT Biomimetics `pkGym`/`QGym` → this repo.
Branch **`port`** is the canonical migration branch (IsaacGym → MuJoCo backend swap).
`main`/`dev` predate the port. `MIGRATION_PLAN.md` is the plan of record: phases 0–3
complete, **Phase 4 (validation + IsaacGym removal) open**. `README_MUJOCO.md` is the
user-facing doc; `README.md` is a legacy stub. Further physics backends
(v-sim) are planned — new engines follow `q2-backend-integration`.

## Environment (uv only)

```bash
uv sync                                     # creates .venv (Python 3.11 — pinned for vsim cp311 wheels)
uv run python -m pytest tests/unit_tests/   # the real test suite (~30 s; vsim tests skip without opt-in)
uv run scripts/train_mujoco.py --task pendulum --device cpu --num_envs 256 --headless --disable_wandb
uv run scripts/play_mujoco.py --task mini_cheetah        # plays latest checkpoint, keyboard teleop on
# vsim backend (licensed GPU engine, ~10× warp throughput; needs .env.vsim):
uv run --env-file .env.vsim scripts/train_mujoco.py --task mini_cheetah --backend vsim --device cuda:0 --num_envs 4096 --headless
bash scripts/run_vsim_tests.sh              # vsim suite (opt-in, local-only)
```

Never use `pip install` / `requirements.txt` / `setup.py` — those are the legacy
IsaacGym-era path. `scripts/train.py`, `scripts/play.py`, `scripts/export_policy.py`
require IsaacGym (Python 3.8 venv, not present here) and will fail in this env.

## Non-negotiables

- **No `try/except` in dev code — fail fast and obviously.** The only sanctioned
  exceptions are the existing `ImportError` guards around `isaacgym` imports
  (Phase 0 policy, removed in Phase 4). Never add a fallback that silently
  degrades behavior.
- Run `uv run ruff format . && uv run ruff check .` at the end of every session.
- Never commit files > 100 KB (CI + pre-commit both enforce this).
- `logs/`, `user/wandb_config.json`, `*.pt` are gitignored — never force-add them.
- Do not run bare `pytest` from the repo root: `tests/integration_tests/` builds
  IsaacGym envs at session start and INTERNALERRORs on mere collection. Always
  target `tests/unit_tests/` (or `gym learning`) explicitly.

## Critical landmine — FIXED on this branch (2026-07-11)

The warp backend used to refresh `root_states`/`rigid_body_states` only inside
property getters, so GPU training ran on frozen base-state observations (and
reported inflated rewards — mini_cheetah ~6.2 fake vs ~1.1 real at iteration 30).
Fixed: `MuJocoWarpBackend._sync_assembled_states()` runs in `step()` and resets;
regression test `tests/unit_tests/test_task_state_liveness.py`. Consequences that
still stand: **all pre-fix GPU training results/checkpoints are untrustworthy**,
`origin/jt/port` carries a now-redundant band-aid line to drop at merge, and
branches without this fix still have the bug. Details:
`.claude/skills/q2-phase4-parity-campaign/SKILL.md` Phase 0.

**Sibling staleness — FIXED 2026-07-24 (branch `vsim`).** Same family, different
tensor: warp's `dof_state` getter returned a per-call `torch.stack([dof_pos,
dof_vel])` **copy**, but the task caches `self.dof_state` once at init
(`fixed_robot`/`legged_robot._init_buffers`), so on GPU that cached reference
froze at the init zeros. Latent for mini_cheetah (its rewards read
`dof_pos`/`dof_vel`, which stayed live) but **fatal for the pendulum**:
`_reward_equilibrium` reads `dof_state` directly, so `abs(dof_state)≈0` →
`sqrdexp(0)=1.0`, pegging equilibrium at its max on GPU. This faked the pendulum
GPU training reward (**~1.16 fake vs ~0.23 real**; the historical "+1.5" GPU
pendulum mark carries this asterisk). Fixed by making warp maintain an assembled
`_dof_state_t` refreshed in `_sync_assembled_states()` and returning a live view
(mirrors the root_states fix); `test_task_state_liveness.py` now also asserts
`dof_state` tracks `dof_pos`/`dof_vel`. CPU and vsim expose `dof_state` as a real
view and were never affected.

## Skills index (`.claude/skills/`)

| Skill | Load when |
|---|---|
| `q2-architecture-contract` | touching `gym/envs/base/*`, backends, state tensors, resets |
| `q2-build-and-env` | env setup, uv/python issues, macOS, GPU install |
| `q2-run-and-train` | running training/play, logs, wandb, checkpoints, teleop |
| `q2-config-system` | any config field question, adding/overriding config axes |
| `q2-testing-and-validation` | running/adding tests, what counts as evidence, CI |
| `q2-debugging-playbook` | any unexplained failure — check here before debugging |
| `q2-failure-archaeology` | before re-investigating anything that smells historical |
| `mujoco-backend-reference` | MuJoCo/mujoco-warp semantics as they apply here |
| `legged-rl-reference` | rewards, observations, PPO/SAC knobs, oscillators |
| `q2-task-authoring` | adding a robot or task end-to-end |
| `q2-backend-integration` | integrating a NEW physics engine (v-sim, Newton, …) |
| `q2-conventions-and-change-control` | style, branches, gates, docs of record |
| `q2-phase4-parity-campaign` | the live campaign: warp parity → IsaacGym removal |
| `q2-research-methodology` | turning a hunch into an accepted result here |
