---
name: q2-build-and-env
description: Recreate the Q2 development environment from scratch — uv, Python versions, mujoco/mujoco-warp, GPU vs CPU vs macOS, and the known install traps (mjpython on macOS, warp JIT delay, legacy venvs, dual packaging debt). Load for any environment/setup/install problem, ImportError, "works on my machine", or CI-vs-local mismatch. NOT for running training (q2-run-and-train) or test invocation details (q2-testing-and-validation).
---

# Q2 Build & Environment

## Canonical setup (all platforms)

```bash
cd Q2
uv sync        # creates .venv from pyproject.toml + uv.lock
```

That is the whole install. Facts behind it (verified 2026-07-12):

- Python: `requires-python = ">=3.11,<3.14"` (`pyproject.toml`), pinned to
  **3.11** by `.python-version` (2026-07-12; the vsim/vlearn wheels are cp311
  only — full suite revalidated identical on 3.11). Ruff targets py311.
- **vsim backend extra**: fully self-contained in-repo since 2026-07-21 —
  `uv sync --extra vsim` installs the `vlearn` wheel from `vendor/vlearn/`
  (gitignored drop zone; see its README for the new-machine procedure).
  Needs system `libczmq4`; every vsim process must start via
  `uv run --env-file .env.vsim` (LD_LIBRARY_PATH is read by the loader at
  process start). License mechanics (verified): `License.key` is looked up
  in the process CWD (a gitignored root symlink handles this);
  `TurboActivate.dat` via `VL_WORKING_DIRECTORY`.
  ⚠️ 2026-07-21: vlearn 0.3.12 vendored, but the key does NOT activate it
  (same key node-locked 0.3.5 in April; trial spent; servers reachable) —
  **vendor-side fix pending**; all vsim runtime is blocked until then.
- Runtime deps: `mujoco>=3.6`, `torch`, `tensordict`, `numpy`, `pygame`, `mss`,
  `wandb`, `pandas`, `matplotlib`.
- The `dev` dependency group (installed by default with `uv sync`) includes
  `mujoco-warp>=3.6`, `pytest`, `ruff`, `pre-commit` — so **a plain `uv sync` on
  Linux already gives you the GPU backend**. There is also a `gpu` extra
  (`uv pip install -e '.[gpu]'`) but you normally don't need it.
- Install pre-commit hooks once per clone: `uv run pre-commit install`
  (hooks: ruff-format, ruff, check-merge-conflict, check-added-large-files 100 KB).

## Verify the environment

```bash
uv run python -c "import mujoco, torch; print(mujoco.__version__, torch.__version__, torch.cuda.is_available())"
uv run python -c "import importlib.util as u; print('warp:', u.find_spec('mujoco_warp') is not None)"
uv run python -m pytest tests/unit_tests/ -q     # expect ~151 passed, 21 skipped in ~30 s
```

The 21 skips are by design (CPU-backend contract tests parametrized over a
`cuda:0` device param). With CUDA present, warp tests genuinely run.

## Platform matrix

| Platform | Backend | Notes |
|---|---|---|
| Linux + NVIDIA | `--device cuda:0` → MuJocoWarpBackend | viewer NOT available with warp — use `--headless` |
| Linux CPU | `--device cpu` → MuJocoCPUBackend | viewer works |
| macOS | always MuJocoCPUBackend | viewer needs `mjpython` (trap below) |

## Known traps

**T1 — macOS GUI viewer needs `mjpython` + a dylib-bearing Python.**
MuJoCo's passive viewer on macOS must run under `mjpython` (ships with the
`mujoco` pip package), and `mjpython` dlopens `libpython3.13.dylib`, which uv's
bundled Python does not ship. Recipe (from README_MUJOCO.md, added in `bb8c981`):

```bash
brew install python@3.13
uv venv --python /opt/homebrew/opt/python@3.13/bin/python3.13
uv sync
.venv/bin/mjpython scripts/train_mujoco.py --task mini_cheetah --device cpu --num_envs 64
```

Or skip the viewer with `--headless`. The CPU backend raises a RuntimeError with
these exact instructions if a viewer is requested without mjpython
(`gym/envs/base/mujoco_cpu_backend.py:153-161`).

**T2 — first warp/GPU run "hangs": that's kernel JIT compilation.**
The first `mjw.forward` triggers warp module compilation/loading; with 4096 envs
this can sit silent for minutes before the first iteration prints. Evidence: the
2026-07-10 aborted runs died with `KeyboardInterrupt` inside
`warp/_src/context.py … kernel.module.load` — someone Ctrl-C'd during compile.
Wait it out; subsequent runs reuse the warp kernel cache.

**T3 — legacy paths that look real but are dead in this env:**
- `requirements.txt` + `setup.py` (package "QGym", depends on `isaacgym`) are the
  pre-port pip path. CI on main/dev still uses them; local dev must not.
- `scripts/train.py`, `scripts/play.py`, `scripts/export_policy.py` require
  IsaacGym (Python 3.8, its own venv, not present on this machine). Symptom of
  hitting them: `AttributeError: 'NoneType' object has no attribute
  'parse_arguments'` from `gym/utils/helpers.py` (the isaacgym import guard set
  `gymutil = None`).
- `.venv311/` is a leftover Python 3.11 env from early warp bring-up (Apr 2026).
  The primary env is `.venv` (3.13). Don't resurrect `.venv311` without reason.
- Two egg-info dirs (`QGym.egg-info`, `q2.egg-info`) are build artifacts of the
  old and new package names; both untracked noise.

**T4 — packaging debt (matters only if you build a wheel):**
`[tool.setuptools] packages` in `pyproject.toml` omits `gym.envs.pendulum` and
`gym.envs.cartpole` — editable/`uv run` works (repo root on path), a built wheel
would silently lack those tasks. Also `onnx`/`onnxruntime` are only in the legacy
`requirements.txt`, so the (IsaacGym-only) export script has no deps under uv.
Both are known, unfixed as of 2026-07-10.

**T5 — benign warning:** pygame emits a `pkg_resources is deprecated` UserWarning
on import. Ignore it; it appears in every pytest run.

## When NOT to use this skill

- Training command anatomy, logs, wandb → `q2-run-and-train`.
- Test suite structure and CI reality → `q2-testing-and-validation`.
- Why a *running* sim behaves wrongly → `q2-debugging-playbook`.

## Provenance and maintenance

Verified 2026-07-10 on Linux, RTX 4080, `port` @ `bc2bd96`. Re-verify:

```bash
cat .python-version pyproject.toml | head -30
uv run python -m pytest tests/unit_tests/ -q | tail -1     # pass/skip counts drift with new tests
grep -n "packages" pyproject.toml                          # T4 still missing pendulum/cartpole?
ls scripts/                                                # legacy scripts still present?
```
