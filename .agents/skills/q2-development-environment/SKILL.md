---
name: q2-development-environment
description: Set up, repair, and verify Q2's uv-managed .venv across Linux CPU, Linux NVIDIA GPU, macOS, and the optional licensed VSim backend. Use for fresh checkouts, dependency or Python problems, import failures, platform mismatches, CUDA/Warp setup, VSim loader or license failures, and local-versus-CI discrepancies.
---

# Q2 Development Environment

Read `AGENTS.md`, `.python-version`, `pyproject.toml`, `uv.lock`, and the
relevant setup section of `README_MUJOCO.md` before changing dependencies.
Treat `requirements.txt` and `setup.py` as the separate legacy IsaacGym stack.

## Establish the target

- Use uv to create and manage the repository's `.venv`.
- Use Python 3.11 for the common tested environment and VSim wheel
  compatibility. The package metadata permits Python 3.11 through 3.13.
- Use MuJoCo CPU for portable/headless development and the Linux CPU viewer.
- Use MuJoCo Warp only with a working NVIDIA driver and CUDA-visible PyTorch.
- Use VSim only when the machine has the local wheel, system library, node
  license, and CUDA. Never replace a requested backend with another silently.
- Keep IsaacGym commands in their Python 3.8 environment; do not merge that
  dependency set into the modern uv environment.

## Create or update the environment

Use the locked dependency graph for ordinary development:

```bash
uv sync --frozen
uv run --frozen python -c "import mujoco, torch; print(mujoco.__version__, torch.__version__)"
uv run --frozen python -m pytest -q
```

Use `.venv/bin/python` directly only when a tool must bypass uv resolution.

For MuJoCo Warp:

```bash
uv sync --frozen --extra gpu
uv run --frozen python -c "import torch, mujoco_warp; print(torch.cuda.is_available())"
```

Require the final command to print `True`. If it does not, inspect
`nvidia-smi`, the PyTorch build, and device visibility; do not add a CPU
fallback.

For VSim, follow `thirdparty/vlearn/README.md` exactly. Install the local wheel
with `uv sync --locked --extra vsim`; start every process with
`uv run --env-file .env.vsim ...`; validate using
`bash scripts/run_vsim_tests.sh`. Do not print or inspect license contents.

## Diagnose by boundary

- Resolution/install failure: compare `pyproject.toml`, `uv.lock`, local wheel
  presence, and exact uv flags. Do not casually regenerate the lockfile.
- Python mismatch: check `uv run --frozen python --version` and
  `sys.executable` before debugging imports.
- Warp import but no CUDA: confirm driver visibility and the installed torch
  build separately.
- Long first Warp step: distinguish kernel compilation from a deadlock using
  the traceback and GPU activity before interrupting it.
- VSim loader errors: environment variables must exist before process start;
  in-process changes to `LD_LIBRARY_PATH` are too late.
- VSim license errors: verify paths and activation state without reading,
  logging, or committing secrets. License repair may require the vendor.
- macOS viewer errors: use `mjpython` with the Homebrew-Python recipe in
  `README_MUJOCO.md`, or validate headless behavior.
- CI mismatch: inspect `.github/workflows/` directly. The current workflow
  still uses the legacy pip path and is not the modern local correctness gate.

## Change dependencies

Explain why a dependency belongs in the modern core, `gpu` extra, `vsim`
extra, or dev group. Update `pyproject.toml` and the lockfile together, then
test a clean sync appropriate to every affected platform. Update
`README_MUJOCO.md` when setup steps or supported versions change.
