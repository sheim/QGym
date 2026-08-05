# Q2 Developer Entry Point

Read `AGENTS.md` before changing this repository. It is the durable,
repository-wide developer guide and source of current architecture, style,
environment, and validation rules.

Q2 supports MuJoCo CPU, MuJoCo Warp, and optional licensed VSim. Use
`MIGRATION_PLAN.md` for current backend parity evidence and planned work.

## Common commands

Use the uv-managed `.venv` and locked dependency graph:

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
uv run --frozen python -m pytest -q -m warp
uv run --frozen ruff check .
```

Use `README_MUJOCO.md` for setup, training, playback, and optional VSim
commands. Cross-cutting backend, task, and full-stack tests belong under
`tests/unit_tests/`. Small deterministic implementation tests may live beside
their code under `gym/` or `learning/`, where they can also demonstrate local
usage; collect those explicitly with `pytest gym` or `pytest learning`.
Hardware-specific groups are selected explicitly.

## Current repository skills

Load procedural guidance from `.agents/skills/`:

| Skill | Use for |
|---|---|
| `q2-development-environment` | uv, Python, CUDA/Warp, macOS, or VSim setup |
| `q2-backend-development` | backend contracts, state/reset, contacts, routing |
| `q2-task-authoring` | robots, assets, configs, tasks, and registration |
| `q2-rl-development` | observations, rewards, algorithms, and normalization |
| `q2-train-and-evaluate` | training, resume, playback, and evaluation |
| `q2-testing-and-debugging` | test selection, regressions, and diagnosis |
