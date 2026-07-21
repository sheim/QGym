---
name: q2-conventions-and-change-control
description: Q2's house style, non-negotiables with their rationale and originating incidents, branch topology and merge state, commit/PR conventions, gates every change must pass, and the docs of record (MIGRATION_PLAN.md, README_MUJOCO.md) with maintenance rules. Load before committing, opening a PR, restructuring code, editing the plan/READMEs, or deciding which branch to base work on. NOT a test manual (q2-testing-and-validation) and NOT setup (q2-build-and-env).
---

# Q2 Conventions & Change Control

## Non-negotiables (each with its why)

1. **No `try/except` in dev code — fail fast and obviously.**
   Stated in MIGRATION_PLAN.md "Style Guidelines". Rationale: research code
   with silent fallbacks produces wrong *results*, not crashes — the worst
   failure mode. Historical enforcement: `ebc2925` deliberately REMOVED a
   try/except that silently fell back warp→CPU backend.
   Sanctioned exceptions (grandfathered, removed in Phase 4): the
   `ImportError` guards around `isaacgym` imports and the per-import guards in
   `gym/envs/__init__.py` — these gate *optional dependency presence*, never
   behavior.
2. **Ruff, always, at session end**: `uv run ruff format . && uv run ruff
   check .` (line length 88, py311 target, E722 ignored — see pyproject).
   Pre-commit runs ruff-format + ruff + merge-conflict + large-file hooks;
   install with `uv run pre-commit install`.
3. **No files > 100 KB** — enforced twice (pre-commit `check-added-large-files
   --maxkb=100`, CI `basic_checks.yml` on every push). Origin: robot
   meshes/logs bloating history. Side effect to know: the regression suite's
   reference tensor was never committed, killing that suite — don't "fix" a
   gate by committing binaries; store them outside git.
4. **uv is the only environment manager**; `logs/`, `user/wandb_config.json`,
   `*.pt`, `wandb/` are gitignored — never force-add.
5. **Debug flags don't ship**: `disable_actions`, `disable_gravity` etc. leaked
   into configs once inside a WIP commit (`bfd6b2d`, cleaned in `4538b5e`).
   Before committing, `grep -rn "disable_" gym/envs/*/…config*.py` and justify
   every `True`.
6. **One concern per commit.** `b1bd4f2` mixed a legitimate optimization with
   silent hyperparameter changes — that cost archaeology time. Tuning commits
   say what changed numerically (good example: `c5a598f` lists old→new
   weights).

## Branch topology (state as of 2026-07-10 — re-verify, it moves)

| Branch | Meaning |
|---|---|
| `port` (= `origin/port`, HEAD `bc2bd96`) | canonical IsaacGym→MuJoCo migration branch; base new work here unless told otherwise |
| `origin/jt/port` | JoshuaTchou's continuation, 13 commits ahead. Steve merges it himself, later (decision 2026-07-10). Already pulled into `vsim` (2026-07-11, per Steve): reward/scaling tuning, fusestatic, mjspec config mechanism (+ njmax put_data forwarding fix, values retuned 90→200). Still exclusive to jt/port: SaveStates state-logging tool, the root_states band-aid line (DELETE at merge — superseded by the proper fix), stray `info.txt`. Expect merge conflicts in `mujoco_warp_backend.py` + mini_cheetah configs; vsim's versions are the correct ones (jt/port's njmax route is a silent no-op) |
| `main`, `dev` | pre-port legacy; CI targets these only |
| `origin/<initials>/<topic>` (yl/, lm/, js/, al/, jt/…) | per-person research branches; many abandoned — consult q2-failure-archaeology before mining them |

Naming convention for new branches: `<initials>/<topic>`.

## Gates for a change (run through all that apply)

1. `uv run python -m pytest tests/unit_tests/ -q` green before AND after
   (CI will not save you — it runs nothing on `port`; see
   q2-testing-and-validation).
2. Touched a backend's step/reset/state path → the contract tests are the
   spec: extend them with the invariant your change relies on.
3. Fixed a bug → regression test lands in the SAME change (house pattern:
   `fba2deb`), and the battle gets an entry in `q2-failure-archaeology`.
4. Changed behavior that affects training results (rewards, physics params,
   frequencies) → record before/after curves (see q2-research-methodology);
   never bury it in an unrelated commit.
5. Completed/started a migration phase or discovered a new gotcha → update
   `MIGRATION_PLAN.md` in the same PR (it has a "Notes and known gotchas"
   section and per-phase ✅ marks — keep them truthful).
6. Ruff + no new files > 100 KB + no debug flags left on.

## Commit & PR conventions

- Commit messages: short imperative summaries, no conventional-commits prefixes
  (match `git log` house style). `WIP:` prefix is used honestly — expect
  follow-up cleanup commits.
- PRs use `.github/pull_request_template.md`: issue link, summary with WHY,
  **instructions for reviewers including how to actually test** ("point out
  desired behavior, not just 'check that this appears'"), checklist (regression
  expectations, reviewer assigned, tests for core features, ruff run).
- Historical review flow: PRs into `dev`, periodic `dev`→`main` merges. The
  port branch has been pushed directly without PRs so far.

## Docs of record

| Doc | Role | Maintenance rule |
|---|---|---|
| `MIGRATION_PLAN.md` | technical plan + style rules + gotchas + phase status | update in the same change that alters status; never let ✅ marks lie |
| `README_MUJOCO.md` | user-facing quickstart/CLI/platform notes | update when CLI flags or platform recipes change |
| `README.md` | legacy stub pointing at pkGym lineage | leave alone until Phase 4 rebranding |
| `CLAUDE.md` + `.claude/skills/` | agent-facing knowledge | every skill ends with re-verification commands — run them before trusting volatile facts; update skills when they drift |

## When NOT to use this skill

- What/how to test → `q2-testing-and-validation`.
- Whether a cleanup is safe architecturally → `q2-architecture-contract`.
- Historical "why is it like this" → `q2-failure-archaeology`.

## Provenance and maintenance

Compiled 2026-07-10 from MIGRATION_PLAN.md, pyproject.toml,
.pre-commit-config.yaml, .github/, and git history of `port`/`jt/port`.
Re-verify:

```bash
git branch -a -v --sort=-committerdate | head -8      # topology drift
git log --oneline port..origin/jt/port | wc -l        # jt/port merged yet?
sed -n "/Style Guidelines/,/Architecture/p" MIGRATION_PLAN.md
cat .pre-commit-config.yaml
```
