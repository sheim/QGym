---
name: q2-research-methodology
description: The discipline that turns a hunch into an accepted result in Q2 — the evidence bar (one mechanism explaining all observations, adversarial refutation, predict-numbers-before-running), first-principles proof recipes (energy audits, lockstep comparison, per-term reward decomposition, seed discipline) each with a worked example from this repo's history, the idea lifecycle from branch to merge-or-retirement, and the grounded open-problem list. Load when investigating anomalies, evaluating a tuning/algorithm claim, planning an experiment, or deciding whether a result is real. NOT a symptom table (q2-debugging-playbook) or the parity campaign itself (q2-phase4-parity-campaign).
---

# Q2 Research Methodology

## The evidence bar

A claim is accepted here when:

1. **One mechanism explains ALL observations — including the negatives.**
   Canonical case: warp `root_states` staleness. The getter-side-effect
   mechanism explains (a) frozen values during training, (b) why the one-line
   "fix" works, (c) why the assignment looks like a no-op, (d) why every
   contract test still passes (they re-call the property), AND (e) why CPU is
   unaffected (per-step sync). The jt/port commit stopped at "unknown why this
   works" (`2326b71`) — that is *below* the bar; a mechanism you can't state
   will be deleted by the next cleanup and the bug returns.
2. **It survived an assigned refutation attempt.** Before accepting, actively
   try to kill it: construct the experiment whose outcome would falsify the
   mechanism (see discriminators in q2-debugging-playbook). If you can't
   design a falsifying experiment, you don't have a mechanism yet.
3. **Numbers were predicted before running.** State expected values/direction
   first, then run. Example from the repo: the damped-pendulum tests encode
   predictions as tolerances (energy never rises > 1e-4 J/step; all envs end
   < 0.01 J; CPU-vs-warp lockstep < 1e-3 per step). Write your prediction in
   the PR/notebook before the run so hindsight can't edit it.

## Proof recipes (with in-repo worked examples)

- **Invariant audit** — find a conserved/dissipated quantity and test it.
  Worked example: `tests/unit_tests/test_mujoco_cpu_physics.py` (energy
  monotonicity + convergence). Use for: any new physics config or backend.
- **Lockstep twin experiment** — run two implementations step-for-step from
  identical state; assert bounded divergence. Worked example:
  `test_cross_backend_physics.py` (CPU vs warp, 2000 steps, 1e-3). Use for:
  backend changes, integrator/param changes.
- **Per-term reward decomposition** — the runner logs every registered
  `_reward_*` separately (console + wandb). A "better policy" claim must show
  WHICH terms moved. A term frozen at init is a smoking gun (that's how frozen
  observations manifest). Use for: any tuning claim.
- **State instrumentation** — jt/port's `SaveStates.py` (`8e2cfc7`, currently
  commented out on that branch): buffers chosen env states across training,
  dumps histograms + `saved_states.pt`. This tool exposed the root_states bug.
  Resurrect it rather than reinventing ad-hoc prints.
- **Profiling before optimizing** — both historical speedups came from
  measurement, not intuition: hardcoded coupling Jacobian (`b1bd4f2`, halved
  `_apply_coupling`), switch-per-decimation (`7296c6c`, ~10%). The `ea5cdff`
  reward refactor was *motivated* by making per-reward profiling possible.
- **Seed discipline** — `--seed N` sets python/numpy/torch (+CUDA) seeds;
  unset seeds are drawn randomly and stored in the cfg (recoverable from the
  log's config snapshot). CPU backend is deterministic per seed; warp is not
  guaranteed — determinism claims are CPU-only. Compare runs only with equal
  `batch_size` (rollout length auto-derives from it, `936a4cc`).

## Idea lifecycle (as this repo actually practices it)

1. **Branch** `<initials>/<topic>` off the canonical branch.
2. **Experiment** with honest `WIP:` commits; tuning commits state numeric
   before→after (`c5a598f` is the model citizen).
3. **Verdict** — one of:
   - *Merged*: via PR with reproduction instructions (e.g. `psdNetworks` →
     "working SAC on pendulum, ready for merge"; the humanoid branch merged
     after a year of divergence, `2022c82`).
   - *Retired*: explicit reverts, reasons in message (the horse-branch revert
     trio). Retirement is a *result* — record what was learned.
   - *Stalled*: the graveyard of `lm/GePPO`, `lm/IPG`, `linkedIPG`, `optuna`,
     `TrajBuf`… branches that just stopped. This is the failure mode to avoid:
     if you stop, write down why (one paragraph in the archaeology skill
     beats a dead branch).
4. Bug found on the way → fix + regression test + archaeology entry
   (`fba2deb` pattern).

## Where good ideas have come from here

Instrumentation (SaveStates → root_states bug), profiling (both speedups),
invariant tests (energy audits caught physics-config errors cheaply),
cross-robot generalization pressure (per-joint mean rewards), and porting
friction itself (the URDF-limits and cfrc_ext discoveries are reusable
MuJoCo knowledge now encoded in mujoco-backend-reference).

## Open problems worth a campaign (grounded, 2026-07-10)

For each: the asset this repo holds, first steps HERE, and a falsifiable
"you have a result when…".

1. **Trustworthy massively-parallel MuJoCo-warp RL.** mujoco-warp is young;
   public, tested recipes are scarce. Asset: the backend-agnostic contract-test
   harness (172 tests) — rare in this space. Steps: campaign Phases 0–2.
   Result when: CPU/GPU curves agree per-term at fixed seed/batch on
   mini_cheetah, and the suite pins it.
2. **Domain randomization under MuJoCo** (both plan checkboxes). Asset:
   explicit config axes + a backend seam to implement them per-world in warp.
   Steps: campaign Phase 3 discriminator, then per-world friction in warp
   (model fields are batched there in ways CPU MjModel is not). Result when: a
   test proves per-env friction differs and a train/eval friction-shift
   experiment shows the robustness delta.
3. **Humanoid on MuJoCo.** Registered (`humanoid`, `humanoid_running`, 18
   actuators, mass DR configured) but zero MuJoCo training runs exist. Asset:
   task code is backend-clean already (Phase 3l). Steps: smoke-train CPU 16
   envs; fix fused-body/limit issues surfaced; then GPU post-campaign-Phase-0.
   Result when: humanoid walks under `play_mujoco` teleop from a MuJoCo-trained
   checkpoint.
4. **Deployment export for MuJoCo-trained policies.** Asset:
   `export_network()` (TorchScript+ONNX) already exists; only the IsaacGym-era
   script wraps it. Steps: `scripts/export_mujoco.py` mirroring
   play_mujoco's checkpoint resolution; add onnx deps to pyproject. Result
   when: exported ONNX reproduces the torch policy's actions on recorded obs
   within the old regression tolerance (rtol 1e-5/atol 1e-7 — the standard the
   dead integration test used).
5. **Structured critics (QRCritics zoo) revival** — only if a human asks:
   it's someone's research program (PSD/SAC configs for pendulum exist);
   first step is fixing the `matfncs.py` missing-import bug and a pendulum
   SAC baseline re-run.

## When NOT to use this skill

- You just need to find/fix a bug → `q2-debugging-playbook`.
- You need the standards a change must pass to merge →
  `q2-conventions-and-change-control`.
- You're executing the parity work → `q2-phase4-parity-campaign`.

## Provenance and maintenance

Synthesized 2026-07-10 from git history (`port`, `jt/port`, research
branches), MIGRATION_PLAN.md, and the test suite. Volatile parts: the open
problems list (§ above) and branch-graveyard examples. Re-verify:

```bash
ls logs/ 2>/dev/null                                   # humanoid runs appeared? → update problem 3
git log --oneline port..origin/jt/port | head -3       # jt/port state
grep -rn "SaveStates" learning/ gym/ || echo "state-logging tool not merged"
```
