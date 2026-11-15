# Q-Intervened Height-Q Playback — Plan

Status: **COMPLETED AND ARCHIVED at user request — 2026-11-12. V1–V7 verified;
V8 manual viewer validation remains unverified.**

## Archive — 2026-11-12

Implementation committed in BigRedGym as `6507a13`. Final code-commit checks:
300 portable, 28 gym, 7 learning, and 32 Warp tests passed; Ruff clean.
These automated results do not establish interactive keyboard/viewer behavior.
The user requested archival with that evidence limitation retained. No todos
were created or updated.

Docs mapping at the time: centralized. Active plan: `.opencode/workspace/plans/`;
archive: `.opencode/workspace/plans/archive/` (both since moved into the product
repo under `docs/plans/`); journal:
`docs/journal.md`. `primary_pkg` is not needed.

Runtime model: `opencode-go/deepseek-v4.1-flash` (same model for subagents).

## 1. Goal

Add a maintained, single-environment MuJoCo CPU playback script that loads the
existing frozen `go2trot` PPO actor and the original fair-100 height-Q artifact,
then executes one of two per-decision Q interventions:

- `maxq`: draw eight stochastic policy actions, execute the one with highest Q;
- `rejection`: draw eight stochastic policy actions in fixed order, execute the
  first with `Q >= c`, otherwise execute the max-Q draw.

The mean action is **not** a candidate in either stochastic mode. Existing
`learning/offline/persistent_intervention.py` helpers must be reused, especially
`sample_rejection_candidates` + `select_rejection_action` for rejection.

The script must use the normal task reset path, start and reset with zero
commands, keep keyboard command control, and never let automatic command
resampling silently overwrite user commands. Auto-reset is off by default.

## 2. Baseline (read-only inventory)

BigRedGym (`BigRedGym/`):

- branch `bh574/final/dev/offline_RL`, HEAD `dfc506e449eedad5f57bb208e38b4dca69977e3e`
- no upstream configured for this branch; commits not on any remote:
  `dfc506e` (HEAD) and `1fb64bd` (baseline branch)
- dirty: 16 porcelain entries; 2 staged, 6 unstaged tracked, 19 untracked paths
- outstanding user work to preserve: `.gitmodules`, `genAI_skills/AGENTS.md`,
  `genAI_skills/README_MUJOCO.md`, `learning/offline/persistent_intervention.py`,
  `tests/TEST_CATALOG.md`, `tests/unit_tests/test_persistent_intervention.py`,
  untracked `docs/plans/`, `docs/thirdparty/`, and the threshold/report scripts
- submodule `thirdparty/CORL` at `eeeeef2`, dirty
  (`M algorithms/offline/iql.py`, `?? BRG_INTEGRATION.md`,
  `?? algorithms/offline/iql_smoke.py`); preserve, do not commit
- one worktree: the product checkout itself

`.opencode/` (separate repo):

- branch `course/MAE5110/BigRedGym`, HEAD `469bf24`, ahead 1 of
  `origin/course/MAE5110/BigRedGym`; 10 dirty entries; preserve.

Relevant `/tmp/opencode` inputs (read-only):

| Input | Path | Identity |
|---|---|---|
| Height-Q artifact | `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct` | fair-100, 1,000,000 transitions, 400,000 updates |
| Q weights | same dir `q.pt` | 322 KiB |
| Frozen actor | `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt` | sha256 `f88b1f9b…adf6` |
| Dataset fingerprint | artifact manifest | `6412a878…938cb` |
| Rejection experiment | `/tmp/opencode/height-q-rejection-threshold-20261111-v1` | 248 MiB; do not modify |

## 3. Discovered behavior (file anchors)

Current CLI (`uv run --frozen python scripts/play.py --help`, exit 0):

`--task` required; `--device cpu`; `--num_envs 32`; `--seed`;
`--experiment_name`; `--load_run`; `--checkpoint -1`; `--original_cfg`;
`--headless`; `--backend {mujoco,vsim}`; `--viewer_ui`;
`--keyboard/--no-keyboard`.

`scripts/play.py:78-130` (`setup`):

- sets `episode_length_s = 50`, `commands.resampling_time = 9999`,
  `push_robots.toggle = False`, `init_state.reset_mode = "reset_to_basic"`;
- builds env, builds an `OnPolicyRunner`, `switch_to_eval()`.

`scripts/play.py:133-187` (`play` + main):

- action comes from `runner.get_inference_actions()`, which is deterministic
  actor mean with clean observations (`learning/runners/on_policy_runner.py:327`);
- loop only does `env.step()` + `env.check_exit()`; **it never calls
  `env._reset_idx` or `env.reset_envs()`**, so a fall/timeout does not teleport
  the robot in current playback;
- keyboard and command visualizer are installed after env construction.

Keyboard (`gym/utils/interfaces/teleop_bindings.py:37-115`,
`gym/utils/interfaces/MujocoKeyboardInterface.py`):

- exact keys: `I/K` forward/back, `J/L` strafe, `N/M` yaw, `Up/Down` currently
  no-op TODO, `R` reset; MuJoCo exits via window close/Ctrl-C (the shared help
  text also lists `Esc`, but the MuJoCo adapter does not bind it; Vsim does);
- MuJoCo dispatches discrete key events through
  `env._backend._viewer_key_callback`;
- `TeleopCommands.__init__` seeds **forward 1.0**, and overwrites
  `cfg.commands.resampling_time` with `max_episode_length_s + 1` (51 s here);
- `TeleopCommands.apply("reset")` sets `env.timed_out[:] = True` then calls
  `env.reset()`, which resamples commands through `_reset_idx`; nothing zeroes
  them afterward.

Reset/task path:

- `gym/envs/base/task_skeleton.py:49-65`: `reset()` fills `to_be_reset`,
  calls `_reset_idx`, then one `step()`, then zeroes `episode_length_buf`;
  `step()` starts with `_reset_buffers()` clearing `terminated`/`timed_out`;
- `gym/envs/base/legged_robot.py:49-62, 92-131`: `_post_decimation_step`
  resamples commands on the resampling mask; `_reset_idx` resets system, base
  observations, commands, buffers;
- `gym/envs/go2/go2trot.py:116-173`: reset also randomizes gait phase/frequency,
  zeros `dof_pos_target`, refreshes gait/phase observations;
- `learning/runners/BaseRunner.py:88-93`: original training auto-reset mask is
  `timed_out | terminated` via `env._reset_idx`, with no command zeroing.
- Consequence: with auto-reset off, a fall is a one-step `terminated` flag;
  the robot remains where it fell and simulation continues. No backend/task
  reset is bypassed.

Height-Q runtime contract (artifact manifest):

- obs dim 48, action dim 12, `n_policy_steps = 1`, `control_dt = 0.02`,
  gamma 0.99, reward 0 inside `0.32 ± 0.05 m` and -1 outside, absorbing failure;
- actor obs: `base_ang_vel`, `projected_gravity`, `commands`, `dof_pos_obs`,
  `dof_vel`, `dof_pos_target`, `phase_obs`, `phase_frequency`;
- no `base_height`, `base_lin_vel`, contacts, root state, or other critic-only
  measurements in Q inputs (no Q privilege);
- action name `dof_pos_target` in BRG policy units; `normalize_obs = false`;
  `smooth_exploration = false`; observation noise is the actor noise dict;
- `learning/offline/evaluation.py:40-76`: `load_policy_from_artifact` freezes the
  actor; `_policy_observation` adds artifact-spec noise to scaled task states,
  matching `BaseRunner.get_noise` semantics;
- `learning/offline/persistent_intervention.py:9-107, 176-204`: max-Q uses eight
  stochastic candidates (`candidate0` is the first stochastic draw, not the
  mean); rejection uses eight stochastic candidates only and the verified
  first-feasible-else-max-Q helper. `rollout_threshold` is the deterministic
  mean+eight closest-mean controller and must **not** be used here.

Identity checks already confirmed while planning:

- checkpoint path exists and sha256 matches the artifact policy spec;
- artifact config fingerprint matches current `go2trot` config **after**
  `task_registry.convert_frequencies_to_params` (before playback-only
  overrides), with both default and `--original_cfg` configs.

## 4. Proposed defaults / choices

| Choice | Proposed default | Meaning |
|---|---|---|
| Controller mode | `--mode maxq` | No threshold needed; choose max-Q of eight stochastic draws. |
| Rejection threshold | required only for `rejection` | `Q >= c` first feasible draw; else max-Q draw; mean never a candidate. |
| Initial commands | `[0, 0, 0]` | `vx`, `vy`, `yaw_rate` all zero; keyboard changes only. |
| Command resampling | disabled | Set a practically infinite task resampling time; keyboard is authoritative. |
| Manual `R` reset | normal `env.reset()` then zero commands | Keeps task/backend reset; no immediate run-away from camera. |
| Auto reset | `--auto_reset off` | No timeout/fall teleport; robot stays fallen and simulation continues. |
| Other auto modes | `termination`, `timeout`, `both` | `both` is the original `BaseRunner` mask `terminated | timed_out`; all zero commands after reset. |
| Observation input | artifact noisy policy observation | Matches training/Q distribution; no separate normalizer. |
| Device/envs | `cpu`, exactly 1 env | Passive viewer and single robot; no batch playback in V1. |
| Episode horizon | task config (5 s) | Do not copy `play.py`'s 50 s override; only matters when timeout auto-reset is enabled. |
| Pushes | preserve task config (enabled) | Faithful to artifact training; optional `--no-push_robots` escape hatch if approved. |
| Camera | task config `viewer.pos/lookat` | Same view; no follow camera or camera changes. |
| Viewer UI | hidden by default | Same collision-avoidance reason as `play.py`; `--viewer_ui` opt-in. |
| Warp/VSim | out of scope V1 | CPU viewer + CPU headless only; explicit error for non-CPU. |

## 5. Proposed CLI

```text
scripts/play_height_q.py
  --task go2trot                     # required registry task (go2trot expected)
  --artifact PATH                    # required HeightQ directory (manifest.json + q.pt)
  --checkpoint PATH                  # default: artifact policy_spec.checkpoint
  --mode {maxq,rejection}            # default: maxq
  --threshold C                      # required iff rejection; finite float
  --device cpu                       # default: cpu; V1 rejects cuda/vsim
  --num_envs 1                       # default: 1; must be 1 in V1
  --seed INT                         # default: none -> random like play.py
  --headless                         # no viewer; requires --steps
  --steps N                          # positive; default 0 = run until viewer closes/Ctrl-C
  --auto_reset {off,termination,timeout,both}   # default: off
  --keyboard / --no-keyboard         # default: keyboard on
  --viewer_ui                        # show MuJoCo side panels (default hidden)
  --original_cfg                     # load saved run config; fingerprint still checked
  --experiment_name NAME             # optional with --original_cfg
  --load_run RUN                     # optional with --original_cfg
  --no-push_robots                   # optional playback-only override (if approved)
```

Exact proposed viewer command:

```bash
cd BigRedGym
uv run --frozen python scripts/play_height_q.py \
  --task go2trot \
  --artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --mode rejection --threshold -1.776 \
  --device cpu --auto_reset off
```

Exact proposed headless smoke:

```bash
cd BigRedGym
set -o pipefail
uv run --frozen python scripts/play_height_q.py \
  --task go2trot \
  --artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --mode maxq --device cpu --headless --steps 5 --seed 0 \
  2>&1 | tee /tmp/opencode/q-playback-smoke-20261111.log
echo "exit=${PIPESTATUS[0]}"
```

Run foreground only; no detached jobs.

## 6. Runtime design

New `learning/offline/height_q_playback.py`:

- `HeightQPlaybackController`: owns `FrozenTrotPolicy`, `HeightQ`, mode,
  threshold, and a caller-owned `torch.Generator`; method `act(obs)` returns
  action and small diagnostics (`indices`, `fallback`, `selected_q`, `gain`).
- `validate_playback_contract(env_cfg, train_cfg, artifact, spec, checkpoint)`:
  fail-fast checks below.
- Reuse exactly:
  - max-Q: `sample_candidates` + `select_action(..., critic=artifact)`;
  - rejection: `sample_rejection_candidates` + `select_rejection_action`;
  - no `sample_candidates_with_mean`, no `rollout_threshold`, no mean candidate.

New `scripts/play_height_q.py`:

1. Parse args; import `gym.envs`; `get_cfgs` (optionally `original_cfg`);
   `convert_frequencies_to_params`; set `env_cfg.env.num_envs = 1`.
2. Validate identity **before** any playback override:
   - `HeightQ(artifact)` loads format/schema/return convention; Q obs/action
     widths equal manifest;
   - checkpoint path exists and `sha256_file` equals
     `artifact.manifest["policy_spec"]["checkpoint_sha256"]`;
   - `replay_config_fingerprint(class_to_dict(env_cfg))` equals
     `artifact.manifest["dataset"]["config_fingerprint"]`;
   - `spec.obs_names == train_cfg.actor.obs`; `spec.obs_dims` equal env state
     widths; `spec.action_names == train_cfg.actor.actions`; action width equal
     env action width;
   - `spec.noise == class_to_dict(train_cfg.actor.noise) ==
     spec.actor_cfg["noise"]`;
   - `spec.actor_cfg["normalize_obs"] is False`,
     `train_cfg.actor.normalize_obs is False`, `spec.n_policy_steps == 1`,
     `spec.smooth_exploration is False`, action convention is
     `brg_policy_units`, `spec.control_dt == env_cfg.control.ctrl_dt`;
   - no privileged Q input names (`base_height`, `base_lin_vel`, contact/root
     state names) in `spec.obs_names`.
3. Build env through `task_registry.make_env(..., headless=..., backend="mujoco")`
   with the **normal** task reset path (constructor already calls `reset()`).
4. Load policy via `load_policy_from_artifact(artifact, checkpoint, device)`;
   freeze actor; do not build an optimizer/critic runner; do not write artifacts.
5. Zero `env.commands`; set `env.cfg.commands.resampling_time` to a huge finite
   value (not infinity, which breaks `int(resampling_time / dt)`).
6. Install keyboard with zero initial command, zero-on-reset, and preserved
   resampling time. Install `CommandVisualizer`; camera untouched.
7. Loop:
   - build policy observation with `_policy_observation(env, spec..., obs_gen)`;
   - `controller.act(obs)`; `env.set_states(spec.action_names, action)`;
   - `env.step()`;
   - if `auto_reset != off`, compute mask and call `env._reset_idx(mask)`, then
     zero commands (normal reset path; no direct backend state writes);
   - stop on `--steps`, viewer close/Ctrl-C, or `env.check_exit()`.
8. Print startup identity summary and final `steps`, `resets`, mode, threshold,
   final command; exit non-zero on any validation failure.

`TeleopCommands` extension (default behavior preserved for `play.py`/Vsim):

- optional `initial_commands`; default remains `[1, 0, 0]` seeding;
- optional `reset_commands`; default `None` leaves existing reset behavior;
- optional `preserve_resampling_time`; default `False` keeps the current
  `max_episode_length_s + 1` assignment.
- `MujocoKeyboardInterface(env, commands=None)`; default constructs
  `TeleopCommands(env)` exactly as today.

## 7. Files to edit / add

| File | Change |
|---|---|
| `BigRedGym/scripts/play_height_q.py` | New single-env CPU playback script, CLI, loop, validation, auto-reset helper. |
| `BigRedGym/learning/offline/height_q_playback.py` | New frozen-policy/Q controller and contract validation; reuses verified helpers. |
| `BigRedGym/gym/utils/interfaces/teleop_bindings.py` | Optional zero initial commands, zero-after-reset, preserve resampling; defaults unchanged. |
| `BigRedGym/gym/utils/interfaces/MujocoKeyboardInterface.py` | Optional injected `TeleopCommands`; default unchanged. |
| `BigRedGym/tests/unit_tests/test_height_q_playback.py` | Controller, selection, validation, auto-reset mask tests. |
| `BigRedGym/tests/unit_tests/test_teleop_keyboard.py` | New zero/reset/preserve cases; existing behavior protected. |
| `BigRedGym/tests/TEST_CATALOG.md` | Add playback test coverage; preserve current user edits. |
| `BigRedGym/genAI_skills/README_MUJOCO.md` | Document new playback CLI; preserve current user edits. |
| `BigRedGym/genAI_skills/.agents/skills/q2-train-and-evaluate/SKILL.md` | Add `play_height_q.py` playback entry and evidence boundary. |
| `BigRedGym/docs/offline_height_q.md` | Document runtime modes, reset semantics, artifact identity checks. |
| `docs/plans/q_intervened_playback_plan.md` | This plan. |
| `docs/journal.md` | Minimal plan journal entry. |

## 8. Verification

V1 — CLI/parser only (no sim):

```bash
uv run --frozen python scripts/play_height_q.py --help
```

Expect exit 0 and the flags/defaults above; `--mode maxq`, `--auto_reset off`,
`--device cpu`, `--num_envs 1`, keyboard enabled.

V2 — Controller max-Q unit test (fake policy/Q, no sim): eight stochastic
candidates; `candidate0 == mean + std*eps0`, not the mean; selected index is
`argmax(Q)`; output shape `[1, 12]`; finite.

V3 — Rejection unit test (fake policy/Q): eight candidates only; mean excluded;
first `Q >= c` selected; all-infeasible falls back to max-Q; threshold NaN
rejected; candidate count 8 enforced; no `sample_candidates_with_mean` path.

V4 — Keyboard/reset fake-env tests: new commands start at zero; `I/K/J/L/N/M`
still move the expected axes; `R` invokes normal `reset()` and zeroes commands;
command resampling time is preserved/disabled; existing default `play.py` seeding
still tested.

V5 — Auto-reset mask fake-env tests: `off` returns no mask and never calls
`_reset_idx`; `termination` resets only falls; `timeout` resets only timeout;
`both` equals `terminated | timed_out`; every reset path zeroes commands.

V6 — Identity/spec failure tests: wrong checkpoint hash, missing actor state,
wrong obs/action/noise/normalization/action convention, privileged Q input,
wrong dataset fingerprint, and mode/threshold misuse each raise a clear error;
no fallback substitution.

V7 — CPU headless smoke with the real fair-100 artifact and `model_600`:
the exact command in §5 with `--steps 5 --seed 0`; expect exit 0, finite actions,
`steps=5`, `resets=0`, zero final command, and a saved foreground log at
`/tmp/opencode/q-playback-smoke-20261111.log`. This proves wiring only, not Q
accuracy or policy improvement.

V8 — Manual viewer validation (user only): run the viewer command and confirm
the camera view, keyboard keys, zero initial command, `R` zeroing, and no
automatic resampling. This subagent cannot open a GUI; do not claim V8 from
headless evidence.

Final implementation gate (after approval, not now):

```bash
uv run --frozen python -m pytest tests/unit_tests/test_height_q_playback.py -q
uv run --frozen python -m pytest tests/unit_tests/test_teleop_keyboard.py -q
uv run --frozen python -m pytest -q
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
uv run --frozen ruff format <touched-python-files>
uv run --frozen ruff check .
```

## 9. Constraints / non-goals

- No product edits until approval; no commit; no `todos.md` edit; no plan
  archival.
- Preserve all current dirty threshold/report relocation work and the dirty
  CORL submodule.
- No new Q artifact, no actor fine-tuning, no Q re-fit, no new checkpoint.
- No hardware deployment, no VSim, no Warp/VSim playback in V1. Warp headless is
  a possible follow-up only if separately approved.
- No camera follow changes; no changes to `scripts/play.py` defaults.
- No claim of Q accuracy, policy improvement, or viewer behavior from headless
  smoke.

## 10. Open questions (non-blocking with proposed answers)

1. **Observation noise**: proposed on, using artifact `policy_spec.noise`, to
   match the Q training distribution. If you want clean-observation playback,
   that is an explicit future flag, not the default.
2. **Pushes**: proposed preserving the task config (enabled for `go2trot`), with
   optional `--no-push_robots` if approved. Existing `play.py` disables pushes;
   this new script would differ deliberately for Q-distribution fidelity.
3. **Timeout horizon**: proposed task config (5 s), not `play.py`'s 50 s; only
   observable when `--auto_reset timeout` or `both` is enabled.
4. **Default mode**: proposed `maxq`; use `--mode rejection --threshold C` for
   the threshold controller.

## 11. Execution and results (2026-11-11)

Status: **implementation complete; V1-V7 pass. V8 manual viewer validation is
still required from the user and is not claimed from headless evidence.**

### Files changed

| File | Change |
|---|---|
| `BigRedGym/learning/offline/height_q_playback.py` | New controller, strict artifact/task identity gates, live width checks, instance-local command-resampling hold, auto-reset masks/zeroing. |
| `BigRedGym/scripts/play_height_q.py` | New single-env MuJoCo CPU playback CLI and loop. Rejects non-CPU, VSim, and `num_envs != 1`; headless requires `--steps`; zero commands at startup/reset; optional auto-reset. |
| `BigRedGym/gym/utils/interfaces/teleop_bindings.py` | Backward-compatible optional zero initial commands, zero-after-reset commands, and preserved resampling time. Defaults unchanged. |
| `BigRedGym/gym/utils/interfaces/MujocoKeyboardInterface.py` | Optional injected `TeleopCommands` and deferred event queue. Default immediate dispatch unchanged. |
| `BigRedGym/tests/unit_tests/test_height_q_playback.py` | Controller/selection, CLI rejection, identity failures, live widths, resampling hold, and auto-reset tests. |
| `BigRedGym/tests/unit_tests/test_teleop_keyboard.py` | Manual-mode options, default regression, and deferred-key tests. |
| `BigRedGym/tests/TEST_CATALOG.md` | Playback and expanded teleop coverage. |
| `BigRedGym/genAI_skills/README_MUJOCO.md` | Playback commands, keys, reset semantics, support boundary. |
| `BigRedGym/genAI_skills/.agents/skills/q2-train-and-evaluate/SKILL.md` | Playback entry and evidence boundary. |
| `BigRedGym/docs/offline_height_q.md` | Runtime modes, reset semantics, identity gates, evidence boundary. |
| `docs/plans/q_intervened_playback_plan.md` | This status/results section. |
| `docs/journal.md` | Result entry link. |

### Verification

```text
V1  scripts/play_height_q.py --help                     exit 0
V2  max-Q controller test                               pass
V3  rejection/order/shared-helper tests                 pass
V4  keyboard/default/manual/deferred tests              pass
V5  auto-reset mask/zeroing tests                       pass
V6  identity and width failure tests                    pass
V7  CPU headless real-artifact smoke, 5 steps          exit 0
    /tmp/opencode/q-playback-smoke-20261111.log
    steps=5 resets=0 final_commands=[0.0, 0.0, 0.0]
    artifact fingerprint 6412a8784c88, pushes=True, seed=0
```

Additional real-artifact CPU smokes (all exit 0, foreground, one env):

```text
rejection --threshold -1.776, 5 steps   fallbacks=2, final_commands=zero
--original_cfg --load_run Nov10_...      fingerprint match, 2 steps
--auto_reset both, 260 steps             resets=1 at the 250-step/5 s timeout
logs: /tmp/opencode/q-playback-*-smoke-20261111.log
```

Full gate run from `BigRedGym/` (`/tmp/opencode/q-playback-final-gate-20261111.log`):

```text
uv run --frozen python -m pytest -q        292 passed, 97 deselected
uv run --frozen python -m pytest gym -q     28 passed
uv run --frozen python -m pytest learning -q     7 passed
uv run --frozen ruff check .               All checks passed
git diff --check                           clean
```

Post-review rerun: 43 playback tests passed, 2-step real-artifact headless
smoke exited 0; no playback processes remained.

### Deviations / implementation notes

- Command resampling is disabled with an instance-local `_resample_commands`
  override rather than a large `resampling_time`, per the user requirement.
  Shared task/teleop defaults are untouched; normal training and `play.py`
  retain resampling.
- The MuJoCo key callback runs on the viewer UI thread. Playback uses
  `MujocoKeyboardInterface(..., deferred=True)` to queue key actions and applies
  them, including `R` reset, on the task thread. Default immediate dispatch is
  unchanged for existing callers.
- `--backend {mujoco,vsim}` was added so unsupported VSim playback fails with a
  clear message; only `mujoco` is accepted. `--device` must be exactly `cpu` and
  `--num_envs` exactly one.
- Artifact scales are compared with a float32-round-trip tolerance because JSON
  records 0.3 as 0.30000001192092896. Noise/config dictionaries are exact.
- `--no-push_robots` is implemented as the approved playback-only escape hatch;
  pushes remain enabled by default from the task config.
- Live observation/action tensor widths are checked after environment
  construction (`validate_env_widths`) in addition to config-level validation
  before playback overrides.

### Remaining manual gate (user only)

```bash
cd BigRedGym
uv run --frozen python scripts/play_height_q.py \
  --task go2trot \
  --artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct \
  --checkpoint logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --mode maxq --device cpu --auto_reset off
```

Confirm camera view, zero initial command, held commands without automatic
resampling, `I/K/J/L/N/M` axes, `R` reset with zero commands, viewer close, and
clean Ctrl-C exit. Optionally repeat with `--mode rejection --threshold -1.776`.
