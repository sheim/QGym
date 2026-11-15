# Selectable replay field recording — record any subset of raw simulator state

Status: **COMPLETED AND ARCHIVED — 2026-11-13.** The recorder records the core
transition fields exactly as before and adds an opt-in `--replay_fields`
selection of raw simulator state. A 1M-transition snapshot with every supported
feature was collected from the trained `model_600` policy.

## Archive — 2026-11-13

Implemented and verified (S1–S4, E1–E8, V1–V4). One recorder now produces either
a narrow Q-training snapshot or a wide offline-reward dataset; the core field
contract is unchanged and legacy `brg-offline-replay-v1` snapshots still load.
Automated gates: 20 focused / 321 portable / 28 gym / 7 learning tests, Ruff
clean. Artifact: `/tmp/opencode/replay-1m-all-signals-20261113/replay_iter_0000616`
(1,000,000 rows, 52 fields, 7.458 GB). Durable docs: `docs/offline_height_q.md`
§"Selectable raw-state features". Committed as `Add selectable replay field
recording` in `BigRedGym`. Open: none.

## 1. Goal

### 1.a. Goal statements

- Let the existing transition recorder record any subset of available simulator
  state, selected at run time, instead of a fixed field list.
- Unblock offline reward work whose label needs state the actor observation does
  not carry (contacts, per-body kinematics, torques, per-term rewards, critic
  observation, successor commands/gait state).
- Collect a 1M-transition dataset with every supported signal from a trained
  policy.

### 1.b. Success criteria

- One flag selects groups, individual fields, or `all`; unknown names fail.
- A recording's exact field set is declared in the manifest and validated on
  load; a mismatch is rejected, not silently accepted.
- Core-only recordings and existing v1 snapshots are unaffected.
- `load_snapshot`, `concat_datasets`, `nested_subsets`, `episode_split`, and the
  height/angular relabel paths keep working on wide snapshots.

### 1.c. Constraints

- Do not change the core transition contract or any existing Q artifact format.
- Recording must not consume training RNG or change PPO storage.
- Preserve unrelated worktree changes; commit only this change on approval.
- Do not commit logs, snapshots, or any file over 100 KiB.

### 1.d. Questions

- Which policy produces the dataset? → resume `model_600` (trained go2trot).
- How wide is "all the signal"? → every env tensor plus next-state values and
  per-term rewards.
- Snapshot cadence? → end only (`--replay_snapshot_interval=0`).

## 2. Design

### 2.a. Interfaces

```python
# ========================
# Library: learning/offline/replay_fields.py (new)
# ========================
STEP_FEATURE_ATTRS: dict[str, str]      # recorded name -> env attribute, pre-action
NEXT_FEATURE_ATTRS: dict[str, str]      # recorded name -> env attribute, post-action pre-reset
RUNNER_FEATURES = ("critic_obs", "next_critic_obs", "reward_terms")
REPLAY_FEATURE_GROUPS = {"critic", "root", "base", "joints", "contacts",
                         "bodies", "commands", "gait", "rewards", "episode"}
ALL_REPLAY_FIELDS: tuple[str, ...]      # canonical order, group-major

def resolve_replay_fields(selection) -> tuple[str, ...]:
  # ""/None/"none"/"core" -> (); "all" -> ALL_REPLAY_FIELDS;
  # group or field token expands; unknown token raises ValueError.

def step_feature_names(fields)      # all non-"next_", non-"reward_terms"
def next_feature_names(fields)      # "next_*" and "reward_terms"

# ========================
# Library: learning/offline/contracts.py (modify)
# ========================
REPLAY_FORMAT = "brg-offline-replay-v1"        # legacy core-only (unchanged)
REPLAY_FORMAT_V2 = "brg-offline-replay-v2"     # declares its field set
SCHEMA_VERSION = 1                              # unchanged; Q artifacts unaffected

@dataclass(frozen=True)
class RecorderConfig:
  ...
  fields: tuple[str, ...] = ()   # canonical/ordered/unique, validated when enabled

# ========================
# Library: learning/storage/transition_recorder.py (modify)
# ========================
class TransitionRecorder:
  def __init__(..., extra_fields=(), reward_term_names=(), field_metadata=None)
  # recorded_fields = TRANSITION_FIELDS + extra_fields
  # extra buffers allocated lazily on first append; width fixed for the run
  def append(..., extra_features: dict[str, Tensor] | None = None)
  # extra_features must equal extra_fields exactly; [N, width] float32 finite
  def snapshot(iteration)   # manifest: recorded_fields, extra_fields,
                            # field_dims, core_fields, reward_term_names,
                            # field_metadata, format=REPLAY_FORMAT_V2

# ========================
# Library: learning/offline/dataset.py (modify)
# ========================
def _resolve_recorded_fields(manifest, data) -> (extra_fields, recorded_fields)
  # v1: schema 1 + exactly TRANSITION_FIELDS
  # v2: schema 1 + recorded_fields == core + extra_fields, data keys equal
def _validate_extra_fields(manifest, data, extra_fields, valid_count)
  # float32, (valid_count, field_dims[name]), finite; reward_term_names width
def concat_datasets(...)   # key set must match; canonical field order preserved

# ========================
# Library: learning/runners/on_policy_runner.py (modify)
# ========================
extra_fields = resolve_replay_fields(replay_cfg.get("fields", ()))
reward_term_names = termination_terms + weights   # initialize_rewards_dict order
field_metadata = {"body_names": robot_layout.body_names,
                  "dof_names": actuated_dof_names}

def _replay_features(names, *, rewards_dict=None)  # clones at one time point
def _replay_feature(name, rewards_dict)
  # critic_obs / next_critic_obs -> get_obs(critic_cfg["obs"])
  # reward_terms -> stack(rewards_dict.sum(0).values(), dim=1)
  # else getattr(env, STEP|NEXT_FEATURE_ATTRS[name]); _rigid_body_state -> [N, -1]

# ========================
# CLI: scripts/train.py (modify)
# ========================
--replay_fields GROUPS_OR_FIELDS_OR_ALL   # default "" = core fields only
```

### 2.b. Data structures

```python
# Snapshot data — v2
#   core: actor_obs, action, next_actor_obs, successor_height_m, terminated,
#         timed_out, episode_id, step_id, env_id, behavior_iteration
#   extras: every selected field, float32 [N, width]; widths from the env
# Manifest additions (v2 only)
#   format = brg-offline-replay-v2; recorded_fields; extra_fields;
#   field_dims {name: width}; core_fields; reward_term_names; field_metadata

# Time semantics
#   plain name  -> read before the action (matches the acting observation)
#   next_ name  -> read after the action, before reset (true successor)
#   successor_height_m is the pre-reset height; no duplicate next_base_height
```

### 2.c. Pseudo code

```python
# ---- run ----
env = make_env("go2trot", ...)
runner = make_alg_runner(env, train_cfg)          # loads model_600
recorder = TransitionRecorder(..., extra_fields=resolve_replay_fields("all"))

for iteration in range(...):
  for step in range(num_steps_per_env):
    pre = snapshot(step_feature_names(extra_fields))      # before action
    env.step(); update_rewards_dict(...)
    post = snapshot(next_feature_names(extra_fields),     # after action,
                    rewards_dict=rewards_dict)            # before reset
    reset_envs()
    recorder.append(..., extra_features={**pre, **post})

recorder.snapshot(iteration)   # immutable dir: data.pt + manifest.json
```

## 3. Refined steps

### setup

- S1. Baseline: `git -C BigRedGym status --short`; confirm only intended files.
- S2. Confirm env tensors to expose: `root_states`, `dof_pos`, `dof_vel`,
  `contact_forces` (`[N, num_bodies, 3]`), `_rigid_body_state`,
  `torques`/`dof_pos_target`/`dof_pos_history`/`dof_vel_target`/`tau_ff`/gains,
  `commands`, `base_*`, `projected_gravity`, `base_height`, `phase*`,
  `gait_reference`, `episode_length_buf`. No `dof_acc` or `feet_air_time` exists
  in this codebase, so neither can be recorded.

### edit

- E1. `learning/offline/replay_fields.py` (new) — registry, groups, canonical
  order, `resolve_replay_fields`, step/next partition.
- E2. `learning/offline/contracts.py` (modify) — `REPLAY_FORMAT_V2`;
  `RecorderConfig.fields` with canonical-order validation. `SCHEMA_VERSION`
  untouched so Q artifacts are unaffected.
- E3. `learning/storage/transition_recorder.py` (modify) — lazy extra buffers,
  exact-set/width/finite validation, v2 manifest with declared fields.
- E4. `learning/runners/on_policy_runner.py` (modify) — field resolution,
  reward-term names, body/dof names, pre-action and pre-reset snapshots.
- E5. `scripts/train.py` (modify) — `--replay_fields`.
- E6. `learning/offline/dataset.py` (modify) — v1/v2 resolution, extra-field
  validation, concat over the declared field set.
- E7. `tests/unit_tests/test_offline_replay.py` (modify) — 13 new tests.
- E8. `docs/offline_height_q.md` (modify) — field groups, naming rule,
  per-body caveat, size budget.

### build

- B1. `ruff format` touched files, `ruff check .`.

### verification

- V1. Focused replay tests: round trip, exact-set rejection, width change, NaN,
  reward-term names, legacy v1 accept/reject, missing tensor, concat.
- V2. Full gates: default `pytest -q`, `pytest gym`, `pytest learning`.
- V3. Warp wiring smoke (`--replay_fields=all`, 64 envs, 2 iterations): snapshot
  loads, widths match the env, relabel and episode subsets work.
- V4. 1M collection from `model_600` (4096 envs, 16 iterations), then load and
  inspect the artifact.

## 4. Review

- **Fail closed:** unknown field names, non-canonical `RecorderConfig.fields`,
  a missing/unexpected extra tensor, a width change, non-finite values, and a
  manifest/data field-set mismatch all raise. No silent narrowing.
- **Backward compatibility:** core-only recordings and v1 snapshots are
  unchanged; only v2 declares extras. A v1 manifest carrying extra tensors is
  rejected. `SCHEMA_VERSION` (shared with Q artifacts) is not bumped.
- **Semantics:** `next_` fields are captured pre-reset, so successors are never
  reset states; `reward_terms` come from the runner's weighted dict, not a
  reconstruction.
- **Memory/disk:** all-field `go2trot` is 1856 float32 values per transition
  (7.4 KB). A 1M ring plus the snapshot clone is ~15 GB host RAM; disk is
  7.46 GB.
- **Per-body widths:** `contact_forces` and `rigid_body_states` are per body
  (42 for Go2, including rotor/IMU/radar/camera), so `field_metadata` records
  `body_names`/`dof_names` rather than assuming four feet.
- **Locality:** feature capture is a separate generator-free read; recording
  still does not consume training RNG.

## 5. Execution results — 2026-11-13

- V1 — `pytest tests/unit_tests/test_offline_replay.py -q` → `20 passed`.
- V2 — `pytest -q` → `321 passed, 97 deselected`; `pytest gym` → `28 passed`;
  `pytest learning` → `7 passed`; `ruff check .` → clean.
- V3 — Warp smoke at 64 envs: `--replay_fields=all` produced a v2 snapshot with
  42 extra fields; load, relabel (inclusive height bound), and episode subsets
  all worked. Fix found here: `_rigid_body_state` is flattened across envs and
  needed a `[N, -1]` view.
- V4 — 1M collection: `--num_envs=4096 --max_iterations=16 --resume
  --load_run=Nov10_10-36-22_gpu-final-600-trot-hist-20261110 --checkpoint=600
  --replay_fields=all --replay_capacity=1000000 --replay_snapshot_interval=0`.

Artifact `/tmp/opencode/replay-1m-all-signals-20261113/replay_iter_0000616`:

| property | value |
|---|---|
| rows | 1,000,000 (1,048,576 appended, ring keeps the last 1M) |
| fields | 52 recorded (10 core + 42 extra) |
| per-transition | 1856 float32 + provenance ≈ 7.4 KB |
| size | 7.458 GB |
| format / schema | `brg-offline-replay-v2` / 1 |
| source revision | `146b04180ce21bb38fc3733dcf088cb789a79aa4` |
| config fingerprint | `6412a8784c883ecb3b28c9601ad1871bec4acb97bfee0f5d7b13e77acc9938cb` |
| run dir | `logs/go2trot/Nov13_15-11-03_gpu-1m-all-signals-20261113b` (`model_600.pt`, `model_616.pt`) |
| reward terms | 12 (`termination` first) |
| episode groups | 8,088 `(env_id, episode_id)` pairs, ~124 steps each |
| flags | 8 terminated, 3,997 timed out |
| base height | mean 0.321 m (σ 0.026) |

## 6. Usage

```bash
# narrow: core only (unchanged behavior)
--replay_record

# offline-reward dataset
--replay_record --replay_capacity=1000000 --replay_fields=all

# cheaper label set (~1 KB/transition)
--replay_fields=critic,commands,gait,rewards
```

```python
from learning.offline.dataset import load_snapshot
d = load_snapshot("/tmp/opencode/replay-1m-all-signals-20261113/replay_iter_0000616")
d.data["next_commands"]                      # pre-reset successor command
d.data["contact_forces"].view(len(d), 42, 3) # per-body, feet at 11/20/29/38
```

## 7. Limits and follow-ups

- The 1M set is ~2 episodes per env (4096 envs × ~256 control steps), so
  episode-grouped splits are valid but trajectory diversity is one run lane.
  More per-environment episodes need more steps at fewer envs.
- `dof_acc` and `feet_air_time` do not exist in this codebase; recording them
  would require a task buffer, not a registry entry.
- The wide dataset is not itself a reward: a new reward still needs its own
  spec/format/artifact per the offline-reward conventions.
