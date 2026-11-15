# Part 1 — goal, architecture, steps and the first experiments (S1-S9)

> Part of the replay/relabel Q plan. Index: [README.md](README.md).

Goal: Preserve BRG training transitions and learn validated alternative-reward values for fixed nominal policies.
Status: COMPLETED — 2026-11-11. The replay/relabel/FQE/variance/persistent-intervention pipeline, experiments and evidence are archived here in four parts; see [README.md](README.md); see §17 for the completion record. Scientific limitations remain: persistent Q selection is not safety, `Q^pi`, transfer or policy-improvement validation.
Open questions: Height reference/bound; checkpoint and discount resolution; retention capacity; evaluation domain and acceptance tolerances.

## 1. Goal

### 1.a. Goal statements
- Preserve training transitions on disk and support loading and relabeling them for height, pitch/roll, and joint-velocity rewards.
- Learn alternative-reward action values for explicitly identified, fixed nominal policies from retained training data.
- Establish the empirical suitability of those values for future candidate-action threshold filtering.

### 1.b. Success criteria
- Loaded records retain the measurements and provenance needed to reproduce declared alternative rewards without rerunning physics.
- Independent simulator continuations quantify return-prediction error for nominal and candidate first actions under matching continuation policies.
- Nested 10%, 50%, and 100% dataset experiments report prediction error and threshold false acceptance with uncertainty; decreasing error is a hypothesis, not a guaranteed outcome.

### 1.c. Constraints
- Preserve current stochastic trot-training behavior; exclude MPPI, intervention execution, and new-policy optimization from this plan.
- Limit suitability claims to validated policies, states, candidate-action ranges, reward bounds, and simulation conditions; make no safety guarantee.
- Preserve clean physical reward measurements independently of policy observations; reward relabeling does not change recorded dynamics or termination rules.

### 1.d. Questions
- [open] What reward timing, height/attitude references, joint-wise bounds, and combined versus separate returns are required; are bounds fixed per experiment or critic conditions?
- [open] Which frozen checkpoint and continuation policy are evaluated, how are mixed-policy identities handled, and what discount, horizon, timeout, and observation-noise semantics define the return?
- [open] What retention capacity, dataset-subset construction, candidate-action domain, threshold range, repetitions, and numerical acceptance criteria establish useful evidence?

## 2. High-Level Architecture

### Agreed scope refinements
The first experiment fits one height-reward Q with a constant bound. Other reward channels and bound conditioning are deferred. The reward is -1 outside the reference-height bound and 0 inside, measured after action execution and before reset.

The target is discounted continuing return under one frozen stochastic trot checkpoint, preserving its training-time observation-noise convention. True termination becomes absorbing failure; timeout does not. Finite-horizon Q is deferred to a separately specified experiment or future plan.

Q receives only current trot actor-observation fields and candidate actions. Ground-truth height supplies labels and diagnostics, never critic input. Partial observability is an explicit limitation: observation-only FQE is an approximation, not a guaranteed Markov-state evaluation method.

### 2.a. Interfaces
Proposed implementation locations below are relative to `BigRedGym/`. Paths marked new do not yet imply implemented functionality. Shared dataset, reward and artifact contracts from §2.b live in `learning/offline/contracts.py` (new).

```python
# New: learning/storage/transition_recorder.py — bounded retention and snapshots.
# Existing integration: learning/runners/on_policy_runner.py — capture before reset.
# Existing entry point: scripts/train.py — recorder configuration and lifecycle.
# Recording is independent of PPO storage and must not change behavior or RNG draws.
class TransitionRecorder:
    def append(batch: RecordedTransitionBatch) -> None:
    def snapshot(destination) -> DatasetManifest:

# New: learning/offline/dataset.py — load_snapshot and nested dataset subsets.
# Dataset boundary: validate schema, provenance, valid count and ring ordering.
# Snapshots are immutable training inputs; mutable ring contents are not datasets.
def load_snapshot(source) -> TransitionDataset:

# New: learning/offline/rewards.py — height labels and reward-spec validation.
# Reward boundary: no physics execution or policy-input changes.
# Height is measured in meters at the post-action/pre-reset boundary.
def relabel_height(dataset, specification: HeightRewardSpec) -> LabeledDataset:

# New: learning/offline/nominal_policy.py — frozen checkpoint adapter.
# Reuses existing learning/modules/actor.py; does not change its training behavior.
# Frozen nominal policy: checkpoint parameters and exact observation/action contract.
# Uses the policy's stochastic action distribution, not deterministic playback.
class FrozenTrotPolicy:
    def sample_actions(actor_observations, rng) -> Actions:

# New: learning/algorithms/fqe.py — Q network, target network and FQE updates.
# New: learning/offline/training.py — fit_fqe orchestration and artifact export.
# New: scripts/train_height_q.py — load, relabel, select subsets and invoke fitting.
# Offline evaluation boundary: no actor optimization, no IQL improvement target.
def fit_fqe(dataset, policy, training_spec) -> QArtifact:

# New: learning/offline/q_artifact.py — HeightQ loading and prediction.
# Keep separate from existing learning/corl_artifact.py, which loads actors only.
# Runtime boundary: rejects incompatible field order/scales or action conventions.
class HeightQ:
    def predict(actor_observations, candidate_actions) -> ExpectedReturn:

# New: learning/offline/evaluation.py — rollout return and threshold validation.
# New: scripts/evaluate_height_q.py — simulator setup and report entry point.
# Existing scripts/evaluate_go2trot.py remains the separate trot-performance gate.
# Simulation validation boundary: matched initial states and explicit first actions.
# Subsequent actions follow the frozen nominal policy without an intervention filter.
def validate_q(artifact, policy, evaluation_spec) -> ValidationReport:
```

`learning/offline/` is a proposed new BRG package. CORL's `thirdparty/CORL/algorithms/offline/iql.py` is a reference for reusable network/training patterns, not the FQE objective or a runtime benchmark dependency. The first implementation lives in BRG and does not modify the CORL submodule or invoke its policy-improvement loop. Any adapted CORL code must retain applicable attribution and licensing. Test locations and individual edit steps will be specified in Phase 3.

### 2.b. Data structures
```python
class RecordedTransitionBatch:
    actor_observation         # Exact policy-facing, scaled/noisy observation at action selection.
    action                    # Actual sampled action in BRG policy units; no retrospective clipping.
    next_actor_observation    # Successor before reset; same field/scaling/noise convention.
    successor_height_m        # Clean post-action height, label-only.
    terminated, timed_out     # Independent flags; preserve simultaneous true values.
    run_id, env_id, episode_id, step_id, behavior_iteration

class DatasetManifest:
    schema_version, field_order, units, scales, action_convention
    task_config, resolved_training_config, backend, source_revision
    run_provenance, policy_provenance, collection_interval, valid_count
    retention_spec, snapshot_id, fragment_boundaries
    # Retention capacity remains configurable and unchosen.
    # Ring overwrite may cut episodes; IDs preserve the remaining sequence.
    # Explicit successors keep isolated transitions usable without fabricated terminals.

class HeightRewardSpec:
    reference_height_m, bound_m  # Constants for each artifact; numeric values remain open.
    timing = "post_action_pre_reset"
    inside_reward = 0
    outside_reward = -1
    terminal_continuation = "absorbing_failure"

class QArtifact:
    weights, network_spec, observation_preprocessing
    policy_checkpoint_identity, observation_noise_spec, action_contract
    reward_spec, discount, return_convention
    dataset_snapshot_identity, subset_identity, training_seed
    validated_domain, validation_report_reference
    # No privileged features in inference. Save preprocessing alongside weights.
    # Explicitly identify continuing-return format; future finite-horizon artifacts differ.
```

Observation noise for pre-reset successors must not consume the training RNG stream or overwrite the observations used by PPO. Ordinary successors should reuse already collected matching observations where possible. Ended transitions require a pre-reset observation capture with isolated noise generation. Copy live tensors before reset or later simulator updates mutate them.

The dataset can contain behavior from several PPO iterations; this does not change the frozen evaluation policy. Multiple distinct target policies require explicit target-policy identities and separate evaluation contracts; a pooled dataset does not define a synthetic continuation policy.

### 2.c. Pseudo code
```python
def collect_training_data():
    # Preserve existing PPO action collection and update logic.
    capture_policy_observation_and_action()
    execute_policy_interval()
    capture_height_flags_and_successor_before_reset()
    recorder.append(copied_transition_batch)
    continue_existing_reset_and_PPO_update()
    # Persist coherent snapshots at explicit boundaries; reject incomplete snapshots.

def offline_experiment(snapshot, fixed_policy, reward_spec, gamma):
    require(0 <= gamma < 1)
    data = load_snapshot(snapshot)
    labels = relabel_height(data, reward_spec)
    subsets = nested_subsets(labels, fractions=[0.1, 0.5, 1.0])
    # Keep subset construction and policy identity fixed across comparisons.
    for subset in subsets:
        for seed in declared_training_seeds:
            fit_fqe(subset, fixed_policy, training_spec)

def fqe_target(transition, frozen_target_Q, fixed_policy, gamma):
    r_prime = height_reward(transition.successor_height_m)
    failure_value = -1 / (1 - gamma)
    if transition.terminated:
        # Current reward remains the observed height label.
        # Every subsequent absorbing-failure reward is -1.
        return r_prime + gamma * failure_value
    next_action = fixed_policy.sample_actions(transition.next_actor_observation)
    # Average sampled continuation actions as specified by training configuration.
    # A timeout alone still bootstraps from the true pre-reset successor.
    return r_prime + gamma * frozen_target_Q(next_observation, next_action)

def evaluate_candidate(initial_simulator_state, first_action):
    prediction = Q(policy_facing_observation, first_action)
    # Repeat from matched physical states and initial policy-facing observations.
    # New randomness affects continuation, not the selected first action.
    returns = simulate_first_action_then_frozen_policy()
    # Disable administrative time-limit resets for continuation measurement.
    # On true failure, append the exact absorbing discounted tail analytically.
    compare_prediction_to_mean_return_with_uncertainty()
```

For a nonterminated rollout stopped after H rewards, omitted continuing return lies in `[-gamma**H / (1-gamma), 0]`. Report this truncation interval and sampling uncertainty rather than using the learned Q as its own reference tail. Longer rollouts reduce the tail bound. With all rewards in [-1, 0], continuing Q lies in `[-1/(1-gamma), 0]`.

Independent evaluation covers nominal first actions and a declared candidate-action domain. Report prediction error, action ranking and false acceptance near selected thresholds. False acceptance concerns estimated expected return, not a single unlucky stochastic rollout. Report ambiguous classifications when confidence/tail intervals overlap a threshold.

### Architecture decisions still open
- Numeric height reference/bound, selected checkpoint, and resolved discount must be fixed before an experiment; do not silently adopt config defaults.
- Retention capacity, persistence cadence, subset sampling, training budgets, candidate-action ranges, repetitions and acceptance tolerances remain to be specified.
- Current observations can alias hidden physical states; use saved evaluation-state diagnostics to distinguish approximation error from hidden-state variability without leaking privileged inputs into Q.

## 3. Refined steps

### 3.a. Setup
- Inspect the current BRG developer guide, CORL integration documentation, and affected source before edits; preserve unrelated changes and leave the CORL submodule unchanged.
- Confirm the installed uv environment and dependency-light CORL reference implementation; do not install upstream CORL requirements into BRG.
- Resolve experiment inputs before launching collection or fitting: checkpoint, height reference/bound, discount, capacity, snapshot cadence, sample-selection rule, training budgets and evaluation tolerances.
- Establish the dataset contract for a configurable exact transition capacity, including partial batches at wraparound; retain the most recent transitions and record deterministic ordering within parallel environment batches.
- Require positive capacity and valid snapshot cadence; use explicit user configuration rather than silently choosing a retention budget. Recording remains disabled unless requested.
- Declare candidate-action ranges in BRG policy units and the evaluation initial-state distribution. Evaluation seeds and trajectories must be independent of training samples.

### 3.b. Edit

#### Edit 1 — Define versioned data and value contracts
Purpose: Make recorded data, reward semantics, and inference inputs independently interpretable.

New files:
- `BigRedGym/learning/offline/__init__.py`
- `BigRedGym/learning/offline/contracts.py`

Expected impact:
- Define transition, manifest, reward, policy, training, evaluation and artifact specifications with explicit units and versions.
- Validate file/schema boundaries, including dimensions, finite values, independent boolean flags, positive scales and compatible observation/action ordering.
- Distinguish simulator timeout, actual termination and retention fragments; never encode a ring cut as a terminal event.

#### Edit 2 — Add bounded recording without changing PPO behavior
Purpose: Preserve the last configured number of actual training transitions and export coherent disk snapshots.

New files:
- `BigRedGym/learning/storage/transition_recorder.py`

Modified files:
- `BigRedGym/learning/runners/on_policy_runner.py`
- `BigRedGym/scripts/train.py`

Expected impact:
- Introduce opt-in recorder settings without changing the default PPO path or replacing DictStorage.
- Copy action-selection observations/actions and clean post-action height/flags/successors before reset can overwrite task state.
- Isolate any added observation-noise sampling from training RNG streams; preserve existing ordinary successor observations when their semantics match.
- Retain records in bounded host-memory storage and export immutable snapshots with manifest, valid count, IDs and provenance; no unbounded background queue.
- Use completed-file publication so interrupted exports cannot be mistaken for valid snapshots; define snapshot naming and failure behavior explicitly.
- Preserve the distinction between bounded in-memory retention and disk history: retaining every immutable snapshot grows disk use, so snapshot cleanup must be an explicit separate policy rather than automatic deletion of experiment inputs.
- Preserve PPO checkpoint and optimizer semantics; snapshots do not claim exact simulator or training-resume restoration.
- Support explicitly retained early- and late-training snapshots plus corresponding PPO checkpoints; a last-N ring alone cannot preserve early data after overwrite.

#### Edit 3 — Load, relabel and form reproducible subsets
Purpose: Convert recorded snapshots into independently labeled FQE transition datasets.

New files:
- `BigRedGym/learning/offline/dataset.py`
- `BigRedGym/learning/offline/rewards.py`

Expected impact:
- Load only valid records and verify manifest compatibility before combining data from runs.
- Label each transition from its clean pre-reset successor height using the fixed reward specification; do not mutate source snapshots.
- Retain explicit successors and trajectory IDs while exposing consistently flattened transition batches.
- Generate deterministic nested 10%, 50% and 100% subsets under the agreed selection rule and persist membership identities.
- Define early and late data windows by recorded behavior iteration and form equal-size early, late and mixed datasets before comparing their nested fractions; record mixture proportions and sample IDs.
- Detect overlap between source windows and avoid duplicate transitions in mixtures; keep final evaluation trajectories independent of all training subsets.
- Keep height label data out of critic inputs; reject reward specifications requiring unavailable measurements.

#### Edit 4 — Load the frozen stochastic nominal policy
Purpose: Make the evaluation policy contract identical across FQE targets and simulator continuations.

New files:
- `BigRedGym/learning/offline/nominal_policy.py`

Expected impact:
- Load the selected PPO actor parameters and preprocessing without optimizer updates or observation-statistic changes.
- Sample its learned action distribution using caller-owned RNGs rather than deterministic playback or action clipping.
- Preserve BRG action units and recorded observation ordering/scaling/noise conventions; reject mismatched datasets or checkpoints.
- Record checkpoint identity separately from behavior-policy iteration metadata; mixed behavior data must not silently change the target policy.

#### Edit 5 — Fit and export a continuing-return height Q
Purpose: Implement fixed-policy evaluation rather than offline actor improvement.

New files:
- `BigRedGym/learning/algorithms/fqe.py`
- `BigRedGym/learning/offline/training.py`
- `BigRedGym/learning/offline/q_artifact.py`
- `BigRedGym/scripts/train_height_q.py`

Expected impact:
- Implement a scalar Q and frozen target network with explicit action-sampling count, optimizer, target-update and preprocessing settings.
- For true termination, use observed immediate height reward plus the discounted absorbing-failure value; true termination takes precedence if timeout is also true.
- For timeout alone, bootstrap from the recorded pre-reset successor using the frozen stochastic policy.
- Train only Q parameters and save inference weights, preprocessing, return/reward semantics, target-policy identity and experiment provenance.
- Expose runtime prediction using actor-observation fields and actions only; keep existing actor artifacts unchanged.
- Reject unsupported finite-horizon or bound-conditioned requests rather than interpreting them as continuing fixed-bound values.
- Keep minibatch loading bounded; document peak host/device memory including snapshot buffers and serialization copies.
- Run the early/late/mixed dataset comparison against the same frozen target-policy checkpoint, reward specification and discount, with declared comparable budgets and multiple seeds; changing the behavior-data source must not change the target Q's policy.

#### Edit 6 — Validate nominal and candidate-action values
Purpose: Produce independent evidence of return accuracy and future threshold-filter suitability.

New files:
- `BigRedGym/learning/offline/evaluation.py`
- `BigRedGym/scripts/evaluate_height_q.py`

Expected impact:
- Generate held-out initial conditions and repeated continuations with an explicit candidate first action and frozen stochastic policy thereafter.
- Match full task and randomized dynamics state for paired comparisons; do not claim matching by restoring root pose alone.
- Use reproducible initialization or explicitly validated state restoration; verify matching initial observations and physical-state provenance before paired tests.
- Suppress administrative time-limit resets during continuing evaluation while retaining true-failure detection and the recorded dynamics/command conventions.
- Append the exact absorbing tail following failure and report an analytical omitted-tail interval for surviving finite rollouts.
- Report prediction errors, candidate-action ranking, false acceptance and ambiguous threshold classifications with sampling and truncation uncertainty.
- Compare nested data fractions under declared budgets and seeds; report nonmonotonic or negative results rather than making the decreasing-error hypothesis a software test assertion.
- Report hidden-state variation separately where multiple physical situations share similar observable inputs; never feed those diagnostics into deployed Q.
- Add held-out early-policy and late-policy evaluation-state groups generated using retained checkpoints and independent seeds, then execute each candidate first action and continue with the same frozen target policy.
- Preserve task histories and dynamics when switching from the state-generating policy to the target policy; sample the initial observation under the target policy's agreed observation convention and tag the state-generator checkpoint separately.
- Use reproducible regeneration or validated full-state restoration for repeated continuations; do not infer historical simulator states from observation-only dumps or use early-policy recorded returns as final-policy Q ground truth.
- Report a cross-table of training-data source (early/late/mixed) versus evaluation-state source (early/late), in addition to aggregate metrics, using the same evaluation cases across critics.

### 3.c. Build and static checks
All commands are for the implementer, from `BigRedGym/`; Scaffold does not execute them.

- Use the existing uv-managed Python 3.11 environment and `uv run --frozen`; no new dependency is expected, and any discovered need requires explicit review.
- Run `uv run --frozen ruff format <touched-python-paths>` and `uv run --frozen ruff check <touched-python-paths>` followed by the repository lint gate.
- Run tests in one process, without pytest-xdist; constrain numerical-library threads for verification to `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- No simulator/native package rebuild is proposed. If packaging is required, first confirm the repository's configured build tool; do not add build dependencies solely for this work.
- Run GPU experiments sequentially, with unique run names and measured memory/throughput; recorder staging and snapshots must be included in the memory budget.

### 3.d. Verification

#### Case 1 — Recording and relabeling integrity
New test: `BigRedGym/tests/unit_tests/test_offline_replay.py`.

Run: `uv run --frozen python -m pytest tests/unit_tests/test_offline_replay.py -q`.

Expected:
- Tiny-capacity wraparound, partially filled buffers, oversized append batches and snapshot reload retain exactly the declared valid records and explicit successor associations.
- Synthetic task resets cannot overwrite saved height/successors; simultaneous termination/timeout flags survive recording.
- Boundary heights label correctly; recorder enablement leaves baseline actions and training RNG state unchanged under a controlled fake-environment test.
- Corrupt/incompatible snapshots fail explicitly; interrupted publication is not accepted as a complete dataset.
- Early/late selection, mixture deduplication and nested membership are reproducible and preserve policy/iteration provenance through save/load.

#### Case 2 — FQE semantics and artifact equivalence
New test: `BigRedGym/tests/unit_tests/test_height_fqe.py`.

Run: `uv run --frozen python -m pytest tests/unit_tests/test_height_fqe.py -q`.

Expected:
- Analytic toy processes verify continuing returns, gamma=0, absorbing failure, timeout bootstrapping and simultaneous-flag precedence.
- Frozen-policy sampling and Q inference preserve units and preprocessing; actor parameters remain unchanged during Q fitting.
- Export/reload predictions agree within declared numerical tolerance, and privileged label fields cannot affect predictions when observation/action inputs are held fixed.
- Tail intervals and threshold classifications are correct on deterministic fixtures.

#### Case 3 — Robot pipeline and scientific experiment
Run the proposed recording, fitting and evaluation entry points only after required CLI values and acceptance tolerances have been settled; begin with a short sequential smoke run, then the approved GPU experiment.

Expected:
- A real recorded snapshot passes schema and label checks, produces a loadable Q artifact, and supports fresh nominal/candidate-action continuations.
- The 10%/50%/100% reports share the declared target policy and evaluation distribution and include seed variability, candidate coverage and tail bounds.
- Equal-size early/late/mixed comparisons isolate data composition before nested-fraction comparisons; the evaluation-state groups are generated independently and resume the same frozen target policy after the candidate first action.
- Cross-group reports distinguish behavior-data provenance, evaluation-state-generator policy and target continuation policy; unavailable early snapshots/checkpoints are reported as missing evidence rather than reconstructed from final checkpoints.
- Recorder overhead and peak memory are measured against recording-disabled training; no learning-quality equivalence is inferred from a successful smoke run.
- Prediction and threshold results establish or reject suitability within the tested domain; neither policy safety nor monotonic improvement is assumed.

After focused tests, run the existing default and colocated suites sequentially: `uv run --frozen python -m pytest -q`, `uv run --frozen python -m pytest gym -q`, and `uv run --frozen python -m pytest learning -q`. CUDA collection/evaluation evidence is separate from portable unit-test results.

## 4. Review

### 4.a. Scientific boundaries
- **Continuing-return coverage:** training resets at time limits, but the target continues through them. Recorded pre-reset successors fix transition semantics, not missing coverage of long continuations. Validate longer nominal rollouts explicitly; if they leave dataset support, report insufficient evidence rather than trust extrapolation.
- **Partial observability:** current actor inputs omit height and randomized dynamics. Observation-only FQE can have biased Bellman targets and policy-dependent hidden-state distributions, not merely finite-sample variance. Failure to predict reliably can justify a later history/estimator extension; increasing dataset size alone is not a remedy.
- **Candidate-action support:** Gaussian behavior has formal unbounded support but finite data does not cover all actions. Report empirical support and errors separately for candidate-action regions; do not silently clip original actions or interpret an untested candidate range as validated.
- **Meaning of the return:** absorbing failure deliberately changes future rewards to -1 regardless of unobserved height. The artifact estimates height compliance with a failure penalty, not pure observed-height violations alone. This convention must appear in reports and artifact metadata.
- **Future-filter limitation:** candidate-first-action tests resume the nominal policy afterward. They do not validate repeated MPPI/filter intervention, alternative continuations, or a safety guarantee.

### 4.b. Collection and persistence boundary cases
- Verify observation/action/physics alignment at startup, normalization burn-in, reset and resume. In particular, the current runner obtains observations before optional burn-in; stale observations must not be silently presented as valid state-transition evidence. If encountered, explicitly exclude/flag affected records or seek approval for a narrowly scoped runner correction.
- For actor frequencies below control frequency, record the policy-interval duration and define which ending flags and post-interval height are represented. The first experiment uses the current trot frequency configuration; other timing geometries require separate validation rather than assumed equivalence.
- Copy live tensors before reset, preserve the observation actually used for the action, and distinguish it from independently sampled pre-reset successor noise. Added recorder noise must use a separate generator and must not advance training RNG state.
- Handle empty snapshots, capacity smaller than one parallel batch, wrap at a batch boundary, repeated snapshot requests and resume-created run IDs. Empty datasets are invalid training inputs; snapshot manifests must prevent accidental duplicate ingestion.
- Validate snapshot publication, free-space failures and incomplete files. Do not silently drop records or delete prior experiment snapshots. Disk retention remains an explicit user decision, separate from the last-N in-memory policy.
- Bound memory during snapshot export and loading, including temporary copies. A pause-and-copy or synchronous export is acceptable initially; asynchronous writing must not read storage while ring overwrites mutate it.

### 4.c. Training and inference boundary cases
- Validate `0 <= gamma < 1`, positive training sample counts, nonnegative finite height bound, finite reference height and compatible dimensions. Height exactly on either boundary is inside; the artifact records this inclusive convention.
- A terminal transition uses its observed immediate height reward and then the absorbing tail. For example, an in-bound terminal transition has target `-gamma/(1-gamma)`, not `-1/(1-gamma)`; both flags true must follow this same terminal rule.
- Dataset cuts do not change Bellman targets. Timeout alone bootstraps from a valid physical successor and never from the reset observation; absent successors are rejected rather than repaired by shifting unrelated rows.
- Fit observation preprocessing using only each experiment's training subset, freeze it, and save it with the artifact. Use the target actor's own preprocessing separately; never feed Q-normalized observations into an actor expecting BRG-scaled inputs.
- Freeze all nominal-policy parameters and normalization state. Save the exact checkpoint identity, noise convention and action units so runtime requests cannot silently substitute deterministic playback or a different policy.
- Q's known mathematical range is a diagnostic. Report raw out-of-range predictions rather than hiding instability by clipping outputs before validation.

### 4.d. Evaluation integrity
- For each repeated candidate trial, fix the physical initial state, task hidden state, randomized parameters, initial actor observation and first action; vary continuation randomness independently. If the backend cannot reproduce that setup, use explicitly documented reproducible initializations instead of claiming exact state restoration.
- Separate matched-state action comparisons from comparisons across different hidden states with similar observations. Report both where useful; their uncertainty has different meanings.
- Keep the main evaluation suite independent of the training subsets. Diagnostic trajectory statistics may guide a preliminary bound choice, but freeze bounds, candidate ranges and thresholds before final held-out evaluation.
- For H-step surviving rollouts, include the negative tail interval as well as Monte Carlo sampling uncertainty. Repeated rollout counts and H must make this uncertainty small enough for the chosen threshold tolerances; otherwise classify the result as inconclusive.
- Allocate comparable declared training budgets to the three dataset fractions, and report optimization adequacy. More data with insufficient updates can confound the learning-curve hypothesis.
- Preserve negative results: lower aggregate error can coexist with worse near-threshold false acceptance or candidate ranking. Suitability requires the declared downstream metrics, not only average prediction error.

### 4.e. Remaining gates before execution
- Choose numeric height specification, checkpoint, discount, retention/snapshot settings, subset sampling and training budgets.
- Define evaluation initial conditions, candidate-action coverage, thresholds, repetitions, tail tolerance and numerical acceptance criteria.
- Confirm acceptable recording overhead and disk-retention behavior; leave all outstanding decisions explicit rather than implementing guessed defaults.

Review conclusion: the proposed pipeline is implementable, but the first observation-only critic is an empirical feasibility test. The current plan cannot promise a suitable runtime filter critic until the held-out candidate-action and threshold tests pass. MPPI, intervention execution, other rewards, bound conditioning and finite-horizon Q remain outside this implementation scope.

## 5. Documentation and handoff

### 5.a. Execution journal and todos
The execute agent, not Scaffold, updates `docs/journal.md` according to workspace AGENTS.md §Docs and the dev-journal skill after execution.

Do not create or update todos, including `.opencode/workspace/todos.md`, without the user's explicit approval for that action. Approval of this plan or its execution does not constitute approval to write todos.

Record implemented files, deviations, actual validation commands/results, dataset and checkpoint identities, resource measurements, and remaining experiment gates. Distinguish software smoke success from evidence supporting the Q-accuracy hypothesis. Preserve failed or inconclusive experiments, including false-acceptance uncertainty and hidden-state/coverage limitations.

### 5.b. User and technical documentation
New implementer-owned document: `BigRedGym/docs/offline_height_q.md`.

Cover:
- End-to-end collection, snapshot selection, relabeling, FQE fitting, artifact loading and evaluation commands matching the implemented CLI.
- Dataset versions, observation/action units, height timing, independent termination/timeout flags, bounded memory versus disk retention, and explicit retention of early snapshots/checkpoints.
- Fixed-bound reward definition, inclusive bounds, continuing discount, absorbing failure, pre-reset timeout successors and finite-rollout tail uncertainty.
- Early/late/mixed data comparisons, nested fractions, state-generator versus behavior versus target-policy identities, and held-out candidate-action tests.
- Runtime input requirements and the limitations of observation-only FQE; no claim of universal policy transfer, repeated-filter validity or safety certification.
- Deferred extensions: other constraint rewards, bound conditioning, finite-horizon Q, history/estimators, offline policy optimization and MPPI/intervention integration.

Update existing documents:
- `BigRedGym/README.md`: brief discovery link and opt-in replay/FQE workflow reference; avoid duplicating detailed semantics.
- `BigRedGym/docs/thirdparty/CORL/install.md`: distinguish the existing actor smoke integration from BRG-owned FQE, and identify any adapted CORL components and licensing without claiming the full benchmark stack is supported.
- `BigRedGym/tests/TEST_CATALOG.md`: describe new tests, their analytic/fake-environment oracles and the separate scope of GPU robot experiments.
- `BigRedGym/genAI_skills/README_MUJOCO.md`: record consequential validation limitations and experimental outcomes where relevant, explicitly separating training behavior preservation from offline Q quality.

### 5.c. Matching skill updates
The implementer updates these existing skill documents after behavior is implemented:
- `BigRedGym/genAI_skills/.agents/skills/q2-rl-development/SKILL.md`: FQE versus policy improvement, target-policy contracts, relabeling timing and runtime-observation restrictions.
- `BigRedGym/genAI_skills/.agents/skills/q2-train-and-evaluate/SKILL.md`: implemented recording/fitting/evaluation entry points, artifact provenance and early/late dataset workflows.
- `BigRedGym/genAI_skills/.agents/skills/q2-testing-and-debugging/SKILL.md`: analytic target tests, reset-boundary recording checks, tail uncertainty and threshold-evaluation evidence requirements.

### 5.d. Handoff conditions
- Final approval authorizes the written plan, not evidence claims or guessed numerical experiment settings; unresolved execution gates in §4.e remain explicit.
- The executor must follow workspace phase gates and confirm experiment settings before launching collection/training runs.
- Scaffold writes only this plan. Source edits, documentation/skill updates, journal entries and runtime validation belong to execution; any todo creation or update requires separate explicit user approval.

## 6. Data investigation — 2026-11-10

### 6.a Packages

| Package | Path | Transitions | Size |
|---|---|---:|---:|
| Early (`0–50`) | `logs/go2trot/Nov10_10-35-07_gpu-early-50-trot-hist-20261110/replay/replay_iter_0000050` | 3,276,800 | 1.5 GiB |
| Final (`600`) | `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/replay/replay_iter_0000600` | 1,000,000 | 449 MiB |
| Mix | `logs/go2trot/hist-mix-20261110/replay/replay_iter_0000600` | 2,000,000 | 940 MiB |

The final package resumes from the early `model_50.pt` and contains the last 1M transitions of the 600-iteration run. Mix is an equal-size early/final sample.

### 6.b Height and state histograms

![Early, final, and mixed replay distributions](NOTE-assets)

| Series | Height p5 / p50 / p95 (m) | \|Roll\| p95 / p99 (deg) | \|Pitch\| p95 / p99 (deg) | \|DOF vel\| p95 / p99 (rad/s) |
|---|---|---|---|---|
| Early | 0.140 / **0.198** / 0.308 | 128.9 / 165.8 | 23.7 / 27.3 | 8.02 / 11.40 |
| Final | 0.276 / **0.323** / 0.357 | 6.04 / 8.75 | 6.48 / 9.18 | 4.12 / 6.83 |
| Mix | 0.151 / **0.282** / 0.353 | 105.0 / 153.9 | 22.1 / 25.8 | 6.68 / 10.17 |

Height occupancy around `0.30 m`:

| Band | Early inside | Final inside | Mix inside |
|---|---:|---:|---:|
| ±0.02 | 3.1% | 38.7% | 20.9% |
| ±0.05 | 9.7% | **90.0%** | 49.9% |
| ±0.10 | 47.2% | 99.9% | 73.5% |

The final policy is upright and tight around **0.323 m**; early training is crouched, broader, and contains falls. Even equal-size mix is early-dominated in behavior. Choose a boundary from **final** data; use early/mix as negative examples and for early/late comparisons. A ±0.05 m band gives about 90% final inside and 10% outside; ±0.02 is tight (~39% inside), while ±0.10 is loose (~99.9% inside).

### 6.c Phase distribution and height periodicity

![Phase distribution and phase-locked height](NOTE-assets)

- Phase is essentially uniform over `[0, 2π)`: normalized entropy `0.999999`, max/min bin density `1.009`.
- Gait frequency is `1.02–2.98 Hz`, median `1.99 Hz`; phase advances `0.128–0.375 rad` per control step (`7.3–21.5°`), so a cycle takes about 17–50 steps.
- Height is periodic. The phase-locked mean height amplitude is **~1.8 cm**; the phase-bin mean ranges from ~0.311 m to ~0.329 m, while the 5–95% band spans ~0.265–0.36 m.
- Per-episode height range is broad (median ~0.101 m, p95 ~0.161 m), including transients and episode-to-episode offsets; 7,246 episode fragments were analysed.
- Example traces show clear gait cycles and occasional bound crossings, especially during the initial transient.

### 6.d Phase versus out-of-bound rate

For reference `0.32 m`, all final transitions, 36 phase bins:

| Bound | Overall OOB | Lowest phase bin | Highest phase bin |
|---:|---:|---:|---:|
| ±0.01 m | 66.1% | 60.1% at 75° | 73.3% at 155° |
| ±0.02 m | 38.3% | 31.0% at 75° | 47.7% at 155° |
| ±0.03 m | 20.0% | 14.7% at 75° | 26.3% at 155° |
| ±0.05 m | **5.3%** | **3.2% at 95°** | **8.4% at 205°** |
| ±0.10 m | 0.28% | 0.12% at 75° | 0.59% at 175° |

Upright-only filtering (|roll| and |pitch| ≤ 10°) changes the ±0.05 rate only from 5.3% to 4.8%, so the ±0.05 violations are not mainly falls.

Interpretation: the phase-locked **mean** oscillation (~1.8 cm) is smaller than a ±5 cm bound, so the mean does not cross the boundary. The ±5 cm violations come from within-phase variability, episode offsets, and transients; the rate is still phase-dependent (3.2–8.4%, about 2.6×). For a bound comparable to the oscillation (±2 cm), the mean approaches the boundary and phase dependence is large (31–48%). A constant ±5 cm bound is robust to normal gait phase; a tighter bound should be phase-conditioned or centered on the phase-locked mean.

### 6.e Implication for the first ±0.05 experiment

- Use the final package and `model_600.pt`; use **reference 0.32 m, bound 0.05 m** for the first Q fit.
- Expect about 5% negative (outside-bound) transitions, with a 3–8% phase-dependent rate. The Q receives `phase_obs` and `phase_frequency` in its actor observation, so it can represent the phase dependence, but the reward is a constant band, not a phase-conditioned target.
- Do not choose the bound from the mix; early crouch/falls would bias it low and wide.
- This is distribution evidence, not Q accuracy or policy-quality evidence. The 10/50/100% and fresh-rollout evaluation remain the test.

### 6.f Reset transient: the initial height is out of bound — 2026-11-10

The reference band `0.32 ± 0.05` is `[0.27, 0.37] m`. The reset pose puts the root at
`z = 0.40 m` with the legs only partly extended (`thigh 0.66`, `calf -1.36`), so the
robot starts above the band and falls onto its feet. Measured values:

| Point | Height (m) | Inside band |
|---|---:|---:|
| Root at reset (`base_init_state`) | 0.4000 | 0% |
| First recorded transition, 1M final snapshot | 0.3939 (0.3908–0.3964) | 0.00% |
| `step_id = 1` | 0.3844 (0.3778–0.3895) | 0.00% |
| `step_id = 2` | 0.3754 (0.3685–0.3805) | 0.68% |
| `step_id = 3` | 0.3631 (0.3549–0.3705) | 99.97% |
| `step_id = 4` | 0.3492 | 100% |
| `step_id = 6` | 0.3229 | 100% |

Consequences:

- **Every episode starts with two or three unconditional `-1` rewards** (~`-2.97` discounted), independent of the action taken. The first two steps are essentially ballistic: the spread across 4096 environments is under 1.2 cm, so the candidate action cannot reach the ground contact that would change the height.
- This is action-independent, so it does not by itself distort within-trial candidate ranking, but it does inflate absolute error and bias: the transient is ~12% of a 500-step return whose within-trial candidate spread is ~2.8.
- It explains why the first-step reward is `-1.0` for all 8192 evaluation cases (`first_rewards` mean exactly −1.0 in both 1024-trial reports) and why one-step lookahead cannot discriminate candidates.
- The raw ±0.05 out-of-bound rate of 5.3% includes ~1.2 percentage points of pure reset transient (`step_id` 0–2). Excluding the first three steps, the steady-state rate is ~4.1%.
- Any future horizon or per-step-violation objective must decide explicitly whether the transient counts. Options: skip the first 3 steps in labels and evaluation, lower the reset height so the robot starts in contact, or accept the transient as part of "falling from the reset pose".

## 7. ±0.05 experiment results — 2026-11-10

### 7.a Training setup

Fixed target: final package, `model_600.pt`, reference `0.32 m`, bound `0.05 m`, gamma `0.99`, episode-nested 10/50/100%, batch 256, hidden `[256, 256]`, target tau `0.005`, seed 0.

Two budgets were run:

- **Equal updates:** 100,000 updates for every fraction (pilot). This gives ~256 epochs at 10% and ~26 epochs at 100%, so it is not a fair data-efficiency comparison.
- **Equal epochs (~100):** 40,000 / 200,000 / 400,000 updates for 10/50/100%. This is the fairer comparison and is the primary result.

### 7.b Evaluation protocol

Fresh simulator initializations, 5 trials, 4 continuation repetitions, 8 candidate first actions per trial, 250-step horizon. Two candidate sets:

- Fixed behavior actions sampled from the final package.
- Local candidates: the nominal policy mean plus seven perturbations at the policy’s learned action std.

The second action onward follows the frozen policy. Returns include the exact absorbing-failure tail on termination and a reported omitted-tail interval for surviving rollouts.

### 7.c Results

| Budget | Fraction | Transitions | MAE | RMSE | Ranking pair accuracy | Q range across candidates | Return range across candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal updates | 10% | 101,298 | 2.797 | 3.459 | 0.543 | — | — |
| Equal updates | 50% | 505,905 | 2.863 | 3.549 | 0.450 | — | — |
| Equal updates | 100% | 1,000,000 | 2.492 | 3.327 | 0.386 | — | — |
| Equal epochs | 10% | 101,298 | 3.424 | — | 0.507 | 1.10 | 4.12 |
| Equal epochs | 50% | 505,905 | 3.579 | — | 0.543 | 0.59 | 4.21 |
| Equal epochs | 100% | 1,000,000 | 3.204 | — | 0.464 | 0.41 | 4.09 |

The equal-update MAE decreases from 10% to 100% but is not monotonic (50% is worse than 10%), and ranking falls to 0.386. The equal-epoch 100% MAE is the lowest of its three (3.204 vs 3.424/3.579), but ranking is still near chance (0.464), and the Q’s candidate range shrinks with data (1.10 → 0.59 → 0.41) while the measured return range stays ~4.1.

### 7.d Interpretation

- **The data-efficiency hypothesis is not established.** There is a weak MAE improvement at 100% in both budgets, but it is not monotonic, and candidate-action ranking is at or below chance. A larger dataset is not yet producing a more useful action-value critic.
- **The critic is nearly action-insensitive.** Its within-trial spread over local candidates is only ~0.4–1.1, versus a measured return spread of ~4.1. It behaves more like a state-value predictor than a candidate-action discriminator. That is insufficient for the intended future threshold filter.
- **Likely contributors:** the height reward is 0/–1 with only ~5% violations; a single action’s effect on a 250-step return is weak relative to continuation stochasticity; the absorbing-failure value dominates the return scale; the stochastic FQE target averages over next actions; and 4 repetitions may be too few to resolve small candidate effects. Observation-only Q and hidden physical state remain additional limitations.
- **The equal-update pilot is confounded** by unequal epochs; the equal-epoch rerun is the fairer comparison but still does not show the desired ranking.
- **Do not proceed to MPPI/filter integration on this artifact.** The first Q is not validated as an action-ranking critic.

### 7.e Recommended next investigation

1. Measure **paired candidate returns with more repetitions** (e.g., 20–50) at fixed initial states to determine how much of the ~4.1 return spread is genuine candidate effect versus continuation noise.
2. Evaluate the **nominal action only** and a **shorter horizon** (e.g., 20–50 steps) to test whether the critic’s action sensitivity is present near the current step.
3. Consider a per-step violation objective or a constraint-probability objective rather than a long-horizon 0/–1 return; the current reward may be too sparse and failure-dominated for action ranking.
4. Report Q action sensitivity directly (within-state prediction spread, gradient w.r.t. action) before claiming filter suitability.
5. Only after those diagnostics, decide whether to change the FQE objective, use a deterministic/low-variance continuation target, add history/estimators, or phase-condition the reward.

### 7.f Horizon validation — 2026-11-10

Same 100% critic, checkpoints, trials, local candidates, and continuation seeds; only the evaluation horizon changed. The first H steps are identical across horizons because seeds are fixed.

| H | Max omitted tail (`gamma^H/(1-gamma)`) | Aggregate MAE | RMSE | Ranking | Terminal by H | Mean Q | Mean observed return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 8.106 | 3.216 | 4.555 | 0.450 | 0.0% | −7.974 | −6.099 |
| 500 | 0.657 | 3.331 | 4.407 | 0.493 | 3.1% | −7.974 | −6.534 |
| 750 | 0.053 | 3.308 | 4.365 | 0.486 | 4.4% | −7.974 | −6.568 |
| 1000 | 0.0043 | 3.322 | 4.377 | 0.479 | 4.4% | −7.974 | −6.574 |

Aggregate MAE changes by only +0.106 from H=250 to H=1000 (+3.3%), which is within the predeclared <0.2/<10% tolerance, and stabilizes after H=500. However, this aggregate hides a real horizon failure:

- **Late terminations:** 0% of rollouts terminate within 250 steps, 3.1% within 500, and 4.4% within 750/1000. The 250-step evaluation misses seven late-failure rollouts.
- **Paired per-result MAE:** survivors improve slightly (4.071 → 3.991), but late terminations worsen sharply (5.084 → 9.450). Their observed return falls from −11.2 to −15.6 while Q stays around −6.7; Q is optimistic on post-250 failures.
- **Bias:** at H=1000, Q is pessimistic on survivors (mean bias −1.87) and optimistic on late failures (mean bias +8.9). The aggregate MAE cancels these errors.
- **Tail interval containment:** Q lies inside the H-tail interval for 61.9% of cases at H=250, 1.25% at H=500, and 0% at H=750/1000. The 250-step interval is too wide to be diagnostic; once the interval narrows, Q is outside it almost everywhere.

**Conclusion:** H=250 is **not sufficient** for validating the continuing-return/failure behavior, even though aggregate MAE appears stable. The training task’s nominal episodes are 250 steps; continuing beyond them exposes post-episode failures that are absent from the training distribution, and Q does not predict them reliably. Use **H≥500** for this evaluation, and treat the post-250 failure cases as a coverage limitation of the current dataset. A finite-horizon/truncated objective or longer training episodes may be more defensible than extrapolating a continuing-return Q beyond the collected episode length.

## 8. Evaluation-distribution Q experiment — 2026-11-10

### 8.a Purpose

Test whether the original critic’s poor held-out performance was mainly **state-distribution shift** or a more fundamental action-value problem.

### 8.b Data and training

Collected evaluation-distribution transitions using the H=500 protocol:

- 16 trials (seeds 0–15), 4 repetitions, 8 local candidates
- 512 rollouts, **254,927 transitions**
- 6 terminal transitions (~0.0024% of transitions); no timeout transitions
- First action per rollout is a local candidate; later actions follow the frozen policy

Trained a new FQE critic with the same frozen `model_600.pt`, reward (`0.32 ± 0.05 m`), gamma `0.99`, network, and optimizer settings; 100,000 updates, batch 256.

### 8.c Held-out evaluation

New trials **100–104**, same 4 repetitions and 8 local candidates, H=500. Both critics evaluated on identical cases.

| Critic | Training distribution | MAE | RMSE | Mean Q | Mean return | Bias (Q−return) | Ranking | Q range | Return range | Terminal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | Final 1M behavior window | 2.653 | 2.969 | −6.528 | −4.696 | −1.832 | 0.550 | 0.301 | 1.971 | 0% |
| Eval-distribution | 254,927 evaluation rollouts | **1.300** | **1.507** | −5.057 | −4.693 | **−0.364** | 0.486 | 0.583 | 1.962 | 0% |

### 8.d Interpretation

- **Calibration improved substantially.** Training on the evaluation distribution cut MAE by ~51% (2.653 → 1.300) and reduced the pessimism bias from −1.83 to −0.36 on new trials. The original critic’s calibration problem was partly **state-distribution shift**.
- **Action discrimination did not improve.** Ranking stayed near chance (0.550 → 0.486), and the Q range across local candidates (0.58) remains far below the measured return range (1.96). The critic is still mostly a state-value predictor at the candidate scale.
- **Threshold behavior improved only because of calibration.** The original had 8 false rejects at threshold −5 and 7 at −10; the eval-distribution critic had none, but this reflects its closer mean, not better candidate ranking.
- **Terminal/failure coverage remains weak.** The training data contained only 6 terminal transitions, and the new evaluation trials had no terminal rollouts at H=500. The late-failure issue from §7.f is not resolved by this experiment.
- **Conclusion:** distribution shift is one real problem, but it is not the whole problem. The dominant remaining issue is **action sensitivity / reward geometry**, not merely where the states came from. The eval-distribution critic is better calibrated but still not a validated action-ranking filter critic.

### 8.e Batched 1024-trial evaluation — 2026-11-10

The earlier 5-trial comparison was sample-specific. A batched evaluator now creates one 1024-world environment per candidate batch, with the same trial states reproduced across candidate batches and repetitions. This preserves per-trial candidate pairing while using the GPU in parallel.

Default batched protocol: 1024 trials, 4 repetitions, 8 local candidates, H=500, seeded 10,000.

| Critic | Cases | MAE | RMSE | Mean Q | Mean return | Bias (Q−return) | Ranking | Q range | Return range | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original 1M | 8,192 | **2.534** | **3.393** | −6.789 | −6.440 | −0.349 | 0.503 | 0.387 | 2.807 | 1.35% |
| Eval-distribution 254k | 8,192 | 2.972 | 4.516 | −5.477 | −6.440 | +0.963 | 0.495 | 0.618 | 2.771 | 1.32% |

Findings:

- **The small-trial calibration advantage did not hold.** On 5 held-out trials, the eval-distribution critic had MAE 1.300 vs 2.653 for the original. On 1024 fresh trials, the original is better: 2.534 vs 2.972. The eval-distribution critic overfit its 16 training trials / 254k targeted transitions; the original 1M behavior dataset generalizes better.
- **Ranking is chance for both.** 0.503 vs 0.495 with 28,672 pairs; the larger sample confirms there is no reliable candidate ordering. Per-trial swings remain.
- **Action range is still too small.** Q range 0.39–0.62 versus return range 2.77–2.81.
- **The eval-distribution critic is optimistic on the new batch** (bias +0.96 vs −0.35 for the original). At threshold −5, it has 857 false accepts out of 8,192 cases vs 323 for the original; the original has 492 false rejects. Neither is suitable as a threshold filter.
- **Late failures are present at H=500** in the 1024-trial batch (about 1.3% terminal), unlike the five-trial sample. The batched evaluation is therefore more representative.

Implementation: `scripts/evaluate_height_q_batched.py`. The initial-state check compares fresh 1024-world environments and allows only floating-point-level differences.

### 8.f One- and two-step lookahead ranking — 2026-11-10

The batched evaluator records the first two transitions per rollout and computes:

```text
one_step(s, a_c) = r1 + gamma * Q(s1, pi_mean(s1))                    # non-terminal
one_step(s, a_c) = r1 + gamma * failure_value                         # terminal at step 1

two_step(s, a_c) = r1 + gamma * r2 + gamma^2 * Q(s2, pi_mean(s2))     # survives both
two_step(s, a_c) = r1 + gamma * r2 + gamma^2 * failure_value          # terminal at step 2
two_step(s, a_c) = r1 + gamma * failure_value                         # terminal at step 1
```

The scores are averaged over 4 repetitions and ranked against the full measured return on the same 1024 trials. Values below are from the latest seed-10,000 batch; run-to-run Warp float differences are small but not zero.

| Critic | Score | MAE | RMSE | Bias | Ranking | Score range | Return range | False accept / reject at −5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original | `Q(s,a)` | 2.534 | 3.393 | −0.352 | 0.5040 | 0.387 | 2.792 | 323 / 495 |
| Original | one-step lookahead | 2.524 | 3.374 | −0.455 | 0.5083 | 0.540 | 2.792 | 270 / 507 |
| Original | two-step lookahead | **2.495** | **3.356** | −0.334 | **0.5337** | 0.596 | 2.792 | 283 / 439 |
| Eval-distribution | `Q(s,a)` | 2.975 | 4.528 | +0.963 | 0.4950 | 0.618 | 2.798 | 862 / 243 |
| Eval-distribution | one-step lookahead | 2.840 | 4.196 | +0.516 | 0.5060 | 0.872 | 2.798 | 630 / 409 |
| Eval-distribution | two-step lookahead | **2.779** | **4.154** | +0.485 | 0.4983 | 0.663 | 2.798 | 576 / 388 |

Findings:

- **Two-step lookahead is the best tested score for the original critic:** MAE 2.534 → 2.495 and ranking 0.5040 → 0.5337. The monotonic improvement from direct to one-step to two-step suggests the original Q’s state-value component carries some useful signal.
- **The eval-distribution critic improves in calibration but not ranking:** MAE 2.975 → 2.779 and bias +0.963 → +0.485, while ranking stays at chance (0.4983). Its two-step score does not recover candidate discrimination.
- **The action range is still too small:** score range 0.60–0.66 versus return range about 2.8.
- **Threshold behavior improves with lookahead but remains unsafe:** at −5, false accepts fall from 323 to 283 for the original and from 862 to 576 for the eval-distribution critic. At −2, the eval-distribution critic drops from 32 false accepts to 0, but the original still has 11 false rejects.
- **Conclusion:** two-step lookahead is worth one limited follow-up (3–5 steps) for the original critic, because ranking moves above chance. It is not a solution: 0.534 is far from reliable, the eval-distribution critic does not benefit in ranking, and neither is ready for a threshold filter. A shorter-horizon or per-step violation objective, better candidate coverage, and a stronger action-value target remain the more promising directions.

### 8.g Why the ranking test is weak: state support, height observability, and measurement noise — 2026-11-10

**Correction (2026-11-10):** the noise-ceiling/reliability interpretation in (e)
is invalid. Noisy estimated means were treated as truth and sorting repetitions
destroys paired covariance. Retain the historical numbers for provenance only;
do not use their ceilings or repetition-budget forecasts as evidence.

Three separate questions were conflated in the earlier interpretation. Measured separately below.

**(a) The Q never receives height.** The artifact input is the 48-dim actor observation: `base_ang_vel` 3, `projected_gravity` 3, `commands` 3, `dof_pos_obs` 12, `dof_vel` 12, `dof_pos_target` 12, `phase_obs` 2, `phase_frequency` 1. There is no height channel, so "Q evaluated at height 0.4" is not literal; the reset pose is encoded implicitly.

**(b) Height is highly recoverable from that observation** — the physical coupling is real. Predicting the post-action height from the pre-action state on the 1M snapshot (800k train / 200k test):

| Input | MAE | R² |
|---|---:|---:|
| `projected_gravity` only (3) | 1.88 cm | 0.092 |
| `dof_pos_obs` only (12) | 1.45 cm | 0.378 |
| `dof_vel` only (12) | 1.90 cm | 0.034 |
| gravity + joints, linear | 1.24 cm | 0.587 |
| all 48 obs, linear | 1.18 cm | 0.635 |
| all 48 obs, MLP 256×256 | **0.68 cm** | **0.874** |

Height std in the test set is 2.57 cm, so the state determines the height regime; the information is present, not missing. Gravity alone is weak; joint angles carry the most.

**(c) The evaluation states are in support but under-covered.** Standardized 48-dim nearest-neighbour distances to the 1M training set:

| Query | median NN | p90 | max |
|---|---:|---:|---:|
| held-out training rows (baseline) | 2.62 | 3.64 | 6.59 |
| 1024 evaluation initial states | **0.55** | 0.87 | 1.22 |

No evaluation state exceeds the training p99 NN distance, so this is **not extrapolation**. But 100% of their nearest neighbours are `step_id ≤ 2` rows — the reset transient, 1.19% of the dataset. The reset cluster is a narrow, near-degenerate pocket that receives ~1.2% of the training signal.

Heights in the data, for completeness: the 1M snapshot has max 0.4105 m and only 26 rows above 0.40 m (0.0026%); 1.73% exceed 0.37 m. The 250k eval-distribution snapshot has max 0.4059 m and 0.89% above 0.37 m. Height near 0.40 m is essentially absent, matching the earlier observation.

**(d) The Q does not break specifically at the reset cluster.** State-wide check on the 512 eval-distribution rollouts (254,927 transitions), comparing `Q(s,a)` with the measured discounted return from each state under the same reward and gamma:

| Slice | n | MAE | Bias | R² | corr |
|---|---:|---:|---:|---:|---:|
| all | 254,927 | 3.17 | −0.79 | **0.511** | **+0.733** |
| `step_id` 0–2 (reset) | 1,536 | 3.80 | −0.28 | 0.396 | +0.633 |
| `step_id` 101–499 | 203,215 | 3.24 | −0.98 | 0.525 | +0.745 |
| inside band | 243,001 | 2.96 | −0.84 | 0.389 | +0.652 |
| below band | 9,656 | 8.31 | +0.09 | 0.546 | +0.772 |

The Q carries substantial state-value signal (R² 0.51) and is *not* systematically mis-calibrated at the reset states (bias −0.28 there, versus −0.98 at steady state). So the near-chance ranking is not explained by the Q being blind or wrong in the reset region.

**(e) The ranking measurement is noise-dominated.** From the 1024-trial × 8-candidate × 4-repetition returns:

| Quantity | Value |
|---|---:|
| within-(trial, candidate) repetition std | 2.15 |
| within-trial candidate-mean std | 0.95 |
| std of a 4-repetition mean estimate | 1.07 |
| within-trial split-half (2 vs 2) reliability | 0.555 |
| Spearman–Brown reliability, 4 repetitions | 0.714 |
| variance share: between-trial / between-candidate / repetition | 78% / 4.2% / 18% |

A perfect predictor of the true candidate means, scored against the measured 4-repetition means, reaches only **0.678** pairwise ranking accuracy. The corresponding ceilings are 0.717 / 0.757 / 0.792 / 0.824 at 8 / 16 / 32 / 64 repetitions.

Interpretation: the frozen policy's own action stochasticity over 500 steps contributes an order of magnitude more variance than the candidate's first action (repetition variance 6.79 versus within-trial candidate variance 1.58). Candidate differences are real but small relative to the continuation noise, so 4 repetitions cannot resolve ranking well. The two-step score at 0.534 is below the 0.678 ceiling but not far enough below it to conclude the critic is useless for ranking; the experiment as configured cannot separate a good critic from a mediocre one.

Consequences for the next iteration:

- Raise repetitions to 16–32 before drawing further ranking conclusions (ceiling 0.76–0.79), or reduce label noise directly (shorter label horizon, or common random numbers across candidates at the trajectory level).
- Report the ceiling alongside every ranking number, otherwise "near chance" is uninterpretable.
- Do not invest in more lookahead depth for the ranking question until the measurement floor is fixed; 3–5 step lookahead on top of a 0.68-ceiling test would not be decisive.









## 9. Mid-gait insertion experiment — 2026-11-10

**Superseded evidence:** this historical evaluator varied prefix states across
repetitions, generated candidates at reset, overwrote direct predictions, lost
prefix survival, and restored incomplete/clamped state. Its fixed-state variance,
ranking-resolution, exact-restore, and first-reward independence claims are not
validated. Rates must be computed from raw binary rewards, never thresholded
averages. The corrected experiment below must pass native restoration gates.

### 9.a Motivation

§8.g leaves three facts that make the reset-state ranking test uninformative:

- 100% of evaluation queries land in the reset cluster, which is 1.19% of the training data and where the Q's state-value calibration is already fine.
- At the reset pose the robot is falling from 0.40 m, so the first two steps are ballistic and the candidate action cannot change the first reward.
- Four repetitions put the pairwise-ranking ceiling at 0.678.

The first two are properties of *where* the Q is queried, and both disappear if the candidate actions are inserted mid-gait.

### 9.b Method

`scripts/evaluate_height_q_batched.py` gains `--insert_step k` (default 0, the original protocol).

```text
reset --(frozen policy, k steps)--> insertion state --(candidate a_c)--> H-step measured return
```

- `Go2Trot.capture_state()` / `Go2Trot.restore_state()` snapshot and replay the full task state (DOF position/velocity, root state, command, phase, phase frequency, target, episode counter, common step counter) plus the physics commit, and clear the per-episode termination flags.
- The first candidate of a repetition runs the prefix and snapshots the result; the rest restore the snapshot. **Re-running the prefix is not equivalent**: see the deviation note below.
- The insertion observation is drawn from a fresh generator seeded only by the repetition, so all candidates see identical observation noise.
- `predictions[candidate] = Q(s_insert, a_c)`; the measured return, one-step, and two-step scores all start at the insertion step. Prefix rewards are excluded.
- `episode_length_s` is extended to cover `(k + H) * dt + 1` so an administrative timeout cannot reset a world mid-measurement.
- Worlds that terminate during the prefix are excluded from all aggregates and counted in the report.
- `insert_step = 0` keeps the original code path unchanged and reproduces its numbers exactly.

### 9.b.1 Deviation: replaying the prefix does not reproduce the state

The first implementation re-ran the prefix per candidate with repetition-scoped seeds, assuming that would reproduce the state. It does not:

| Prefix length | max abs state difference between two identical replays |
|---:|---|
| 20 steps | 1.6e-6 m height, 9.6e-4 in `base_ang_vel` |
| 50 steps | 2.6e-2 m height, 6.8 in `dof_vel` |
| 100 steps | 2.5e-2 m height, 4.7e1 in `dof_vel` |

Isolation showed the policy network and environment construction are both bit-reproducible (identical `mean_actions` on repeated calls; identical friction, mass, gains, commands, and phase for an env built after an unrelated rollout). With a constant action the physics drifts only 2.4e-5 in 100 steps, so the state is quasi-deterministic; the divergence appears only when the policy closes the loop, which amplifies the residual solver noise. The replay approach is therefore abandoned in favour of snapshot/restore.

Restoring is exact: two restores of one snapshot followed by one step differ by 2e-5 in observation and 3e-8 m in height, i.e. the same per-step Warp noise as any step. Restore also clamps joint positions to the asset limits (`reset_state` contract), which moved `dof_pos` by up to 8e-3 rad; the clamp is deterministic and identical for every candidate, and the affected joints sit at their limits during a trot.

Teleport fidelity against a true continuation, one step after restore with the same action: height 2.3e-5 m, `dof_pos` 8e-3 rad, `dof_vel` 1.2 rad/s. The state is self-consistent and in-distribution (height 0.32 m, 99% in band), but it is not the uninterrupted trajectory, so absolute returns are not compared against k=0 runs.

### 9.c Acceptance checks

| Check | Result |
|---|---|
| `insert_step = 0` reproduces the previous smoke metrics exactly | 0.5223 ranking, 4.2254 MAE — identical |
| Insertion state reproduced across candidates | passes at `rtol=1e-4, atol=1e-5` because the state is restored, not replayed |
| Two restores of one snapshot agree | 2e-5 observation, 3e-8 m height after one step |
| Insertion point is mid-gait, not transient | at `k = 100`, height 0.3197 m, 99.2% inside band |
| `capture_state` does not alias live buffers | unit test in `gym/envs/go2/test_go2trot.py` |

### 9.d Protocol

`k = 100` (2 s, steady-state gait, mid-command since commands resample every 3 s), `H = 500`, 1024 trials, 16 repetitions, 8 candidates, seed 10000. Runs: original critic at `k = 0` (control) and `k = 100`, plus the eval-distribution critic at `k = 100`. The `k = 0` control at 16 repetitions separates the effect of the insertion point from the effect of the repetition count.

### 9.e Results

All runs: 1024 trials, 8 candidates, H=500, seed 10000, original critic unless stated. Pairwise-ranking 95% CIs are trial-level bootstraps (400 resamples over trials).

| Run | Score | MAE | Bias | Ranking | 95% CI | Ceiling (16 reps) |
|---|---|---:|---:|---:|---|---:|
| `k = 0` original | direct | 2.326 | −0.340 | 0.4985 | [0.4900, 0.5083] | 0.686 |
| `k = 0` original | one-step | 2.317 | −0.443 | 0.5212 | [0.5111, 0.5319] | 0.686 |
| `k = 0` original | two-step | 2.297 | −0.318 | 0.5395 | [0.5296, 0.5494] | 0.686 |
| `k = 100` original | direct | 2.045 | +0.714 | 0.5179 | [0.5088, 0.5268] | 0.655 |
| `k = 100` original | one-step | 2.016 | +0.029 | 0.5220 | [0.5134, 0.5324] | 0.655 |
| `k = 100` original | two-step | **1.904** | +0.652 | **0.5348** | [0.5261, 0.5439] | 0.655 |
| `k = 100` eval-dist | direct | 2.494 | +0.304 | 0.5053 | [0.4966, 0.5130] | 0.655 |
| `k = 100` eval-dist | one-step | 2.384 | −0.174 | 0.5192 | [0.5112, 0.5276] | 0.655 |
| `k = 100` eval-dist | two-step | 2.276 | +0.225 | 0.5234 | [0.5140, 0.5319] | 0.655 |

No trials were excluded at either insertion point.

Continuation noise, which sets the ceiling:

| | repetition sd | within-trial candidate sd | sd of a 16-rep mean | target mean |
|---|---:|---:|---:|---:|
| `k = 0` | 3.00 | 0.76 | 0.75 | −6.45 |
| `k = 100` | 5.02 | 0.89 | 1.26 | −5.03 |

Findings:

- **Mid-gait insertion did not rescue ranking.** Two-step moves 0.5395 → 0.5348, i.e. not at all. Direct `Q(s,a)` moves from chance (0.4985, CI includes 0.5) to 0.5179 (CI excludes 0.5), but that is a shift of two points, not a fix.
- **Every ranking number is now statistically resolved and none is near its ceiling.** The ceilings are 0.686 and 0.655; the critic reaches 11–22% of the achievable margin over chance ((acc − 0.5) / (ceiling − 0.5)): 0.0 / 11% / 21% at `k = 0`, and 12% / 14% / 22% at `k = 100`.
- **The one-step lookahead fixes bias but not ranking.** At `k = 100` the direct bias is +0.714 and the one-step bias is +0.029, the best calibration of any configuration tested, while ranking moves only 0.5179 → 0.5220.
- **The eval-distribution critic remains worse than the original** at the same states: direct 0.5053 vs 0.5179 and MAE 2.494 vs 2.045. Training on 16 evaluation trials did not help even when queried on that same kind of mid-gait state.
- **MAE is not comparable across `k`.** The target distribution differs (mean −6.45 at `k = 0` versus −5.03 at `k = 100`), and the mid-gait state is reached by teleport, not by the uninterrupted trajectory. Only ranking and error-relative-to-target are interpretable across runs.
- **The mid-gait continuation is much noisier** (repetition sd 5.02 versus 3.00), so its ranking ceiling at 16 repetitions is *lower*, not higher. Sharpening this test further needs roughly 64 repetitions to reach a 0.79 ceiling.

Conclusion: querying the critic where the data actually lives, where the robot is in contact, and where the action can change the height within a few steps still yields weak action discrimination. The limiting factor is the critic's action sensitivity, not the evaluation distribution and not the measurement precision. A usable filter would need a different value target or a different learning signal, not another insertion point.

### 9.f First-step reward distribution

The reward at the insertion step is binary (`0` inside `0.32 ± 0.05 m`, `-1` outside) and is **independent of the candidate action** at both insertion points:

| | `k = 0` (reset) | `k = 100` (mid-gait) |
|---|---:|---:|
| insertion height mean | 0.3961 m | 0.3203 m |
| inside band at insertion | 0.0% | 97.4% |
| P(first reward = −1) | **1.0000** | **0.0117** |
| P(first reward = 0) | 0.0000 | 0.9883 |
| per-candidate −1 rate (8 candidates) | all 1.0000 | all 0.0117 |
| trials where all 8 candidates share one value | 1024 / 1024 | 1024 / 1024 |
| trials always −1 / always 0 / mixed | 1024 / 0 / 0 | 12 / 1012 / 0 |
| P(first-step termination) | 0.0 | 0.0 |

A single 20 ms control step moves the base by about 1 cm, while the band is 10 cm wide, so the binary reward is decided by the insertion height rather than by the candidate. The 12 always-outside trials have insertion heights 0.248–0.272 m, i.e. already below the band before the candidate is applied.

Consequence: the immediate reward carries **no** candidate information at either insertion point, so every bit of ranking signal has to come from the multi-step return. That is consistent with one-step lookahead being almost identical to direct `Q(s,a)` and two-step lookahead gaining only ~0.02–0.03. It also means the band reward is a weak per-step learning signal for action discrimination; a continuous height-error shaping term would carry candidate information in the first step, if that is what future experiments need.

The distribution is critic-independent: candidates come from the policy checkpoint and the environment, not from the Q artifact, so both critics see identical first-step rewards.

### 9.g How diverse are the k=100 insertion states

Reproducing repetition 0's prefix (1024 worlds, seed 10000) and comparing the captured 48-dim observations against the 1M training snapshot:

| Observation group | k=100 std | training std | ratio |
|---|---:|---:|---:|
| `base_ang_vel` | 3.234 | 3.129 | 1.03 |
| `projected_gravity` | 0.0311 | 0.0461 | 0.67 |
| `commands` | 0.2434 | 0.2411 | 1.01 |
| `dof_pos_obs` | 0.1400 | 0.1451 | 0.96 |
| `dof_vel` | 1.8292 | 1.8741 | 0.98 |
| `dof_pos_target` | 1.0350 | 1.0518 | 0.98 |
| `phase_obs` | 0.7072 | 0.7071 | 1.00 |
| `phase_frequency` | 0.5746 | 0.5803 | 0.99 |

Per-dimension standard deviations, averaged over components.

Distance structure in standardized 48-dim space:

| Set | nearest training neighbour (median) | pairwise among the set (median) |
|---|---:|---:|
| 1024 reset states (`k = 0`) | **0.55** | ≈0 (identical pose) |
| 1024 insertion states (`k = 100`) | **2.52** | **9.10** |
| held-out training rows | 2.62 | 9.10 (400 random rows) |

Physical spread at `k = 100`: height std 2.16 cm (training 2.56 cm), range 0.193–0.402 m; mean absolute tilt about 2.2°; mean per-joint position range about 1.01 rad across trials.

Interpretation:

- **The insertion states are an ordinary sample of the training distribution**, not a dense pocket. Their nearest-neighbour distance (2.52) matches held-out training rows (2.62), and their pairwise spread (9.10) equals that of 400 random training states (9.10). Compare the reset cluster at 0.55, which is near-duplicate territory.
- Every channel except `projected_gravity` matches the training spread to within 5%. The gravity ratio of 0.67 means the mid-gait states are more upright than the average training state, which is expected because the training data includes falls and tilts.
- This closes the state-coverage objection: the earlier ranking results cannot be blamed on querying a narrow or degenerate state family. At `k = 100` the Q is queried on a representative set of states and still ranks at 0.52–0.53.

