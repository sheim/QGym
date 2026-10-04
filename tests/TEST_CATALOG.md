# Regression test catalog

This catalog records the purpose and review disposition of the portable unit
tests and the colocated `gym` and `learning` tests. It is a behavior review, not
a test-count target. Generated collected cases, parameter IDs, markers,
fixtures, source hashes, and available timing evidence live in
`logs/streamlining/test_inventory.json`; generated artifacts are not committed.
The implementation sequence is in [STREAMLINING_PLAN.md](../STREAMLINING_PLAN.md).

## Reading the review

**Keep** means the test protects a supported behavior. **Consolidate** and
**strengthen** are future review work, not authorization to delete the current
coverage. A replacement must still detect the named counterexample on the same
relevant backend/task axes. Two tests are not redundant merely because both
assert a shape, use a mock, or exercise a reset.

Oracle descriptions distinguish:

- **Independent:** an analytic prediction, hand-calculated example, recorded
  external reference, or native measurement through a separate code path.
- **Metamorphic:** permutation, isolation, invariance, or controlled changes to
  inputs with a predicted relation between outputs.
- **Differential:** agreement with a separate implementation. CPU/Warp agreement
  alone cannot expose an error in their common model loader.
- **Interaction:** a recording fake checks messages, ordering, or protocol
  propagation. It does not establish real simulator or learner behavior.
- **Static:** declarations, defaults, or schema properties. Keep these only when
  they protect an intentional public contract; avoid preserving implementation
  choices or testing a fixture against itself.

Valid-state shape/type assertions belong in tests where they establish the
backend or policy interface. Tests must not require new runtime shape/type
guards. Invalid-input cases that only preserve a custom internal assertion
should be reconsidered with that assertion; errors expressing a real supported
API, physical parameter, or experiment contract have a different purpose.

The case inventory records collection separately from execution. Optional
cases are collected without running fixtures. A missing duration or outcome is
**unmeasured**, not a pass or a zero-cost test. Timing review must include setup,
call, and teardown, and compare matching source/dependency revisions.

## Backend state, reset, and frame conventions

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_backend_contract.py](unit_tests/test_backend_contract.py) | Real pendulum CPU/Warp/VSim: public schema, live writable state, independent worlds, selected reset, signed gravity and torque response. Schema plus metamorphic and physical oracles. | **Keep.** Gravity now checks the predicted sign in every world for both horizontal poses; the previous absolute-velocity check accepted reversed gravity. This is the shared fixed-base contract, not a replacement for floating-base contacts. |
| [test_legged_backend_contract.py](unit_tests/test_legged_backend_contract.py) | CPU/Warp floating-base schema, quaternion convention, friction slots, body-origin velocity, limit enforcement, reset persistence, and trajectory agreement. Native and differential oracles alongside schema checks. | **Consolidate later:** repeated shape/count/identity methods into a backend-parametrized floating-base schema case. Preserve body-origin velocity and material semantics; a fixed pendulum cannot cover them. |
| [test_vsim_legged_contract.py](unit_tests/test_vsim_legged_contract.py) | Licensed VSim floating-base contract, support against gravity, weight-carrying contact forces, and partial reset. Physical weight and state-isolation oracles. | **Consolidate later** only the repeated schema/identity cases with the floating-base suite. Keep VSim contact weight and no-ground-penetration evidence. |
| [test_root_velocity_frames.py](unit_tests/test_root_velocity_frames.py) | CPU/Warp/VSim world angular velocity under nonidentity yaw/tilt, reset and root-only writes, sparse reset, and cached stepping. Native object velocities and independently constructed rotation increments expose cancelling read/write frame errors. | **Keep.** A quaternion round trip or CPU/Warp differential test is not replacement coverage. |
| [test_root_velocity_observation.py](unit_tests/test_root_velocity_observation.py) | Full Go2Trot CPU/Warp actor and critic body-frame velocities at 100 Hz, with cached references and contact-free motion. Independent native body velocities. | **Keep.** Covers task observation assembly after the backend boundary; intentionally overlaps the frame suite at a different layer. |
| [test_mujoco_cpu_reset_liveness.py](unit_tests/test_mujoco_cpu_reset_liveness.py) | CPU setup/reset publishes current body pose, velocity, and contacts immediately; selected worlds only; no simulation-time advance. Native state and contact oracles; forwarding spies count native refreshes. | **Keep.** Spies supplement physical/state assertions and do not replace them. Immediate reset publication is distinct from step liveness. |
| [test_task_state_liveness.py](unit_tests/test_task_state_liveness.py) | CPU/Warp task-held root/body/DOF references evolve without a getter refreshing them, and atomic reset preserves the requested spawn height. Metamorphic cached-state oracle. | **Keep; isolation repaired.** Config pairs are deep-copied and every created backend closes in `finally`. Future consolidation with VSim must preserve cached-reference inspection before getters. |
| [test_vsim_task_state_liveness.py](unit_tests/test_vsim_task_state_liveness.py) | Licensed VSim task-held root and body buffers evolve and alias the backend's persistent state. | **Keep; isolation repaired.** Fresh copied configs and fixture cleanup in `finally`. Shares the liveness scenario, but currently lacks the CPU/Warp DOF-cache assertion; do not call them exact duplicates. |
| [test_task_reset_observations.py](unit_tests/test_task_reset_observations.py) | CPU/Warp selected resets refresh actor/critic derived quantities while preserving unselected worlds; empty reset leaves caches intact. Per-world sentinels and observation comparisons. | **Keep.** Backend native refresh alone does not prove task-derived observation freshness. |

## Physics, contacts, assets, and canonical routing

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_cross_backend_physics.py](unit_tests/test_cross_backend_physics.py) | CPU/Warp pendulum trajectories from identical varied initial states. Tight differential oracle over a long rollout. | **Keep.** Measure cost before shortening; pair with analytic sign/energy tests because shared MuJoCo mistakes can agree. |
| [test_mujoco_cpu_physics.py](unit_tests/test_mujoco_cpu_physics.py) | Damping dissipates pendulum mechanical energy and converges. Independent energy/settling behavior across damping values. | **Keep.** Physical correctness is not established by stable finite tensors alone. |
| [test_vsim_parity.py](unit_tests/test_vsim_parity.py) | VSim small-angle period against analytic mechanics; damped energy envelope against CPU. Independent and differential oracles. | **Keep.** The two oracles detect different errors. Replace mandatory-MuJoCo `importorskip` in a later cleanup; explicitly requested evidence must not disappear. |
| [test_legged_termination.py](unit_tests/test_legged_termination.py) | CPU/Warp upside-down drop produces nonzero base contact force through task state. Real collision/contact oracle. | **Keep; isolation repaired.** Config copies and guaranteed backend close. The current name says termination, but the assertion proves contact publication, not `terminated`; strengthen or rename separately. |
| [test_vsim_legged_termination.py](unit_tests/test_vsim_legged_termination.py) | The same base-contact publication through VSim link sensors and task tensors. | **Keep; isolation repaired.** Preserve backend-specific sensor coverage. Same termination-versus-contact naming limitation as the CPU/Warp test. |
| [test_robot_layout.py](unit_tests/test_robot_layout.py) | Canonical names and URDF order, permuted DOF/motor/sensor routing, robot-only body order, and named gait trajectory wrapping. Permutation sentinels and known names. | **Keep.** Review the mock-backend metadata-only case for consolidation with schema coverage. Native torque expectations derived from a production map need the independent permutation cases alongside them. |
| [test_urdf_limits.py](unit_tests/test_urdf_limits.py) | Actual mini-cheetah effort/velocity values reach canonical DOF properties instead of permissive fallback limits. Independent known asset values. | **Keep.** The finite/bounded check can be folded into the exact-value case if every actuator remains covered. |
| [test_mini_cheetah_inertias.py](unit_tests/test_mini_cheetah_inertias.py) | URDF principal moments are positive and satisfy physical triangle inequalities; base inertia matches the external model. Physical invariant plus reference values. | **Keep.** Exact reference values are an intentional asset contract, unlike arbitrary benchmark defaults. |
| [test_vsim_asset.py](unit_tests/test_vsim_asset.py) | Converted XML retains collision geometry and joint limits, injects motors/sensors/dynamics, and resolves mesh paths. XML fixture and mutated-input sentinels; no license required. | **Keep.** Not a VSim engine execution test. Fixed-base collision geometry protects the invisible-pendulum regression. |

## Domain randomization lifecycle and physical application

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [base/test_domain_randomization.py](../gym/envs/base/test_domain_randomization.py) | Pure sampler: isolated seeded streams, startup-only friction/mass, independent PD axes, selected/empty reset, and nominal-relative gain rebuilding. Recording backend plus metamorphic comparisons. | **Keep.** Physical-range and missing-required-config errors are intentional contracts. Recording calls do not prove native physical application; that is covered below. |
| [test_domain_randomization.py](unit_tests/test_domain_randomization.py) | CPU/Warp/VSim applied friction/mass/gains, startup ownership, selected task/gait reset buffers, and predicted friction/acceleration effects. Native readbacks plus real physical response. | **Keep; consolidate shared scenario setup later.** Preserve both parameter readback and dynamics checks. The multiccd crash-pose regression is separate evidence, not a cosmetic config assertion. |
| [test_vsim_domain_randomization_regression.py](unit_tests/test_vsim_domain_randomization_regression.py) | VSim physical-axis set topology, nominal-range equivalence, sparse/empty reset isolation, and per-world weight normalization at 100 Hz. Native topology and controlled comparisons. | **Keep.** Pure topology proxy tests cannot replace execution of the actual production randomization path. |
| [go2/test_go2trot.py](../gym/envs/go2/test_go2trot.py) | Contact strength normalizes using each world's applied total mass. Two hand-calculated worlds with different mass. | **Keep.** Small local example complements the VSim end-to-end weight test. |

## Task registry, scaling, rewards, and actions

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_task_registry.py](unit_tests/test_task_registry.py) | Exact manifest registration, explicit learning-class resolution, selected reset binding, absence of unsupported legacy config, and an actual CPU action step for every declared task. Manifest/static and integration oracles. | **Keep.** The stale `collapse_fixed_joints` declaration was removed and this gate passes in the recorded baseline. One step does not establish runner construction or learning. |
| [test_task_skeleton.py](unit_tests/test_task_skeleton.py) | Named state concatenation/distribution, scaling round trips, and reset flags with a minimal real task subclass. Hand-calculated values and round trips. | **Keep; one assignment-only test removed.** Reassess `test_set_states_wrong_total_dim_raises` with any removal of its custom runtime assertion; preserve correct slicing/scaling behavior without requiring that assertion type. |
| [test_go2trot_command_bounds.py](unit_tests/test_go2trot_command_bounds.py) | Full desired positions include gait/default offsets, legal requests are unchanged, extreme raw samples cannot poison applied observations/history/rewards, and limits follow permuted actuators with passive DOFs. Real task plus adversarial numeric/permutation cases. | **Keep.** Protects the demonstrated training feedback instability; preserving raw policy samples and finite applied history are separate obligations. |
| [test_mini_cheetah_ref_vsim_config.py](unit_tests/test_mini_cheetah_ref_vsim_config.py) | Position-limit penalty normalization, squared rather than fourth-power yaw error, and axis-aligned command support. Hand-calculated rewards and sampled distribution properties. | **Keep.** Revisit the probabilistic sample threshold only with the intended command distribution; the class name does not imply a licensed engine dependency. |
| [test_coupling.py](../gym/envs/mit_humanoid/test_coupling.py) | Humanoid coupling matches an external recorded numeric example; sampled history advances selected worlds without touching others. External reference and isolation sentinels. | **Keep.** Preserve reference provenance. The history expectation uses `torch.roll`, so selection sentinels and explicit newest-slot checks carry the independent value. |

## Evaluation, physical metrics, and analysis

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_legged_eval_metrics.py](unit_tests/test_legged_eval_metrics.py) | Named command coverage, full applied targets, known tracking/contact/saturation metrics, fixed commands after resets, and balanced impulse scheduling. Hand-built signals and explicit numerical expectations. | **Keep.** Consolidate trivial command-position constants only if command support and balance remain checked. |
| [test_legged_signal_analysis.py](unit_tests/test_legged_signal_analysis.py) | Synthetic height drift, a balanced trot, per-world mass normalization, and foot-relative clearance yield known metrics. Independent signal constructions. | **Keep.** Provides interpretation oracles that real-rollout smoke tests cannot supply. |
| [test_compare_policy_observations.py](unit_tests/test_compare_policy_observations.py) | Comparison excludes samples after either run terminates and aggregates named components correctly. Hand-calculated masks and zero-gap metamorphic case. | **Keep.** Identical-input checks alone are weak; retain the nontrivial mask/aggregation cases and add shifted inputs only when changing distance formulas. |
| [test_eval_applied_actions.py](unit_tests/test_eval_applied_actions.py) | Actual evaluator artifact distinguishes bounded applied commands from extreme raw policy outputs at 100 Hz. Controlled policy with a real CPU task and saved NPZ. | **Keep.** Complements task-side bounds by checking the data consumer and artifact contract. |
| [test_eval_domain_randomization_trajectory.py](unit_tests/test_eval_domain_randomization_trajectory.py) | Center-of-mass velocity includes angular motion of a local COM offset; subprocess commands preserve trajectory protocol. Hand-calculated mechanics plus interaction checks. | **Keep.** A command-list check does not prove a completed trajectory run. |
| [test_validate_domain_randomization.py](unit_tests/test_validate_domain_randomization.py) | Friction mode before setup, known distribution shifts, missing-pair versus marginal statistics, and explicit comparison inputs. Numerical and protocol oracles. | **Keep.** Paired/marginal distinction protects interpretation under missing evaluations. |
| [test_policy_io.py](../gym/utils/test_policy_io.py) | Observation/action units, canonical component/history labels, first-episode masks, and phase-bin statistics. Hand-calculated conversions and masks. | **Keep.** Reassess shape-rejection-only coverage if the underlying analysis API changes; keep valid-data unit and ordering assertions. |
| [test_compare_backend_unapplied_actions.py](unit_tests/test_compare_backend_unapplied_actions.py) | Open-loop comparison reduces correct axes, finds first activity, forwards zero-motor settings, and rejects artifacts with applied actions. Numeric and artifact/protocol sentinels. | **Keep.** Artifact rejection preserves the experiment's independent variable; it is not a generic internal tensor guard. |

## Evaluation CLI and saved configuration

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_eval_go2_policy.py](unit_tests/test_eval_go2_policy.py) | Explicit/latest checkpoint selection, engine/device mapping, and policy-I/O recording command construction. Temporary files and CLI interaction checks. | **Keep; consolidate parser-only cases if helpful.** No claim of actual policy quality from command construction. |
| [test_original_cfg.py](unit_tests/test_original_cfg.py) | Saved config inheritance loads without contaminating live modules, selected run stays pinned, and carried snapshots take precedence. Real temporary Python modules/files. | **Keep.** Protects reproducibility; source-config loading is not model/optimizer checkpoint-resume coverage. |
| [test_run_config_diff.py](../gym/utils/test_run_config_diff.py) | Config diff follows only relevant imports and honors carried snapshots. Controlled file trees and expected diffs. | **Keep.** Snapshot precedence overlaps the loader at a different consumer; share fixtures only if they stay simple. |

## Benchmark protocols and campaign evidence

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_benchmark_domain_randomization.py](unit_tests/test_benchmark_domain_randomization.py) | Reset masks cover worlds at the declared rate, native set counts are reported, and fixed-friction experiments retain physical DR topology. Counting and recording fakes. | **Keep; strengthen configuration-only topology checks with the existing native regression.** No CI speed assertion is inferred from these tests. |
| [test_benchmark_vsim_environment_sets.py](unit_tests/test_benchmark_vsim_environment_sets.py) | Valid set partitions preserve environment count, topology proxy forwards other operations, nominal physics and 100 Hz remain fixed, and trials are nonempty. Partition invariants and interaction tests. | **Keep.** Proxy and config coverage are controls, not performance or native-engine evidence. |
| [test_benchmark_simulation.py](unit_tests/test_benchmark_simulation.py) | Paired timing gates distinguish improvement, regression, noise, insufficient pairs, incompatible workloads, and profiled results. Different requested initial physics invalidates a comparison; roundoff in derived contact readings does not. Actual CPU workloads restore repeatable state and schedule exactly the intended per-world reset rate. Controlled samples plus physical/state replay oracles. | **Keep.** The synthetic gate tests do not assert CI machine speed; the real workload cases protect reset and restore controls. CPU replay does not prove restoration of opaque native VSim solver history. |
| [test_profile_simulation.py](unit_tests/test_profile_simulation.py) | Host profiles retain timed-stack weights in data and escaped SVG output; CUDA summaries distinguish overlapping work from elapsed time; capture commands preserve the bounded protocol; both profilers mark results ineligible for speed gates. Synthetic records, parsed SVG, and a temporary SQLite database provide numerical and artifact oracles. | **Keep.** Rejecting actual marked artifacts catches accidental use of profiler overhead in speed comparisons. Command construction does not establish that a real profiler captured the requested region. |
| [test_domain_randomization_campaign.py](unit_tests/test_domain_randomization_campaign.py) | Seed pairing, balanced command/friction samples, artifact identity/completeness, source provenance, resume constraints, and failed/finalized cell handling. Deliberately missing/changed artifacts and hand-calculated effects. | **Keep; consolidate/strengthen selectively.** Large fake-registry cases prove config isolation/wiring only. Replace arbitrary exact warmup defaults with intended measurement invariants when that protocol changes. |
| [test_full_domain_randomization_campaign.py](unit_tests/test_full_domain_randomization_campaign.py) | Training/evaluation scope, source checkpoint identity/finiteness, capacity failures, stop handling, interleaved scheduling, and failed-training dependency gates. Temporary artifacts and controlled child-process/state fakes. | **Keep; one self-fixture test removed.** Fixed matrix counts are interpretable only alongside uniqueness and dependency coverage. Scheduler fakes deliberately avoid expensive training while exercising failure transitions. |

## Learning, inference, math, and logging

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_runner_inference.py](unit_tests/test_runner_inference.py) | Supported runner inference paths use clean observations and runner reset reuses the task-owned mask. Recording fake actors and mask sentinels. | **Consolidate later** equivalent runner cases while retaining off-policy action transformation. These tests do not prove optimizer, storage, or save/load behavior. |
| [test_regression_pendulum_training.py](unit_tests/test_regression_pendulum_training.py) | The registered CPU pendulum constructs its actual PPO runner, performs updates, saves and resumes optimizer state, and emits finite evaluation artifacts at 100 Hz. Independent synthetic catch signals, an exact initial-state grid, saturated torque inputs, and seeded network comparisons check evaluation and initialization controls. | **Keep.** This closes the runner/update/checkpoint smoke gap without repairing the fixture's config. It does not establish a learned catch-rate threshold or cover normalizer lifecycle, since this protocol disables normalization. |
| [test_normalize.py](../learning/modules/utils/test_normalize.py) | Running mean/variance combination and normalization on known input. Hand-calculated scalar moments. | **Keep; coverage gap remains** for update-versus-freeze timing and serialization/resume, especially repeated optimizer passes over the same rollout. |
| [test_usecase.py](../learning/utils/PBRS/test_usecase.py) | Potential-based shaping produces known rewards across pre/post-step potentials. Small real interface example with numeric expectations. | **Keep.** It is not a substitute for PPO/GAE return tests. |
| [test_logger.py](../learning/utils/logger/test_logger.py) | Finished-episode and iteration values aggregate correctly and counters clear. Controlled logger inputs. | **Keep.** Check global logger isolation when introducing real runner tests. |
| [test_utils.py](../learning/utils/tests/test_utils.py) | Horizon conversion and zero-weight reward removal. Small hand-calculated utility examples. | **Keep; combine adjacent tiny cases if useful.** Preserve physical-time discount semantics under frequency changes. |

## Deployment, bootstrap, and threading

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_deploy_optional_sdk.py](unit_tests/test_deploy_optional_sdk.py) | Simulation/offline imports and help work with SDK/DDS deliberately unavailable; hardware invocation reports the optional extra. Isolated subprocess import blockers. | **Keep.** Optional-dependency boundaries must work even on a machine where the SDK happens to be installed. |
| [test_go2_deploy_obs.py](unit_tests/test_go2_deploy_obs.py) | Canonical deployment joint/gyro/gravity ordering and scaling, action mapping, and CPU-task observation parity. Hand-built messages plus real task comparison. | **Keep.** Strict full-observation xfail remains explicit debt: residual clipping differs from full gait/default target projection. Other parity assertions must stay active. |
| [test_unitree_sdk.py](unit_tests/test_unitree_sdk.py) | Real optional SDK/DDS/CRC imports, default command contents, damping-only emergency command, and CRC response to a changed motor field. Actual native dependency and message mutation. | **Keep; Linux opt-in, no robot connection.** Recomputing CRC with the same SDK tests integration, not independent checksum algorithm correctness. Mutation coverage prevents a constant checksum from passing. |
| [test_recurrent_thread.py](unit_tests/test_recurrent_thread.py) | Real Linux timer/thread lifecycle, periodic callbacks, timeout behavior, failure results, descriptor close, and replacement-instance cleanup. Observed OS resources plus narrowly injected errors. | **Keep.** Broadly mocking timers would lose the resource/lifecycle oracle; elapsed-time bounds are scheduler-sensitive and should be reviewed with measured cost. |
| [test_fetch_unitree_sdk.py](unit_tests/test_fetch_unitree_sdk.py) | Official-pin bootstrap mechanics preserve existing revisions and staged/unstaged/untracked work, tolerate ignored caches, and report failures. Temporary local Git repositories, offline. | **Keep.** Exercises actual Git behavior rather than comparing generated command strings. |

## UI, geometry, and common math

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_teleop_keyboard.py](unit_tests/test_teleop_keyboard.py) | Key bindings, held-key edges, reset/exit behavior, command axes/limits, and viewer callback/UI wiring. Event fakes and explicit command changes. | **Consolidate later** repeated declaration/default cases. Keep key-edge and axis behavior. Replace mandatory-MuJoCo import skips separately. |
| [test_vsim_command_visualizer.py](unit_tests/test_vsim_command_visualizer.py) | Yaw/arrow/arc geometry, robot-heading transforms, visibility, and drawing-hook wiring. Analytic geometry and lightweight fake viewer. | **Consolidate later** equivalent geometric examples, not all viewer tests. Names are VSim-specific but pure geometry requires no license. |
| [test_math.py](../gym/utils/math/test_math.py) | Angle wrapping, bounded random sampling, and exponential-average update. Hand-calculated and range invariants. | **Keep.** Random-output dimensions alone are weak; bounds and numerical behavior are the useful assertions. |

## Changes completed in the first cleanup slice

- Deep-copy environment/runner configs in the four task liveness/contact files
  before changing episode lengths, initial state, pushes, frequencies, or seeds.
  Close their CPU/Warp backends and VSim fixtures in `finally` blocks.
- Strengthen the shared pendulum gravity/torque test with opposite signed
  initial poses and inputs, asserting the expected response in each world.
- Remove `test_protocol_fixture_has_deliberate_training_geometry` from the full
  campaign suite: it called only its own `_protocol()` fixture and asserted its
  constants. Production matrix, checkpoint, and scheduling tests remain.
- Remove `TestConstruction.test_device_stored` from the task skeleton suite: it
  checked only that construction copied the string `"cpu"`. Valid tensor
  operations, state scaling, reset behavior, and actual backend device/schema
  tests remain. No runtime device behavior was retired.

No other review candidate in this document has been removed. Existing runtime
shape/type guards, algorithm behavior, and physics settings are outside this
test-only slice.

## Measured baseline: 2026-09-07

The portable gate passed with 307 passes, one strict expected failure, and
94 optional cases deselected in 15.64 seconds. Separate colocated runs passed
32 `gym` tests in 1.71 seconds and six `learning` tests in 1.43 seconds. These
are pytest's reported elapsed times; the generated reports also retain reporter
wall times and each case's setup, call, and teardown durations. Separate
optional gates passed all 31 Warp cases in 13.20 seconds and 47 VSim cases in
6.30 seconds, including the signed gravity test and repaired liveness/contact
fixtures. The 16 Unitree cases
were collected but not executed during this cleanup slice.

| Portable file | Sum of setup, call, and teardown | Why this cost remains justified |
|---|---:|---|
| `test_regression_pendulum_training.py` | 4.72 s | Real optimization/checkpoint subprocesses and explicit rejection of insufficient learning evidence. |
| `test_mujoco_cpu_physics.py` | 3.94 s | Analytic energy dissipation and convergence through actual physics. |
| `test_deploy_optional_sdk.py` | 2.68 s | Fresh-process imports with optional dependencies deliberately unavailable. |
| `test_domain_randomization.py` | 2.13 s | Native physical application and the demonstrated multiccd crash pose. |
| All other portable files | 1.74 s | Remaining contract, task, analysis, and control cases combined. |

All portable test phases sum to 15.22 seconds. Colocated test phases total only
0.017 seconds for `gym` and 0.003 seconds for `learning`; their elapsed times
mostly reflect imports and collection. Timing alone does not justify removing
these small local examples.

Reports are `logs/streamlining/portable_test_durations.json`,
`gym_test_durations.json`, `learning_test_durations.json`, and
`benchmark_test_durations.json` in the same directory, with optional execution
in `warp_test_durations.json` and `vsim_test_durations.json`. The final inventory
joins the latest relevant measurement to each collected case and preserves
source hashes. The final portable and focused benchmark runs recorded no
source changes during execution; the earlier report remains available as
`portable_test_durations_initial.json`. Separate GPU correctness tests ran on
the same machine, so these measurements are diagnostic baseline costs, not
CI timing limits or simulation-speed evidence.

## Gates, debt, and next reviews

Portable collection/execution is defined by `pyproject.toml`: the default gate
targets `tests/unit_tests` and excludes `warp`, `vsim`, and `unitree`. Colocated
tests need explicit `pytest gym` and `pytest learning` calls. Requested Warp and
VSim fixtures fail on missing prerequisites; a few mandatory MuJoCo imports
still use `importorskip` and should be made explicit. Unitree tests are opt-in
and do not connect to hardware.

GitHub's unit workflow runs default tests, both colocated roots, and Ruff on
Ubuntu/Python 3.11 for `main`/`dev`/`bigred`/`cdx` pushes and pull requests,
with manual dispatch available. It does not build distribution packages.
It does not establish Warp/VSim correctness, hardware timing, or policy
learning. Its separate file-size workflow checks the event's change range
on pushes and pull requests, with manual dispatch available.

Keep the deployment observation xfail visible in baseline evidence. The
previous stale-configuration registry failure is closed by the prerequisite
config cleanup and its passing test result, not by deleting the check. Never
reinterpret unexecuted optional tests as passes.

Before broader removals, address fixture/global-state isolation, then review
duplicate cases against a concrete counterexample. The new pendulum regression
provides real PPO update/checkpoint execution; independent storage/GAE,
timeout/termination, and normalizer update/freeze/save/load cases remain review
work. A smoke update is not evidence of pendulum learning; the separately
calibrated 100 Hz training protocol owns that claim.

The separate seed-7 GPU calibration failed the proposed learning threshold:
both Warp and VSim caught only 38 of 256 initial states (14.84%) after
400 updates, despite finite training/evaluation and successful checkpoint
restoration and continuation. The policies predominantly damped motion toward
the downward equilibrium. Keep that failed quality evidence alongside the
passing execution smoke; do not lower the threshold to turn it into a pass.
The recorded analysis in `logs/streamlining/pendulum/diagnosis.md` separates
reward/horizon and exploration hypotheses from demonstrated backend errors.
