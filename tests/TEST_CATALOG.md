# Regression test catalog

This catalog records the purpose and review disposition of the portable unit
tests and the colocated `gym` and `learning` tests. It is a behavior review, not
a test-count target. Historical case inventories and timing reports were
generated, ignored artifacts and are not included in this checkout. Current
limitations and evidence boundaries are recorded in
[limitations and validation scope](../genAI_skills/README_MUJOCO.md#limitations-and-validation-scope).

## Reading the review

`test_fixed_state_variance.py` uses analytic shared-noise returns to verify
state-major candidate indexing, covariance-preserving paired contrasts and
repeated-batch variance, observed terminal rewards plus absorbing tails, and a
fake environment for paired commands/pushes. Native Warp snapshot fidelity is
separately gated by `scripts/evaluate_height_q_variance.py`; portable tests alone
do not establish restore fidelity or critic quality.

The extension also checks saved-world selection across flattened state, model and
history fields; lane/candidate axes; reproducible Gaussian scaling with no clipping;
and distinct lane/continuation draws with candidate sharing. The native gates in
`scripts/extend_height_q_variance.py` additionally check all 82 saved fields,
source-to-lane next-step fidelity and actual command/push pairing.

`test_persistent_intervention.py` covers the sample-eight controller contract with
an analytic policy/critic: candidate0 has exactly the nominal stochastic marginal,
lanes share one sample stream but produce distinct draws, batched Q selection equals
per-candidate prediction, and a fake environment verifies per-decision selection,
preterminal denominators, absorbing tails and masks. It also covers the
threshold-rejection rule: first feasible stochastic draw in fixed order (not
highest-Q or closest-to-mean), max-Q fallback, deterministic ties, attempt
accounting, and short exact negative controls (permissive rejection equals plain,
restrictive rejection equals the eight-way best, permissive distance selection
equals the mean). The simulator experiments in
`scripts/evaluate_height_q_persistent.py` and
`scripts/evaluate_height_q_rejection.py` are separately gated (native restore,
CPU/GPU Q agreement, short identical-controller controls, and an actual
short-horizon controller smoke); portable tests establish neither restore
fidelity nor policy quality.

`test_reward_tradeoff.py` protects the exact post-step training-reward dispatch
used by the original-reward tradeoff experiment: nonzero-weight filtering,
ordinary-versus-termination masks, dt scaling, terminal-only termination term,
zero reward after terminal, per-term discounted sums equal to the reported
original return, timeout failure, and bitwise parity with the real
`OnPolicyRunner.update_rewards_dict` on a synthetic environment. It also covers
the dual-objective rollout on a fake environment (original return has no
absorbing failure tail while the height return keeps it) and the portable
Student-t/state-bootstrap helpers. `scripts/evaluate_height_q_reward_tradeoff.py`
is separately gated on native Warp restore fidelity, CPU/GPU Q agreement, a real
environment-step reward-dispatch parity check, and short H5 height/selection
parity against the existing persistent rollout. Portable tests do not establish
restore fidelity, Q quality, or a state population.

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
| [test_domain_randomization.py](unit_tests/test_domain_randomization.py) | CPU/Warp/VSim applied friction/mass/gains, startup ownership, selected task/gait reset buffers, core DR bundle overrides, and predicted friction/acceleration effects. Native readbacks plus real physical response. | **Keep; consolidate shared scenario setup later.** Preserve both parameter readback and dynamics checks. The multiccd crash-pose regression is separate evidence, not a cosmetic config assertion. |
| [test_vsim_domain_randomization_regression.py](unit_tests/test_vsim_domain_randomization_regression.py) | VSim physical-axis set topology, nominal-range equivalence, sparse/empty reset isolation, and per-world weight normalization at 100 Hz. Native topology and controlled comparisons. | **Keep.** Pure topology proxy tests cannot replace execution of the actual production randomization path. |
| [go2/test_go2trot.py](../gym/envs/go2/test_go2trot.py) | Contact strength normalizes using each world's applied total mass. Two hand-calculated worlds with different mass. | **Keep.** Small local example complements the VSim end-to-end weight test. |

## Task registry, scaling, rewards, and actions

`test_vsim_contact_frames.py` checks a standing Go2 at four yaw angles: mean
world contact forces must balance gravity within 3% of body weight. This catches
rotating environment-frame forces as if they were link-local; it requires the
licensed CUDA setup and explicit `Q2_VSIM_TESTS=1` opt-in.

`test_go2trot_performance.py` validates the fixed-command performance gate with
analytic balanced trots and counterexamples: standing on every foot, inverted
gait timing, single-leg loading, alternating target jitter, missed commands,
early failures, and load that swaps between strides. These are metric tests;
policy quality requires running `scripts/evaluate_go2trot.py --check` on an
explicit checkpoint and backend.

The colocated `gym/envs/go2/test_go2trot.py` also checks consecutive-action
smoothness and symmetric yaw-error rewards against hand-calculated values.

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_task_registry.py](unit_tests/test_task_registry.py) | Exact manifest registration, explicit learning-class resolution, selected reset binding, absence of unsupported legacy config, and an actual CPU action step for every declared task. Manifest/static and integration oracles. | **Keep.** The stale `collapse_fixed_joints` declaration was removed and this gate passes in the recorded baseline. One step does not establish runner construction or learning. |
| [test_task_skeleton.py](unit_tests/test_task_skeleton.py) | Named state concatenation/distribution, scaling round trips, and reset flags with a minimal real task subclass. Hand-calculated values and round trips. | **Keep; one assignment-only test removed.** Reassess `test_set_states_wrong_total_dim_raises` with any removal of its custom runtime assertion; preserve correct slicing/scaling behavior without requiring that assertion type. |
| [test_go2trot_command_bounds.py](unit_tests/test_go2trot_command_bounds.py) | Full desired positions include gait/default offsets, legal requests are unchanged, extreme raw samples cannot poison applied observations/history/rewards, and limits follow permuted actuators with passive DOFs. Real task plus adversarial numeric/permutation cases. | **Keep.** Protects the demonstrated training feedback instability; preserving raw policy samples and finite applied history are separate obligations. |
| [test_mini_cheetah_ref_vsim_config.py](unit_tests/test_mini_cheetah_ref_vsim_config.py) | Position-limit penalty normalization, squared rather than fourth-power yaw error, and axis-aligned command support. Hand-calculated rewards and sampled distribution properties. | **Keep.** Revisit the probabilistic sample threshold only with the intended command distribution; the class name does not imply a licensed engine dependency. |
| [test_coupling.py](../gym/envs/mit_humanoid/test_coupling.py) | Humanoid coupling matches an external recorded numeric example; sampled history advances selected worlds without touching others. External reference and isolation sentinels. | **Keep.** Preserve reference provenance. The history expectation uses `torch.roll`, so selection sentinels and explicit newest-slot checks carry the independent value. |

## Saved configuration

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_original_cfg.py](unit_tests/test_original_cfg.py) | Saved config inheritance loads without contaminating live modules, selected run stays pinned, and carried snapshots take precedence. Real temporary Python modules/files. | **Keep.** Protects reproducibility; source-config loading is not model/optimizer checkpoint-resume coverage. |

## Learning, inference, math, and logging

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_runner_inference.py](unit_tests/test_runner_inference.py) | Supported runner inference paths use clean observations and runner reset reuses the task-owned mask. Recording fake actors and mask sentinels. | **Consolidate later** equivalent runner cases while retaining off-policy action transformation. These tests do not prove optimizer, storage, or save/load behavior. |
| [test_normalize.py](../learning/modules/utils/test_normalize.py) | Running mean/variance combination and normalization on known input. Hand-calculated scalar moments. | **Keep; coverage gap remains** for update-versus-freeze timing and serialization/resume, especially repeated optimizer passes over the same rollout. |
| [test_usecase.py](../learning/utils/PBRS/test_usecase.py) | Potential-based shaping produces known rewards across pre/post-step potentials. Small real interface example with numeric expectations. | **Keep.** It is not a substitute for PPO/GAE return tests. |
| [test_logger.py](../learning/utils/logger/test_logger.py) | Finished-episode and iteration values aggregate correctly and counters clear. Controlled logger inputs. | **Keep.** Check global logger isolation when introducing real runner tests. |
| [test_utils.py](../learning/utils/tests/test_utils.py) | Horizon conversion and zero-weight reward removal. Small hand-calculated utility examples. | **Keep; combine adjacent tiny cases if useful.** Preserve physical-time discount semantics under frequency changes. |
| [test_offline_replay.py](unit_tests/test_offline_replay.py) | Bounded replay ring ordering/overwrite, pre-reset successor and independent flags, recording-RNG isolation, immutable snapshot publication, schema validation, height relabeling, nested transition/episode subsets, and mixture deduplication. Hand-built transition batches and deterministic fixtures. | **Keep.** It protects the replay contract, not simulator rollout or Q accuracy. Add a real-run snapshot smoke outside the portable gate. |
| [test_height_fqe.py](unit_tests/test_height_fqe.py) | Fixed-policy FQE targets for continuing reward, timeout bootstrapping, terminal absorbing failure and flag precedence; Q artifact round-trip; frozen actor loading and non-training; finite-rollout tail bounds; batched-evaluator CLI defaults that pin `insert_step = 0`. Analytic zero-Q and a fake height/termination environment. | **Keep.** It does not establish learned-Q accuracy or robot behavior; evaluate candidate actions and thresholds on fresh simulator continuations separately. |
| [test_reward_tradeoff.py](unit_tests/test_reward_tradeoff.py) | Exact training-reward dispatch (nonzero filtering, ordinary/termination masks, dt scale, terminal-only termination term, zero after terminal, per-term sum), bitwise parity with `OnPolicyRunner.update_rewards_dict`, dual-objective rollout without an original absorbing tail, timeout failure, and portable paired-t/state-bootstrap helpers. | **Keep.** The simulator reward-parity and restore gates live in `scripts/evaluate_height_q_reward_tradeoff.py`; this test does not establish native fidelity or a state population. |
| [test_height_q_playback.py](unit_tests/test_height_q_playback.py) | Single-environment playback controller: eight stochastic max-Q draws, rejection first-feasible/max-Q fallback, mean excluded, shared sample order, finite threshold and mode validation; pre-simulator checkpoint hash/config-fingerprint/actor/noise/scale/normalization/privileged-input gates; instance-local command-resampling hold; auto-reset union masks and zeroing. Analytic policy/critic and fake task tensors. | **Keep.** Protects wiring and identity, not Q accuracy or policy quality. Manual CPU viewer/keyboard/thread behavior and the real artifact smoke remain external evidence. |

## Deployment, bootstrap, and threading

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_deploy_optional_sdk.py](unit_tests/test_deploy_optional_sdk.py) | Simulation/offline imports and help work with SDK/DDS deliberately unavailable; hardware invocation reports the optional extra. Isolated subprocess import blockers. | **Keep.** Optional-dependency boundaries must work even on a machine where the SDK happens to be installed. |
| [test_go2_deploy_obs.py](unit_tests/test_go2_deploy_obs.py) | Canonical joint/gyro/gravity ordering and scaling, static-posture offsets, active gait frequency, and unrestricted target assembly using hand-built messages. | **Keep.** Covers current reward-only gait semantics. The CPU-task observation-parity test and its expected-failure variant were removed; these utility tests do not establish hardware behavior or checkpoint compatibility. |
| [test_unitree_sdk.py](unit_tests/test_unitree_sdk.py) | Real optional SDK/DDS/CRC imports, default command contents, damping-only emergency command, and CRC response to a changed motor field. Actual native dependency and message mutation. | **Keep; Linux opt-in, no robot connection.** Recomputing CRC with the same SDK tests integration, not independent checksum algorithm correctness. Mutation coverage prevents a constant checksum from passing. |
| [test_recurrent_thread.py](unit_tests/test_recurrent_thread.py) | Real Linux timer/thread lifecycle, periodic callbacks, timeout behavior, failure results, descriptor close, and replacement-instance cleanup. Observed OS resources plus narrowly injected errors. | **Keep.** Broadly mocking timers would lose the resource/lifecycle oracle; elapsed-time bounds are scheduler-sensitive and should be reviewed with measured cost. |
| [test_fetch_unitree_sdk.py](unit_tests/test_fetch_unitree_sdk.py) | Official-pin bootstrap mechanics preserve existing revisions and staged/unstaged/untracked work, tolerate ignored caches, and report failures. Temporary local Git repositories, offline. | **Keep.** Exercises actual Git behavior rather than comparing generated command strings. |

## UI, geometry, and common math

| File | Protected behavior and oracle | Disposition and overlap |
|---|---|---|
| [test_teleop_keyboard.py](unit_tests/test_teleop_keyboard.py) | Key bindings, held-key edges, reset/exit behavior, command axes/limits, viewer callback/UI wiring, default forward seeding, optional zero-initial/zero-after-reset/preserved-resampling options, and deferred UI-thread event draining. Event fakes and explicit command changes. | **Consolidate later** repeated declaration/default cases. Keep key-edge, axis, and default/manual-mode behavior. Replace mandatory-MuJoCo import skips separately. |
| [test_vsim_command_visualizer.py](unit_tests/test_vsim_command_visualizer.py) | Yaw/arrow/arc geometry, robot-heading transforms, visibility, and drawing-hook wiring. Analytic geometry and lightweight fake viewer. | **Consolidate later** equivalent geometric examples, not all viewer tests. Names are VSim-specific but pure geometry requires no license. |
| [test_math.py](../gym/utils/math/test_math.py) | Angle wrapping, bounded random sampling, and exponential-average update. Hand-calculated and range invariants. | **Keep.** Random-output dimensions alone are weak; bounds and numerical behavior are the useful assertions. |

## Tooling cleanup

Campaign, trajectory, specialized parity, calibration, policy-evaluation,
benchmark/profile, and analysis programs and their exclusive tests are retired.
The extracted evaluator, trainer-CLI, PPO checkpoint, and pendulum-control tests
are also absent; no replacement coverage is claimed. Core DR override checks
remain in `test_domain_randomization.py`, and saved-configuration loading remains
covered by `test_original_cfg.py`.

Native physics, state/reset contracts, retained task/config tests, focused
learning-code tests, assets, and deployment coverage remain. Earlier
fixture-isolation fixes and independent signed-gravity checks remain. VSim
backend code, its optional extra, and marked tests are retained, while the local
test launcher and support workflow are deferred.

## Historical baseline: 2026-09-07

Before tooling retirement, the portable gate recorded 307 passes, one strict
expected failure, and 94 optional cases deselected in 15.64 seconds. Separate
runs recorded 32 colocated `gym` tests, six `learning` tests, 31 Warp cases,
and 47 VSim cases passing. The 16 Unitree cases were collected but unexecuted.
These are historical source/dependency measurements, not current pass counts
or timing limits; retired cases are included in those totals.

Those measurements were recorded in generated reports under
`logs/streamlining/`. The reports were ignored artifacts and are not included in
this checkout. Measure the current suite afresh before making timing or coverage
claims.

## Gates, debt, and next reviews

Portable collection/execution is defined by `pyproject.toml`: the default gate
targets `tests/unit_tests` and excludes `warp`, `vsim`, and `unitree`. Colocated
tests need explicit `pytest gym` and `pytest learning` calls. Requested Warp and
VSim fixtures fail on missing prerequisites; a few mandatory MuJoCo imports
still use `importorskip` and should be made explicit. Unitree tests are opt-in
and do not connect to hardware.

GitHub's unit workflow runs default tests, both colocated roots, Ruff, and a
package build on Ubuntu/Python 3.11 for `main`/`dev` pushes and pull requests,
with manual dispatch available.
It does not establish Warp/VSim correctness, hardware timing, or policy
learning. Its separate file-size workflow checks the event's change range on
pushes and pull requests, with manual dispatch available. `uv.lock` is exempt
from the 100 KiB limit as dependency metadata.

The removed deployment observation-parity test supplies no passing parity
evidence. The previous stale-configuration registry failure is closed by the prerequisite
config cleanup and its passing test result, not by deleting the check. Never
reinterpret unexecuted optional tests as passes.

Before broader removals, address fixture/global-state isolation, then review
duplicate cases against a concrete counterexample. There is no PPO
update/checkpoint or pendulum-control smoke test in CI. Independent storage/GAE,
timeout/termination, and normalizer update/freeze/save/load cases remain review
work. Retained native tests do not establish learning quality or model/optimizer
checkpoint restoration.

The historical seed-7 GPU calibration failed its proposed learning threshold:
both Warp and VSim caught only 38 of 256 initial states (14.84%) after
400 updates, despite finite training/evaluation and successful checkpoint
restoration and continuation in that historical run. The policies predominantly
damped motion toward the downward equilibrium. This remains failed quality
evidence; it does not describe an active execution or learning-quality gate.
