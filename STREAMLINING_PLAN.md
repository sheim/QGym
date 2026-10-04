# Simulation, training, and test streamlining

Establish useful regression gates before changing implementation for speed or
readability. The order is: repair prerequisites, protect simulation speed,
protect pendulum learning, audit and simplify tests, then profile and refactor
production code. This document records the plan and execution status. Generated
evidence is under `logs/streamlining/`; the current findings are recorded below.

All new simulation and training experiments use 100 Hz. Training profiles
set policy, control, and physics frequencies explicitly. Existing research
artifacts remain historical evidence, not automatically valid baselines.

## 0. Establish a usable starting point

- Record the current source and asset hashes, lockfile, resolved configuration,
  hardware, test collection, failures, expected failures, and test durations.
  Include relevant untracked source files: a commit hash alone does not describe
  this research worktree. Put generated evidence under `logs/streamlining/`.
- Repair the pendulum runner/config mismatch before hiding it behind a test
  override. `OnPolicyRunner.__init__` reads `algorithm.rollout_size`, but
  `PendulumRunnerCfg` and `FixedRobotCfgPPO` only declare `storage_size`.
  Use one explicit rollout field consistently across supported PPO configs and
  documentation; the current consumer is `rollout_size`. Keep optimizer
  `batch_size` separate. Do not add aliases, fallback reads, or implicit rollout
  truncation. Exercise actual runner construction and updates in the regression.
- Resolve the existing registry failure caused by the unused
  `collapse_fixed_joints` field. Check its consumer and intended support, then
  remove the stale configuration or implement the intended behavior in a
  separate change. Keep the registry test.
- Keep the deployment observation parity xfail visible: deployment clips
  residual actions against absolute position limits. Its resolution is a
  separate behavior change, not a prerequisite for simulator timing.
- Use fresh config instances and guaranteed backend cleanup in fixtures and
  benchmark workers. Several existing tests mutate shared registry instances.

Exit: the relevant correctness baseline is understood, pendulum can use the
real PPO runner, and each remaining failure has an explicit disposition.

## 1. Regression tests for simulation speed

### Workloads

Start from `scripts/benchmark_domain_randomization.py` and
`scripts/benchmark_vsim_environment_sets.py`. Reuse their timing and reset
scheduling code through one small regression entry point and comparator. Avoid
building a general benchmark framework or another campaign orchestration stack.

| Workload | Backend and size | Purpose |
|---|---|---|
| Pendulum, nominal physics | CPU 256; Warp and VSim 4,096 | Fixed-base stepping and task overhead |
| Go2Trot, nominal physics | CPU 256; Warp and VSim 4,096 | Contacts, floating-base state, and task lifecycle |
| Go2Trot, identical nominal parameters | VSim 4,096 environments in one set versus 4,096 sets | Preserve the environment-set slowdown discriminator with DR disabled |

Each cell compares against its own reference on the same machine. Different
CPU/GPU batch sizes serve different regression workloads; this matrix is not
a cross-engine speed ranking. Add single-environment latency and size sweeps
only when investigating scaling. CPU full-DR training is outside this push.

Measure these separately:

1. Native backend step, including the public state refresh required by its
   contract, with precomputed nontrivial torques.
2. Full task step followed by an empty reset: the runner's usual no-reset path.
3. Full task step with deterministic, staggered timeout-rate resets.
4. Empty, one-environment, sparse, and full backend/task resets as diagnostic
   component measurements. Resetting every environment every step is a stress
   case, not representative training throughput.
5. Setup, compilation/graph capture, and memory consumption outside steady-state
   timing. Isolated component measurements are not additive patch-cost estimates.

Use a repeatable contact workload for Go2Trot. Preserve the existing settled,
zero-torque topology probe as a separately labelled discriminator. When DR code
changes, also exercise its actual production startup path with fixed friction
and nominal mass scales; the topology proxy alone cannot cover that path.

### Measurement and acceptance

- Explicitly set 100 Hz control and physics, decimation 1, fixed solver/graph
  settings, and disabled pushes. Record actual native set counts.
- Run one cell per process and one GPU job at a time. Warm compilation, capture,
  stepping, and each reset path before measuring. Start calibration with 100
  warmup calls; verify that timing has stabilized rather than assuming it has.
- Calibrate batch length to about 0.5–2 seconds, then freeze that step count for
  reference/candidate comparisons. Collect five timed batches per process.
  Synchronize simulator/Torch work before and after whole batches; exclude
  per-step synchronization, logging, profiling, and scalar readbacks.
- Restore equivalent physical and task state before trials: commands, phase,
  history, episode counters, reset inputs, and RNG state as well as robot state.
  An all-environment reset alone does not do this. Generate torque inputs and
  reset masks outside timing. Check state evolution, applied inputs, and public
  observations outside timing so skipped work cannot count as a speedup.
- Compare alternating reference/candidate fresh processes. Calibrate noise with
  at least five same-revision process pairs first. Processes, not batches within
  one process, are independent timing samples.
- Report raw times, median milliseconds per step, environment steps/second,
  reset counts, spread, and memory. Include CPU/thread settings, GPU/driver,
  native engine versions, Torch/CUDA, resolved config, and source/asset hashes.
- Begin with a **provisional 5% end-to-end slowdown threshold**. Calibrate and
  freeze a per-workload tolerance against same-revision variability before
  making it blocking. Use absolute time as well as percentage for tiny reset
  operations. If noise obscures a useful threshold, improve measurement before
  promoting the gate. Do not automatically widen tolerances or replace baselines
  after failures.
- A failing comparison keeps a failing exit status. A confirmation run is
  diagnostic evidence, not retry-until-green behavior. Changes to workload,
  hardware, physics, or dependencies require a deliberately established new
  reference; do not silently compare incompatible results.

Fix known worker limitations before adopting their numbers: the DR worker
inherits task frequencies, its `none` profile still calls an empty reset, its
default GPU trial is short, and its descriptive `friction-fixed` topology label
disagrees with native readback. The set worker lacks full task-step and nonempty
reset profiles. Keep the original vendor Ant benchmark vendor-facing.

Exit: a reproducible reference, a comparator that detects a deliberate slowdown,
and documented runtime/noise on the designated machine. Portable CI tests the
protocol/comparator; hardware timing runs explicitly on the reference host.

## 2. Pendulum training regression on Warp and VSim

Train through `scripts.train.setup()` → `OnPolicyRunner` → `PPO2` → `DictStorage`.
Use the real actor, critic, optimizer, resets, and checkpoint path. A process
that merely finishes two updates is a smoke test, not evidence of learning.

Start calibration with this explicit shared profile:

| Setting | Initial calibration value |
|---|---|
| Backends | MuJoCo Warp and VSim, `cuda:0`, separate processes |
| Policy / control / physics | 100 / 100 / 100 Hz |
| Training environments | 512 |
| `rollout_size` | 65,536: 128 consecutive steps/environment, 1.28 seconds |
| Optimizer `batch_size` | 16,384 |
| `max_gradient_steps` | 24 for each actor and critic update |
| Actor/critic networks | Existing `[128, 64, 32]`, tanh |
| Learning rate / entropy coefficient | Existing `1e-4` / `0.01` |
| Observation normalization/noise | Disabled, as in the current pendulum profile |
| Discount / GAE physical horizons | 0.8 / 2.0 seconds |
| Training reset distribution | Angle `[-pi, pi]`, angular velocity `[-5, 5]` |
| Calibration checkpoints | 0, 100, 200, 400 |

The current task uses 25 Hz policy/control and 200 Hz physics. Preserve its
nominal physical discount horizons through the existing frequency converter:
set `algorithm.discount_horizon=0.8` and
`algorithm.GAE_bootstrap_horizon=2.0`. At 100 Hz these give `gamma=0.9875`,
`lam=0.995`. Keeping the old `0.95/0.98` constants would shorten their horizons
fourfold. Persist the
resolved profile and apply it to evaluation as well; saved source files or
`--original_cfg` alone do not preserve arbitrary runtime overrides.

Generate actor and critic initial state once per seed, load the same states
into both backends' fresh runners, and record their hashes. Seed the environment,
policy sampling, episode offsets, and minibatch sampling explicitly after
construction, then redraw the initial environment state and episode counters.
Reseeding alone does not replace state already generated during setup. Equal
seed integers alone do not establish equal initial weights.
Process isolation also avoids the runner's module-global logger/storage state.

### Learning criterion

Use deterministic evaluation on a fixed 16×16 angle/velocity grid, 10-second
episodes, and no resets during evaluation. Evaluate the untrained policy and
every declared trained checkpoint with the same profile. Reuse the pendulum
recording path in `scripts/eval_policy.py` and the physical catch definition in
`scripts/pendulum_fidelity.py`: wrapped `|theta| < 0.14 rad` and `|omega| < 0.5
rad/s` throughout the final second. The analytic controller is a physics
reference, not the policy under test.

- Proposed calibration target: at least **80% caught and held**, and at least
  **50 percentage points improvement over initialization**, on each backend.
  These are proposed targets, not claims about current achievable performance.
- First calibrate seed 7 at the declared checkpoints. Select the smallest
  common fixed training budget that achieves useful performance with margin;
  validate it with seeds 17 and 27 before freezing the gate. Do not choose a
  different winning checkpoint for each backend or seed.
- After calibration, require both absolute physical competence and retention of
  reference quality within a tolerance established from repeated runs. A model
  can still regress materially while remaining above a coarse 80% floor.
- Record per-term rewards, final angle/velocity errors, time to sustained catch,
  raw policy actions, applied torque/saturation, losses, finite model/optimizer
  state, collected samples, and collection/optimization wall time. Record
  `env.torques` after stepping: the evaluator's existing pendulum
  `applied_actions` contains requested `tau_ff`, before physical torque clipping.
  Inspect finiteness in the test harness, not through new production tensor guards.
- Verify deterministic actions survive save/load into a fresh runner, both
  optimizer states restore for resume, and one additional update succeeds. The
  runner interprets the iteration budget as additional iterations on resume.
  Current checkpoints omit simulator/RNG state: do not promise bitwise equivalence to
  uninterrupted training.

If learning misses the physical target, diagnose it before weakening the test.
Current reward ambiguities deserve attention: equilibrium uses unwrapped angle
although observations are periodic, and the energy formula omits the pole's COM
inertia. Any necessary reward correction must be separately reviewed and
measured before freezing a reference, not bundled into a readability refactor.

Execution tiers:

- Portable: actual config/runner construction, two updates, and checkpoint
  smoke, plus focused learning math tests. No claim of swing-up learning.
- Regular GPU regression: the frozen seed/budget on each backend. Target at
  most ten minutes per backend; measure this during calibration rather than
  promising it. Record compilation separately.
- Learning/refactor acceptance: three seeds on both backends, native evaluation,
  and final-policy transfer in both directions. Each backend passes against its
  own baseline; exact equality of cross-engine trajectories is not required.

Exit: both engines demonstrably learn under a fixed protocol, and the tests
detect disabled optimization or broken action/observation wiring. This profile
does not cover normalization; keep dedicated tests for that behavior.

## 3. Catalog and evaluate the unit tests

The initial static inventory found **55 files / 333 test functions before
parametrization**: 45/298 under `tests/unit_tests`, 6/29 under `gym`, and 4/6 under
`learning`. These are declarations, not passing or collected-case counts.

Create a complete collected-case inventory and duration report under
`logs/streamlining/`, plus a reviewable `tests/TEST_CATALOG.md` organized by
behavior. For each case record its production entry point, invariant,
counterexample, oracle, backend/task axes, dependencies/marker, fixture
isolation, cost, failures/xfails, duplicate relationship, disposition, and
replacement coverage. Record setup/teardown time too. Review all three source
roots and explicitly selected optional groups; bare pytest omits colocated tests.

Initial family-level assessment:

| Family | Initial direction |
|---|---|
| Backend state, reset, frames | Keep native/independent oracles and cached-state invariants; consolidate schemas |
| Physics, contacts, assets, routing | Keep predicted physical responses and canonical permutation sentinels |
| DR lifecycle and physical effects | Keep sampling/application separation and physics checks; share equivalent backend scenarios |
| Registry, scaling, rewards, actions | Separate behavioral contracts from assertions about incidental defaults |
| Evaluation, metrics, saved configuration | Preserve numerical oracles, applied actions, config isolation, and checkpoint selection |
| Benchmark protocols and campaign evidence | Preserve comparable workloads, provenance, scheduling, pairing, and failed-cell handling |
| Learning, inference, math, logging | Address PPO/GAE/storage/checkpoint gaps; simplify shallow mocked inference cases |
| Deployment, bootstrap, threading | Keep optional dependency boundaries, real CRC smoke, and resource cleanup |
| UI, geometry, common math | Keep meaningful event/numerical behavior; combine small equivalent cases |

Concrete review candidates, not a deletion list:

- **Keep:** rotated-root tests in `test_root_velocity_frames.py`, the real
  applied-command regression in `test_go2trot_command_bounds.py`, sparse-reset
  liveness, friction/mass physical effects in `test_domain_randomization.py`,
  and paired campaign aggregation. These catch demonstrated or plausible
  behavioral failures using independent evidence.
- **Consolidate:** repeated shape/identity cases in
  `test_legged_backend_contract.py` and `test_vsim_legged_contract.py`, using a
  shared backend-parametrized contract. Preserve separate contact, floating-base,
  reset, and canonical-routing obligations. Backend-specific tests remain where
  their native semantics or failure mechanisms differ.
- **Strengthen:** `test_gravity_and_applied_torque_have_expected_signs` currently
  checks nonzero absolute velocity for the gravity response. Assert the predicted
  sign. Large fake-registry tests should state that they verify wiring/config
  isolation; add a small real integration path when actual execution is the claim.
- **Remove or replace after review:**
  `test_protocol_fixture_has_deliberate_training_geometry` only checks its own
  fixture; `test_device_stored` checks an assignment; benchmark tests that freeze
  arbitrary warmup defaults should assert the relevant experimental invariant.
- **Reassess invalid-input tests:** tests that exist solely to require custom
  shape/type assertions should not preserve those production guards. Keep shape,
  type, and numerical assertions in tests of valid behavior. Preserve errors
  that express an actual supported API or algorithm contract.

Sequence: fix fixture config isolation and backend cleanup; strengthen weak
oracles; consolidate equivalent cases; remove demonstrated redundancy; fill the
important gaps. The gaps include hand-calculated GAE/timeout/termination cases,
rollout storage/temporal ordering, actual PPO updates, and normalization
update/freeze/save/load behavior.

For proposed removals, name the behavior being retired or the replacement that
still detects the counterexample. Use selected historical bug reintroductions
or small deliberate faults in an isolated checkout to verify valuable tests.
No test-count reduction quota, wholesale mutation framework, or blanket removal
of mocks. Measure protected behavior, failure clarity, isolation, and feedback
time instead of counting assertions.

Exit: every case is classified, retained cases have a clear purpose, removed
cases have a rationale, and portable/colocated/applicable optional suites run
without accidental dependency skips or new xfails. Missing mandatory MuJoCo
must fail, not disappear behind `importorskip`. Keep the existing deployment
xfail as explicit debt until its behavior is corrected.

## 4. Profile Warp and VSim, inspect flame graphs, then refactor

Run the same warmed 100 Hz Go2Trot workload on Warp and VSim, one GPU process at
a time. Capture the full task/reset path and, after the training gate is ready,
a short pendulum rollout/update window on both backends. Keep setup and JIT out
of the steady-state capture. Record workload, source, and tool versions beside
each artifact so the flame graphs can be compared meaningfully.

Produce an inspectable Python/host flame graph for each backend, plus a CUDA
timeline from Nsight Systems covering native simulator kernels, copies,
synchronization, and CPU launch gaps. Use NVTX labels for stepping, resets,
observation assembly, rollout collection, and optimization. A Torch-only trace
does not establish coverage of VSim's native kernels: state capture limitations
explicitly rather than presenting incomplete coverage as a complete GPU profile.

Save the raw traces, portable flame-graph/Speedscope artifacts, and a concise
assessment of the dominant stacks, kernels, transfers, and idle gaps under
`logs/streamlining/profiles/`. Distinguish self time, inclusive time, CPU waiting,
and GPU execution. Rank concrete refactoring candidates by measured cost and
the regression that protects each change. Profiled times are diagnostic and
must not replace unprofiled speed measurements.

Use the new workloads to inspect CPU native loops and contact refresh,
Warp/Torch boundaries, VSim command/state copies, empty and sparse reset work,
task tensor allocation/concatenation, and PPO collection/optimization. Inspect
repeated critic evaluation and logger/scalar synchronization as measured
candidates. Profilers are diagnostic runs, separate from acceptance timings.

Apply the requested style throughout:

- Required values are explicit config fields and direct accesses. Missing
  attributes/keys and invalid operations should fail directly.
- Do not introduce production shape/type checks, automatic casts/reshapes to
  rescue inputs, catch-and-default paths, or silent backend substitutions.
  Shape/type assertions belong in tests.
- Resolve configured mappings and dispatch once where possible. Keep functions
  small and concrete; abstract repeated behavior only when it reduces real
  duplication. Do not replace straightforward code with a generic framework.
- Keep resource cleanup in `finally`/context managers. Distinguish cleanup from
  catching errors and continuing with invented defaults.
- Preserve atomic reset ordering, canonical routing, in-place public state, and
  the learning/action/observation contracts. An optimization that skips required
  work is not a valid performance improvement.

Follow the reachability audit in `MIGRATION_PLAN.md` for learning-stack pruning.
No runner becomes removable just because its code looks old; configured SAC/PSD
paths require a support decision and their own evidence. Remove tests of a
retired path with that path, not in advance.

Keep correctness fixes, performance changes, readability changes, and research
tuning separately reviewable. For each slice state the hypothesis, show the
smallest relevant counterexample/test, run the applicable new regression gates,
and report the measured effect. Do not silently update baselines to accept a
slowdown or poorer learning.

## Deliverables and execution

1. Prerequisite fixes and a classified starting test report.
2. Simulation-speed worker/comparator, frozen workload profiles, reference
   artifacts, and demonstrated slowdown detection.
3. Pendulum train/evaluate regression, calibrated seed/budget/threshold profile,
   checkpoint coverage, and Warp/VSim reference results.
4. Complete test catalog and small, justified test-cleanup changes.
5. Warp/VSim flame graphs and native CUDA timelines, their assessment, and
   ranked profiling findings, followed by measured production refactors.

Keep small profiles, comparator logic, tests, and guidance in source control;
keep timing samples, checkpoints, traces, and bulky catalogs under `logs/`.
Run GPU/VSim gates explicitly on available hardware without overlapping jobs.
Current GitHub CI runs portable/colocated tests and Ruff on Ubuntu;
it does not establish GPU performance or learning. Add a hardware runner only
when its availability is established, rather than allowing requested GPU tests
to skip. Document the actual local and CI commands when implementation exists.

## Execution evidence, 2026-09-07

The first implementation slice adds:

- `scripts/benchmark_simulation.py`: controlled task/backend stepping and reset
  profiles, process-level paired comparisons, provenance, initial-state and
  repeated-state evidence. Profiler timings and incompatible protocols cannot
  enter the speed comparison. The comparator detects a deliberate 20% slowdown
  and preserves inconclusive/failing exit statuses.
- `scripts/regression_pendulum_training.py`: actual PPO training, fixed native
  evaluations, common initial weights, physical torque records, and fresh-runner
  optimizer/checkpoint restoration plus one additional update.
- `scripts/profile_simulation.py`: native Nsight CUDA captures and filtered
  host Speedscope/SVG flame graphs. `--rate` controls host sampling independently
  of the fixed 100 Hz simulation. Nsight ranges distinguish task/backend steps,
  resets, and post-physics derived state; this is not every observation operation.
- `tests/TEST_CATALOG.md`: all test files reviewed, collected-case inventory and
  setup/call/teardown durations, four fixture-isolation repairs, a signed gravity
  oracle, and removal of two assignment/self-fixture tests. Learning math and
  normalizer lifecycle gaps remain explicit review items.

The required PPO `rollout_size` is now declared by `FixedRobotCfgPPO`, and unused
PPO `storage_size` declarations are removed. Off-policy storage sizes remain.
The stale `collapse_fixed_joints` registry failure is resolved by removing the
unused field. No reward, physics, or hot-path optimization is bundled into this
slice.

Final validation: 307 portable tests, 32 colocated `gym` tests, six colocated
`learning` tests, 31 explicitly selected Warp tests, and 47 licensed VSim tests
passed. The one known deployment observation xfail remains; the 16 Unitree
cases were not rerun in this slice. Full Ruff and a wheel/sdist build with
`--no-sources` passed. Per-case phase reports and the joined collection inventory are
under `logs/streamlining/`; test durations are diagnostic rather than speed gates.

### Speed evidence

The seven core cells use the full task step and deterministic timeout-rate reset
path. Five batches per process and five alternating same-revision process pairs
are recorded in `logs/streamlining/speed/references/summary.json`, with every raw
measurement beside it. Reference batch lengths are 50–1,000 steps, calibrated
to roughly 0.5–2 seconds; both VSim Go2 topologies use the same 200-step window.
All seven cells passed the provisional 5% check across 70 fresh processes and
350 timed batches. Their paired confidence intervals remain within about 2%
of equal speed. This establishes a same-session reference, not repeated-day
stability. The [speed assessment](logs/streamlining/speed/references/assessment.md)
records all cell medians and intervals. The matched nominal VSim Go2 set-count
contrast is 4.549 versus 5.316 ms/step, a 16.9% increase with singleton sets.
The setup/warmup probes and the earlier CPU pilot remain separate artifacts.
Only this full-task profile has a process-level noise calibration; isolated
backend/reset profiles are available as diagnostic commands, not promoted gates.

Initial-condition identity compares root/DOF positions and velocities. Contact
forces are derived solver results: Warp repeated the requested state exactly
while initial contact outputs varied by at most about 0.00023 N across these
process pairs. They remain finite-state/physics evidence, rather than an exact
initial-condition key. A regression test preserves this distinction. Performance
acceptance still requires the relevant correctness tests; a speed ratio alone
does not certify equivalent physics.

### Learning calibration failed the physical target

| Native backend | Caught at 0 / 100 / 200 / 400 updates | Training wall time | Checkpoint/resume |
|---|---|---:|---|
| Warp | 0 / 30 / 37 / 38 of 256 | 303.09 s | Passed, including update 401 |
| VSim | 0 / 31 / 36 / 38 of 256 | 101.89 s | Passed, including update 401 |

Both final catch rates are **14.84%**. The proposed 80% catch / 50-point
improvement criterion is unchanged and fails. These are calibration results,
not acceptable learning references. Initial checkpoint hashes match, and all
checked model, optimizer, rollout metrics, and evaluation tensors are finite.
No additional seeds or reward changes were run after this shared failure.

About 82% of reward improvement comes from reducing the angular-velocity penalty;
only 2/128 starts at least 90 degrees from upright are caught. The known
unwrapped-angle reward alias is real, but recomputing it on the saved trajectories
does not establish it as the main cause. The next discriminator is the existing
analytic swing-up controller on the same grid, comparing physical success and
discounted returns under the current objective before choosing a training change.
See [the offline diagnosis](logs/streamlining/pendulum/diagnosis.md), manifests,
evaluation arrays, and checkpoint summaries under `logs/streamlining/pendulum/`.

### Profile assessment and refactoring order

Actual captures cover Warp and both nominal VSim set topologies. Open the
[Warp flame graph](logs/streamlining/profiles/warp_host_200hz/host.flamegraph.svg),
[VSim shared-set flame graph](logs/streamlining/profiles/vsim_host_200hz/host.flamegraph.svg),
or [VSim singleton-set flame graph](logs/streamlining/profiles/vsim_sets_host_200hz/host.flamegraph.svg).
The [assessment](logs/streamlining/profiles/assessment.md) links the native CUDA
timelines and records overlap, stack coverage, observer effects, and provenance.
The original 1,000 Hz VSim host captures lagged; 200 Hz recaptures supersede their
percentages. Physics remains at 100 Hz in every capture.

1. Investigate Warp graph capture and solver temporary allocations. The native
   conditional loop synchronizes with the host outside capture; GPU kernel
   intervals occupy only about 12% of the captured wall window. Both stepping
   and resets incur this path. Preserve reset/state/parity contracts.
2. Cache immutable Go2 command choices on the device. Constructing the tensor
   twice per step introduces stream synchronization. The large host-stack share
   includes waiting for physics, so it is not a predicted speedup percentage.
3. Keep VSim topology costs distinct from host preparation. Matched captures
   have the same kernel counts and saved sample states, but singleton sets raise
   GPU kernel interval-union time by about 20%. Use unprofiled timings to assess
   the end-to-end cost; this nominal discriminator does not change DR sampling.
4. Review reset/observation work after the dominant costs. The captured masks
   select four or five environments per step; these traces do not measure an
   empty-reset shortcut or justify skipping required refreshes.

Broad learning refactoring and a rollout/update flame graph remain pending a
useful physical learning gate. The simulation profiles above exclude policy
inference, reward aggregation, and PPO optimization.

## Commands for generating and opening the evidence

Run these from `/home/heim/Repos/pkGym`. These commands document the recorded
run names; use new output paths for a new experiment. Training and profiling
reject existing output directories, while a benchmark `--output` can overwrite
its JSON. Run simulation, training, and profiling jobs sequentially, without
competing workloads during timing. No rerun is needed just to open the results.

### Environment and output directories

```bash
cd /home/heim/Repos/pkGym
uv sync --frozen --extra vsim --group profiling
mkdir -p logs/streamlining/speed logs/streamlining/profiles
```

The recorded runs used the following optional shell settings to use cached
dependencies and keep uv's cache under `/tmp`:

```bash
export UV_OFFLINE=true
export UV_CACHE_DIR=/tmp/q2-uv-cache
```

Set these after dependency installation; offline mode requires a populated
cache. VSim commands also load `.env.vsim`. Nsight Systems is installed
separately; `py-spy` is supplied by the `profiling` dependency group.

### Training, checkpoints, evaluations, and diagnosis

The commands used for the two calibration runs were:

```bash
uv run --frozen -m scripts.regression_pendulum_training \
    --backend mujoco --device cuda:0 \
    --output logs/streamlining/pendulum/warp_seed7 \
    > logs/streamlining/warp_training.log 2>&1

uv run --frozen --env-file .env.vsim -m scripts.regression_pendulum_training \
    --backend vsim --device cuda:0 \
    --output logs/streamlining/pendulum/vsim_seed7 \
    > logs/streamlining/vsim_training.log 2>&1
```

The resolved defaults were seed 7, 400 updates, 512 training environments,
65,536 rollout samples, 16,384-sample minibatches, 24 gradient steps, and
256 evaluation environments for ten seconds. Policy/control/physics are
100/100/100 Hz. Each worker generates `manifest.json`, initial weights,
training logs/checkpoints, `evaluation_{0,100,200,400}.npz`, `evaluations.json`,
resume artifacts, and `summary.json`. It performs evaluation and the one-update
resume itself; there is no separate playback command required to create those
outputs. `--require-learning` makes the physical criterion affect the exit
status; it was not used for these initial calibration runs.

The local offline analysis helper regenerates the numerical diagnosis:

```bash
uv run --frozen python logs/streamlining/pendulum/diagnose_calibration.py \
    > logs/streamlining/pendulum/diagnosis_stdout.txt
```

It writes `diagnosis.json` from existing evaluation artifacts. `diagnosis.md`
is the written interpretation, not an automatically generated report.

### Simulation speed logs and paired comparisons

For example, the Go2 CPU calibration command was:

```bash
uv run --frozen -m scripts.benchmark_simulation run \
    --backend cpu --task go2trot --num-envs 256 \
    --profile task_timeout --steps 50 --warmup 100 --repeats 5 \
    --output logs/streamlining/speed/cpu_go2_calibration.json \
    > logs/streamlining/speed/cpu_go2_calibration.log 2>&1
```

The saved local drivers generated the complete probe matrix and A/A reference
matrix, one child process per measurement:

```bash
uv run --frozen --env-file .env.vsim \
    python logs/streamlining/run_speed_probes.py \
    > logs/streamlining/speed_probes.log 2>&1

uv run --frozen --env-file .env.vsim \
    python -m logs.streamlining.run_speed_aa \
    > logs/streamlining/speed_aa.log 2>&1
```

After the comparator correction, the second invocation was repeated with
stdout redirected to `logs/streamlining/speed_aa_resume.log`. The saved A/A
driver now reuses existing measurement JSONs; it does **not** take fresh timings
for those paths. Change its output root for a new calibration. These drivers
are local experiment artifacts under ignored `logs/`, not repository entry
points available in a fresh checkout.

The reference driver invokes `scripts.benchmark_simulation run` with seed 7,
one Torch thread, `task_timeout`, 100 warmup calls, and five repeats. Its matrix is:

| Task | Backend | Environments | Sets | Timed steps |
|---|---|---:|---:|---:|
| Pendulum | Warp | 4,096 | 1 | 250 |
| Pendulum | VSim | 4,096 | 1 | 1,000 |
| Go2Trot | Warp | 4,096 | 1 | 50 |
| Go2Trot | VSim | 4,096 | 1 | 200 |
| Go2Trot | VSim | 4,096 | 4,096 | 200 |
| Pendulum | CPU | 256 | 1 | 200 |
| Go2Trot | CPU | 256 | 1 | 50 |

Each cell directory contains `0_a.json` through `4_b.json` and corresponding
stdout logs. To recompute a cell's comparison from its saved measurements:

```bash
uv run --frozen -m scripts.benchmark_simulation compare \
    --reference logs/streamlining/speed/references/go2trot_warp_1/{0,1,2,3,4}_a.json \
    --candidate logs/streamlining/speed/references/go2trot_warp_1/{0,1,2,3,4}_b.json \
    --threshold 0.05 \
    --output logs/streamlining/speed/references/go2trot_warp_1/comparison.json
```

This example uses Bash/Zsh brace expansion. For an actual change, supply fresh
candidate files paired with the reference process order. `summary.json` is
written by the local A/A driver; `assessment.md` was assembled from those
process medians and comparison results.

### Raw host profiles, flame graphs, and native CUDA traces

The following loop reproduces the three accepted host captures and the three
native CUDA captures with their recorded directory names. It invokes the same
commands used individually during the run; execute it only with new paths when
preserving existing evidence.

```bash
for name in warp vsim vsim_sets; do
    case "$name" in
        warp) backend=warp; sets=1 ;;
        vsim) backend=vsim; sets=1 ;;
        vsim_sets) backend=vsim; sets=4096 ;;
    esac

    uv run --frozen --env-file .env.vsim -m scripts.profile_simulation \
        --tool py-spy --rate 200 --backend "$backend" --sets "$sets" \
        --task go2trot --num-envs 4096 --profile task_timeout \
        --steps 250 --warmup 100 \
        --output "logs/streamlining/profiles/${name}_host_200hz" \
        > "logs/streamlining/profiles/${name}_host_200hz.log" 2>&1

    uv run --frozen --env-file .env.vsim -m scripts.profile_simulation \
        --tool nsys --backend "$backend" --sets "$sets" \
        --task go2trot --num-envs 4096 --profile task_timeout \
        --steps 250 --warmup 100 \
        --output "logs/streamlining/profiles/${name}_cuda" \
        > "logs/streamlining/profiles/${name}_cuda.log" 2>&1
done
```

The original host captures used `--rate 1000` and directories `${name}_host`;
they are retained as sampling-quality diagnostics. The rate is profiler
sampling frequency, not simulation frequency: physics/control remain 100 Hz.

`--tool py-spy` launches the benchmark with `py-spy record --native --idle
--format speedscope`. It then filters stacks to `profiled_steps` and renders
the SVG automatically. Its outputs are:

- `host.raw.speedscope.json`: complete captured host samples, including setup.
- `host.speedscope.json`: interactive profile data for the timed batch only.
- `host.flamegraph.svg`: static rendering of that filtered profile.
- `capture.json`, `capture.log`, and `workload.json`: command/tool metadata,
  profiler diagnostics, and the profiled workload result.

`--tool nsys` uses CUDA/NVTX/OS-runtime tracing, CUDA graph node tracing, and a
`cudaProfilerApi` capture range. It generates `cuda.nsys-rep`, `cuda.sqlite`,
and the same metadata/log/workload files. `capture.json` preserves the exact
expanded profiler/child command. All profiler timings are excluded from speed
acceptance.

To regenerate the numerical cross-capture assessment without a GPU run:

```bash
uv run --frozen python logs/streamlining/profiles/analyze_profiles.py \
    > logs/streamlining/profiles/analysis_stdout.txt
```

This local helper writes `assessment.json`; `assessment.md` is the separately
written interpretation of its kernel, NVTX, overlap, and sampling summaries.

### Open the interactive profiles

For host flame graphs, open the browser viewer and the folder containing the
profile data:

```bash
xdg-open https://www.speedscope.app/
xdg-open logs/streamlining/profiles/warp_host_200hz
xdg-open logs/streamlining/profiles/vsim_host_200hz
```

Drag `host.speedscope.json` onto the viewer for the timed region, or
`host.raw.speedscope.json` for the complete capture. Select **Left Heavy** for
the aggregated flame-graph view. Speedscope reads these files in the browser.
See the [viewer instructions](https://github.com/jlfwong/speedscope#usage).
If Node/npm is installed, its command-line alternative is:

```bash
npm install -g speedscope
speedscope logs/streamlining/profiles/warp_host_200hz/host.speedscope.json
speedscope logs/streamlining/profiles/vsim_host_200hz/host.speedscope.json
```

Use `vsim_sets_host_200hz` to inspect singleton sets. For native CUDA timelines,
open the raw Nsight reports with the installed desktop viewer:

```bash
nsys-ui logs/streamlining/profiles/warp_cuda/cuda.nsys-rep
nsys-ui logs/streamlining/profiles/vsim_cuda/cuda.nsys-rep
nsys-ui logs/streamlining/profiles/vsim_sets_cuda/cuda.nsys-rep
```

### Correctness logs, test inventory, and package outputs

The normal repository checks are:

```bash
uv run --frozen python -m pytest -q
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
uv run --frozen python -m pytest tests/unit_tests -q -m warp
bash scripts/run_vsim_tests.sh
uv run --frozen ruff check .
```

For the recorded per-case setup/call/teardown reports, temporary local collectors
wrapped these checks. The portable, colocated, and Warp invocations were:

```bash
uv run --frozen python /tmp/q2_streamlining_test_report.py \
    logs/streamlining/portable_test_durations.json -q \
    > logs/streamlining/portable_tests.log 2>&1
uv run --frozen python /tmp/q2_streamlining_test_report.py \
    logs/streamlining/gym_test_durations.json -q gym \
    > logs/streamlining/gym_tests.log 2>&1
uv run --frozen python /tmp/q2_streamlining_test_report.py \
    logs/streamlining/learning_test_durations.json -q learning \
    > logs/streamlining/learning_tests.log 2>&1
uv run --frozen python /tmp/q2_streamlining_test_report.py \
    logs/streamlining/warp_test_durations.json -q tests/unit_tests -m warp \
    > logs/streamlining/warp_tests.log 2>&1
```

VSim retained the real guard/license preflight and loaded a temporary pytest
plugin to collect the same report schema:

```bash
Q2_TEST_DURATION_REPORT=logs/streamlining/vsim_test_durations.json \
PYTHONPATH=/tmp PYTEST_PLUGINS=q2_streamlining_pytest_report \
    bash scripts/run_vsim_tests.sh > logs/streamlining/vsim_tests.log 2>&1
```

Those `/tmp` helpers are session-local, not installed repository tools. The
joined `test_inventory.json` was assembled from collection and duration reports;
`tests/TEST_CATALOG.md` is the manual behavior/cost review. A fresh checkout can
collect all case IDs without executing optional fixtures using:

```bash
uv run --frozen python -m pytest --collect-only -q -m '' \
    tests/unit_tests gym learning > logs/streamlining/test_collection.log
```

The final package build used the installed build backend and wrote artifacts
outside the source tree:

```bash
uv build --no-sources --no-build-isolation --no-cache --offline \
    --python .venv/bin/python --out-dir /tmp/q2-streamlining-final-dist \
    > logs/streamlining/build.log 2>&1
```
