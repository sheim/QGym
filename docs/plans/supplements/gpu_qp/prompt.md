# Investigation: GPU QP, SQP, and NLP for a discrete-time Q-derived CBF

## Instructions for the build agent

Perform a disposable feasibility investigation, not a product implementation. The user will launch you separately.

Workspace: `/home/brad/Documents/test_projects/test_legged_robot`.
Product: `/home/brad/Documents/test_projects/test_legged_robot/BigRedGym`.

Read workspace/product `AGENTS.md` and load relevant environment, backend, RL, and testing skills. Preserve unrelated changes. Do not modify product source, configs, lockfiles, or the existing `.venv`. Do not commit or add todos without approval.

Create a unique temporary directory under `/tmp/opencode/gpu-qp-<date>-<suffix>/`. All environments, clones, experimental code, caches you control, large artifacts, and profiler traces belong there. Keep it until review.

Create the permanent report directory explicitly:

`/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/docs/plans/supplements/gpu_qp/`

Write `REPORT.md` there, with small sanitized reproduction scripts under `scripts/`. Copy this investigation prompt there as `prompt.md`, leaving this source prompt intact. These documentation writes are the only permitted changes to the original workspace. Do not overwrite different existing files without approval. Keep each permanent file below 100 KB; no binaries, environments, checkpoints, or raw traces there.

## Purpose

Find and test released, off-the-shelf GPU solvers suitable for a discrete-time CBF controller. The controller must be extensible across:

- QP using a locally linearized constraint.
- SQP using repeated local QPs.
- General constrained nonlinear optimization.

The QP/NLP numerical solve must execute on GPU, not merely its model or derivatives. A CPU solver calling GPU model evaluations does not qualify. Do not conflate JIT compilation, differentiability, or GPU tensor inputs with GPU optimization.

Initial scope is one evaluation environment with 12 action variables, not large-scale controller batching. Slower-than-real-time functional operation is acceptable. Report batching support as future capability, not a required implementation.

## Mathematical problem

Let the nominal policy be pi, the loaded fixed-policy critic be Q_r'^pi, and:

```text
h(s) = Q_r'^pi(obs(s), pi(obs(s))) - c
safe_set = {s : h(s) >= 0}
g(u) = h(F(s, u)) - (1 - beta) * h(s),  0 < beta <= 1

minimize  0.5 * (u - u_nom)^T W (u - u_nom)
subject to g(u) >= 0
           u_min <= u <= u_max
```

W must be positive definite. F predicts one complete Q2 control interval from the full prediction state, including the history and task fields needed to construct successor observations. Actions are Q2 policy-unit joint targets, not direct torques.

The Q-defined set is an experimental return superlevel set, not a previously certified barrier. Investigate feasibility and numerical behavior; do not require or claim a forward-invariance proof.

The nonlinear problem is not generally a QP. A QP variant linearizes g around a reference action; label that approximation explicitly. Optional slack and trust-region bounds must be explicit. Never equate a feasible relaxed or linearized problem with satisfying the original nonlinear constraint.

Use the nominal policy convention recorded in the artifact. Distinguish a policy-mean action from expectation over a stochastic policy. If testing a deterministic mean approximation to an artifact trained for stochastic continuation, name that choice rather than claiming exact value equivalence.

## Existing evidence to inspect

Locate and read the MJX Go2Trot conversion report, starting at:

`/tmp/opencode/mjx-go2trot-20261115-a/REPORT.md`

Also inspect any permanent copy under `BigRedGym/docs/plans/`. If files have moved, locate them rather than guessing results.

Reported findings to verify before reuse:

- MuJoCo/MJX 3.11.0 with JAX/JAXlib and CUDA13 plugin 0.9.2 executed on the RTX 5090 with Python 3.11.
- Original Go2Trot conversion fails on cylinder-box collisions. A named diagnostic model omits 64 such self-collision pairs and runs on GPU.
- Float64 prediction matched tested CPU reference states closely.
- Forward-mode derivatives work; reverse-mode through the tested physics solver fails because of dynamic loops.
- Combined Torch/JAX CUDA-library compatibility was not fully established.

Reuse that temporary experiment read-only where useful; do not mutate or delete it. Keep omitted collision pairs and other model approximations explicit. Do not claim baseline physics equivalence.

Read `BigRedGym/docs/plans/supplements/full_gpu_mppi/REPORT.md` for GPU profiling lessons. That report observed scalar device-to-host reads: those are real transfers even if not observation/action payloads. Do not relabel them as zero traffic.

## 1. Survey solver candidates

Start with QPax and JAXopt BoxOSQP for QP; research current GPU SQP and general constrained NLP alternatives. Prefer released packages. Consider other frameworks if needed, but assess integration and transfer costs with the JAX model.

For each serious candidate record:

- Release/version, license, maintained source and documentation, Python/JAX/CUDA requirements.
- Supported constraints, precision, warm starts, iteration/time limits, statuses, infeasibility behavior, slack handling, and batching.
- Whether the optimizer itself runs on GPU and whether iteration decisions require host reads.
- Required derivatives: gradient, Jacobian, Hessian, Hessian-vector product, or only function values.
- Whether derivatives can be supplied explicitly instead of invoking reverse-mode through MJX.
- Whether an advertised NLP solver actually handles general nonlinear inequalities; penalties alone are not equivalent to a constrained solve.

Do not force a recommendation if no suitable released GPU NLP exists. Distinguish an off-the-shelf QP solve from the custom SQP/merit/line-search loop we would need to write around it.

Choose at least one QP candidate and one credible nonlinear route for hands-on testing if available. If no nonlinear route can be tested, give a source-backed reason and mark it untested rather than substituting CPU optimization.

## 2. Isolated dependency compatibility

Inspect current Q2 pins and the existing MJX experiment freeze. Build isolated environments preserving Q2's Python, Torch, NumPy, MuJoCo, and Warp versions where possible. Record every necessary change.

Test actual same-process imports and small GPU computations involving Torch, JAX, the solver, and MJX/Warp as appropriate; successful dependency resolution alone is insufficient. Record CUDA wheel changes and device placement. Do not modify Q2's existing environment to perform this check.

## 3. Test numerical behavior before robot integration

Use small 12-variable cases with known solutions or independently checked reference solutions:

- Nominal action already feasible; active barrier; action bounds active.
- Infeasible hard constraints; an explicitly relaxed counterpart.
- Poor scaling, nearly zero constraint gradient, and iteration-limit termination.
- Nonlinear constraints where a feasible linearized QP proposal violates the true constraint.

A CPU reference solver is allowed only for offline validation, never as the controller's implementation or silent fallback.

Report original constraint residuals, bounds, stationarity/optimality diagnostics where available, solver status, and slack. Nonconvergence and non-finite outputs must be distinguishable from true infeasibility; do not treat one failed local NLP solve as proof no feasible action exists.

## 4. Test MJX and Q integration

Use the existing named MJX diagnostic model with the same PD/control-step semantics. Test differentiation through the full composite g(u), not only a scalar position probe.

Locate a real compatible policy and Q artifact read-only. For supported network architectures, implement a disposable JAX evaluation and compare it against Torch, including observation normalization, action scaling, and policy outputs. Do not train new artifacts. Do not expose privileged physics inputs to a Q trained only on actor observations.

Reconstruct the actual successor observation, including required history, commands, phase, and previous-action fields. Keep terminal behavior explicit. If this cannot be done faithfully in the investigation budget, report the blocker and separate simpler model/barrier tests from real-Q evidence.

Use forward-mode action derivatives where MJX reverse-mode is unavailable. Determine whether the optimizer accepts those derivatives. For second-order methods, examine quasi-Newton or supplied approximations instead of assuming exact Hessians work. Differentiating through the optimization solution is not required for this inference controller.

Test a few actual states: free flight, sustained contact, contact change, and partial/full torque saturation where available. Include cases inside and outside the Q-defined set if obtainable; record chosen thresholds and do not claim they were independently calibrated.

For each solver proposal, re-evaluate the original nonlinear g(u). A solved local QP is not sufficient evidence. Report refinement behavior and explicit failure outcomes; do not silently execute an unchecked proposal.

## 5. Propose the smallest replaceable solver interface

Return concise Python-adjacent interface sketches, not production code. Separate:

1. Model/barrier evaluation and supplied derivatives.
2. The full constrained controller problem.
3. Solver implementations and their saved state.

The same controller-facing problem should support a linearized-QP strategy, SQP, or general NLP without pretending each solver consumes identical low-level inputs. Describe capabilities and adapters: QP matrices versus nonlinear callbacks, first/second derivatives, bounds, warm starts, and device/dtype requirements.

Suggested controller-facing shape, revise as evidence warrants:

```python
class ConstrainedSolver:
    @classmethod
    def build(cls, settings, problem_spec):
        # Prepare solver state and compiled functions.

    def solve(self, problem, initial_action, warm_start):
        # Return action, status, residuals, and updated saved state.
```

Clarify who linearizes constraints, controls repeated solves, checks the original constraint, and decides fallback. Do not hide SQP inside a supposedly plain QP solver or let different solvers silently change the objective/constraints.

## 6. Audit GPU execution and measure cost

Profile the warmed full boundary:

```text
state, nominal action, artifacts already on GPU
  → model/barrier evaluation and derivatives
  → optimization iterations
  → original-constraint check
  → action, status, and residuals on GPU
```

Use Nsight Systems if available; label all stages. Report separately:

- Model, derivative, Q/policy, solver, and final-check device placement.
- Host↔device copy counts/bytes, including scalar predicates and status reads.
- Host synchronization and CPU decisions between iterations.
- Device-to-device exchange and stream ordering across Torch/JAX if used.
- Compilation versus warmed latency and memory.

CPU kernel launches are allowed; CPU numerical QP/NLP solves do not qualify. Any host-controlled optimization or scalar reads must be disclosed rather than called fully GPU resident. If profiler evidence is unavailable, mark strict residency unverified.

Test one controller first, then a small batch only if cheap. Preserve a conservative GPU-memory budget and do not disturb unrelated jobs. Do not extrapolate unmeasured CUDA-graph speedups or performance at scale.

## Final report

Write:

`/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/docs/plans/supplements/gpu_qp/REPORT.md`

Include:

1. Verdict matrix for each candidate: installed compatibility, GPU numerical solve, supplied forward derivatives, original-constraint satisfaction, failure handling, zero host transfers, and real-MJX/real-Q integration. Use passed/failed/untested with conditions.
2. Exact package/source versions, licenses, GPU/driver, artifact identities, model modifications, and environment freeze/reproduction instructions.
3. Numerical results, original residuals, solver/slack outcomes, timings, memory, and profiler trace locations.
4. Minimal replaceable interfaces with explicit QP/SQP/NLP differences.
5. Recommended first implementation and alternatives, with unresolved blockers and evidence limits. Separate functional feasibility from safety and intervention quality.
6. Reproduction scripts and cleanup instructions. List permanent documentation writes separately and confirm original implementation/configuration/environment were unchanged.

Keep all permanent files under 100 KB. Record raw evidence locations in `/tmp` and identify what would be lost on cleanup. Return the report path, temporary directory, recommended candidate, and brief blockers in your final response.
