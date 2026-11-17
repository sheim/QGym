# Investigation: fully GPU MPPI for Q2 Go2Trot

## Instructions for the general execution agent

The user will launch you separately. Perform this investigation; do not merely draft another plan.

Workspace root: `/home/brad/Documents/test_projects/test_legged_robot`.

First read workspace and product `AGENTS.md`, and load the relevant Q2 environment, backend, RL, and testing skills. Follow their rules while preserving unrelated changes.

### Explicit permanent documentation locations

Create this directory if it does not exist:

`/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/docs/plans/supplements/full_gpu_mppi/`

Move this prompt from:

`/home/brad/Documents/test_projects/test_legged_robot/.opencode/workspace/plans/investigations/full-gpu-mppi/prompt.md`

to:

`/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/docs/plans/supplements/full_gpu_mppi/prompt.md`

Preserve its contents. Confirm the destination exists with matching contents before removing the source. Do not overwrite a different existing destination; ask the user if one exists.

Write the final report at:

`/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/docs/plans/supplements/full_gpu_mppi/REPORT.md`

Put small, sanitized reproduction scripts under:

`/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/docs/plans/supplements/full_gpu_mppi/scripts/`

Keep each permanent file below 100 KB. Do not put environments, binaries, checkpoints, raw recordings, or profiler traces in these documentation directories. Preserve an existing report rather than overwriting it without approval. Do not commit anything unless the user asks.

These documentation writes and the requested prompt move are the exceptions to the isolation rule below. Do not modify product implementation, configuration, dependency files, or the existing environment. Do not add operational todos without approval.

## Goal

Determine whether a controller using released `pytorch-mppi` can perform this complete computation on GPU:

**candidate generation → physics prediction → nominal-policy rollout → Q/cost evaluation → action selection**

Require no CPU↔GPU data transfers in the warmed controller hot path. CPU orchestration and kernel launches are allowed. Report CPU synchronization separately from data transfers.

This is a feasibility investigation, not product implementation.

## Temporary isolation

- Product checkout: `/home/brad/Documents/test_projects/test_legged_robot/BigRedGym/`.
- Create a unique experiment directory under `/tmp/opencode/`, for example `/tmp/opencode/full-gpu-mppi-<date>-<unique-suffix>/`. Report its exact absolute path.
- Keep source copies, isolated uv environments, experimental scripts, caches you control, large results, and profiler traces inside that temporary directory.
- Preserve current relevant source, including uncommitted changes. Do not modify the original checkout's implementation, `.venv`, `pyproject.toml`, or `uv.lock`.
- Do not copy secrets, licenses, or large logs indiscriminately.
- Keep the temporary directory until review; document cleanup. The permanent report and reproduction instructions must remain useful after temporary artifacts are deleted, with any lost raw evidence explicitly identified.

## Candidate implementation

Use:

- Released `pytorch-mppi`, initially version 0.9.1 if compatible.
- Q2's MuJoCo Warp physics path.
- Existing Torch nominal-policy and Q artifacts on CUDA.
- Torch/Warp GPU array exchange where needed.

Do not install MPPI's upstream `test` extra: it introduces OpenAI Gym, conflicting with Q2's `gym` namespace. Do not assume the package's advertised GPU support establishes end-to-end compatibility.

## 1. Establish dependencies and available artifacts

Inspect actual Q2 dependency versions and GPU/driver information. Start by preserving Q2's locked Torch, NumPy, MuJoCo, mujoco-warp, and warp-lang versions.

Install candidate dependencies only into the isolated environment. Record dependency changes and conflicts.

Locate a compatible nominal policy and Q artifact read-only. Record checkpoint hashes, observation/action conventions, normalization, reward/return convention, and control interval.

If compatible artifacts are unavailable, distinguish synthetic wiring tests from real-artifact evidence. Do not train replacements.

## 2. Define the rollout semantics before testing

Explain how the library represents candidate trajectories and how the nominal policy and Q participate.

Distinguish:

- Sampling action sequences and scoring them with physics rollouts.
- Using the nominal policy inside predicted rollouts.
- Using Q as a terminal value or another explicitly defined objective.

Implement at least one coherent experiment that actually exercises physics, policy inference, and Q evaluation inside prediction. State its equations or concise pseudocode.

Do not silently change MPPI's sampling/update rule to accommodate policy-guided actions. If proper integration requires a subclass, adapter, or upstream changes, identify exactly what is required. A random-shooting test alone does not demonstrate MPPI.

## 3. Build a disposable prediction adapter

Reuse the actual configured Go2Trot model and action semantics:

- Canonical/native state and action ordering.
- Action scaling and default joint-position offsets.
- PD control and torque clipping.
- All physics substeps per control interval.
- Required physics state, model parameters, commands, observation history, and policy state.

Branch predictions into separate GPU worlds without advancing the evaluation environment.

Verify that each candidate starts from the intended state and physical parameters. Exclude automatic resets from predictions or implement explicit terminal handling.

Start with one evaluation environment and many candidate worlds. Separately assess multiple evaluation environments: candidate batching is not automatically independent-controller batching.

## 4. Verify prediction and controller correctness

Before measuring throughput:

- Compare a candidate prediction against the normal Q2 step from matched state/action/model inputs.
- Check predicted observations against Q2's observation construction.
- Check policy and Q outputs against their ordinary artifact inference paths.
- Confirm a deliberately different candidate action changes the predicted trajectory/cost.
- Confirm the evaluator's state is unchanged by controller rollouts.
- Check terminal masking, action bounds, finite outputs, reset behavior, and reproducibility under explicit seeds.

Use numerical tolerances appropriate to Warp; do not claim bitwise determinism without evidence.

## 5. Audit the complete GPU hot path

Define the measured boundary as:

```text
controller.action(input_already_on_gpu) → action_on_gpu
```

Artifact loading, initial model upload, compilation, and later report writing are outside this boundary.

After warm-up:

1. Profile several complete calls with Nsight Systems, using ranges for candidate generation, physics, observation construction, policy, Q/cost evaluation, and MPPI update.
2. Inspect CUDA transfer events, including host-to-device, device-to-host, and unified-memory migrations if applicable.
3. Identify host synchronization separately: `.item()`, Python branching on GPU values, blocking conversions, and explicit synchronization.
4. Inspect Torch/Warp exchange for copies, aliasing/lifetime hazards, stream ordering, and device placement.
5. Check that per-step state restoration and model updates do not introduce hidden host transfers.

Device-to-device copies are allowed but should be counted and explained. CUDA kernel launches from Python are allowed.

If Nsight is unavailable, use available profiling tools and source inspection, but label absence of CPU↔GPU traffic as unverified, not passed merely because tensors report `cuda`.

## 6. Measure practical cost

Warm up before timing and synchronize at measurement boundaries, not unnecessarily inside the controller.

Start small and increase candidate count and horizon cautiously. Report:

- Candidate count, horizon, number of evaluation environments.
- Physics steps and policy/Q calls per decision.
- Compilation/setup versus warmed decision latency.
- Physics, policy, Q, and controller-update timings.
- Peak memory and allocator settings.
- Transfer counts/bytes and synchronization findings.
- Throughput and any failed/OOM configurations.

Do not exhaust the GPU or disrupt unrelated jobs. Establish a conservative memory budget from available resources.

Assess whether CUDA graphs or other optimization would be necessary, but do not make them prerequisites merely to demonstrate residency.

## Final report

Write the report at the permanent `BigRedGym/docs/plans/supplements/full_gpu_mppi/REPORT.md` path specified above, not only in `/tmp`.

Include:

1. Verdict table: dependency coexistence, actual MPPI integration, GPU physics, GPU policy rollout, GPU Q evaluation, GPU selection, no host transfers, and multi-environment support—each passed, failed, or untested.
2. Exact environment: versions, sources, GPU/driver, artifact identities, and source provenance.
3. Algorithm definition: how policy and Q are used, and whether upstream MPPI remains unchanged.
4. Correctness evidence: state/observation/inference comparisons and isolation checks.
5. Profiling evidence: trace paths, measured boundaries, transfer/synchronization findings, timing and memory tables.
6. API implications: required controller inputs, dynamics methods, saved controller state, resets, and batching limits.
7. Blockers and alternatives: distinguish technical impossibility from missing adapters or performance work.
8. Reproduction and cleanup: exact commands, scripts, seeds, temporary directory, and confirmation that original product implementation/configuration/environment were unchanged. List the permanent documentation files created and the prompt relocation separately.

Keep the report concise and portable. Use relative links to permanent reproduction scripts and explicit absolute paths for temporary evidence. Do not claim fully GPU execution from package documentation or tensor device labels alone.

In your final response, return the permanent report path, relocated prompt path, temporary experiment directory, and a brief verdict with remaining blockers.
