# Go2Trot MJX-JAX feasibility investigation

Intended executor: OpenCode Go / DS v4.1 Flash execution agent.

## Task

Test whether Q2's actual Go2Trot MuJoCo model can be converted to MJX-JAX and used for GPU prediction and differentiation.

This is an isolated feasibility experiment, not a product implementation. Do not modify the workspace checkout, its environment, or its lockfile.

## Workspace and isolation

- Workspace: `/home/brad/Documents/test_projects/test_legged_robot`
- Product: `BigRedGym/`
- Read workspace `AGENTS.md`, `BigRedGym/genAI_skills/AGENTS.md`, and the relevant Q2 backend, environment, and testing skills.
- Create a unique experiment directory under `/tmp/opencode/`. Keep all copies, clones, virtual environments, scripts, caches you control, and results inside it.
- Copy the necessary product files or create an independent checkout. Preserve the current relevant source, including uncommitted changes; a clean Git clone alone may not represent the current implementation.
- Never copy license files, `.env` secrets, `.venv`, or large logs indiscriminately.
- Use an isolated uv environment. Do not install into Q2's existing `.venv`.
- Keep the experiment directory until we review the report. Document how to delete it.

## Questions to answer

1. Can the configured `MjModel` used by Go2Trot convert through `mjx.put_model()` to MJX-JAX, without changing robot geometry, dynamics, or contact settings?
2. Can it execute on this machine's NVIDIA GPU?
3. Can we differentiate predictions with respect to Go2Trot's joint-target actions, including PD control and torque limits?
4. How closely does one full control-step prediction match the existing Q2 backend?
5. What changes, limitations, runtime costs, or dependency conflicts would integration introduce?

Distinguish MJX-JAX from MJX-Warp: a successful Warp execution does not establish JAX differentiation.

## Dependency investigation

Inspect actual `pyproject.toml`, `uv.lock`, and installed/device information rather than trusting these preliminary observations:

- Python 3.11
- Torch 2.13.0
- NumPy 2.4.6
- MuJoCo and mujoco-warp 3.11.0
- warp-lang 1.16.0
- NVIDIA RTX 5090, CUDA 13-era driver/runtime

Start with MuJoCo/MJX version alignment that preserves Q2's locked MuJoCo version. Check published wheel availability and Python/CUDA requirements.

Earlier metadata research suggested JAX/JAXlib 0.7.2 as a Python-3.11-compatible candidate, but GPU compatibility was not verified. Do not treat this as a required or working pin.

If the baseline cannot work, explain the blocker before trying a clearly separated alternative environment. Never silently upgrade Q2's physics stack.

## Experiment

### 1. Trace the actual model construction

Identify the source files and config fields that produce Go2Trot's MuJoCo model. Record physics timestep, control timestep, substeps, solver/integrator, contacts, actuator setup, and domain randomization.

Use the actual generated model, not a simplified hand-written Go2 XML. Record which environment lane's physical parameters are tested.

### 2. Convert and execute

Convert the model and required initial state to MJX-JAX. Demonstrate device placement and execute GPU steps. Report unsupported features, conversion warnings, compilation failures, and non-finite results.

Do not disable contacts or change solver settings silently. Any such change creates a separately named diagnostic variant—not a successful baseline.

### 3. Reproduce one Q2 control interval

Build a disposable prediction adapter covering:

- Canonical/native joint and action ordering.
- Action scaling and default-position offsets.
- PD control, including when it is recomputed.
- Torque clipping.
- All physics substeps in one control interval.
- State fields required to initialize equivalent predictions.

Do not replace the control interval with one physics step and call them equivalent.

### 4. Check prediction agreement

Use a small, reproducible set of valid states/actions covering:

- Standing/contact.
- A moving state, preferably from an actual short Go2Trot rollout.
- Near-contact-change behavior if obtainable.
- A torque-saturation case.

Compare identical starting states, physical parameters, and actions against Q2's current backend. Prefer Warp for the target comparison; CPU MuJoCo may provide an additional reference.

Report numerical differences in joint position/velocity, root position/orientation/velocity, and relevant contacts. Handle quaternion sign equivalence correctly. Separate one-step error from short-rollout divergence.

If a trained policy is needed, locate a suitable artifact read-only and record its identity. Do not train a policy. Report unavailable cases rather than inventing inputs or evidence.

### 5. Check derivatives

Test derivatives of the full control-step prediction with respect to joint-target actions.

- State which forward/reverse differentiation modes work.
- Compare autodiff directional derivatives with finite differences across several step sizes.
- Include contact and saturation cases.
- Separate expected zero derivatives from clipping from failed or uninformative contact derivatives.
- Check finiteness and repeatability.

A scalar probe of predicted state is sufficient for this experiment; converting a Q network or implementing a complete CBF is out of scope.

### 6. Measure cost

Report compilation time separately from warmed execution. Synchronize GPU work before timing.

Start with small batches, then test a few practical batch sizes within available memory. Measure forward prediction and derivative computation separately. Record GPU memory and failures; do not launch uncontrolled large compilations or exhaust the GPU.

## Report

Write `REPORT.md` inside the experiment directory and return its absolute path.

Include:

1. Verdict: model conversion, GPU execution, action derivatives, prediction agreement—each marked passed, failed, or untested.
2. Exact environment: dependency versions, wheel/source origins, Git revisions where used, Python, GPU, driver, and device placement.
3. Model provenance: source/config paths, physical settings, randomization, and model modifications.
4. Reproduction: commands, scripts, seeds, and artifact identities.
5. Results: numerical comparison and derivative tables, timings, memory, warnings, and failures.
6. Integration implications: what can be reused unchanged, what needs a JAX adapter, and remaining blockers for a Q-derived CBF.
7. Evidence limits: distinguish successful conversion from accurate prediction and useful gradients.
8. Cleanup: experiment directory and deletion instructions; confirm the original workspace and environment were not modified.

Keep conclusions empirical. Do not claim "compatible," "differentiable," or "equivalent" solely because an import or `put_model()` succeeds.
