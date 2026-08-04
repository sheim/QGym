---
name: q2-backend-development
description: Implement, modify, or review Q2 physics backends and their task-facing state contract. Use for changes under gym/envs/base involving SimBackend, MuJoCo CPU or Warp, VSim, RobotLayout, state tensors, reset ordering, quaternion conversion, contacts, asset import, backend selection, or a new physics engine.
---

# Q2 Backend Development

Read `AGENTS.md`, `gym/envs/base/sim_backend.py`,
`gym/envs/base/robot_layout.py`, the affected backend, and its contract tests.
Read the relevant current section of `MIGRATION_PLAN.md` for parity decisions;
do not rely on dated branch/status summaries in `.claude/skills/`.

## Preserve the public contract

- Leave every public tensor valid after `setup()` and updated in place after
  `step()` and reset. Tasks cache tensor objects; property getter side effects
  do not satisfy liveness.
- Expose canonical `RobotLayout` order for DOFs, bodies, torques, contacts,
  `dof_state`, and `rigid_body_states`. Keep engine-native indices private.
- Keep `dof_state` persistent with writable `dof_pos`/`dof_vel` views. If an
  engine stores positions and velocities separately, gather into a persistent
  assembled buffer and scatter public reset writes back to native storage.
- Follow write-then-commit resets. Do not refresh native root state over a
  pending public `root_states` write between DOF and root commits.
- Keep task quaternions scalar-last `[x, y, z, w]`; convert once at the
  backend boundary.
- Interpret input torques as canonical full-DOF generalized forces. Apply
  free-joint offsets and native permutations in the backend.
- Expose robot-only bodies and world-frame contact force vectors in Newtons.
  Build semantic contact groups in canonical space.

## Make a backend change

1. State the engine's native conventions: quaternion order, pose/velocity
   layout, contact force meaning and frame, actuation interface, asset losses,
   batching, device support, and deterministic behavior.
2. Decide which tensors are native views and which are assembled copies. Name
   the exact refresh and scatter sites for each.
3. Implement fixed-base pendulum setup/step/reset first. Fail immediately on
   unsupported devices or joint types.
4. Add the backend fixture and run shared contract and damped-energy tests.
5. Compare pendulum lockstep or analytic behavior against MuJoCo CPU.
6. Add floating-base root state, canonical body state, ground/contact support,
   and partial resets.
7. Add named routing, legged contract, termination, and task-level liveness
   tests before RL smoke tests.
8. Wire selection through `task_registry.select_backend()` without fallback.
9. Benchmark only after correctness gates pass; record environment count,
   timestep, rollout geometry, device, warmup, and warnings.

## Engine-specific scars

- MuJoCo imports URDF effort/velocity limits incompletely; Q2 reparses them in
  `urdf_limits.py`. Preserve that path and add asset-specific tests.
- MuJoCo may fuse fixed-linked bodies; Q2 disables `fusestatic` so canonical
  bodies such as feet remain addressable.
- MuJoCo uses scalar-first quaternions, angular-first body velocities, and
  `[torque, force]` spatial contact arrays. CPU and Warp must both call the
  post-constraint RNE operation before exposing contact forces.
- MuJoCo Warp native arrays are zero-copy, but canonical/swizzled tensors are
  assembled. Refresh them eagerly. Forward `njmax` to Warp data allocation;
  a constraint overflow is incorrect physics, not a harmless warning.
- MuJoCo CPU and Warp model construction belongs in
  `mujoco_backend_base.py`. Avoid backend-specific physics drift.
- VSim motor, articulation, body, and sensor order are independent. Route by
  exact native indices/names, rotate link-attached force components to world
  axes, and close its process singleton cleanly.
- Per-engine config pass-throughs must be explicit, validated, and covered by
  a test showing that the engine actually consumes them.

## Validate

Start with targeted tests, then run:

```bash
uv run --frozen python -m pytest -q
```

For VSim changes also run:

```bash
bash scripts/run_vsim_tests.sh
```

A GPU-specific test skipped on a CPU machine is not evidence that the GPU path
works. For physics claims, add a predicted invariant, lockstep comparison, or
fidelity probe. For state/reset/contact bugs, land the regression test with
the fix and update `MIGRATION_PLAN.md` if campaign evidence changes.
