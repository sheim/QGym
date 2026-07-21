---
name: mujoco-backend-reference
description: MuJoCo and mujoco-warp domain knowledge as it applies to Q2 — quaternion/coordinate conventions vs IsaacGym, qpos/qvel/free-joint layout, qfrc_applied, cfrc_ext semantics, the MjSpec URDF pipeline (discardvisual, fusestatic, .dae stripping), warp's put_model/put_data/zero-copy views/JIT/njmax, and per-backend step/reset mechanics. Load when writing or reviewing backend code, translating IsaacGym idioms, or reasoning about MuJoCo model/data fields. NOT Q2's own invariants (q2-architecture-contract) and NOT general RL knowledge (legged-rl-reference).
---

# MuJoCo Backend Reference (as applied in Q2)

Definitions once: **MJCF** = MuJoCo's XML model format; **MjSpec** = mutable
in-memory model description (edit, then `compile()` → `MjModel`); **warp** =
NVIDIA Warp, the kernel framework `mujoco_warp` (a.k.a. mjwarp) builds on;
**nworld** = warp's batch dimension (= num_envs).

## Conventions that differ from IsaacGym (memorize these)

| Concept | MuJoCo | IsaacGym / Q2 task layer |
|---|---|---|
| Quaternion order | `[w,x,y,z]` scalar-FIRST | `[x,y,z,w]` scalar-LAST |
| Floating base | free joint: `nq = 7 + n_joints`, `nv = 6 + n_joints` | root state tensor `[13]` |
| Root velocities | `qvel[0:3]` linear, `qvel[3:6]` angular | same order in root_states 7:10 / 10:13 |
| Body spatial velocity `cvel` | `[ang(3), lin(3)]` — ANGULAR FIRST | rigid_body_states wants lin 7:10, ang 10:13 |
| Contact forces `cfrc_ext` | body frame, `[torque(3), force(3)]`, per body | world-frame force `[N, bodies, 3]` |
| Applied forces | `qfrc_applied` in generalized coords | per-DOF torques |

Q2's swizzles: `WXYZ_TO_XYZW = [1,2,3,0]`, `XYZW_TO_WXYZ = [3,0,1,2]`
(`gym/envs/base/mujoco_backend_base.py:19-20`). They appear in exactly four
places per backend (root read, root write, body-quat read, and reset) — if you
add a fifth, you're probably in the wrong layer.

**`cfrc_ext` trap (cost a real bug):** `mj_step` / warp `forward+euler` leave
`cfrc_ext` at ZERO. You must call `mujoco.mj_rnePostConstraint(m, d)` (CPU) or
`mjw.rne_postconstraint(m, d)` (warp) after stepping to populate contact
forces. Q2 does this in both backends' `step()`. Also note Q2 takes
`cfrc_ext[..., 3:6]` (the force half) and the values are body-frame — contact
*thresholds* tuned on IsaacGym world-frame forces may need revisiting
(MIGRATION_PLAN "Gotchas").

## The URDF → MjSpec pipeline (Q2-specific, `mujoco_backend_base.py:88-172`)

Order of operations in `_load_model`:
1. `_parse_urdf_limits(path)` — ElementTree pass caching `{joint: (effort,
   velocity)}`, because **MuJoCo's URDF importer discards `<limit
   effort/velocity>`** (it expects them on actuators; Q2 creates no actuators
   and drives `qfrc_applied` directly).
2. `_load_urdf_spec(path)` — pre-processes the URDF XML: strips `<visual>`
   blocks whose meshes are `.dae/.collada` (MuJoCo can't decode them), then
   injects `<mujoco><compiler discardvisual="false" strippath="false"/></mujoco>`.
   The injected tag does double duty: keeps supported visual meshes AND keeps
   the resulting MjSpec mutable — without it, later `add_texture`/`add_material`
   calls are **silently pruned at compile**. `spec.modelfiledir` is set so
   relative mesh paths resolve.
3. `spec.compiler.balanceinertia = True`.
4. Floating base: if `not cfg.asset.fix_base_link`, add a free joint named
   `root` to the first body.
5. Menagerie-style visuals (azimuth 150, elevation −20, shadowsize 4096,
   gradient skybox, directional light); checker ground plane only when
   `cfg.terrain.mesh_type == "plane"` (friction from terrain cfg, ground size
   field `[0,0,0.05]` = infinite plane).
6. `spec.compile()` → MjModel; then timestep, gravity, per-DOF
   damping/armature are poked directly on the model.

Post-compile facts used downstream: free joint detection `nq == nv + 1`;
DOF names skip joint 0 when floating; body/joint names via `mj_id2name` (empty
names get `body_i`/`joint_i` placeholders).

**`fusestatic`:** MuJoCo fuses bodies connected by rigid joints; mini_cheetah's
`foot` merged into `shank` and broke name lookups (and starved GRF rewards of
feet). `spec.compiler.fusestatic = False` in `_load_model` (pulled from
jt/port into `vsim` 2026-07-11) keeps them — mini_cheetah now has 18 bodies
incl. 4 feet. Side effect: model changes shift chaotic CPU↔warp divergence
curves; see the two-window lockstep test.

## CPU backend mechanics (`mujoco_cpu_backend.py`)

- One shared `MjModel`, one `MjData` per env; `step()` = Python loop:
  `d.qfrc_applied[off:] = torques[i]; mj_step(m, d)` then a full
  numpy→torch sync of every state tensor (`_sync_state_from_mujoco`).
  Cost scales linearly in num_envs; tensors are honest copies, always fresh.
- Resets: write torch views → `d.qpos/qvel` → `mj_forward` per env.
- Rendering: `mujoco.viewer.launch_passive(m, datas[0], key_callback=...)` —
  env 0 only, lazily created; key callback goes through a `_viewer_key_callback`
  indirection (viewer exists before interfaces attach).

## Warp backend mechanics (`mujoco_warp_backend.py`)

- Build: `wp.init()`; `wp.ScopedDevice(device)` wraps ALL warp calls;
  `mjw.put_model(mjm)`; `mjw.put_data(mjm, MjData(mjm), nworld=num_envs)`.
- **Zero-copy views:** `wp.to_torch(d.qpos)` etc. give torch tensors sharing
  warp storage — writes from torch land in the sim with no copy. Q2 keeps
  views of qpos, qvel, qfrc_applied, cfrc_ext, xpos, xquat, cvel.
- Step = `qfrc copy → mjw.forward → mjw.euler → mjw.rne_postconstraint` (this
  is mj_step decomposed; forward+euler ≙ semi-implicit Euler).
- Reset = write into zero-copy views, then `mjw.forward` (no data upload
  needed — the write already happened in-place).
- **Assembled tensors are NOT zero-copy:** `root_states` and
  `rigid_body_states` are scratch tensors filled by swizzling the views —
  refreshed by `_sync_assembled_states()` in `step()`, `setup()`, and the
  reset methods (fix landed 2026-07-11; before that the refresh lived in the
  property getters, the historical staleness bug — q2-architecture-contract
  W1/W2). `dof_state` is a per-call `torch.stack` copy (interleaved view over
  separate qpos/qvel arrays is impossible). Direct views (`dof_pos`,
  `dof_vel`, `contact_forces` slice) ARE always live.
- **JIT:** first `mjw.forward` compiles kernels — minutes-long silent pause on
  first run, cached afterwards. Don't Ctrl-C it (see debugging playbook row 2).
- **Capacity knobs:** `njmax` (constraint-Jacobian rows per world) and
  `opt.ccd_iterations`. "nefc overflow" warnings mean constraints are being
  SILENTLY DROPPED — wrong physics, not just noise. Two traps: (1) warp
  ignores the legacy `mjModel.njmax` field — Q2's warp backend forwards it to
  `mjw.put_data(njmax=...)`; (2) values are per-model — mini_cheetah needs
  200 with fusestatic off (was 90 before). `ccd_iterations` is a fixed
  per-step cost: 50→100 halved training throughput (13.2k→7.1k steps/s,
  measured 2026-07-11). Configure both via `cfg.mjspec_attributes` /
  `cfg.mjspec_option_attributes` (q2-config-system).
- Sliced views like `qpos[:, 7:]` are non-contiguous; MIGRATION_PLAN flags
  verifying writes propagate through `mjw.forward` — the contract tests cover
  this for dof state.
- No viewer exists for warp; headless only.

## Fixed-base vs floating-base handling

- Fixed-base (pendulum, cartpole): asserts `nq == nv`; **all contacts disabled**
  (`geom_contype[:] = 0`, `geom_conaffinity[:] = 0`) — there is no ground
  plane interaction at all; do not add contact-based rewards to fixed robots.
- Floating-base: offsets qpos 7 / qvel 6; torques applied to
  `qfrc_applied[:, 6:]`; contacts left enabled; ground plane from terrain cfg.

## When NOT to use this skill

- Q2's own tensor/lifecycle contracts → `q2-architecture-contract`.
- Symptom lookup → `q2-debugging-playbook`.
- Reward/obs semantics → `legged-rl-reference`.
- Upstream API details beyond what Q2 uses → mujoco.readthedocs.io (allowed
  fetch domain in project settings).

## Provenance and maintenance

Facts verified against `port` @ `bc2bd96`, mujoco≥3.6 / mujoco-warp≥3.6 pins,
2026-07-10. Re-verify:

```bash
uv run python -c "import mujoco; print(mujoco.__version__)"
sed -n "88,175p" gym/envs/base/mujoco_backend_base.py     # pipeline drift
grep -n "rne_postconstraint\|fusestatic\|njmax" gym/envs/base/*.py
```
