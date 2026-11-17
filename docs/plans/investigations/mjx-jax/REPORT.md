# Go2Trot MJX-JAX feasibility investigation — REPORT

Experiment directory: `/tmp/opencode/mjx-go2trot-20261115-a`

Date: 2026-11-15

Source checkout: `BigRedGym` at `77488a0b98a28bed1bb58cf5a2862503f14696ae`
(branch `bh574/final/dev/offline_RL`, clean worktree). The experiment uses an
independent copy at `/tmp/opencode/mjx-go2trot-20261115-a/checkout`.

## 1. Verdict

| Question | Verdict | Basis |
|---|---|---|
| Convert the configured Go2Trot `MjModel` through `mjx.put_model()` without changing geometry, dynamics, or contact settings | **FAILED** | MJX 3.11.0 has no `(cylinder, box)` collision function. The baseline model contains 64 such candidate pairs (robot self-collision, e.g. base↔calf, head↔thigh, cross-leg hip↔thigh). `put_model()` raises `NotImplementedError` before any execution. MJX 4.0.1 has the same missing pair. |
| Execute on this machine's NVIDIA GPU | **PASSED for the diagnostic variant only** | JAX placed model/data on `CudaDevice(id=0)` (`default_backend() == "gpu"`) and executed both float32 and float64 steps. The unmodified baseline never executes because conversion fails. |
| Differentiate a full control-step prediction w.r.t. joint-target actions, including PD and torque limits | **PARTIAL** | Forward mode (`jax.jacfwd`/`jvp`) works in float64 and matches central finite differences to ~1e-9. Reverse mode (`jax.grad`, `jax.jacrev`, `vjp`) fails for every case: MJX's Newton/CG solver uses `jax.lax.while_loop` with a dynamic stop condition. |
| Match one control-step prediction to the existing Q2 backend | **PASSED in float64 for the diagnostic variant; FAILED in default float32** | Float64 MJX matches the recorded Q2 CPU backend to 3.8e-15 (qpos) / 1.8e-13 (qvel) per step and 2.8e-8 / 3.0e-7 over 100 chained steps. Float32 MJX differs by 3.1e-4 (qpos) / 1.6e-2 (qvel) per step and diverges to 2.9 m / 56 m/s over 100 steps. Against raw MuJoCo Warp (float32), float64 MJX agrees to Warp's own floor: 5.2e-7 / 2.6e-5. |
| Integrate into Q2 | **Not ready as a drop-in backend** | Needs a diagnostic collision-model change, a JAX-only state/PD adapter, float64 for accuracy, forward-mode-only derivatives, no per-lane DR batching, and a large throughput gap to Warp. Details in §6. |

**Important distinction.** The successful execution and gradients above are for
`artifacts/variants/model_env0_diag_nocylbox.mjb`, a separately named
diagnostic variant that removes exactly the 64 unsupported cylinder-box
self-collision pairs via collision bitmasks. It is not a successful conversion
of the baseline model. On the tested rollout the diagnostic model is
bit-identical to the baseline in MuJoCo CPU (see §5.1), but it changes contact
filtering and is not evidence for states where those self-collisions occur.

MJX-JAX was tested here. MuJoCo Warp is a separate engine/path; Q2's Warp
backend was used only as a comparison reference, not as evidence for JAX
differentiation.

## 2. Exact environment

### Machine

- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, x86_64.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB.
- Driver: 580.178.04 (CUDA 13.0 driver).
- CPU/GPU placement: all JAX/MJX arrays and executions on `cuda:0`;
  `jax.default_backend()` returned `gpu`, `jax.devices()` returned
  `[CudaDevice(id=0)]`. No CPU fallback was used.

### Isolated experiment environment

Created under `/tmp/opencode/mjx-go2trot-20261115-a/env` with
`uv 0.13.7`, Python 3.11.18. Never installed into Q2's `.venv`.

| Package | Version | Origin |
|---|---|---|
| Python | 3.11.18 | uv-managed standalone |
| `mujoco` | 3.11.0 | PyPI wheel `mujoco-3.11.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, sha256 `16f94f05…e598d1f1` (same wheel as Q2's lock) |
| `mujoco-mjx` | 3.11.0 | PyPI wheel `mujoco_mjx-3.11.0-py3-none-any.whl`, sha256 `5c3a96a9…49db846` |
| `jax` / `jaxlib` | 0.9.2 | PyPI; `jaxlib-0.9.2-cp311-cp311-manylinux_2_27_x86_64.whl`, sha256 `9ac995b4…39351f` |
| `jax-cuda13-plugin` / `-pjrt` | 0.9.2 | PyPI; x86_64 CPython 3.11 wheels |
| `numpy` | 2.4.6 | PyPI (matches Q2) |
| `scipy` | 1.17.1 | pulled by `mujoco-mjx` |
| `warp-lang` / `mujoco-warp` | 1.16.0 / 3.11.0 | added later only to test coexistence with Q2's GPU stack |
| CUDA runtime wheels | `nvidia-*-cu13` 13.x (e.g. runtime 13.5.37, cuDNN 9.28.1.4, NCCL 2.32.6) | `jax[cuda13]`'s `with-cuda` extra |

`uv pip freeze` is saved in `results/freeze_mjx.txt`.

Q2's reference runs used the existing `BigRedGym/.venv` read-only
(Python 3.11.18, `mujoco 3.11.0`, `torch 2.13.0+cu130`, `numpy 2.4.6`,
`mujoco-warp 3.11.0`, `warp-lang 1.16.0`) with
`PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disabled`. No `uv sync`, no install, and
no lockfile change was made.

### Dependency notes

- `mujoco-mjx 3.11.0` declares `mujoco>=3.11.0.dev0` and unpinned `jax`; we
  pinned `mujoco==3.11.0` to preserve Q2's locked physics version.
- Current newest JAX (0.12.0.1) requires Python ≥3.12. On Q2's Python 3.11,
  JAX 0.9.2 was used; 0.10.0 also declares ≥3.11 but was not tested.
- `jax[cuda13]` pulls newer `nvidia-*-cu13` wheels than Q2 currently has
  (Q2: cuDNN 9.20, NCCL 2.29.7; JAX extra resolved cuDNN 9.28, NCCL 2.32).
  Installing into Q2's existing environment could upgrade those packages and
  affect Torch/Warp. This was not tested in Q2's environment.
- `mujoco-mjx[warp]` pins `warp-lang==1.14.0` (3.11.0), which conflicts with
  Q2's `warp-lang 1.16.0`. The JAX path does not need that extra. We installed
  `warp-lang 1.16.0` + `mujoco-warp 3.11.0` alongside `mujoco-mjx 3.11.0` and
  confirmed `import mujoco.mjx` and `impl="jax"` still work.
- `mujoco-mjx 4.0.1` was tested in a second isolated environment (`env4`,
  MuJoCo 4.0.1). Its collision table still has no `(cylinder, box)` function,
  so upgrading MuJoCo/MJX does not remove the baseline blocker.

## 3. Model provenance

### Source path

The model is produced by Q2's normal Go2Trot construction path:

- `gym/utils/task_registry.py` — `convert_frequencies_to_params()` /
  `set_control_and_sim_dt()`, `make_env()`, backend selection.
- `gym/envs/go2/go2trot_config.py` — task config (PD gains, default angles,
  control frequencies, DR).
- `gym/envs/base/legged_robot_config.py` — MuJoCo options (`njmax=256`,
  `solref=[0.005, 1.0]`, `ccd_iterations=50`, `disableflags=MULTICCD`).
- `gym/envs/base/mujoco_backend_base.py` `_load_model()` — URDF → `MjSpec`
  (`balanceinertia=True`, `fusestatic=False`), free joint, flat ground, solver
  and friction settings, `dof_damping`/`dof_armature`.
- `gym/envs/base/mujoco_cpu_backend.py` — one `mj_step` per control step,
  torques through `qfrc_applied`, state copy.
- `gym/envs/base/domain_randomization.py` — startup friction/mass and episode
  PD gain randomization.
- `gym/envs/go2/go2_config.py` — robot layout (canonical FL, FR, RL, RR),
  rotor inertia, joint damping, torque limits.

Asset: `resources/robots/go2/urdf/go2.urdf`.

### Control interval

`Go2TrotCfg.control.ctrl_frequency = 50`,
`desired_sim_frequency = 50` → `decimation = 1`, `ctrl_dt = 0.02 s`,
`sim_dt = 0.02 s`. Therefore one Q2 control interval is exactly one physics
step **by configuration**, not by assumption. The adapter still emulates the
full Q2 step path; if decimation were >1, it would need to recompute PD and
call `mjx.step` once per substep.

### Model facts (env 0, seed 0)

- `nq=19`, `nv=18`, `nu=0`, `njnt=13`, `nbody=43`, `ngeom=28`.
  Free joint + 12 hinge joints; no MuJoCo actuators (torques via
  `qfrc_applied`).
- `timestep=0.02`, Euler integrator, Newton solver, pyramidal cone,
  `jacobian=AUTO`, `iterations=100`, `ls_iterations=50`, `tolerance=1e-8`,
  `impratio=1`, gravity `[0,0,-9.81]`.
- `solref=[0.005,1.0]` and default `solimp=[0.9,0.95,0.001,0.5,2]` on all
  geoms.
- `disableflags=524288` (`mjDSBL_MULTICCD`), `ccd_iterations=50`.
- `dof_damping=0.01` for joint DOFs; armature
  `[0.002268,0.002268,0.005484] × 4` in native order.
- Ground plane + robot geoms share sliding friction 0.7481283 (startup DR
  sample), `[f, 0.005, 0.0001]`.
- Startup DR samples for this lane:
  - contact friction `0.7481283` (range `[0.5, 1.0]`);
  - 42 link mass/inertia scales in `[0.9, 1.2]` (recorded in
    `artifacts/policy616/meta.json`);
  - episode PD gain scales sampled on reset: `p_gains` in `[18.03, 21.27]`,
    `d_gains` in `[0.402, 0.547]`.
- Torque limits (canonical): hip/thigh `23.7 N·m`, calf `45.43 N·m`.
- Default joint positions: hip `0.0`, thigh `0.96`, calf `-1.36` rad.
- Canonical ↔ native DOF order is the identity for Go2
  (`FL,FR,RL,RR × hip,thigh,calf`). The canonical ↔ native **body** order is a
  non-identity 42-element permutation and is recorded in `meta.json`; it is
  required for contact-force comparison.
- Model file: `artifacts/policy616/model_env0.mjb`, sha256
  `52eaf8c98a57724a4d553aea4ee1c2369e8b59618bc319e96e5257ad1118b7a8`.
  This is env 0's private model copy after startup link-mass/friction DR, built
  with `discard_visual=True` (link-mass DR + headless), i.e. the headless
  training model.

### Rollout artifact

- Policy checkpoint: `BigRedGym/logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/model_616.pt`
  (read-only), sha256
  `07a13a1b1a2d6947e040977afb4b9caa966ac09e48b92aac9e593c6428ae2a61`.
- `model_616.pt` is loaded with the current Go2Trot runner config; actor,
  critic and optimizers load successfully. Deterministic inference
  (`act_inference`) drives a 100-step (2.0 s) Q2 CPU rollout at ~0.72 m/s;
  root height stays 0.294–0.385 m, so it is a real trot, not a fall.
- Commands were fixed to `[1.0, 0, 0]` after reset so the moving state is a
  forward trot. Three extra probes are recorded: zero action from the
  post-reset state ("stand"/free fall to first contact), target 1.0 rad from
  that resulting state, and a fully saturated target (`10.0 rad` on all 12
  joints) from the post-target-1.0 state.

## 4. Reproduction

All scripts are in `/tmp/opencode/mjx-go2trot-20261115-a/scripts`.

### Environment

```bash
EXP=/tmp/opencode/mjx-go2trot-20261115-a
cd "$EXP"
uv venv --python 3.11 env
uv pip install --python env/bin/python \
  "mujoco==3.11.0" "mujoco-mjx==3.11.0" "jax[cuda13]==0.9.2"
```

Optional coexistence test: also install `warp-lang==1.16.0 mujoco-warp==3.11.0`
into the same isolated env; `import mujoco.mjx` and `impl="jax"` still work.

### Checkout copy

```bash
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude 'logs' \
  --exclude 'thirdparty' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '.ruff_cache' --exclude '*.pt' \
  /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/ "$EXP/checkout/"
```

### Q2 reference export (Q2 venv, read-only)

```bash
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disabled \
  /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/.venv/bin/python \
  scripts/export_reference.py \
  --checkout "$EXP/checkout" \
  --out "$EXP/artifacts/policy616" \
  --steps 100 \
  --checkpoint /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113/model_616.pt
```

This writes `model_env0.mjb`, `meta.json`, `rollout.npz` and the extra probe
states (stand, target 1.0, full saturation). `--seed 0` is the default.

### Diagnostic variants and conversion

```bash
env/bin/python scripts/make_variants.py
```

Writes `artifacts/variants/model_env0_diag_nocylbox.mjb` (D1),
`model_env0_diag_noselfcollision.mjb` (D2), and
`artifacts/variants/conversion_report.json`.

### One-step and chained comparison

```bash
env/bin/python scripts/compare_step.py --x64   # float64
env/bin/python scripts/compare_step.py         # float32
```

### Warp reference (Q2 venv, raw Q2 step pipeline)

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/.venv/bin/python \
  scripts/warp_reference.py \
  --model "$EXP/artifacts/variants/model_env0_diag_nocylbox.mjb" \
  --artifacts "$EXP/artifacts/policy616" \
  --out "$EXP/results/warp_d1.json"
```

`compare_warp.py` (float32 and `--x64`) compares MJX against the saved Warp
post-states.

### Derivatives, cost, contact audit

```bash
env/bin/python scripts/derivatives.py          # float64
env/bin/python scripts/derivatives.py --f32    # float32
env/bin/python scripts/cost.py --f32
env/bin/python scripts/cost.py                 # float64
env/bin/python scripts/contact_audit.py
```

The Warp raw throughput reference is `scripts/warp_cost.py`, run with Q2's
venv.

Seeds/determinism: env seed 0; DR generators seeded from `cfg.seed`; policy
inference deterministic; finite-difference probe direction seeded with
`numpy.random.default_rng(20261115)`.

## 5. Results

### 5.1 Conversion

Baseline model:

```
mujoco.mjx.put_model(model_env0.mjb, impl="jax")
NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented.
```

The model has 319 candidate geom pairs from its collision masks; 64 are
cylinder-box pairs, all robot self-collisions:

- `base` box vs each calf cylinder (`*_calf`, `*_calflower`, `*_calflower1`),
- `Head_upper` cylinder vs each thigh box,
- hip cylinder vs opposite/sibling thigh boxes.

There are no cylinder-box pairs involving the ground plane. The QC check with
MuJoCo 4.0.1 / `mujoco-mjx 4.0.1` reports the same missing function.

Diagnostic variants (explicitly named, not baselines):

- **D1 `model_env0_diag_nocylbox.mjb`**: collision bitmasks keep every
  ground–robot pair, every sphere/cylinder/box self pair except cylinder-box.
  255 supported candidate pairs, 0 unsupported. `put_model()` succeeds.
  D1 is bit-identical to the baseline model in MuJoCo CPU over the entire
  100-step rollout and both probes (`cpuD1_vs_cpuBase = 0.0` for qpos and qvel
  at every step). A field-wise comparison of the two `MjModel`s found only
  `geom_contype` and `geom_conaffinity` differ (27 robot geoms; the ground
  plane is unchanged) — no geometry, mass, inertia, gain, limit or solver
  field differs. A direct contact audit over the same rollout found 256 active
  contacts, all plane-sphere (203) or plane-cylinder (53), and **zero**
  cylinder-box contacts. This makes D1 behaviorally identical for the tested
  states, but not in general: a fall that pushes a calf into the base or legs
  into each other could activate a pair that D1 removes.
- **D2 `model_env0_diag_noselfcollision.mjb`**: all robot self-collision
  disabled (27 pairs). Also converts; not used for the numerical results.

### 5.2 GPU execution

`mjx.put_model`/`make_data`/`mjx.step` ran on `cuda:0`: `model.qpos0.device`
and `data.qpos.device` both reported `cuda:0`, `jax.default_backend()` was
`gpu`, and `jax.devices()` was `[CudaDevice(id=0)]`. Float64 and float32 both
executed contacts, and the 100-step chained rollouts produced no non-finite
values. GPU placement is by JAX default; no explicit `device_put` was needed
because JAX's default backend was the GPU.

### 5.3 One-step agreement

Reference definitions:

- **Q2 CPU** = the recorded Q2 backend post-step state (`mj_step` + Q2's
  `rnePostConstraint`/kinematics/com refresh) from the 100-step policy rollout.
- **MuJoCo CPU harness** = a fresh `MjData` at the same qpos/qvel with the
  same torques. This reproduced the recorded Q2 state to 4.4e-15 (qpos) /
  2.2e-13 (qvel), so warm-start history is not a confounder in float64.
- **Warp** = raw `mujoco_warp` using Q2's exact step pipeline (forward, euler,
  rne_postconstraint, kinematics, com_pos, com_vel) on the same MjModel, with
  `njmax=256` and zeroed warm start.
- Quaternion comparison is sign-invariant; the orientation-angle column is
  arccos-limited near identity, so quaternion component errors are the better
  measure there.

One-step worst case over 100 recorded states (identical start state, model,
parameters, and applied torques):

| Case | qpos max | qvel max | root pos | root orientation | root lin vel | root ang vel | contact force |
|---|---:|---:|---:|---:|---:|---:|---:|
| MJX float64 vs Q2 CPU | 3.8e-15 | 1.8e-13 | 5.6e-17 | 8.4e-8 rad* | 4.9e-16 | 2.1e-14 | 8.4e-13 N |
| MJX float32 vs Q2 CPU | 3.1e-4 | 1.6e-2 | 1.8e-6 | 4.7e-4 rad* | 8.9e-5 | 4.6e-3 | 5.5e-2 N |
| Warp float32 vs Q2 CPU | 5.2e-7 | 2.6e-5 | 1.0e-7 | 5.4e-4 rad* | 1.1e-6 | 1.5e-5 | 2.6e-4 N |
| MJX float64 vs Warp f32 | 5.2e-7 | 2.6e-5 | 1.0e-7 | 5.4e-4 rad* | 1.1e-6 | 1.5e-5 | 2.6e-4 N |
| MJX float32 vs Warp f32 | 3.1e-4 | 1.6e-2 | 1.8e-6 | 5.8e-4 rad* | 8.9e-5 | 4.6e-3 | 5.5e-2 N |

\* `2*acos(|q1·q2|)` is limited by the dtype: ~1e-7 rad for float64 and
~1e-4 rad for float32 even for identical quaternions. Quaternion component
differences were ~1e-16 (float64) and ~1e-7 (float32). The angle column is
shown for completeness, not as a precision claim.

Representative per-case errors (MJX vs Q2 CPU, float64):

| State | contacts | qpos | qvel | contact force |
|---|---:|---:|---:|---:|
| `t=0` free flight | 0 | 2.2e-16 | 4.4e-15 | 0 |
| `t=4` contact onset | 12 | 4.4e-16 | 2.1e-14 | 1.1e-13 |
| `t=10` sustained contact | 4 | 4.4e-16 | 2.5e-14 | 5.7e-14 |
| `t=99` | 5 | 6.7e-16 | 2.5e-14 | 8.5e-14 |
| full torque saturation (target 10 rad) | 0 | 4.8e-15 | 2.4e-13 | 0 |

The float32 one-step error is dominated by a spurious joint acceleration in
free flight: with zero qvel and zero torque, MuJoCo CPU gives exactly zero
joint `qacc`, while float32 MJX gives ~1e-2 rad/s² joint accelerations under
gravity. With `jax_enable_x64`, MJX reproduces CPU `qacc` to 1e-15. This is a
precision/conditioning effect (free joint + armature), not a contact or
conversion defect.

### 5.4 Short-rollout divergence

Open-loop 100-step (2.0 s) chained replay from the first recorded state using
the recorded torque sequence:

| Engine | qpos max | qvel max |
|---|---:|---:|
| MJX float64 | 2.8e-8 m | 3.0e-7 m/s |
| MJX float32 | 2.93 m | 56.1 m/s |
| Warp float32 | 0.515 m | 5.57 m/s |
| MuJoCo CPU (D1 / baseline) | 0 | 0 |

The float32 and Warp short-rollout divergences are chaotic amplification of
their one-step error through contact transitions, not an additional per-step
error. One-step error is the meaningful comparison for a CBF/prediction use;
the 100-step number shows that open-loop float32 predictions are not
quantitatively useful after a couple of seconds for this contact-rich gait.
Float64 MJX remains within solver-tolerance of the CPU trajectory.

Warp's reference was not bitwise reproducible across fresh processes: repeated
runs gave one-step qpos 5.2e-7–1.1e-6 and chained 0.48–0.68 m. This is
consistent with float32 contact/atomic ordering; it means Warp itself has a
noise floor larger than the float64 MJX–CPU difference.

### 5.5 Contacts

Q2 exposes `contact_forces` as `cfrc_ext[body, 3:6]` after
`mj_rnePostConstraint`. MJX's `rne_postconstraint` was used for the comparison.
Float64 MJX matches Q2 contact forces to 8.4e-13 N; float32 to 5.5e-2 N.
The contact audit over the 100-step rollout found only plane-sphere (feet) and
plane-cylinder (calf) contacts; no robot self-contacts occurred.

### 5.6 Derivatives

The differentiated function is the full one-step prediction: PD torque from
the pre-step qpos/qvel and the 12 joint targets, per-joint torque clipping,
`qfrc_applied`, then `mjx.step`. The scalar probe is a fixed random dot product
of the 19-element post-step qpos. All results below are float64 unless noted.

| Case | contacts | saturated joints | gradient norm | forward-mode | reverse-mode | FD vs autodiff (best) | repeat |
|---|---:|---|---:|---|---|---:|---:|
| free flight, `t=0` | 0 | none | 0.5165 | works | fails | 1.9e-11 at eps 1e-3 | 0 |
| sustained contact, `t=10` | 4 | none | 0.4432 | works | fails | 1.5e-11 at eps 1e-3 | 6.9e-17 |
| contact onset, `t=4` | 12 | none | 0.3166 | works | fails | 1.9e-11 at eps 1e-3 | 5.6e-17 |
| full saturation | 0 | all 12 | 0.0 (exact) | works | fails | 0 (function constant) | 0 |
| partial saturation | 4 | hips 0,3,6,9 | 0.3274 | works | fails | 6.5e-13 at eps 1e-3 | 1.5e-16 |

Forward-mode gradient vs central finite differences at eps
`1e-2, 1e-3, 1e-4, 1e-5, 1e-6`:

| Case | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---:|---:|---:|---:|---:|
| free flight | 1.8e-9 | 1.9e-11 | 2.6e-12 | 2.3e-11 | 2.5e-10 |
| sustained contact | 1.5e-9 | 1.5e-11 | 2.5e-12 | 2.1e-11 | 1.6e-10 |
| contact onset | 1.9e-9 | 1.9e-11 | 2.6e-12 | 3.0e-11 | 3.3e-10 |
| full saturation | 2.0e-9 | 2.0e-8 | 2.0e-7 | 2.0e-6 | 0 |

For full saturation the true gradient is exactly zero (all torques are
clipped constants); the growing relative FD error at small eps is just
roundoff in a constant function. For partial saturation, the four hip rows are
exactly zero (clipping), the other eight rows are nonzero, and FD at eps 1e-2
shows a 4.7e-3 discrepancy because that perturbation crosses a contact
non-smoothness; at eps ≤1e-3 it agrees to 7e-13.

Directional derivative along a fixed random unit direction `v` (float64):

| Case | autodiff `g·v` | AD−FD, eps 1e-2 | eps 1e-3 | eps 1e-4 | eps 1e-5 |
|---|---:|---:|---:|---:|---:|
| free flight | 0.213505777996 | 5.3e-9 | 5.3e-11 | 5.4e-13 | 5.0e-12 |
| sustained contact | -0.074235260851 | 1.9e-10 | 2.0e-12 | 2.7e-13 | 1.4e-12 |
| contact onset | 0.097260287688 | 2.4e-10 | 2.4e-12 | 3.5e-13 | 9.2e-12 |
| full saturation | 0 | 0 | 0 | 0 | 0 |
| partial saturation | 0.042548643558 | 8.6e-12 | 2.7e-13 | 3.8e-13 | 2.7e-11 |

The directional test is the more meaningful comparison for a CBF constraint:
it projects the full Jacobian onto one direction, and the autodiff value
agrees with finite differences to 1e-9–1e-13 (float64).

Reverse mode fails for every case, including contact-free states, with:

```
ValueError: Reverse-mode differentiation does not work for lax.while_loop or
lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using
fori_loop with static start/stop.
```

Both `mjSOL_NEWTON` (Q2's solver) and `mjSOL_CG` fail the same way. The cause is
the solver's dynamic `while_loop` over iterations/tolerance. Float32 forward
gradients are finite and agree with float64 to 2.2e-8 absolute (<1e-7
relative); float32 finite differences are noisy at small eps (1e-3 at 1e-6),
so float32 autodiff is more trustworthy than float32 FD.

Policy-action scaling: Q2's policy action is converted to the internal target
by `target = action * scales["dof_pos_target"]`, with
`scales["dof_pos_target"] = 4 × [0.2, 0.3, 0.3]`. The Jacobian w.r.t. policy
actions is therefore the internal-target Jacobian multiplied column-wise by
that vector. Both are stored in `results/derivatives_*.json`.

### 5.7 Cost

JAX allocator peak memory; all times after `block_until_ready`. Compilation is
per batch shape and separate from warmed execution. The MJX forward timings
cover `mjx.step` only; the contact-force comparison additionally calls
`mjx.rne_postconstraint`, which is excluded from this table. The Warp raw step
includes its `rne_postconstraint`/kinematics/com refresh, so the timing ratio
is conservative (biased in MJX's favour).

Forward one control step (one physics step for Go2Trot):

| Batch | f32 compile | f32 warm step | f32 steps/s | f64 compile | f64 warm step | f64 steps/s | f32 peak mem | f64 peak mem |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.7 s | 3.53 ms | 283 | 11.6 s | 5.62 ms | 178 | 0.12 GB | 0.12 GB |
| 16 | 15.9 s | 5.72 ms | 2,796 | 13.6 s | 11.5 ms | 1,390 | 0.12 GB | 0.15 GB |
| 128 | 15.6 s | 6.31 ms | 20,275 | 15.4 s | 22.4 ms | 5,704 | 0.15 GB | 0.15 GB |
| 1024 | 21.9 s | 32.3 ms | 31,684 | 20.0 s | 145 ms | 7,072 | 0.48 GB | 0.91 GB |
| 4096 | 22.9 s | 200 ms | 20,529 | 21.5 s | 575 ms | 7,119 | 1.81 GB | 3.61 GB |

Forward-mode gradient of the scalar probe (one gradient per world):

| Batch | f32 compile | f32 warm | f32 gradients/s | f64 compile | f64 warm | f64 gradients/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.3 s | 8.94 ms | 112 | 14.7 s | 12.1 ms | 82.7 |
| 16 | 17.2 s | 7.52 ms | 2,127 | 14.6 s | 23.8 ms | 672 |

Raw MuJoCo Warp (same model, same Q2 step pipeline, `njmax=256`, zeroed warm
start) for the same forward step:

| Batch | first step | warm step | steps/s |
|---:|---:|---:|---:|
| 1 | 0.274 s | 3.53 ms | 283 |
| 128 | 4.6 ms | 3.60 ms | 35,548 |
| 1024 | 6.0 ms | 4.65 ms | 220,179 |
| 4096 | 9.4 ms | 7.75 ms | 528,576 |

At the largest batch, MJX float32 is ~26× slower than raw Warp for the same
step, and float64 MJX is ~74× slower. At batch 1024 the gap is ~7× (f32) /
~31× (f64). The current workspace's measured end-to-end Go2Trot training
throughput is ~176k env-steps/s on Warp at 4096 envs; the raw Warp numbers
above are the engine-only cost, so they bound that end-to-end figure from
above. No batch failed or OOM'd at
4096 on this 32 GB GPU.

### 5.8 Warnings and failures

- Baseline conversion: `NotImplementedError` for cylinder-box, before any
  compile/step. This is a hard blocker, not a warning.
- Optional `warp`/`mujoco_warp` imports print
  `Failed to import warp: No module named 'warp'` when warp is absent. This is
  harmless and disappears when warp is installed. Installing warp 1.16 +
  mujoco-warp 3.11 alongside MJX 3.11 works for the JAX path.
- Large batched compiles print XLA/Triton `xtile_compiler.cc` E-level
  diagnostics (`kCustom`/Triton fusion) for some shapes. The runs completed,
  outputs were finite, and timings were repeatable; the diagnostics were not
  investigated further.
- MJX's public `Data.ncon` is deprecated and reports a static capacity (694),
  not the number of active contacts. Active contacts must be counted from
  `data._impl.contact.dist < 0` (this matches CPU `data.ncon`).
- Warp reference is not bitwise deterministic across processes; see §5.4.
- No non-finite state or gradient was observed in float64. Float32 rollouts
  diverged but remained finite.

## 6. Integration implications

### Reusable unchanged

- Q2's model construction is reusable exactly as-is up to the `MjModel`:
  the `.mjb` binary loads in a separate MuJoCo 3.11/MJX 3.11 process and
  preserves nq/nv/geometry/masses/gains/limits/solver settings.
- For Go2, the canonical-to-native DOF mapping is the identity; the canonical
  body permutation is available from the backend/layout and only matters for
  contact-force routing.
- Q2's PD formula, default positions, p/d gains, torque limits and
  `dof_pos_target` scaling are directly reusable in a small pure-JAX function.
- The state needed for an equivalent one-step prediction is small:
  `qpos[19]`, `qvel[18]`, 12 canonical targets, p/d gains, limits and the
  default positions. Phase, commands and gait references affect the policy's
  observation and reward, not this physics step.

### What a JAX adapter must own

- A pure-JAX PD torque function and `qfrc_applied` assembly in native order
  (identity for Go2, but do not hard-code it).
- One `mjx.step` per control interval; if `decimation > 1`, recompute PD and
  step once per substep, because Q2 recomputes torques every substep.
- Static shapes/batch size; JIT compilation per batch shape. Changing batch
  size recompiles.
- Float64 (`jax_enable_x64`) if quantitative prediction or gradients are
  required. Default float32 is not adequate for this model.
- Contact-force extraction with `mjx.rne_postconstraint` and the canonical body
  permutation.
- Policy-action scaling (`target = action * scale`) if the derivative is
  wanted in policy units.

### Changes / dependency cost

- Add `mujoco-mjx` and JAX (+ CUDA plugin) to the dependency graph, keeping
  `mujoco==3.11.0`. On Python 3.11 the current JAX newest is unavailable;
  JAX 0.9.2 was the tested choice.
- `jax[cuda13]` may upgrade Q2's existing `nvidia-*-cu13` packages; this must
  be validated against Torch 2.13 + Warp 1.16 before adopting.
- `mujoco-mjx[warp]` would pin an older `warp-lang` (1.14.0 for MJX 3.11) and
  must not be installed into Q2's GPU extra; the JAX path does not need it.
- `put_model` cannot be run on the baseline model, so Q2 cannot advertise
  "MJX support" without either (a) a documented collision-model limitation and
  an explicit opt-in diagnostic mask, (b) replacing cylinder/box colliders
  with supported primitives (changes geometry/contact), or (c) an MJX
  extension implementing cylinder-box collision. None preserves the baseline
  contact model automatically.

### Remaining blockers for a Q-derived CBF

1. **Model conversion**: cylinder-box self-collision is unsupported. D1 shows
   the rest of the model is convertible, but D1 is a contact-filtering change,
   not the configured model.
2. **Reverse-mode gradients**: a scalar forward-mode Jacobian works and is
   enough for a 12-action CBF (12 JVPs), but any use that requires a VJP
   through the simulator (e.g. backpropagating a Q loss through contact
   dynamics) is blocked by the solver's dynamic `while_loop`. Reverse mode
   would need a fixed-iteration solver (`lax.scan`) or a custom VJP, neither
   of which is in stock MJX 3.11.
3. **Accuracy/cost**: float64 is required for prediction accuracy; it is
   1.6–2.9× slower than float32 and ~74× slower than raw Warp at batch 4096.
   Float32 one-step errors are small in joint position but 1.6e-2 in joint
   velocity and 5.5e-2 N in contact force; float32 open-loop rollouts diverge.
4. **Per-env domain randomization**: the experiment used one lane's model
   (seed 0) with its friction/mass/PD samples. MJX models are static pytrees;
   per-env mass/friction would need a batched-model wrapper or one model per
   lane, with recompilation. This is untested and would be a design change.
5. **Contact nonsmoothness**: gradients are local through the active contact
   set. A CBF that must reason about contact making/breaking needs either
   nonsmooth analysis or a smooth contact model; MJX autodiff does not remove
   that issue.

## 7. Evidence limits

- One environment lane (env 0 of a one-env construction), seed 0, with one
  sampled DR realization. Other lanes have different masses/friction/gains.
- One trained policy artifact and a 100-step (2 s) rollout. No policy was
  trained; no long-horizon or multi-episode statistics.
- The comparison uses the Q2 CPU backend as the float64 ground truth. Warp is
  a second reference and is itself float32/non-bitwise-deterministic.
- The numerical agreement and gradients are for D1, the diagnostic
  no-cylinder-box-self-collision model. Baseline conversion never executed.
  The 100-step rollout had no cylinder-box contacts, so D1 is bit-identical
  there; this does not prove equivalence for falling/self-contact states.
- No VSim, Go2 hardware, training-quality, or full RL smoke test was run.
- Float32 long-rollout divergence is chaotic amplification; it is not a
  per-step precision claim.
- MJX 4.0.1 was checked only for the missing collision function; no full
  upgrade experiment was run.
- Cost numbers are single-process measurements on this GPU, with the JAX
  allocator's peak memory (not process RSS), and include XLA/Triton
  compilation diagnostics.

## 8. Cleanup

The experiment directory is self-contained:

```bash
rm -rf /tmp/opencode/mjx-go2trot-20261115-a
```

It contains the checkout copy, isolated `env/` and alternative `env4/`,
artifacts, results and this report. It is not referenced by Q2.

The original workspace was not modified:

- `BigRedGym` remained at `77488a0b98a28bed1bb58cf5a2862503f14696ae`,
  worktree clean, `.venv` and `uv.lock` untouched.
- No files were installed into Q2's environment. Reference runs used
  `PYTHONDONTWRITEBYTECODE=1` and the existing interpreter read-only.
- `.opencode` had pre-existing uncommitted changes; this experiment did not
  modify them. No commit was made anywhere.
