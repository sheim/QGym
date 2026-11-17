# GPU QP, SQP and NLP for a discrete-time Q-derived CBF — investigation report

**Status:** investigation complete, 2026-11-15. Disposable feasibility work; no
product source, config, lockfile, `.venv` or artifact was modified.

**Question.** For one Q2 `go2trot` environment with 12 policy-unit joint-target
actions, can the constrained solve in

```text
h(s) = Q_r'^pi(obs(s), pi(obs(s))) - c
g(u) = h(F(s, u)) - (1 - beta) h(s)
minimise 0.5 (u - u_nom)^T W (u - u_nom)   s.t.  g(u) >= 0,  u_min <= u <= u_max
```

run **on the GPU** — as a locally linearised QP, as SQP over repeated QPs, or as
a general constrained NLP — against a real MJX one-step predictor and the real
fitted height-Q?

**Short answer.** Yes for the QP layer: two released GPU QP solvers (QPAX and
cuPIQP) solve the 12-variable/37-row problem on the device in 1.2–1.4 ms
(f64, 7 iterations), and batching 256 problems costs 5.2–5.6 µs each. SQP on top
of that QP is feasible and reaches 12–45 ms per SQP iteration with no host
decisions inside the loop;
a full 8-iteration filter decision costs 73–357 ms depending on how many inner
QP iterations the data needs, i.e. 4–18× slower than Q2's 20 ms control
interval. The hard limits are elsewhere:

1. **The local QP does not certify the nonlinear constraint.** Fourteen of the
   18 solved linearised-QP proposals measured here violated the original `g(u)`
   by 0.0006 to 2.3 units, although every one satisfied its own linearisation;
   the four that satisfied it started from an already-nearly-satisfied point or
   overshot (+2.00 from -9.61). In the known-solution projection test the
   violation was 2.28 — the QP lands on the tangent plane, outside the ball.
   The original constraint must be re-evaluated and the proposal
   rejected/refined: a solved local QP is not evidence.
2. **The barrier is mostly active, and hard to satisfy in one step.** With
   `beta = 0.5`, `c` = median of `h`: the nominal policy already satisfies
   `g(u) >= 0` in 46/100 rollout states, and the worst violation is -0.41
   Q-units. With `beta = 1`, `c` = 90th percentile (a one-step "get back into
   the set" demand) only 10/100 states pass and the worst violation is -9.61.
   SQP (validated against the analytic solution of a projection test, where it
   converges to 7e-4 in 4–7 iterations) reduces these violations by 1–2 orders
   of magnitude and reaches feasibility in one of nine real states; it does not
   certify `g >= 0` in 8 iterations. Q is a discounted return, so one 0.02 s
   step can move it by about one discounted reward — that is the structural
   reason, not an arithmetic one.
3. **A released GPU NLP exists but is not a controller.** `jaxipm` (GPU-batched
   IPOPT) solves the same 12-variable nonlinear problem correctly, but needs
   Python >= 3.12 (Q2 pins `== 3.11.*`), derives derivatives symbolically from
   the callables (not from supplied forward-mode Jacobians), and takes 7.2–7.5 s
   per single solve. It is a throughput design, not a latency design.
4. **The nominal objective works against feasibility.** Minimising
   `0.5 ||u - u_nom||^2` under the linearised constraint lowered the true `g`
   relative to the nominal in 2 of 8 states (`beta = 0.5`: standing probe
   -0.280 -> -0.370, full saturation +3.411 -> +1.122) even though the
   linearised constraint was satisfied in every case: the QP moves along the
   objective's descent direction and the linearisation is not conservative.

Temporary experiment directory (keep until reviewed):
`/tmp/opencode/gpu-qp-20261115-a/`. Permanent reproductions: [`scripts/`](scripts/),
[`scripts/README.md`](scripts/README.md).

---

## 1. Verdict matrix

| Candidate | Version / licence | Installed & runnable with Q2 pins | GPU numerical solve | Supplied (forward) derivatives | Original-constraint check | Failure handling | Zero host transfers | Real MJX + real-Q integration |
|---|---|---|---|---|---|---|---|---|
| **QPAX** (QP, PDIP) | 0.1.4, Apache-2.0, JAX | **PASS** (isolated; needs JAX 0.9.2 on py3.11) | **PASS** (cuda:0, f32/f64) | **PASS** (takes QP matrices, not model callbacks) | n/a (QP only) | **PARTIAL** — `converged` flag + iteration count; hard-infeasible QP returns **NaN with no status**; elastic variant returns per-row slack | **FAIL (measured)** — elastic QR path: 137 one-byte DtoH reads + 137 stream syncs per decision (invariant to inner iteration count) | **PASS** as the inner QP of SQP; 12–45 ms per device-side SQP iteration |
| **JAXopt BoxOSQP** (QP, ADMM) | 0.8.5, Apache-2.0, JAX | **PARTIAL** — f64 works; **f32 broken** with JAX 0.9.2 (`lax.triangular_solve` dtype error, `cond` branch dtype mismatch) | **PASS** (f64) | **PASS** | n/a | **PARTIAL** — returns a mid-point at `max_iter` with status 0 on infeasible input (0.5 violation, 20 000 iterations); the raising path exists only with `jit=False` (source, not exercised here) | untested (device `while_loop` with `jit=True`); 228 ms per solve dominates | **FAIL as a controller primitive** (46–81 iters, 228 ms, 200x QPAX) |
| **cuPIQP** (QP, proximal IPM) | 0.1.0, BSD-2, Warp-native | **FAIL as-is** — requires `warp-lang>=1.17` (Q2 locks 1.16.0) and adds `nvmath-python`, `cuda-bindings`, `cuda-core`, `socu`; tested only in its own environment | **PASS** (cuda:0, f32/f64, batched, GPU-only inputs) | **PASS** (QP matrices; also accepts JAX/Warp/torch arrays zero-copy) | n/a | **PASS** — explicit `CUPIQP_PRIMAL_INFEASIBLE` / `MAX_ITER_REACHED` / `NUMERICAL_ISSUES` status enum | **PASS (by design, partially measured)** — `solve()` queues and returns in 0.037 ms; all iterations device-side with a CUDA-graph option | **UNTESTED with MJX** (warp-version conflict; no MJX path) |
| **jaxipm** (general NLP, GPU IPOPT) | 0.0.1, arXiv 2606.26341, Python >= 3.12 | **FAIL** — Python >= 3.12 vs Q2 `== 3.11.*`; needs cuDSS (`nvidia-cudss-cu13`) | **PASS** (cuda:0; 3.7e-10 from the analytic solution) | **NO** — derivatives come from `jax2sympy` symbolic differentiation of the callables, with `override_sparse_funcs` hooks; not a "give me the Jacobian" API | n/a (NLP) | device-side `while_loop` with termination codes | not measured for residency | **REJECTED** for this model: would have to differentiate MJX symbolically; 7.2–7.5 s per single solve |
| **NVIDIA cuOpt** (QP) | 26.10.1 (`cuopt-cu13`), proprietary EULA | **UNTESTED** — installable (pulls the RAPIDS stack: cudf, cupy, numba, raft), not installed | documented GPU barrier/PDLP QP (LP+QP GA in 26.10) | n/a | n/a | documented | documented asynchronous | **N/A** — no nonlinear inequalities; QP only |
| **MadNLP + MadNLPGPU / MadIPM** (general NLP) | MIT, Julia | **UNTESTED** — Julia runtime, no Python-native path | documented fully GPU NLP (IPOPT-line search, cuDSS/cuSOLVER KKT) | n/a | n/a | documented | documented | **N/A** in this budget; would need a Julia sidecar and model re-declaration (ExaModels) |
| CPU-only QP/NLP (OSQP, Clarabel, PIQP, ProxQP, Ipopt, SLSQP) | — | runnable | **FAIL by definition** | — | — | — | — | used only as offline references (scipy SLSQP) |

Interpretation of the "zero host transfers" column: no candidate achieved an
unqualified pass. QPAX's *physics* boundary is clean (MJX's solver loop is
device-side), but 137 one-byte device-to-host reads and 137 stream
synchronisations per controller decision were measured on the QP path
(section 8). Those are real transfers, not bookkeeping artifacts.

---

## 2. What was actually implemented

* `h(s)` uses the **artifact's own** deterministic policy mean
  (`actor.act_inference`) and the Q fitted for that same actor, with the Q input
  normalisation from the artifact manifest and **clean** (noise-free)
  observations. This is a mean-action approximation, not the expectation of the
  stochastic actor.
* The Q consumes **only** the 48-D actor observation and the action; no
  privileged physics quantity (contact force, foot position, true root height)
  is ever passed to it, and `h` is therefore a function of the same information
  the deployed actor has.
* `F(s, u)` is one complete Q2 control interval: PD torque
  `p*(u*scale + default - q) - d*qdot`, per-joint torque clipping, one
  `mjx.step` (decimation is 1 for `go2trot`, so one substep), then the successor
  48-D observation including the previous-action channel, the advanced gait
  phase (`phase += 2 pi * 0.02 * phase_frequency`), commands and normalised
  joint state.
* `W = I` in policy units, action bounds `[-2, +2]` policy units (the clamp the
  earlier MPPI study used), and an explicit L-infinity trust region of 0.25
  policy units per SQP iteration.
* Slack is explicit: the SQP inner QP is the **elastic** QPAX QP with a linear
  penalty of 10 per unit of slack per row, and the reported slack is measured
  against the original rows.
* Threshold `c` is the **median** (or 90th percentile) of `Q(obs_t, pi(obs_t))`
  over the recorded rollout. It is a chosen threshold, not a calibrated safe
  set; the Q-defined set is an experimental return superlevel set, not a
  certified barrier, and no forward-invariance claim is made.
* Terminal behaviour: `go2trot`'s `step()` only raises flags, so predictions are
  reset-free. This investigation evaluates only non-terminal states; the
  artifact's absorbing-failure value is **not** injected into `h`, and that is a
  known simplification, not an equivalence claim.

---

## 3. Environment, versions, provenance

### Machine

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB, sm_120 |
| driver | 580.178.04 (CUDA 13.0) |
| other GPU load | 647 MiB used by desktop processes before/after every run; no foreign compute process |

### Product pins (unchanged, used read-only)

product revision `5c7a7930bdf22636776b3745c4cbb121e265dc56` (branch
`bh574/final/dev/offline_RL`), Python 3.11.18, torch 2.13.0+cu130, mujoco 3.11.0,
mujoco-warp 3.11.0, warp-lang 1.16.0, numpy 2.4.6, tensordict 0.13.0.

### Isolated environments (never Q2's `.venv`)

| env | purpose | contents |
|---|---|---|
| `env/` | all JAX numerics | the earlier MJX freeze (jax/jaxlib/jax-cuda13-plugin 0.9.2, mujoco 3.11.0, mujoco-mjx 3.11.0, numpy 2.4.6, scipy 1.17.1) + `qpax 0.1.4` + `jaxopt 0.8.5` |
| `env_coexist/` | same-process dependency check | the locked Q2 GPU stack (torch 2.13.0+cu130, mujoco, mujoco-warp, warp-lang 1.16.0, tensordict) **plus** the JAX stack above |
| `env_cupiqp/` | cuPIQP | `cupiqp 0.1.0`, `warp-lang 1.19.0`, `nvmath-python 1.0.0`, `cuda-bindings 13.5.0`, `cuda-core 1.3.0`, `socu 0.1.0`, `nvtx 0.2.17` |
| `env_jaxipm/` | jaxipm | Python **3.12.16**, `jaxipm 0.0.1`, jax 0.12.0.1 (+cuda13), `spineax 0.0.5`, `jax2sympy`, `sympy`, plus `nvidia-cudss-cu13` and a local `libcudss.so.0 -> libcudss.so.1` symlink (spineax loads the `0` soname) |

### Dependency compatibility (section 2 of the brief)

Installing the JAX CUDA13 stack next to the locked Q2 GPU stack **upgrades 12
`nvidia-*` wheels**; resolution alone is not enough, so this was tested by
running both frameworks in one process:

| package | Q2 lock | after adding JAX cuda13 |
|---|---|---|
| nvidia-cublas | 13.1.1.3 | 13.10.0.20 |
| nvidia-cudnn-cu13 | 9.20.0.48 | 9.28.1.4 |
| nvidia-nccl-cu13 | 2.29.7 | 2.32.6 |
| nvidia-cusolver | 12.0.4.66 | 12.4.1.2 |
| nvidia-cusparse | 12.6.3.3 | 12.9.4.2 |
| nvidia-cuda-runtime / nvrtc / cupti / nvjitlink / nvshmem | 13.0.x / 13.3.x / 3.4.5 | 13.5.37 / 13.5.37 / 13.5.25 / 13.5.37 / 3.8.0 |
| (also) absl-py, etils, fsspec, glfw, zipp | 2.5.0 / 1.14.0 / 2026.7.0 / 2.10.2 / 4.1.0 | 2.5.1 / 1.15.0 / 2026.9.0 / 2.11.0 / 4.1.2 |

Same-process checks in `env_coexist` (`scripts/coexistence.py`) — **9/9 pass**:

| check | result |
|---|---|
| torch CUDA matmul + conv2d | ok, `cuda:0` |
| JAX compute | ok, backend `gpu`, `CudaDevice(id=0)` |
| MJX `mjx.step` **after** torch allocations | ok (nq=19, device `cuda:0`) |
| torch after JAX/MJX | ok, finite, 6.77 GB free |
| Warp 1.16 kernel launch | ok, `mean=3.0` |
| `import mujoco_warp` | ok |
| QPAX solve after torch+Warp | ok, 5 iterations |
| `jaxopt` import | ok |
| `tensordict` import | ok (0.13.0) |

`mujoco-mjx[warp]` was not used (it pins `warp-lang==1.14.0`); the JAX path does
not need it. cuPIQP cannot currently share Q2's environment (warp 1.17+).

Freezes: `results/coexist_freeze_torch.txt`, `results/coexist_freeze_all.txt`,
`/tmp/opencode/mjx-go2trot-20261115-a/results/freeze_mjx.txt`.

---

## 4. Model and artifact provenance

| item | identity |
|---|---|
| MJX model | `/tmp/opencode/mjx-go2trot-20261115-a/artifacts/variants/model_env0_diag_nocylbox.mjb`, sha256 `f55cf10b…c19b30b` |
| baseline model it was derived from | `…/artifacts/policy616/model_env0.mjb`, sha256 `52eaf8c9…1118b7a8`; the model regenerated by this investigation's own Q2 CPU rollout is **byte-identical** to it (`52eaf8c9…`) |
| policy | `BigRedGym/logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt`, sha256 `f88b1f9b…3adf6` |
| Q | `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct` (format `brg-height-q-v1`, schema 1), `q.pt` sha256 `e59d738a…d2236`, manifest sha256 `db267188…90e0e9` |
| Q/policy compatibility | artifact contract fingerprint `6412a878…9938cb` (validated by the earlier MPPI study for this exact pair) |
| reference rollout | 100 Q2 CPU control steps at 50 Hz, env 0, seed 0, commands `[1,0,0]`, produced by `scripts/torch_reference.py` with the product's own interpreter; `artifacts/nov10/torch_ref.npz` |
| model modification | the MJX model is the **diagnostic** variant that removes exactly the 64 unsupported (cylinder, box) self-collision pairs. It is not the baseline contact model; the earlier MJX report showed it is bit-identical on this rollout (all contacts are plane–sphere/plane–cylinder) but not in general |

Observation convention: 48 = 3 `base_ang_vel` + 3 `projected_gravity` +
3 `commands` + 12 `dof_pos_obs` + 12 `dof_vel` + 12 `dof_pos_target` + 2
`phase_obs` + 1 `phase_frequency`, divided by the scales recorded in the Q
artifact manifest. Two MuJoCo/Torch details had to be matched exactly and were
caught by the parity gate: the free-joint quaternion is `(w,x,y,z)` in `qpos`
and `(x,y,z,w)` in Q2's `root_states`, and `base_ang_vel` is the body-local
`qvel[3:6]` (Q2 rotates the world-frame `root_states` angular velocity back).

---

## 5. QP layer: numerical behaviour (12 variables)

Battery: one linearised barrier row plus the 24 box rows (37 rows for the
multi-constraint case). References are closed-form for the single-half-space and
bound-active cases (`u* = u_nom + lambda W^-1 a`), `clip(u_nom)` for the bound
case, and scipy SLSQP (offline validation only) for the 8-row case.

| case | reference | QPAX f64 iters / flag | QPAX f64 dual res | original viol. | ref err | BoxOSQP f64 iters | cuPIQP f64 iters / status | cuPIQP f64 ref err |
|---|---|---|---|---|---|---|---|---|
| inactive barrier | analytic | 5 / 1 | 2.2e-7 | 0 | 1.9e-7 | 36 | 5 / SOLVED | 1e-10 |
| active barrier | analytic | 5 / 1 | 4.4e-7 | 0 | 1.2e-6 | 46 | 6 / SOLVED | 8e-10 |
| bounds active | clip | 6 / 1 | 1.7e-8 | 0 | 6.2e-6 | 71 | 6 / SOLVED | 8.9e-9 |
| infeasible | none exists | 60 / 0, **x = NaN** | NaN | NaN | — | 20 000 / status 0, viol 0.500 | 22 / **PRIMAL_INFEASIBLE** | — |
| infeasible, elastic | best compromise | 7 / 1, **slack_max 0.984** | — | 0.984 (reported as slack) | — | — | — | — |
| poor scaling (cond 1e12) | analytic | 60 / 0 | 1.7e-11 | 0 | **0.0** (exact) | 20 000 / status 0 | 19 / SOLVED (gap 2e-4) | **11.2** (obj 3x worse) |
| nearly-zero gradient | analytic | 4 / 1 | 8.7e-15 | 0 | 0.0 | 1 | 3 / SOLVED | 0.0 |
| 8 rows + bounds | scipy SLSQP | 7 / 1 | 3.2e-10 | 0 | 1.8e-6 | 81 | 7 / SOLVED | 6.8e-9 |
| iteration limit 2 | — | 2 / 0 | 4.5e-3 | 0.25 | 0.25 | — | 3 / MAX_ITER_REACHED | 0.045 |

Additional observations:

* **`converged` is a solver-internal KKT norm, not the answer quality.** With
  `solver_tol=1e-9` QPAX reports "not converged" after 60 iterations while the
  primal-dual residual is already 1.8e-10 and the solution is exact. The default
  `solver_tol=1e-5` is the usable setting.
* **float32 does not meet the default QPAX tolerance.** In f32 the flag stayed 0
  (30–60 iterations) with residuals ~1.9e-4 and solutions accurate to 1e-4..3e-4.
  f32 is fine for the QP; it is the *flag* that needs a relaxed tolerance.
* **Hard infeasibility is indistinguishable from divergence in QPAX.** The plain
  solve returned non-finite `x`; only the elastic variant made infeasibility
  visible, as slack. cuPIQP was the only candidate with an explicit infeasible
  status.
* **Ill-conditioning is the shared failure mode.** At 1e12 weight spread QPAX
  f64 still returned the exact solution (objective gap 0), BoxOSQP ended 3.2–4.0
  away (it varies between repeats: ADMM at `max_iter`), cuPIQP reported SOLVED
  with a 3x worse objective, and cuPIQP f32 was 6571 away — which is why
  scaling and primal-dual residuals must be checked by the caller, not inferred
  from `status`.

### Timing (one QP, 12 vars / 37 rows, warmed, `block_until_ready`)

| solver | dtype | compile | warm solve | iterations |
|---|---|---|---|---|
| QPAX (tol 1e-5) | f64 | 281 ms | **1.16 ms** | 7 |
| QPAX (tol 1e-9) | f64 | 288 ms | 8.84 ms | 60 (max) |
| QPAX (tol 1e-5) | f32 | 361 ms | 2.44 ms | 30 (max, flag 0) |
| BoxOSQP | f64 | 791 ms | 227.8 ms | 46 |
| cuPIQP | f64 | ~2.5–3 s first kernel load | **1.38 ms** sync / **0.037 ms** queued | 7 |
| cuPIQP | f32 | idem | 0.585 ms | 5 |

### Batching (256 independent QPs, analytic references)

| solver | dtype | total | per solve | all converged | max ref err |
|---|---|---|---|---|---|
| QPAX `vmap` | f64 | 1.44 ms | 5.64 µs | 256/256 | 5.9e-4 |
| QPAX `vmap` | f32 | 5.65 ms | 22.1 µs | 0/256 (flag) | 2.8e-4 |
| cuPIQP native batch | f64 | 1.33 ms | 5.19 µs | 256/256 SOLVED | 1.0e-6 |
| cuPIQP native batch | f32 | 0.61 ms | 2.39 µs | 256/256 SOLVED | 2.0e-2 |

Batching support is a future-capability statement: the controller scope here is
one evaluation environment.

---

## 6. Nonlinear layer: linearised QP vs SQP

### 6.1 Synthetic known-solution case (projection onto a ball)

`min 0.5||u-u_nom||^2 s.t. ||u||^2 <= r^2`, `||u_nom|| = 8.40`, `r = 6.72`,
analytic solution `u* = r u_nom / ||u_nom||`:

| strategy | linearised residual | **true** `g` | distance to `u*` | iterations |
|---|---|---|---|---|
| nominal | -25.386 | -25.386 | 1.679 | — |
| **one linearised QP** | **+0.0005 (satisfied)** | **-2.284 (violated)** | 0.168 | 7, converged, zero slack |
| SQP, trust 0.25 | — | **+2.5e-5** | **7.2e-4** | 7 |
| SQP, trust 1.0 | — | +1.9e-5 | 7.8e-4 | 4 |

Two results matter here. The single-QP row is exactly the case the brief asks
about: a converged, slack-free local QP satisfies its own linearisation to 5e-4
while the true quadratic constraint is violated by 2.28, because the solution
lies on the tangent plane outside the ball. That is a model error, not a solver
error, and only re-evaluating `g` catches it. The SQP rows show the machinery
(elastic QP + true-`g` backtracking + trust region) converging to the analytic
optimum to 7e-4 in 4–7 iterations, so the QP/acceptance loop itself is sound;
where it does not converge (the real states in §6.2) the one-step requirement is
the plausible cause, not the arithmetic.

### 6.2 Real robot states (MJX predictor + real Q)

Two regimes, same states, `h` sampled over the 100-step rollout
(min -3.94, max -0.82, median -1.07, mean -1.13, std 0.37 in raw Q units).

**Strict regime:** `c` = 90th percentile (-0.970), `beta = 1.0`, i.e.
`g(u) = h(F(s,u))` — "successor inside the set". Nominal satisfies `g >= 0` in
**10/100** states (range -1.942 … +0.150).

| state | contacts | nominal `g` | linearised-QP true `g` | SQP 8-iter true `g` | SQP status |
|---|---|---|---|---|---|
| `t=0` free flight (post-reset) | 0 | -1.942 | **-0.791** | -0.041 | 1 (iteration limit) |
| `t=1` first contact | 4 | -1.340 | **-0.815** | -0.279 | 1 |
| `t=4` contact onset | 12 | -0.056 | **-0.010** | -0.003 | 1 |
| `t=10` sustained contact | 4 | -0.027 | **-0.0006** | **+0.00006** | 0 (feasible, 3 iters) |
| `t=28` | 4 | -0.247 | **-0.041** | -0.006 | 1 |
| `t=99` late rollout | 5 | -0.136 | **-0.046** | -0.032 | 1 |
| standing probe (zero action) | 4 | -1.698 | **-1.232** | -0.596 | 1 |
| target 1.0 rad probe | 4 | -1.729 | **-1.139** | -0.625 | 1 |
| full saturation (target 10 rad) | 0 | -9.614 | **+2.000** | -0.152 | 1 |

**Moderate regime:** `c` = median (-1.068), `beta = 0.5` (halve the violation
at most per step). Nominal satisfies `g >= 0` in **46/100** states (range -0.409
… +0.484).

| state | nominal `g` | linearised-QP true `g` | SQP true `g` |
|---|---|---|---|
| `t=0` | -0.409 | -0.222 | -0.035 |
| `t=4` | -0.055 | -0.010 | -0.003 |
| `t=10` | +0.038 | +0.038 | +0.038 |
| `t=28` | -0.158 | -0.006 | -0.0001 |
| `t=99` | -0.026 | +0.012 | +0.0004 |
| standing probe | -0.280 | **-0.370 (worse)** | -0.102 |
| target 1.0 probe | -0.358 | -0.321 | -0.158 |
| full saturation | +3.411 | **+1.122 (worse)** | +3.411 |

Reading: the linearisation is not conservative in either direction — it
overshoots to a "safe" +2.00 where the truth is -9.61 at full saturation, and it
degrades a satisfied constraint (-0.280 -> -0.370 for the standing probe).
SQP reduces the violation by 1–2 orders of magnitude and reaches feasibility in
one case, but the residual after 8 iterations is still negative in eight of
nine states; that is a statement about the one-step requirement and the
acceptance rule, not about arithmetic. Both regimes are reported because they
say different things: `beta = 0.5` gives a mostly-satisfiable constraint with
violations of order 0.1, `beta = 1` turns it into a one-step reachability
problem that this Q cannot support from a violation of 9.6.

### 6.3 Cost of SQP

| configuration | per decision | note |
|---|---|---|
| one composite evaluation `g(u)` | **4.0 ms** | warm, f64, 1 GPU step + obs + policy + Q |
| one `g` + forward Jacobian | **10.7 ms** | `jacfwd` = 12 JVPs through the MJX step |
| one device-side SQP iteration (value+jac, elastic QP, 6 backtracks) | **~12–40 ms** | inner QP iterations are data-dependent |
| 8-iteration decision, fully device-side | **98.8 ms** (`t=0`), 97.9 ms (`t=1`), 356.9 ms (`t=4`), 272.2 ms (`t=28`), 227.1 ms (`t=50`) | status 1: iteration limit |
| feasible decision, 2 iterations, device-side | **72.9 ms** (`t=10`) | status 0 |
| same loops driven from Python | **0.36–3.6 s** per decision | 5–25× slower: per-iteration host read of `g` and `jac` (DtoH + sync); the first state also pays 18 s of compilation |

Median over the six profiled states and 10 repeats each: **162.8 ms**, minimum
72.6 ms. The Python-controlled loop is retained in
`scripts/sqp_controller.py` because it is what produces the per-iteration logs
above; the device-side loop (`scripts/device_sqp.py`) is the one whose
residency was profiled.

---

## 7. Real MJX + real Q integration

| gate | result |
|---|---|
| one-step physics parity, JAX float64 vs Q2 CPU rollout (100 states) | qpos **1.5e-7**, qvel **7.5e-6** (the Q2 record is float32, so this is its floor) |
| 48-D observation parity vs Q2's own observation code | **9.4e-7** max abs |
| policy-mean action parity (JAX vs Torch `act_inference`) | **5.3e-7** max abs; same vs the recorded rollout actions |
| Q parity (JAX vs `HeightQ.predict`, including obs normalisation) | **4.4e-6** max abs |
| **end-to-end predictor**: `h(F(s_t, pi(s_t)))` vs recorded `h(s_{t+1})`, 99 steps | **1.16e-7** max abs, 2.5e-8 rms |
| reconstructed `h(s_t)` from `(qpos, qvel, prev target, phase)` vs `Q` on Q2's recorded 48-D observation | **4.8e-6** max abs |
| forward-mode gradient of the full composite vs central differences (eps 1e-6) | 5.9e-9 abs, 9.1e-8 rel |
| directional derivative `jac^T v` vs finite difference | 3.0e-9 |
| reverse mode `jax.grad` through MJX | unavailable (previous report: `lax.while_loop` with dynamic stop) |
| second-order `jax.hessian` of the composite | **fails**: `TypeError: unsupported operand type(s) for +: 'NoneType' and 'tuple'` inside `solver.solve`'s `while_loop` |

Consequences: the derivative the controller needs (one 12-vector Jacobian per
iteration) exists and is accurate; anything requiring a VJP or an exact Hessian
of the Lagrangian does not. SQP therefore uses the exact objective Hessian
(`W`) with a zero constraint-curvature term (a Gauss-Newton/least-squares
Lagrangian approximation) — a standard variant, and it must be labelled as
such.

`h` over the rollout: min -3.94, max -0.82, median -1.07, mean -1.13,
std 0.37 (raw `Q` values). States inside and outside the chosen superlevel set
are both in the sample (`h0 = -2.87` at `t=0`, `+0.03` at `t=10` with the median
threshold; `h0 = -2.97` / `+0.89` with the 90th-percentile threshold).

---

## 8. GPU execution audit

Warmed boundary, 5 decisions of the `t=0` state (8 SQP iterations each), Nsight
Systems 2026.1.3, `-c cudaProfilerApi`, trace
`traces/device-sqp.nsys-rep` (temporary):

| metric | value | per decision |
|---|---|---|
| total wall (device-side loop) | — | **98.8 ms** |
| GPU kernel time (window) | **90.3 ms** (15 130 kernels) | 18.1 ms (**18 %**) |
| `cuGraphLaunch` | 8 005 | 1 601 |
| `cudaLaunchKernel` + `cuLaunchKernel` | 9 640 + 5 490 | ~3 000 |
| device→host copies | 685 copies, **685 bytes total**, every one **1 byte** | 137 copies, 137 B |
| host→device copies | 0 | 0 |
| `cuStreamSynchronize` | 685 (395 ms cumulative) | 137 |
| device→device copies | 3 600, 1.42 MB | 720, 284 KB |
| largest kernels | cuSOLVER `getrf_wo_pivot` 46.7 µs x 840, `geqr2_smem` 53.3 µs x 320, `trsv_*` | — |

Attribution: the kernel mix and the 1-byte reads both point at cuSOLVER's dense
factorisations (QPAX's elastic solver hard-codes a QR/LU backend), not at the
physics. The control run on a feasible state (2 SQP iterations instead of 8,
3 decisions) gave **the same 137 reads and 137 syncs per decision**
(410/410 in total) with 4 835 graph launches per decision — so the count is
**not** proportional to the number of inner QP iterations, and exact per-call
attribution was not established. What is established: the physics boundary is
device-side, at least one readback per decision is not, and the payload (action,
residuals, status) crosses once at the end. Strict "zero host transfers" is
therefore **unverified**, and the honest description is "device-resident
optimisation with a small, non-payload readback path".

Memory:

* JAX's own allocation for the composite peaked at **19 MB** (`bytes_in_use`
  1.6 MB) — the model, weights and step buffers are small.
* With default settings the XLA pool grew to its **25.2 GB limit** as soon as
  real work started (`peak_pool_bytes == bytes_limit`), which shows up as ~25 GB
  in `nvidia-smi` while the process runs. With
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` the same work kept the pool at **149 MB**.
  Any controller sharing the card with a training run must set one of these.
* No configuration OOM'd; the largest QP batch tried was 256 and the largest
  device-SQP decision 8 iterations.

---

## 9. Smallest replaceable solver interface

Three layers, kept separate because they consume genuinely different inputs.

### 9.1 Model/barrier evaluation and supplied derivatives

```python
class BarrierModel:
    """Owns the predictor and the artifact; the only object that touches MJX."""

    @classmethod
    def build(cls, model_path, policy, q, scales, *, beta, threshold_c):
        ...

    def h(self, state) -> Float:                  # h(s) = Q(obs(s), pi(obs(s))) - c
        ...

    def g(self, u, state) -> Float:               # h(F(s,u)) - (1-beta) h(s)
        ...

    def g_and_jac(self, u, state) -> tuple[Float, Array[12]]:
        """Value and dg/du. Forward mode only (12 JVPs); no VJP, no Hessian."""

    def step(self, u, state) -> State:            # the F(s,u) transition, for rollout
        ...

    def terminal(self, state) -> Bool:            # explicit; not injected into h here
        ...
```

Capabilities it must advertise, because they decide which solver can consume it:
`jacobian="forward"`, `hessian=None`, `jacobian_cost=12 * step_cost`,
`supports_batch=True`.

### 9.2 The controller problem (solver-agnostic)

```python
@dataclass(frozen=True)
class ConstraintProblem:
    u_nom: Array[12]          # policy-mean action, policy units
    u_lo: Array[12]           # box, policy units
    u_hi: Array[12]
    W: Array[12, 12]          # positive definite, policy units
    barrier: BarrierModel     # nonlinear g and its supplied Jacobian
    state: State              # full prediction state (qpos/qvel/phase/commands/prev target)
    trust_region: float | None
    slack_penalty: float | None   # None => hard constraints only
```

### 9.3 Solver implementations and saved state

```python
class ConstrainedSolver(Protocol):
    @classmethod
    def build(cls, settings, problem_spec) -> "ConstrainedSolver": ...
    def solve(self, problem, initial_action, warm_start) -> "SolverResult": ...

@dataclass
class SolverResult:
    u: Array[12]
    status: Literal["feasible", "iteration_limit", "infeasible", "trust_collapse",
                    "nonfinite"]
    g_original: Float         # re-evaluated on the full nonlinear g
    linearised_residual: Float
    slack: Array | None
    iterations: int
    stationarity: Float | None
    saved_state: Any
```

| strategy | consumes | owns | must not pretend |
|---|---|---|---|
| `LinearisedQP` | `g`, `jac` at one reference, bounds, W | one QP; **no** feasibility claim | that a solved QP satisfies `g` |
| `TrustRegionSQP` | `g_and_jac` per iteration | linearisation, QP sequence, merit/acceptance, trust update, final `g` check | that it is a plain QP solver; that acceptance proves feasibility |
| `ElasticQP` (relaxed) | as `LinearisedQP` + slack penalty | explicit slack reporting | that the relaxed solution satisfies the hard problem |
| `NLP` (jaxipm/MadNLP) | callables + derivative hooks | its own SQP/IPM | that a forward-mode Jacobian is enough — these need symbolic/AD Hessians |

Who does what, explicitly: **the model** linearises nothing; **the strategy**
linearises and owns repeated solves; **the controller** re-evaluates the original
`g(u*)` on the device, decides accept/reject/fallback, and is the only place
that can choose to execute the nominal action instead. Fallback order:
`u*` if `g(u*) >= 0` and status is feasible -> best feasible iterate from the
same solve -> nominal `u_nom` -> hold previous safe action. Never execute an
unchecked proposal; never treat non-convergence as infeasibility.

Device/dtype requirements found: `float64` for the prediction and `h`;
QPAX wants `f32`-friendly tolerances or an explicit relaxed `solver_tol` in f64;
cuPIQP requires GPU-resident inputs (no silent host copies) and a Warp version
Q2 does not currently pin.

---

## 10. Recommendation

**First implementation:** `TrustRegionSQP` over **QPAX elastic QP** (f64, single
environment) with the device-side loop of `scripts/device_sqp.py`, the explicit
`g` re-check on every proposal, and the status enum above. Rationale: it is the
only combination measured here that (a) solves on the GPU, (b) accepts supplied
forward-mode derivatives, (c) reports slack and iteration limits, and (d) is
already integrated with the real MJX predictor and the real Q.

**Alternatives, in order:** cuPIQP if the write path is a Warp-native controller
— it is faster per solve (1.38 ms sync, 0.037 ms queued for the same QP), has an
explicit infeasibility status, native batching (5.19 µs/solve at B=256) and
CUDA-graph support — but it needs `warp-lang>=1.17` (a Q2 dependency bump with
its own mujoco-warp/regression work). BoxOSQP is not competitive (200x slower,
f32 broken, no infeasibility status). jaxipm is the only released GPU NLP found
that handles this problem class, and it is unusable here (Python 3.12, symbolic
derivatives of the model, 7 s per solve) — keep it as the reference for "what a
GPU NLP would have to be", not as the controller.

**Blocking issues, in priority order:**

1. **The barrier's formulation, not the solver.** `beta <= 0.5` leaves a
   mostly-satisfiable constraint with violations of order 0.1 Q-units; `beta = 1`
   turns it into a one-step reachability demand on a discounted return and can
   be violated by 9.6. Choose `beta`/`c` from the artefact's own return
   semantics (or use a multi-step/horizon condition), and define what happens
   when the state is outside the set (currently: nothing feasible).
2. **No feasibility guarantee on the real states.** The loop reaches the
   analytic optimum on the projection test but on the real states stalls at
   `g ≈ -0.0001…-0.6` (one state reached +6e-5) — the one-step requirement looks
   locally infeasible there. Add an exact-penalty merit function or a phase-1
   feasibility QP, and make "no certified action" an explicit controller state.
3. **Throughput.** 73–357 ms per decision (median 163 ms) vs a 20 ms interval.
   The physics is 18 % of it; the rest is launch/sync overhead in the QP path.
   The ways forward are a cheaper QP backend (fewer cuSOLVER calls, CUDA-graph
   capture, cuPIQP), not more GPU FLOPs.
4. **Contact non-smoothness.** The gradient is local to the active contact set;
   no amount of GPU speed fixes that for a CBF that must reason about contact
   make/break.
5. **Dependency surface.** JAX cuda13 rewrites 12 `nvidia-*` wheels under Q2's
   torch/Warp stack; it worked here, but it is a packaging and regression
   decision, not a drop-in.

**Not established (do not read as claims):** control quality, safety,
intervention benefit, forward invariance, that the Q set is a barrier, that any
of these proposals improves the robot, behaviour under `num_envs > 1`, VSim,
hardware, or long-horizon stability. Functional feasibility of the solve is
established; safety and policy quality are not.

---

## 11. Reproduction

All commands run from `/tmp/opencode/gpu-qp-20261115-a` with
`PYTHONPATH=scripts` where a script imports a sibling. Full details in
[`scripts/README.md`](scripts/README.md).

```bash
# reference data (product interpreter, read-only)
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disabled \
  BigRedGym/.venv/bin/python scripts/torch_reference.py \
    --checkout /tmp/opencode/mjx-go2trot-20261115-a/checkout \
    --out artifacts/nov10 --steps 100 \
    --checkpoint BigRedGym/logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
    --q-artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct
PYTHONDONTWRITEBYTECODE=1 BigRedGym/.venv/bin/python scripts/export_weights.py ... 

# QP battery
env/bin/python scripts/qp_battery.py                       # QPAX + BoxOSQP
PYTHONPATH=scripts env_cupiqp/bin/python scripts/cupiqp_battery.py

# composite barrier, parity and gradients
env/bin/python scripts/cbf_composite.py

# SQP (Python-controlled, per-iteration log) - two regimes
env/bin/python scripts/sqp_controller.py --threshold-quantile 0.5 --beta 0.5 \
  --out results/sqp_moderate.json      # also prints the ball-projection case
env/bin/python scripts/sqp_controller.py --threshold-quantile 0.9 --beta 1.0 \
  --out results/sqp_strict.json

# device-side SQP + profile
env/bin/python scripts/device_sqp.py --states 0,1,4,10,28,50
nsys profile -o traces/device-sqp --trace=cuda --capture-range=cudaProfilerApi \
  --capture-range-end=stop env/bin/python scripts/device_sqp.py \
  --states 0 --profile --out results/device_sqp_profile.json

# dependency coexistence
env_coexist/bin/python scripts/coexistence.py

# GPU NLP candidate (isolated py3.12)
LD_LIBRARY_PATH=$PWD/env_jaxipm/lib/python3.12/site-packages/nvidia/cu13/lib \
  env_jaxipm/bin/python scripts/jaxipm_test.py
```

### Temporary directory and what is lost on cleanup

`/tmp/opencode/gpu-qp-20261115-a/` contains the four isolated environments, the
`artifacts/nov10` reference rollout (qpos/qvel/obs/action/Q per step), the
weight npz, every result JSON quoted above, and the Nsight traces
(`traces/device-sqp.nsys-rep`, `traces/device-sqp-feasible.nsys-rep`) with their
SQLite exports. The environments and the traces do not survive cleanup (they are
reproducible from the commands above); the fixtures
`artifacts/nov10/torch_ref.npz` and `artifacts/nov10/weights.npz` are needed to
re-run the composite gates and are also reproducible from `torch_reference.py` +
`export_weights.py` against the two external artifacts (policy checkpoint and Q
directory), both of which are themselves outside the repository and temporary.

It also depends read-only on `/tmp/opencode/mjx-go2trot-20261115-a/` (diagnostic
MJX model, checkout copy) and on
`/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct` (Q artifact).
Deleting those breaks reproduction; none was modified.

Cleanup: `rm -rf /tmp/opencode/gpu-qp-20261115-a` after review. No process was
left running; GPU memory returned to the 647 MiB desktop baseline.

### Permanent files created

| file | note |
|---|---|
| `docs/plans/supplements/gpu_qp/REPORT.md` | this report |
| `docs/plans/supplements/gpu_qp/prompt.md` | the investigation brief, byte-identical (sha256 `c572595d…76c8c2`) |
| `docs/plans/supplements/gpu_qp/scripts/README.md` | reproduction instructions |
| `…/scripts/cbf_composite_lib.py` | artefacts, 48-D observation, policy, Q, PD+MJX step |
| `…/scripts/cbf_composite.py` | parity, `h`, `g`, gradient gates |
| `…/scripts/qp_cases.py` | shared 12-variable battery definitions and references |
| `…/scripts/qp_battery.py` | QPAX + BoxOSQP battery, timing, batch |
| `…/scripts/cupiqp_battery.py` | cuPIQP battery, timing, batch |
| `…/scripts/sqp_controller.py` | Python-controlled linearised-QP/SQP comparison |
| `…/scripts/device_sqp.py` | device-side SQP + profiler hooks |
| `…/scripts/jaxipm_test.py` | GPU NLP candidate on the ball problem |
| `…/scripts/coexistence.py` | same-process torch/JAX/MJX/Warp check |
| `…/scripts/torch_reference.py` | Q2/Torch reference rollout exporter |
| `…/scripts/export_weights.py` | actor/Q weights → npz |

### Confirmation of non-interference

* `git -C BigRedGym status --short` reports the pre-existing modified
  `docs/plans/intervention_testing_plan.md`, the pre-existing untracked
  `docs/plans/investigations/mjx-jax/REPORT.md`, the pre-existing
  `docs/plans/supplements/full_gpu_mppi/`, and this new
  `docs/plans/supplements/gpu_qp/` — nothing else. No product source, config,
  `pyproject.toml`, `uv.lock`, `.venv`, `logs/`, or `.opencode` file was
  modified; nothing was committed.
* Every experiment ran in `/tmp/opencode/gpu-qp-20261115-a/`; the product
  interpreter was used read-only with `PYTHONDONTWRITEBYTECODE=1` for the two
  reference exports only.
