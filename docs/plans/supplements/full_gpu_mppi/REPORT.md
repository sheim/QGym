# Fully GPU MPPI for Q2 Go2Trot — feasibility investigation

**Status:** investigation complete, 2026-11-15. Q2 product source, configuration,
dependencies and `.venv` were not modified.

**Question.** Can released `pytorch-mppi` drive this complete computation on the
GPU — candidate generation → physics prediction → nominal-policy rollout →
Q/cost evaluation → action selection — with no CPU↔GPU data transfer in the
warmed hot path, for Q2's MuJoCo Warp `go2trot` task, its frozen PPO actor and
an existing fixed-policy height Q?

**Short answer.** Yes for residency: every stage runs on the device and no
payload data crosses the bus. Two qualifications matter. First, the unmodified
Q2 Warp backend blocks the host once per physics step and once per state
commit; both are removable with bit-identical local replacements. Second, the
decision is not fast: ~92 % of a decision is host-side kernel-launch overhead,
so a 50 Hz controller needs CUDA-graph capture, not more GPU work. Controller
*quality* was not established and remains open.

Temporary experiment directory (keep until reviewed):
`/tmp/opencode/full-gpu-mppi-20261115-a1/`
Permanent reproductions: [`scripts/`](scripts/), [`scripts/README.md`](scripts/README.md).

---

## 1. Verdict

| # | Claim | Verdict | Basis |
|---|---|---|---|
| 1 | Dependency coexistence with Q2's locked stack | **PASS** | resolves and imports with the exact locked torch/mujoco/warp versions; adds `arm-pytorch-utilities 0.5.0`, `pytorch-seed 0.2.0` |
| 2 | Actual MPPI integration (library rule unchanged) | **PASS with caveat** | upstream `MPPI` samples, scores and reweights unchanged; the policy enters through the documented `SpecificActionSampler` hook; the update collapses to ≈1 effective candidate (§3) |
| 3 | GPU physics | **PASS** | Q2's own `MuJocoWarpBackend`/`Go2Trot` code, K candidate worlds per decision |
| 4 | GPU policy rollout | **PASS** | frozen actor rolled forward T steps in dedicated Warp worlds; 2.4e-7 parity with the single-row artifact path |
| 5 | GPU Q evaluation | **PASS** | Q forward on the predicted terminal observation; exact match with `HeightQ.predict`, 4.8e-7 vs a CPU copy |
| 6 | GPU selection | **PASS** | `cost_total` reduce, importance weights, `U` update and `action[0]` all on device |
| 7 | No host↔device **data** transfers | **PASS (qualified)** | 388 B per decision, all 4-byte solver-predicate reads inside `mujoco_warp`'s conditional-graph loop; no observation/action/model/telemetry payload ever crosses |
| 8 | No host **synchronisation** in the hot path | **PASS only after two local backend replacements** | 0/3 decisions with the replacements, 3/3 without; see §5.3 |
| 9 | Multi-environment support | **FAIL (structural)** | `MPPI` holds one state, one nominal sequence and one weight vector; two controllers cost 1.93× latency |

Not assessed: control quality, safety, tracking performance, VSim, hardware,
CUDA-graph capture (assessed, not implemented).

---

## 2. Exact environment

### Hardware and driver

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB, sm_120 |
| driver | 580.178.04 (CUDA 13.0); Warp reports toolkit 12.9 |
| other GPU load during runs | 647–1 400 MiB used, no foreign compute process |

### Python stack (isolated virtual environment, versions pinned to `uv.lock`)

| package | version | note |
|---|---|---|
| python | 3.11.18 | matches `requires-python` |
| torch | 2.13.0+cu130 | locked |
| numpy | 2.4.6 | locked |
| mujoco | 3.11.0 | locked |
| mujoco-warp | 3.11.0 | locked `gpu` extra |
| warp-lang | 1.16.0 | locked |
| tensordict | 0.13.0 | locked; required by Q2 runner imports |
| pandas / matplotlib / wandb | 3.0.5 / 3.11.1 / 0.28.1 | locked; required by Q2 import graph |
| **pytorch-mppi** | **0.9.1** | candidate library, PyPI wheel `pytorch_mppi-0.9.1-py3-none-any.whl` |
| arm-pytorch-utilities | 0.5.0 | new transitive dependency of `pytorch-mppi` |
| pytorch-seed | 0.2.0 | new transitive dependency of `arm-pytorch-utilities` |

Dependency changes and conflicts: none. The candidate library is pure Python,
requires only `torch>=2.0` and `numpy`, and installed against the locked stack
without re-resolution or downgrade. MPPI's `test` extra (`gym`, `pygame`,
`pyglet`, `ray`, `cma`) was deliberately not installed; `gym` would shadow Q2's
`gym` package on import. The experiment ran in
`/tmp/opencode/full-gpu-mppi-20261115-a1/.venv`, never in the checkout's `.venv`;
`pyproject.toml` and `uv.lock` were not touched.

### Source provenance

| item | value |
|---|---|
| product revision | `5c7a7930bdf22636776b3745c4cbb121e265dc56` (branch `bh574/final/dev/offline_RL`) |
| uncommitted product changes before/after | one pre-existing untracked file, `docs/plans/investigations/mjx-jax/REPORT.md`; untouched |
| experiment source copy | `/tmp/opencode/full-gpu-mppi-20261115-a1/BigRedGym/` (rsync of the checkout, excluding `.git`, `.venv`, `logs`, `thirdparty`, `docs`) |
| experiment code | `/tmp/opencode/full-gpu-mppi-20261115-a1/work/` → sanitized copies in [`scripts/`](scripts/) |

### Artifacts

Policy: `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt`
(git-ignored), sha256 `f88b1f9b829675c97ed0c420202cf54456dc414641def34e379e39407ee3adf6`.

Q: `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct`
(**temporary directory**), format `brg-height-q-v1`, schema 1, created
2026-11-10T16:50:18Z, `q.pt` sha256
`e59d738a606979edeb021c86651e78f632bd4b78a99a9a24e80ebecccdad2236`,
manifest sha256 `db267188c3f8cfe5cfe9d54638e318c387537d3320c799374d557fcbb90de0e9`,
trained on 1 000 000 replay transitions from the checkpoint above, artifact
source revision `1fb64bd87b4af49d460da3aecfb70fb45e5cf520`.

Artifact contract used by the controller:

| field | value |
|---|---|
| observation | 48 = 3 `base_ang_vel` + 3 `projected_gravity` + 3 `commands` + 12 `dof_pos_obs` + 12 `dof_vel` + 12 `dof_pos_target` + 2 `phase_obs` + 1 `phase_frequency` |
| action | `dof_pos_target`, 12 policy units, scale `[0.2, 0.3, 0.3]` per leg |
| normalisation | Q input only, `(obs − obs_mean) / obs_std`; the actor uses `normalize_obs=False` |
| reward | `r = 0` if \|height − 0.32\| ≤ 0.05 else `−1`; post-action, pre-reset |
| return | discounted continuing, `γ = 0.99`, termination is absorbing failure with `value = −100` |
| control interval | `ctrl_dt = 0.02 s` (50 Hz), `decimation = 1`, `sim_dt = 0.02 s` |

Provenance was machine-checked, not assumed: `validate_playback_contract`
accepted the adapter's environment configuration, so the artifact config
fingerprint `6412a8784c883ecb3b28c9601ad1871bec4acb97bfee0f5d7b13e77acc9938cb`,
the checkpoint hash, observation names and widths, action convention, scales,
noise model and control interval all match, and the live task tensor widths were
re-checked after construction.

Both artifacts were real and compatible, so **no synthetic stand-in and no
retraining were used**: every number in this report comes from the shipped
policy and the fitted Q above. The only synthetic elements are the seeds and the
evaluation world's fixed command trajectory.

---

## 3. Algorithm definition

### Rollout semantics the library provides

`MPPI` represents a decision as a nominal action sequence `U ∈ R^{T×nu}`. Each
`command(state)` call samples `K` perturbed sequences `U + ε`, clamps them to
`[u_min, u_max]`, rolls each through a user `dynamics(state, u, t)` while
accumulating a user `running_cost(state, u, t)`, adds an optional
`terminal_state_cost`, forms importance weights
`ω_k ∝ exp(−(C_k − min C)/λ)`, and updates `U ← U + Σ_k ω_k ε_k`. The executed
action is `U[0]`.

Three distinct uses of the artifacts are possible, and this investigation used
the middle-plus-terminal combination:

1. sample action sequences and score them by physics rollouts only;
2. put the nominal policy **inside** the predicted rollouts, as the proposal
   distribution;
3. use Q as a terminal value or another explicit objective.

### Objective actually implemented

For a candidate sequence `u_0…u_{T−1}` in policy units, `γ`, `r_h` and `F = |−100|`
taken from the artifact:

```text
C(u) = Σ_{t<T, still active}  −γ^t · r_h(s_t)          physics running cost
     + γ^tterm · F             if the candidate falls at step tterm
     − γ^T · Q(s_T, π(s_T))    if still active          learned terminal value
```

`r_h` and the failure magnitude are the artifact's own reward channel and
terminal convention, so the truncated objective is the quantity Q was fitted to
estimate; the horizon truncation is bootstrapped by Q, not invented.

Pseudocode of one decision:

```text
1  scatter the evaluator state s0 into K candidate Warp worlds
2  policy proposal: in M dedicated worlds, roll π for T steps
      obs_t = observations(world) ; a_t = π(obs_t) + std·ε ; step(world, a_t)
   → M closed-loop nominal sequences (injected through
     SpecificActionSampler.sample_trajectories, the library's documented hook)
3  upstream MPPI: K−M Gaussian perturbations around U, plus the M policy
   sequences, clamped to [−2, +2] policy units
4  for t in 0..T−1:   write PD targets, run one full control interval of
                      Q2 physics (decimation substeps, torque clipping),
                      rebuild the 48-D actor observation, add the height cost
5  terminal cost: π(s_T), then Q(s_T, a) on the device
6  importance weights, U update, execute U[0]
```

### Observation convention

Both the controller input and every predicted observation are **clean** (no
observation noise). This is what Q was fitted on — the replay recorder stores
the pre-reset clean observation — and it is what Q2's guidance requires of
inference. The artifact's noise model and `normalize_obs=False` are still
validated as part of the contract check, but no noise is injected at run time,
so a controller here is not reproducing the stochastic observation channel the
policy was trained with. `n_policy_steps == 1` and `control_dt == 0.02 s` were
likewise validated, not assumed.

### Is upstream MPPI unchanged?
Yes. `InstrumentedMPPI` subclasses `MPPI` only to wrap
`_compute_perturbed_action_and_noise` and `_compute_weighting` in NVTX ranges;
it calls `super()` and changes nothing else. The sampling rule, the bounding
rule, the importance weights and the update are the library's. The policy
enters exclusively through the library's `SpecificActionSampler` extension
point, which is documented as "encoding trajectory prior and domain knowledge
to create biases".

Consequences worth stating plainly:

- Policy sequences receive the same action cost as Gaussian samples
  (`perturbation_cost = Σ_t U_t · (λ ε_t Σ⁻¹)`), so an aggressive policy prior
  is penalised by the unchanged rule. This is a property of the library, not of
  the adapter.
- Because `12·T` is a high-dimensional control space, the weighting is
  degenerate in practice: the effective number of contributing candidates
  (`1/Σω²`) was **1.0–1.1** at `λ ∈ {0.1, 0.5, 2, 10}` with `K = 256, T = 4`.
  Selection behaves as *best-sample* selection, so the "MPPI" label here means
  "physics-scored candidate set", not a smooth path-integral average.

### What the adapter had to do that the library does not provide

- A `dynamics` that owns the real state in Warp and returns the predicted
  observation as the MPPI state vector (`nx = 48`); the library's `state`
  argument is a mirror of the world, not the state store.
- A policy-proposal sampler that must run physics **before** the main rollout
  loop, therefore needs its own worlds (see §6, batching limits).
- Explicit terminal handling, because Q2's `task.step()` raises
  `terminated`/`timed_out` flags and never resets; `_active` masking is the
  adapter's job.

---

## 4. Correctness evidence

All gates are reproduced by [`scripts/verify_prediction_parity.py`](scripts/verify_prediction_parity.py)
and [`scripts/verify_residency.py`](scripts/verify_residency.py).

### 4.1 One-step parity against the ordinary Q2 step

The evaluation world was warmed 40 control steps with the real policy, then for
5 further steps: capture state → branch into 4 candidate worlds → step with the
same action → compare against the same step taken in the evaluation world with
Q2's own code path.

| quantity | worst deviation (candidate 0 vs evaluation env) |
|---|---|
| 48-D scaled actor observation | 5.7e-6 |
| `dof_pos` | 2.4e-7 |
| `root_states` | 1.4e-6 |

Per-trial observations were 1.9e-6 … 2.1e-5 depending on how much contact
warm-start history the snapshot carried. This is the expected Warp behaviour,
not a bitwise match: contact/constraint solver history beyond
`qacc_warmstart`/`cqacc_warmstart` is not part of the snapshot (Q2 documents the
same limitation for its own `restore_state`).

### 4.2 Cross-candidate determinism (the ranking noise floor)

Eight branches of one state stepped with one action are **not** bit-identical:
identical `qpos`/`qvel` inputs give observation differences up to 3e-6 when the
branches are fresh, and up to 2e-5 once a previous decision's solver residue is
in the buffer, located in `base_ang_vel` and `dof_vel`. Copying
`qacc_warmstart` and `cqacc_warmstart` in the snapshot removed the dominant
source (6e-5 → 3e-6); the residue is the engine's own reproducibility limit. For
scale: a deliberate 0.5-policy-unit action change moves the observation by 9.0
(max) / 0.61 (mean), ~2 000× the noise floor. No bitwise-determinism claim is
made anywhere.

### 4.3 Inference parity with the artifact paths

| comparison | max deviation |
|---|---|
| actor, 4-row batch vs single row | 2.4e-7 |
| batched Q (mean/std on device) vs `HeightQ.predict` | 0.0 (exact) |
| Q on CUDA vs the same artifact on CPU | 4.8e-7 |

Example Q values, `K = 4` policy draws, height reward inside the band:
`[−1.721693, −1.787296, −1.711395, −1.735615]`.

The hot path uses a sync-free equivalent of `HeightQ.predict`, because that
method ends with `if not torch.isfinite(prediction).all()`, which reads a
boolean back to the host. The fast path performs the same arithmetic and gate 3
shows it is exact; the finiteness check belongs in a slower validation wrapper,
not in a controller.

### 4.4 Other gates

| gate | result |
|---|---|
| deliberately different candidate action changes the outcome | observation delta 9.0 max; height 0.312572 → 0.313220 |
| evaluator state unchanged by controller rollouts | `torch.equal` on `dof_pos`, `dof_vel`, `root_states` after 3 rollout decisions |
| height cost matches the artifact convention | in-band → 0.0, out-of-band → −1.0, `γ = 0.99`, failure 100 |
| explicit seeds reproducible | two `manual_seed(99)` generators give identical draws |
| action bounds / finiteness | candidates clamped to ±2 policy units; a non-finite Q raises in the artifact path and never appeared in the fast path |
| commit path fidelity | the sync-free all-worlds commit is **bit-identical** to `SimBackend.reset_state` on `dof_pos`, `dof_vel`, `root_states`, `contact_forces`; only the *following* step differs (1e-7…1.2e-5), i.e. the engine floor again |

---

## 5. Profiling evidence

Boundary: `controller.action(input_already_on_gpu) → action_on_gpu`. Artifact
loading, environment construction, Warp kernel compilation and reporting are
outside it. Timings synchronise at that boundary only.

### 5.1 Trace artifacts (temporary)

| file | content |
|---|---|
| `/tmp/opencode/full-gpu-mppi-20261115-a1/traces/hot-patched-k256-t8.nsys-rep` | K=256, T=8, M=8, sync-free backend, 6 warmed decisions |
| `/tmp/opencode/full-gpu-mppi-20261115-a1/traces/hot-k256-t8.nsys-rep` | same, unmodified backend (shows the extra synchronisations) |
| `/tmp/opencode/full-gpu-mppi-20261115-a1/traces/warp-only.nsys-rep` | bare `env.step()` loop, no controller code |
| `/tmp/opencode/full-gpu-mppi-20261115-a1/traces/warp-launch.nsys-rep` | bare `wp.launch` loop (control for the transfer attribution) |
| `/tmp/opencode/full-gpu-mppi-20261115-a1/results/cost-k*-t*.json` | measured tables below |

Nsight capture used `-c cudaProfilerApi`; `--capture-range=nvtx` produced no
report on this installation, so the hot window is delimited with
`torch.cuda.profiler.start()/stop()`. NVTX child ranges remain available via
`GMPPI_NVTX=1` and `nsys stats -r nvtx_sum`.

### 5.2 Per-range share of one decision (K=256, T=8, M=8, patched)

| range | ms per decision | note |
|---|---|---|
| `controller/decision` (total) | **97.6** | 6 decisions, 97.0–102.5 ms spread |
| `physics/step` (48 calls) | 48.4 | 6.05 ms per control step of 256 worlds |
| `policy/rollout` (6 calls) | 47.3 | 5.9 ms per control step of **8** worlds |
| `mppi/candidate_generation` | 47.4 | nests `policy/rollout` |
| `cost/height_reward` (48 calls) | 0.48 | 59 µs per call |
| `terminal/policy+Q` (6 calls) | 0.31 | actor + Q over 256 rows |
| `mppi/update` (6 calls) | 0.08 | weights, `U` update, selection |

GPU busy vs wall for the same window: **30 582 kernels (5 097 per decision),
7.98 ms of GPU time against 97.6 ms wall — 92 % of the decision is host-side.**
A control step is ~250 Warp kernel launches; at ~20 µs of host time per launch
that is ~5 ms regardless of world count. The consequence is visible in the
table below: 16× more candidates costs 11 %.

### 5.3 Transfer and synchronisation audit

Hot window (6 decisions), patched backend:

| metric | value |
|---|---|
| host→device copies | **0** |
| device→host copies | 582 (97 per decision), **2 328 B total, 388 B per decision**, every one 4 bytes |
| device→device copies | 840 (140 per decision), 33.1 MB (5.5 MB per decision) |
| `cudaMallocAsync` / `cudaFreeAsync` | 5 772 / 5 772 (allocator churn from per-step temporaries) |
| `cudaMemcpyAsync` | 1 422 |

Attribution of the 4-byte host reads: a bare `env.step()` loop with no
controller code reproduces them (≈4.7 per step), a bare `wp.launch` loop
produces **zero** transfers, and correlation with kernel activity places most of
them immediately after the constraint-solver kernel `_solve_done`. The source is
`mujoco_warp`'s solver:

```python
nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)
wp.capture_while(nsolving, while_body=_solver_iteration, ...)
```

The graph-conditional solver loop reads its 1-int predicate on the host each
iteration. It is a scalar control read, not a data transfer, and it cannot be
removed without changing the engine. (Setting `opt.graph_conditional = False`
is strictly worse here: 4.5 ms → 20.2 ms per control step at K=8, because the
loop then runs all 100 configured iterations.)

Device-to-device traffic (140 copies, 5.5 MB per decision) is the state scatter,
the MPPI terminal-state storage (`[1,K,T,48]` plus actions) and the assembled
canonical buffers. All explained, all device-local.

**Synchronisation, unmodified backend:** `torch.cuda.set_sync_debug_mode("error")`
reported a host synchronisation in **3/3** decisions. Two sites, both in
`gym/envs/base/mujoco_warp_backend.py`, both caused by advanced indexing on a
non-contiguous view of a Warp-backed tensor:

| site | cause | frequency |
|---|---|---|
| `reset_state` | `torch.where(mask, rs[:,3:7][:, XYZW_TO_WXYZ], qpos_view, out=qpos_view)` | once per state commit |
| `_sync_assembled_states` | `rs[:,3:7] = self._qpos_t[:,3:7][:, WXYZ_TO_XYZW]` and the `xquat` equivalent | **once per `step()`** |

**Synchronisation, with two local replacements:** **0/3** decisions.
[`scripts/gmppi_patch_warp_sync.py`](scripts/gmppi_patch_warp_sync.py) replaces
`_sync_assembled_states` with a slice-only rewrite that is **bit-identical** on
`dof_pos`, `dof_vel`, `root_states`, `rigid_body_states` and `contact_forces`,
and the adapter commits state with contiguous `copy_` instead of
`torch.where(..., out=<strided view>)`, also bit-identical. The cost of *not*
patching is small in wall time — 51.0 ms vs 49.5 ms per decision at K=256,
T=4 — because the step is launch-bound, but it invalidates a residency claim.

**Warp/Torch exchange.** No copies are made between the two frameworks: all
Warp data is reached through `wp.to_torch` views, so a write is a device write.
Aliasing hazard found and handled: `_dof_state_t` and `_root_states_t` are
persistent assembled buffers that `step()` refreshes eagerly; a controller must
copy state *in* only through those views and must never hold a stale reference
across `reset_state`. Lifetime hazard: `MPPI.U` is rebound (not mutated) by
`_command`, so a caller must read `get_action_sequence()` rather than keep a
reference. Stream ordering needed no explicit work because everything runs on
the default stream; a multi-stream design would have to reconcile Warp's
`ScopedDevice` stream with Torch's.

### 5.4 Practical cost

`K` = candidates, `T` = control steps in the prediction horizon, `M` = 8
policy-rollout worlds, one evaluation environment, `patched`, warmed, median of
12 decisions:

| K | T | candidates·steps per decision | decision latency | decisions/s | physics steps/s | torch peak (alloc/reserved) | GPU memory (process) |
|---|---|---|---|---|---|---|---|
| 64 | 4 | 256 | 48.6 ms | 20.6 | 5 273 | 34 / 34 MiB | 1 323 MiB |
| 256 | 4 | 1 024 | 50.7 ms | 19.7 | 20 207 | 36 / 36 MiB | 1 325 MiB |
| 256 | 8 | 2 048 | 92.3 ms | 10.8 | 22 178 | 36 / 38 MiB | 1 325 MiB |
| 1024 | 4 | 4 096 | 53.6 ms | 18.6 | 76 361 | 44 / 64 MiB | 1 415 MiB |
| 1024 | 8 | 8 192 | 98.8 ms | 10.1 | 82 948 | 47 / 64 MiB | 1 415 MiB |

These are the values written to `results/cost-k*-t*.json` by
`scripts/measure_cost.py`. The NVTX range totals in §5.2 read ~5 % higher
(97.6 ms) because that window was collected under the profiler.

Component medians at K=256, T=8: scatter 4.7 ms (includes one `mjw.forward`),
physics+observation 36.6 ms for 8 steps, policy rollout 41.9 ms for 8 steps of 8
worlds, terminal policy+Q 0.15 ms, decision 92 ms. Compilation/setup is outside
every number above; the first Warp step of a process pays JIT kernel loading
(tens of milliseconds per kernel, cached in `~/.cache/warp/1.16.0`).

No configuration failed and nothing came close to OOM: the process held
1.3–1.4 GiB of 31 GiB, and `K` from 64 to 1 024 added ~90 MiB. The budget was
kept conservative (≤ 2 GiB planned, ≤ 5 % of the card used) so unrelated GPU
work could continue; the largest configuration tried was `K = 1 024`.

Interpretation of the table:

- Latency is **insensitive to K** and roughly linear in `T`: the ceiling is
  `2·T` sequential engine steps (one physics pass plus one policy pass), each
  dominated by fixed host cost.
- Physics throughput is fine (22 000–83 000 control steps/s), so the GPU is not
  the limiter.
- Consequences for a real controller: at 50 Hz (20 ms/decision) the current
  design reaches only 10–21 Hz, and `T = 8` misses by 5×. Because 92 % is host
  launch overhead, CUDA-graph capture of a whole decision is the enabling
  optimisation, not a nicety — it would replace ~5 100 launches and ~96 ms of
  host time with one graph launch against ~8 ms of GPU work, i.e. ≈ 10 ms per
  decision at these sizes. Warp supports capture (`wp.capture_*`) and
  `mujoco_warp` already uses a device-side conditional loop, so the capture
  target is the adapter's Python control flow, which must become loop-free with
  preallocated buffers. This was assessed, not implemented: the feasibility
  question is residency, and residency holds without it.

### 5.5 Multi-environment

Two evaluation environments with two controllers: **49.9 ms → 96.4 ms
(1.93×)**. Candidate batching does not amortise controllers, and memory doubles
per extra prediction world set. §6 explains why this is structural.

---

## 6. API implications

**Controller inputs.** The controller cannot work from an observation alone. It
needs, all on device: the 48-D observation; the task buffers that define the
next step (`dof_pos`, `dof_vel`, `root_states`, `dof_pos_target`,
`dof_pos_history`, `commands`, `phase`, `phase_frequency`,
`episode_length_buf`, `p_gains`, `d_gains`); the solver warm start
(`qacc_warmstart`, `cqacc_warmstart`); and — once, at construction — the
physical identity (per-world `body_mass`, `body_inertia`, `link_mass`,
`link_inertia`, `geom_friction`, `pair_friction`, plus one `mjw.set_const`).
Startup domain randomisation is fixed for an environment's lifetime, so a
one-time copy suffices; episodic gain randomisation is carried in the scattered
task tensors.

**Dynamics method.** `dynamics(state, u, t) → next_observation` must own the
authoritative state; the `state` argument is a mirror. `nx = 48` (the actor
observation) is the only MPPI state vector, and `terminal_state_cost` is what
makes the truncation bootstrapped.

**Saved controller state.** `U` (`T×12` nominal sequence), `u_init`, the action
generator state, `_active`/`_terminated_now`, and the sampler's pending branch
state. `MPPI.reset()` re-samples `U` from the noise model; a task reset must
clear `_active` and re-scatter before the next decision.

**Resets and terminal handling.** `LeggedRobot.step()` never resets: it raises
`terminated`/`timed_out` flags only, so predictions are auto-reset-free by
construction. The adapter converts a per-candidate termination at step `t` into
`γ^t·F` and stops accumulating reward for that candidate. Command resampling and
push disturbances are instance-disabled (`env._resample_commands`,
`env._push_robots`) because they draw from the global CUDA RNG and a branch
cannot know the evaluator's future draws; the registered config is unchanged, so
the artifact fingerprint still validates.

**Batching limits.** `MPPI` keeps one `state`, one `U`, one `omega` and one
`cost_total`; there is no batch dimension over controllers. Running `N`
evaluation environments therefore needs `N` controllers and `N` prediction world
sets (measured: 1.93× latency for the second). Additionally, the policy proposal
needs its own worlds because `sample_trajectories` runs before the main rollout
loop and every `step()` advances *all* worlds of a Warp model — measured cost of
that second pass is one extra engine-step series, which is why `M = 8` worlds
cost as much as `K = 1024`.

---

## 7. Blockers and alternatives

None of these is a technical impossibility; they differ in how much work they
imply.

| blocker | kind | alternative / work |
|---|---|---|
| 92 % of a decision is host launch overhead; 10–21 Hz instead of 50 Hz | performance work | CUDA-graph capture of the whole decision (Warp supports it; ~10 ms projected at K=256, T=8); otherwise reduce `T`, or accept a slower-than-real-time supervisor |
| Q2's Warp backend synchronises the host once per `step()` and once per `reset_state` | **missing adapter / small upstream fix** | slice-only `_sync_assembled_states` and `copy_`-based commit; both bit-identical, [patch provided](scripts/gmppi_patch_warp_sync.py). Should land as a Q2 change with a regression test |
| `mujoco_warp`'s conditional solver loop reads a 1-int predicate to the host | engine property | accept; it is 388 B per decision and not payload data. `opt.graph_conditional = False` is 4.5× slower |
| MPPI weights collapse to ≈1 effective candidate | algorithmic | reformulate the temperature relative to the action-cost term, use a policy-parameterised update (which would change the library's rule and needs its own justification), or accept best-sample selection and call it that |
| no controller batching in `pytorch-mppi` | library limitation | one controller per evaluation environment; or an outer batch implementation, which is a rewrite of `MPPI`, not an adapter |
| policy proposals pay MPPI's unchanged action cost | library semantics | sample the prior closer to `U` (smaller policy noise), or exclude proposals from the action cost via a subclass — an explicit rule change |
| `HeightQ.predict` reads a finiteness flag back to the host | missing adapter | sync-free equivalent (exact, gate 3) with the check moved to a validation wrapper |
| 4-byte DtoH from the solver loop | engine property | accept and report |

Not attempted, and therefore still open: any statement about control quality,
safety, tracking, or that MPPI improves on the nominal policy. The evidence here
is residency, parity and cost.

---

## 8. Reproduction and cleanup

### Commands

Full instructions, seeds and artifact flags are in
[`scripts/README.md`](scripts/README.md). From the checkout root, in the
isolated environment:

```bash
PYTHONPATH=<checkout>:<scripts> python verify_prediction_parity.py
PYTHONPATH=<checkout>:<scripts> python verify_residency.py --patch-warp-sync
PYTHONPATH=<checkout>:<scripts> python measure_cost.py --samples 256 --horizon 8 \
  --policy-sequences 8 --patch-warp-sync --out costs.json
GMPPI_NVTX=1 nsys profile -o hot --trace=cuda,nvtx -c cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  PYTHONPATH=<checkout>:<scripts> python profile_hot_path.py \
  --samples 256 --horizon 8 --patch-warp-sync
```

Seeds: evaluation world 7, candidate predictor 1, policy worlds 2, controller
generator 3, commit-parity predictor 11, policy draws 1234/99.

### Temporary directory

`/tmp/opencode/full-gpu-mppi-20261115-a1/` — 5.5 GiB, kept for review. Contains:

| path | content | survives? |
|---|---|---|
| `.venv/` | isolated environment (locked stack + `pytorch-mppi`) | no — recreate from `scripts/README.md` |
| `BigRedGym/` | rsync source copy at the recorded revision | no — checkout is authoritative |
| `work/` | experiment scripts before sanitizing | no — copies live in `scripts/` |
| `results/cost-k*-t*.json` | latency/memory tables in §5.4 | **no** — values transcribed into this report |
| `traces/*.nsys-rep` | Nsight Systems captures in §5.1 | **no** — raw evidence lost on cleanup; the report's numbers and the reproduction commands remain |
| `locked-reqs.txt` | `uv export` of the locked stack | reproducible |

The Q artifact at `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/` is
also temporary and is not part of this directory; deleting it removes the ability
to re-run any gate with this exact Q. The checkpoint under `logs/` is
git-ignored and equally not durable. **Reproduction therefore needs both
external artifacts to be preserved separately; this report records their
hashes so a surviving copy can be verified.**

Cleanup: `rm -rf /tmp/opencode/full-gpu-mppi-20261115-a1` after review. No
cleanup was performed, and no process was left running.

### Confirmation of non-interference

- Product implementation, configuration and dependency files untouched:
  `git -C BigRedGym status --short` reports only the pre-existing untracked
  `docs/plans/investigations/mjx-jax/REPORT.md` plus this new
  `docs/plans/supplements/full_gpu_mppi/` directory.
- The checkout's `.venv`, `pyproject.toml` and `uv.lock` were never written;
  every command used the isolated environment.
- No `logs/`, checkpoint, cache or profiler trace was written inside the
  checkout. One accidental artifact (`uv export` output) was produced in the
  checkout during setup and removed immediately; it is recorded here for
  completeness.

### Permanent files created

| file | note |
|---|---|
| `docs/plans/supplements/full_gpu_mppi/REPORT.md` | this report |
| `docs/plans/supplements/full_gpu_mppi/prompt.md` | the investigation brief, relocated unchanged |
| `docs/plans/supplements/full_gpu_mppi/scripts/README.md` | reproduction instructions |
| `docs/plans/supplements/full_gpu_mppi/scripts/gmppi_adapter.py` | prediction adapter |
| `docs/plans/supplements/full_gpu_mppi/scripts/gmppi_controller.py` | MPPI controller |
| `docs/plans/supplements/full_gpu_mppi/scripts/gmppi_common.py` | shared artifact/eval setup |
| `docs/plans/supplements/full_gpu_mppi/scripts/gmppi_patch_warp_sync.py` | sync-free assembled-state refresh |
| `docs/plans/supplements/full_gpu_mppi/scripts/verify_prediction_parity.py` | gates 1–6 |
| `docs/plans/supplements/full_gpu_mppi/scripts/verify_residency.py` | gates 7–8 |
| `docs/plans/supplements/full_gpu_mppi/scripts/measure_cost.py` | latency/memory harness |
| `docs/plans/supplements/full_gpu_mppi/scripts/profile_hot_path.py` | Nsight target |
| `docs/plans/supplements/full_gpu_mppi/scripts/probes.py` | supporting probes |

Prompt relocation (separate from the documentation writes):
`.opencode/workspace/plans/investigations/full-gpu-mppi/prompt.md` →
`docs/plans/supplements/full_gpu_mppi/prompt.md`, contents verified identical by
sha256 (`58bd905586155402748e4c4796ba75253bf013bd95c9c1f73b8060abddc93c74`)
before the source was removed; the now-empty source directory was removed and
the sibling `investigations/mjx-jax/` was left untouched.

Nothing was committed.
