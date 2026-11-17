# GPU-QP investigation scripts

Sanitised reproductions for
[`../REPORT.md`](../REPORT.md). They are the scripts that produced the report's
numbers, with the investigation's own absolute paths left as CLI defaults so the
evidence stays traceable. Nothing here is product code, and none of it is
imported by Q2.

## Environments

Four isolated environments (never Q2's `.venv`) were used. Create them from
`/tmp/opencode/gpu-qp-20261115-a` (the experiment directory) and run the scripts
from there with `PYTHONPATH=scripts`.

```bash
EXP=/tmp/opencode/gpu-qp-20261115-a
cd "$EXP"

# 1. JAX numerics: the earlier MJX freeze + QPAX + JAXopt
uv venv --python 3.11 env
uv pip install --python env/bin/python \
  -r /tmp/opencode/mjx-go2trot-20261115-a/results/freeze_mjx.txt qpax jaxopt

# 2. Same-process torch/JAX/MJX/Warp check: locked Q2 GPU stack + JAX stack
uv venv --python 3.11 env_coexist
uv pip install --python env_coexist/bin/python -r <locked Q2 GPU requirements>
uv pip install --python env_coexist/bin/python \
  -r /tmp/opencode/mjx-go2trot-20261115-a/results/freeze_mjx.txt qpax jaxopt

# 3. cuPIQP (needs warp-lang >= 1.17, so it cannot share Q2's environment)
uv venv --python 3.11 env_cupiqp
uv pip install --python env_cupiqp/bin/python cupiqp numpy scipy

# 4. jaxipm (requires Python >= 3.12 and cuDSS)
uv venv --python 3.12 env_jaxipm
uv pip install --python env_jaxipm/bin/python jaxipm nvidia-cudss-cu13
ln -sf libcudss.so.1 \
  env_jaxipm/lib/python3.12/site-packages/nvidia/cu13/lib/libcudss.so.0
```

`<locked Q2 GPU requirements>` is `uv export --frozen --extra gpu --no-dev
--no-hashes` of `BigRedGym` with the `-e .` line removed.

## Inputs (all external, all read-only)

| input | path | identity in the report |
|---|---|---|
| MJX diagnostic model | `/tmp/opencode/mjx-go2trot-20261115-a/artifacts/variants/model_env0_diag_nocylbox.mjb` | sha256 `f55cf10b…` |
| policy checkpoint | `BigRedGym/logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt` | sha256 `f88b1f9b…` |
| Q artifact | `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct` | `q.pt` sha256 `e59d738a…` |
| Q2 checkout copy | `/tmp/opencode/mjx-go2trot-20261115-a/checkout` | product rev `5c7a7930…` |

Every script takes these as flags (`--model`, `--artifacts`, `--q-manifest`,
`--checkpoint`, `--q-artifact`, `--checkout`); the defaults are the paths above.

## Order of operations

```bash
cd "$EXP"; export PYTHONPATH=scripts

# (1) Torch reference: 100-step Q2 CPU rollout + 48-D obs + policy + Q, plus probes
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disabled \
  /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/.venv/bin/python \
  scripts/torch_reference.py --checkout /tmp/opencode/mjx-go2trot-20261115-a/checkout \
  --out artifacts/nov10 --steps 100 \
  --checkpoint /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt \
  --q-artifact /tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct

# (2) actor + Q weights to npz (same interpreter)
PYTHONDONTWRITEBYTECODE=1 \
  /home/brad/Documents/test_projects/test_legged_robot/BigRedGym/.venv/bin/python \
  scripts/export_weights.py --checkpoint <ckpt> --q-artifact <dir> \
  --out artifacts/nov10/weights.npz

# (3) composite barrier: parity, successor-prediction gate, gradient gate
env/bin/python scripts/cbf_composite.py            # -> results/cbf_composite.json

# (4) QP batteries
env/bin/python scripts/qp_battery.py               # QPAX + BoxOSQP
PYTHONPATH=scripts env_cupiqp/bin/python scripts/cupiqp_battery.py

# (5) SQP: Python-controlled (per-iteration log), two regimes
env/bin/python scripts/sqp_controller.py --threshold-quantile 0.5 --beta 0.5 \
  --out results/sqp_moderate.json
env/bin/python scripts/sqp_controller.py --threshold-quantile 0.9 --beta 1.0 \
  --out results/sqp_strict.json

# (6) device-side SQP (no host decisions inside the loop) + profile
env/bin/python scripts/device_sqp.py --states 0,1,10,28,50
nsys profile -o traces/device-sqp --trace=cuda --capture-range=cudaProfilerApi \
  --capture-range-end=stop env/bin/python scripts/device_sqp.py \
  --states 0 --profile --out results/device_sqp_profile.json

# (7) dependency coexistence
env_coexist/bin/python scripts/coexistence.py

# (8) GPU NLP candidate
LD_LIBRARY_PATH=$PWD/env_jaxipm/lib/python3.12/site-packages/nvidia/cu13/lib \
  env_jaxipm/bin/python scripts/jaxipm_test.py
```

## Script map

| file | role |
|---|---|
| `cbf_composite_lib.py` | artifact loading, 48-D observation, actor, Q, PD + `mjx.step` |
| `cbf_composite.py` | parity gates, `h(s)`, `g(u)`, forward-mode gradient gate |
| `qp_cases.py` | shared 12-variable QP battery with analytic/scipy references |
| `qp_battery.py` | QPAX and BoxOSQP: correctness, limits, timing, 256-way batch |
| `cupiqp_battery.py` | cuPIQP on the same battery; status enum, async timing, native batch |
| `sqp_controller.py` | linearised QP vs trust-region SQP from Python, per-iteration log, ball case |
| `device_sqp.py` | whole SQP in one `lax.while_loop`; `--profile` wraps decisions in `cuProfilerStart/Stop` |
| `jaxipm_test.py` | jaxipm on the closed-form ball projection (py3.12 env) |
| `coexistence.py` | torch + JAX/MJX + Warp + QPAX in one process |
| `torch_reference.py` | Q2/Torch reference rollout exporter (product interpreter, read-only) |
| `export_weights.py` | actor/Q state dicts to npz |

## Notes and gotchas found while doing this

* The free-joint quaternion is `(w,x,y,z)` in `qpos` and `(x,y,z,w)` in Q2's
  `root_states`; `base_ang_vel` is the body-local `qvel[3:6]`. Both mistakes are
  visible immediately in `cbf_composite.py`'s parity gate.
* `h(s_t)` must be evaluated on the observation Q was fitted on: the
  `dof_pos_target` channel holds the **previous** step's target
  (`target_pre[t]`), not the action about to be applied. Using the latter biases
  `h` by up to 2.6 Q-units and shifts every `g`. `sqp_controller.py` and
  `device_sqp.py` now cross-check the reconstructed `h` against `Q` on Q2's own
  recorded 48-D observation (agreement 4.8e-6) before using it.
* `torch_reference.py` records `p_gains`/`d_gains` **per step**: episodic gain
  randomisation runs on every reset, so a value read after the rollout is not
  the value the rollout used.
* `jaxopt.BoxOSQP` with JAX 0.9.2: `params_eq=None` and float32 both fail;
  use a single stacked `A` matrix with `l <= A x <= u`, f64 only.
* `jax.hessian` of the composite raises inside MJX's solver `while_loop`; only
  forward mode works.
* JAX's XLA pool grows to its 25 GB limit by default; set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing the GPU.
