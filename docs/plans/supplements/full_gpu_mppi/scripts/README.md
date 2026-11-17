# Reproduction scripts — full-GPU MPPI investigation

Small, sanitized drivers for the experiment described in [`../REPORT.md`](../REPORT.md).
They are experiment code, not product code: nothing here is imported by Q2.

## Prerequisites

Both artifacts are external to the repository, so they are CLI options:

| input | used by the report | note |
|---|---|---|
| `--artifact` | `/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/100pct` | fixed-policy height Q; **temporary**, may be deleted |
| `--checkpoint` | `logs/go2trot/Nov10_10-36-22_gpu-final-600-trot-hist-20261110/model_600.pt` | git-ignored |

`--artifact` defaults to `$GMPPI_ARTIFACT` or the temporary path above;
`--checkpoint` defaults to `$GMPPI_CHECKPOINT` or the path under the product
checkout found from this directory. Both are validated against the other:
`validate_playback_contract` rejects a mismatch in checkpoint hash, observation
names/widths, action convention, scales, noise, control interval, or the
environment configuration fingerprint.

The experiment was run in an isolated virtual environment containing exactly the
locked Q2 stack plus the candidate library. From the product checkout root:

```bash
uv venv --python 3.11 /tmp/opencode/full-gpu-mppi-<id>/.venv
uv pip install --python /tmp/opencode/full-gpu-mppi-<id>/.venv/bin/python \
  torch==2.13.0 numpy==2.4.6 mujoco==3.11.0 mujoco-warp==3.11.0 \
  warp-lang==1.16.0 tensordict==0.13.0 pandas==3.0.5 matplotlib==3.11.1 \
  wandb==0.28.1 pytorch_mppi-0.9.1-py3-none-any.whl
```

Do not install MPPI's `test` extra: it pulls `gym`, which collides with Q2's
`gym` package name.

## Running

All commands assume `PYTHONPATH=<product checkout>:<this directory>` and the
isolated interpreter. Run from this directory.

```bash
# 1  correctness gates before any timing (exit status 1 on deviation)
python verify_prediction_parity.py

# 2  residency gates: sync-free commit parity + no host sync in the hot path
python verify_residency.py --patch-warp-sync

# 3  warmed latency, memory and component timings
python measure_cost.py --samples 256 --horizon 8 --policy-sequences 8 \
  --patch-warp-sync --out costs.json

# 4  Nsight Systems hot window, then the transfer/sync/range reports
GMPPI_NVTX=1 nsys profile -o hot --trace=cuda,nvtx -c cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  python profile_hot_path.py --samples 256 --horizon 8 --patch-warp-sync
nsys stats --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  hot.nsys-rep

# 5  supporting probes
python probes.py --probe step       # step cost vs world count
python probes.py --probe launch     # GPU busy fraction of one step
python probes.py --probe weighting  # importance-weight concentration vs lambda
python probes.py --probe multi-env  # two controllers, two evaluation worlds
```

`--patch-warp-sync` installs the slice-only replacement for Q2's assembled-state
refresh, which is bit-identical to the original and does not synchronise the
stream. Without it, the same gates report one host synchronisation per
`SimBackend.reset_state` and per `_sync_assembled_states` call.

## Files

| file | role |
|---|---|
| `gmppi_adapter.py` | `PredictionTask`: a Q2 `Go2Trot` instance used as a batched counterfactual predictor; state capture/scatter, physical-identity copy |
| `gmppi_controller.py` | upstream `pytorch_mppi.MPPI` wired to Warp physics, the frozen actor and the height Q; policy-prior sampler; NVTX labels |
| `gmppi_patch_warp_sync.py` | isolated `_sync_assembled_states` replacement (experiment scope only) |
| `gmppi_common.py` | artifact loading, evaluation-world construction, path defaults |
| `verify_prediction_parity.py` | gates 1–6 |
| `verify_residency.py` | gates 7–8 |
| `measure_cost.py` | warmed latency, memory, component timings |
| `profile_hot_path.py` | Nsight Systems target |
| `probes.py` | supporting measurements quoted in the report |

## Seeds

Evaluation world `seed=7`, candidate predictor `seed=1`, policy-rollout worlds
`seed=2`, controller generator `seed=3`, parity gates `seed=11` and policy
draws seeded explicitly (`1234`, `99`). Warp is not bit-reproducible across
worlds: identical candidate worlds diverge at ~3e-6 in scaled observations after
one step, which is the ranking noise floor.
