#!/usr/bin/env bash
# mini_cheetah_ref cross-engine parity benchmark (Phase 4).
#
# Trains the reference-gait policy on each backend, then builds two eval sets:
#   * transfer matrix  (reset_to_range, aggregate reward + survival)
#   * DOF samples      (reset_to_basic, identical IC -> cross-backend DOF RMS)
# Inspect with notebooks/mini_cheetah_ref.py.
#
# vsim needs its env file; run this from the repo root.  cpu training is the
# long pole (contact-rich, single-threaded) — it runs last; comment it out and
# reuse a prior cpu run if you only want to refresh the GPU rows.
set -euo pipefail

ITERS=${ITERS:-250}
SEED=${SEED:-7}
GPU_ENVS=${GPU_ENVS:-4096}
CPU_ENVS=${CPU_ENVS:-256}
EVAL_ENVS=${EVAL_ENVS:-256}   # transfer matrix batch (cpu-eval is slow)
T_END=${T_END:-5.0}
VSIM="uv run --env-file .env.vsim"
MJ="uv run"

train() {  # backend device envs exp  [--backend vsim]
  local runner=$MJ; [ "${5:-}" = "vsim" ] && runner=$VSIM
  $runner scripts/train_mujoco.py --task mini_cheetah_ref --device "$2" \
    --num_envs "$3" --max_iterations "$ITERS" --seed "$SEED" --headless \
    --disable_wandb --experiment_name "$4" ${5:+--backend vsim}
}

echo "== training =="
train mujoco cuda:0 "$GPU_ENVS" mini_cheetah_ref_warp
train vsim   cuda:0 "$GPU_ENVS" mini_cheetah_ref_vsim vsim
train mujoco cpu    "$CPU_ENVS" mini_cheetah_ref_cpu

# eval one (train_label, eval_backend, eval_device, eval_label, mode, out, extra)
eval_cell() {
  local runner=$MJ; [ "$2" = "vsim" ] && runner=$VSIM
  $runner scripts/eval_policy.py --task mini_cheetah_ref \
    --ckpt "logs/mini_cheetah_ref_$1" --train_label "$1" \
    --eval_backend "$2" --eval_device "$3" --eval_label "$4" \
    --num_envs "$6" --t_end "$T_END" --reset_mode "$5" --out "$7" ${8:-}
}

echo "== transfer matrix (reset_to_range) =="
for tr in cpu warp vsim; do
  eval_cell "$tr" mujoco cpu    cpu  reset_to_range "$EVAL_ENVS" "logs/mc_rl_eval/${tr}__cpu.npz"
  eval_cell "$tr" mujoco cuda:0 warp reset_to_range "$EVAL_ENVS" "logs/mc_rl_eval/${tr}__warp.npz"
  eval_cell "$tr" vsim   cuda:0 vsim reset_to_range "$EVAL_ENVS" "logs/mc_rl_eval/${tr}__vsim.npz"
done

echo "== DOF samples (reset_to_basic, identical IC) =="
for tr in cpu warp vsim; do
  eval_cell "$tr" mujoco cpu    cpu  reset_to_basic 4 "logs/mc_rl_sample/${tr}__cpu.npz"  --record_dof
  eval_cell "$tr" mujoco cuda:0 warp reset_to_basic 4 "logs/mc_rl_sample/${tr}__warp.npz" --record_dof
  eval_cell "$tr" vsim   cuda:0 vsim reset_to_basic 4 "logs/mc_rl_sample/${tr}__vsim.npz" --record_dof
done
echo "== done -> notebooks/mini_cheetah_ref.py =="
