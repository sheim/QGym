#!/usr/bin/env bash
# mini_cheetah_ref cross-engine parity benchmark (Phase 4).
#
# Trains the reference-gait policy on each backend, then builds two eval sets:
#   * transfer matrix  (reset_to_range, aggregate reward + survival)
#   * DOF samples      (reset_to_basic, identical IC -> cross-backend DOF RMS)
# Inspect with notebooks/mini_cheetah_ref.py.
#
# vsim needs its env file; run this from the repo root. CPU training is the
# long pole (contact-rich, single-threaded) and runs last.
#
# PPO derives its temporal rollout horizon as batch_size / num_envs. Keep the
# same environment count and batch size on every backend: changing only the GPU
# environment count changes GAE from a temporal rollout into independent
# one-step samples and is not a parity experiment.
set -euo pipefail

ITERS=${ITERS:-250}
SEED=${SEED:-7}
TRAIN_ENVS=${TRAIN_ENVS:-256}
BATCH_SIZE=${BATCH_SIZE:-4096}
EVAL_ENVS=${EVAL_ENVS:-256}   # transfer matrix batch (cpu-eval is slow)
T_END=${T_END:-5.0}
if (( BATCH_SIZE < TRAIN_ENVS || BATCH_SIZE % TRAIN_ENVS != 0 )); then
  echo "BATCH_SIZE must be an integer multiple of TRAIN_ENVS"
  exit 2
fi

train() {  # backend device envs experiment
  local backend=$1
  local device=$2
  local envs=$3
  local experiment=$4
  local -a runner=(uv run)
  local -a backend_args=()
  if [[ "$backend" == "vsim" ]]; then
    runner+=(--env-file .env.vsim)
    backend_args=(--backend vsim)
  fi
  "${runner[@]}" scripts/train_mujoco.py \
    --task mini_cheetah_ref \
    --device "$device" \
    --num_envs "$envs" \
    --batch_size "$BATCH_SIZE" \
    --max_iterations "$ITERS" \
    --seed "$SEED" \
    --headless \
    --disable_wandb \
    --experiment_name "$experiment" \
    "${backend_args[@]}"
}

echo "== training: $TRAIN_ENVS envs, $((BATCH_SIZE / TRAIN_ENVS)) steps/env =="
train mujoco cuda:0 "$TRAIN_ENVS" mini_cheetah_ref_warp
train vsim   cuda:0 "$TRAIN_ENVS" mini_cheetah_ref_vsim
train mujoco cpu    "$TRAIN_ENVS" mini_cheetah_ref_cpu

# eval one (train_label, eval_backend, eval_device, eval_label, mode, out, extra)
eval_cell() {
  local train_label=$1
  local eval_backend=$2
  local eval_device=$3
  local eval_label=$4
  local reset_mode=$5
  local num_envs=$6
  local output=$7
  local -a extra=("${@:8}")
  local -a runner=(uv run)
  if [[ "$eval_backend" == "vsim" ]]; then
    runner+=(--env-file .env.vsim)
  fi
  "${runner[@]}" scripts/eval_policy.py \
    --task mini_cheetah_ref \
    --ckpt "logs/mini_cheetah_ref_$train_label" \
    --train_label "$train_label" \
    --eval_backend "$eval_backend" \
    --eval_device "$eval_device" \
    --eval_label "$eval_label" \
    --num_envs "$num_envs" \
    --t_end "$T_END" \
    --reset_mode "$reset_mode" \
    --out "$output" \
    "${extra[@]}"
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
