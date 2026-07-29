#!/usr/bin/env bash
# Evaluate one Mini Cheetah reference policy in its training backend.
#
# Usage:
#   scripts/eval_mc_ref_hardware.sh mujoco cpu CHECKPOINT LABEL
#   scripts/eval_mc_ref_hardware.sh vsim cuda:0 CHECKPOINT LABEL
#
# Environment overrides:
#   TASK=mini_cheetah_ref EVAL_ENVS=288 T_END=5.0 SETTLING_TIME=0.5 SEED=0
set -euo pipefail

if (( $# < 4 )); then
  echo "usage: $0 <mujoco|vsim> <device> <checkpoint> <label>"
  exit 2
fi

BACKEND=$1
DEVICE=$2
CHECKPOINT=$3
LABEL=$4
TASK=${TASK:-mini_cheetah_ref}
EVAL_ENVS=${EVAL_ENVS:-288}
T_END=${T_END:-5.0}
SETTLING_TIME=${SETTLING_TIME:-0.5}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-logs/mc_ref_hardware}

if [[ "$BACKEND" != "mujoco" && "$BACKEND" != "vsim" ]]; then
  echo "backend must be mujoco or vsim"
  exit 2
fi
if (( EVAL_ENVS % 9 != 0 )); then
  echo "EVAL_ENVS must be divisible by 9 (one balanced block per command case)"
  exit 2
fi

RUNNER=(uv run)
if [[ "$BACKEND" == "vsim" ]]; then
  RUNNER=(uv run --env-file .env.vsim)
fi

mkdir -p "$OUT_DIR"

evaluate() {
  local mode=$1
  local suffix=$2
  "${RUNNER[@]}" scripts/eval_policy.py \
    --task "$TASK" \
    --ckpt "$CHECKPOINT" \
    --train_label "$LABEL" \
    --eval_backend "$BACKEND" \
    --eval_device "$DEVICE" \
    --eval_label "$LABEL-$suffix" \
    --num_envs "$EVAL_ENVS" \
    --t_end "$T_END" \
    --seed "$SEED" \
    --reset_mode "$mode" \
    --command_profile hardware \
    --settling_time "$SETTLING_TIME" \
    --out "$OUT_DIR/${LABEL}__${suffix}.npz"
}

# Nominal isolates steady tracking and gait quality. Robust adds randomized
# task initial conditions; domain randomization remains whatever the config
# declares, and pushes stay disabled for both.
evaluate reset_to_basic nominal
evaluate reset_to_range robust

echo "wrote native scorecards under $OUT_DIR"
