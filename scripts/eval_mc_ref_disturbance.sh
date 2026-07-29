#!/usr/bin/env bash
# Paper-style Mini Cheetah velocity-impulse failure-rate sweep.
#
# Usage:
#   scripts/eval_mc_ref_disturbance.sh mujoco cuda:0 CHECKPOINT LABEL
#   scripts/eval_mc_ref_disturbance.sh vsim cuda:0 CHECKPOINT LABEL
#
# Defaults mirror Table I where practical: 3 m/s forward command, 36 planar
# directions, and 50 impulse times at 0.01 s spacing (1800 trials/magnitude).
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
EVAL_ENVS=${EVAL_ENVS:-1800}
T_END=${T_END:-8.0}
IMPULSE_START=${IMPULSE_START:-5.0}
IMPULSE_STAGGER=${IMPULSE_STAGGER:-0.5}
IMPULSE_DIRECTIONS=${IMPULSE_DIRECTIONS:-36}
MAGNITUDES=${MAGNITUDES:-"1.5 2.0 2.5 3.0 3.5"}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-logs/mc_ref_disturbance}

if [[ "$BACKEND" != "mujoco" && "$BACKEND" != "vsim" ]]; then
  echo "backend must be mujoco or vsim"
  exit 2
fi
expected_envs=$(
  uv run python -c \
    "print(round(float('$IMPULSE_STAGGER') * 100) * $IMPULSE_DIRECTIONS)"
)
if (( EVAL_ENVS != expected_envs )); then
  echo "EVAL_ENVS=$EVAL_ENVS; $expected_envs gives one trial per direction/time pair"
fi

RUNNER=(uv run)
if [[ "$BACKEND" == "vsim" ]]; then
  RUNNER=(uv run --env-file .env.vsim)
fi

mkdir -p "$OUT_DIR"
for magnitude in $MAGNITUDES; do
  suffix=${magnitude//./p}
  "${RUNNER[@]}" scripts/eval_policy.py \
    --task "$TASK" \
    --ckpt "$CHECKPOINT" \
    --train_label "$LABEL" \
    --eval_backend "$BACKEND" \
    --eval_device "$DEVICE" \
    --eval_label "$LABEL-impulse-$magnitude" \
    --num_envs "$EVAL_ENVS" \
    --t_end "$T_END" \
    --seed "$SEED" \
    --reset_mode reset_to_basic \
    --command_profile forward_3p0 \
    --settling_time 0.5 \
    --velocity_impulse "$magnitude" \
    --impulse_start_time "$IMPULSE_START" \
    --impulse_stagger_time "$IMPULSE_STAGGER" \
    --impulse_directions "$IMPULSE_DIRECTIONS" \
    --out "$OUT_DIR/${LABEL}__impulse_${suffix}.npz"
done

echo "wrote disturbance sweep under $OUT_DIR"
