#!/bin/bash
source /home/lmolnar/miniconda3/etc/profile.d/conda.sh

# Args
LOAD_RUN="Jul13_01-49-59_PPO32_S16"
CHECKPOINT=700
N_RUNS=5

# Set directories
QGYM_DIR="/home/lmolnar/workspace/QGym"
RS_DIR="/home/lmolnar/workspace/Robot-Software"

QGYM_LOG_DIR="${QGYM_DIR}/logs/mini_cheetah_ref/${LOAD_RUN}"

# Change default folder name and copy to RS
mv ${QGYM_LOG_DIR}/exported ${QGYM_LOG_DIR}/exported_${CHECKPOINT}
cp ${QGYM_LOG_DIR}/exported_${CHECKPOINT}/* ${RS_DIR}/config/systems/quadruped/controllers/policies

for i in $(seq 1 $N_RUNS)
do
    # Store LCM logs in file labeled with checkpoint
    FILE=${RS_DIR}/logging/lcm_logs/${CHECKPOINT}
    if [ -f "$FILE" ]; then
        rm "$FILE"
    fi

    # Run logging script in background
    ${RS_DIR}/logging/scripts/run_lcm_logger.sh ${CHECKPOINT} &
    PID1=$!

    # Run quadruped script and cancel logging script when done
    ${RS_DIR}/build/bin/run_quadruped m s
    kill $PID1

    # Convert logs to .mat and copy to QGym
    conda deactivate
    conda activate robot-sw
    ${RS_DIR}/logging/scripts/sim_data_convert.sh ${CHECKPOINT}
    cp ${RS_DIR}/logging/matlab_logs/${CHECKPOINT}.mat ${QGYM_LOG_DIR}

    # Finetune in QGym
    conda deactivate
    conda activate qgym
    python ${QGYM_DIR}/scripts/finetune.py --task=mini_cheetah_finetune --headless --load_run=${LOAD_RUN} --checkpoint=${CHECKPOINT}

    # Copy policy to RS
    CHECKPOINT=$((CHECKPOINT + 1))
    cp ${QGYM_LOG_DIR}/exported_${CHECKPOINT}/* ${RS_DIR}/config/systems/quadruped/controllers/policies
done