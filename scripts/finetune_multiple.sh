#!/bin/bash
source /home/lmolnar/miniconda3/etc/profile.d/conda.sh

# Args
LOAD_RUN="Jul25_16-06-36_LinkedIPG_50Hz_nu02_v08"
CHECKPOINT=1000
N_RUNS=10
INTER_NU=0.9  # can be fixed or adaptive
EVAL=false  # no finetuning, just evaluate without exploration (set in RS)

# Set directories
QGYM_DIR="/home/lmolnar/workspace/QGym"
RS_DIR="/home/lmolnar/workspace/Robot-Software"

QGYM_LOG_DIR="${QGYM_DIR}/logs/mini_cheetah_ref/${LOAD_RUN}"

# Change default folder name and copy to RS
mv ${QGYM_LOG_DIR}/exported ${QGYM_LOG_DIR}/exported_${CHECKPOINT}
cp ${QGYM_LOG_DIR}/exported_${CHECKPOINT}/* ${RS_DIR}/config/systems/quadruped/controllers/policies

for i in $(seq 1 $N_RUNS)
do
    # Store logs in files labeled by checkpoint
    LCM_FILE=${RS_DIR}/logging/lcm_logs/${CHECKPOINT}
    MAT_FILE=${RS_DIR}/logging/matlab_logs/${CHECKPOINT}.mat
    rm ${LCM_FILE}
    rm ${MAT_FILE}

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
    # INTER_NU=$(echo "0.05 * $i" | bc)  # adaptive
    if [ "$EVAL" = false ] ; then
        conda deactivate
        conda activate qgym
        python ${QGYM_DIR}/scripts/finetune.py --task=mini_cheetah_finetune --headless --load_run=${LOAD_RUN} --checkpoint=${CHECKPOINT} --inter_nu=${INTER_NU}
    fi

    # Copy policy to RS
    CHECKPOINT=$((CHECKPOINT + 1))
    cp ${QGYM_LOG_DIR}/exported_${CHECKPOINT}/* ${RS_DIR}/config/systems/quadruped/controllers/policies
done