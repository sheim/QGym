import os
from learning import LEGGED_GYM_LQRC_DIR
from learning.modules.lqrc.plotting import plot_training_data_dist

read_path = "CholeskyPlusConst_fixed_mean_var_all_obs.npy"
output_path = "CholeskyPlusConst_fixed_mean_var_data_dist_1_redo.png"
save_path = os.path.join(LEGGED_GYM_LQRC_DIR, "logs")

if __name__ == "__main__":
    plot_training_data_dist(
        LEGGED_GYM_LQRC_DIR + f"/{read_path}", save_path + f"/{output_path}"
    )
