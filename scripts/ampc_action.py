import pickle
import math
import random
import os
import time
from gym import LEGGED_GYM_ROOT_DIR

import torch
from torch import nn  # noqa F401
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


from tqdm import tqdm

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    create_uniform_generator,
)
from utils import (
    DEVICE,
    find_flatline_index,
    is_lipschitz_continuous,
    max_pairwise_distance,
    calc_neighborhood_radius,
    process_batch,
    in_box,
    grid_search_u,
)
from critic_params_ampc import critic_params
from tensordict import TensorDict

from learning.modules.lqrc.plotting import (
    plot_learning_progress,
    plot_binned_errors_ampc,
    plot_variable_lr,
    plot_eigenval_hist,
    plot_multiple_critics_w_data,
)
from learning.modules.lqrc.utils import get_latent_matrix

# ampc imports and setup
from collections import defaultdict
import functools
from learning.modules.ampc.wheelbot import (
    load_dataset,
    N,
    nu,
    nx,
    ntheta,
    x_max,
    x_min,
    WheelbotBatchSimulation,
    plot_wheelbot,
    plot_wheelbot_all_inits,
    plot_time_to_failure,
    plot_u_diff,
    WheelbotOneStepMPC,
    WheelbotSimulation,
)

ONE_STEP_MPC = False
GRID_SEARCH = False


# make dir for saving this run's results
time_str = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# set up critics
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

critic_names = [
    # "OuterProduct",
    # "OuterProductLatent",
    # "PDCholeskyInput",
    "CholeskyLatent",
    "DenseSpectralLatent",
    "Critic",
]
print("Loading data")
# load data
# with open(f"{LEGGED_GYM_ROOT_DIR}/learning/modules/lqrc/dataset.pkl", "rb") as f:
#     data = pickle.load(f)
# with open(f"{LEGGED_GYM_ROOT_DIR}/learning/modules/lqrc/v_dataset.pkl", "rb") as f:
#     data = pickle.load(f)
with open(
    f"{LEGGED_GYM_ROOT_DIR}/learning/modules/ampc/simple_unicycle/casadi/100_unicycle_dataset.pkl",
    "rb",
) as f:
    data = pickle.load(f)
x0 = np.array(data["x0"])  # (3478114, 10)
cost = np.array(data["cost"])  # (3478114,)
# optimal_u = np.array(data["U"])  # (3478114, 2)

print(
    f"Raw data mean {x0.mean(axis=0)} \n median {np.median(x0, axis=0)} \n max {x0.max(axis=0)} \n min {x0.min(axis=0)}"
)

# remove top 1% of cost values and corresponding states
num_to_remove = math.ceil(0.01 * len(cost))
top_indices = np.argsort(cost)[-num_to_remove:]
mask = np.ones(len(cost), dtype=bool)
mask[top_indices] = False
x0 = x0[mask]
cost = cost[mask]
# optimal_u = optimal_u[mask]
n_samples = x0.shape[0]

# min max normalization to put state and cost on [0, 1]
x0_min = x0.min(axis=0)
x0_max = x0.max(axis=0)
x0 = (x0 - x0_min) / (x0_max - x0_min)
cost_min = cost.min()
cost_max = cost.max()
cost = (cost - cost_min) / (cost_max - cost_min)

print(
    f"Normalized data mean {x0.mean(axis=0)} \n median {np.median(x0, axis=0)} \n max {x0.max(axis=0)} \n min {x0.min(axis=0)}"
)

# make batch for one step MPC eval before adding non-fs synthetic data points
batch_terminal_eval = 10
mpc_eval_ix = random.sample(list(range(x0.shape[0])), batch_terminal_eval)
mpc_mask = np.zeros(len(cost), dtype=bool)
mpc_mask[mpc_eval_ix] = True
mpc_eval_x0 = x0[mpc_mask]
mpc_eval_cost = cost[mpc_mask]
# mpc_eval_optimal_u = optimal_u[mpc_mask]
# ensure this validation batch is not seen in training data
x0 = x0[~mpc_mask]
cost = cost[~mpc_mask]
# optimal_u = optimal_u[~mpc_mask]


# add in non-fs synthetic data points
print("Building KDTree")
# Build the KDTree to get pairwise point distances
start = time.time()
tree = KDTree(x0)
d_max = max_pairwise_distance(tree)

midpt = np.mean(x0, axis=0)
n_synthetic = 0.2 * n_samples
# random_x0 = np.concatenate(
#     (
#         np.random.uniform(
#             low=midpt[0] - d_max,
#             high=midpt[0] + d_max,
#             size=(int(n_synthetic // 2), x0.shape[1]),
#         ),
#         np.random.uniform(
#             low=midpt[1] - d_max,
#             high=midpt[1] + d_max,
#             size=(int(n_synthetic // 2), x0.shape[1]),
#         ),
#     )
# )
random_x0 = np.concatenate(
    (
        np.random.uniform(
            low=-0.5,
            high=1.5,
            size=(int(n_synthetic // 2), x0.shape[1]),
        ),
        np.random.uniform(
            low=-0.5,
            high=1.5,
            size=(int(n_synthetic // 2), x0.shape[1]),
        ),
    )
)

radius = calc_neighborhood_radius(tree, x0)
print("Creating non feasible state data points")
# filter random_x0 in batches
chunk_size = 100
x0_non_fs_list = []
for i in range(0, len(random_x0), chunk_size):
    batch = random_x0[i : i + chunk_size]
    Y_filtered_batch = process_batch(batch, tree, radius)
    if len(Y_filtered_batch) > 0:
        x0_non_fs_list.append(Y_filtered_batch)
print("")

# concatenate all filtered batches and create matching cost array
x0_non_fs = np.concatenate(x0_non_fs_list) if x0_non_fs_list else None
cost_non_fs = np.ones((x0_non_fs.shape[0],))
# union non feasible and feasible states
x0 = np.concatenate((x0, x0_non_fs))
cost = np.concatenate((cost, cost_non_fs))

# hack to see data dist
plt.hist(cost, bins=100)
plt.savefig(os.path.join(save_path, "data_dist.png"), dpi=300)


print(
    "Dataset Lipschitz continuity with threshold of 1e-6",
    is_lipschitz_continuous(
        x0, cost, threshold=100
    ),  # reduced threshold due to 0 to 1 normalization
)

# ampc set up
K_W = np.array([400e-3, 40e-3, 3e-3, 3e-3])
K_R = np.array([1.3e0, 1.6e-1, 0.8e-04, 4e-04])
K = np.array(
    [
        [0, 0, K_W[0], 0, 0, K_W[1], K_W[2], 0, K_W[3], 0],
        [0, K_R[0], 0, 0, K_R[1], 0, 0, K_R[2], 0, K_R[3]],
    ]
)

terminal_set_fcn = functools.partial(in_box, x_min=x_min, x_max=x_max)
onestepmpc = WheelbotOneStepMPC()
check_terminal_nth_epoch = 100  # 500
eval_frac_in_terminal = defaultdict(list)

# turn numpy arrays to torch before training
x0 = torch.from_numpy(x0).float().to(DEVICE)
cost = torch.from_numpy(cost).float().to(DEVICE)

print(
    "cost mean", cost.mean(), "cost median", cost.median(), "cost std dev", cost.std()
)

# set up constants
total_data = x0.shape[0]
n_dims = x0.shape[1]
graphing_data = {
    data_name: {name: {} for name in critic_names}
    for data_name in [
        "critic_obs",
        "values",
        "cost",
        "error",
    ]
}
test_error = {name: [] for name in critic_names}
lr_history = {name: [] for name in critic_names}

# set up training
max_gradient_steps = 500  # 1000
batch_size = 512
n_training_data = int(0.6 * total_data)
n_validation_data = total_data - n_training_data
print(f"training data: {n_training_data}, validation data: {n_validation_data}")
rand_perm = torch.randperm(total_data)
train_idx = rand_perm[0:n_training_data]
test_idx = rand_perm[n_training_data:]

data = TensorDict(
    {"critic_obs": x0.unsqueeze(dim=0), "cost": cost.unsqueeze(dim=0)},
    batch_size=(1, total_data),
    device=DEVICE,
)

t_to_fail = {name: [] for name in critic_names}
u_diff = {name: None for name in critic_names}
u_diff_per_batch = np.zeros((max_gradient_steps // check_terminal_nth_epoch, 2))

standard_offset = 0
for ix, name in enumerate(critic_names):
    torch.cuda.empty_cache()
    print("")
    params = critic_params[name]
    if "critic_name" in params.keys():
        params.update(critic_params[params["critic_name"]])
    params["num_obs"] = n_dims

    critic_class = globals()[name]
    critic = critic_class(**params).to(DEVICE)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=0.01)
    # critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        critic_optimizer, mode="min", factor=0.5, patience=100, threshold=1e-5
    )

    # train new critic
    mean_value_loss = 0
    counter = 0

    generator = create_uniform_generator(
        data[:1, train_idx],
        batch_size,
        max_gradient_steps=max_gradient_steps,
    )

    if GRID_SEARCH:
        model = WheelbotSimulation()

    for batch in generator:
        # print offset to check it's working as intended
        if counter == 0:
            if ix == 0:
                standard_offset = batch["cost"].mean()
            print(f"{name} value offset before mean assigning", critic.value_offset)
            with torch.no_grad():
                critic.value_offset.copy_(standard_offset)
            print(f"{name} value offset after mean assigning", critic.value_offset)
        # extract matrix transform for latent if applicable
        if "Latent" in name:
            latent_weight = get_latent_matrix(
                batch["critic_obs"].shape, critic.latent_NN, device=DEVICE
            )
            latent_weight = latent_weight.cpu().detach().numpy()
            # latent_bias = latent_bias.cpu().detach().numpy()
            # print(latent_bias)

        # calculate loss and optimize
        value_loss = critic.loss_fn(
            batch["critic_obs"].squeeze(), batch["cost"].squeeze()
        )
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
        lr_scheduler.step(value_loss)
        counter += 1
        # pointwise prediction test error
        with torch.no_grad():
            actual_error = (
                (
                    data["cost"][0, test_idx].squeeze()
                    - critic.evaluate(data["critic_obs"][0, test_idx])
                ).pow(2)
            ).to("cpu")
        test_error[name].append(actual_error.detach().mean().numpy())
        lr_history[name].append(lr_scheduler.get_last_lr()[0])
        # perform closed-loop simulation as test error metric
        if (
            ONE_STEP_MPC
            and name != "Critic"
            and (counter % check_terminal_nth_epoch == 0)
        ):
            # N_eval = int(3 * N)
            N_eval = 30
            X_sim_ = (x0_max - x0_min) * mpc_eval_x0 + x0_min
            X_sim_cl_ = np.repeat(X_sim_[:, :, np.newaxis], N_eval + 1, axis=2)
            U_sim_cl_ = np.zeros((np.shape(X_sim_cl_)[0], nu, N_eval))

            epoch_ix = (counter // check_terminal_nth_epoch) - 1
            u_diff_per_batch[epoch_ix, 0] = counter
            eigen_vals = []
            for b in range(batch_terminal_eval):
                onestepmpc.reset()
                for k in range(N_eval):
                    if GRID_SEARCH:
                        u, xnext = grid_search_u(
                            critic,
                            (X_sim_cl_[b, :, k] - x0_min) / (x0_max - x0_min),
                            model,
                            np.array([-0.5, -0.5]),
                            np.array([0.5, 0.5]),
                            0.05,
                        )
                    else:
                        # A = cost_scale*np.copy(compute_single_A_filtered(model, X_sim_cl_[b,:,k]))
                        prediction = critic.evaluate(
                            torch.from_numpy(
                                (X_sim_cl_[b, :, k] - x0_min) / (x0_max - x0_min)
                            )
                            .float()
                            .to(DEVICE)
                            .unsqueeze(0),
                            return_all=True,
                        )
                        A = prediction["A"].squeeze().cpu().detach().numpy()
                        try:
                            eigen_vals.extend(
                                [eig.real for eig in np.linalg.eigvals(A).tolist()]
                            )
                        except:
                            print(
                                "NaNs stopped eigenvalue computation, adding NaNs to eigenvalue list"
                            )
                            eigen_vals.extend([float("nan") for _ in range(A.shape[0])])
                        if "Latent" in name:
                            A = latent_weight.T @ A @ latent_weight
                        # denormalize A before sending it to one step MPC
                        # if np.isnan(np.amax(A)):
                        #     print(f"Max elem of A is nan at batch {b} and step {k}")
                        # if k == N_eval - 1:
                        #     print("Max elem of A at final step", np.amax(A))
                        A = (cost_max - cost_min) * A + cost_min
                        # print(f"k {k} max A after denorm", np.amax(A))
                        u, xnext, _ = onestepmpc.run(X_sim_cl_[b, :, k], A)
                    u_applied = K @ X_sim_cl_[b, :, k] + u
                    # if k == 0:
                    #     print(f"{np.count_nonzero(A)=}, {np.linalg.eigvals(A)=}, {A=}")
                    # print("applied torque", u_applied, "at step", k)
                    if k == 0:
                        row_ix = (
                            np.asarray(
                                mpc_eval_x0
                                == (X_sim_cl_[b, :, k] - x0_min) / (x0_max - x0_min)
                            )
                            .all(axis=1)
                            .nonzero()
                        )
                        diff = u - mpc_eval_optimal_u[row_ix]
                        avg_diff = np.sqrt(np.sum(np.square(diff)))
                        u_diff_per_batch[epoch_ix, 1] += avg_diff
                    U_sim_cl_[b, :, k] = np.copy(u)
                    X_sim_cl_[b, :, k + 1] = np.copy(xnext)
            u_diff_per_batch[epoch_ix, 1] = (
                u_diff_per_batch[epoch_ix, 1] / batch_terminal_eval
            )
            u_diff[name] = u_diff_per_batch

            # compute fraction of states in terminal set
            frac_in_terminal = np.mean(
                np.vectorize(terminal_set_fcn)(
                    X_sim_cl_[:batch_terminal_eval, :6, N_eval]
                )
            )
            eval_frac_in_terminal[name].append(frac_in_terminal)

            plot_traj_outdir = f"{save_path}/traj_graphs"
            if not os.path.exists(plot_traj_outdir):
                os.makedirs(plot_traj_outdir)
            random_indices = random.sample(list(range(batch_terminal_eval)), 5)
            plot_wheelbot_all_inits(
                X_sim_cl_[random_indices],
                U_sim_cl_[random_indices],
                plot_labels=["cl_sim"],
                filename=f"{plot_traj_outdir}/{name}_{counter}_plot_traj",
                show=False,
            )
            t_to_fail_samples = []
            for ix in random_indices:
                fail_ix = find_flatline_index(
                    X_sim_cl_[ix, 0]
                )  # arbitrarily picking first state for now
                t_to_fail_samples.append(fail_ix)
            t_to_fail_samples = np.array(t_to_fail_samples)
            t_to_fail[name].append(
                [counter, t_to_fail_samples.mean(), t_to_fail_samples.std()]
            )
            print(
                "Time to failure mean",
                t_to_fail_samples.mean(),
                "and std",
                t_to_fail_samples.std(),
            )
            if not GRID_SEARCH:
                plot_eigenval_hist(
                    eigen_vals,
                    fn=f"{plot_traj_outdir}/{name}_{counter}_plot_eigenval",
                    title="Histogram of Eigenvalues Across Batch",
                )
    print(f"{name} average error: ", actual_error.mean().item())
    print(f"{name} max error: ", actual_error.max().item())

    with torch.no_grad():
        graphing_data["error"][name] = actual_error
        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = critic.evaluate(data[0, :]["critic_obs"])
        graphing_data["cost"][name] = data[0, :]["cost"]

    print(
        f"{name} mean",
        graphing_data["values"][name].mean().item(),
        "median",
        graphing_data["values"][name].median().item(),
        "std dev",
        graphing_data["values"][name].std().item(),
    )
for key in critic_names:
    t_to_fail[key] = np.array(t_to_fail[key])

plot_learning_progress(
    test_error,
    "Pointwise Error on Test Set \n (Comparison of Normed Vals Used in Supervised Training)",
    fn=f"{save_path}/test_error",
)
plot_variable_lr(lr_history, f"{save_path}/lr_history")

plot_binned_errors_ampc(
    graphing_data,
    save_path + "/ampc",
    title_add_on=f"Value Function at {max_gradient_steps} Epochs, With Offset",
    lb=0.0,
    ub=0.75,
    step=0.025,
    tick_step=10,
)

graphing_data["critic_obs"]["Unicycle"] = x0
graphing_data["values"]["Unicycle"] = cost
graphing_data["cost"]["Unicycle"] = cost
graphing_data["error"]["Unicycle"] = torch.zeros_like(cost)

# make data square number
sqrt_num = math.isqrt(x0.shape[0])
sq_num = sqrt_num**2
for critic in graphing_data["critic_obs"].keys():
    graphing_data["critic_obs"][critic] = graphing_data["critic_obs"][critic][
        :sq_num, :
    ]
    graphing_data["values"][critic] = graphing_data["values"][critic][:sq_num]
    graphing_data["cost"][critic] = graphing_data["cost"][critic][:sq_num]

plot_multiple_critics_w_data(
    graphing_data["critic_obs"],
    graphing_data["values"],
    graphing_data["cost"],
    title=f"Learning Unicycle MPC",
    display_names={
        "Unicycle": "Unicycle Ground Truth",
        "OuterProduct": "Outer Product",
        "CholeskyLatent": "Cholesky Latent",
        "DenseSpectralLatent": "Spectral Latent",
        "Critic": "Critic",
    },
    grid_size=sqrt_num,
    fn=save_path + f"/{len(critic_names)}",
    data=data[0, train_idx]["critic_obs"],
    extension="png",
    task="Unicycle",
    log_norm=False,
)

if ONE_STEP_MPC:
    plot_time_to_failure(t_to_fail, f"{save_path}/time_to_fail")
    plot_u_diff(u_diff, f"{save_path}/u_error")
