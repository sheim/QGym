import pickle
import math
import time
from gym import LEGGED_GYM_ROOT_DIR

import torch
from torch import nn  # noqa F401
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree

from tqdm import tqdm

from learning.modules.critic import Critic  # noqa F401
from learning.modules.lqrc import *  # noqa F401
from learning.utils import (
    create_uniform_generator,
)
from utils import DEVICE
from critic_params_ampc import critic_params
from tensordict import TensorDict

from learning.modules.lqrc.plotting import (
    plot_learning_progress,
    plot_binned_errors_ampc
)
from learning.modules.utils.neural_net import export_network

# make dir for saving this run's results
time_str = time.strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "offline_critics_graph", time_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# set up critics
for critic_param in critic_params.values():
    critic_param["device"] = DEVICE

critic_names = [
    # "Critic",
    # "OuterProduct",
    # "PDCholeskyInput",
    "CholeskyLatent",
    # "DenseSpectralLatent",
]
print("Loading data")
# load data
with open(f"{LEGGED_GYM_ROOT_DIR}/learning/modules/lqrc/dataset.pkl", "rb") as f:
    data = pickle.load(f)
x0 = np.array(data["x0"]) # (3478114, 10)
cost = np.array(data["cost"]) # (3478114,)

# remove top 1% of cost values and corresponding states
num_to_remove = int(0.01 * len(cost))
top_indices = np.argsort(cost)[-num_to_remove:]
mask = np.ones(len(cost), dtype=bool)
mask[top_indices] = False
x0 = x0[mask]
cost = cost[mask]

# min max normalization to put state and cost on [0, 1]
# x0_min = x0.min(axis=0)
# x0_max = x0.max(axis=0)
# x0 = (x0 - x0_min) / (x0_max - x0_min)
# cost_min = cost.min()
# cost_max = cost.max()
# cost = (cost - cost_min) / (cost_max - cost_min)

d_max = 0

print("Building KDTree")
# Build the KDTree to get pairwise point distances
start = time.time()
tree = KDTree(x0)

# random_x0 = np.random.uniform(low=-0.2, high=1.2, size=(10000, x0.shape[1]))
random_x0 = np.random.uniform(low=cost.min(), high=cost.max(), size=(10000, x0.shape[1]))

def process_batch(batch):
    indices = tree.query_ball_point(batch, d_max)
    mask = np.array([len(idx) == 0 for idx in indices])
    return batch[mask]

print("Creating non feasible state data points")
# filter random_x0 in batches
chunk_size = 3
x0_non_fs_list = []
for i in range(0, len(random_x0), chunk_size):
    batch = random_x0[i:i+chunk_size]
    Y_filtered_batch = process_batch(batch)
    if len(Y_filtered_batch) > 0:
        x0_non_fs_list.append(Y_filtered_batch)
print("")

# concatenate all filtered batches and create matching cost array
x0_non_fs = np.concatenate(x0_non_fs_list) if x0_non_fs_list else None
cost_non_fs = np.ones((x0_non_fs.shape[0],))
# union non feasible and feasible states
x0 = np.concatenate((x0, x0_non_fs))
cost = np.concatenate((cost, cost_non_fs))

# #hack to see data dist
# import matplotlib.pyplot as plt
# plt.hist(cost, bins=100)
# plt.savefig(os.path.join(save_path, "data_dist.png"), dpi=300)

# ampc imports and setup
from collections import defaultdict
import functools
from learning.modules.ampc.wheelbot import load_dataset, N, nu, nx, ntheta, x_max, x_min, WheelbotBatchSimulation, plot_wheelbot, WheelbotOneStepMPC
def in_box(x, x_min, x_max):
    return np.all(np.logical_and(x >= x_min,  x <= x_max ))
terminal_set_fcn = functools.partial(in_box, x_min=x_min, x_max=x_max)
onestepmpc = WheelbotOneStepMPC()
check_terminal_nth_epoch = 100
eval_frac_in_terminal = defaultdict(list)

# turn numpy arrays to torch before training
x0 = torch.from_numpy(x0).float().to(DEVICE)
cost = torch.from_numpy(cost).float().to(DEVICE)

print("cost mean", cost.mean(), "cost median", cost.median(), "cost std dev", cost.std())

# set up constants
total_data = x0.shape[0]
n_dims = x0.shape[1]
graphing_data = {data_name: {name: {} for name in critic_names}
            for data_name in [
                "critic_obs",
                "values",
                "cost",
                "error",
            ]}
test_error = {name: [] for name in critic_names}

# set up training
max_gradient_steps = 4000
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
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # train new critic
    mean_value_loss = 0
    counter = 0

    generator = create_uniform_generator(
        data[:1, train_idx],
        batch_size,
        max_gradient_steps=max_gradient_steps,
    )
    for batch in generator:
        if counter == 0:
            if ix == 0:
                standard_offset = batch["cost"].mean()
            print(f"{name} value offset before mean assigning", critic.value_offset)
            with torch.no_grad():
                critic.value_offset.copy_(standard_offset)
            print(f"{name} value offset after mean assigning", critic.value_offset)
        value_loss = critic.loss_fn(
            batch["critic_obs"].squeeze(), batch["cost"].squeeze()
        )

        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
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
        # perform closed-loop simulation as test error metric
        if name != "Critic" and (counter % check_terminal_nth_epoch == 0):
            batch_terminal_eval = batch["critic_obs"].shape[0]  #100
            N_eval = int(3*N)
            # X_sim_ = np.copy(np.asarray(jax.numpy.copy(X_)))
            X_sim_ = batch["critic_obs"].cpu().detach().numpy()
            X_sim_cl_ = np.repeat(X_sim_[:,:,np.newaxis], N_eval+1, axis=2)
            U_sim_cl_ = np.zeros((np.shape(X_sim_cl_)[0], nu, N_eval))

            print("X sim shape", X_sim_.shape)
            print("X sim cl shape", X_sim_cl_.shape)
            print("U sim cl shape", U_sim_cl_.shape)
            
            for b in range(batch_terminal_eval):
                onestepmpc.reset()
                for k in range(N_eval):
                    print("k", k)
                    # A = cost_scale*np.copy(compute_single_A_filtered(model, X_sim_cl_[b,:,k]))
                    prediction  = critic.evaluate(torch.from_numpy(X_sim_cl_[b,:,k]).to(DEVICE).unsqueeze(0), return_all=True)
                    A = prediction["A"].squeeze().cpu().detach().numpy()
                    # if k == 0:
                    #     print(f"{np.count_nonzero(A)=}, {np.linalg.eigvals(A)=}, {A=}")
                    u, xnext, status = onestepmpc.run(X_sim_cl_[b,:,k], A)
                    U_sim_cl_[b,:,k] = np.copy(u)
                    X_sim_cl_[b,:,k+1] = np.copy(xnext)
            
            # compute fraction of states in terminal set
            frac_in_terminal = np.mean(np.vectorize(terminal_set_fcn)(X_sim_cl_[:batch_terminal_eval,:6,N_eval]))
            eval_frac_in_terminal[name].append(frac_in_terminal)
            
            for idx in range(min(batch_terminal_eval, 100)):
                labels = ["cl_sim"]
                plot_outdir = f"plots/value_{time_str}/{counter}"
                if not os.path.exists(plot_outdir):
                    os.makedirs(plot_outdir)
                plot_wheelbot([X_sim_cl_[idx]],[U_sim_cl_[idx]],labels,filename=f"{plot_outdir}/plot_{idx}", show=False)

    print(f"{name} average error: ", actual_error.mean().item())
    print(f"{name} max error: ", actual_error.max().item())

    with torch.no_grad():
        graphing_data["error"][name] = actual_error
        graphing_data["critic_obs"][name] = data[0, :]["critic_obs"]
        graphing_data["values"][name] = critic.evaluate(
            data[0, :]["critic_obs"]
        )
        graphing_data["cost"][name] = data[0, :]["cost"]
    # print("exporting", name)
    # export_network(critic, name, save_path, n_dims)
    print(f"{name} mean", graphing_data["values"][name].mean(), "median", graphing_data["values"][name].median(), "std dev", graphing_data["values"][name].std())

plot_learning_progress(test_error, "AMPC VF Test Error", fn=f"{save_path}/test_error")


# plot_binned_errors_ampc(
#     graphing_data,
#     save_path + "/ampc",
#     title_add_on=f"Value Function for AMPC",
# )
