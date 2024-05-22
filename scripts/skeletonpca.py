# dataset creation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def prepare_clustered_data(mean, std, data, clustersize):
    s = np.vstack(
        [
            np.random.normal(mean[0], std[0], size=(1, clustersize)),
            np.random.normal(mean[1], std[1], size=(1, clustersize)),
            np.random.normal(mean[2], std[2], size=(1, clustersize)),
        ]
    )
    data = np.hstack([data, s])
    return data


def normalize(data):
    mean = np.reshape(np.mean(data, axis=1), (data.shape[0], 1))
    norm_std = data - mean
    return norm_std


def order_eigenvalues(eigvals):
    order = np.zeros(eigvals.shape)
    eigvals_copy = eigvals.copy()
    for i in range(1, 1 + eigvals.shape[0]):
        order[np.argmax(eigvals_copy)] = i
        eigvals_copy[np.argmax(eigvals_copy)] = float("-inf")
    return order


def construct_eigvec_matrix(eigvals, eigvecs, order):
    W = np.empty((eigvecs.shape[0], 0))
    for i in range(1, 1 + eigvals.shape[0]):
        index = np.squeeze(np.argwhere(order == i))
        W = np.hstack([W, eigvecs[:, index : index + 1]])
    return W


def cov_value(x, y):
    mean_x = sum(x) / float(len(x))
    mean_y = sum(y) / float(len(y))

    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]

    sum_value = sum([sub_y[i] * sub_x[i] for i in range(len(x))])
    denom = float(len(x) - 1)

    cov = sum_value / denom
    return cov


def covariance(arr):
    c = [[cov_value(a, b) for a in arr] for b in arr]
    return c


def my_pca(normalized):
    print("starting pca")
    A = np.cov(normalized)
    eigvals, eigvecs = np.linalg.eig(A)
    var = eigvals / np.sum(eigvals)
    print("found all")

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    sums = []
    for i in range(12):
        sums.append(np.sum(eigvals[0:i+1]))
    # plt.plot(sums)
    # plt.bar([1,2,3,4,5,6,7,8,9,10,11,12], eigvals)
    # plt.title("Scree Plot", fontsize=20)
    # plt.xlabel("Component Number", fontsize=20)
    # plt.ylabel("Eigenvalue", fontsize=20)
    # plt.savefig("screeplot.png")
    # plt.show()
    W = np.zeros((eigvecs.shape[0], eigvecs.shape[1]))
    W[:, 0:3] = eigvecs[:, 0:3]

    proj_data = np.matmul(W.T, normalized)
    print("Eigenvectors (columns): \n" + str(eigvecs))
    np.save("pca_components_oldqgym", eigvecs)
    print("Eigenvalues: " + str(eigvals))
    #np.save("eigvals", eigvals)
    print("Variances: " + str(var))
    print(proj_data.shape)
    return proj_data, eigvecs, eigvals, var


def sklearnpca(normalized):
    pca = PCA(n_components=6)
    proj_data = pca.fit_transform(normalized).T
    print("sklearn pca var" + str(pca.explained_variance_ratio_))
    print(np.sum(pca.explained_variance_ratio_))

    #for plotting variances
    fig = plt.figure(figsize=(10, 10))
    sums = []
    for i in range(12):
        sums.append(np.sum(pca.explained_variance_ratio_[0:i]))
    plt.plot(sums)
    plt.title("Explained Variances across PCs", fontsize=20)
    plt.xlabel("PCs", fontsize=20)
    plt.ylabel("Total Explained Variance", fontsize=20)
    
    print(pca.components_.T)

    #np.save("pca_components_ref_withpcascaling", pca.components_.T)
    eigvecs = pca.components_.T
    var = pca.explained_variance_ratio_
    print(eigvecs.shape)
    return proj_data, eigvecs, var


full_data = dict(np.load("new_logs_oldqgym.npz"))
data_names = list(full_data.keys())

# data for indivdual leg
# set_num = 4 #from 1-4 for 4 legs
# data = full_data['dof_pos_obs'][:,(set_num-1)*3:set_num*3]
# print(data.shape)

# # data for actuator type
# actuator_type = 3
# full_data = full_data['dof_pos_obs']
# data = np.empty((full_data.shape[0],0))
# for d in range(4):
#     data = np.hstack((data, full_data[:,actuator_type+d*3-1:actuator_type+d*3]))
# #data = data.T
# print(data.shape)

# all 12 actuators
data = full_data["dof_pos_obs"]
print(data.shape)

normalized = normalize(data)

# plot for sanity check
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter3D(data[:,0],data[:,1],data[:,2],s=5)
# ax.set_aspect('equal')
# ax.set_xlabel('$X$')
# ax.set_ylabel('$Y$')
# ax.set_zlabel('$Z$')
# plt.show()

proj_data, eigvecs, eigvals, var = my_pca(normalized.T)
#proj_data, eigvecs, var = sklearnpca(normalized)

"""
# plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(proj_data[0, :], proj_data[1, :], proj_data[2, :], s=5)
# ax.scatter3D(normalized[:,0],normalized[:,1],normalized[:,2],s=5)

# for i in range(3):
# #         #no arrow heads
# #         #ax.plot([0, eigvecs[0,i]], [0, eigvecs[1,i]], [0, eigvecs[2,i]], color='blue', alpha=0.8, lw=3)
# #         #with arrow heads + scaling based on variance

# #UNCOMMENT FOR EIGVEC ARROWS
#         a = Arrow3D([0,eigvecs[0,i]*var[i]], [0,eigvecs[1,i]*var[i]], [0,eigvecs[2,i]*var[i]], mutation_scale=5, lw=1, arrowstyle="-|>", color="r")
#         ax.add_artist(a)
a = Arrow3D([0, 1], [0, 0], [0, 0], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
ax.add_artist(a)
a = Arrow3D([0, 0], [0, 1], [0, 0], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
ax.add_artist(a)
a = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
ax.add_artist(a)
ax.set_aspect("equal")
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.set_zlabel("$Z$")
plt.draw()
plt.title("Principal Component Analysis of Dataset", fontsize=20)
plt.show()
# plt.savefig("pca_figures/same_actuators.png")

# [[ 0.1908192  -0.14334909 -0.06542918]
#  [ 0.21286251 -0.16837439 -0.26777735]
#  [ 0.21671466 -0.0384895  -0.2714394 ]
#  [ 0.10711524 -0.15630009 -0.10363912]
#  [-0.62923506  0.06820159  0.10970734]
#  [-0.15028135  0.05048668  0.45711409]
#  [-0.11397995 -0.12248484 -0.0692966 ]
#  [ 0.41294512  0.50485976  0.55272695]
#  [-0.31539877 -0.26916481  0.21056653]
#  [-0.15398403 -0.11891383 -0.13825684]
#  [-0.12318052  0.68874044 -0.48912235]
#  [ 0.34560296 -0.29521193  0.07484593]]

"""