# dataset creation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit


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


def my_pca(normalized, num_actuators):
    print("starting pca")
    A = np.cov(normalized)
    eigvals, eigvecs = np.linalg.eig(A)
    var = eigvals / np.sum(eigvals)
    print("found all")

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    sums = []
    for i in range(num_actuators):
        sums.append(np.sum(eigvals[0 : i + 1]))
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
    # np.save("eigvals", eigvals)
    print("Variances: " + str(var))
    print(proj_data.shape)
    return proj_data, eigvecs, eigvals, var


def sklearnpca(normalized, num_actuators):
    pca = PCA(n_components=num_actuators)
    proj_data = pca.fit_transform(normalized).T
    print("sklearn pca var" + str(pca.explained_variance_ratio_))
    print(np.sum(pca.explained_variance_ratio_))

    # for plotting variances
    # fig = plt.figure(figsize=(10, 10))
    sums = []
    for i in range(num_actuators):
        sums.append(np.sum(pca.explained_variance_ratio_[0:i]))
    # plt.plot(sums)
    # plt.title("Explained Variances across PCs", fontsize=20)
    # plt.xlabel("PCs", fontsize=20)
    # plt.ylabel("Total Explained Variance", fontsize=20)
    # plt.show()
    print(pca.components_.T)

    np.save("humanoid_pcas", pca.components_.T)
    eigvecs = pca.components_.T
    var = pca.explained_variance_ratio_
    # print(eigvecs.shape)
    return proj_data, eigvecs, var


def est_func(position_data, phase_data):
    # Normalize phase data to the range [0, 2π]
    phase_data_normalized = (
        2
        * np.pi
        * (phase_data - np.min(phase_data))
        / (np.max(phase_data) - np.min(phase_data))
    )

    # Define a more complicated sinusoidal model with multiple frequencies
    def complicated_sinusoidal_model(phase, A1, omega1, phi1, A2, omega2, phi2):
        return A1 * np.sin(omega1 * phase + phi1) + A2 * np.sin(omega2 * phase + phi2)

    # Initial guess for the parameters: Amplitudes, frequencies, and phase shifts
    initial_guess = [1, 1, 0, 0.5, 2, 0]

    # Fit the model to your data
    params, _ = curve_fit(
        complicated_sinusoidal_model,
        phase_data_normalized,
        position_data,
        p0=initial_guess,
    )

    # Extract the fitted parameters
    (
        A1_fitted,
        omega1_fitted,
        phi1_fitted,
        A2_fitted,
        omega2_fitted,
        phi2_fitted,
    ) = params

    # Generate fitted data for visualization
    fitted_position = complicated_sinusoidal_model(
        phase_data_normalized,
        A1_fitted,
        omega1_fitted,
        phi1_fitted,
        A2_fitted,
        omega2_fitted,
        phi2_fitted,
    )

    # Plot the original data and the fitted curve
    plt.plot(phase_data_normalized, position_data, "bo", label="Collected Data")
    plt.plot(
        phase_data_normalized, fitted_position, "r-", label="Fitted Sinusoidal Function"
    )
    plt.xlabel("Phase (normalized to 0 to 2π)")
    plt.ylabel("Position (DOF)")
    plt.legend()
    plt.show()

    # Output the fitted parameters
    print(
        f"Fitted Parameters for Sinusoid 1: Amplitude: {A1_fitted}, Frequency: {omega1_fitted}, Phase Shift: {phi1_fitted}"
    )
    print(
        f"Fitted Parameters for Sinusoid 2: Amplitude: {A2_fitted}, Frequency: {omega2_fitted}, Phase Shift: {phi2_fitted}"
    )
    return np.array(
        [
            [A1_fitted, omega1_fitted, phi1_fitted],
            [A2_fitted, omega2_fitted, phi2_fitted],
        ]
    )


def est_more(position_data, phase_data):
    # Define your model function with 5 sinusoids
    def complicated_sinusoidal_model(
        phase,
        A1,
        omega1,
        phi1,
        A2,
        omega2,
        phi2,
        A3,
        omega3,
        phi3,
        A4,
        omega4,
        phi4,
        A5,
        omega5,
        phi5,
    ):
        return (
            A1 * np.sin(omega1 * phase + phi1)
            + A2 * np.sin(omega2 * phase + phi2)
            + A3 * np.sin(omega3 * phase + phi3)
            + A4 * np.sin(omega4 * phase + phi4)
            + A5 * np.sin(omega5 * phase + phi5)
        )

    # Normalize phase data to the range [0, 2π]
    phase_data_normalized = (
        2
        * np.pi
        * (phase_data - np.min(phase_data))
        / (np.max(phase_data) - np.min(phase_data))
    )

    # Initial guess for the parameters: Amplitudes, frequencies, and phase shifts
    initial_guess = [1, 1, 0, 0.5, 2, 0, 0.3, 3, 0, 0.2, 4, 0, 0.1, 5, 0]

    # Attempt fitting with increased maxfev
    try:
        params, _ = curve_fit(
            complicated_sinusoidal_model,
            phase_data_normalized,
            position_data,
            p0=initial_guess,
            maxfev=2000,
        )
    except RuntimeError as e:
        print("Error during curve fitting:", e)
        # Handle error or provide fallback
        plt.plot(phase_data_normalized, position_data, "bo", label="Collected Data")
        plt.xlabel("Phase (normalized to 0 to 2π)")
        plt.ylabel("Position (DOF)")
        plt.legend()
        plt.show()
        params = None

    if params is not None:
        # Extract fitted parameters
        (
            A1_fitted,
            omega1_fitted,
            phi1_fitted,
            A2_fitted,
            omega2_fitted,
            phi2_fitted,
            A3_fitted,
            omega3_fitted,
            phi3_fitted,
            A4_fitted,
            omega4_fitted,
            phi4_fitted,
            A5_fitted,
            omega5_fitted,
            phi5_fitted,
        ) = params

        # Generate fitted data for visualization
        fitted_position = complicated_sinusoidal_model(
            phase_data_normalized,
            A1_fitted,
            omega1_fitted,
            phi1_fitted,
            A2_fitted,
            omega2_fitted,
            phi2_fitted,
            A3_fitted,
            omega3_fitted,
            phi3_fitted,
            A4_fitted,
            omega4_fitted,
            phi4_fitted,
            A5_fitted,
            omega5_fitted,
            phi5_fitted,
        )

        # Plot the original data and the fitted curve
        plt.plot(phase_data_normalized, position_data, "bo", label="Collected Data")
        plt.plot(
            phase_data_normalized,
            fitted_position,
            "r-",
            label="Fitted Sinusoidal Function",
        )
        plt.xlabel("Phase (normalized to 0 to 2π)")
        plt.ylabel("Position (DOF)")
        plt.legend()
        plt.show()

        # Output the fitted parameters
        print(
            f"Fitted Parameters for Sinusoid 1: Amplitude: {A1_fitted}, Frequency: {omega1_fitted}, Phase Shift: {phi1_fitted}"
        )
        print(
            f"Fitted Parameters for Sinusoid 2: Amplitude: {A2_fitted}, Frequency: {omega2_fitted}, Phase Shift: {phi2_fitted}"
        )
        print(
            f"Fitted Parameters for Sinusoid 3: Amplitude: {A3_fitted}, Frequency: {omega3_fitted}, Phase Shift: {phi3_fitted}"
        )
        print(
            f"Fitted Parameters for Sinusoid 4: Amplitude: {A4_fitted}, Frequency: {omega4_fitted}, Phase Shift: {phi4_fitted}"
        )
        print(
            f"Fitted Parameters for Sinusoid 5: Amplitude: {A5_fitted}, Frequency: {omega5_fitted}, Phase Shift: {phi5_fitted}"
        )

        return np.array(
            [
                [A1_fitted, omega1_fitted, phi1_fitted],
                [A2_fitted, omega2_fitted, phi2_fitted],
                [A3_fitted, omega3_fitted, phi3_fitted],
                [A4_fitted, omega4_fitted, phi4_fitted],
                [A5_fitted, omega5_fitted, phi5_fitted],
            ]
        )


# AIC calculation function
def calculate_aic(n_params, residuals, n_points):
    residual_sum_of_squares = np.sum(residuals**2)
    aic = 2 * n_params + n_points * np.log(residual_sum_of_squares / n_points)
    return aic


# General sinusoidal model for n sinusoids
def sinusoidal_model(phase, *params):
    n_sinusoids = len(params) // 3
    result = np.zeros_like(phase)
    for i in range(n_sinusoids):
        A = params[i * 3]
        omega = params[i * 3 + 1]
        phi = params[i * 3 + 2]
        result += A * np.sin(omega * phase + phi)
    return result


def fit_sinusoidal_models(position_data, phase_data, max_sinusoids=5):
    # Normalize phase data to the range [0, 2π]
    phase_data_normalized = (
        2
        * np.pi
        * (phase_data - np.min(phase_data))
        / (np.max(phase_data) - np.min(phase_data))
    )

    best_aic = np.inf
    best_params = None
    best_model = None
    best_n_sinusoids = 0

    for n_sinusoids in range(1, max_sinusoids + 1):
        # Initial guess for parameters: Amplitudes, frequencies, and phase shifts
        initial_guess = []
        for i in range(n_sinusoids):
            initial_guess.extend(
                [1 / (i + 1), (i + 1), 0]
            )  # Start with decreasing amplitudes, increasing frequencies

        try:
            # Fit the model with n sinusoids
            params, _ = curve_fit(
                sinusoidal_model,
                phase_data_normalized,
                position_data,
                p0=initial_guess,
                maxfev=5000,
            )

            # Calculate residuals and AIC
            fitted_position = sinusoidal_model(phase_data_normalized, *params)
            residuals = position_data - fitted_position
            aic = calculate_aic(len(params), residuals, len(position_data))

            # Select the model with the lowest AIC
            if aic < best_aic:
                best_aic = aic
                best_params = params
                best_model = fitted_position
                best_n_sinusoids = n_sinusoids

        except RuntimeError as e:
            print(f"Error fitting with {n_sinusoids} sinusoids:", e)
            continue

    if best_params is not None:
        n_sinusoids = len(params) // 3  # Number of sinusoids
        best_params = np.reshape(
            best_params, (best_n_sinusoids, 3)
        ).T  # Transpose to make it 3xn

        print(f"Best model has {best_n_sinusoids} sinusoids with AIC: {best_aic}")

        # Plot the best fit
        plt.plot(phase_data_normalized, position_data, "bo", label="Collected Data")
        plt.plot(
            phase_data_normalized,
            best_model,
            "r-",
            label=f"Fitted {best_n_sinusoids} Sinusoids",
        )
        plt.xlabel("Phase (normalized to 0 to 2π)")
        plt.ylabel("Position (DOF)")
        plt.legend()
        plt.show()

        return best_params, best_n_sinusoids, best_aic
    else:
        print("No suitable model was found.")
        return None, None, None


mode = "regular"
if mode == "regular":
    full_data = dict(np.load("data_source_straight_v3_a0.npz"))
    data = full_data["dof_pos_obs"][:, 0:10]
    normalized = normalize(data)
    proj_data, eigvecs, var = sklearnpca(normalized, num_actuators=10)

else:
    full_data = dict(np.load("data_source_straight.npz"))
    data_names = list(full_data.keys())
    phase = full_data["phase"].reshape(1001, 16, 1)
    # phase_wrapped = np.mod(phase[:,0,0], 2 * np.pi)
    phase_flat = np.mod(phase[:, 0, 0], 2 * np.pi)

    start_idx = np.where(np.isclose(phase_flat, 0, atol=0.1))[0][
        0
    ]  # Adjust tolerance as needed
    end_idx = (
        np.where(np.isclose(phase_flat[start_idx + 1 :], 0, atol=0.1))[0][0]
        + start_idx
        + 1
    )
    print(f"Start of first period: {start_idx}")
    print(f"End of first period: {end_idx}")

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
    pcas_allphases = np.empty((12, 3, 1001))
    og_data = full_data["dof_pos_obs"]
    for i in range(start_idx, end_idx + 1):
        data = og_data.reshape(1001, 16, 12)[i, :, :]
        normalized = normalize(data)
        proj_data, eigvecs, var = sklearnpca(normalized)
        pcas_allphases[:, :, i] = eigvecs

    fig = plt.figure(figsize=(10, 10))
    plt.plot(pcas_allphases[0, 0, :])

    plt.plot(phase[:, 0, 0])
    plt.show()

    # fig = plt.figure(figsize=(10, 10))
    # # plt.scatter(phase[start_idx:end_idx+1,0,0], pcas_allphases[0,0,start_idx:end_idx+1])
    # plt.scatter(phase[:,0,0], pcas_allphases[0,0,:])

    plt.show()

    # sinusoids_3pcs = []
    # for k in range(0,3):
    #     sinusoid_params = []
    #     for j in range(0,12):
    #         position_data = pcas_allphases[j,0,start_idx:end_idx+1]
    #         phase_data = phase[start_idx:end_idx+1,0,0]
    #         num_sinusoids_list = [1, 2, 3, 4, 5]
    #         best_params, best_n_sinusoids, best_aic = fit_sinusoidal_models(position_data, phase_data, max_sinusoids=5)
    #         if best_params.shape[1] < 5:
    #             padded_params = np.pad(best_params, ((0, 0), (0, 5 - best_params.shape[1])), 'constant')
    #         else:
    #             padded_params = best_params

    #         sinusoid_params.append(padded_params)
    #     sinusoid_params = np.array(sinusoid_params)
    #     sinusoids_3pcs.append(sinusoid_params)
    #     print(sinusoid_params.shape)
    # sinusoids_3pcs = np.array(sinusoids_3pcs)
    # print(sinusoids_3pcs.shape)
    # np.save("time_based_pcas", sinusoids_3pcs)


# plotting
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter3D(proj_data[0, :], proj_data[1, :], proj_data[2, :], s=5)

# ax.scatter3D(normalized[:,0],normalized[:,1],normalized[:,2],s=5)

# for i in range(3):
# #         #no arrow heads
#     ax.plot([0, eigvecs[0,i]], [0, eigvecs[1,i]], [0, eigvecs[2,i]], color='blue', alpha=0.8, lw=3)
# with arrow heads + scaling based on variance

# UNCOMMENT FOR EIGVEC ARROWS
# a = Arrow3D([0,eigvecs[0,i]*var[i]], [0,eigvecs[1,i]*var[i]], [0,eigvecs[2,i]*var[i]], mutation_scale=5, lw=1, arrowstyle="-|>", color="r")
# ax.add_artist(a)


# a = Arrow3D([0, 1], [0, 0], [0, 0], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
# ax.add_artist(a)
# a = Arrow3D([0, 0], [0, 1], [0, 0], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
# ax.add_artist(a)
# a = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
# ax.add_artist(a)
# ax.set_aspect("equal")
# ax.set_xlabel("$X$")
# ax.set_ylabel("$Y$")
# ax.set_zlabel("$Z$")
# plt.draw()
# plt.title("Principal Component Analysis of Dataset", fontsize=20)
# plt.show()


# plt.savefig("pca_figures/same_actuators.png")
