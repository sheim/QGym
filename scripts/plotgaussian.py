import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

data = dict(np.load("gaussian_scalings_logs.npz"))
pca_scalings = data["pca_scalings"]
dof_pos_obs = data["dof_pos_obs"]


print(dof_pos_obs[0, 0])
