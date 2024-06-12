# dataset creation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA

#full_data = (np.load("pca_scalings_training_myrunner.npz",allow_pickle=True)['pca_scalings'])
full_data = np.load("pca_components_ref_withpcascaling.npy")
print(full_data)

# for i in range(0,full_data.shape[0],4000):
#     var = abs(full_data[i,:]) / np.sum(np.abs(full_data[i,:]))
#     sums = []
#     for j in range(var.shape[0]):
#         sums.append(np.sum(var[0:j+1]))
#     print(var)
#     print(sums)
#     plt.bar([1,2,3,4,5,6,7,8,9,10,11,12], sums)
#     plt.title("Scree Plot", fontsize=20)
#     plt.xlabel("Component Number", fontsize=20)
#     plt.ylabel("Variance", fontsize=20)
#     plt.show()


