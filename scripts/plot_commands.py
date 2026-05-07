import torch
import matplotlib.pyplot as plt

commands = torch.load("commands.pt", weights_only=True).detach().cpu().numpy()
labels = ["x command", "y command", "yaw command", "height command"]

fig, axs = plt.subplots(4)

for i in range(len(labels)):
    axs[i].hist(commands[:, i], bins=20, cumulative=True)
    axs[i].set_title(labels[i])

plt.suptitle("cumulative histogram of resampled commands")
plt.tight_layout()
plt.savefig("commands.png")
print("saved figure")

rand_float = torch.load("rand_float.pt", weights_only=True).detach().cpu().numpy()
m9 = torch.load("m9.pt", weights_only=True).detach().cpu().numpy()
m1 = torch.load("random_mult.pt", weights_only=True).detach().cpu().numpy()

print(rand_float[:10])
print(m9[:10])
print(m1[:10])
