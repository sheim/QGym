import torch
import matplotlib.pyplot as plt


commands = torch.load("commands.pt", weights_only=True).detach().cpu().numpy()
labels = ["x command", "y command", "yaw command", "height command"]

fig, axs = plt.subplots(4)

for i in range(len(labels)):
    axs[i].hist(commands[:, i], bins=10)
    axs[i].set_title(labels[i])

plt.tight_layout()
plt.savefig("commands.png")
print("saved figure")
