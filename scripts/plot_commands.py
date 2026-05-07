import torch

commands = torch.load("commands.pt", weights_only=True).detach().cpu().numpy()
base_height = torch.load("base_height.pt", weights_only=True).detach().cpu().numpy()
labels = ["x command", "y command", "yaw command", "height command"]

"""
fig, axs = plt.subplots(4)

for i in range(len(labels)):
    axs[i].hist(commands[:, i], bins=20, cumulative=True)
    axs[i].set_title(labels[i])

plt.suptitle("cumulative histogram of resampled commands")
plt.tight_layout()
plt.savefig("commands.png")
print("saved figure")
"""

print(commands)
print("commands shape:")
print(commands[:, 3:4].shape)
print(base_height)
print(base_height.shape)

"""
# debug distribution of commands
if not os.path.isfile("commands.pt"):
    torch.save(self.commands, "commands.pt")
    torch.save(rand_float, "rand_float.pt")
    torch.save(mask_9, "m9.pt")
    torch.save(random_mult, "random_mult.pt")
    print("saved commands.pt")
"""
