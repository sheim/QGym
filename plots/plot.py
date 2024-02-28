import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
name = "distribution_smooth"
data = pd.read_csv(name + ".csv")

# Plot the data
n = 200
plt.plot(data.iloc[:n, 0])
plt.plot(data.iloc[:n, 1])
plt.xlabel("timestep")
plt.ylabel("action")
plt.title("Smoothing every rollout")
plt.legend(["mean", "sample"])
# plt.show()

# Save the plot to a file
plt.savefig(name + ".png")
