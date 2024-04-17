import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
name = "plots/distribution_pink"
data = pd.read_csv(name + ".csv")

# Plot the data (last n steps)
n = 200
plt.plot(data.iloc[-n:, 0])
plt.plot(data.iloc[-n:, 1])
plt.xlabel("timestep")
plt.ylabel("action (NN output)")
plt.title("gSDE (sample_freq=8)")
plt.legend(["mean", "sample"])
# plt.show()

# Save the plot to a file
plt.savefig(name + ".png")
