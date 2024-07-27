from gym import LEGGED_GYM_ROOT_DIR

import os
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = f"{LEGGED_GYM_ROOT_DIR}/logs/mini_cheetah_ref/"
LOAD_RUN = "Jul26_11-58-37_LinkedIPG_100Hz_nu02_v08"

PLOT_LOSSES = {
    "Nu=0.9 sim": "losses_nu09_sim.csv",
    "Nu=0.9 no sim": "losses_nu09_nosim.csv",
}

LABELS = ["value_loss", "surrogate_loss", "q_loss", "offpol_loss"]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
plt.suptitle(" IPG Finetuning Losses")

for i, label in enumerate(LABELS):
    for name, file in PLOT_LOSSES.items():
        data = pd.read_csv(os.path.join(ROOT_DIR, LOAD_RUN, file))
        axs[i // 2, i % 2].plot(data[label], label=name)
        axs[i // 2, i % 2].set_title(label)
        axs[i // 2, i % 2].set_xlabel("Checkpoint")
        axs[i // 2, i % 2].set_ylabel("Loss")
        axs[i // 2, i % 2].legend()

plt.tight_layout()
plt.show()
