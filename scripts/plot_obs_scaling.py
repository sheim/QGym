import numpy as np
import os

LOG_FILE = "obs_logs.npz"
ENV_ID = 0
SAVE_DIR = "obs_scaling_output"
os.makedirs(SAVE_DIR, exist_ok=True)

data = np.load(LOG_FILE, allow_pickle=True)

# extract observation variables from keys
obs_vars = set()
for key in data.keys():
    print(key)
    if key.endswith("_raw"):
        obs_vars.add(key[:-4])  # remove '_raw' suffix

print(f"Found observation variables: {obs_vars}")


def save_txt(var_name, values, save_dir):
    """save multi-component array to a single .txt file"""
    if values.ndim > 2:
        # reshape to 2D: timesteps × components
        values_2d = values.reshape(values.shape[0], -1)
    else:
        values_2d = values
    save_path = os.path.join(save_dir, f"{var_name}.txt")
    np.savetxt(save_path, values_2d, fmt="%.6f")
    print(f"[saved txt] {save_path}")


for var in obs_vars:
    raw_key = f"{var}_raw"
    scaled_key = f"{var}_scaled"

    raw = data[raw_key][ENV_ID]
    scaled = data[scaled_key][ENV_ID] if scaled_key in data else raw

    # save raw and scaled as separate .txt files
    save_txt(f"{var}_raw", raw, SAVE_DIR)
    save_txt(f"{var}_scaled", scaled, SAVE_DIR)
