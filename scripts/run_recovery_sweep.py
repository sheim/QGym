# run_recovery_sweep.py

import subprocess
import os

RESULTS_DIR = "recovery_sweep"
os.makedirs(RESULTS_DIR, exist_ok=True)

torque_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
latency_values = [0, 1, 2, 3, 4, 5]

for torque in torque_values:
    for latency in latency_values:
        print(f"\nRunning torque={torque}, latency={latency}")

        cmd = [
            "python",
            "scripts/play.py",
            "--task",
            "horse_osc_belay",
            "--torque_scale",
            str(torque),
            "--latency_steps",
            str(latency),
            "--max_steps",
            "2500",
            "--results_dir",
            RESULTS_DIR,
            "--headless",
        ]

        subprocess.run(cmd, check=True)

print("Sweep complete.")
