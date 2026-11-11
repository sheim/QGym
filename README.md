# Installation

Clone the repo:

```git clone https://github.com/LampLighterLab/BigRedGym.git```

Then, enter the BigRedGym folder.

Create the venv using uv:

```uv sync --frozen```

Note: you will need to use the ```--frozen``` flag every time you run a python script in this repo when the optional VSim wheel or Unitree SDK checkout is absent. example: ```uv run --frozen scripts/train.py --task=go2trot  --device=cuda:0 --headless --max_iterations=300```

## VS Code debugger

After `uv sync --frozen`, open the repository folder in VS Code with the Microsoft
Python and Python Debugger extensions installed. Select **Train Go2 (viewer)** in
Run and Debug, then press **F5**.

The shared [.vscode/launch.json](.vscode/launch.json) selects `.venv/bin/python`
on Linux, `.venv/Scripts/python.exe` on Windows, and `.venv/bin/mjpython` on macOS.
Edit the single `args` list to change the task, device, environment count, or
other CLI options. macOS users who encounter the library-loading error should
apply the workaround below first.

## macOS

The [MuJoCo passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
requires `mjpython` on macOS. Start with fewer environments than the Go2 trot
default of 4096 when running on CPU:

```sh
.venv/bin/mjpython scripts/train.py --task=go2trot --device=cpu --num_envs=16
```

If `mjpython` fails with `Library not loaded: @executable_path/../lib/libpython3.11.dylib`,
it is looking for uv's Python library relative to `.venv`. For this error, link
the existing library into `.venv/lib`, then rerun the command above:

```sh
ln -s "$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib/libpython3.11.dylib" .venv/lib/libpython3.11.dylib
```

This workaround is local to the virtual environment; repeat it if you recreate
`.venv` and encounter the error again.

For training without a viewer, use regular Python with `--headless`:

```sh
uv run --frozen scripts/train.py --task=go2trot --device=cpu --num_envs=16 --headless
```

# Training

To train, run ```scripts/train.py```:

```
uv run --frozen scripts/train.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--num_envs: how many environments to run in parallel in training
--max_iterations: how many training iterations to run (typically 300-500 is sufficient for the go2)
--headless: disable the viewer window during training
--save_interval: override checkpoint interval in learning iterations
--seed: random seed for rng
--batch_size: override batch size from cfg
--experiment_name: Experiment folder under logs/ (default: task config value)
--run_name: Run label appended to the timestamp in the log folder and W&B name
--resume: Resume optimizer and model state from an existing run
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

For example, `--task go2trot --run_name tracking_trial` saves to
`logs/go2trot/<timestamp>_tracking_trial/` and uses `<timestamp>_tracking_trial`
as the W&B display name. Use `--experiment_name` to change the parent folder.

## Offline replay and height Q

Opt in to bounded transition recording without changing the PPO path:

```bash
uv run --frozen scripts/train.py --task=go2trot --device=cpu --headless \
  --num_envs=1024 --max_iterations=600 --run_name=trot-replay \
  --replay_record --replay_capacity=1000000 --replay_snapshot_interval=50
```

Then relabel height, fit a fixed-policy Q, and validate it against fresh
simulator rollouts:

```bash
uv run --frozen scripts/train_height_q.py --snapshot <snapshot> \
  --checkpoint <checkpoint> --output /tmp/opencode/height-q \
  --height_reference 0.3 --height_bound 0.05 --gamma 0.99 \
  --subset_group episode --fractions 0.1 0.5 1.0

uv run --frozen scripts/evaluate_height_q.py \
  --artifact /tmp/opencode/height-q/height_q/data/100pct \
  --load_run <run> --checkpoint <same-checkpoint> \
  --trials 20 --repetitions 3 --horizon 100 \
  --thresholds -1.0 -0.5 0.0 --candidate_actions <candidates.pt> \
  --output /tmp/opencode/height-q-eval.json
```

See [offline replay and fixed-policy height Q](docs/offline_height_q.md) for
snapshot/artifact schemas, terminal and timeout semantics, early/late/mixed
comparisons, and evidence boundaries. These artifacts use actor observations and
do not import CORL.

# Testing

To test your policy in simulation, run ```scripts/play.py```:

```
uv run --frozen scripts/play.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--experiment_name: Override experiment_name (log dir is logs/<experiment_name>/...)
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

Go2Trot policies observe sine/cosine gait phase and phase frequency. To resume or
play an older checkpoint trained without these inputs, add `--original_cfg` to
use the configuration saved with that run.

The automated regression suite runs with `uv run --frozen python -m pytest -q`.
See [limitations and validation scope](genAI_skills/README_MUJOCO.md#limitations-and-validation-scope)
for what the retained tests cover.

# Note on AI-generated files
Some of the code has been worked on by an AI agent, in the case where code segments have been heavily edited by AI, the AI-generated files in [genAI_skills](genAI_skills/) may be of some value in understanding the code, especially [README_MUJOCO.md](genAI_skills/README_MUJOCO.md).
