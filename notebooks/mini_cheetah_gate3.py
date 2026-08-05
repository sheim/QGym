"""Visual report for the mini_cheetah_ref Gate 3 parity campaign.

This notebook focuses on the controlled 2026-07-28 campaign:

* why the original CPU/GPU training comparison was invalid;
* corrected 16-step training curves;
* native command and gait-reference tracking;
* final-checkpoint and iteration-100 transfer matrices;
* late VSim-specific policy exploitation;
* deterministic CPU/Warp trajectory equivalence.

Run from the repository root:

    uv run marimo edit notebooks/mini_cheetah_gate3.py
    uv run marimo run notebooks/mini_cheetah_gate3.py

The notebook is robust to partial logs and explains which sections are missing.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, json, mo, np, plt


@app.cell
def _(Path, json, np):
    repo_logs = Path("logs")
    backend_order = ["cpu", "warp", "vsim"]
    backend_colors = {
        "cpu": "#4C78A8",
        "warp": "#F58518",
        "vsim": "#E45756",
    }

    def pick_data_dir(*names):
        """Choose the newest existing named data directory."""
        _candidates = [repo_logs / _name for _name in names]
        _existing = [_path for _path in _candidates if _path.exists()]
        if not _existing:
            return None
        return max(_existing, key=lambda _path: _path.stat().st_mtime)

    def load_latest_vitals(roots, min_iteration=250):
        """Load the newest completed run from an explicit set of experiment roots."""
        _runs = [
            _run
            for _root in roots
            if _root.exists()
            for _run in _root.iterdir()
            if (_run / "vitals.jsonl").exists()
        ]
        _completed = []
        for _run in _runs:
            _rows = [
                json.loads(_line)
                for _line in (_run / "vitals.jsonl").read_text().splitlines()
                if _line.strip()
            ]
            if _rows and _rows[-1].get("iteration", -1) >= min_iteration:
                _completed.append((_run, _rows))
        if not _completed:
            return None, None
        _latest, _rows = max(
            _completed,
            key=lambda _item: _item[0].stat().st_mtime,
        )
        _columns = {
            _key: np.asarray([_row.get(_key, np.nan) for _row in _rows])
            for _key in _rows[-1]
        }
        return _columns, _latest

    def load_eval_dir(path):
        _out = {}
        if path is None:
            return _out
        for _file in sorted(path.glob("*.npz")):
            _data = np.load(_file, allow_pickle=False)
            _key = (str(_data["train_label"]), str(_data["eval_label"]))
            _out[_key] = {_name: _data[_name] for _name in _data.files}
        return _out

    def load_sweep_dir(path):
        _out = {}
        if path is None or not path.exists():
            return _out
        for _file in sorted(path.glob("*.npz")):
            _iteration = int(_file.stem.split("__", maxsplit=1)[0])
            _data = np.load(_file, allow_pickle=False)
            _out[(_iteration, str(_data["eval_label"]))] = {
                _name: _data[_name] for _name in _data.files
            }
        return _out

    def load_named_npz_dir(path):
        _out = {}
        if path is None or not path.exists():
            return _out
        for _file in sorted(path.glob("*.npz")):
            _data = np.load(_file, allow_pickle=False)
            _out[_file.stem] = {_name: _data[_name] for _name in _data.files}
        return _out

    return (
        backend_colors,
        backend_order,
        load_eval_dir,
        load_latest_vitals,
        load_named_npz_dir,
        load_sweep_dir,
        pick_data_dir,
        repo_logs,
    )


@app.cell
def _(
    backend_order,
    load_eval_dir,
    load_latest_vitals,
    load_named_npz_dir,
    load_sweep_dir,
    pick_data_dir,
    repo_logs,
):
    vitals = {}
    large_batch_vitals = {}
    campaign_paths = {}
    for _backend in backend_order:
        _baseline_root = (
            repo_logs / f"mini_cheetah_ref_{_backend}_h16"
            if _backend != "cpu"
            else repo_logs / "mini_cheetah_ref_cpu"
        )
        _columns, _run = load_latest_vitals([_baseline_root])
        if _columns is not None:
            vitals[_backend] = _columns
        campaign_paths[f"{_backend} baseline training"] = _run

        if _backend != "cpu":
            _large_columns, _large_run = load_latest_vitals(
                [repo_logs / f"mini_cheetah_ref_{_backend}"]
            )
            if _large_columns is not None:
                large_batch_vitals[_backend] = _large_columns
            campaign_paths[f"{_backend} batch-32768 training"] = _large_run

    final_eval_path = pick_data_dir("mc_rl_eval_h16", "mc_rl_eval")
    large_batch_eval_path = repo_logs / "mc_rl_eval_b32768"
    sample_path = pick_data_dir("mc_rl_sample_h16", "mc_rl_sample")
    iteration100_path = repo_logs / "mc_rl_eval_h16_i100"
    sweep_path = repo_logs / "mc_rl_eval_h16" / "vsim_ckpt_sweep"
    large_batch_sweep_path = large_batch_eval_path / "vsim_ckpt_sweep"
    native_tracking_path = repo_logs / "mc_native_tracking"

    final_evals = load_eval_dir(final_eval_path)
    large_batch_evals = load_eval_dir(
        large_batch_eval_path if large_batch_eval_path.exists() else None
    )
    iteration100_evals = load_eval_dir(
        iteration100_path if iteration100_path.exists() else None
    )
    samples = load_eval_dir(sample_path)
    sweep_evals = load_sweep_dir(sweep_path)
    large_batch_sweep_evals = load_sweep_dir(large_batch_sweep_path)
    native_tracking = load_named_npz_dir(native_tracking_path)

    campaign_paths.update(
        {
            "baseline final transfer": final_eval_path,
            "batch-32768 final transfer": (
                large_batch_eval_path if large_batch_eval_path.exists() else None
            ),
            "iteration-100 transfer": (
                iteration100_path if iteration100_path.exists() else None
            ),
            "baseline VSim checkpoint sweep": (
                sweep_path if sweep_path.exists() else None
            ),
            "batch-32768 VSim checkpoint sweep": (
                large_batch_sweep_path if large_batch_sweep_path.exists() else None
            ),
            "deterministic samples": sample_path,
            "native command tracking": (
                native_tracking_path if native_tracking_path.exists() else None
            ),
        }
    )
    return (
        campaign_paths,
        final_evals,
        iteration100_evals,
        large_batch_evals,
        large_batch_sweep_evals,
        large_batch_vitals,
        native_tracking,
        samples,
        sweep_evals,
        vitals,
    )


@app.cell
def _(
    campaign_paths,
    final_evals,
    iteration100_evals,
    large_batch_evals,
    large_batch_sweep_evals,
    large_batch_vitals,
    mo,
    native_tracking,
    samples,
    sweep_evals,
    vitals,
):
    _rows = "\n".join(
        f"| {_name} | {'✅' if _path is not None else '—'} | `{_path}` |"
        for _name, _path in campaign_paths.items()
    )
    mo.md(
        f"""
        # Mini Cheetah Gate 3 — controlled parity campaign

        The corrected experiment holds PPO rollout geometry constant across
        MuJoCo CPU, MuJoCo Warp, and VSim. The main result is asymmetric:
        CPU/Warp-trained policies transfer broadly, while the final VSim policy
        specializes to VSim after initially transferring well.

        **Loaded:** {len(vitals)}/3 baseline training runs ·
        {len(large_batch_vitals)}/2 large-batch GPU runs ·
        {len(final_evals)}/9 baseline transfer cells ·
        {len(large_batch_evals)}/9 large-batch transfer cells ·
        {len(iteration100_evals)}/9 iteration-100 cells ·
        {len(sweep_evals) + len(large_batch_sweep_evals)} checkpoint-screen
        cells · {len(samples)}/9 deterministic samples ·
        {len(native_tracking)}/5 native tracking cases.

        | Artifact | Found | Path |
        |---|:---:|---|
        {_rows}
        """
    )
    return


@app.cell
def _(campaign_paths, mo):
    def _play_command(run_path, backend, device, headless=False):
        if run_path is None:
            return "# checkpoint not found"
        _prefix = "uv run --env-file .env.vsim" if backend == "vsim" else "uv run"
        _parts = [
            _prefix,
            "scripts/play.py",
            "--task mini_cheetah_ref",
            f"--backend {backend}",
            f"--device {device}",
            "--num_envs 1",
            "--seed 0",
            f"--experiment_name {run_path.parent.name}",
            f"--load_run {run_path.name}",
            "--checkpoint 250",
        ]
        if headless:
            _parts.extend(["--headless", "--no-keyboard"])
        return " \\\n  ".join(_parts)

    _specs = [
        (
            "CPU baseline — native MuJoCo CPU viewer",
            campaign_paths.get("cpu baseline training"),
            "mujoco",
            "cpu",
            False,
        ),
        (
            "Warp baseline — exact native CUDA runtime (headless)",
            campaign_paths.get("warp baseline training"),
            "mujoco",
            "cuda:0",
            True,
        ),
        (
            "VSim baseline — native VSim viewer",
            campaign_paths.get("vsim baseline training"),
            "vsim",
            "cuda:0",
            False,
        ),
        (
            "Warp batch 32768 — exact native CUDA runtime (headless)",
            campaign_paths.get("warp batch-32768 training"),
            "mujoco",
            "cuda:0",
            True,
        ),
        (
            "VSim batch 32768 — native VSim viewer",
            campaign_paths.get("vsim batch-32768 training"),
            "vsim",
            "cuda:0",
            False,
        ),
    ]
    _blocks = "\n\n".join(
        f"**{_label}**\n\n```bash\n"
        f"{_play_command(_path, _backend, _device, _headless)}\n```"
        for _label, _path, _backend, _device, _headless in _specs
    )
    _warp_visible = _play_command(
        campaign_paths.get("warp baseline training"),
        "mujoco",
        "cpu",
    )
    mo.md(
        f"""
        ## Run the policies in their native simulators

        These commands resolve an explicit experiment, run directory, and
        checkpoint, so later runs cannot silently change what is loaded.

        {_blocks}

        MuJoCo Warp has no interactive viewer. To render the Warp-trained
        baseline policy, run the same MuJoCo model through the CPU viewer:

        ```bash
        {_warp_visible}
        ```

        The viewer commands enable keyboard velocity commands by default.
        Close the viewer window to stop playback.
        """
    )
    return


@app.cell
def _(backend_colors, backend_order, mo, np, plt):
    _invalid_steps = np.asarray([16, 1, 1])
    _controlled_steps = np.asarray([16, 16, 16])
    _x = np.arange(len(backend_order))
    _width = 0.36

    _fig, _ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    _bars_old = _ax.bar(
        _x - _width / 2,
        _invalid_steps,
        _width,
        label="original",
        color="#BAB0AC",
    )
    _bars_new = _ax.bar(
        _x + _width / 2,
        _controlled_steps,
        _width,
        label="controlled",
        color=[backend_colors[_backend] for _backend in backend_order],
    )
    _ax.bar_label(_bars_old, fontsize=9)
    _ax.bar_label(_bars_new, fontsize=9)
    _ax.set(
        xticks=_x,
        xticklabels=[_backend.upper() for _backend in backend_order],
        ylabel="consecutive steps / env / PPO update",
        title="The original benchmark changed the temporal rollout horizon",
        ylim=(0, 19),
    )
    _ax.legend()
    mo.vstack(
        [
            mo.md(
                """
                ## 1. Benchmark validity

                `num_steps_per_env = batch_size // num_envs`. With batch size
                4096, using 256 CPU environments but 4096 GPU environments
                compared 16-step GAE rollouts with one-step samples. The
                controlled campaign uses 256 environments and 16 steps
                everywhere; throughput scaling is a separate experiment.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, vitals):
    _available_metrics = sorted(
        {
            _key
            for _columns in vitals.values()
            for _key in _columns
            if _key != "iteration"
        }
    )
    _default = "rewards/total_rewards"
    training_metric = mo.ui.dropdown(
        options=_available_metrics,
        value=(
            _default
            if _default in _available_metrics
            else (_available_metrics[0] if _available_metrics else None)
        ),
        label="Training metric",
    )
    training_metric
    return (training_metric,)


@app.cell
def _(backend_colors, backend_order, mo, plt, training_metric, vitals):
    def _training_plot():
        if not vitals or training_metric.value is None:
            return mo.md("_No controlled training vitals found._")
        _fig, _axes = plt.subplots(
            1,
            2,
            figsize=(12.5, 4.2),
            constrained_layout=True,
        )
        for _backend in backend_order:
            if _backend not in vitals:
                continue
            _columns = vitals[_backend]
            _axes[0].plot(
                _columns["iteration"],
                _columns[training_metric.value],
                label=_backend,
                color=backend_colors[_backend],
                lw=1.8,
            )
            _axes[1].plot(
                _columns["iteration"],
                _columns["rewards/reference_traj"],
                label=_backend,
                color=backend_colors[_backend],
                lw=1.8,
            )
        for _ax in _axes:
            _ax.axvline(100, color="#666666", ls="--", lw=1, alpha=0.7)
            _ax.axvline(150, color="#666666", ls=":", lw=1, alpha=0.7)
            _ax.set_xlabel("iteration")
            _ax.grid(alpha=0.2)
        _axes[0].set(
            ylabel=training_metric.value,
            title=training_metric.value,
        )
        _axes[1].set(
            ylabel="weighted reward term",
            title="reference trajectory reward",
        )
        _axes[0].legend()
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## 2. Equal-horizon training

                All three backends now learn the task. The vertical guides mark
                iterations 100 and 150, the interval where the VSim policy's
                target-backend transfer collapses even as its native training
                terms remain healthy.
                """
            ),
            _training_plot(),
        ]
    )
    return


@app.cell
def _(
    backend_colors,
    large_batch_vitals,
    mo,
    plt,
    training_metric,
    vitals,
):
    def _batch_training_plot():
        if not large_batch_vitals or training_metric.value is None:
            return mo.md("_No completed batch-32768 GPU training runs found._")
        _fig, _axes = plt.subplots(
            1,
            2,
            figsize=(12.5, 4.2),
            constrained_layout=True,
        )
        for _backend in ("warp", "vsim"):
            for _label, _dataset, _style in (
                ("batch 4096", vitals, "-"),
                ("batch 32768", large_batch_vitals, "--"),
            ):
                if _backend not in _dataset:
                    continue
                _columns = _dataset[_backend]
                _axes[0].plot(
                    _columns["iteration"],
                    _columns[training_metric.value],
                    label=f"{_backend}, {_label}",
                    color=backend_colors[_backend],
                    ls=_style,
                    lw=1.7,
                )
                _axes[1].plot(
                    _columns["iteration"],
                    _columns["rewards/reference_traj"],
                    label=f"{_backend}, {_label}",
                    color=backend_colors[_backend],
                    ls=_style,
                    lw=1.7,
                )
        _axes[0].set(
            xlabel="iteration",
            ylabel=training_metric.value,
            title=f"{training_metric.value}: batch comparison",
        )
        _axes[1].set(
            xlabel="iteration",
            ylabel="weighted reward term",
            title="reference reward: batch comparison",
        )
        for _ax in _axes:
            _ax.grid(alpha=0.2)
            _ax.legend(fontsize=8)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ### Large-batch follow-up

                Batch 32768 uses 2048 environments, retaining 16 temporal
                steps per environment but collecting eight times as many
                transitions per iteration. The CPU run is intentionally absent:
                its projected runtime was 86 minutes, so this is a focused GPU
                robustness test rather than a replacement full gate.
                """
            ),
            _batch_training_plot(),
        ]
    )
    return


@app.cell
def _(backend_colors, large_batch_vitals, mo, plt, vitals):
    _tracking_metrics = [
        ("rewards/tracking_lin_vel", "linear command tracking reward"),
        ("rewards/tracking_ang_vel", "yaw command tracking reward"),
        ("rewards/reference_traj", "gait-reference tracking reward"),
    ]
    _fig, _axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.2),
        constrained_layout=True,
    )
    for _backend in ("cpu", "warp", "vsim"):
        _datasets = [("batch 4096", vitals, "-")]
        if _backend != "cpu":
            _datasets.append(("batch 32768", large_batch_vitals, "--"))
        for _batch_label, _dataset, _style in _datasets:
            if _backend not in _dataset:
                continue
            _columns = _dataset[_backend]
            for _ax, (_metric, _title) in zip(_axes, _tracking_metrics):
                _ax.plot(
                    _columns["iteration"],
                    _columns[_metric],
                    label=f"{_backend}, {_batch_label}",
                    color=backend_colors[_backend],
                    ls=_style,
                    lw=1.6,
                )
                _ax.set(
                    xlabel="iteration",
                    ylabel="weighted reward term",
                    title=_title,
                )
    for _ax in _axes:
        _ax.grid(alpha=0.2)
        _ax.legend(fontsize=7)
    mo.vstack(
        [
            mo.md(
                """
                ## 3. Native training: command and reference tracking

                These are the tracking terms observed while each policy trains
                in its own backend. Solid lines are the controlled batch-4096
                campaign; dashed lines are the GPU batch-32768 follow-up.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(backend_colors, mo, native_tracking, np, plt):
    _case_specs = [
        ("CPU · b4096", "b4096__cpu", "cpu", "-", "o"),
        ("Warp · b4096", "b4096__warp", "warp", "-", "o"),
        ("VSim · b4096", "b4096__vsim", "vsim", "-", "o"),
        ("Warp · b32768", "b32768__warp", "warp", "--", "s"),
        ("VSim · b32768", "b32768__vsim", "vsim", "--", "s"),
    ]
    _tracking_axes = [
        ("forward velocity", 0, "lin"),
        ("lateral velocity", 1, "lin"),
        ("yaw rate", 2, "ang"),
    ]

    def _valid_samples(cell, settle_seconds=0.5):
        _terminated = cell["terminated"]
        _valid = np.ones_like(_terminated, dtype=bool)
        _valid[1:] = ~np.maximum.accumulate(_terminated[:-1], axis=0)
        _settle = int(round(settle_seconds * float(cell["ctrl_hz"])))
        _valid[:_settle] = False
        return _valid

    def _response(cell, component, velocity_kind):
        _source = (
            cell["base_lin_vel"] if velocity_kind == "lin" else cell["base_ang_vel"]
        )
        return _source[:, :, component]

    def _per_env_average(values, valid):
        _averages = np.full(values.shape[1], np.nan)
        for _env_index in range(values.shape[1]):
            _mask = valid[:, _env_index]
            if np.any(_mask):
                _averages[_env_index] = np.mean(values[_mask, _env_index])
        return _averages

    def _native_command_plot():
        if not native_tracking:
            return mo.md("_No native command-tracking trajectories found._")
        _fig, _axes = plt.subplots(
            1,
            3,
            figsize=(16, 4.4),
            constrained_layout=True,
        )
        for _ax, (_title, _component, _velocity_kind) in zip(_axes, _tracking_axes):
            _all_commands = [
                _cell["commands"][0, :, _component]
                for _key, _cell in native_tracking.items()
                if _key in {_spec[1] for _spec in _case_specs}
            ]
            if not _all_commands:
                _ax.set_axis_off()
                continue
            _command_min = min(float(np.min(_value)) for _value in _all_commands)
            _command_max = max(float(np.max(_value)) for _value in _all_commands)
            _edges = np.linspace(_command_min, _command_max, 11)
            _ax.plot(
                [_command_min, _command_max],
                [_command_min, _command_max],
                color="#777777",
                ls=":",
                lw=1.2,
                label="ideal",
            )
            for _label, _key, _backend, _style, _marker in _case_specs:
                _cell = native_tracking.get(_key)
                if _cell is None:
                    continue
                _valid = _valid_samples(_cell)
                _command = _cell["commands"][0, :, _component]
                _achieved = _per_env_average(
                    _response(_cell, _component, _velocity_kind),
                    _valid,
                )
                _bin = np.digitize(_command, _edges[1:-1])
                _x_values = []
                _y_values = []
                for _bin_index in range(len(_edges) - 1):
                    _selected = (_bin == _bin_index) & np.isfinite(_achieved)
                    if np.any(_selected):
                        _x_values.append(float(np.mean(_command[_selected])))
                        _y_values.append(float(np.mean(_achieved[_selected])))
                _ax.plot(
                    _x_values,
                    _y_values,
                    marker=_marker,
                    ms=4,
                    color=backend_colors[_backend],
                    ls=_style,
                    lw=1.4,
                    label=_label,
                )
            _unit = "m/s" if _velocity_kind == "lin" else "rad/s"
            _ax.set(
                xlabel=f"commanded ({_unit})",
                ylabel=f"achieved mean ({_unit})",
                title=_title,
            )
            _ax.grid(alpha=0.2)
        _axes[0].legend(fontsize=7)
        return _fig

    _metric_names = ["forward", "lateral", "yaw", "gait reference"]
    _metric_values = {}
    for _label, _key, _backend, _style, _marker in _case_specs:
        _cell = native_tracking.get(_key)
        if _cell is None:
            continue
        _valid = _valid_samples(_cell)
        _values = []
        for _title, _component, _velocity_kind in _tracking_axes:
            _command = _cell["commands"][:, :, _component]
            _actual = _response(_cell, _component, _velocity_kind)
            _values.append(
                float(np.sqrt(np.mean((_actual[_valid] - _command[_valid]) ** 2)))
            )
        _reference = _cell.get("reference_dof_rmse")
        _values.append(
            float(np.sqrt(np.mean(_reference[_valid] ** 2)))
            if _reference is not None
            else np.nan
        )
        _survival = 100.0 * float(np.mean(_cell["survived"]))
        _metric_values[_label] = (_backend, _style, _values, _survival)

    def _native_error_plot():
        if not _metric_values:
            return mo.md("")
        _fig, (_ax_error, _ax_survival) = plt.subplots(
            1,
            2,
            figsize=(15, 4.4),
            gridspec_kw={"width_ratios": [2.2, 1]},
            constrained_layout=True,
        )
        _x = np.arange(len(_metric_names))
        _width = 0.15
        _offset = -0.5 * _width * (len(_metric_values) - 1)
        _survival_values = []
        _survival_colors = []
        _survival_hatches = []
        for _case_index, (
            _label,
            (_backend, _style, _values, _survival),
        ) in enumerate(_metric_values.items()):
            _bars = _ax_error.bar(
                _x + _offset + _case_index * _width,
                _values,
                _width,
                label=_label,
                color=backend_colors[_backend],
                alpha=0.55 if _style == "--" else 0.9,
                hatch="//" if _style == "--" else None,
            )
            _ax_error.bar_label(
                _bars,
                fmt="%.2f",
                fontsize=6,
                rotation=90,
                padding=2,
            )
            _survival_values.append(_survival)
            _survival_colors.append(backend_colors[_backend])
            _survival_hatches.append("//" if _style == "--" else None)
        _ax_error.set(
            xticks=_x,
            xticklabels=_metric_names,
            ylabel="RMSE (m/s, rad/s, or rad)",
            title="native held-out tracking error after 0.5 s settling",
        )
        _ax_error.legend(fontsize=7, ncol=3)
        _ax_error.grid(axis="y", alpha=0.2)

        _survival_bars = _ax_survival.bar(
            range(len(_metric_values)),
            _survival_values,
            color=_survival_colors,
        )
        for _bar, _hatch in zip(_survival_bars, _survival_hatches):
            _bar.set_hatch(_hatch)
        _ax_survival.bar_label(_survival_bars, fmt="%.1f%%", fontsize=8)
        _ax_survival.set(
            xticks=range(len(_metric_values)),
            xticklabels=list(_metric_values),
            ylabel="survived full evaluation (%)",
            title="native robustness context",
            ylim=(0, 108),
        )
        _ax_survival.tick_params(axis="x", labelrotation=35)
        _ax_survival.grid(axis="y", alpha=0.2)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## 4. Native evaluation: commanded versus achieved motion

                Each policy is evaluated only in the backend where it trained.
                Curves show binned per-environment responses to fixed held-out
                commands; the dotted diagonal is perfect tracking. RMSE excludes
                the first 0.5 seconds and masks samples after termination. Gait
                reference error is direct joint-position RMSE, not a weighted
                reward proxy. Native survival is shown alongside the conditional
                tracking error so early failures remain visible.
                """
            ),
            _native_command_plot(),
            _native_error_plot(),
        ]
    )
    return


@app.cell
def _(backend_order, np):
    def eval_matrix(evals, metric, scale=1.0):
        _matrix = np.full((len(backend_order), len(backend_order)), np.nan)
        for _i, _train in enumerate(backend_order):
            for _j, _evaluate in enumerate(backend_order):
                _cell = evals.get((_train, _evaluate))
                if _cell is not None:
                    _matrix[_i, _j] = float(np.mean(_cell[metric])) * scale
        return _matrix

    def draw_heatmap(ax, matrix, title, fmt, cmap, vmin=None, vmax=None):
        _image = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(
            range(len(backend_order)),
            [f"eval {_backend}" for _backend in backend_order],
        )
        ax.set_yticks(
            range(len(backend_order)),
            [f"train {_backend}" for _backend in backend_order],
        )
        ax.set_title(title)
        for _i in range(matrix.shape[0]):
            for _j in range(matrix.shape[1]):
                if not np.isnan(matrix[_i, _j]):
                    ax.text(
                        _j,
                        _i,
                        fmt.format(matrix[_i, _j]),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                    )
        return _image

    return draw_heatmap, eval_matrix


@app.cell
def _(
    draw_heatmap,
    eval_matrix,
    final_evals,
    iteration100_evals,
    mo,
    plt,
):
    def _transfer_plot():
        if not final_evals:
            return mo.md("_No final transfer matrix found._")
        _fig, _axes = plt.subplots(
            2,
            2,
            figsize=(12.5, 8),
            constrained_layout=True,
        )
        _datasets = [
            ("iteration 100", iteration100_evals),
            ("iteration 250", final_evals),
        ]
        for _column, (_label, _evals) in enumerate(_datasets):
            if not _evals:
                _axes[0, _column].set_axis_off()
                _axes[1, _column].set_axis_off()
                continue
            _reward = eval_matrix(_evals, "mean_reward")
            _survival = eval_matrix(_evals, "survived", 100.0)
            _reward_image = draw_heatmap(
                _axes[0, _column],
                _reward,
                f"{_label}: mean reward",
                "{:.2f}",
                "viridis",
                vmin=5.5,
                vmax=8.5,
            )
            _survival_image = draw_heatmap(
                _axes[1, _column],
                _survival,
                f"{_label}: survival",
                "{:.0f}%",
                "magma",
                vmin=0,
                vmax=100,
            )
            _fig.colorbar(_reward_image, ax=_axes[0, _column], shrink=0.8)
            _fig.colorbar(_survival_image, ax=_axes[1, _column], shrink=0.8)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## 5. Cross-backend transfer

                At iteration 100, VSim transfer is strong but CPU training is
                incomplete. At iteration 250, CPU and Warp policies transfer
                broadly, while the VSim-trained policy collapses on both
                MuJoCo implementations. This rules out a basic CPU/Warp runtime
                mismatch and exposes a late, asymmetric policy specialization.
                """
            ),
            _transfer_plot(),
        ]
    )
    return


@app.cell
def _(
    draw_heatmap,
    eval_matrix,
    final_evals,
    large_batch_evals,
    mo,
    plt,
):
    def _batch_transfer_plot():
        if not large_batch_evals:
            return mo.md("_No batch-32768 transfer matrix found._")
        _baseline = eval_matrix(final_evals, "survived", 100.0)
        _large = eval_matrix(large_batch_evals, "survived", 100.0)
        _delta = _large - _baseline
        _fig, _axes = plt.subplots(
            1,
            3,
            figsize=(16, 4.2),
            constrained_layout=True,
        )
        _base_image = draw_heatmap(
            _axes[0],
            _baseline,
            "batch 4096 survival",
            "{:.0f}%",
            "magma",
            vmin=0,
            vmax=100,
        )
        draw_heatmap(
            _axes[1],
            _large,
            "batch 32768 survival",
            "{:.0f}%",
            "magma",
            vmin=0,
            vmax=100,
        )
        _delta_image = draw_heatmap(
            _axes[2],
            _delta,
            "change (percentage points)",
            "{:+.0f}",
            "coolwarm",
            vmin=-30,
            vmax=30,
        )
        _fig.colorbar(_base_image, ax=_axes[:2], shrink=0.8)
        _fig.colorbar(_delta_image, ax=_axes[2], shrink=0.8)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ### Does batch 32768 close transfer?

                No. It improves final VSim-native survival from 69.5% to 96.9%
                and keeps Warp broadly portable, but VSim-to-CPU/Warp remains
                below 10%. The CPU row is the unchanged baseline policy.
                """
            ),
            _batch_transfer_plot(),
        ]
    )
    return


@app.cell
def _(
    backend_colors,
    final_evals,
    large_batch_evals,
    large_batch_sweep_evals,
    mo,
    np,
    plt,
    sweep_evals,
):
    def _checkpoint_plot():
        if not sweep_evals and not large_batch_sweep_evals:
            return mo.md("_No VSim checkpoint sweep found._")
        _fig, (_ax_reward, _ax_survival) = plt.subplots(
            1,
            2,
            figsize=(12.5, 4.2),
            constrained_layout=True,
        )
        _datasets = (
            ("batch 4096", 4096, sweep_evals, final_evals, "-"),
            (
                "batch 32768",
                32768,
                large_batch_sweep_evals,
                large_batch_evals,
                "--",
            ),
        )
        for _batch_label, _batch_size, _sweep, _final_evals, _style in _datasets:
            for _evaluate in ("cpu", "vsim"):
                _points = []
                for (_iteration, _label), _cell in _sweep.items():
                    if _label != _evaluate:
                        continue
                    _points.append(
                        (
                            _iteration,
                            float(np.mean(_cell["mean_reward"])),
                            100.0 * float(np.mean(_cell["survived"])),
                        )
                    )
                _final = _final_evals.get(("vsim", _evaluate))
                if _final is not None:
                    _points.append(
                        (
                            250,
                            float(np.mean(_final["mean_reward"])),
                            100.0 * float(np.mean(_final["survived"])),
                        )
                    )
                _points.sort()
                if not _points:
                    continue
                _iterations, _rewards, _survival = map(np.asarray, zip(*_points))
                _transitions_millions = _iterations * _batch_size / 1e6
                _line_label = f"{_batch_label}, eval {_evaluate}"
                _ax_reward.plot(
                    _transitions_millions,
                    _rewards,
                    marker="o",
                    ls=_style,
                    label=_line_label,
                    color=backend_colors[_evaluate],
                    lw=1.6,
                )
                _ax_survival.plot(
                    _transitions_millions,
                    _survival,
                    marker="o",
                    ls=_style,
                    label=_line_label,
                    color=backend_colors[_evaluate],
                    lw=1.6,
                )
        for _ax in (_ax_reward, _ax_survival):
            _ax.set_xlabel("collected transitions (millions)")
            _ax.grid(alpha=0.2)
            _ax.legend(fontsize=8)
        _ax_reward.set(ylabel="mean reward", title="VSim policy reward")
        _ax_survival.set(
            ylabel="survival (%)",
            title="VSim policy survival",
            ylim=(-3, 103),
        )
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## 6. When VSim transfer breaks

                The checkpoint screens use 64 environments at iterations
                50–200; final points use 256. The x-axis accounts for the
                eightfold difference in transitions per update. The larger
                batch widens the transferable window substantially, but CPU
                survival eventually falls to 9% while VSim-native survival
                remains 97%.
                """
            ),
            _checkpoint_plot(),
        ]
    )
    return


@app.cell
def _(mo, samples):
    _policies = sorted({_train for _train, _evaluate in samples})
    _first_sample = next(iter(samples.values()), None)
    _joint_names = (
        [str(_name) for _name in _first_sample["dof_names"]]
        if _first_sample is not None and "dof_names" in _first_sample
        else []
    )
    sample_policy = mo.ui.dropdown(
        options=_policies,
        value=(_policies[0] if _policies else None),
        label="Policy",
    )
    sample_joint = mo.ui.dropdown(
        options=_joint_names,
        value=(_joint_names[1] if len(_joint_names) > 1 else None),
        label="Joint",
    )
    mo.hstack([sample_policy, sample_joint], justify="start", gap=1)
    return sample_joint, sample_policy


@app.cell
def _(
    backend_colors,
    mo,
    np,
    plt,
    sample_joint,
    sample_policy,
    samples,
):
    def _trajectory_plot():
        if sample_policy.value is None or sample_joint.value is None:
            return mo.md("_No deterministic trajectory samples found._")
        _cells = {
            _evaluate: _cell
            for (_train, _evaluate), _cell in samples.items()
            if _train == sample_policy.value
        }
        if "cpu" not in _cells:
            return mo.md("_The selected policy has no CPU reference sample._")
        _names = [str(_name) for _name in _cells["cpu"]["dof_names"]]
        _joint_index = _names.index(sample_joint.value)
        _hz = float(_cells["cpu"]["ctrl_hz"])
        _cpu = _cells["cpu"]["dof_traj"][:, 0, :]
        _time = np.arange(len(_cpu)) / _hz

        _fig, _axes = plt.subplots(
            2,
            2,
            figsize=(12.5, 8),
            constrained_layout=True,
        )
        _pair_labels = []
        _pair_rms = []
        for _evaluate in ("warp", "vsim"):
            if _evaluate not in _cells:
                continue
            _trajectory = _cells[_evaluate]["dof_traj"][:, 0, :]
            _length = min(len(_cpu), len(_trajectory))
            _rms_time = np.sqrt(
                np.mean((_cpu[:_length] - _trajectory[:_length]) ** 2, axis=1)
            )
            _axes[0, 0].plot(
                _time[:_length],
                np.maximum(_rms_time, 1e-10),
                label=f"CPU vs {_evaluate}",
                color=backend_colors[_evaluate],
                lw=1.5,
            )
            _pair_labels.append(f"CPU–{_evaluate}")
            _pair_rms.append(
                float(np.sqrt(np.mean((_cpu[:_length] - _trajectory[:_length]) ** 2)))
            )

        _axes[0, 0].axhline(0.05, color="#777777", ls="--", lw=1)
        _axes[0, 0].set(
            xlabel="time (s)",
            ylabel="all-DOF RMS (rad)",
            title="closed-loop divergence from MuJoCo CPU",
            yscale="log",
        )
        _axes[0, 0].legend()
        _axes[0, 0].grid(alpha=0.2)

        for _evaluate, _cell in sorted(_cells.items()):
            _angle = _cell["dof_traj"][:, 0, _joint_index]
            _local_time = np.arange(len(_angle)) / float(_cell["ctrl_hz"])
            _velocity = np.gradient(_angle, 1.0 / float(_cell["ctrl_hz"]))
            _axes[0, 1].plot(
                _local_time,
                _angle,
                label=_evaluate,
                color=backend_colors[_evaluate],
                lw=1.3,
            )
            _axes[1, 0].plot(
                _angle,
                _velocity,
                label=_evaluate,
                color=backend_colors[_evaluate],
                lw=1.1,
                alpha=0.85,
            )
        _axes[0, 1].set(
            xlabel="time (s)",
            ylabel="position (rad)",
            title=sample_joint.value,
        )
        _axes[1, 0].set(
            xlabel="position (rad)",
            ylabel="finite-difference velocity (rad/s)",
            title=f"{sample_joint.value} phase portrait",
        )
        _axes[0, 1].legend()
        _axes[1, 0].legend()

        _bars = _axes[1, 1].bar(
            _pair_labels,
            _pair_rms,
            color=[
                backend_colors["warp"],
                backend_colors["vsim"],
            ][: len(_pair_rms)],
        )
        _axes[1, 1].bar_label(
            _bars,
            labels=[f"{_value:.2e}" for _value in _pair_rms],
            fontsize=9,
        )
        _axes[1, 1].set(
            ylabel="five-second RMS (rad)",
            title="full-trajectory discrepancy",
            yscale="log",
        )
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## 7. Deterministic trajectory evidence

                The same clean observation, command, gait phase, and basic
                state are used on every backend. CPU/Warp remain at numerical
                precision for all three policies; VSim separates shortly after
                standing contact. The phase portrait uses finite-difference
                velocity because the sample artifact stores joint position.
                """
            ),
            _trajectory_plot(),
        ]
    )
    return


@app.cell
def _(backend_order, final_evals, large_batch_evals, mo, np):
    def _failures(evals):
        _failed = []
        for _train in backend_order:
            for _evaluate in backend_order:
                if _train == _evaluate:
                    continue
                _cell = evals.get((_train, _evaluate))
                if _cell is not None:
                    _survival = 100.0 * float(np.mean(_cell["survived"]))
                    if _survival < 50.0:
                        _failed.append(f"{_train}→{_evaluate} ({_survival:.1f}%)")
        return ", ".join(_failed) if _failed else "none"

    _baseline_failures = _failures(final_evals)
    _large_batch_failures = _failures(large_batch_evals)
    mo.md(
        f"""
        ## Gate verdict

        > **Gate 3 remains open.** Training convergence, CPU/Warp equivalence,
        > deterministic trajectory consistency, and CPU/Warp-to-VSim transfer
        > pass. Baseline off-diagonal failures: **{_baseline_failures}**.
        > Batch-32768 failures: **{_large_batch_failures}**.

        The larger batch makes the transferable VSim window substantially
        wider, but the final policy still specializes to VSim. Increasing batch
        size alone is therefore not a gate-closing fix. The next controlled
        experiment should predeclare either a source-only early-stopping rule or
        a robustness intervention such as contact/domain randomization.

        Reproduce the corrected campaign:

        ```bash
        ITERS=250 SEED=7 TRAIN_ENVS=256 BATCH_SIZE=4096 \\
          EVAL_ENVS=256 T_END=5.0 bash scripts/run_mc_ref_benchmark.sh
        ```

        Large-batch GPU follow-up:

        ```bash
        ITERS=250 SEED=7 TRAIN_ENVS=2048 BATCH_SIZE=32768 \\
          EVAL_ENVS=256 T_END=5.0 bash scripts/run_mc_ref_benchmark.sh
        ```

        The full command includes CPU training, which takes approximately
        86 minutes on this machine. The recorded follow-up stopped that leg and
        reused the controlled CPU row.
        """
    )
    return


if __name__ == "__main__":
    app.run()
