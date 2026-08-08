"""Compare controlled Go2 policy-evaluation artifacts.

Generate artifacts first:

    uv run --frozen scripts/eval_go2_policy.py \
        --policy baseline=logs/go2/BASELINE_RUN \
        --policy current=logs/go2/CURRENT_RUN \
        --iterations 100 250 500

Then open this report:

    GO2_EVAL_DIR=logs/go2_evaluation \
        uv run --frozen marimo edit notebooks/go2_policy_evaluation.py

The report compares physical metrics, not only training reward. Reward curves
remain useful diagnostics but are not comparable after reward definitions or
weights change.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from gym.utils.helpers import class_to_dict
    from gym.utils.original_cfg import (
        OriginalCfgError,
        load_original_cfgs_from_run,
        original_cfg_source_dir,
    )
    from gym.utils.policy_io import (
        component_scales_from_names,
        first_episode_mask,
        phase_binned_stats,
        policy_io_in_space,
    )
    from gym.utils.run_config_diff import LoggedConfigError, diff_logged_run_configs

    return (
        LoggedConfigError,
        OriginalCfgError,
        Path,
        class_to_dict,
        component_scales_from_names,
        diff_logged_run_configs,
        first_episode_mask,
        json,
        load_original_cfgs_from_run,
        mo,
        np,
        original_cfg_source_dir,
        os,
        phase_binned_stats,
        plt,
        policy_io_in_space,
    )


@app.cell
def _(Path, json, np, os):
    artifact_root = Path(os.environ.get("GO2_EVAL_DIR", "logs/go2_evaluation"))

    def _load_artifact(_path):
        with np.load(_path, allow_pickle=False) as _data:
            return {"path": _path, **{_key: _data[_key] for _key in _data.files}}

    artifacts = []
    if artifact_root.exists():
        for _path in sorted(artifact_root.glob("*.npz")):
            _artifact = _load_artifact(_path)
            if str(_artifact.get("task", "")) == "go2":
                artifacts.append(_artifact)

    modes = list(
        dict.fromkeys(str(_artifact["reset_mode"]) for _artifact in artifacts)
    )
    eval_labels = list(
        dict.fromkeys(str(_artifact["eval_label"]) for _artifact in artifacts)
    )

    def metric_mean(_artifact, _name, _command=None):
        _key = f"metric_{_name}"
        if _key not in _artifact:
            return np.nan
        _values = np.asarray(_artifact[_key])
        _valid = np.isfinite(_values)
        if _command is not None:
            _valid &= _artifact["command_case"] == _command
        return float(np.mean(_values[_valid])) if np.any(_valid) else np.nan

    def steady_response(_artifact):
        _start = int(
            round(float(_artifact["settling_time_s"]) * float(_artifact["ctrl_hz"]))
        )
        _commands = np.asarray(_artifact["eval_commands"])
        _actual = np.full_like(_commands, np.nan, dtype=float)
        _signals = (
            np.asarray(_artifact["base_lin_vel"])[:, :, 0],
            np.asarray(_artifact["base_lin_vel"])[:, :, 1],
            np.asarray(_artifact["base_ang_vel"])[:, :, 2],
        )
        _terminated = np.asarray(_artifact["terminated"])
        for _env_index in range(len(_commands)):
            _failures = np.flatnonzero(_terminated[:, _env_index])
            _end = int(_failures[0]) if len(_failures) else len(_terminated)
            if _end <= _start:
                continue
            for _axis, _signal in enumerate(_signals):
                _actual[_env_index, _axis] = np.mean(
                    _signal[_start:_end, _env_index]
                )
        return _commands, _actual

    vitals = {}
    for _artifact in artifacts:
        _label = str(_artifact["train_label"])
        _run_dir = Path(str(_artifact["checkpoint_path"])).parent
        _vitals_path = _run_dir / "vitals.jsonl"
        if not _vitals_path.exists():
            continue
        _rows = [
            json.loads(_line)
            for _line in _vitals_path.read_text().splitlines()
            if _line.strip()
        ]
        if _rows:
            vitals[_label] = {
                _key: [_row.get(_key, np.nan) for _row in _rows]
                for _key in _rows[-1]
            }
    return (
        artifact_root,
        artifacts,
        eval_labels,
        metric_mean,
        modes,
        steady_response,
        vitals,
    )


@app.cell
def _(artifacts, eval_labels, modes, mo):
    _options = modes or ["reset_to_range"]
    _default = "reset_to_range" if "reset_to_range" in _options else _options[0]
    reset_mode = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="initial-state protocol",
    )
    _backend_options = eval_labels or ["mujoco-cpu"]
    eval_backend = mo.ui.dropdown(
        options=_backend_options,
        value=_backend_options[0],
        label="evaluation backend",
    )
    mo.hstack(
        [eval_backend, reset_mode, mo.md(f"**Artifacts found:** {len(artifacts)}")],
        justify="start",
        gap=2,
    )
    return eval_backend, reset_mode


@app.cell
def _(artifacts, eval_backend, reset_mode):
    selected = [
        _artifact
        for _artifact in artifacts
        if str(_artifact["reset_mode"]) == reset_mode.value
        and str(_artifact["eval_label"]) == eval_backend.value
    ]
    latest = {}
    for _artifact in selected:
        _label = str(_artifact["train_label"])
        _iteration = int(_artifact["checkpoint_iteration"])
        if _label not in latest or _iteration > int(
            latest[_label]["checkpoint_iteration"]
        ):
            latest[_label] = _artifact
    return latest, selected


@app.cell
def _(artifact_root, eval_backend, latest, mo, reset_mode, selected):
    _loaded = ", ".join(
        f"{_label}@{int(_artifact['checkpoint_iteration'])}"
        for _label, _artifact in latest.items()
    )
    _protocol_fields = (
        "num_envs",
        "duration_s",
        "seed",
        "settling_time_s",
        "contact_threshold_n",
        "command_profile",
        "ctrl_hz",
    )
    _protocols = {
        tuple(str(_artifact.get(_field, "missing")) for _field in _protocol_fields)
        for _artifact in selected
    }
    _protocol_status = (
        "✅ Protocol metadata matches across the selected artifacts."
        if len(_protocols) <= 1
        else "⚠️ Selected artifacts use different evaluation settings."
    )
    mo.md(
        f"""
        # Go2 policy evaluation

        Controlled fixed-command evaluation on **{eval_backend.value}** for
        **{reset_mode.value}**. The
        report uses the same command cases, seed, duration, settling period,
        contact threshold, and backend recorded in each artifact.

        **Latest checkpoints:** {_loaded or "none"}

        **Artifact directory:** `{artifact_root}`

        {_protocol_status}

        Training reward is shown only as a diagnostic. Use the physical
        scorecard below when reward terms or weights differ between runs.
        """
    )
    return


@app.cell
def _(latest, metric_mean, mo, np):
    _specs = [
        ("Survival", "survival", "%", 100.0, ".1f"),
        ("Forward RMSE", "tracking_vx_rmse", "m/s", 1.0, ".3f"),
        ("Yaw RMSE", "tracking_yaw_rmse", "rad/s", 1.0, ".3f"),
        ("Base tilt", "base_tilt_rms", "deg", 1.0, ".2f"),
        ("Trot classified", "gait_trot_classified", "%", 100.0, ".1f"),
        ("Trot RPD error", "gait_rpd_trot_error", "rad", 1.0, ".3f"),
        ("Swing clearance p95", "swing_clearance_p95_mean", "mm", 1000.0, ".1f"),
        ("Mechanical power", "mechanical_power_mean", "W", 1.0, ".1f"),
    ]
    _header = "| Metric | " + " | ".join(latest) + " |\n"
    _header += "|---|" + "---:|" * len(latest) + "\n"
    _rows = []
    for _title, _metric, _unit, _scale, _format in _specs:
        _cells = []
        for _artifact in latest.values():
            _value = metric_mean(_artifact, _metric) * _scale
            _cells.append(
                f"{_value:{_format}} {_unit}" if np.isfinite(_value) else "—"
            )
        _rows.append(f"| {_title} | " + " | ".join(_cells) + " |")
    mo.md("## Latest-checkpoint scorecard\n\n" + _header + "\n".join(_rows))
    return


@app.cell
def _(mo, plt, vitals):
    def _training_plot():
        if not vitals:
            return mo.md("_No matching `vitals.jsonl` files found._")
        _panels = (
            ("rewards/total_rewards", "total reward"),
            ("rewards/tracking_lin_vel", "tracking linear reward"),
            ("rewards/tracking_ang_vel", "tracking yaw reward"),
            ("rewards/trot_contact", "trot-contact reward"),
            ("actor/action_std", "action standard deviation"),
            ("episode_time", "episode duration"),
        )
        _fig, _axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for _axis, (_key, _title) in zip(_axes.flat, _panels):
            for _label, _columns in vitals.items():
                if _key in _columns:
                    _axis.plot(
                        _columns["iteration"],
                        _columns[_key],
                        label=_label,
                    )
            _axis.set(xlabel="training iteration", title=_title)
            _axis.grid(alpha=0.2)
        _axes.flat[0].legend(fontsize=8)
        return _fig

    mo.vstack(
        [
            mo.md(
                "## Training diagnostics\n\n"
                "Only compare reward curves where the reward definition is unchanged."
            ),
            _training_plot(),
        ]
    )
    return


@app.cell
def _(mo, selected):
    policy_io_artifacts = {
        f"{str(_artifact['train_label'])}@"
        f"{int(_artifact['checkpoint_iteration'])}": _artifact
        for _artifact in selected
        if "actor_observations" in _artifact and "policy_actions" in _artifact
    }
    _options = list(policy_io_artifacts) or ["no policy-I/O artifacts"]
    policy_io_run = mo.ui.dropdown(
        options=_options,
        value=_options[0],
        searchable=True,
        label="policy checkpoint",
    )
    mo.vstack(
        [
            mo.md(
                "## Observation and action explorer\n\n"
                "Switch between task-normalized values and task units using the "
                "scales saved with the policy's original run config. This is "
                "separate from any learned observation normalizer inside the "
                "network."
            ),
            policy_io_run,
        ]
    )
    return policy_io_artifacts, policy_io_run


@app.cell
def _(
    OriginalCfgError,
    Path,
    class_to_dict,
    component_scales_from_names,
    load_original_cfgs_from_run,
    mo,
    np,
    original_cfg_source_dir,
    policy_io_artifacts,
    policy_io_run,
):
    policy_io_artifact = policy_io_artifacts.get(policy_io_run.value)
    policy_io_catalog = {}
    policy_io_scaling_error = None
    if policy_io_artifact is not None:
        _scale_specs = (
            (
                "actor_observation_scales",
                "actor_observation_fields",
                "actor_observation_names",
            ),
            (
                "critic_observation_scales",
                "critic_observation_fields",
                "critic_observation_names",
            ),
            ("action_scales", "action_fields", "action_names"),
        )
        if any(_key not in policy_io_artifact for _key, _, _ in _scale_specs):
            try:
                _original_env_cfg, _ = load_original_cfgs_from_run(
                    str(policy_io_artifact["task"]),
                    Path(str(policy_io_artifact["checkpoint_path"])).parent,
                )
                _original_scales = class_to_dict(_original_env_cfg.scaling)
                for _scale_key, _fields_key, _names_key in _scale_specs:
                    policy_io_artifact[_scale_key] = np.asarray(
                        component_scales_from_names(
                            policy_io_artifact[_fields_key],
                            policy_io_artifact[_names_key],
                            _original_scales,
                        ),
                        dtype=np.float32,
                    )
                policy_io_artifact["policy_io_scale_source"] = str(
                    original_cfg_source_dir(
                        Path(str(policy_io_artifact["checkpoint_path"])).parent
                    )
                )
            except OriginalCfgError as _error:
                policy_io_scaling_error = str(_error)

        _groups = (
            (
                "actor_observations",
                "actor_observation_names",
                "actor_observation_scales",
                "actor obs",
                "normalized",
            ),
            (
                "critic_observations",
                "critic_observation_names",
                "critic_observation_scales",
                "critic obs",
                "normalized",
            ),
            (
                "policy_actions",
                "action_names",
                "action_scales",
                "policy output",
                "normalized",
            ),
            (
                "applied_actions",
                "action_names",
                "action_scales",
                "applied action",
                "unnormalized",
            ),
        )
        if policy_io_scaling_error is None:
            for (
                _array_key,
                _names_key,
                _scales_key,
                _group_label,
                _source_space,
            ) in _groups:
                for _index, _name in enumerate(policy_io_artifact[_names_key]):
                    _key = f"{_array_key}:{_index}"
                    policy_io_catalog[_key] = {
                        "label": f"{_group_label} | {_name}",
                        "name": str(_name),
                        "source": _array_key,
                        "source_space": _source_space,
                        "scale": float(policy_io_artifact[_scales_key][_index]),
                        "values": policy_io_artifact[_array_key][:, :, _index],
                    }

    _signal_options = {
        _series["label"]: _key for _key, _series in policy_io_catalog.items()
    }
    _default_needles = (
        "actor obs | base_lin_vel.x",
        "actor obs | phase_obs.sin",
        "policy output | dof_pos_target.FL_thigh_joint",
        "applied action | dof_pos_target.FL_thigh_joint",
    )
    _default_signals = [
        _label for _label in _default_needles if _label in _signal_options
    ]
    if not _default_signals:
        _default_signals = list(_signal_options)[:2]
    policy_io_signals = mo.ui.multiselect(
        options=_signal_options,
        value=_default_signals,
        max_selections=8,
        label="signals (up to 8)",
        full_width=True,
    )
    _commands = (
        ["all"]
        if policy_io_artifact is None or "command_case" not in policy_io_artifact
        else [
            "all",
            *dict.fromkeys(
                str(_command) for _command in policy_io_artifact["command_case"]
            ),
        ]
    )
    policy_io_command = mo.ui.dropdown(
        options=_commands,
        value=("forward_1p0" if "forward_1p0" in _commands else _commands[0]),
        label="stats/phase command",
    )
    policy_io_mode = mo.ui.dropdown(
        options={
            "time trace": "time",
            "oscillator-phase profile": "phase",
            "distribution": "distribution",
            "pairwise relationship": "relationship",
        },
        value="time trace",
        label="plot",
    )
    policy_io_space = mo.ui.dropdown(
        options={
            "task-normalized": "normalized",
            "task units": "unnormalized",
        },
        value="task-normalized",
        label="value space",
    )
    policy_io_standardize = mo.ui.checkbox(
        value=False,
        label="standardize plotted signals",
    )
    policy_io_exclude_settling = mo.ui.checkbox(
        value=True,
        label="exclude settling window from stats/patterns",
    )
    _num_envs = (
        1
        if policy_io_artifact is None
        else int(policy_io_artifact["actor_observations"].shape[1])
    )
    _default_environment = 0
    if policy_io_artifact is not None and "command_case" in policy_io_artifact:
        _forward_environments = [
            _index
            for _index, _command in enumerate(policy_io_artifact["command_case"])
            if str(_command) == "forward_1p0"
        ]
        if _forward_environments:
            _default_environment = _forward_environments[0]
    policy_io_environment = mo.ui.slider(
        0,
        max(0, _num_envs - 1),
        value=_default_environment,
        step=1,
        show_value=True,
        label="time-trace environment",
    )
    mo.vstack(
        [
            (
                mo.md(f"⚠️ Original scaling config unavailable: `{policy_io_scaling_error}`")
                if policy_io_scaling_error is not None
                else mo.md(
                    "Scaling source: "
                    f"`{policy_io_artifact.get('policy_io_scale_source', 'unknown')}`"
                    if policy_io_artifact is not None
                    else "Scaling source: no artifact selected"
                )
            ),
            policy_io_signals,
            mo.hstack(
                [
                    policy_io_command,
                    policy_io_mode,
                    policy_io_space,
                    policy_io_environment,
                    policy_io_standardize,
                    policy_io_exclude_settling,
                ],
                justify="start",
                gap=1,
            ),
        ]
    )
    return (
        policy_io_artifact,
        policy_io_catalog,
        policy_io_command,
        policy_io_environment,
        policy_io_exclude_settling,
        policy_io_mode,
        policy_io_signals,
        policy_io_space,
        policy_io_standardize,
    )


@app.cell
def _(
    first_episode_mask,
    mo,
    np,
    phase_binned_stats,
    plt,
    policy_io_artifact,
    policy_io_catalog,
    policy_io_command,
    policy_io_environment,
    policy_io_exclude_settling,
    policy_io_mode,
    policy_io_run,
    policy_io_signals,
    policy_io_space,
    policy_io_standardize,
    policy_io_in_space,
):
    def _policy_io_analysis():
        if policy_io_artifact is None:
            return mo.md(
                "_No selected artifact contains policy I/O. Regenerate it with "
                "`scripts/eval_go2_policy.py`._"
            )
        _selected_keys = list(policy_io_signals.value)
        if not _selected_keys:
            return mo.md("_Choose at least one observation or action signal._")

        _valid = first_episode_mask(policy_io_artifact["terminated"])
        _command = policy_io_command.value
        if _command != "all":
            _valid &= np.asarray(policy_io_artifact["command_case"])[None, :] == _command
        _settling_steps = 0
        if policy_io_exclude_settling.value:
            _settling_steps = int(
                round(
                    float(policy_io_artifact["settling_time_s"])
                    * float(policy_io_artifact["ctrl_hz"])
                )
            )
            _valid[:_settling_steps] = False
        if not np.any(_valid):
            return mo.md("_No valid samples remain for this command and window._")

        def _converted_values(_key):
            _series = policy_io_catalog[_key]
            return policy_io_in_space(
                _series["values"],
                _series["scale"],
                _series["source_space"],
                policy_io_space.value,
            )

        _rows = []
        for _key in _selected_keys:
            _series = policy_io_catalog[_key]
            _samples = _converted_values(_key)[_valid]
            _samples = _samples[np.isfinite(_samples)]
            if len(_samples):
                _quantiles = np.quantile(_samples, [0.05, 0.5, 0.95])
                _rows.append(
                    {
                        "signal": _series["label"],
                        "samples": int(len(_samples)),
                        "mean": f"{np.mean(_samples):.5g}",
                        "std": f"{np.std(_samples):.5g}",
                        "min": f"{np.min(_samples):.5g}",
                        "p05": f"{_quantiles[0]:.5g}",
                        "median": f"{_quantiles[1]:.5g}",
                        "p95": f"{_quantiles[2]:.5g}",
                        "max": f"{np.max(_samples):.5g}",
                    }
                )

        def _plot_values(_key):
            _values = np.asarray(_converted_values(_key), dtype=float)
            if not policy_io_standardize.value:
                return _values
            _samples = _values[_valid]
            _mean = np.nanmean(_samples)
            _std = np.nanstd(_samples)
            return (_values - _mean) / (_std if _std > 1e-12 else 1.0)

        _mode = policy_io_mode.value
        _fig, _axis = plt.subplots(
            figsize=(12.5, 5.5),
            constrained_layout=True,
            subplot_kw=({"projection": "polar"} if _mode == "phase" else None),
        )
        if _mode == "time":
            _env_index = int(policy_io_environment.value)
            _time = np.arange(_valid.shape[0]) / float(policy_io_artifact["ctrl_hz"])
            _env_valid = first_episode_mask(policy_io_artifact["terminated"])[
                :, _env_index
            ]
            _env_valid[:_settling_steps] = False
            for _key in _selected_keys:
                _values = _plot_values(_key)[:, _env_index].copy()
                _values[~_env_valid] = np.nan
                _axis.plot(_time, _values, label=policy_io_catalog[_key]["label"])
            _env_command = str(policy_io_artifact["command_case"][_env_index])
            _axis.set(
                xlabel="time (s)",
                ylabel=(
                    "z-score"
                    if policy_io_standardize.value
                    else policy_io_space.value
                ),
                title=f"{policy_io_run.value}: env {_env_index} ({_env_command})",
            )
        elif _mode == "phase":
            _actor_names = [
                str(_name) for _name in policy_io_artifact["actor_observation_names"]
            ]
            if "phase_obs.sin" not in _actor_names or "phase_obs.cos" not in _actor_names:
                plt.close(_fig)
                return mo.md("_The selected policy has no oscillator-phase observation._")
            _actor_obs = policy_io_artifact["actor_observations"]
            _actor_scales = policy_io_artifact["actor_observation_scales"]
            _sin_index = _actor_names.index("phase_obs.sin")
            _cos_index = _actor_names.index("phase_obs.cos")
            _phase = np.mod(
                np.arctan2(
                    policy_io_in_space(
                        _actor_obs[:, :, _sin_index],
                        _actor_scales[_sin_index],
                        "normalized",
                        "unnormalized",
                    ),
                    policy_io_in_space(
                        _actor_obs[:, :, _cos_index],
                        _actor_scales[_cos_index],
                        "normalized",
                        "unnormalized",
                    ),
                ),
                2.0 * np.pi,
            )
            for _key in _selected_keys:
                _centers, _mean, _std, _counts = phase_binned_stats(
                    _phase,
                    _plot_values(_key),
                    _valid,
                )
                _closed_phase = np.concatenate([_centers, _centers[:1] + 2 * np.pi])
                _closed_mean = np.concatenate([_mean, _mean[:1]])
                _closed_std = np.concatenate([_std, _std[:1]])
                _line = _axis.plot(
                    _closed_phase,
                    _closed_mean,
                    label=policy_io_catalog[_key]["label"],
                )[0]
                _axis.fill_between(
                    _closed_phase,
                    _closed_mean - _closed_std,
                    _closed_mean + _closed_std,
                    color=_line.get_color(),
                    alpha=0.12,
                )
            _axis.set_theta_zero_location("N")
            _axis.set_theta_direction(-1)
            _axis.set_title(
                f"{policy_io_run.value}: {_command} mean ± std by phase"
            )
        elif _mode == "distribution":
            for _key in _selected_keys:
                _samples = _plot_values(_key)[_valid]
                _axis.hist(
                    _samples[np.isfinite(_samples)],
                    bins=60,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=policy_io_catalog[_key]["label"],
                )
            _axis.set(
                xlabel=(
                    "z-score"
                    if policy_io_standardize.value
                    else policy_io_space.value
                ),
                ylabel="density",
                title=f"{policy_io_run.value}: {_command} distributions",
            )
        else:
            if len(_selected_keys) < 2:
                plt.close(_fig)
                return mo.md("_Choose at least two signals for a relationship plot._")
            _x_key = _selected_keys[0]
            _x = _plot_values(_x_key)[_valid]
            for _key in _selected_keys[1:]:
                _y = _plot_values(_key)[_valid]
                _finite = np.isfinite(_x) & np.isfinite(_y)
                _x_finite = _x[_finite]
                _y_finite = _y[_finite]
                if not len(_x_finite):
                    continue
                _sample_indices = np.linspace(
                    0,
                    len(_x_finite) - 1,
                    min(5000, len(_x_finite)),
                    dtype=int,
                )
                _correlation = (
                    np.corrcoef(_x_finite, _y_finite)[0, 1]
                    if np.std(_x_finite) > 0 and np.std(_y_finite) > 0
                    else np.nan
                )
                _axis.scatter(
                    _x_finite[_sample_indices],
                    _y_finite[_sample_indices],
                    s=5,
                    alpha=0.18,
                    label=f"{policy_io_catalog[_key]['label']} (r={_correlation:.3f})",
                )
            _axis.set(
                xlabel=policy_io_catalog[_x_key]["label"],
                ylabel="selected signals",
                title=f"{policy_io_run.value}: {_command} relationships",
            )
        _axis.grid(alpha=0.2)
        _axis.legend(fontsize=7, loc="best")
        return mo.vstack(
            [
                mo.ui.table(
                    _rows,
                    selection=None,
                    pagination=True,
                    page_size=8,
                    show_column_summaries=False,
                ),
                _fig,
            ]
        )

    _policy_io_analysis()
    return


@app.cell
def _(metric_mean, mo, np, plt, selected):
    def _progression_plot():
        if not selected:
            return mo.md("_No checkpoint evaluations for this protocol._")
        _specs = (
            ("survival", "survival", 100.0, "%"),
            ("forward RMSE", "tracking_vx_rmse", 1.0, "m/s"),
            ("base tilt", "base_tilt_rms", 1.0, "deg"),
            ("trot RPD error", "gait_rpd_trot_error", 1.0, "rad"),
            ("swing clearance p95", "swing_clearance_p95_mean", 1000.0, "mm"),
            ("mechanical power", "mechanical_power_mean", 1.0, "W"),
        )
        _fig, _axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        _labels = list(
            dict.fromkeys(str(_artifact["train_label"]) for _artifact in selected)
        )
        for _axis, (_title, _metric, _scale, _unit) in zip(_axes.flat, _specs):
            for _label in _labels:
                _cells = sorted(
                    (
                        _artifact
                        for _artifact in selected
                        if str(_artifact["train_label"]) == _label
                    ),
                    key=lambda _artifact: int(_artifact["checkpoint_iteration"]),
                )
                _x = [int(_cell["checkpoint_iteration"]) for _cell in _cells]
                _y = [metric_mean(_cell, _metric) * _scale for _cell in _cells]
                _valid = np.isfinite(_y)
                _axis.plot(
                    np.asarray(_x)[_valid],
                    np.asarray(_y)[_valid],
                    marker="o",
                    label=_label,
                )
            _axis.set(xlabel="checkpoint iteration", ylabel=_unit, title=_title)
            _axis.grid(alpha=0.2)
        _axes.flat[0].legend(fontsize=8)
        return _fig

    mo.vstack([mo.md("## Physical metrics over training"), _progression_plot()])
    return


@app.cell
def _(latest, mo, np, plt, steady_response):
    def _calibration_plot():
        if not latest:
            return mo.md("_No latest-checkpoint artifacts loaded._")
        _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        _axis_specs = (
            (0, "forward velocity", "m/s"),
            (1, "lateral velocity", "m/s"),
            (2, "yaw rate", "rad/s"),
        )
        for _label, _artifact in latest.items():
            _commands, _actual = steady_response(_artifact)
            for _axis, (_component, _title, _unit) in zip(_axes, _axis_specs):
                _valid = np.isfinite(_actual[:, _component])
                _axis.scatter(
                    _commands[_valid, _component],
                    _actual[_valid, _component],
                    s=18,
                    alpha=0.45,
                    label=_label,
                )
                _limits = np.asarray(_axis.get_xlim())
                _limits = np.array(
                    [
                        min(_limits[0], np.min(_commands[:, _component])),
                        max(_limits[1], np.max(_commands[:, _component])),
                    ]
                )
                _axis.plot(_limits, _limits, color="black", ls="--", lw=1)
                _axis.set(
                    xlabel=f"commanded {_unit}",
                    ylabel=f"achieved {_unit}",
                    title=_title,
                )
                _axis.grid(alpha=0.2)
        _axes[0].legend(fontsize=8)
        return _fig

    mo.vstack(
        [
            mo.md(
                "## Command calibration\n\n"
                "Each point is one rollout after the settling window; the dashed "
                "line is perfect tracking."
            ),
            _calibration_plot(),
        ]
    )
    return


@app.cell
def _(latest, mo, np, plt):
    def _phase_plot():
        _phase_cells = {
            _label: _artifact
            for _label, _artifact in latest.items()
            if "foot_contact_by_phase" in _artifact
            and "foot_clearance_by_phase" in _artifact
        }
        if not _phase_cells:
            return mo.md("_No phase-binned foot artifacts found._")
        _commands = (
            ("forward_1p0", "1 m/s forward"),
            ("forward_3p0", "3 m/s forward"),
        )
        _num_columns = len(_commands) * len(_phase_cells)
        _fig, _axes = plt.subplots(
            3,
            _num_columns,
            figsize=(4.4 * _num_columns, 12.5),
            squeeze=False,
            constrained_layout=True,
            subplot_kw={"projection": "polar"},
        )
        _foot_colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
        _force_values = []
        _clearance_values = []
        for _policy_index, (_label, _artifact) in enumerate(_phase_cells.items()):
            _phase = np.asarray(_artifact["gait_phase_bin_centers"])
            _closed_phase = np.concatenate([_phase, _phase[:1] + 2.0 * np.pi])
            for _speed_index, (_command, _speed_label) in enumerate(_commands):
                _column = _policy_index * len(_commands) + _speed_index
                _mask = _artifact["command_case"] == _command
                if not np.any(_mask):
                    _axes[0, _column].set_title(
                        f"{_label}: {_speed_label}\n(no command samples)"
                    )
                    continue
                _has_force = "foot_contact_force_z_by_phase" in _artifact
                for _foot_index, _foot_name in enumerate(_artifact["foot_names"]):
                    _contact = np.nanmean(
                        _artifact["foot_contact_by_phase"][_mask, _foot_index],
                        axis=0,
                    )
                    _clearance = 1000.0 * np.nanmean(
                        _artifact["foot_clearance_by_phase"][_mask, _foot_index],
                        axis=0,
                    )
                    _closed_contact = np.concatenate([_contact, _contact[:1]])
                    _closed_clearance = np.concatenate(
                        [_clearance, _clearance[:1]]
                    )
                    _clearance_values.append(
                        _closed_clearance[np.isfinite(_closed_clearance)]
                    )
                    _axes[0, _column].plot(
                        _closed_phase,
                        _closed_contact,
                        color=_foot_colors[_foot_index],
                        label=str(_foot_name).replace("_foot", ""),
                    )
                    if _has_force:
                        _force_samples = _artifact[
                            "foot_contact_force_z_by_phase"
                        ][_mask, _foot_index]
                        _force_mean = np.nanmean(_force_samples, axis=0)
                        _force_std = np.nanstd(_force_samples, axis=0)
                        _closed_force_mean = np.concatenate(
                            [_force_mean, _force_mean[:1]]
                        )
                        _closed_force_std = np.concatenate(
                            [_force_std, _force_std[:1]]
                        )
                        _force_lower = _closed_force_mean - _closed_force_std
                        _force_upper = _closed_force_mean + _closed_force_std
                        _force_values.extend(
                            [
                                _force_lower[np.isfinite(_force_lower)],
                                _force_upper[np.isfinite(_force_upper)],
                            ]
                        )
                        _axes[1, _column].plot(
                            _closed_phase,
                            _closed_force_mean,
                            color=_foot_colors[_foot_index],
                        )
                        _axes[1, _column].fill_between(
                            _closed_phase,
                            _force_lower,
                            _force_upper,
                            color=_foot_colors[_foot_index],
                            alpha=0.14,
                        )
                    _axes[2, _column].plot(
                        _closed_phase,
                        _closed_clearance,
                        color=_foot_colors[_foot_index],
                    )
                if not _has_force:
                    _axes[1, _column].text(
                        0.5,
                        0.5,
                        "regenerate artifact\nfor force data",
                        transform=_axes[1, _column].transAxes,
                        ha="center",
                        va="center",
                    )
                _axes[0, _column].set_title(f"{_label}: {_speed_label}")

        _nonempty_force = [_values for _values in _force_values if len(_values)]
        if _nonempty_force:
            _finite_force = np.concatenate(_nonempty_force)
            _force_min = min(0.0, float(np.min(_finite_force)))
            _force_max = max(0.0, float(np.max(_finite_force)))
        else:
            _force_min, _force_max = 0.0, 1.0
        _force_padding = max(1.0, 0.05 * (_force_max - _force_min))
        _nonempty_clearance = [
            _values for _values in _clearance_values if len(_values)
        ]
        if _nonempty_clearance:
            _finite_clearance = np.concatenate(_nonempty_clearance)
            _clearance_min = min(0.0, float(np.min(_finite_clearance)))
            _clearance_max = max(0.0, float(np.max(_finite_clearance)))
        else:
            _clearance_min, _clearance_max = 0.0, 1.0
        _clearance_padding = max(
            5.0,
            0.05 * (_clearance_max - _clearance_min),
        )
        _zero_phase = np.linspace(0.0, 2.0 * np.pi, 257)
        for _column in range(_num_columns):
            for _axis in _axes[:, _column]:
                _axis.axvspan(0.0, np.pi, color="#dddddd", alpha=0.4)
                _axis.axvline(np.pi, color="black", ls="--", lw=1)
                _axis.set_theta_zero_location("N")
                _axis.set_theta_direction(-1)
                _axis.set_thetagrids(
                    [0, 90, 180, 270],
                    labels=["0", "π/2", "π", "3π/2"],
                )
                _axis.set_rlabel_position(135)
                _axis.grid(alpha=0.2)
            _axes[0, _column].set_ylim(0.0, 1.0)
            _axes[0, _column].set_rticks([0.0, 0.5, 1.0])
            _axes[1, _column].set_ylim(
                _force_min - _force_padding,
                _force_max + _force_padding,
            )
            _axes[1, _column].plot(
                _zero_phase,
                np.zeros_like(_zero_phase),
                color="black",
                ls=":",
                lw=1,
            )
            _axes[2, _column].set_ylim(
                _clearance_min - _clearance_padding,
                _clearance_max + _clearance_padding,
            )
            _axes[2, _column].plot(
                _zero_phase,
                np.zeros_like(_zero_phase),
                color="black",
                ls=":",
                lw=1,
            )
        _handles, _labels = _axes[0, 0].get_legend_handles_labels()
        if _labels:
            _fig.legend(
                _handles,
                _labels,
                loc="upper center",
                ncol=len(_labels),
                bbox_to_anchor=(0.5, 1.03),
            )
        _axes[0, 0].set_ylabel("contact probability", labelpad=42)
        _axes[1, 0].set_ylabel(
            "vertical foot contact force (N)",
            labelpad=42,
        )
        _axes[2, 0].set_ylabel(
            "clearance above stance height (mm)",
            labelpad=42,
        )
        return _fig

    mo.vstack(
        [
            mo.md(
                "## Foot contact, force, and clearance by oscillator phase\n\n"
                "Polar angle is local leg phase, shown for both 1 and 3 m/s "
                "forward commands. Gray is the Go2 reference stance half-cycle. "
                "The middle row is mean vertical foot contact force; translucent "
                "bands show ±1 standard deviation across evaluated environments. "
                "Clearance is foot height minus that foot's median stance height, "
                "so asset offsets do not masquerade as lift. Dotted radial lines "
                "mark zero force and zero clearance."
            ),
            _phase_plot(),
        ]
    )
    return


@app.cell
def _(latest, mo, np, plt):
    def _stability_plot():
        if not latest:
            return mo.md("_No latest-checkpoint artifacts loaded._")
        _specs = (
            ("metric_base_height_std", "base-height std", 1000.0, "mm"),
            ("metric_base_tilt_rms", "base tilt", 1.0, "deg"),
            (
                "metric_swing_clearance_p95_mean",
                "swing clearance p95",
                1000.0,
                "mm",
            ),
        )
        _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for _axis, (_key, _title, _scale, _unit) in zip(_axes, _specs):
            _values = []
            _names = []
            for _label, _artifact in latest.items():
                if _key not in _artifact:
                    continue
                _finite = _artifact[_key][np.isfinite(_artifact[_key])] * _scale
                if len(_finite):
                    _values.append(_finite)
                    _names.append(_label)
            if _values:
                _axis.violinplot(_values, showmedians=True, showextrema=True)
                _axis.set_xticks(range(1, len(_names) + 1), _names, rotation=20)
            _axis.set(ylabel=_unit, title=_title)
            _axis.grid(axis="y", alpha=0.2)
        return _fig

    mo.vstack([mo.md("## Stability and swing distributions"), _stability_plot()])
    return


@app.cell
def _(metric_mean, mo, np, plt, selected):
    def _tradeoff_plot():
        if not selected:
            return mo.md("_No checkpoint evaluations for this protocol._")
        _fig, _axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
        _labels = list(
            dict.fromkeys(str(_artifact["train_label"]) for _artifact in selected)
        )
        for _label in _labels:
            _cells = sorted(
                (
                    _artifact
                    for _artifact in selected
                    if str(_artifact["train_label"]) == _label
                ),
                key=lambda _artifact: int(_artifact["checkpoint_iteration"]),
            )
            _tracking = np.asarray(
                [
                    metric_mean(_cell, "tracking_vx_rmse", "forward_1p0")
                    for _cell in _cells
                ]
            )
            _power = np.asarray(
                [
                    metric_mean(_cell, "mechanical_power_mean", "forward_1p0")
                    for _cell in _cells
                ]
            )
            _valid = np.isfinite(_tracking) & np.isfinite(_power)
            _axis.plot(
                _tracking[_valid],
                _power[_valid],
                marker="o",
                label=_label,
            )
            for _cell, _x, _y, _show in zip(_cells, _tracking, _power, _valid):
                if _show:
                    _axis.annotate(
                        str(int(_cell["checkpoint_iteration"])),
                        (_x, _y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
        _axis.set(
            xlabel="1 m/s forward tracking RMSE (m/s; lower is better)",
            ylabel="mechanical power (W; lower is better)",
            title="tracking–effort tradeoff by checkpoint",
        )
        _axis.grid(alpha=0.2)
        _axis.legend(fontsize=8)
        return _fig

    mo.vstack([mo.md("## Tracking versus effort"), _tradeoff_plot()])
    return


@app.cell
def _(latest, mo):
    _options = list(latest) or ["baseline"]
    _default = "baseline" if "baseline" in _options else _options[0]
    reference_policy = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="before/reference policy",
    )
    reference_policy
    return (reference_policy,)


@app.cell
def _(
    LoggedConfigError,
    Path,
    diff_logged_run_configs,
    latest,
    mo,
    reference_policy,
):
    _reference_label = reference_policy.value
    _reference = latest.get(_reference_label)
    _sections = []
    if _reference is None:
        _sections.append("_No reference policy artifact is loaded._")
    else:
        _reference_checkpoint = str(_reference["checkpoint_path"])
        _reference_run = Path(_reference_checkpoint).parent.name
        for _label, _artifact in latest.items():
            if _label == _reference_label:
                continue
            _checkpoint = str(_artifact["checkpoint_path"])
            _run = Path(_checkpoint).parent.name
            _sections.append(
                f"### `{_reference_label}` ({_reference_run}) → "
                f"`{_label}` ({_run})"
            )
            try:
                _changes = diff_logged_run_configs(
                    _reference_checkpoint,
                    _checkpoint,
                    "go2",
                )
            except LoggedConfigError as _error:
                _sections.append(f"⚠️ {_error}")
                continue
            if not _changes:
                _sections.append("✅ The saved task and inherited configs are identical.")
                continue
            _addition_count = sum(_change.additions for _change in _changes)
            _deletion_count = sum(_change.deletions for _change in _changes)
            _sections.append(
                f"**{len(_changes)} config file(s) changed: "
                f"+{_addition_count} / −{_deletion_count} lines.**"
            )
            for _change in _changes:
                _sections.extend(
                    [
                        f"#### `{_change.path}` — {_change.status}, "
                        f"+{_change.additions} / −{_change.deletions}",
                        f"```diff\n{_change.unified_diff}\n```",
                    ]
                )
    if not _sections:
        _sections.append("_Load at least two labeled policies to compare configs._")
    mo.md(
        "## Saved training-config differences\n\n"
        "Compared directly from each run's saved source snapshot. Red lines were "
        "present only in the reference run; green lines are in the comparison "
        "run. Saved code is read as text and is not executed. This shows config "
        "source defaults; command-line or W&B overrides are not reconstructed.\n\n"
        + "\n\n".join(_sections)
    )
    return


@app.cell
def _(latest, metric_mean, mo, np, reference_policy):
    _reference = latest.get(reference_policy.value)
    _specs = (
        ("Survival", "survival", 100.0, "%", "higher"),
        ("Forward RMSE", "tracking_vx_rmse", 1.0, "m/s", "lower"),
        ("Yaw RMSE", "tracking_yaw_rmse", 1.0, "rad/s", "lower"),
        ("Base tilt", "base_tilt_rms", 1.0, "deg", "lower"),
        ("Trot RPD error", "gait_rpd_trot_error", 1.0, "rad", "lower"),
        ("Swing clearance", "swing_clearance_p95_mean", 1000.0, "mm", "context"),
        ("Power", "mechanical_power_mean", 1.0, "W", "lower"),
    )
    _rows = []
    if _reference is not None:
        for _title, _metric, _scale, _unit, _direction in _specs:
            _before = metric_mean(_reference, _metric) * _scale
            for _label, _artifact in latest.items():
                if _label == reference_policy.value:
                    continue
                _after = metric_mean(_artifact, _metric) * _scale
                _delta = _after - _before
                if _direction == "lower":
                    _interpretation = -_delta
                    _meaning = "improvement"
                elif _direction == "higher":
                    _interpretation = _delta
                    _meaning = "improvement"
                else:
                    _interpretation = _delta
                    _meaning = "change"
                if np.isfinite(_before) and np.isfinite(_after):
                    _rows.append(
                        f"| {_label} | {_title} | {_before:.3g} | {_after:.3g} | "
                        f"{_interpretation:+.3g} {_unit} {_meaning} |"
                    )
    _table = "\n".join(_rows) or "| — | No comparable policies loaded | — | — | — |"
    mo.md(
        f"""
        ## Before/after deltas

        Positive **improvement** respects whether a metric should rise or fall;
        swing clearance is reported as an unsigned design **change**.

        | Policy | Metric | Before | After | Direction-aware delta |
        |---|---|---:|---:|---:|
        {_table}
        """
    )
    return


if __name__ == "__main__":
    app.run()
