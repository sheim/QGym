"""Native-backend Mini Cheetah tuning scorecard.

Run from the repository root:

    uv run marimo edit notebooks/mini_cheetah_tuning.py
    uv run marimo run notebooks/mini_cheetah_tuning.py
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
    artifact_root = Path("logs/mc_ref_hardware/baseline_b32768")
    disturbance_root = Path("logs/mc_ref_disturbance/baseline_b32768_360")
    backend_colors = {"mujoco": "#4C78A8", "vsim": "#E45756"}

    def _load_pair(backend, suffix):
        _matches = sorted(artifact_root.glob(f"{backend}-*__{suffix}.npz"))
        if not _matches:
            return None
        _data = np.load(_matches[-1], allow_pickle=False)
        return {_key: _data[_key] for _key in _data.files}

    native = {
        _backend: _load_pair(_backend, "nominal") for _backend in ("mujoco", "vsim")
    }
    native = {_key: _value for _key, _value in native.items() if _value is not None}

    disturbance = {}
    disturbance_pre = {}
    for _backend in ("mujoco", "vsim"):
        _points = []
        _pre_points = []
        for _path in sorted(
            disturbance_root.glob(f"{_backend}-*__impulse_*.summary.json")
        ):
            _summary = json.loads(_path.read_text())
            _magnitude = _summary["protocol"]["velocity_impulse_m_per_s"]
            _failure = _summary["results"]["overall"]["disturbance_failure"]["mean"]
            _pre_failure = _summary["results"]["overall"][
                "disturbance_pre_impulse_failure"
            ]["mean"]
            _points.append((_magnitude, _failure))
            _pre_points.append((_magnitude, _pre_failure))
        disturbance[_backend] = sorted(_points)
        disturbance_pre[_backend] = sorted(_pre_points)
    return (
        artifact_root,
        backend_colors,
        disturbance,
        disturbance_pre,
        disturbance_root,
        native,
    )


@app.cell
def _(artifact_root, disturbance_root, mo, native):
    mo.md(
        f"""
        # Mini Cheetah native-backend tuning scorecard

        This report separates policy quality from training reward. It follows
        the evaluation ideas in
        [Zhang et al. (2024)](https://arxiv.org/abs/2402.08662):

        - Fig. 3-style body-weight-normalized load distributions for all feet;
        - touchdown relative phase difference (RPD), with ideal trot
          `(LF, RH, LH) = (π, π, 0)` relative to RF;
        - Table I-style phase-staggered planar velocity-impulse failure rates.

        It additionally measures base-height steadiness and 500 Hz torque/joint
        velocity spectra for motor safety.

        **Loaded native artifacts:** {len(native)}/2

        **Native path:** `{artifact_root}`

        **Disturbance path:** `{disturbance_root}`
        """
    )
    return


@app.cell
def _(mo, native, np):
    def _mean(backend, metric, command=None):
        _cell = native.get(backend)
        if _cell is None:
            return np.nan
        _values = _cell[f"metric_{metric}"]
        _mask = np.isfinite(_values)
        if command is not None:
            _mask &= _cell["command_case"] == command
        return float(np.mean(_values[_mask])) if np.any(_mask) else np.nan

    _rows = []
    _specs = [
        ("Survival", "survival", "", 100.0, ".1f"),
        ("Base-height std", "base_height_std", "mm", 1000.0, ".1f"),
        ("Base-height range", "base_height_range", "mm", 1000.0, ".1f"),
        ("Torque power >10 Hz", "torque_fft_high_frequency_ratio", "%", 100.0, ".1f"),
        (
            "Joint-velocity power >10 Hz",
            "joint_velocity_fft_high_frequency_ratio",
            "%",
            100.0,
            ".1f",
        ),
        ("Moving trials classified trot", "gait_trot_classified", "%", 100.0, ".1f"),
        ("Trot RPD error", "gait_rpd_trot_error", "rad", 1.0, ".3f"),
        ("GRF balance CV", "grf_balance_cv", "", 1.0, ".3f"),
    ]
    for _label, _metric, _unit, _scale, _format in _specs:
        _values = [_mean(_backend, _metric) * _scale for _backend in ("mujoco", "vsim")]
        _rows.append(
            f"| {_label} | {_values[0]:{_format}} {_unit} | "
            f"{_values[1]:{_format}} {_unit} |"
        )
    mo.md(
        """
        ## Headline native metrics

        | Metric | MuJoCo Warp | VSim |
        |---|---:|---:|
        """
        + "\n".join(_rows)
        + """

        The FFT cutoff is a diagnostic, not yet a motor-certified limit.
        Gait harmonics and contact impulses legitimately contribute broadband
        power; tune against the full PSD and worst joint, then finalize limits
        from actuator/controller data.
        """
    )
    return


@app.cell
def _(mo, native):
    _first = next(iter(native.values()), None)
    _joint_names = (
        [str(_name) for _name in _first["actuated_dof_names"]]
        if _first is not None
        else []
    )
    spectrum_joint = mo.ui.dropdown(
        options=_joint_names,
        value=(_joint_names[0] if _joint_names else None),
        label="Joint spectrum",
    )
    spectrum_joint
    return (spectrum_joint,)


@app.cell
def _(backend_colors, mo, native, np, plt, spectrum_joint):
    def _spectral_plot():
        if not native or spectrum_joint.value is None:
            return mo.md("_No spectral artifacts found._")
        _fig, _axes = plt.subplots(
            1,
            2,
            figsize=(14, 4.5),
            constrained_layout=True,
        )
        for _backend, _cell in native.items():
            _names = [str(_name) for _name in _cell["actuated_dof_names"]]
            _joint = _names.index(spectrum_joint.value)
            _frequency = _cell["fft_frequency_hz"]
            for _ax, _key, _title, _unit in (
                (_axes[0], "torque_psd_by_joint", "joint torque", "(N m)²/Hz"),
                (
                    _axes[1],
                    "joint_velocity_psd_by_joint",
                    "joint velocity",
                    "(rad/s)²/Hz",
                ),
            ):
                _ax.semilogy(
                    _frequency[1:],
                    np.maximum(_cell[_key][_joint, 1:], 1e-12),
                    label=_backend,
                    color=backend_colors[_backend],
                    lw=1.4,
                )
                _ax.set(
                    xlabel="frequency (Hz)",
                    ylabel=f"PSD {_unit}",
                    title=f"{spectrum_joint.value}: {_title}",
                    xlim=(0, min(100, float(_frequency[-1]))),
                )
        for _ax in _axes:
            _ax.axvline(2.5, color="#777777", ls=":", lw=1, label="gait 2.5 Hz")
            _ax.axvline(10.0, color="#222222", ls="--", lw=1, label="HF cutoff")
            _ax.grid(alpha=0.2)
        _axes[0].legend(fontsize=8)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## Motor spectra at the 500 Hz PD/physics rate

                PSDs use detrended, Hann-windowed signals after the 0.5-second
                settling period. Scalar metrics include spectral centroid,
                dominant frequency, fraction above 10 Hz, gait-band fraction,
                and the worst single-joint high-frequency fraction.
                """
            ),
            _spectral_plot(),
        ]
    )
    return


@app.cell
def _(backend_colors, mo, native, plt):
    _fig, _axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.3),
        constrained_layout=True,
    )
    for _backend, _cell in native.items():
        _height_violin = _axes[0].violinplot(
            1000.0 * _cell["metric_base_height_std"],
            positions=[list(native).index(_backend)],
            widths=0.75,
            showmedians=True,
        )
        _range_violin = _axes[1].violinplot(
            1000.0 * _cell["metric_base_height_range"],
            positions=[list(native).index(_backend)],
            widths=0.75,
            showmedians=True,
        )
        for _violin in (_height_violin, _range_violin):
            for _body in _violin["bodies"]:
                _body.set_facecolor(backend_colors[_backend])
                _body.set_alpha(0.65)
    for _ax, _title in zip(
        _axes,
        ("base-height standard deviation", "base-height peak-to-peak range"),
    ):
        _ax.set(
            xticks=range(len(native)),
            xticklabels=list(native),
            ylabel="mm",
            title=_title,
        )
        _ax.grid(axis="y", alpha=0.2)
    mo.vstack(
        [
            mo.md(
                """
                ## Survival with base-height steadiness

                Height is evaluated by mean, standard deviation, range, and
                drift. Only the variability metrics are selection objectives;
                the mean height is context and need not equal the reward target.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(backend_colors, mo, native, plt):
    def _grf_plot():
        if not native:
            return mo.md("_No GRF artifacts found._")
        _fig, _axes = plt.subplots(
            1,
            len(native),
            figsize=(13, 4.6),
            squeeze=False,
            constrained_layout=True,
        )
        for _ax, (_backend, _cell) in zip(_axes[0], native.items()):
            _mask = _cell["command_case"] == "forward_1p0"
            _loads = _cell["grf_body_weight_by_foot"][_mask]
            _violins = _ax.violinplot(
                [_loads[:, _foot] for _foot in range(_loads.shape[1])],
                showmeans=True,
                showextrema=True,
            )
            for _body in _violins["bodies"]:
                _body.set_facecolor(backend_colors[_backend])
                _body.set_alpha(0.65)
            _ax.axhline(0.25, color="#222222", ls="--", lw=1, label="balanced")
            _ax.set(
                xticks=range(1, len(_cell["foot_names"]) + 1),
                xticklabels=[
                    str(_name).replace("_foot", "") for _name in _cell["foot_names"]
                ],
                ylabel="episode-average vertical GRF / body weight",
                title=f"{_backend}: 1 m/s forward",
            )
            _ax.grid(axis="y", alpha=0.2)
            _ax.legend(fontsize=8)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## Balanced leg use — analogous to paper Fig. 3

                Every point is one rollout's time-averaged vertical GRF for one
                leg, normalized by nominal robot weight. Balanced cyclic use is
                centered near 0.25 on all four legs.
                """
            ),
            _grf_plot(),
        ]
    )
    return


@app.cell
def _(backend_colors, mo, native, np, plt):
    def _gait_plot():
        if not native:
            return mo.md("_No gait artifacts found._")
        _fig, _axes = plt.subplots(
            1,
            len(native),
            figsize=(13, 4.8),
            squeeze=False,
            constrained_layout=True,
        )
        _ideal = np.asarray([np.pi, np.pi, 0.0])
        for _ax, (_backend, _cell) in zip(_axes[0], native.items()):
            _mask = _cell["command_case"] == "forward_1p0"
            _rpd = _cell["gait_rpd_mean"][_mask]
            for _component in range(3):
                _values = _rpd[:, _component]
                _values = _values[np.isfinite(_values)]
                _jitter = np.linspace(-0.12, 0.12, max(len(_values), 1))
                _ax.scatter(
                    _component + _jitter[: len(_values)],
                    _values,
                    s=18,
                    alpha=0.6,
                    color=backend_colors[_backend],
                )
                _ax.scatter(
                    [_component],
                    [_ideal[_component]],
                    marker="x",
                    s=90,
                    color="black",
                    linewidths=2,
                )
            _ax.set(
                xticks=range(3),
                xticklabels=["LF−RF", "RH−RF", "LH−RF"],
                yticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                yticklabels=["0", "π/2", "π", "3π/2", "2π"],
                ylabel="touchdown relative phase",
                title=f"{_backend}: 1 m/s forward",
                ylim=(-0.15, 2 * np.pi + 0.15),
            )
            _ax.grid(alpha=0.2)
        return _fig

    mo.vstack(
        [
            mo.md(
                """
                ## Trot consistency from touchdown RPD

                Touchdowns are debounced by half a nominal stance period.
                RPD is computed for each complete RF cycle and circularly
                averaged. Black crosses mark ideal trot `(π, π, 0)`.
                """
            ),
            _gait_plot(),
        ]
    )
    return


@app.cell
def _(backend_colors, disturbance, disturbance_pre, mo, plt):
    _magnitudes = sorted(
        {
            _magnitude
            for _points in disturbance.values()
            for _magnitude, _failure in _points
        }
    )
    _fig, _ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for _backend, _points in disturbance.items():
        if not _points:
            continue
        _x, _y = zip(*_points)
        _ax.plot(
            _x,
            [100.0 * _value for _value in _y],
            marker="o",
            label=_backend,
            color=backend_colors[_backend],
            lw=1.7,
        )
    _ax.set(
        xlabel="planar velocity impulse magnitude (m/s)",
        ylabel="conditional post-impulse failure rate (%)",
        title="phase- and direction-staggered disturbance rejection",
        xticks=_magnitudes,
        ylim=(0, 100),
    )
    _ax.grid(alpha=0.2)
    _ax.legend()

    _rows = []
    for _magnitude in _magnitudes:
        _values = []
        for _backend in ("mujoco", "vsim"):
            _lookup = dict(disturbance.get(_backend, []))
            _values.append(100.0 * _lookup.get(_magnitude, float("nan")))
        _rows.append(f"| {_magnitude:.1f} | {_values[0]:.1f}% | {_values[1]:.1f}% |")
    _pre_rows = []
    for _backend in ("mujoco", "vsim"):
        _values = [100.0 * _value for _, _value in disturbance_pre.get(_backend, [])]
        _pre_rate = sum(_values) / len(_values) if _values else float("nan")
        _pre_rows.append(f"| {_backend} | {_pre_rate:.1f}% |")
    mo.vstack(
        [
            mo.md(
                """
                ## Disturbance rejection — analogous to paper Table I

                These preliminary baselines use 360 trials per magnitude:
                36 directions and ten impulse times distributed across one
                0.5-second phase window. The production wrapper defaults to all
                1,800 direction/time pairs used by the paper-style protocol.
                The plotted rates are conditional on surviving until the
                impulse, keeping failures during the five-second settling
                period separate.

                | Impulse (m/s) | MuJoCo post | VSim post |
                |---:|---:|---:|
                """
                + "\n".join(_rows)
                + """

                | Backend | Pre-impulse settling failure |
                |---|---:|
                """
                + "\n".join(_pre_rows)
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Selection implications

    - Require survival first, then check height steadiness; a crouched but
      steady gait is acceptable, an oscillating base is not.
    - Use FFT ratios and full per-joint PSDs together. The 10 Hz split is a
      comparison feature until motor/controller frequency limits are known.
    - Require the 1 m/s forward case to classify as trot with low RPD
      variation and four comparable GRF distributions.
    - Use the impulse failure curve as a hard robustness discriminator
      before selecting a checkpoint for hardware.
    """)
    return


if __name__ == "__main__":
    app.run()
