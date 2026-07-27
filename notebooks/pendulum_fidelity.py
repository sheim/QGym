"""Interactive inspection of the pendulum cross-engine physics-fidelity run.

Loads the npz dumps produced by ``scripts/pendulum_fidelity.py run`` (one per
backend, under ``logs/fidelity/``) and lets you scrub the initial-condition grid
to compare per-engine trajectories, plus aggregate divergence views.

    uv run marimo edit notebooks/pendulum_fidelity.py      # interactive
    uv run marimo run  notebooks/pendulum_fidelity.py      # read-only app

Generate the data first (see that script's docstring).  mujoco-cpu is the
deterministic reference every divergence is measured against.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Mirror scripts/pendulum_fidelity.py (pendulum.urdf, joint frame).
    I_JOINT, MGL = 1.0267, 9.81

    def wrap(x):
        """Wrap angle(s) to [-pi, pi]."""
        return np.arctan2(np.sin(x), np.cos(x))

    return I_JOINT, MGL, Path, mo, np, plt, wrap


@app.cell
def _(Path, mo, np):
    fdir = Path("logs/fidelity")
    files = sorted(fdir.glob("*.npz")) if fdir.exists() else []
    data = {}
    for _f in files:
        _d = np.load(_f, allow_pickle=True)
        data[str(_d["label"])] = {k: _d[k] for k in _d.files}

    # Prefer mujoco-cpu as the reference; fall back to whatever loaded first.
    ref_label = next((k for k in data if k.startswith("mujoco-cpu")), None)
    if ref_label is None and data:
        ref_label = next(iter(data))

    mo.stop(
        not data,
        mo.md(
            f"**No data in `{fdir}/`.** Run `scripts/pendulum_fidelity.py run` for "
            "each backend first (see this notebook's docstring)."
        ),
    )

    n_steps, n_envs = data[ref_label]["theta"].shape
    grid_pts = int(round(n_envs**0.5))
    ctrl_hz = float(data[ref_label]["ctrl_hz"])
    t = np.arange(n_steps) / ctrl_hz
    # reset_to_uniform grid = cartesian_prod(lin_pos, lin_vel):
    #   env e -> theta index e // grid_pts, omega index e % grid_pts
    lin_pos = np.linspace(-np.pi, np.pi, grid_pts)
    lin_vel = np.linspace(-5.0, 5.0, grid_pts)
    return ctrl_hz, data, grid_pts, lin_pos, lin_vel, n_envs, ref_label, t


@app.cell
def _(data, mo, n_envs, ref_label):
    mo.md(
        f"""
        # Pendulum physics-fidelity

        Engines loaded: **{", ".join(data)}**  ·  reference: **{ref_label}**
        ·  grid: **{n_envs} envs**.  θ = 0 is upright; divergence is the wrapped
        angle difference vs the reference (pure physics, identical control law).
        """
    )
    return


@app.cell
def _(grid_pts, mo):
    theta_idx = mo.ui.slider(0, grid_pts - 1, value=grid_pts - 1, label="θ₀ index")
    omega_idx = mo.ui.slider(0, grid_pts - 1, value=grid_pts // 2, label="ω₀ index")
    mo.vstack(
        [
            mo.md("**Pick an initial condition** (θ₀ ∈ [−π, π], ω₀ ∈ [−5, 5]):"),
            theta_idx,
            omega_idx,
        ]
    )
    return omega_idx, theta_idx


@app.cell
def _(lin_pos, lin_vel, mo, omega_idx, theta_idx):
    mo.md(
        f"Selected IC: **θ₀ = {lin_pos[theta_idx.value]:+.3f} rad**, "
        f"**ω₀ = {lin_vel[omega_idx.value]:+.3f} rad/s**"
    )
    return


@app.cell
def _(data, mo):
    # The engines nearly coincide, so give each a decreasing width + distinct
    # dash so an underlying curve shows from beneath the one on top, and let the
    # user toggle any of them off.
    _order = list(data)
    _dashes = ["-", "--", ":", "-."]

    def estyle(label):
        i = _order.index(label)
        return {
            "lw": max(0.9, 3.6 - 1.3 * i),
            "ls": _dashes[i % len(_dashes)],
            "zorder": i + 1,
        }

    engine_sel = mo.ui.multiselect(options=_order, value=_order, label="show engines")
    mo.hstack([mo.md("**Toggle engines:**"), engine_sel], justify="start", gap=1)
    return engine_sel, estyle


@app.cell
def _(data, engine_sel, estyle, grid_pts, omega_idx, plt, ref_label, t, theta_idx, wrap):
    def _plot():
        e = theta_idx.value * grid_pts + omega_idx.value
        fig, ((ax_th, ax_ph), (ax_tau, ax_div)) = plt.subplots(
            2, 2, figsize=(11, 6.5), constrained_layout=True
        )
        for label, d in data.items():
            if label not in engine_sel.value:
                continue
            st = estyle(label)
            th = wrap(d["theta"][:, e])
            ax_th.plot(t, th, label=label, **st)
            ax_ph.plot(th, d["omega"][:, e], label=label, **st)
            ax_tau.plot(t, d["tau"][:, e], label=label, **st)
            if label != ref_label:
                div = wrap(d["theta"][:, e] - data[ref_label]["theta"][:, e])
                ax_div.plot(t, div, label=f"{label} − {ref_label}", **st)

        ax_th.axhline(0, color="k", lw=0.5, ls=":")
        ax_th.set(xlabel="t (s)", ylabel="θ (rad, wrapped)", title="Angle vs time")
        ax_th.legend(fontsize=8)
        ax_ph.set(xlabel="θ (rad)", ylabel="ω (rad/s)", title="Phase portrait")
        ax_tau.axhline(5, color="r", lw=0.5, ls=":")
        ax_tau.axhline(-5, color="r", lw=0.5, ls=":")
        ax_tau.set(xlabel="t (s)", ylabel="τ (N·m)", title="Control torque (±5 limit)")
        ax_div.set(xlabel="t (s)", ylabel="Δθ (rad)", title=f"Divergence vs {ref_label}")
        ax_div.legend(fontsize=8)
        return fig

    _plot()
    return


@app.cell
def _(data, engine_sel, estyle, mo, np, plt, ref_label, t, wrap):
    def _plot():
        fig, (ax_rms, ax_catch) = plt.subplots(
            1, 2, figsize=(11, 3.6), constrained_layout=True
        )
        for label, d in data.items():
            if label != ref_label and label in engine_sel.value:
                div = wrap(d["theta"] - data[ref_label]["theta"])  # [T, N]
                ax_rms.plot(
                    t, np.sqrt(np.mean(div**2, axis=1)), label=label, **estyle(label)
                )
        ax_rms.set(
            xlabel="t (s)",
            ylabel="RMS Δθ over grid (rad)",
            title=f"Mean divergence vs {ref_label}",
            yscale="log",
        )
        ax_rms.legend(fontsize=8)

        for label, d in data.items():
            if label not in engine_sel.value:
                continue
            tw = wrap(d["theta"])
            hold = int(round(float(d["ctrl_hz"])))
            up = (np.abs(tw[-hold:]) < 0.14).all(0) & (
                np.abs(d["omega"][-hold:]) < 0.5
            ).all(0)
            first = np.argmax(d["mode"], axis=0) / float(d["ctrl_hz"])
            ax_catch.hist(first[up], bins=30, alpha=0.5, label=f"{label} ({up.sum()})")
        ax_catch.set(
            xlabel="time to catch (s)", ylabel="# envs", title="Catch-time distribution"
        )
        ax_catch.legend(fontsize=8)
        return fig

    mo.vstack([mo.md("## Aggregate over the grid"), _plot()])
    return


@app.cell
def _(data, engine_sel, grid_pts, lin_vel, mo, np, plt, ref_label, wrap):
    def _plot():
        others = [
            label
            for label in data
            if label != ref_label and label in engine_sel.value
        ]
        if not others:
            return mo.md("_Only the reference engine is loaded — no heatmap._")
        short = int(round(1.5 * float(data[ref_label]["ctrl_hz"])))
        fig, axes = plt.subplots(
            1,
            len(others),
            figsize=(5.2 * len(others), 4.2),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, label in zip(axes[0], others):
            div = wrap(data[label]["theta"][:short] - data[ref_label]["theta"][:short])
            rms = np.sqrt(np.mean(div**2, axis=0)).reshape(grid_pts, grid_pts)
            im = ax.imshow(
                rms,
                origin="lower",
                aspect="auto",
                extent=[lin_vel[0], lin_vel[-1], -np.pi, np.pi],
                cmap="viridis",
            )
            ax.set(
                xlabel="ω₀ (rad/s)",
                ylabel="θ₀ (rad)",
                title=f"{label} short-horizon RMS Δθ",
            )
            fig.colorbar(im, ax=ax, label="RMS Δθ (rad)")
        return fig

    mo.vstack([mo.md(f"## Per-IC divergence vs {ref_label} over the (θ₀, ω₀) grid"), _plot()])
    return


if __name__ == "__main__":
    app.run()
