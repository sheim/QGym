"""Visualize the deterministic mini-cheetah drop test across physics backends.

The notebook loads ``logs/mc_fid/drop_*.npz`` produced by
``scripts/mini_cheetah_fidelity.py`` and compares MuJoCo CPU, MuJoCo Warp, and
vsim against the CPU trajectory.

Generate fresh inputs:

    mkdir -p logs/mc_fid
    uv run scripts/mini_cheetah_fidelity.py run --probe drop \
      --backend mujoco --device cpu --out logs/mc_fid/drop_cpu.npz
    uv run scripts/mini_cheetah_fidelity.py run --probe drop \
      --backend mujoco --device cuda:0 --out logs/mc_fid/drop_warp.npz
    uv run --env-file .env.vsim scripts/mini_cheetah_fidelity.py run \
      --probe drop --backend vsim --device cuda:0 \
      --out logs/mc_fid/drop_vsim.npz

Run interactively:

    uv run marimo edit notebooks/mini_cheetah_drop_fidelity.py
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

    def quat_distance(a, b):
        """Sign-invariant angular distance between xyzw quaternion arrays."""
        an = a / np.linalg.norm(a, axis=-1, keepdims=True)
        bn = b / np.linalg.norm(b, axis=-1, keepdims=True)
        dot = np.abs(np.sum(an * bn, axis=-1))
        return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))

    return Path, mo, np, plt, quat_distance


@app.cell
def _(Path, mo, np):
    data_dir = Path("logs/mc_fid")
    paths = sorted(data_dir.glob("drop_*.npz")) if data_dir.exists() else []
    data = {}
    for path in paths:
        loaded = np.load(path, allow_pickle=False)
        if str(loaded["probe"]) == "drop":
            data[str(loaded["label"])] = {key: loaded[key] for key in loaded.files}

    ref_label = next((label for label in data if label == "mujoco-cpu"), None)
    if ref_label is None and data:
        ref_label = next(iter(data))

    mo.stop(
        not data,
        mo.md(
            f"**No drop data found in `{data_dir}/`.** Run the three commands "
            "in this notebook's docstring first."
        ),
    )
    mo.stop(
        len(data) < 2,
        mo.md(
            f"Only **{next(iter(data))}** is loaded. Generate at least one more "
            "backend before comparing trajectories."
        ),
    )

    ref = data[ref_label]
    n_steps, n_envs = ref["base_z"].shape
    ctrl_hz = float(ref["ctrl_hz"])
    t = (np.arange(n_steps) + 1) / ctrl_hz
    dof_names = [str(name) for name in ref["dof_names"]]
    return ctrl_hz, data, dof_names, n_envs, ref, ref_label, t


@app.cell
def _(data, mo, n_envs, ref_label, t):
    mo.md(f"""
    # Mini-cheetah drop-test fidelity

    Loaded **{", ".join(data)}** with **{n_envs} identical environments**
    over **{t[-1]:.2f} s**. All divergence is measured against
    **{ref_label}**.

    The robot starts at `z=0.5 m` with its default joint pose held by PD.
    Differences before impact indicate integration/state mismatches;
    differences at and after impact primarily expose contact-solver behavior.
    """)
    return


@app.cell
def _(data, dof_names, mo, n_envs):
    labels = list(data)
    engine_sel = mo.ui.multiselect(options=labels, value=labels, label="show engines")
    env_idx = mo.ui.slider(0, n_envs - 1, value=0, step=1, label="environment index")
    default_joints = [name for name in dof_names if name.endswith("_hfe")]
    if not default_joints:
        default_joints = dof_names[: min(4, len(dof_names))]
    joint_sel = mo.ui.multiselect(
        options=dof_names,
        value=default_joints,
        label="phase-portrait joints",
    )
    mo.hstack([engine_sel, env_idx, joint_sel], justify="start", gap=2)
    return engine_sel, env_idx, joint_sel, labels


@app.cell
def _(
    data,
    engine_sel,
    env_idx,
    labels,
    plt,
    quat_distance,
    ref,
    ref_label,
    t,
):
    def _plot_selected():
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
        ax_z, ax_vz, ax_q, ax_grf = axes.flat
        e = env_idx.value
        styles = ["-", "--", ":", "-."]
        for i, label in enumerate(labels):
            if label not in engine_sel.value:
                continue
            d = data[label]
            style = {"ls": styles[i % len(styles)], "lw": max(1.0, 2.8 - 0.5 * i)}
            ax_z.plot(t, d["base_z"][:, e], label=label, **style)
            if "base_lin_vel" in d:
                ax_vz.plot(t, d["base_lin_vel"][:, e, 2], label=label, **style)
            dq = quat_distance(d["base_quat"][:, e], ref["base_quat"][:, e])
            ax_q.plot(t, dq, label=label, **style)
            ax_grf.plot(t, d["grf"][:, e].sum(axis=-1), label=label, **style)

        ax_z.set(xlabel="time (s)", ylabel="base z (m)", title="Vertical trajectory")
        ax_vz.set(
            xlabel="time (s)",
            ylabel="vertical velocity (m/s)",
            title="Vertical velocity",
        )
        ax_q.set(
            xlabel="time (s)",
            ylabel="orientation error (rad)",
            title=f"Orientation divergence vs {ref_label}",
        )
        ax_grf.set(
            xlabel="time (s)", ylabel="summed foot GRF (N)", title="Contact response"
        )
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        ax_z.legend(fontsize=8)
        return fig

    _plot_selected()
    return


@app.cell
def _(data, engine_sel, env_idx, joint_sel, labels, mo, np, plt):
    def _plot_joint_phase_portraits():
        joints = joint_sel.value
        if not joints:
            return mo.md("_Select at least one joint._")

        ncols = min(2, len(joints))
        nrows = int(np.ceil(len(joints) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 4 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        styles = ["-", "--", ":", "-."]
        for ax, joint_name in zip(axes.flat, joints):
            for engine_index, label in enumerate(labels):
                if label not in engine_sel.value:
                    continue
                d = data[label]
                backend_dof_names = [str(name) for name in d["dof_names"]]
                joint_index = backend_dof_names.index(joint_name)
                position = d["dof_pos"][:, env_idx.value, joint_index]
                velocity = d["dof_vel"][:, env_idx.value, joint_index]
                ax.plot(
                    position,
                    velocity,
                    linestyle=styles[engine_index % len(styles)],
                    linewidth=1.5,
                    label=label,
                )
                ax.scatter(position[0], velocity[0], marker="o", s=24)
                ax.scatter(position[-1], velocity[-1], marker="x", s=30)
            ax.set(
                xlabel="joint position (rad)",
                ylabel="joint velocity (rad/s)",
                title=joint_name,
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        for ax in axes.flat[len(joints) :]:
            ax.set_visible(False)
        return fig

    mo.vstack(
        [
            mo.md(
                "## Joint phase portraits\n\n"
                "Circles mark the first recorded state; crosses mark the final state."
            ),
            _plot_joint_phase_portraits(),
        ]
    )
    return


@app.cell
def _(data, engine_sel, labels, np, plt, quat_distance, ref, ref_label, t):
    def _plot_aggregate():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        ax_z, ax_q, ax_f = axes
        styles = ["-", "--", ":", "-."]
        for i, label in enumerate(labels):
            if label == ref_label or label not in engine_sel.value:
                continue
            d = data[label]
            style = {"ls": styles[i % len(styles)], "lw": 2}
            dz = d["base_z"] - ref["base_z"]
            dq = quat_distance(d["base_quat"], ref["base_quat"])
            # Total force is invariant to backend-specific foot/sensor order.
            df = d["grf"].sum(axis=-1) - ref["grf"].sum(axis=-1)
            ax_z.plot(t, np.sqrt(np.mean(dz**2, axis=1)), label=label, **style)
            ax_q.plot(t, np.sqrt(np.mean(dq**2, axis=1)), label=label, **style)
            ax_f.plot(
                t,
                np.sqrt(np.mean(df**2, axis=1)),
                label=label,
                **style,
            )

        ax_z.set(xlabel="time (s)", ylabel="RMS Δz (m)", title="Base-height divergence")
        ax_q.set(
            xlabel="time (s)", ylabel="RMS angle (rad)", title="Orientation divergence"
        )
        ax_f.set(xlabel="time (s)", ylabel="RMS ΔGRF (N)", title="Contact divergence")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        return fig

    _plot_aggregate()
    return


@app.cell
def _(data, engine_sel, env_idx, labels, mo, plt, t):
    def _plot_feet():
        available = [label for label in labels if label in engine_sel.value]
        if not available:
            return mo.md("_Select at least one engine._")
        fig, axes = plt.subplots(
            len(available),
            1,
            figsize=(11, 2.5 * len(available)),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for ax, label in zip(axes[:, 0], available):
            d = data[label]
            names = (
                [str(name) for name in d["grf_names"]]
                if "grf_names" in d
                else [str(name) for name in d["foot_names"]]
                if "foot_names" in d
                else [f"foot {i}" for i in range(d["grf"].shape[-1])]
            )
            for i, name in enumerate(names):
                ax.plot(t, d["grf"][:, env_idx.value, i], label=name)
            ax.set(ylabel="GRF (N)", title=label)
            ax.grid(alpha=0.2)
            ax.legend(ncol=len(names), fontsize=8)
        axes[-1, 0].set_xlabel("time (s)")
        return fig

    mo.vstack([mo.md("## Per-foot impact timing"), _plot_feet()])
    return


@app.cell
def _(ctrl_hz, data, mo, np, ref, ref_label):
    rows = []
    settle_steps = max(1, int(round(0.5 * ctrl_hz)))
    for label, d in data.items():
        total_grf = d["grf"].sum(axis=-1)
        total_grf_rms = np.sqrt(np.mean((total_grf - ref["grf"].sum(axis=-1)) ** 2))
        hit = total_grf > 1.0
        first_idx = np.argmax(hit, axis=0)
        first_idx = np.where(hit.any(axis=0), first_idx, -1)
        impact = np.where(first_idx >= 0, (first_idx + 1) / ctrl_hz, np.nan)
        rows.append(
            {
                "engine": label,
                "impact time (s)": f"{np.nanmean(impact):.4f}",
                "peak total GRF (N)": f"{total_grf.max():.1f}",
                "settled z (m)": f"{d['base_z'][-settle_steps:].mean():.4f}",
                "z RMS vs CPU (m)": (
                    "—"
                    if label == ref_label
                    else f"{np.sqrt(np.mean((d['base_z'] - ref['base_z']) ** 2)):.3e}"
                ),
                "total GRF RMS vs CPU (N)": (
                    "—" if label == ref_label else f"{total_grf_rms:.3e}"
                ),
            }
        )

    mo.vstack(
        [
            mo.md("## Summary"),
            mo.ui.table(rows, selection=None),
            mo.md(
                "Impact uses the first sample where summed foot force exceeds "
                "1 N. Settled height is averaged over the final 0.5 s."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
