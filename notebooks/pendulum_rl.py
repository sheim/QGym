"""Inspect pendulum RL: training vitals + the cross-engine transfer matrix.

Two data sources:

* **Training vitals** — ``logs/pendulum_<backend>/<run>/vitals.jsonl``, one JSON
  object per iteration (written by learning.utils.logger.Logger).
* **Transfer eval** — ``logs/rl_eval/<train>__<eval>.npz`` from
  ``scripts/eval_policy.py``: a policy trained on backend A, evaluated on
  backend B over the deterministic reset_to_uniform grid.

    uv run marimo edit notebooks/pendulum_rl.py      # interactive
    uv run marimo run  notebooks/pendulum_rl.py      # read-only app

Robust to partial data: whatever isn't on disk yet is simply skipped.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, json, mo, np, plt


@app.cell
def _(Path, json):
    # Training vitals: newest run per pendulum_<backend> experiment dir.
    backends = ["cpu", "warp", "vsim"]
    vitals = {}
    for _b in backends:
        _exp = Path("logs") / f"pendulum_{_b}"
        if not _exp.exists():
            continue
        _runs = [d for d in _exp.iterdir() if (d / "vitals.jsonl").exists()]
        if not _runs:
            continue
        _latest = max(_runs, key=lambda d: d.stat().st_mtime)
        with open(_latest / "vitals.jsonl") as _f:
            rows = [json.loads(line) for line in _f if line.strip()]
        if rows:
            vitals[_b] = {k: [r.get(k) for r in rows] for k in rows[-1]}
    return backends, vitals


@app.cell
def _(Path, np):
    # Transfer-eval npzs: keyed by (train_label, eval_label).
    edir = Path("logs/rl_eval")
    evals = {}
    if edir.exists():
        for _f in sorted(edir.glob("*.npz")):
            d = np.load(_f, allow_pickle=True)
            evals[(str(d["train_label"]), str(d["eval_label"]))] = {
                k: d[k] for k in d.files
            }
    train_labels = sorted({k[0] for k in evals})
    eval_labels = sorted({k[1] for k in evals})
    return eval_labels, evals, train_labels


@app.cell
def _(evals, mo, vitals):
    mo.md(
        f"""
        # Pendulum RL — vitals & transfer matrix

        Training runs loaded: **{", ".join(vitals) or "none"}**  ·
        transfer-eval cells loaded: **{len(evals)}**.
        Train on backend A, evaluate the deterministic policy on backend B.
        """
    )
    return


@app.cell
def _(vitals):
    # Decreasing width + distinct dash so overlapping curves separate.
    _order = list(vitals)
    _dashes = ["-", "--", ":", "-."]

    def estyle(label):
        i = _order.index(label)
        return {
            "lw": max(0.9, 3.4 - 1.2 * i),
            "ls": _dashes[i % len(_dashes)],
            "zorder": i + 1,
        }

    return (estyle,)


@app.cell
def _(mo, vitals):
    _keys = sorted({k for v in vitals.values() for k in v if k != "iteration"})
    _default = "rewards/total_rewards"
    metric = mo.ui.dropdown(
        options=_keys,
        value=_default if _default in _keys else (_keys[0] if _keys else None),
    )
    engine_sel = mo.ui.multiselect(
        options=list(vitals), value=list(vitals), label="engines"
    )
    mo.hstack(
        [mo.md("**Metric:**"), metric, mo.md("**Engines:**"), engine_sel],
        justify="start",
        gap=1,
    )
    return engine_sel, metric


@app.cell
def _(engine_sel, estyle, metric, mo, plt, vitals):
    def _plot():
        if not vitals or metric.value is None:
            return mo.md("_No training vitals on disk yet._")
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for label, v in vitals.items():
            if label in engine_sel.value and metric.value in v:
                ax.plot(v["iteration"], v[metric.value], label=label, **estyle(label))
        ax.set(xlabel="iteration", ylabel=metric.value, title=metric.value)
        ax.legend(fontsize=8)
        return fig

    _plot()
    return


@app.cell
def _(engine_sel, estyle, mo, plt, vitals):
    def _plot():
        if not vitals:
            return mo.md("")
        panels = [
            "rewards/total_rewards",
            "algorithm/mean_value_loss",
            "algorithm/mean_surrogate_loss",
            "actor/action_std",
            "actor/entropy",
            "steps_per_s",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        for ax, key in zip(axes.flat, panels):
            for label, v in vitals.items():
                if label in engine_sel.value and key in v:
                    ax.plot(v["iteration"], v[key], label=label, **estyle(label))
            ax.set_title(key, fontsize=9)
            ax.tick_params(labelsize=7)
            if key == "steps_per_s":
                ax.set_yscale("log")
        axes.flat[0].legend(fontsize=8)
        return fig

    mo.vstack([mo.md("## Key training vitals"), _plot()])
    return


@app.cell
def _(eval_labels, evals, mo, np, plt, train_labels):
    def _matrix(metric):
        m = np.full((len(train_labels), len(eval_labels)), np.nan)
        for i, tr in enumerate(train_labels):
            for j, ev in enumerate(eval_labels):
                cell = evals.get((tr, ev))
                if cell is not None:
                    m[i, j] = float(np.mean(cell[metric]))
        return m

    def _heatmap(ax, m, title, fmt, cmap):
        im = ax.imshow(m, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(eval_labels)), eval_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(train_labels)), train_labels)
        ax.set(xlabel="eval backend", ylabel="train backend", title=title)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                            fontsize=9, color="w")
        return im

    def _plot():
        if not evals:
            return mo.md("_No transfer-eval npzs on disk yet — run scripts/eval_policy.py._")
        fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        rm = _matrix("mean_reward")
        sm = _matrix("success") * 100.0
        im_r = _heatmap(ax_r, rm, "mean reward", "{:+.2f}", "viridis")
        im_s = _heatmap(ax_s, sm, "upright success %", "{:.0f}", "magma")
        fig.colorbar(im_r, ax=ax_r)
        fig.colorbar(im_s, ax=ax_s)
        return fig

    mo.vstack(
        [
            mo.md("## Transfer matrix (train ↓ × eval →)"),
            _plot(),
            mo.md(
                "Diagonal = native (train and eval on the same engine); "
                "off-diagonal = sim-to-sim policy transfer."
            ),
        ]
    )
    return


@app.cell
def _(evals, mo):
    _keys = list(evals)
    cell = mo.ui.dropdown(
        options={f"{tr} → {ev}": (tr, ev) for tr, ev in _keys},
        value=(f"{_keys[0][0]} → {_keys[0][1]}" if _keys else None),
        label="inspect cell",
    )
    mo.hstack([mo.md("**Detail cell:**"), cell], justify="start", gap=1)
    return (cell,)


@app.cell
def _(cell, evals, mo, np, plt):
    def _plot():
        if not evals or cell.value is None:
            return mo.md("_No cell selected._")
        d = evals[cell.value]
        theta = d["theta"]
        n = theta.shape[1]
        g = int(round(n**0.5))
        tw = np.arctan2(np.sin(theta), np.cos(theta))
        t = np.arange(theta.shape[0]) / float(d["ctrl_hz"])

        fig, (ax_tr, ax_map) = plt.subplots(
            1, 2, figsize=(11, 4.2), constrained_layout=True
        )
        # sample of trajectories, coloured by final success
        ok = d["success"]
        for e in range(0, n, max(1, n // 60)):
            ax_tr.plot(t, tw[:, e], lw=0.6, color="green" if ok[e] else "red", alpha=0.4)
        ax_tr.axhline(0, color="k", lw=0.5, ls=":")
        ax_tr.set(xlabel="t (s)", ylabel="θ (rad)",
                  title=f"{cell.value[0]} → {cell.value[1]}  (green=upright)")

        succ = d["success"].reshape(g, g)
        im = ax_map.imshow(
            succ, origin="lower", aspect="auto",
            extent=[-5, 5, -np.pi, np.pi], cmap="RdYlGn", vmin=0, vmax=1,
        )
        ax_map.set(xlabel="ω₀ (rad/s)", ylabel="θ₀ (rad)",
                   title=f"success over ICs ({100 * ok.mean():.0f}%)")
        fig.colorbar(im, ax=ax_map)
        return fig

    mo.vstack([mo.md("## Per-cell detail"), _plot()])
    return


if __name__ == "__main__":
    app.run()
