"""Inspect mini_cheetah_ref RL: training vitals, cross-engine transfer matrix,
and cross-backend DOF-trajectory divergence.

Data sources:

* **Training vitals** — ``logs/mini_cheetah_ref_<backend>/<run>/vitals.jsonl``.
* **Transfer eval** — ``logs/mc_rl_eval/<train>__<eval>.npz`` (reset_to_range,
  aggregate reward + survival: policy trained on A, scored on B).
* **Sample DOF trajectories** — ``logs/mc_rl_sample/<train>__<eval>.npz``
  (reset_to_basic → identical IC across backends, so the same policy's DOF
  trajectory on different engines is directly comparable).

    uv run marimo edit notebooks/mini_cheetah_ref.py      # interactive
    uv run marimo run  notebooks/mini_cheetah_ref.py      # read-only app

Robust to partial data.  mini_cheetah is floating-base + contact-rich, so
cross-engine DOF trajectories diverge (contact chaos) — the RMS heatmap
quantifies that; cpu~=warp (same MuJoCo solver), vsim differs (contact model).
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
    backends = ["cpu", "warp", "vsim"]
    vitals = {}
    for _b in backends:
        _exp = Path("logs") / f"mini_cheetah_ref_{_b}"
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
    return (vitals,)


@app.cell
def _(Path, np):
    def _load(dirname):
        d = Path("logs") / dirname
        out = {}
        if d.exists():
            for f in sorted(d.glob("*.npz")):
                z = np.load(f, allow_pickle=True)
                out[(str(z["train_label"]), str(z["eval_label"]))] = {
                    k: z[k] for k in z.files
                }
        return out

    evals = _load("mc_rl_eval")
    samples = _load("mc_rl_sample")
    train_labels = sorted({k[0] for k in evals})
    eval_labels = sorted({k[1] for k in evals})
    return eval_labels, evals, samples, train_labels


@app.cell
def _(evals, mo, samples, vitals):
    mo.md(f"""
    # mini_cheetah_ref — vitals · transfer · DOF divergence

    Training runs: **{", ".join(vitals) or "none"}**  ·  transfer cells:
    **{len(evals)}**  ·  DOF-sample cells: **{len(samples)}**.  Floating-base,
    contact-rich; a policy trained on backend A is scored on backend B.
    """)
    return


@app.cell
def _(vitals):
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
            "rewards/reference_traj",
            "rewards/tracking_lin_vel",
            "algorithm/mean_value_loss",
            "actor/action_std",
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
    def _matrix(metric, scale=1.0):
        m = np.full((len(train_labels), len(eval_labels)), np.nan)
        for i, tr in enumerate(train_labels):
            for j, ev in enumerate(eval_labels):
                cell = evals.get((tr, ev))
                if cell is not None:
                    m[i, j] = float(np.mean(cell[metric])) * scale
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
            return mo.md("_No transfer-eval npzs yet — run scripts/eval_policy.py._")
        fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        im_r = _heatmap(ax_r, _matrix("mean_reward"), "mean reward", "{:.2f}",
                        "viridis")
        im_s = _heatmap(ax_s, _matrix("survived", 100.0), "survival %", "{:.0f}",
                        "magma")
        fig.colorbar(im_r, ax=ax_r)
        fig.colorbar(im_s, ax=ax_s)
        return fig

    mo.vstack(
        [
            mo.md("## Transfer matrix (train ↓ × eval →)"),
            _plot(),
            mo.md("Diagonal = native; off-diagonal = sim-to-sim policy transfer."),
        ]
    )
    return


@app.cell
def _(mo, samples):
    def _rms(a, b):
        import numpy as np

        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _plot():
        import numpy as np
        import matplotlib.pyplot as plt

        if not samples:
            return mo.md(
                "_No DOF-sample npzs yet — run eval_policy with --record_dof "
                "--reset_mode reset_to_basic._"
            )
        keys = sorted(samples)  # (train, eval) pairs
        labels = [f"{t}→{e}" for t, e in keys]
        n = len(keys)
        # env 0 (reset_to_basic => identical IC across backends), dof trajectory
        trajs = [samples[k]["dof_traj"][:, 0, :] for k in keys]
        tmin = min(t.shape[0] for t in trajs)
        trajs = [t[:tmin] for t in trajs]
        rms = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                rms[i, j] = _rms(trajs[i], trajs[j])

        fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.0 * n + 1.5),
                               constrained_layout=True)
        im = ax.imshow(rms, cmap="inferno")
        ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), labels, fontsize=8)
        ax.set_title("DOF-trajectory RMS between policy×backend runs (rad)")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{rms[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="w" if rms[i, j] < rms.max() * 0.6 else "k")
        fig.colorbar(im, ax=ax)
        return fig

    mo.vstack(
        [
            mo.md("## Cross-backend DOF-trajectory divergence"),
            _plot(),
            mo.md(
                "Same fixed IC (reset_to_basic) per run.  For a fixed train "
                "policy, the spread across eval backends is physics divergence "
                "under that policy — expect cpu↔warp small, vsim larger "
                "(contact chaos amplifies the contact-model gap)."
            ),
        ]
    )
    return


@app.cell
def _(mo, samples):
    _tr = sorted({k[0] for k in samples})
    policy = mo.ui.dropdown(options=_tr, value=(_tr[0] if _tr else None),
                           label="train policy")
    mo.hstack([mo.md("**Sample DOF traj — policy:**"), policy], justify="start", gap=1)
    return (policy,)


@app.cell
def _(mo, plt, policy, samples):
    def _plot():
        import numpy as np

        if not samples or policy.value is None:
            return mo.md("_No sample selected._")
        cells = {e: v for (t, e), v in samples.items() if t == policy.value}
        if not cells:
            return mo.md("")
        names = None
        for v in cells.values():
            if "dof_names" in v:
                names = [str(x) for x in v["dof_names"]]
                break
        # show first joint of each leg (0,3,6,9) to keep it readable
        show = [0, 3, 6, 9]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
        for ax, j in zip(axes.flat, show):
            for e, v in sorted(cells.items()):
                tr = v["dof_traj"][:, 0, j]
                t = np.arange(len(tr)) / float(v["ctrl_hz"])
                ax.plot(t, tr, label=e, lw=1.2)
            jn = names[j] if names else f"dof {j}"
            ax.set(xlabel="t (s)", ylabel="angle (rad)", title=jn)
            ax.legend(fontsize=8)
        return fig

    mo.vstack(
        [mo.md(f"### {policy.value} policy: same joint on each eval backend"), _plot()]
    )
    return


if __name__ == "__main__":
    app.run()
