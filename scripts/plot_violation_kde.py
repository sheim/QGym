"""KDE of violations per rollout for the three initial conditions (fig14).

Reads the per-rollout violation counts written by
``scripts/plot_contact_intervention_traces.py`` (``stats.json`` in each trace
run directory) and plots, for each initial condition, the density of the
violations-per-rollout distribution under the plain policy and under max-Q.

Gaussian KDE with a Silverman bandwidth, reflected at zero so the density does
not leak below the support of a count.

Usage:

    uv run --frozen python scripts/plot_violation_kde.py \
      --source logs/go2trot/<run> --output <path>/fig14_three_initial_conditions_kde.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent.parent / (
    "docs/plans/supplements/foot_contact_study"
)
sys.path.insert(0, str(STUDY_DIR))

from figure_io import save_figure  # noqa: E402

# (label, run directory name) in display order: worst, best, benign.
CONDITIONS = (
    ("state 22 (worst plain)", "traces_state22"),
    ("state 86 (mid)", "traces_state86"),
    ("state 61 (smallest difference)", "traces_smallest_delta"),
)
COLORS = {"plain": "tab:orange", "maxq": "tab:blue"}
LABELS = {"plain": "plain", "maxq": "max-Q"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Run directory with traces_*")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def silverman_bandwidth(values: np.ndarray) -> float:
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    iqr = (
        float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) > 1 else 0.0
    )
    scale = min(std, iqr / 1.349) if std > 0 and iqr > 0 else max(std, iqr)
    n = len(values)
    bandwidth = 0.9 * scale * n ** (-0.2) if scale > 0 else 0.5
    return max(bandwidth, 0.15)  # counts are integers; keep the curve from spiking


def reflected_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE reflected at zero: counts cannot be negative."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros_like(grid)
    bandwidth = silverman_bandwidth(values)
    total = np.zeros_like(grid)
    for value in values:
        for sign in (1.0, -1.0):
            total += np.exp(-0.5 * ((grid - sign * value) / bandwidth) ** 2)
    return total / (len(values) * bandwidth * np.sqrt(2.0 * np.pi))


def main():
    args = build_parser().parse_args()
    source = Path(args.source)
    data = {}
    for label, directory in CONDITIONS:
        path = source / directory / "stats.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        stats = json.loads(path.read_text())
        values = {
            name: np.asarray(stats["violations"][name]["values"], dtype=float)
            for name in ("plain", "maxq")
        }
        data[label] = {
            "state": int(stats["state"]),
            "values": values,
            "rollouts": int(stats["rollouts"]["plain"]),
            "training_reward": stats["training_reward_per_step"],
        }

    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.18})
    # Per-panel scales: these distributions live on very different supports
    # (max 43 versus max 6 violations), so one shared axis hides two of them.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (label, entry) in zip(axes, data.items(), strict=True):
        top = max(float(entry["values"][name].max()) for name in ("plain", "maxq"))
        grid = np.linspace(0.0, top + 1.5, 512)
        peak = 0.0
        for name in ("plain", "maxq"):
            values = entry["values"][name]
            density = reflected_kde(values, grid)
            peak = max(peak, float(density.max()))
            ax.fill_between(grid, 0.0, density, color=COLORS[name], alpha=0.25, lw=0)
            ax.plot(grid, density, color=COLORS[name], lw=1.8, label=LABELS[name])
            ax.axvline(values.mean(), color=COLORS[name], ls="--", lw=1.0, alpha=0.9)
            # Rug: the counts are discrete, the KDE is only a smoothing.
            ax.plot(
                values,
                np.full_like(values, -0.04 * peak),
                "|",
                color=COLORS[name],
                ms=5,
                mew=1.0,
                clip_on=False,
            )
        ax.set_ylim(-0.12 * peak, 1.15 * peak)
        ax.set_xlim(0, grid[-1])
        mean_plain = entry["values"]["plain"].mean()
        mean_maxq = entry["values"]["maxq"].mean()
        zero_plain = float((entry["values"]["plain"] == 0).mean())
        zero_maxq = float((entry["values"]["maxq"] == 0).mean())
        ax.set_title(
            f"{label}\nplain {mean_plain:.2f} vs max-Q {mean_maxq:.2f} "
            f"violations/rollout ({entry['rollouts']} each)\n"
            f"zero-violation rollouts {zero_plain:.0%} vs {zero_maxq:.0%}",
            fontsize=8,
        )
        ax.set_xlabel("violations per rollout (500 steps)")
    axes[0].set_ylabel("density (Gaussian KDE, reflected at 0)")
    handles = [
        Line2D([], [], color=COLORS["plain"], lw=2, label="plain"),
        Line2D([], [], color=COLORS["maxq"], lw=2, label="max-Q"),
        Line2D([], [], color="0.35", ls="--", lw=1, label="mean"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, dpi=args.dpi, layout=False)
    hd = output.with_name(output.stem + "_hd.png")
    # The tracked copy is palette-reduced; this one is the full-quality raster.
    fig.savefig(hd, dpi=args.dpi * 1.5, pil_kwargs={"optimize": True})
    fig.savefig(hd.with_suffix(".svg"))
    plt.close(fig)
    print(f"hd copies: {hd.name}, {hd.with_suffix('.svg').name}")
    for label, entry in data.items():
        lines = {
            name: {
                "mean": float(entry["values"][name].mean()),
                "sd": float(entry["values"][name].std(ddof=1)),
                "zero_fraction": float((entry["values"][name] == 0).mean()),
                "bandwidth": silverman_bandwidth(entry["values"][name]),
            }
            for name in ("plain", "maxq")
        }
        print(label, json.dumps(lines))


if __name__ == "__main__":
    main()
