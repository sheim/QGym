"""Plot yaw angular-velocity tracking against the reference for traced rollouts.

Reads ``trace_<controller>.pt`` files written by
``scripts/evaluate_angular_velocity_q_persistent.py --trace_steps``. Each file
holds per-step ``ang_vel`` and post-step ``command`` tensors in world order
(lane-major, then 128 insertion states) plus a liveness mask. The left panel
shows the mean yaw rate against the piecewise-constant reference; the right
panel shows the per-world tracking error. Vertical markers are the detected
command-resampling boundaries (3.0 s = 150 control steps at dt=0.02).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

CHANGE_THRESHOLD = 0.01


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument(
        "--controllers",
        nargs="+",
        default=None,
        help="Controller names to plot (1-4); defaults to baseline plus the "
        "first max-Q artifact.",
    )
    p.add_argument("--resample_s", type=float, default=3.0)
    p.add_argument("--output", default=None)
    p.add_argument("--title", default=None)
    return p


def default_controllers(directory: Path) -> list[str]:
    provenance = json.loads((directory / "provenance.json").read_text())
    names = ["baseline", provenance["artifact_names"][0]]
    return [name for name in names if (directory / f"trace_{name}.pt").is_file()]


@torch.no_grad()
def load_trace(directory: Path, name: str, lanes: int = 8) -> dict:
    path = directory / f"trace_{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing trace file: {path}")
    raw = torch.load(path, weights_only=True)
    alive = raw["alive"].bool()
    if alive.shape[0] % lanes:
        raise ValueError(f"{path} does not hold a whole number of lanes")
    steps = alive.shape[1]
    mask = alive.reshape(lanes, -1, steps)
    count = mask.sum(1)
    result = {"steps": steps, "control_dt": float(raw["control_dt"])}
    series = {}
    for key in ("ang_vel", "command"):
        values = raw[key].double().reshape(lanes, -1, steps)
        series[key] = values
        lane_state_mean = (values * mask).sum(1) / count.clamp(min=1)
        result[key] = {
            "mean": lane_state_mean.mean(0),
            "std": lane_state_mean.std(0, unbiased=False),
            "count": count.sum(0),
        }
    error = series["ang_vel"] - series["command"]
    error_mean = (error * mask).sum(1) / count.clamp(min=1)
    result["error"] = {
        "mean": error_mean.mean(0),
        "std": error_mean.std(0, unbiased=False),
        "world_step_rms": float((error.square() * mask).sum() / mask.sum().clamp(min=1))
        ** 0.5,
        "world_step_mean_absolute": float(
            (error.abs() * mask).sum() / mask.sum().clamp(min=1)
        ),
    }
    result["seconds"] = float(raw.get("seconds", 0.0))
    return result


def metrics(trace: dict) -> dict:
    mask = trace["ang_vel"]["count"] > 0
    return {
        "mean_absolute_of_lane_mean_error_rad_s": float(
            trace["error"]["mean"][mask].abs().mean()
        ),
        "world_step_yaw_rmse_rad_s": trace["error"]["world_step_rms"],
        "world_step_mean_absolute_yaw_error_rad_s": trace["error"][
            "world_step_mean_absolute"
        ],
        "mean_between_lane_sd_rad_s": float(trace["ang_vel"]["std"][mask].mean()),
        "alive_world_steps_fraction": float(mask.double().sum() / mask.numel()),
    }


def resample_boundaries(command_mean: torch.Tensor, dt: float) -> list[float]:
    changes = (command_mean.diff().abs() > CHANGE_THRESHOLD).nonzero().flatten()
    return [float((index + 1) * dt) for index in changes]


def plot(directory: Path, controllers: list[str], args) -> Path:
    traces = {name: load_trace(directory, name) for name in controllers}
    report = {
        name: {"seconds": trace["seconds"], **metrics(trace)}
        for name, trace in traces.items()
    }
    first = traces[controllers[0]]
    steps = first["steps"]
    dt = first["control_dt"]
    time = torch.arange(steps, dtype=torch.float64) * dt
    boundaries = resample_boundaries(first["command"]["mean"], dt)
    report["resample_boundaries_s"] = boundaries
    report["resample_interval_steps"] = args.resample_s / dt
    colors = {"baseline": "#1f77b4"}
    for index, name in enumerate(controllers[1:]):
        colors[name] = ["#d62728", "#2ca02c", "#9467bd"][index % 3]
    figure, axes = plt.subplots(
        len(controllers),
        2,
        figsize=(13, 2.8 * len(controllers)),
        squeeze=False,
        sharex=True,
    )
    for row, name in enumerate(controllers):
        trace = traces[name]
        label = "plain (candidate 0)" if name == "baseline" else "max-Q"
        rate_axis, error_axis = axes[row]
        for axis in (rate_axis, error_axis):
            for boundary in boundaries:
                axis.axvline(boundary, color="0.7", linewidth=0.8, ls=":")
            axis.grid(alpha=0.3)
        rate_axis.plot(
            time,
            trace["command"]["mean"],
            color="black",
            linewidth=1.2,
            ls="--",
            label="reference (post-resample command)",
        )
        rate_axis.fill_between(
            time,
            trace["ang_vel"]["mean"] - trace["ang_vel"]["std"],
            trace["ang_vel"]["mean"] + trace["ang_vel"]["std"],
            color=colors[name],
            alpha=0.25,
            label=f"{label} mean±SD across lanes",
        )
        rate_axis.plot(
            time, trace["ang_vel"]["mean"], color=colors[name], linewidth=1.3
        )
        rate_axis.set_title(f"{name}: yaw rate vs reference")
        rate_axis.set_ylabel("yaw rate [rad/s]")
        error_axis.axhline(0.0, color="black", linewidth=0.8)
        error_axis.fill_between(
            time,
            trace["error"]["mean"] - trace["error"]["std"],
            trace["error"]["mean"] + trace["error"]["std"],
            color=colors[name],
            alpha=0.25,
            label=f"{label} error ±SD across lanes",
        )
        error_axis.plot(time, trace["error"]["mean"], color=colors[name], linewidth=1.3)
        mean_error = metrics(trace)["world_step_yaw_rmse_rad_s"]
        error_axis.set_title(
            f"{name}: systematic error; world-step yaw RMSE={mean_error:.4f}"
        )
        error_axis.set_ylabel("mean(ang_vel - reference) [rad/s]")
        if row == len(controllers) - 1:
            rate_axis.set_xlabel("time [s]  (dotted = detected command resample)")
            error_axis.set_xlabel("time [s]  (dotted = detected command resample)")
    if args.title:
        figure.suptitle(args.title)
    figure.tight_layout()
    output = (
        Path(args.output)
        if args.output
        else directory / "angular_velocity_tracking.png"
    )
    figure.savefig(output, dpi=150)
    plt.close(figure)
    (directory / "tracking_metrics.json").write_text(json.dumps(report, indent=2))
    return output


def main():
    args = build_parser().parse_args()
    directory = Path(args.directory)
    controllers = args.controllers or default_controllers(directory)
    if not 1 <= len(controllers) <= 4:
        raise ValueError("plot 1-4 controllers")
    output = plot(directory, controllers, args)
    print(json.dumps({"plot": str(output), "controllers": controllers}))


if __name__ == "__main__":
    main()
