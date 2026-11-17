"""One initial condition, many rollouts: plain policy versus max-Q, mean +/- sd.

Restores the eight replicated worlds of a single insertion state, runs the
frozen policy (plain) and the max-Q controller over several blocks of shared
command/push noise, and plots the time-domain mean +/- std of the per-step
signals across all rollouts.

Usage:

    uv run --frozen python scripts/plot_contact_intervention_traces.py \
      --source logs/go2trot/<run>/contact_ranking \
      --artifact logs/go2trot/<run>/offline_q_60k/contact_q/data/100pct \
      --checkpoint logs/go2trot/<run>/model_616.pt --load_run <run> \
      --state 22 --blocks 8 --horizon 500 --output logs/go2trot/<run>/traces_state22
"""

import argparse
import copy
import json
from pathlib import Path

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    ContactForceRewardSpec,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.evaluation import load_policy_from_artifact
from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    install_paired_environment_noise,
)
from learning.offline.persistent_intervention import rollout_contact
from learning.offline.q_artifact import ContactQ
from learning.offline.training_reward import TrainingRewardModel

STUDY_DIR = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "plans"
    / "supplements"
    / "foot_contact_study"
)

LANES = 8
STATES = 128  # the Warp insertion adapter requires its fixed 1024-world inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Insertion export directory")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument(
        "--state",
        type=int,
        default=-1,
        help="-1 screens all 128 states and picks the one with the smallest "
        "plain-versus-max-Q violation difference",
    )
    parser.add_argument(
        "--screen_blocks",
        type=int,
        default=4,
        help="Blocks used by the screening pass when --state -1",
    )
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7800000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI for the PNG")
    parser.add_argument(
        "--docs_name",
        default=None,
        help="Also write a <=100 KiB copy under the study figures directory with "
        "this file name (the committed copy).",
    )
    return parser


def masked_mean_std(
    trace: torch.Tensor, alive: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step mean and std over *alive* rollouts only.

    Dead rollouts are zero-filled, so an unmasked average mixes real values with
    zeros and reports a spurious spread (and a mean drifting to zero).
    """
    mask = alive if trace.ndim == alive.ndim else alive.unsqueeze(1)
    mask = mask.expand_as(trace).float()
    count = mask.sum(0).clamp_min(1.0)
    mean = (trace * mask).sum(0) / count
    variance = (((trace - mean) * mask).square()).sum(0) / count
    return mean, variance.sqrt()


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    import gym.envs  # noqa: F401
    import sys

    sys.path.insert(0, str(STUDY_DIR))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(Path(args.source) / "insertion.pt", weights_only=True)
    artifact = ContactQ(args.artifact)
    reward_spec = ContactForceRewardSpec(**artifact.manifest["reward_spec"])
    policy, spec = load_policy_from_artifact(artifact, args.checkpoint, args.device)
    if sha256_file(args.checkpoint) != spec.checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")

    cfg, train = copy.deepcopy(
        task_registry.get_cfgs(
            "go2trot",
            original_cfg=False,
            experiment_name="go2trot",
            load_run=args.load_run,
        )
    )
    cfg.env.num_envs = STATES * LANES
    train.runner.device = args.device
    task_registry.convert_frequencies_to_params(cfg, train)
    if (
        replay_config_fingerprint(class_to_dict(cfg))
        != artifact.manifest["dataset"]["config_fingerprint"]
    ):
        raise ValueError("environment fingerprint mismatch")
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s, (args.horizon + 150) * cfg.control.ctrl_dt + 1
    )
    cfg.seed = saved["seed"]
    env = task_registry.make_env("go2trot", cfg, args.device, True, "mujoco")
    try:
        artifact.q.to(env.device)
        artifact.obs_mean = artifact.obs_mean.to(env.device)
        artifact.obs_std = artifact.obs_std.to(env.device)
        # The adapter only supports its 1024-world inventory, so restore every
        # state and keep only the target state's eight lanes per rollout.
        snapshot = WarpInsertionSnapshot(env)
        snapshot.load_export(saved["snapshot"])
        snapshot.restore()
        snapshot.assert_restored()
        install_paired_environment_noise(env, LANES)
        observations = saved["insertion_observations"].to(env.device)
        names = list(env.robot_layout.body_names)
        feet = torch.tensor(
            [names.index(name) for name in reward_spec.foot_names],
            dtype=torch.long,
            device=env.device,
        )
        if args.state < 0:
            # Screening pass: same controllers, all 128 states, its own seeds so
            # the plotted rollouts stay independent of the selection.
            deltas = torch.zeros(STATES, dtype=torch.float64)
            for block in range(args.screen_blocks):
                seed = args.seed + 100 + 10 * block
                plain_run = rollout_contact(
                    env,
                    policy,
                    spec,
                    snapshot,
                    observations,
                    None,
                    seed,
                    reward_spec,
                    feet,
                    args.horizon,
                )
                maxq_run = rollout_contact(
                    env,
                    policy,
                    spec,
                    snapshot,
                    observations,
                    artifact,
                    seed,
                    reward_spec,
                    feet,
                    args.horizon,
                )
                deltas += (
                    plain_run["violation_steps"].reshape(STATES, LANES).mean(1)
                    - maxq_run["violation_steps"].reshape(STATES, LANES).mean(1)
                ).double()
            deltas /= args.screen_blocks
            args.state = int(deltas.argmin())
            ranked = torch.argsort(deltas)
            print(
                f"screened {STATES} states on {args.screen_blocks} blocks: "
                f"smallest delta state {args.state} ({float(deltas[args.state]):.3f}), "
                f"median {float(deltas.median()):.2f}, largest "
                f"{float(deltas.max()):.2f}"
            )
            print(
                "five smallest:",
                [
                    (int(index), round(float(deltas[index]), 2))
                    for index in ranked[:5].tolist()
                ],
            )
        # The adapter only supports its 1024-world inventory, so keep only the
        # chosen state's eight lanes per rollout from here on.
        rows = slice(args.state * LANES, (args.state + 1) * LANES)
        training_model = TrainingRewardModel(
            env,
            class_to_dict(train.critic.reward.weights),
            class_to_dict(train.critic.reward.termination_weight),
            gamma=float(train.algorithm.gamma),
        )
        traces = {"plain": [], "maxq": []}
        counts = {"plain": [], "maxq": []}
        training = {"plain": [], "maxq": []}
        for block in range(args.blocks):
            seed = args.seed + 10 * block
            for name, critic in (("plain", None), ("maxq", artifact)):
                result = rollout_contact(
                    env,
                    policy,
                    spec,
                    snapshot,
                    observations,
                    critic,
                    seed,
                    reward_spec,
                    feet,
                    args.horizon,
                    training_model=training_model,
                    trace=True,
                )
                traces[name].append(
                    {
                        key.split("trace_")[1]: value[rows]
                        for key, value in result.items()
                        if key.startswith("trace_")
                    }
                )
                counts[name].append(result["violation_steps"][rows])
                training[name].append(
                    (result["training_reward"] / result["steps"].clamp_min(1))[rows]
                )
            print(f"block {block} done", flush=True)

        stacked = {}
        for name in traces:
            stacked[name] = {}
            for key in traces[name][0]:
                value = torch.stack([entry[key] for entry in traces[name]])
                if key == "foot_load":
                    stacked[name][key] = value.reshape(
                        -1, value.shape[-2], value.shape[-1]
                    )
                else:
                    stacked[name][key] = value.reshape(-1, args.horizon)
        per_rollout = {name: torch.cat(counts[name]).reshape(-1) for name in counts}
        stats = {
            "state": args.state,
            "blocks": args.blocks,
            "rollouts": {name: int(per_rollout[name].numel()) for name in counts},
            "horizon": args.horizon,
            "reward_spec": artifact.manifest["reward_spec"],
            "violations": {
                name: {
                    "mean": float(per_rollout[name].mean()),
                    "std": float(per_rollout[name].std(unbiased=False)),
                    "max": float(per_rollout[name].max()),
                    "zero_fraction": float((per_rollout[name] == 0).double().mean()),
                    "values": per_rollout[name].tolist(),
                }
                for name in counts
            },
            "training_reward_per_step": {
                name: float(torch.cat(training[name]).mean()) for name in training
            },
            "per_step": {
                name: {
                    key: {
                        "mean": stacked[name][key].mean(0).tolist(),
                        "std": stacked[name][key].std(0, unbiased=False).tolist(),
                    }
                    for key in ("violation", "max_load", "support", "base_height")
                }
                for name in stacked
            },
        }

        # ── figure: mean +/- sd over rollouts, plain vs max-Q ──────────────
        plt.rcParams.update(
            {
                "figure.dpi": 100,
                "savefig.dpi": args.dpi,
                "axes.grid": True,
                "grid.alpha": 0.25,
                # Keep SVG text as text: smaller file, and it scales crisply.
                "svg.fonttype": "none",
            }
        )
        time = (torch.arange(args.horizon).float() * 0.02).numpy()
        colors = {"plain": "tab:orange", "maxq": "tab:blue"}

        def smooth(values: torch.Tensor, window: int = 9) -> torch.Tensor:
            """Centred moving average, so gait oscillation does not hide the level."""
            pad = window // 2
            padded = torch.cat(
                [values[:1].expand(pad), values, values[-1:].expand(pad)]
            )
            kernel = torch.ones(window) / window
            return torch.conv1d(padded.view(1, 1, -1), kernel.view(1, 1, -1)).view(-1)

        def time_panel(ax, key, label, reference=None, foot=None, lower_bound=None):
            for name in ("plain", "maxq"):
                series = (
                    stacked[name]["foot_load"][:, foot, :]
                    if foot is not None
                    else stacked[name][key]
                )
                mean, std = masked_mean_std(series, stacked[name]["alive"])
                lower = mean - std
                if lower_bound is not None:
                    lower = lower.clamp_min(lower_bound)
                # Every second sample: halving the plotted point count keeps the
                # fills from looking like noise, which is what forces a low
                # palette depth when the PNG has to fit 100 KiB.
                step = 2
                ax.fill_between(
                    time[::step],
                    lower[::step],
                    (mean + std)[::step],
                    color=colors[name],
                    alpha=0.28,
                    linewidth=0,
                )
                (line,) = ax.plot(
                    time[::step],
                    smooth(mean)[::step].numpy(),
                    color=colors[name],
                    lw=2.0,
                    label=name,
                    zorder=5,
                )
                # White halo: keeps both means readable over the fills, which is
                # the main legibility problem in this dense figure.
                line.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=3.6, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
            if reference is not None:
                ax.axhline(reference, color="k", ls=":", lw=1)
            ax.set_xlim(0, time[-1])
            ax.set_ylabel(label)
            ax.grid(alpha=0.12, linewidth=0.6)

        fig, axes = plt.subplots(3, 2, figsize=(12.5, 9))
        thresholds = torch.tensor(
            artifact.manifest["reward_spec"]["thresholds_bw"]
        ).view(1, -1, 1)
        foot_violations = {
            name: (stacked[name]["foot_load"] > thresholds).sum(dim=(0, 2)).tolist()
            for name in stacked
        }
        stats["per_foot_violation_steps"] = foot_violations
        plain_foot = int(torch.tensor(foot_violations["plain"]).argmax())
        time_panel(
            axes[0, 0],
            "max_load",
            "max foot load / bw",
            reference=float(max(artifact.manifest["reward_spec"]["thresholds_bw"])),
        )
        time_panel(axes[0, 1], "support", "total support / bw", reference=1.0)
        time_panel(
            axes[1, 0],
            "violation",
            "violation probability (±1 sd, ≥0)",
            lower_bound=0.0,
        )
        time_panel(
            axes[1, 1],
            None,
            f"{reward_spec.foot_names[plain_foot]} load / bw",
            reference=reward_spec.thresholds_bw[plain_foot],
            foot=plain_foot,
        )
        time_panel(axes[2, 0], "base_height", "base height (m)")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]):
            ax.set_xlabel("time (s)")
        ax = axes[2, 1]
        top = int(max(per_rollout["plain"].max().item(), 1))
        bins = torch.arange(0, top + 2) - 0.5
        ax.hist(
            per_rollout["plain"].numpy(),
            bins=bins.numpy(),
            alpha=0.65,
            color=colors["plain"],
            label="plain",
        )
        ax.hist(
            per_rollout["maxq"].numpy(),
            bins=bins.numpy(),
            alpha=0.65,
            color=colors["maxq"],
            label="max-Q",
        )
        ax.set_xlabel("violations per rollout (500 steps)")
        ax.set_ylabel("rollouts")
        ax.legend(fontsize=7, loc="upper right")
        block_means = {
            name: [
                float(torch.cat(counts[name])[i * LANES : (i + 1) * LANES].mean())
                for i in range(args.blocks)
            ]
            for name in counts
        }
        handles = [
            Line2D([], [], color=colors["plain"], lw=2.0, label="plain mean"),
            Line2D([], [], color=colors["maxq"], lw=2.0, label="max-Q mean"),
            Patch(facecolor=colors["plain"], alpha=0.35, label="plain ±1 sd"),
            Patch(facecolor=colors["maxq"], alpha=0.35, label="max-Q ±1 sd"),
            Line2D([], [], color="k", ls=":", lw=1.2, label="reference / threshold"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=5,
            fontsize=8.5,
            frameon=False,
            borderaxespad=0.0,
        )
        fig.suptitle(
            f"Initial condition {args.state}: {stats['rollouts']['plain']} rollouts "
            f"each; plain {stats['violations']['plain']['mean']:.1f} +/- "
            f"{stats['violations']['plain']['std']:.1f} vs max-Q "
            f"{stats['violations']['maxq']['mean']:.1f} +/- "
            f"{stats['violations']['maxq']['std']:.1f} violations/rollout"
        )
        # Reserve a band for the figure legend so it cannot overlap the bottom
        # row's axis label (labels sit outside the axes rect).
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(
            output / f"state_{args.state:03d}_traces.png",
            pil_kwargs={"optimize": True},
        )
        fig.savefig(output / f"state_{args.state:03d}_traces.svg")
        if args.docs_name:
            from figure_io import save_figure

            study_figures = STUDY_DIR / "figures"
            study_figures.mkdir(parents=True, exist_ok=True)
            tracked = study_figures / args.docs_name
            save_figure(fig, tracked, dpi=args.dpi, layout=False)
            # Full-quality copies stay untracked (figures/.gitignore covers
            # *_hd.*): vector for zooming, 300 dpi PNG for raster use.
            hd_png = tracked.with_name(tracked.stem + "_hd.png")
            fig.savefig(hd_png, pil_kwargs={"optimize": True})
            fig.savefig(hd_png.with_suffix(".svg"))
            print(f"hd copies: {hd_png.name}, {hd_png.with_suffix('.svg').name}")
        plt.close(fig)
        torch.save(
            {
                name: {key: value.cpu() for key, value in stacked[name].items()}
                for name in stacked
            },
            output / "rollouts.pt",
        )
        stats["alive"] = {
            name: {
                "rollouts_alive_at_end": int(stacked[name]["alive"][:, -1].sum()),
                "mean_alive_steps": float(stacked[name]["alive"].sum(1).float().mean()),
            }
            for name in stacked
        }
        (output / "block_means.json").write_text(
            json.dumps(block_means, indent=2) + "\n"
        )
        (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats["violations"], indent=1))
        print(f"figure={output / f'state_{args.state:03d}_traces.png'}")
    finally:
        if hasattr(env, "_backend"):
            env._backend.close()


if __name__ == "__main__":
    main()
