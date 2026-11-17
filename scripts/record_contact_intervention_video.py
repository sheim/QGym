"""Record a 10 s headless video of go2trot: plain policy versus max-Q intervention.

Both runs start from the *same* seeded reset state (identical initial qpos/qvel,
phase, frequency and command sequence), differ only in the action selector, and
render offscreen with MuJoCo's EGL renderer. The intervention is the persistent
max-Q rule used in the offline evaluation: eight stochastic draws per decision,
execute the one with the highest predicted contact-constraint return.

The initial condition is a task reset state, not a recorded insertion snapshot
(that snapshot is a Warp-only adapter and Warp has no renderer). Domain
randomisation is disabled so both runs share one physical model; the numbers are
therefore not directly comparable to the Warp experiments, only the qualitative
behaviour is.

Usage:

    MUJOCO_GL=egl uv run --frozen python scripts/record_contact_intervention_video.py \
      --artifact logs/go2trot/<run>/offline_q_60k/contact_q/data/100pct \
      --checkpoint logs/go2trot/<run>/model_616.pt \
      --output logs/go2trot/<run>/videos --scan_seeds 30 --seed -1
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

import mujoco

from gym.utils.task_registry import task_registry
from learning.offline.contracts import ContactForceRewardSpec
from learning.offline.evaluation import (
    _policy_observation,
    load_policy_from_artifact,
)
from learning.offline.q_artifact import ContactQ
from learning.offline.rewards import contact_force_reward

LABELS = {"plain": "plain policy", "maxq": "max-Q intervention"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="go2trot")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Reset seed; -1 scans --scan_seeds and uses the worst plain run.",
    )
    parser.add_argument("--scan_seeds", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_side_by_side", action="store_true")
    return parser


def build_env(args):
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(args.task, original_cfg=True)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(args.seconds + 2.0, 5.0)
    env_cfg.commands.resampling_time = 3.0
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    # One physical model for both runs: a per-run mass/friction draw would
    # confound the visual comparison with domain randomisation.
    env_cfg.domain_randomization.startup.link_mass_scale_range = None
    env_cfg.domain_randomization.startup.contact_friction_range = None
    env_cfg.seed = 0
    train_cfg.seed = 0
    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = -1
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    return task_registry.make_env(args.task, env_cfg, args.device, True, "mujoco")


def make_camera(model, width, height):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = body
    camera.distance = 1.15
    camera.azimuth = 135.0
    camera.elevation = -18.0
    # The task's model ships a 640x480 offscreen framebuffer; enlarge it before
    # the renderer is constructed.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    return camera, mujoco.Renderer(model, height=height, width=width)


def burn_label(frame, text):
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 26], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(image)


@torch.no_grad()
def drive(
    env, policy, policy_spec, artifact, reward_spec, label, args, seed, on_step=None
):
    """Run one controller from a reset seed; return counts and optional frames."""
    feet = env.feet_indices.to(env.device)
    steps = int(args.seconds / env.dt)
    stride = max(int(round(1.0 / (env.dt * args.fps))), 1)
    data = env._backend._datas[0]
    frames = []
    violations = 0
    terminated_steps = 0
    min_height = 1.0

    torch.manual_seed(seed)
    env.reset()
    action_rng = torch.Generator(device=env.device).manual_seed(seed + 7)
    for step in range(steps):
        obs = _policy_observation(
            env,
            policy_spec.obs_names,
            policy_spec.obs_dims,
            {},
            env.scales,
            action_rng,
        )
        if label == "plain":
            action = policy.sample_actions(obs, action_rng)
        else:
            mean = policy.mean_actions(obs)
            eps = torch.randn(
                1, 8, mean.shape[-1], device=env.device, generator=action_rng
            )
            candidates = mean[:, None] + policy.actor.std.detach() * eps
            scores = artifact.predict(
                obs[:, None].expand(-1, 8, -1).flatten(0, 1),
                candidates.flatten(0, 1),
            ).reshape(1, 8)
            action = candidates[0, int(scores.argmax(-1))].unsqueeze(0)
        env.set_states(["dof_pos_target"], action)
        env.step()
        load = env.contact_forces.index_select(1, feet)[..., 2] / env.bodyweight
        reward = contact_force_reward(load, reward_spec) * (~env.terminated)
        violations += int((reward < 0).sum())
        terminated_steps += int(env.terminated.sum())
        min_height = min(min_height, float(env.base_height.min()))
        if on_step is not None and step % stride == 0:
            on_step(data)
    return {
        "controller": label,
        "seed": seed,
        "steps": steps,
        "violation_steps": violations,
        "terminated_steps": terminated_steps,
        "min_base_height_m": min_height,
        "frames": len(frames),
    }


def encode(frames, path, width, height, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()


def main():
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    artifact = ContactQ(args.artifact)
    policy, policy_spec = load_policy_from_artifact(
        artifact, args.checkpoint, args.device
    )
    artifact.q.to(env.device)
    artifact.obs_mean = artifact.obs_mean.to(env.device)
    artifact.obs_std = artifact.obs_std.to(env.device)
    reward_spec = ContactForceRewardSpec(**artifact.manifest["reward_spec"])

    model = env._backend._model_for_env(0)
    camera, renderer = make_camera(model, args.width, args.height)
    results = {"scan": [], "videos": []}
    try:
        seed = args.seed
        if seed < 0:
            # A benign reset state shows nothing; pick the seed whose plain run
            # violates the constraint most, then record that one.
            for candidate in range(args.scan_seeds):
                result = drive(
                    env,
                    policy,
                    policy_spec,
                    artifact,
                    reward_spec,
                    "plain",
                    args,
                    candidate,
                )
                results["scan"].append(
                    {
                        "seed": candidate,
                        "violation_steps": result["violation_steps"],
                        "terminated_steps": result["terminated_steps"],
                        "min_base_height_m": result["min_base_height_m"],
                    }
                )
            # Prefer an upright 10 s clip, then the most violations.
            results["scan"].sort(
                key=lambda entry: (
                    entry["terminated_steps"] > 0,
                    -entry["violation_steps"],
                )
            )
            seed = results["scan"][0]["seed"]
            print(
                "best plain seeds (upright first):",
                [
                    (e["seed"], e["violation_steps"], e["terminated_steps"])
                    for e in results["scan"][:5]
                ],
                flush=True,
            )
        for label in ("plain", "maxq"):
            frames = []

            def on_step(data, _frames=frames, _label=label):
                renderer.update_scene(data, camera=camera)
                _frames.append(burn_label(renderer.render(), LABELS[_label]))

            result = drive(
                env,
                policy,
                policy_spec,
                artifact,
                reward_spec,
                label,
                args,
                seed,
                on_step=on_step,
            )
            path = output / f"{label}.mp4"
            encode(frames, path, args.width, args.height, args.fps)
            result["frames"] = len(frames)
            result["video"] = str(path)
            results["videos"].append(result)
    finally:
        renderer.close()
        if hasattr(env, "_backend"):
            env._backend.close()
    if not args.no_side_by_side and shutil.which("ffmpeg"):
        side = output / "side_by_side.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(output / "plain.mp4"),
                "-i",
                str(output / "maxq.mp4"),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "23",
                str(side),
            ],
            check=True,
        )
        results["side_by_side"] = str(side)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "scan"}, indent=2))


if __name__ == "__main__":
    main()
