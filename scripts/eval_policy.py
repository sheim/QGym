"""Evaluate a deterministic policy on a selected physics backend.

Pendulum evaluations support the deterministic ``reset_to_uniform`` grid used
by the Phase 4 transfer matrix. Legged evaluations additionally provide fixed
hardware command profiles, physical gait and actuator metrics, and optional
velocity-impulse tests. Every run writes an NPZ artifact; hardware scorecards
also write a human-readable JSON summary.

    uv run scripts/eval_policy.py --ckpt logs/pendulum_cpu --train_label cpu \
        --eval_backend mujoco --eval_device cpu --out logs/rl_eval/cpu__cpu.npz

    # cross-engine: a cpu-trained policy evaluated under vsim
    uv run --env-file .env.vsim scripts/eval_policy.py --ckpt logs/pendulum_cpu \
        --train_label cpu --eval_backend vsim --out logs/rl_eval/cpu__vsim.npz
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from gym.envs.base.domain_randomization import (
    DOMAIN_RANDOMIZATION_MODES,
    apply_domain_randomization_override,
    get_domain_randomization_range,
    set_domain_randomization_range,
)
from gym.utils.helpers import class_to_dict, set_seed
from gym.utils.legged_eval_metrics import (
    LeggedMetricAccumulator,
    actuated_position_reference,
    apply_command_profile,
    metric_metadata,
    summarize_metrics,
    velocity_impulse_schedule,
)
from gym.utils.original_cfg import load_original_cfgs_from_run, original_cfg_source_dir
from gym.utils.policy_io import state_component_names, state_component_scales
from gym.utils.task_registry import task_registry
from gym.utils.torch_quat import quat_rotate_inverse


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_ckpt(path):
    """Accept a run dir (pick the highest-iteration model) or a .pt file."""
    if os.path.isfile(path):
        return path
    models = [
        f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")
    ]
    if not models:
        # maybe a logs/<exp> dir with timestamped run subdirs — take the latest run
        runs = sorted(
            (os.path.join(path, d) for d in os.listdir(path)),
            key=os.path.getmtime,
        )
        if not runs:
            raise FileNotFoundError(f"no checkpoints under {path}")
        return resolve_ckpt(runs[-1])
    latest = max(models, key=lambda f: int(f[len("model_") : -len(".pt")]))
    return os.path.join(path, latest)


def set_deterministic_basic_state(env):
    """Make the basic-mode robot state, command, and gait phase device-independent."""
    reset_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.no_grad():
        env._reset_system(reset_mask)
        for name in ("dof_pos_target", "dof_vel_target", "tau_ff", "dof_pos_history"):
            if hasattr(env, name):
                getattr(env, name).zero_()

        if hasattr(env, "commands"):
            env.commands.zero_()
            env.commands[:, 0] = 1.0

        if hasattr(env, "base_quat"):
            env.base_quat[:] = env.root_states[:, 3:7]
            env.base_lin_vel[:] = quat_rotate_inverse(
                env.base_quat, env.root_states[:, 7:10]
            )
            env.base_ang_vel[:] = quat_rotate_inverse(
                env.base_quat, env.root_states[:, 10:13]
            )
            env.projected_gravity[:] = quat_rotate_inverse(
                env.base_quat, env.gravity_vec
            )
            env.base_height = env.root_states[:, 2:3]
        if hasattr(env, "dof_pos_obs"):
            env.dof_pos_obs = env.dof_pos - env.default_dof_pos

        if hasattr(env, "phase"):
            env.phase.zero_()
            env.phase_obs[:, 0] = 0.0
            env.phase_obs[:, 1] = 1.0
            if hasattr(env, "phase_frequency"):
                low, high = map(float, env.cfg.control.gait_freq)
                env.phase_frequency.fill_(0.5 * (low + high))
            if hasattr(env, "_update_gait_reference"):
                env._update_gait_reference()
            if hasattr(env, "_update_cmd_switch"):
                env._update_cmd_switch()

        env.episode_length_buf.zero_()
        env._reset_buffers()


def configure_contact_friction_grid(env_cfg, friction_grid):
    """Configure native per-environment friction storage before backend setup."""
    if friction_grid is None:
        return
    set_domain_randomization_range(
        env_cfg,
        "contact_friction_range",
        [float(value) for value in friction_grid],
    )


def configure_contact_friction_dr(env_cfg, mode, friction_grid=None):
    """Resolve friction DR before backend topology is created."""
    if mode == "off" and friction_grid is not None:
        raise ValueError("contact-friction DR off cannot be combined with a grid")
    if mode == "off":
        set_domain_randomization_range(env_cfg, "contact_friction_range", None)
        return
    if friction_grid is not None:
        set_domain_randomization_range(
            env_cfg,
            "contact_friction_range",
            [float(value) for value in friction_grid],
        )
    if (
        mode == "on"
        and get_domain_randomization_range(env_cfg, "contact_friction_range") is None
    ):
        raise ValueError(
            "contact-friction DR on requires a configured range or explicit grid"
        )
    if mode not in ("config", "on"):
        raise ValueError(f"unknown contact-friction DR mode {mode!r}")


def crossed_contact_friction_grid(command_cases, low, high):
    """Give every command case the same evenly spaced friction levels."""
    command_cases = np.asarray(command_cases)
    values = np.empty(len(command_cases), dtype=np.float32)
    for command_case in dict.fromkeys(command_cases.tolist()):
        indices = np.flatnonzero(command_cases == command_case)
        values[indices] = np.linspace(low, high, len(indices), dtype=np.float32)
    return values


def crossed_parameter_samples(command_cases, low, high, width, seed):
    """Give every command case the same deterministic parameter samples."""
    command_cases = np.asarray(command_cases)
    values = np.empty((len(command_cases), width), dtype=np.float32)
    case_indices = [
        np.flatnonzero(command_cases == command_case)
        for command_case in dict.fromkeys(command_cases.tolist())
    ]
    sample_count = max(map(len, case_indices))
    samples = (
        np.random.default_rng(seed)
        .uniform(low, high, size=(sample_count, width))
        .astype(np.float32)
    )
    for indices in case_indices:
        values[indices] = samples[: len(indices)]
    return values


def configure_scale_range(env_cfg, name, values):
    """Replace one enabled episodic tensor range before task setup."""
    if values is None:
        return
    if get_domain_randomization_range(env_cfg, name) is None:
        raise ValueError(f"{name} is disabled by the selected DR mode")
    set_domain_randomization_range(env_cfg, name, [float(value) for value in values])


def apply_parameter_samples(
    env,
    command_cases,
    seed,
    stiffness_range=None,
    damping_range=None,
    link_mass_range=None,
):
    """Apply deterministic, command-balanced PD and mass samples."""
    randomizer = env.domain_randomizer
    num_envs = env.num_envs
    if stiffness_range is None:
        stiffness_t = randomizer.episode_scale("p_gains")
        stiffness = (
            np.ones((num_envs, env.num_actuators), dtype=np.float32)
            if stiffness_t is None
            else stiffness_t.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    else:
        stiffness = crossed_parameter_samples(
            command_cases, *stiffness_range, env.num_actuators, seed + 101
        )
        stiffness_t = torch.as_tensor(stiffness, device=env.device)
        randomizer.set_episode_scale(
            "p_gains",
            torch.ones(num_envs, dtype=torch.bool, device=env.device),
            stiffness_t,
        )

    if damping_range is None:
        damping_t = randomizer.episode_scale("d_gains")
        damping = (
            np.ones((num_envs, env.num_actuators), dtype=np.float32)
            if damping_t is None
            else damping_t.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    else:
        damping = crossed_parameter_samples(
            command_cases, *damping_range, env.num_actuators, seed + 202
        )
        damping_t = torch.as_tensor(damping, device=env.device)
        randomizer.set_episode_scale(
            "d_gains",
            torch.ones(num_envs, dtype=torch.bool, device=env.device),
            damping_t,
        )

    if link_mass_range is None:
        link_mass_t = randomizer.link_mass_scale
        link_mass = (
            np.ones((num_envs, env.num_bodies), dtype=np.float32)
            if link_mass_t is None
            else link_mass_t.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    else:
        link_mass = crossed_parameter_samples(
            command_cases, *link_mass_range, env.num_bodies, seed + 303
        )
        link_mass_t = torch.as_tensor(link_mass, device=env.device)
        randomizer.link_mass_scale.copy_(link_mass_t)
        env._backend.set_link_mass_scale(
            torch.arange(num_envs, device=env.device), link_mass_t
        )
    return stiffness, damping, link_mass


def current_contact_friction(env):
    """Return one recorded contact coefficient per environment.

    Fixed-base tasks do not construct ``DomainRandomizer`` because contacts are
    disabled for them.  Retain the evaluator's general-task behavior by
    recording their configured nominal coefficient instead.
    """
    randomizer = getattr(env, "domain_randomizer", None)
    if randomizer is None:
        terrain = getattr(env.cfg, "terrain", None)
        nominal = float(getattr(terrain, "dynamic_friction", 1.0))
        return np.full(env.num_envs, nominal, dtype=np.float32)
    return (
        randomizer.contact_friction.detach().cpu().numpy().astype(np.float32, copy=True)
    )


def build(
    task,
    eval_backend,
    eval_device,
    num_envs,
    t_end,
    ckpt,
    reset_mode,
    seed,
    contact_friction_grid=None,
    contact_friction_dr="config",
    original_cfg=False,
    mujoco_njmax=None,
    domain_randomization="config",
    stiffness_scale_range=None,
    damping_scale_range=None,
    link_mass_scale_range=None,
    control_at_sim_frequency=False,
    mujoco_geom_solref=None,
):
    import gym.envs  # noqa: F401 — registers tasks

    # reset_to_uniform lays ICs on a sqrt(N)xsqrt(N) grid (pendulum) — needs a
    # perfect square.  Legged tasks use reset_to_range (distributional eval).
    if reset_mode == "reset_to_uniform":
        root = int(math.isqrt(num_envs))
        if root * root != num_envs:
            raise ValueError(f"num_envs must be a perfect square; got {num_envs}")

    checkpoint_path = resolve_ckpt(ckpt)
    if original_cfg:
        registered_env_cfg, registered_train_cfg = load_original_cfgs_from_run(
            task, Path(checkpoint_path).parent
        )
    else:
        registered_env_cfg, registered_train_cfg = task_registry.get_cfgs(task)
    # Registry configs are process-wide instances. Evaluation overrides must
    # not leak into a later programmatic build in the same process.
    env_cfg = copy.deepcopy(registered_env_cfg)
    train_cfg = copy.deepcopy(registered_train_cfg)
    if mujoco_geom_solref is not None:
        env_cfg.mujoco.solref = mujoco_geom_solref
    configure_contact_friction_dr(
        env_cfg,
        contact_friction_dr,
        contact_friction_grid,
    )
    if domain_randomization == "off" and contact_friction_grid is not None:
        raise ValueError(
            "domain randomization off cannot be combined with a friction grid"
        )
    if contact_friction_dr != "config" and domain_randomization != "config":
        raise ValueError(
            "contact-friction and domain-randomization overrides cannot both "
            "be explicit"
        )
    apply_domain_randomization_override(env_cfg, domain_randomization)
    configure_scale_range(env_cfg, "p_gains", stiffness_scale_range)
    configure_scale_range(env_cfg, "d_gains", damping_scale_range)
    configure_scale_range(env_cfg, "link_mass_scale_range", link_mass_scale_range)
    if mujoco_njmax is not None:
        env_cfg.mujoco.njmax = int(mujoco_njmax)
    env_cfg.env.num_envs = num_envs
    env_cfg.init_state.reset_mode = reset_mode
    # Keep the eval controlled: no pushes, fixed commands for the whole episode.
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = t_end + 10.0
    if control_at_sim_frequency:
        env_cfg.control.ctrl_frequency = env_cfg.control.desired_sim_frequency
    # reset_to_uniform runs one long episode; range-reset legged eval keeps the
    # task's own episode length so survival-to-timeout is meaningful.
    if reset_mode == "reset_to_uniform":
        env_cfg.env.episode_length_s = t_end + 10.0
    env_cfg.seed = seed
    train_cfg.seed = seed
    train_cfg.runner.device = eval_device
    train_cfg.runner.resume = False  # we load the checkpoint explicitly below
    if hasattr(train_cfg, "logging"):
        train_cfg.logging.enable_local_saving = False
    set_seed(seed)

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg, log_root=None)  # no run dir on disk

    env = task_registry.make_env(
        task, env_cfg, device=eval_device, headless=True, backend=eval_backend
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(checkpoint_path, load_optimizer=False)
    runner.switch_to_eval()
    if reset_mode == "reset_to_basic":
        set_deterministic_basic_state(env)
    return env, runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="pendulum")
    p.add_argument("--ckpt", required=True, help="run dir or model_*.pt")
    p.add_argument("--train_label", required=True, help="e.g. cpu / warp / vsim")
    p.add_argument("--eval_backend", choices=["mujoco", "vsim"], default="mujoco")
    p.add_argument("--eval_device", default="cpu")
    p.add_argument("--eval_label", default=None, help="defaults to backend/device")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--t_end", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--reset_mode",
        default="reset_to_uniform",
        help="reset_to_uniform (pendulum grid) or reset_to_range (legged)",
    )
    p.add_argument(
        "--record_dof",
        action="store_true",
        help="record the full dof_pos trajectory [T,N,ndof] (for cross-backend "
        "DOF-RMS comparison; pair with --reset_mode reset_to_basic so ICs match)",
    )
    p.add_argument(
        "--record_tracking",
        action="store_true",
        help="record commands and base velocity trajectories for legged-policy "
        "command-tracking analysis",
    )
    p.add_argument(
        "--record_policy_io",
        action="store_true",
        help="record all configured actor/critic observations, raw policy "
        "outputs, and scaled environment action fields",
    )
    p.add_argument(
        "--command_profile",
        choices=["sampled", "hardware", "go2", "forward_3p0"],
        default="sampled",
        help="sampled uses task commands; hardware applies a fixed nine-case "
        "suite; go2 adds the 3 m/s training-speed extreme",
    )
    p.add_argument(
        "--settling_time",
        type=float,
        default=0.5,
        help="seconds excluded before accumulating hardware-oriented metrics",
    )
    p.add_argument(
        "--contact_threshold",
        type=float,
        default=20.0,
        help="contact-force threshold in newtons for gait-quality metrics",
    )
    p.add_argument(
        "--contact_friction_grid",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="evaluate an explicit crossed friction grid. Every command case "
        "receives the same evenly spaced LOW..HIGH levels; LOW=HIGH is a "
        "fixed-friction domain.",
    )
    p.add_argument(
        "--contact-friction-dr",
        choices=["config", "on", "off"],
        default="config",
        help="Use the task config, require friction DR, or disable it before setup.",
    )
    p.add_argument(
        "--domain-randomization",
        choices=DOMAIN_RANDOMIZATION_MODES,
        default="config",
        help="Use the configured DR bundle, disable every axis, or isolate "
        "friction, PD gains, or link mass.",
    )
    p.add_argument(
        "--stiffness-scale-range",
        "--stiffness-scale-grid",
        dest="stiffness_scale_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="deterministically sample per-actuator stiffness scales from LOW..HIGH",
    )
    p.add_argument(
        "--damping-scale-range",
        "--damping-scale-grid",
        dest="damping_scale_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="deterministically sample per-actuator damping scales from LOW..HIGH",
    )
    p.add_argument(
        "--link-mass-scale-range",
        "--link-mass-scale-grid",
        dest="link_mass_scale_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="deterministically sample per-link mass/inertia scales from LOW..HIGH",
    )
    p.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load the environment and runner configs saved beside the checkpoint.",
    )
    p.add_argument(
        "--mujoco_njmax",
        type=int,
        default=None,
        help="Override MuJoCo/Warp constraint capacity for evaluation.",
    )
    p.add_argument(
        "--velocity_impulse",
        type=float,
        default=0.0,
        help="paper-style planar base-velocity impulse magnitude in m/s",
    )
    p.add_argument(
        "--impulse_start_time",
        type=float,
        default=5.0,
        help="start time for phase-staggered velocity impulses",
    )
    p.add_argument(
        "--impulse_stagger_time",
        type=float,
        default=0.5,
        help="time window over which impulses are staggered across environments",
    )
    p.add_argument(
        "--impulse_directions",
        type=int,
        default=36,
        help="number of evenly spaced planar impulse directions",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    eval_label = args.eval_label or (
        args.eval_backend
        if args.eval_backend == "vsim"
        else f"mujoco-{args.eval_device}"
    )

    checkpoint_path = os.path.abspath(resolve_ckpt(args.ckpt))
    checkpoint_sha256 = sha256_file(checkpoint_path)
    env, runner = build(
        task=args.task,
        eval_backend=args.eval_backend,
        eval_device=args.eval_device,
        num_envs=args.num_envs,
        t_end=args.t_end,
        ckpt=checkpoint_path,
        reset_mode=args.reset_mode,
        seed=args.seed,
        contact_friction_grid=args.contact_friction_grid,
        contact_friction_dr=args.contact_friction_dr,
        original_cfg=args.original_cfg,
        mujoco_njmax=args.mujoco_njmax,
        domain_randomization=args.domain_randomization,
        stiffness_scale_range=args.stiffness_scale_range,
        damping_scale_range=args.damping_scale_range,
        link_mass_scale_range=args.link_mass_scale_range,
    )
    weights = runner.critic_cfg["reward"]["weights"]  # {term: weight}, zeros removed
    terms = list(weights)
    n_steps = int(args.t_end * float(env.cfg.control.ctrl_frequency))
    N, dev = args.num_envs, env.device
    is_pendulum = args.task == "pendulum"
    if not is_pendulum and not 0 <= args.settling_time < args.t_end:
        raise ValueError(
            f"settling_time must be in [0, t_end); got {args.settling_time}"
        )
    if args.contact_threshold <= 0:
        raise ValueError("contact_threshold must be positive")
    control_frequency = float(env.cfg.control.ctrl_frequency)
    applied_link_mass = None
    robot_mass_kg = None
    if args.velocity_impulse < 0:
        raise ValueError("velocity_impulse cannot be negative")
    if args.impulse_directions <= 0:
        raise ValueError("impulse_directions must be positive")
    if args.velocity_impulse:
        impulse_end = args.impulse_start_time + args.impulse_stagger_time
        if args.impulse_start_time < 0 or impulse_end >= args.t_end:
            raise ValueError(
                "impulse interval must start at or after zero and end before t_end"
            )
        impulse_steps, impulse_angles, impulse_delta_velocity = (
            velocity_impulse_schedule(
                N,
                control_frequency,
                args.impulse_start_time,
                args.impulse_stagger_time,
                args.velocity_impulse,
                args.impulse_directions,
            )
        )
    else:
        impulse_steps = np.full(N, -1, dtype=np.int64)
        impulse_angles = np.full(N, np.nan, dtype=np.float32)
        impulse_delta_velocity = np.zeros((N, 2), dtype=np.float32)
    command_cases = (
        np.full(N, "pendulum", dtype="<U16")
        if is_pendulum
        else apply_command_profile(env, args.command_profile)
    )
    if args.contact_friction_grid is not None:
        friction_low, friction_high = args.contact_friction_grid
        applied_contact_friction = crossed_contact_friction_grid(
            command_cases,
            friction_low,
            friction_high,
        )
        all_env_ids = torch.arange(N, dtype=torch.long, device=dev)
        env._backend.set_contact_friction(
            all_env_ids,
            torch.as_tensor(applied_contact_friction, device=dev),
        )
    else:
        applied_contact_friction = current_contact_friction(env)
    if is_pendulum:
        applied_stiffness_scale = None
        applied_damping_scale = None
        applied_link_mass_scale = None
    else:
        (
            applied_stiffness_scale,
            applied_damping_scale,
            applied_link_mass_scale,
        ) = apply_parameter_samples(
            env,
            command_cases,
            args.seed,
            args.stiffness_scale_range,
            args.damping_scale_range,
            args.link_mass_scale_range,
        )
        applied_link_mass = (
            env._backend.link_mass.detach().cpu().numpy().astype(np.float32)
        )
        robot_mass_kg = applied_link_mass.sum(axis=1)
    eval_commands = (
        None
        if is_pendulum
        else env.commands[:, :3].detach().cpu().numpy().astype(np.float32)
    )
    legged_accumulator = (
        None
        if is_pendulum
        else LeggedMetricAccumulator(
            env,
            settle_steps=int(
                round(args.settling_time * float(env.cfg.control.ctrl_frequency))
            ),
            contact_threshold=args.contact_threshold,
            num_steps=n_steps,
            robot_mass_kg=robot_mass_kg,
        )
    )

    per_term_sum = {t: torch.zeros(N, device=dev) for t in terms}
    base_z = np.empty((n_steps, N), dtype=np.float32)
    terminated = np.zeros((n_steps, N), dtype=np.bool_)
    ever_term = torch.zeros(N, dtype=torch.bool, device=dev)
    first_term = torch.full((N,), n_steps, dtype=torch.long, device=dev)
    theta = np.empty((n_steps, N), dtype=np.float32) if is_pendulum else None
    omega = np.empty((n_steps, N), dtype=np.float32) if is_pendulum else None
    # projected_gravity z: -1 upright, 0 on its side (legged uprightness).
    upright = None if is_pendulum else np.empty((n_steps, N), dtype=np.float32)
    dof_traj = (
        np.empty((n_steps, N, env.num_dof), dtype=np.float32)
        if args.record_dof
        else None
    )
    record_tracking = (
        args.record_tracking
        and not is_pendulum
        and hasattr(env, "commands")
        and hasattr(env, "base_lin_vel")
        and hasattr(env, "base_ang_vel")
    )
    command_traj = (
        np.empty((n_steps, N, 3), dtype=np.float32) if record_tracking else None
    )
    base_lin_vel_traj = (
        np.empty((n_steps, N, 3), dtype=np.float32) if record_tracking else None
    )
    base_ang_vel_traj = (
        np.empty((n_steps, N, 3), dtype=np.float32) if record_tracking else None
    )
    has_position_reference = (
        not is_pendulum and actuated_position_reference(env) is not None
    )
    reference_dof_rmse = (
        np.empty((n_steps, N), dtype=np.float32)
        if record_tracking and has_position_reference
        else None
    )
    if args.record_policy_io:
        original_env_cfg, _ = load_original_cfgs_from_run(
            args.task, Path(checkpoint_path).parent
        )
        original_scales = class_to_dict(original_env_cfg.scaling)
        actor_observation_fields = list(runner.actor_cfg["obs"])
        critic_observation_fields = list(runner.critic_cfg["obs"])
        action_fields = list(runner.actor_cfg["actions"])
        actor_observation_names = state_component_names(env, actor_observation_fields)
        critic_observation_names = state_component_names(env, critic_observation_fields)
        action_names = state_component_names(env, action_fields)
        actor_observation_scales = state_component_scales(
            env, actor_observation_fields, original_scales
        )
        critic_observation_scales = state_component_scales(
            env, critic_observation_fields, original_scales
        )
        action_scales = state_component_scales(env, action_fields, original_scales)
        policy_io_scale_source = original_cfg_source_dir(Path(checkpoint_path).parent)
        actor_observations = np.empty(
            (n_steps, N, len(actor_observation_names)), dtype=np.float32
        )
        critic_observations = np.empty(
            (n_steps, N, len(critic_observation_names)), dtype=np.float32
        )
        policy_actions = np.empty((n_steps, N, len(action_names)), dtype=np.float32)
        applied_actions = np.empty_like(policy_actions)
    else:
        actor_observations = None
        critic_observations = None
        policy_actions = None
        applied_actions = None

    with torch.no_grad():
        for k in range(n_steps):
            alive_before_step = ~ever_term
            if actor_observations is not None:
                actor_observations[k] = (
                    runner.get_obs(runner.actor_cfg["obs"]).detach().cpu().numpy()
                )
                critic_observations[k] = (
                    runner.get_obs(runner.critic_cfg["obs"]).detach().cpu().numpy()
                )
            actions = runner.get_inference_actions()
            if policy_actions is not None:
                policy_actions[k] = actions.detach().cpu().numpy()
            runner.set_actions(
                runner.actor_cfg["actions"],
                actions,
                runner.actor_cfg["disable_actions"],
            )
            impulse_envs = np.flatnonzero(impulse_steps == k)
            if len(impulse_envs):
                impulse_envs_device = torch.as_tensor(
                    impulse_envs,
                    dtype=torch.long,
                    device=dev,
                )
                delta_velocity = torch.as_tensor(
                    impulse_delta_velocity[impulse_envs],
                    dtype=env.root_states.dtype,
                    device=dev,
                )
                env.root_states[impulse_envs_device, 7:9] += delta_velocity
                env._backend.set_all_root_states()
            base_z[k] = env.root_states[:, 2].detach().cpu().numpy()
            if dof_traj is not None:
                dof_traj[k] = env.dof_pos.detach().cpu().numpy()
            if record_tracking:
                command_traj[k] = env.commands[:, :3].detach().cpu().numpy()
                base_lin_vel_traj[k] = env.base_lin_vel.detach().cpu().numpy()
                base_ang_vel_traj[k] = env.base_ang_vel.detach().cpu().numpy()
            if reference_dof_rmse is not None:
                reference_target = actuated_position_reference(env)
                reference_target = reference_target + env.default_dof_pos.index_select(
                    1, env.actuated_dof_indices
                )
                actuated_position = env.dof_pos.index_select(
                    1, env.actuated_dof_indices
                )
                reference_dof_rmse[k] = (
                    torch.sqrt(
                        torch.mean((reference_target - actuated_position) ** 2, dim=1)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            if is_pendulum:
                theta[k] = env.dof_pos[:, 0].detach().cpu().numpy()
                omega[k] = env.dof_vel[:, 0].detach().cpu().numpy()
            else:
                upright[k] = env.projected_gravity[:, 2].detach().cpu().numpy()
            env.step()
            if applied_actions is not None:
                applied_actions[k] = (
                    torch.cat(
                        [getattr(env, name) for name in runner.actor_cfg["actions"]],
                        dim=-1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            term = env.terminated
            if legged_accumulator is not None:
                legged_accumulator.update(k, alive_before_step & ~term)
            for reward_name, weight in weights.items():
                per_term_sum[reward_name] += weight * runner.reward_functions[
                    reward_name
                ]().to(dev)
            terminated[k] = env.terminated.detach().cpu().numpy()
            newly = env.terminated & ~ever_term
            first_term[newly] = k
            ever_term |= env.terminated

    # Mean over steps => same scale as the training total_rewards curve.
    per_term_mean = {t: (per_term_sum[t] / n_steps).cpu().numpy() for t in terms}
    mean_reward = np.sum([per_term_mean[t] for t in terms], axis=0)
    survived = (~ever_term).detach().cpu().numpy()
    episode_steps = torch.where(ever_term, first_term + 1, first_term)
    ep_len = (
        (episode_steps.float() / float(env.cfg.control.ctrl_frequency)).cpu().numpy()
    )
    hardware_metrics = {}
    hardware_artifacts = {}
    if legged_accumulator is not None:
        hardware_metrics = legged_accumulator.finalize(survived)
        hardware_artifacts = legged_accumulator.artifacts
        hardware_metrics["survival"] = survived.astype(np.float32)
        hardware_metrics["episode_duration"] = ep_len.astype(np.float32)
        if args.velocity_impulse:
            first_term_step = first_term.cpu().numpy()
            pre_impulse_failure = first_term_step < impulse_steps
            post_impulse_failure = np.full(N, np.nan, dtype=np.float32)
            eligible = ~pre_impulse_failure
            post_impulse_failure[eligible] = (~survived[eligible]).astype(np.float32)
            hardware_metrics["disturbance_failure"] = post_impulse_failure
            hardware_metrics["disturbance_pre_impulse_failure"] = (
                pre_impulse_failure.astype(np.float32)
            )
            hardware_metrics["disturbance_total_failure"] = (~survived).astype(
                np.float32
            )

    env._backend.close()

    extra = {}
    if is_pendulum:
        tw = np.arctan2(np.sin(theta), np.cos(theta))
        hold = int(round(float(env.cfg.control.ctrl_frequency)))
        success = (np.abs(tw[-hold:]) < 0.14).all(0) & (
            np.abs(omega[-hold:]) < 0.5
        ).all(0)
        extra = {"theta": theta, "omega": omega, "success": success}
        headline = f"upright {100 * success.mean():.1f}%"
    else:
        extra = {
            "upright": upright,
            "command_case": command_cases,
            "eval_commands": eval_commands,
            "phase_frequency_hz": (
                env.phase_frequency.detach().cpu().numpy().astype(np.float32)
                if hasattr(env, "phase_frequency")
                else np.full(N, np.nan, dtype=np.float32)
            ),
            "hardware_metric_names": np.asarray(list(hardware_metrics)),
            "hardware_metric_metadata": np.asarray(json.dumps(metric_metadata())),
            "actuated_dof_names": np.asarray(env.actuated_dof_names),
            "body_names": np.asarray(env.robot_layout.body_names),
            "foot_names": np.asarray(env.robot_layout.body_groups["feet"]),
            "link_mass_kg": applied_link_mass,
            "stiffness_scale": applied_stiffness_scale,
            "damping_scale": applied_damping_scale,
            "link_mass_scale": applied_link_mass_scale,
            "robot_mass_kg": robot_mass_kg,
            "impulse_step": impulse_steps,
            "impulse_direction_rad": impulse_angles,
            "impulse_delta_velocity": impulse_delta_velocity,
        }
        extra.update(
            {f"metric_{name}": values for name, values in hardware_metrics.items()}
        )
        extra.update(hardware_artifacts)
        headline = f"survival {100 * survived.mean():.1f}%"
    if dof_traj is not None:
        extra["dof_traj"] = dof_traj
        extra["dof_names"] = np.array(env.dof_names)
    if record_tracking:
        extra.update(
            {
                "commands": command_traj,
                "base_lin_vel": base_lin_vel_traj,
                "base_ang_vel": base_ang_vel_traj,
            }
        )
    if reference_dof_rmse is not None:
        extra["reference_dof_rmse"] = reference_dof_rmse
    if actor_observations is not None:
        extra.update(
            {
                "actor_observations": actor_observations,
                "actor_observation_fields": np.asarray(actor_observation_fields),
                "actor_observation_names": np.asarray(actor_observation_names),
                "actor_observation_scales": np.asarray(
                    actor_observation_scales, dtype=np.float32
                ),
                "critic_observations": critic_observations,
                "critic_observation_fields": np.asarray(critic_observation_fields),
                "critic_observation_names": np.asarray(critic_observation_names),
                "critic_observation_scales": np.asarray(
                    critic_observation_scales, dtype=np.float32
                ),
                "policy_actions": policy_actions,
                "applied_actions": applied_actions,
                "action_fields": np.asarray(action_fields),
                "action_names": np.asarray(action_names),
                "action_scales": np.asarray(action_scales, dtype=np.float32),
                "policy_io_scale_source": str(policy_io_scale_source),
            }
        )

    print(
        f"[{args.train_label} -> {eval_label}] mean reward "
        f"{mean_reward.mean():+.3f}  |  {headline}"
    )
    if hardware_metrics:

        def _finite_mean(values):
            finite = values[np.isfinite(values)]
            return float(np.mean(finite)) if len(finite) else math.nan

        _mean = {
            name: _finite_mean(values) for name, values in hardware_metrics.items()
        }
        print(
            "hardware metrics | "
            f"vx {_mean['tracking_vx_rmse']:.3f} m/s | "
            f"vy {_mean['tracking_vy_rmse']:.3f} m/s | "
            f"yaw {_mean['tracking_yaw_rmse']:.3f} rad/s | "
            f"tilt {_mean['base_tilt_rms']:.2f} deg | "
            f"target accel {_mean['target_acceleration_rms']:.1f} rad/s^2 | "
            f"torque {_mean['torque_utilization_rms']:.2f} limit | "
            f"slip {_mean['foot_slip_speed_rms']:.3f} m/s"
        )
        print(
            "gait quality     | "
            f"height std {1000 * _mean['base_height_std']:.1f} mm | "
            f"trot {100 * _mean['gait_trot_classified']:.1f}% | "
            f"RPD error {_mean['gait_rpd_trot_error']:.3f} rad | "
            f"GRF CV {_mean['grf_balance_cv']:.3f}"
        )
        if "swing_clearance_p95_mean" in _mean:
            print(
                "swing clearance | "
                f"mean p95 {1000 * _mean['swing_clearance_p95_mean']:.1f} mm | "
                f"lowest foot {1000 * _mean['swing_clearance_p95_min']:.1f} mm"
            )
        if args.velocity_impulse:
            print(
                "disturbance      | "
                f"{args.velocity_impulse:g} m/s | "
                f"pre {100 * _mean['disturbance_pre_impulse_failure']:.1f}% | "
                f"post {100 * _mean['disturbance_failure']:.1f}% | "
                f"total {100 * _mean['disturbance_total_failure']:.1f}%"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    output_path = Path(args.out)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            train_label=args.train_label,
            eval_label=eval_label,
            task=args.task,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_iteration=np.int64(runner.it),
            reset_mode=args.reset_mode,
            command_profile=args.command_profile,
            num_envs=np.int64(N),
            duration_s=np.float32(args.t_end),
            seed=np.int64(args.seed),
            settling_time_s=np.float32(args.settling_time),
            contact_threshold_n=np.float32(args.contact_threshold),
            contact_friction_dr=args.contact_friction_dr,
            domain_randomization=args.domain_randomization,
            stiffness_scale_range=np.asarray(
                args.stiffness_scale_range or [np.nan, np.nan], dtype=np.float32
            ),
            damping_scale_range=np.asarray(
                args.damping_scale_range or [np.nan, np.nan], dtype=np.float32
            ),
            link_mass_scale_range=np.asarray(
                args.link_mass_scale_range or [np.nan, np.nan], dtype=np.float32
            ),
            original_cfg=np.bool_(args.original_cfg),
            mujoco_njmax=np.int64(
                -1 if args.mujoco_njmax is None else args.mujoco_njmax
            ),
            contact_friction=applied_contact_friction.astype(np.float32),
            contact_friction_grid=np.asarray(
                args.contact_friction_grid
                if args.contact_friction_grid is not None
                else [np.nan, np.nan],
                dtype=np.float32,
            ),
            mean_reward=mean_reward.astype(np.float32),
            survived=survived,
            ep_len=ep_len.astype(np.float32),
            base_z=base_z,
            terminated=terminated,
            terms=np.array(terms),
            ctrl_hz=float(env.cfg.control.ctrl_frequency),
            **{f"term_{t}": per_term_mean[t].astype(np.float32) for t in terms},
            **extra,
        )
    temporary_path.replace(output_path)
    if hardware_metrics:
        summary_path = os.path.splitext(args.out)[0] + ".summary.json"
        summary = {
            "protocol": {
                "task": args.task,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_iteration": runner.it,
                "train_label": args.train_label,
                "eval_label": eval_label,
                "num_envs": N,
                "seed": args.seed,
                "duration_s": args.t_end,
                "settling_time_s": args.settling_time,
                "command_profile": args.command_profile,
                "contact_threshold_n": args.contact_threshold,
                "contact_friction_grid": args.contact_friction_grid,
                "contact_friction_dr": args.contact_friction_dr,
                "domain_randomization": args.domain_randomization,
                "stiffness_scale_range": args.stiffness_scale_range,
                "damping_scale_range": args.damping_scale_range,
                "link_mass_scale_range": args.link_mass_scale_range,
                "original_cfg": args.original_cfg,
                "mujoco_njmax": args.mujoco_njmax,
                "robot_mass_kg": float(np.mean(robot_mass_kg)),
                "robot_mass_range_kg": [
                    float(np.min(robot_mass_kg)),
                    float(np.max(robot_mass_kg)),
                ],
                "velocity_impulse_m_per_s": args.velocity_impulse,
                "impulse_start_time_s": args.impulse_start_time,
                "impulse_stagger_time_s": args.impulse_stagger_time,
                "impulse_directions": args.impulse_directions,
            },
            "metric_definitions": metric_metadata(),
            "results": summarize_metrics(hardware_metrics, command_cases),
        }
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")
        print(f"wrote {summary_path}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
