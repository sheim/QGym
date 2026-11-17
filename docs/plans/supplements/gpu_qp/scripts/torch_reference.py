"""Export a Torch/Q2 reference for the height-Q CBF investigation.

Run with the product's own locked interpreter (read-only) so the reference uses
Q2's real observation scaling, real policy artifact and real Q artifact:

  PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=disabled \
    BigRedGym/.venv/bin/python scripts/torch_reference.py \
      --checkout <Q2 checkout copy> --out <dir>

Writes ``torch_ref.npz`` (qpos/qvel pre-step, previous target, phase, the 48-D
scaled observation, the policy-mean action, and the height-Q value) plus the
env-0 ``MjModel`` and the parameters a JAX reimplementation needs.

Observation convention recorded in the Q artifact: policy-mean action
(``actor.act_inference``), clean observation (no observation noise), Q input
normalised with the artifact's ``obs_mean``/``obs_std``.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--q-artifact", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.checkout)
    import mujoco  # noqa: E402
    import gym.envs  # noqa: E402,F401
    from gym.utils.helpers import set_seed  # noqa: E402
    from gym.utils.task_registry import task_registry  # noqa: E402
    from learning.offline.q_artifact import HeightQ  # noqa: E402

    env_cfg, train_cfg = task_registry.get_cfgs("go2trot")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 100
    env_cfg.commands.resampling_time = 9999
    env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.seed = args.seed
    train_cfg.seed = args.seed
    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = False
    train_cfg.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        "go2trot", env_cfg, device="cpu", headless=True, backend="mujoco"
    )
    backend = env._backend
    model = backend._model_for_env(0)
    data = backend._datas[0]

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.load(args.checkpoint)
    runner.switch_to_eval()
    obs_names = list(runner.actor_cfg["obs"])
    q = HeightQ(args.q_artifact)

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "model_env0.mjb")
    mujoco.mj_saveModel(model, model_path)

    env.commands[:, 0] = 1.0
    env.commands[:, 1:] = 0.0
    env._refresh_base_observations(torch.ones(1, dtype=torch.bool, device=env.device))

    nq, nv, ndof = model.nq, model.nv, env.num_actuators
    T = args.steps
    rec = {k: np.zeros(shape) for k, shape in {
        "qpos_pre": (T, nq), "qvel_pre": (T, nv), "target_pre": (T, ndof),
        "phase": (T, 2), "obs": (T, 48), "action": (T, ndof), "q_value": (T,),
        "qpos_post": (T, nq), "qvel_post": (T, nv), "ncon_post": (T,),
        "torque": (T, ndof), "p_gains": (T, ndof), "d_gains": (T, ndof),
    }.items()}
    rec["ncon_post"] = np.zeros(T, dtype=np.int64)

    for t in range(T):
        with torch.no_grad():
            obs = env.get_states(obs_names)
            act = runner.alg.actor.act_inference(obs)
            qv = q.predict(obs, act)
        rec["qpos_pre"][t] = data.qpos
        rec["qvel_pre"][t] = data.qvel
        rec["target_pre"][t] = env.dof_pos_target[0].detach().cpu().numpy()
        rec["phase"][t] = [float(env.phase[0, 0]), float(env.phase_frequency[0, 0])]
        rec["obs"][t] = obs[0].detach().cpu().numpy()
        rec["action"][t] = act[0].detach().cpu().numpy()
        rec["q_value"][t] = float(qv[0])
        rec["p_gains"][t] = env.p_gains[0].detach().cpu().numpy()
        rec["d_gains"][t] = env.d_gains[0].detach().cpu().numpy()
        runner.set_actions(runner.actor_cfg["actions"], act)
        env.step()
        rec["torque"][t] = env.torques[0].detach().cpu().numpy()
        rec["qpos_post"][t] = data.qpos
        rec["qvel_post"][t] = data.qvel
        rec["ncon_post"][t] = data.ncon

    # Standing / saturated probes from a fresh reset.
    probes = {}
    env._reset_idx(torch.ones(1, dtype=torch.bool, device=env.device))
    env._refresh_base_observations(torch.ones(1, dtype=torch.bool, device=env.device))
    for name, target in [("stand", 0.0), ("sat1", 1.0), ("satfull", 10.0)]:
        with torch.no_grad():
            obs = env.get_states(obs_names)
            act = runner.alg.actor.act_inference(obs)
            qv = q.predict(obs, act)
        probes[f"{name}_qpos"] = np.array(data.qpos)
        probes[f"{name}_qvel"] = np.array(data.qvel)
        probes[f"{name}_target"] = np.array(env.dof_pos_target[0])
        probes[f"{name}_obs"] = obs[0].numpy()
        probes[f"{name}_action"] = act[0].numpy()
        probes[f"{name}_q"] = np.array(float(qv[0]))
        env.dof_pos_target[:] = target
        env.step()

    meta = {
        "checkout": os.path.abspath(args.checkout),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "q_artifact": os.path.abspath(args.q_artifact),
        "model_mjb": os.path.abspath(model_path),
        "model_sha256": sha256_file(model_path),
        "seed": args.seed,
        "obs_names": obs_names,
        "ctrl_dt": env_cfg.control.ctrl_dt,
        "sim_dt": env_cfg.sim_dt,
        "decimation": int(env_cfg.control.decimation),
        "max_episode_length": int(env.max_episode_length),
        "default_dof_pos": env.default_dof_pos.flatten().tolist(),
        "p_gains": rec["p_gains"][0].tolist(),
        "d_gains": rec["d_gains"][0].tolist(),
        "gain_constancy": {"p_gains": float(np.abs(np.diff(rec["p_gains"], axis=0)).max()) if T > 1 else 0.0, "d_gains": float(np.abs(np.diff(rec["d_gains"], axis=0)).max()) if T > 1 else 0.0},
        "torque_limits": env.actuated_torque_limits.tolist(),
        "canonical_to_native_dof": backend._canonical_to_native_dof_np.tolist(),
        "dof_pos_limits_lo": env._actuated_dof_pos_limits[:, 0].tolist(),
        "dof_pos_limits_hi": env._actuated_dof_pos_limits[:, 1].tolist(),
        "config_fingerprint": None,
    }
    np.savez(os.path.join(args.out, "torch_ref.npz"), **rec, **probes)
    with open(os.path.join(args.out, "torch_ref_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps({k: meta[k] for k in
                      ("checkpoint_sha256", "model_sha256", "obs_names")}, indent=1))
    print("q range", rec["q_value"].min(), rec["q_value"].max())
    print("action absmax", np.abs(rec["action"]).max())


if __name__ == "__main__":
    main()
