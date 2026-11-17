"""Shared pieces for the GPU-QP investigation: artifacts, obs, policy, Q, MJX step.

Everything is float64 JAX.  The policy is the artifact's own trained actor
(deterministic mean, ``act_inference``), and the Q is the artifact fitted for
that actor; no observation noise is injected (clean observation, as the Q was
fitted on).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

N_ACT = 12
OBS_DIM = 48
SIM_DT = 0.02


def load_artifacts(path, q_manifest):
    w = np.load(f"{path}/weights.npz")
    meta = json.load(open(f"{path}/torch_ref_meta.json"))
    man = json.load(open(q_manifest))
    J = lambda k: jnp.asarray(w[k], jnp.float64)  # noqa: E731
    return {
        "actor": [J("actor_NN_0_weight").T, J("actor_NN_0_bias"),
                  J("actor_NN_2_weight").T, J("actor_NN_2_bias"),
                  J("actor_NN_4_weight").T, J("actor_NN_4_bias"),
                  J("actor_NN_6_weight").T, J("actor_NN_6_bias")],
        "q": [J("q_net_0_weight").T, J("q_net_0_bias"),
              J("q_net_2_weight").T, J("q_net_2_bias"),
              J("q_net_4_weight").T, J("q_net_4_bias")],
        "obs_mean": jnp.asarray(man["obs_mean"], jnp.float64),
        "obs_std": jnp.asarray(man["obs_std"], jnp.float64),
        "default_dof": jnp.asarray(meta["default_dof_pos"], jnp.float64),
        "p_gains": jnp.asarray(meta["p_gains"], jnp.float64),
        "d_gains": jnp.asarray(meta["d_gains"], jnp.float64),
        "tau_lim": jnp.asarray(meta["torque_limits"], jnp.float64),
        "pos_lo": jnp.asarray(meta["dof_pos_limits_lo"], jnp.float64),
        "pos_hi": jnp.asarray(meta["dof_pos_limits_hi"], jnp.float64),
        "scale": {k: jnp.asarray(v, jnp.float64)
                  for k, v in man["policy_spec"]["scales"].items()},
        "meta": meta,
        "manifest": man,
    }


def mlp(layers, x, act):
    """layers = [W0,b0,W1,b1,...]; activation between layers, none after last."""
    n = len(layers) // 2
    for i in range(n):
        x = x @ layers[2 * i] + layers[2 * i + 1]
        if i < n - 1:
            x = act(x)
    return x


def policy(art, obs):
    return mlp(art["actor"], obs, jax.nn.elu)


def q_value(art, obs, action):
    x = (obs - art["obs_mean"]) / art["obs_std"]
    return jnp.squeeze(mlp(art["q"], jnp.concatenate([x, action]), jax.nn.relu))


def quat_rotate_inverse(q, v):
    """Q2 gym/utils/torch_quat.py quat_rotate_inverse, quaternion [x,y,z,w]."""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = jnp.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (q_vec @ v) * 2.0
    return a - b + c


def obs_from(qpos, qvel, prev_target, commands, phase, phase_freq, art):
    """48-D actor observation of Q2's Go2Trot, in the artifact's scaling.

    ``prev_target`` is the internal (scaled) joint target of the previous
    control interval; dividing by the target scale recovers the previous
    policy action, which is the ``dof_pos_target`` observation channel.
    """
    # MuJoCo stores the free-joint quaternion as (w, x, y, z); Q2's
    # quat_rotate_inverse expects (x, y, z, w).
    base_quat = qpos[jnp.array([4, 5, 6, 3])]
    sc = art["scale"]
    # Q2's root_states ang_vel is world-frame and base_ang_vel rotates it back,
    # which is exactly the body-local free-joint qvel[3:6].
    base_ang_vel = qvel[3:6] / sc["base_ang_vel"]
    proj_grav = quat_rotate_inverse(base_quat, jnp.array([0.0, 0.0, -1.0]))
    return jnp.concatenate([
        base_ang_vel,
        proj_grav,
        commands / sc["commands"],
        (qpos[7:] - art["default_dof"]) / sc["dof_pos_obs"],
        qvel[6:] / sc["dof_vel"],
        prev_target / sc["dof_pos_target"],
        jnp.array([jnp.sin(phase), jnp.cos(phase)]),
        jnp.array([phase_freq]),
    ])


def build_model(path):
    import mujoco
    import mujoco.mjx as mjx

    mjm = mujoco.MjModel.from_binary_path(path)
    return mjx.put_model(mjm, impl="jax"), mjm


def build_step(mjx_model, art):
    """One Q2 control interval: PD torque, per-joint clip, one mjx.step."""

    def step(qpos, qvel, u):
        target = u * art["scale"]["dof_pos_target"]
        tau = jnp.clip(
            art["p_gains"] * (target + art["default_dof"] - qpos[7:])
            - art["d_gains"] * qvel[6:], -art["tau_lim"], art["tau_lim"])
        qfrc = jnp.zeros(mjx_model.nv, jnp.float64).at[6:].set(tau)
        import mujoco.mjx as mjx

        data = mjx.make_data(mjx_model).replace(
            qpos=qpos, qvel=qvel, qfrc_applied=qfrc)
        out = mjx.step(mjx_model, data)
        return out.qpos, out.qvel

    return step


def successor_obs(step, art, qpos, qvel, u, commands, phase, phase_freq):
    qpos2, qvel2 = step(qpos, qvel, u)
    target_prev = u * art["scale"]["dof_pos_target"]
    phase2 = jnp.mod(phase + 2.0 * jnp.pi * SIM_DT * phase_freq,
                     2.0 * jnp.pi)
    return qpos2, qvel2, obs_from(qpos2, qvel2, target_prev, commands, phase2,
                                 phase_freq, art)
