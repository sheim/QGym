"""Real-MJX + real-Q composite barrier: parity, h(s), g(u), derivatives.

Reuses, read-only, the MJX diagnostic model and the recorded Q2 rollout from the
earlier MJX experiment, plus a Torch reference exported by ``torch_reference.py``
(the policy that the Q artifact was fitted for).

Run:  env/bin/python scripts/cbf_composite.py --out results/cbf_composite.json
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import cbf_composite_lib as lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/nov10")
    ap.add_argument(
        "--model",
        default="/tmp/opencode/mjx-go2trot-20261115-a/artifacts/variants/"
                "model_env0_diag_nocylbox.mjb")
    ap.add_argument("--q-manifest",
                    default="/tmp/opencode/height-q-exp-20261110-final-0p05-fair-v1/"
                            "100pct/manifest.json")
    ap.add_argument("--out", default="results/cbf_composite.json")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--threshold-quantile", type=float, default=0.5)
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)

    art = lib.load_artifacts(args.artifacts, args.q_manifest)
    model, mjm = lib.build_model(args.model)
    step = jax.jit(lib.build_step(model, art))
    ref = np.load(f"{args.artifacts}/torch_ref.npz")
    T = ref["qpos_pre"].shape[0]
    commands = jnp.array([1.0, 0.0, 0.0], jnp.float64)
    rep = {"device": str(jax.devices()[0]), "T": T, "beta": args.beta,
           "model": args.model,
           "model_sha256_note": "diagnostic no-cylinder-box variant "
                                "(sha f55cf10b...); baseline model .mjb "
                                "(sha 52eaf8c9...) is byte-identical to the "
                                "earlier MJX experiment's env-0 model",
           "obs_dim": lib.OBS_DIM}

    qpos_all = jnp.asarray(ref["qpos_pre"], jnp.float64)
    qvel_all = jnp.asarray(ref["qvel_pre"], jnp.float64)
    target_all = jnp.asarray(ref["target_pre"], jnp.float64)
    phase_all = jnp.asarray(ref["phase"], jnp.float64)
    act_all = jnp.asarray(ref["action"], jnp.float64)

    def obs_vec(qpos, qvel, target, phase, freq):
        return lib.obs_from(qpos, qvel, target, commands, phase, freq, art)

    obs_j = jax.jit(jax.vmap(obs_vec))(qpos_all, qvel_all, target_all,
                                       phase_all[:, 0], phase_all[:, 1])
    act_j = jax.jit(jax.vmap(lambda o: lib.policy(art, o)))(obs_j)
    q_fn = lambda o, a: lib.q_value(art, o, a)  # noqa: E731
    q_j = jax.jit(jax.vmap(q_fn))(obs_j, act_j)

    rep["parity"] = {
        "obs_max_abs_vs_torch": float(jnp.abs(obs_j - ref["obs"]).max()),
        "action_max_abs_vs_torch": float(
            jnp.abs(act_j - ref["action"]).max()),
        "q_max_abs_vs_torch": float(jnp.abs(q_j - ref["q_value"]).max()),
        "action_max_abs_vs_recorded_rollout": float(
            jnp.abs(act_j - act_all).max()),
        "note": "policy-mean (act_inference) action, clean observation, Q input "
                "normalised with the artifact's obs_mean/obs_std",
    }

    # one-step physics parity of the JAX step against Q2's CPU rollout
    qpos_p, qvel_p = jax.jit(jax.vmap(
        lambda q, v, a: step(q, v, a)))(qpos_all, qvel_all, act_all)
    rep["step_parity_vs_q2_cpu"] = {
        "qpos_max_abs": float(jnp.abs(qpos_p - ref["qpos_post"]).max()),
        "qvel_max_abs": float(jnp.abs(qvel_p - ref["qvel_post"]).max()),
        "per_step_qpos_max_abs": np.abs(
            np.asarray(qpos_p) - ref["qpos_post"]).max(axis=1).tolist(),
    }

    c = float(jnp.quantile(q_j, args.threshold_quantile))
    rep["h_stats"] = {
        "min": float(q_j.min()), "max": float(q_j.max()),
        "median": float(jnp.median(q_j)), "mean": float(q_j.mean()),
        "std": float(q_j.std()), "threshold_c": c,
        "threshold_rule": f"median of Q(obs_t, pi(obs_t)) over the {T}-step "
                          "rollout (not independently calibrated)",
    }
    h_all = q_j - c

    def succ_h(u, qpos, qvel, phase, freq):
        """h(F(s,u)) = Q(obs_next, pi(obs_next)) - c."""
        _, _, o2 = lib.successor_obs(step, art, qpos, qvel, u, commands,
                                     phase, freq)
        return lib.q_value(art, o2, lib.policy(art, o2)) - c

    succ_h_v = jax.vmap(succ_h, in_axes=(0, 0, 0, 0, 0))

    # end-to-end check of the predictor: h(F(s_t, pi(s_t))) vs the recorded
    # h(s_{t+1}) on the same trajectory
    h_pred = jax.jit(succ_h_v)(act_all[:-1], qpos_all[:-1], qvel_all[:-1],
                               phase_all[:-1, 0], phase_all[:-1, 1])
    dh = np.asarray(h_pred - h_all[1:])
    rep["successor_prediction"] = {
        "h_pred_vs_recorded_max_abs": float(np.abs(dh).max()),
        "h_pred_vs_recorded_rms": float(np.sqrt((dh**2).mean())),
        "n": int(dh.size),
    }

    def make_g(t):
        qpos, qvel, phase, freq = qpos_all[t], qvel_all[t], phase_all[t, 0], phase_all[t, 1]
        h0 = h_all[t]

        def g(u):
            return succ_h(u, qpos, qvel, phase, freq) - (1.0 - args.beta) * h0

        return g, float(h0)

    g_chk, h0_chk = make_g(10)
    u_nom_t = act_all[10]
    jac_chk = jax.jit(jax.jacfwd(g_chk))(u_nom_t)
    eps = 1e-6
    fd = np.zeros(lib.N_ACT)
    for i in range(lib.N_ACT):
        e = np.zeros(lib.N_ACT)
        e[i] = eps
        fd[i] = float((g_chk(u_nom_t + e) - g_chk(u_nom_t - e)) / (2 * eps))
    rep["gradient_check"] = {
        "state": 10, "h0": h0_chk, "g": float(g_chk(u_nom_t)),
        "jac_max_abs": float(jnp.abs(jac_chk).max()),
        "jac_vs_fd_max_abs": float(np.abs(np.asarray(jac_chk) - fd).max()),
        "jac_vs_fd_rel": float(np.abs(np.asarray(jac_chk) - fd).max()
                              / max(1e-12, np.abs(fd).max())),
        "fd_eps": eps,
    }

    # directional derivative along one fixed direction, all rollout states
    rng = np.random.default_rng(20261115)
    v = rng.normal(size=lib.N_ACT)
    v /= np.linalg.norm(v)
    jac_all = jax.jit(jax.vmap(jax.jacfwd(make_g(10)[0])))(u_nom_t[None])[0]
    rep["directional"] = {
        "state": 10,
        "autodiff": float(jnp.dot(jac_all, v)),
        "fd": float((g_chk(u_nom_t + 1e-6 * v) - g_chk(u_nom_t - 1e-6 * v)) / 2e-6),
    }

    print(json.dumps(rep, indent=1))
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
