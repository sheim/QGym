"""Device-side SQP: no host decisions inside the optimisation loop.

The whole constrained solve - linearisation (forward-mode Jacobian through MJX),
elastic QP (QPAX), backtracking on the true nonlinear constraint, trust-region
update and termination - runs inside one ``jax.lax.while_loop``.  Only the final
action, the original-constraint value and a status integer are read back.

Run:
  env/bin/python scripts/device_sqp.py --out results/device_sqp.json
  ... --profile          # wrap the warmed decisions in cuProfilerStart/Stop
"""

import argparse
import ctypes
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import qpax

import cbf_composite_lib as lib

N = 12
PENALTY = 10.0


def build_qp_solver(lo, hi):
    """Elastic QP with one linearised barrier row + box + Linf trust region."""

    def solve(W, u_nom, jac, b, u_center, trust):
        eye = jnp.eye(N)
        G = jnp.concatenate([
            -jac[None, :],
            eye, -eye,
            eye, -eye,
        ], axis=0)
        h = jnp.concatenate([
            jnp.asarray([b]),
            u_center + trust, -u_center + trust,
            hi, -lo,
        ])
        q = -W @ u_nom
        x, t, s1, s2, z1, z2, conv, it = qpax.solve_qp_elastic(
            W, q, G, h, PENALTY)
        slack = jnp.maximum(G @ x - h, 0.0)
        return x, slack, conv, it

    return solve


def build_device_sqp(g, gjac, qp_solve, lo, hi, max_sqp=8, backtracks=6):
    """One jitted device-side SQP.

    ``g(u, state...)`` value; ``gjac`` returns (value, gradient).
    """

    def run(u_nom, W, state):
        def cond(carry):
            u, trust, it, gv, done = carry
            return jnp.logical_not(done)

        def body(carry):
            u, trust, it, gv, done = carry
            gval, jac = gjac(u, *state)
            b = gval - jac @ u
            x, slack, conv, qp_it = qp_solve(W, u_nom, jac, b, u, trust)
            d = x - u

            def bt(c, _):
                alpha, ok, gtry = c
                g_now = g(u + alpha * d, *state)
                better = g_now > gval + 1e-9
                alpha = jnp.where(ok, alpha, jnp.where(better, alpha, 0.5 * alpha))
                ok = ok | better
                return (alpha, ok, jnp.where(ok, g_now, gtry)), None

            (alpha, accepted, g_try), _ = jax.lax.scan(
                bt, (jnp.asarray(1.0), jnp.asarray(False), gval),
                None, length=backtracks)
            u_next = jnp.where(accepted, u + alpha * d, u)
            g_next = jnp.where(accepted, g_try, gval)
            trust_next = jnp.where(accepted, jnp.minimum(2.0 * trust, 1.0),
                                   0.5 * trust)
            it_next = it + 1
            done_next = (g_next >= 0.0) | (it_next >= max_sqp) | (trust_next < 1e-3)
            return (u_next, trust_next, it_next, g_next, done_next)

        carry = (u_nom, jnp.asarray(0.25), jnp.asarray(0), jnp.asarray(0.0),
                 jnp.asarray(False))
        u, trust, it, gv, done = jax.lax.while_loop(cond, body, carry)
        # status: 0 feasible, 1 iteration limit, 2 trust-region collapse
        status = jnp.where(gv >= 0.0, 0, jnp.where(trust < 1e-3, 2, 1))
        return u, gv, it, status, trust

    return run


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
    ap.add_argument("--out", default="results/device_sqp.json")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--threshold-quantile", type=float, default=0.9)
    ap.add_argument("--states", default="0,1,10,28")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)

    art = lib.load_artifacts(args.artifacts, args.q_manifest)
    model, mjm = lib.build_model(args.model)
    step = jax.jit(lib.build_step(model, art))
    ref = np.load(f"{args.artifacts}/torch_ref.npz")
    commands = jnp.array([1.0, 0.0, 0.0], jnp.float64)
    qpos_all = jnp.asarray(ref["qpos_pre"], jnp.float64)
    qvel_all = jnp.asarray(ref["qvel_pre"], jnp.float64)
    phase_all = jnp.asarray(ref["phase"], jnp.float64)
    act_all = jnp.asarray(ref["action"], jnp.float64)

    def obs_vec(qpos, qvel, target, phase, freq):
        return lib.obs_from(qpos, qvel, target, commands, phase, freq, art)

    def h_of(qpos, qvel, target, phase, freq):
        o = obs_vec(qpos, qvel, target, phase, freq)
        return lib.q_value(art, o, lib.policy(art, o))

    q_fn = lambda o: lib.q_value(art, o, lib.policy(art, o))  # noqa: E731

    # h(s_t) must use the observation Q was fitted on: its dof_pos_target
    # channel is the *previous* step's internal target (target_pre[t]), not the
    # action about to be applied.
    target_all = jnp.asarray(ref["target_pre"], jnp.float64)
    h_all = jax.jit(jax.vmap(h_of))(qpos_all, qvel_all, target_all,
                                    phase_all[:, 0], phase_all[:, 1])
    # cross-check against Q evaluated on Q2's own recorded 48-D observation
    q_from_obs = jax.jit(jax.vmap(q_fn))(jnp.asarray(ref["obs"], jnp.float64))
    print("h from reconstructed state vs Q2 obs: max abs",
          float(jnp.abs(h_all - q_from_obs).max()))
    c = float(jnp.quantile(h_all, args.threshold_quantile))
    beta = args.beta

    def g(u, qpos, qvel, phase, freq, h0):
        qpos2, qvel2 = step(qpos, qvel, u)
        target_prev = u * art["scale"]["dof_pos_target"]
        phase2 = jnp.mod(phase + 2.0 * jnp.pi * 0.02 * freq, 2.0 * jnp.pi)
        o2 = obs_vec(qpos2, qvel2, target_prev, phase2, freq)
        return lib.q_value(art, o2, lib.policy(art, o2)) - c - (1.0 - beta) * h0

    g_jac_fwd = jax.jacfwd(g, argnums=0)

    def gjac(u, *state):
        return g(u, *state), g_jac_fwd(u, *state)
    lo = -2.0 * jnp.ones(N, jnp.float64)
    hi = 2.0 * jnp.ones(N, jnp.float64)
    W = jnp.eye(N, dtype=jnp.float64)
    qp_solve = build_qp_solver(lo, hi)
    sqp = jax.jit(build_device_sqp(g, gjac, qp_solve, lo, hi))

    results = {"device": str(jax.devices()[0]), "beta": beta,
               "threshold_c": c,
               "threshold_quantile": args.threshold_quantile,
               "qp": "qpax.solve_qp_elastic, penalty 10.0, slack per row",
               "max_sqp_iterations": 8, "trust0": 0.25, "backtracks": 6,
               "states": {}}
    idx = [int(s) for s in args.states.split(",")]
    timings = []
    for t in idx:
        state = (qpos_all[t], qvel_all[t], phase_all[t, 0], phase_all[t, 1],
                 h_all[t] - c)
        u0 = act_all[t]
        g0 = float(g(u0, *state))
        out = sqp(u0, W, state)
        jax.block_until_ready(out[0])
        u, gv, it, status, trust = out
        per = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = sqp(u0, W, state)
            jax.block_until_ready(out[0])
            per.append((time.perf_counter() - t0) * 1e3)
        timings.extend(per)
        results["states"][t] = {
            "g_nominal": g0, "g_final": float(gv), "iterations": int(it),
            "status": int(status), "trust": float(trust),
            "ms_median": float(np.median(per)),
            "ms_min": float(np.min(per)),
            "action_norm_from_nominal": float(
                jnp.linalg.norm(u - u0)),
            "u": np.asarray(u).tolist(),
        }
        print(t, "g_nom", round(g0, 4), "-> g", round(float(gv), 4),
              "it", int(it), "status", int(status), "ms", round(float(np.median(per)), 2))

    results["warm_ms_median_all"] = float(np.median(timings))
    results["warm_ms_min_all"] = float(np.min(timings))

    if args.profile:
        cuda = ctypes.CDLL("libcuda.so.1")
        t = idx[0]
        state = (qpos_all[t], qvel_all[t], phase_all[t, 0], phase_all[t, 1],
                 h_all[t] - c)
        u0 = act_all[t]
        sqp(u0, W, state)
        jax.block_until_ready(sqp(u0, W, state)[0])
        cuda.cuProfilerStart()
        for _ in range(5):
            out = sqp(u0, W, state)
            jax.block_until_ready(out[0])
        cuda.cuProfilerStop()
        jax.effects_barrier()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
