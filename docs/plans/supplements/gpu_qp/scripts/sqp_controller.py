"""SQP and linearized-QP controllers on the real composite barrier.

Problem (policy units, 12 actions):
    minimise   0.5 (u - u_nom)^T W (u - u_nom)
    subject to g(u) = h(F(s,u)) - (1-beta) h(s) >= 0      (nonlinear, MJX)
               u_min <= u <= u_max                        (box)
plus an explicit L-inf trust region per SQP iteration.

Variants compared on the same state:
  nominal        u = u_nom (the policy-mean action)
  linqp          one QP on the linearisation at u_nom (no re-check)
  sqp            repeated QPs at u_k with true-g acceptance and trust region

Also runs a synthetic known-solution nonlinear case (projection onto a ball)
where the linearised QP proposal provably violates the true constraint.

Run:
  env/bin/python scripts/sqp_controller.py --out results/sqp_controller.json
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import qpax

import cbf_composite_lib as lib

N = 12
U_LO = -2.0
U_HI = 2.0


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def solve_qp_box(W, u_nom, a, b, lo, hi, u_center=None, trust=None):
    """QP: 0.5 u^T W u - (W u_nom)^T u  s.t. -a^T u <= b, box, trust.

    a, b encode the linearised barrier a^T u + b >= 0.
    """
    Q = W
    q = -W @ u_nom
    G = [(-a)[None, :]]
    h = [jnp.asarray([b])]
    if trust is not None:
        # |u - u_center|_inf <= trust  ->  u <= c+t, -u <= -c+t
        c = u_center
        G.append(jnp.eye(N))
        h.append(c + trust)
        G.append(-jnp.eye(N))
        h.append(-c + trust)
    G.append(jnp.eye(N))
    h.append(jnp.full((N,), hi))
    G.append(-jnp.eye(N))
    h.append(jnp.full((N,), -lo))
    G = jnp.concatenate(G, axis=0)
    h = jnp.concatenate(h)
    # Elastic (always-feasible) relaxation: per-row slack with a linear penalty,
    # so an infeasible hard constraint is reported as slack, not as a NaN.
    x, t, s1, s2, z1, z2, conv, it = qpax.solve_qp_elastic(
        Q, q, G, h, penalty=10.0)
    slack = jnp.maximum(G @ x - h, 0.0)
    return x, int(it), int(conv), float(slack.max()), float(slack.sum())


def sqp(gval_jac, u0, u_nom, W, lo, hi, max_iter=8, trust0=0.25,
        trust_min=1e-3, accept_tol=1e-9):
    """Trust-region SQP with true-constraint acceptance."""
    out = []
    u = u0
    trust = trust0
    for k in range(max_iter):
        g, jac = gval_jac(u)
        g, jac = float(g), np.asarray(jac, dtype=np.float64)
        row = {"iter": k, "g": g, "trust": trust,
               "grad_norm": float(np.linalg.norm(jac))}
        if g >= 0.0 and k > 0:
            row.update({"action": "stop_feasible"})
            out.append(row)
            break
        # linearisation: g(u_k) + jac^T (u - u_k) >= 0
        b = g - jac @ np.asarray(u)
        x, it, conv, slack_max, slack_sum = solve_qp_box(
            W, u_nom, jnp.asarray(jac), jnp.asarray(float(b)),
            jnp.asarray(lo), jnp.asarray(hi), u_center=jnp.asarray(u),
            trust=jnp.asarray(trust))
        d = np.asarray(x) - np.asarray(u)
        row.update({"qp_iter": it, "qp_converged": conv,
                    "qp_slack_max": slack_max, "qp_slack_sum": slack_sum,
                    "step_inf": float(np.abs(d).max()),
                    "qp_terms": None})
        # true-constraint acceptance with backtracking on the step
        alpha, accepted = 1.0, False
        for _ in range(12):
            g_try, _ = gval_jac(jnp.asarray(u + alpha * d))
            if float(g_try) > g + accept_tol:
                accepted = True
                break
            alpha *= 0.5
        row.update({"alpha": alpha, "accepted": accepted,
                    "g_trial": float(g_try)})
        out.append(row)
        if accepted:
            u = jnp.asarray(u + alpha * d)
            trust = min(2.0 * trust, 1.0)
        else:
            trust *= 0.5
            if trust < trust_min:
                out[-1]["action"] = "stop_trust_min"
                break
    return u, out


def synthetic_case(art=None):
    """Known-solution nonlinear case: min 0.5||u-u_nom||^2 s.t. ||u||^2 <= r^2."""
    rng = np.random.default_rng(3)
    u_nom = 1.6 * rng.normal(size=N)
    r = 0.8 * np.linalg.norm(u_nom)
    u_ref = r * u_nom / np.linalg.norm(u_nom)
    W = jnp.eye(N, dtype=jnp.float64)

    def gfun(u):
        return float(r**2 - u @ u)

    def gval_jac(u):
        uj = jnp.asarray(u, jnp.float64)
        return r**2 - uj @ uj, -2.0 * uj

    lo = -100.0 * np.ones(N)
    hi = 100.0 * np.ones(N)
    u0 = jnp.asarray(u_nom)
    g0, jac0 = gval_jac(u0)
    b = float(g0) - float(jac0 @ np.asarray(u0))
    x_lin, it, conv, smax, ssum = solve_qp_box(
        W, jnp.asarray(u_nom), jac0, jnp.asarray(b), jnp.asarray(lo),
        jnp.asarray(hi))
    x_sqp, log = sqp(gval_jac, u0, jnp.asarray(u_nom), W, lo, hi)
    x_sqp_long, log_long = sqp(gval_jac, u0, jnp.asarray(u_nom), W, lo, hi,
                               max_iter=40, trust0=1.0)
    return {
        "u_nom_norm": float(np.linalg.norm(u_nom)), "r": float(r),
        "ref_norm": float(np.linalg.norm(u_ref)),
        "g_nominal": g0,
        "linqp": {"g_true": gfun(np.asarray(x_lin)),
                  "g_linearised": float(g0 + jac0 @ (np.asarray(x_lin) - np.asarray(u0))),
                  "dist_to_ref": float(np.linalg.norm(np.asarray(x_lin) - u_ref)),
                  "qp_iter": it, "slack_max": smax},
        "sqp": {"g_true": gfun(np.asarray(x_sqp)),
                "dist_to_ref": float(np.linalg.norm(np.asarray(x_sqp) - u_ref)),
                "iterations": len(log), "log": log},
        "sqp_long": {"g_true": gfun(np.asarray(x_sqp_long)),
                     "dist_to_ref": float(np.linalg.norm(
                         np.asarray(x_sqp_long) - u_ref)),
                     "iterations": len(log_long),
                     "last_steps": [r.get("step_inf") for r in log_long][-5:]},
    }


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
    ap.add_argument("--out", default="results/sqp_controller.json")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--threshold-quantile", type=float, default=0.5)
    ap.add_argument("--weight-mode", default="identity")
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
    print("h stats", float(h_all.min()), float(h_all.max()), "c", c)

    def make_problem(beta):
        def succ(u, qpos, qvel, phase, freq):
            qpos2, qvel2 = step(qpos, qvel, u)
            target_prev = u * art["scale"]["dof_pos_target"]
            phase2 = jnp.mod(phase + 2.0 * jnp.pi * 0.02 * freq, 2.0 * jnp.pi)
            o2 = obs_vec(qpos2, qvel2, target_prev, phase2, freq)
            return lib.q_value(art, o2, lib.policy(art, o2)) - c

        def g(u, qpos, qvel, phase, freq, h0):
            return succ(u, qpos, qvel, phase, freq) - (1.0 - beta) * h0

        return g

    g = make_problem(args.beta)
    g_jac = jax.jit(jax.jacfwd(g, argnums=0))
    g_val = jax.jit(g)
    g_and_jac = jax.jit(lambda u, q, v, p, f, h0: (
        g(u, q, v, p, f, h0), jax.jacfwd(g, argnums=0)(u, q, v, p, f, h0)))

    # nominal feasibility of the decrease condition over the whole rollout
    g_scan = jax.jit(jax.vmap(g))(
        act_all, qpos_all, qvel_all, phase_all[:, 0], phase_all[:, 1],
        h_all - c)
    g_scan = np.asarray(g_scan)
    worst = np.argsort(g_scan)[:3]
    results_prefix = {
        "g_nominal_scan": {
            "mean": float(g_scan.mean()), "min": float(g_scan.min()),
            "max": float(g_scan.max()),
            "fraction_satisfied": float((g_scan >= 0).mean()),
            "worst_states": [int(i) for i in worst],
            "worst_values": [float(g_scan[i]) for i in worst],
            "values": g_scan.tolist(),
        }
    }
    print("scan0", float(g_scan[0]), "h0_0", float(h_all[0]-c))
    print("nominal g scan", {k: v for k, v in results_prefix["g_nominal_scan"].items()
                             if k not in ("values", "worst_states")},
          "worst", worst.tolist())

    states = {}
    for name, t in (("free_flight", 0), ("contact_onset", 4),
                    ("sustained_contact", 10), ("late_rollout", 99),
                    ("worst_state_0", int(worst[0])),
                    ("worst_state_1", int(worst[1])),
                    ("worst_state_2", int(worst[2]))):
        states[name] = dict(qpos=qpos_all[t], qvel=qvel_all[t],
                            phase=phase_all[t, 0], freq=phase_all[t, 1],
                            h0=h_all[t] - c, u_nom=act_all[t], contacts=None)
    for name in ("stand", "sat1", "satfull"):
        states[name] = dict(
            qpos=jnp.asarray(ref[f"{name}_qpos"], jnp.float64),
            qvel=jnp.asarray(ref[f"{name}_qvel"], jnp.float64),
            phase=jnp.asarray(float(ref["phase"][0, 0]), jnp.float64),
            freq=jnp.asarray(float(ref["phase"][0, 1]), jnp.float64),
            h0=None, u_nom=jnp.asarray(ref[f"{name}_action"], jnp.float64),
            obs=jnp.asarray(ref[f"{name}_obs"], jnp.float64))

    W = jnp.eye(N, dtype=jnp.float64)
    lo = U_LO * jnp.ones(N, jnp.float64)
    hi = U_HI * jnp.ones(N, jnp.float64)
    results = {"device": str(jax.devices()[0]), "beta": args.beta,
               "threshold_c": c, "threshold_quantile": args.threshold_quantile,
               "W": "identity", "bounds": [U_LO, U_HI],
               "qp_relaxation": "elastic QP, linear slack penalty 10.0 per row",
               "beta_note": "g(u) = h(F(s,u)) - (1-beta) h(s)",
               "weight_note": "W = I in policy units; nominal objective "
                              "0.5 (u-u_nom)^T (u-u_nom)",
               "states": {}}
    results.update(results_prefix)

    for name, st in states.items():
        h0 = st["h0"]
        if h0 is None:
            o = st["obs"]
            h0 = lib.q_value(art, o, lib.policy(art, o)) - c
        rec = {"h0": float(h0), "u_nom": np.asarray(st["u_nom"]).tolist()}
        args_ = (st["qpos"], st["qvel"], st["phase"], st["freq"], h0)
        u_nom = st["u_nom"]
        t0 = time.perf_counter()
        g_nom = float(g_val(u_nom, *args_))
        jac_nom = np.asarray(g_jac(u_nom, *args_))
        t_eval = (time.perf_counter() - t0) * 1e3
        rec.update({"g_nominal": g_nom, "grad_norm": float(np.linalg.norm(jac_nom)),
                    "g_jac_eval_ms": t_eval,
                    "nominal_feasible": bool(g_nom >= 0.0)})
        # single linearised QP at u_nom
        b = g_nom - jac_nom @ np.asarray(u_nom)
        t0 = time.perf_counter()
        x_lin, it, conv, smax, ssum = solve_qp_box(
            W, u_nom, jnp.asarray(jac_nom), jnp.asarray(b), lo, hi)
        t_qp = (time.perf_counter() - t0) * 1e3
        g_lin_true = float(g_val(jnp.asarray(x_lin), *args_))
        jac_lin = np.asarray(g_jac(jnp.asarray(x_lin), *args_))
        rec["linqp"] = {
            "g_linearised": float(g_nom + jac_nom @ (np.asarray(x_lin) - np.asarray(u_nom))),
            "g_true": g_lin_true,
            "violation": max(0.0, -g_lin_true),
            "qp_iter": it, "qp_converged": conv, "qp_slack_max": smax,
            "qp_slack_sum": ssum, "qp_ms": t_qp,
            "dist_from_nominal": float(np.linalg.norm(np.asarray(x_lin) - np.asarray(u_nom))),
            "grad_norm_at_proposal": float(np.linalg.norm(jac_lin)),
        }
        # SQP
        def gp(u):
            return g_and_jac(u, *args_)

        t0 = time.perf_counter()
        u_sqp, log = sqp(gp, u_nom, u_nom, W, lo, hi)
        t_sqp = (time.perf_counter() - t0) * 1e3
        g_sqp = float(g_val(u_sqp, *args_))
        rec["sqp"] = {
            "g_true": g_sqp, "violation": max(0.0, -g_sqp),
            "iterations": len(log), "wall_ms": t_sqp, "log": log,
            "dist_from_nominal": float(np.linalg.norm(np.asarray(u_sqp) - np.asarray(u_nom))),
            "u": np.asarray(u_sqp).tolist(),
        }
        results["states"][name] = rec
        print(name, "g_nom", round(g_nom, 5), "lin_qp g_true",
              round(g_lin_true, 5), "sqp g", round(g_sqp, 5),
              "iters", len(log), "ms", round(t_sqp, 1))

    # cold vs warm timing of the composite boundary
    st = states["sustained_contact"]
    a0 = (st["qpos"], st["qvel"], st["phase"], st["freq"], st["h0"])
    u = st["u_nom"]
    t0 = time.perf_counter()
    out = g_and_jac(u, *a0)
    jax.block_until_ready(out[1])
    results["timing"] = {"first_call_ms": (time.perf_counter() - t0) * 1e3}
    t0 = time.perf_counter()
    for _ in range(20):
        out = g_and_jac(u, *a0)
    jax.block_until_ready(out[1])
    results["timing"]["warm_g_plus_jac_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    for _ in range(20):
        out = g_val(u, *a0)
    jax.block_until_ready(out)
    results["timing"]["warm_g_only_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    results["timing"]["jacobian_forward_mode"] = "jacfwd (12 JVPs); reverse mode " \
        "through MJX raises on the solver while_loop"

    # second-order check: exact Hessian of the composite is not available
    try:
        h = jax.hessian(g)(u, *a0)
        jax.block_until_ready(h)
        results["hessian"] = {"ok": True, "max_abs": float(jnp.abs(h).max())}
    except Exception as exc:  # noqa: BLE001
        results["hessian"] = {"ok": False,
                              "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    results["synthetic_ball"] = synthetic_case()
    print("synthetic", json.dumps({k: v for k, v in results["synthetic_ball"].items()
                                   if k != "sqp"}, default=_json_default))
    with open(args.out, "w") as f:
        for k, v in results.items():
            try:
                json.dumps(v, default=_json_default)
            except TypeError as exc:
                print("NOT JSON:", k, exc)
                results[k] = str(v)
        json.dump(results, f, indent=1, default=_json_default)


if __name__ == "__main__":
    main()
