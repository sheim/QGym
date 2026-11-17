"""Small 12-variable QP battery: QPAX vs JAXopt BoxOSQP vs reference.

Every case is a 12-action box + a few linear constraints.  References are
closed-form where possible (one active half-space, active bounds) and scipy
SLSQP for the multi-constraint case -- offline validation only, never part of a
controller.

Run:  env/bin/python scripts/qp_battery.py --out results/qp_battery.json
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import qpax
from jaxopt import BoxOSQP

from qp_cases import (  # noqa: F401
    BOX_HI, BOX_LO, N, batch_case, build_case, cases, kkt_report,
    qp_objective, single_constraint_ref, with_bounds)


def run_qpax(case, dtype, max_iter=30, solver_tol=None, linear_solver=None):
    W = jnp.asarray(case["W"], dtype)
    u_nom = jnp.asarray(case["u_nom"], dtype)
    G, h = with_bounds(case["G"], case["h"], case.get("box"))
    Gj, hj = jnp.asarray(G, dtype), jnp.asarray(h, dtype)
    Q = W
    q = -W @ u_nom
    kw = {"max_iter": max_iter}
    if solver_tol is not None:
        kw["solver_tol"] = solver_tol
    if linear_solver is not None:
        kw["linear_solver"] = linear_solver

    def solve(Q, q, Gj, hj):
        return qpax.solve_qp(
            Q, q, jnp.zeros((0, N), dtype), jnp.zeros((0,), dtype), Gj, hj, **kw
        )

    solve_jit = jax.jit(solve)
    x, s, z, y, conv, iters = solve_jit(Q, q, Gj, hj)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    rep = kkt_report(x, case["W"], case["u_nom"], G, h)
    rep["dual_residual"] = float(np.abs(
        case["W"] @ x - case["W"] @ case["u_nom"] + G.T @ z).max())
    rep.update({"solver": "qpax", "dtype": np.dtype(dtype).name,
                "converged_flag": int(conv), "pdip_iter": int(iters),
                "x": x.tolist()})
    if case["ref"] is not None:
        ref = np.asarray(case["ref"], dtype=np.float64)
        rep["ref_abs_err"] = float(np.abs(x - ref).max())
        rep["ref_rel_err"] = float(
            np.abs(x - ref).max() / max(1e-12, np.abs(ref).max()))
        rep["obj"] = qp_objective(x, case["W"], case["u_nom"])
        rep["ref_obj"] = qp_objective(ref, case["W"], case["u_nom"])
        rep["ref_obj_gap"] = rep["obj"] - rep["ref_obj"]
    return rep


def run_qpax_elastic(case, dtype, penalty=1e2):
    W = jnp.asarray(case["W"], dtype)
    u_nom = jnp.asarray(case["u_nom"], dtype)
    G, h = with_bounds(case["G"], case["h"], case.get("box"))
    Q = W
    q = -W @ u_nom

    def solve(Q, q, Gj, hj):
        return qpax.solve_qp_elastic(Q, q, Gj, hj, penalty)

    x, t, s1, s2, z1, z2, conv, iters = jax.jit(solve)(
        Q, q, jnp.asarray(G, dtype), jnp.asarray(h, dtype)
    )
    x = np.asarray(x, np.float64)
    slack = np.maximum(G @ x - h, 0.0)
    rep = kkt_report(x, case["W"], case["u_nom"], G, h)
    rep.update({"solver": "qpax_elastic", "dtype": np.dtype(dtype).name,
                "converged_flag": int(conv), "pdip_iter": int(iters),
                "slack_total": float(slack.sum()),
                "slack_max": float(slack.max()),
                "x": x.tolist()})
    return rep


def run_boxosqp(case, dtype, maxiter=4000, tol=1e-9, jit=True):
    W = jnp.asarray(case["W"], dtype)
    u_nom = jnp.asarray(case["u_nom"], dtype)
    G, h = with_bounds(case["G"], case["h"], case.get("box"))
    Q = W
    q = -W @ u_nom
    Aj = jnp.asarray(G, dtype)
    l = -jnp.inf * jnp.ones(G.shape[0], dtype)
    u = jnp.asarray(h, dtype)
    solver = BoxOSQP(jit=jit, maxiter=maxiter, tol=tol)
    # jit=True: the iteration loop is a device-side while_loop; no host reads
    # inside iterations.  An outer jax.jit() around run() breaks jaxopt's
    # internal custom_vjp, so the call is left un-jitted.
    sol = solver.run(None, (Q, q), Aj, (l, u))
    x = np.asarray(sol.params[0][0], np.float64)
    y = np.asarray(sol.params[1], np.float64)
    rep = kkt_report(x, case["W"], case["u_nom"], G, h)
    rep["dual_residual"] = float(np.abs(
        case["W"] @ x - case["W"] @ case["u_nom"] + G.T @ y).max())
    rep.update({"solver": "boxosqp", "dtype": np.dtype(dtype).name,
                "converged_flag": int(sol.state.status),
                "pdip_iter": int(sol.state.iter_num),
                "solver_error": float(sol.state.error),
                "x": x.tolist()})
    if case["ref"] is not None:
        ref = np.asarray(case["ref"], dtype=np.float64)
        rep["ref_abs_err"] = float(np.abs(x - ref).max())
        rep["ref_rel_err"] = float(
            np.abs(x - ref).max() / max(1e-12, np.abs(ref).max()))
        rep["obj"] = qp_objective(x, case["W"], case["u_nom"])
        rep["ref_obj"] = qp_objective(ref, case["W"], case["u_nom"])
        rep["ref_obj_gap"] = rep["obj"] - rep["ref_obj"]
    return rep


def timing(fn, iters=50):
    fn()
    jax.block_until_ready(fn())
    starts = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - starts) / iters * 1e3


def batch_case(rng, B):
    """B independent active-barrier QPs with analytic solutions."""
    w = np.tile(np.arange(1, N + 1, dtype=np.float64), (B, 1))
    W = np.array([np.diag(wi) for wi in w])
    u_nom = 0.5 * rng.normal(size=(B, N))
    a = rng.normal(size=(B, N))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = 0.1 * np.ones(B)
    ref = np.array([single_constraint_ref(W[i], u_nom[i], a[i], b[i]) for i in range(B)])
    return W, u_nom, a, b, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/qp_battery.json")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)

    all_cases = cases()
    results = {"device": str(jax.devices()[0]), "backend": jax.default_backend(),
               "n": N, "cases": {}, "timing_ms": {}, "batch": {}}

    for case in all_cases:
        entry = {}
        for dtype, tag in ((jnp.float64, "f64"), (jnp.float32, "f32")):
            tol = 1e-9 if tag == "f64" else 1e-4
            try:
                entry[f"qpax_{tag}"] = run_qpax(case, dtype, max_iter=60)
            except Exception as exc:  # noqa: BLE001 - record failures verbatim
                entry[f"qpax_{tag}"] = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                entry[f"boxosqp_{tag}"] = run_boxosqp(
                    case, dtype, tol=tol, maxiter=20000)
            except Exception as exc:  # noqa: BLE001
                entry[f"boxosqp_{tag}"] = {"error": f"{type(exc).__name__}: {exc}"}
        entry["qpax_elastic_f64"] = run_qpax_elastic(case, jnp.float64)
        entry["qpax_f64_iterlimit"] = run_qpax(case, jnp.float64, max_iter=2)
        if case["name"] == "multi_constraint":
            entry["qpax_f64_qr"] = run_qpax(
                case, jnp.float64, max_iter=60, solver_tol=1e-9,
                linear_solver=qpax.LinearSolver.QR,
            )
            entry["qpax_f64_tight"] = run_qpax(
                case, jnp.float64, max_iter=60, solver_tol=1e-9)
            entry["qpax_f64_iterlimit"] = run_qpax(case, jnp.float64, max_iter=2)
            entry["scipy_success"] = case["scipy_success"]
        results["cases"][case["name"]] = entry

    # ---- timing, single QP, float64 and float32 -------------------------
    case = all_cases[6]  # multi_constraint
    for dtype, tag in ((jnp.float64, "f64"), (jnp.float32, "f32")):
        W = jnp.asarray(case["W"], dtype)
        u_nom = jnp.asarray(case["u_nom"], dtype)
        G, h = with_bounds(case["G"], case["h"])
        Q = W
        q = -W @ u_nom
        Gj, hj = jnp.asarray(G, dtype), jnp.asarray(h, dtype)
        Aj = jnp.zeros((0, N), dtype)
        bj = jnp.zeros((0,), dtype)

        for tol_tag, solver_tol, max_it in (("tol1e5", 1e-5, 30),
                                            ("tol1e9", 1e-9, 60)):
            f = jax.jit(lambda Q, q, G, h: qpax.solve_qp(
                Q, q, Aj, bj, G, h, solver_tol=solver_tol, max_iter=max_it))
            st = time.perf_counter()
            out = f(Q, q, Gj, hj)
            jax.block_until_ready(out)
            compile_ms = (time.perf_counter() - st) * 1e3
            results["timing_ms"][f"qpax_{tag}_{tol_tag}"] = {
                "compile": compile_ms,
                "warm_solve": timing(lambda: f(Q, q, Gj, hj)),
                "iterations": int(out[5]),
                "converged": int(out[4]),
                "primal_dual_residual": float(np.abs(np.asarray(
                    jnp.asarray(case["W"], np.float64) @ out[0]
                    - jnp.asarray(case["W"], np.float64) @ jnp.asarray(case["u_nom"], np.float64)
                    + jnp.asarray(G, np.float64).T @ out[2])).max()),
            }
        oq = BoxOSQP(jit=True, maxiter=4000,
                     tol=1e-5 if tag == "f64" else 1e-4,
                     eq_qp_solve="lu")
        l = -jnp.inf * jnp.ones(G.shape[0], dtype)
        u = hj
        g = lambda Q, q, A, l, u: oq.run(None, (Q, q), A, (l, u))
        try:
            st = time.perf_counter()
            out = g(Q, q, Gj, l, u)
            jax.block_until_ready(out.params)
            compile_ms = (time.perf_counter() - st) * 1e3
            results["timing_ms"][f"boxosqp_{tag}"] = {
                "compile": compile_ms,
                "warm_solve": timing(lambda: g(Q, q, Gj, l, u)),
                "iterations": int(out.state.iter_num),
            }
        except Exception as exc:  # noqa: BLE001
            results["timing_ms"][f"boxosqp_{tag}"] = {
                "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    # ---- batched solve, analytic reference ------------------------------
    rng = np.random.default_rng(7)
    B = args.batch
    Wb, u_nomb, ab, bb, refb = batch_case(rng, B)
    for dtype, tag in ((jnp.float64, "f64"), (jnp.float32, "f32")):
        Wj = jnp.asarray(Wb, dtype)
        ub = jnp.asarray(u_nomb, dtype)
        aj = jnp.asarray(ab, dtype)
        bj = jnp.asarray(bb, dtype)
        Gb = -aj[:, None, :]
        hb = bj[:, None]
        eye = jnp.eye(N, dtype=dtype)
        Gfull = jnp.concatenate([Gb, jnp.broadcast_to(eye, (B, N, N)),
                                 -jnp.broadcast_to(eye, (B, N, N))], axis=1)
        hfull = jnp.concatenate([hb, jnp.broadcast_to(jnp.asarray(BOX_HI, dtype), (B, N)),
                                 jnp.broadcast_to(jnp.asarray(-BOX_LO, dtype), (B, N))], axis=1)
        qb = -jnp.einsum("bij,bj->bi", Wj, ub)
        A0 = jnp.zeros((B, 0, N), dtype)
        b0 = jnp.zeros((B, 0), dtype)
        f = jax.jit(jax.vmap(lambda Q, q, G, h: qpax.solve_qp(
            Q, q, A0[0], b0[0], G, h, solver_tol=1e-5, max_iter=60)))
        st = time.perf_counter()
        out = f(Wj, qb, Gfull, hfull)
        jax.block_until_ready(out[0])
        compile_ms = (time.perf_counter() - st) * 1e3
        x = np.asarray(out[0], np.float64)
        warm = timing(lambda: f(Wj, qb, Gfull, hfull), iters=20)
        results["batch"][f"qpax_{tag}"] = {
            "B": B, "compile_ms": compile_ms, "ms_total": warm,
            "us_per_solve": warm * 1e3 / B,
            "solves_per_s": B / (warm * 1e-3),
            "max_ref_abs_err": float(np.abs(x - refb).max()),
            "max_constraint_violation": float(np.maximum(
                np.einsum("bpn,bn->bp", np.asarray(Gfull, np.float64), x)
                - np.asarray(hfull, np.float64), 0.0).max()),
            "converged_fraction": float(np.mean(np.asarray(out[4]) == 1)),
            "iterations_max": int(np.max(np.asarray(out[5]))),
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "cases"}, indent=1))
    for name, entry in results["cases"].items():
        row = {k: (v.get("converged_flag"), v.get("pdip_iter"),
                   round(v.get("constraint_violation", float("nan")), 10),
                   round(v.get("ref_abs_err", float("nan")), 10),
                   v.get("error", "")[:60])
               for k, v in entry.items() if isinstance(v, dict)}
        print(name, json.dumps(row))


if __name__ == "__main__":
    main()
