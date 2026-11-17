"""cuPIQP (Warp-native GPU QP) on the same 12-variable battery as QPAX.

cuPIQP is GPU-only: every input must already be a GPU array, and results stay in
Warp arrays.  This script runs in the isolated ``env_cupiqp`` environment
(warp-lang 1.19, cupiqp 0.1.0) and reuses the case definitions of
``qp_battery.py`` by import.

Run (in env_cupiqp):
  python scripts/cupiqp_battery.py --out results/cupiqp_battery.json
"""

import argparse
import json
import time

import numpy as np
import warp as wp

from cupiqp import DenseSolver, Status

import qp_cases as qb


def to_wp(a, dtype):
    return wp.from_numpy(np.ascontiguousarray(a, dtype=np.float64),
                         dtype=wp.float64 if dtype == "f64" else wp.float32)


def solve_case(case, dtype="f64", max_iter=250):
    W, u_nom = case["W"], case["u_nom"]
    G, h = qb.with_bounds(case["G"], case["h"], case.get("box"))
    n = W.shape[0]
    P = to_wp(W, dtype)
    c = to_wp(-W @ u_nom, dtype)
    Gw = to_wp(G, dtype)
    h_lo = to_wp(-np.inf * np.ones(G.shape[0]), dtype)
    h_up = to_wp(h, dtype)
    x_lo = to_wp(-np.inf * np.ones(n), dtype)
    x_up = to_wp(np.inf * np.ones(n), dtype)
    solver = DenseSolver(dtype=wp.float64 if dtype == "f64" else wp.float32)
    solver.settings.max_iter = max_iter
    solver.setup(P=P, c=c, G=Gw, h_l=h_lo, h_u=h_up, x_l=x_lo, x_u=x_up)
    solver.solve()
    solver.result.x.numpy()  # host copy after the solve
    x = np.asarray(solver.result.x.numpy()[0], dtype=np.float64)
    info = solver.result.info
    status = int(info.status.numpy()[0])
    iters = int(info.iter.numpy()[0])
    rep = qb.kkt_report(x, case["W"], case["u_nom"], G, h)
    rep.update({
        "solver": "cupiqp-dense", "dtype": dtype,
        "status": Status(status).name, "iterations": iters,
        "primal_res": float(info.primal_res.numpy()[0]),
        "dual_res": float(info.dual_res.numpy()[0]),
        "duality_gap": float(info.duality_gap.numpy()[0]),
        "x": x.tolist(),
    })
    if case.get("ref") is not None:
        ref = np.asarray(case["ref"], dtype=np.float64)
        rep["ref_abs_err"] = float(np.abs(x - ref).max())
        rep["ref_obj_gap"] = float(
            qb.qp_objective(x, case["W"], case["u_nom"])
            - qb.qp_objective(ref, case["W"], case["u_nom"]))
    return rep, solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cupiqp_battery.json")
    args = ap.parse_args()
    wp.init()
    print("warp device:", wp.get_device())

    cases = qb.cases()
    out = {"warp": str(wp.get_device()), "cases": {}, "timing": {}, "batch": {}}
    for case in cases:
        entry = {}
        for dtype in ("f64", "f32"):
            try:
                rep, _ = solve_case(case, dtype)
            except Exception as exc:  # noqa: BLE001
                rep = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            entry[dtype] = rep
        entry["iterlimit_f64"] = solve_case(case, "f64", max_iter=3)[0]
        out["cases"][case["name"]] = entry
        print(case["name"],
              {k: (v.get("status"), v.get("iterations"),
                   None if "ref_abs_err" not in v else round(v["ref_abs_err"], 10),
                   v.get("error", "")[:50])
               for k, v in entry.items() if isinstance(v, dict)})

    # timing: warm solve of the multi-constraint case
    case = [c for c in cases if c["name"] == "multi_constraint"][0]
    for dtype in ("f64", "f32"):
        rep, solver = solve_case(case, dtype)
        for _ in range(5):
            solver.solve()
        wp.synchronize()
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            solver.solve()
        wp.synchronize()
        warm = (time.perf_counter() - t0) / n * 1e3
        # solve() alone, without the host synchronise (asynchronous queue)
        t0 = time.perf_counter()
        for _ in range(n):
            solver.solve()
        queue_ms = (time.perf_counter() - t0) / n * 1e3
        wp.synchronize()
        out["timing"][dtype] = {"warm_sync_ms": warm, "queue_ms": queue_ms,
                                "iterations": rep["iterations"]}

    # batch: 256 instances of the active-barrier case with analytic solutions
    rng = np.random.default_rng(7)
    B = 256
    Wb, u_nomb, ab, bb, refb = qb.batch_case(rng, B)
    n = qb.N
    for dtype in ("f64", "f32"):
        wpf = wp.float64 if dtype == "f64" else wp.float32
        eye = np.eye(n)
        Gb = np.concatenate([-ab[:, None, :],
                             np.broadcast_to(eye, (B, n, n)),
                             np.broadcast_to(-eye, (B, n, n))], axis=1)
        hb = np.concatenate([bb[:, None],
                             np.broadcast_to(qb.BOX_HI, (B, n)),
                             np.broadcast_to(-qb.BOX_LO, (B, n))], axis=1)
        P = wp.from_numpy(Wb, dtype=wpf)
        c = wp.from_numpy(-np.einsum("bij,bj->bi", Wb, u_nomb), dtype=wpf)
        Gw = wp.from_numpy(np.ascontiguousarray(Gb), dtype=wpf)
        hl = wp.from_numpy(np.full((B, Gb.shape[1]), -np.inf), dtype=wpf)
        hu = wp.from_numpy(np.ascontiguousarray(hb), dtype=wpf)
        solver = DenseSolver(dtype=wpf)
        solver.setup(P=P, c=c, G=Gw, h_l=hl, h_u=hu)
        solver.solve()
        x = np.asarray(solver.result.x.numpy(), dtype=np.float64)
        st = solver.result.info.status.numpy()
        it = solver.result.info.iter.numpy()
        for _ in range(3):
            solver.solve()
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            solver.solve()
        wp.synchronize()
        total = (time.perf_counter() - t0) / 10 * 1e3
        out["batch"][dtype] = {
            "B": B, "ms_total": total, "us_per_solve": total * 1e3 / B,
            "solves_per_s": B / (total * 1e-3),
            "all_solved": bool(np.all(st == Status.CUPIQP_SOLVED)),
            "max_ref_abs_err": float(np.abs(x - refb).max()),
            "iterations_max": int(it.max()),
            "max_constraint_violation": float(np.maximum(
                np.einsum("bpn,bn->bp", Gb, x) - hb, 0).max()),
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
