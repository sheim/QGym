"""Shared 12-variable QP battery definitions (numpy/scipy only).

Used by ``qp_battery.py`` (QPAX / BoxOSQP) and ``cupiqp_battery.py`` (cuPIQP) so
all candidates are compared on identical problems and references.
"""

import numpy as np
from scipy.optimize import minimize

N = 12
BOX_LO = -2.0 * np.ones(N)
BOX_HI = 2.0 * np.ones(N)


def qp_objective(x, W, u_nom):
    d = x - u_nom
    return 0.5 * float(d @ W @ d)


def kkt_report(x, W, u_nom, G, h, tol=1e-7):
    """Original (untransformed) constraint residual and stationarity."""
    x = np.asarray(x, dtype=np.float64)
    g = h - G @ x  # constraints are G x <= h
    active = g < tol
    grad = W @ (x - u_nom)
    # Multipliers from the active set: grad = -G_active^T lam, lam >= 0
    lam_res = None
    if active.sum() > 0:
        Ga = G[active]
        # least-squares multipliers for the active rows
        lam, *_ = np.linalg.lstsq(-Ga.T, grad, rcond=None)
        lam_res = float(np.abs(-Ga.T @ lam - grad).max())
        lam_min = float(lam.min())
    else:
        lam_res = 0.0
        lam_min = 0.0
    return {
        "constraint_violation": float(np.maximum(-g, 0.0).max()),
        "bound_violation": float(
            np.maximum(np.maximum(BOX_LO - x, x - BOX_HI), 0.0).max()
        ),
        "stationarity": float(np.abs(grad - (-G.T @ _lam_or_zero(active, W, x, u_nom, G, h))).max()),
        "active_rows": int(active.sum()),
        "active_dual_nonneg": lam_min >= -1e-9,
        "complementarity": float(np.abs(g * _lam_or_zero(active, W, x, u_nom, G, h)).max()),
    }


def _lam_or_zero(active, W, x, u_nom, G, h):
    grad = W @ (x - u_nom)
    lam = np.zeros(G.shape[0])
    if active.sum() == 0:
        return lam
    Ga = G[active]
    sol, *_ = np.linalg.lstsq(-Ga.T, grad, rcond=None)
    lam[active] = sol
    return lam


def single_constraint_ref(W, u_nom, a, b):
    """Closed form for one inequality a^T u + b >= 0 (i.e. -a^T u <= b)."""
    W = np.asarray(W, dtype=np.float64)
    u_nom = np.asarray(u_nom, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    denom = a @ np.linalg.solve(W, a)
    slack = a @ u_nom + b
    if slack >= 0.0:
        return u_nom.copy()
    lam = -slack / denom
    return u_nom + lam * np.linalg.solve(W, a)


def build_case(name, W, u_nom, halfspaces):
    """halfspaces: list of (a, b) meaning a^T u + b >= 0."""
    G = np.array([-a for a, _ in halfspaces], dtype=np.float64)
    h = np.array([b for _, b in halfspaces], dtype=np.float64)
    return {"name": name, "W": W, "u_nom": u_nom, "G": G, "h": h,
            "halfspaces": halfspaces}


def with_bounds(G, h, box=None):
    lo, hi = (BOX_LO, BOX_HI) if box is None else box
    Gb = np.vstack([G, np.eye(N), -np.eye(N)])
    hb = np.concatenate([h, hi, -lo])
    return Gb, hb


def cases():
    rng = np.random.default_rng(20261115)
    out = []

    # 1. nominal feasible, barrier inactive
    W = np.eye(N)
    u_nom = 0.1 * np.ones(N)
    out.append(build_case("inactive", W, u_nom, [(np.eye(N)[0], 1.0)]))
    out[-1]["ref"] = u_nom.copy()

    # 2. one active barrier, varied weights (analytic reference)
    w = np.arange(1, N + 1, dtype=np.float64)
    W = np.diag(w)
    u_nom = np.array([0.5, -0.4, 0.3, -0.2, 0.1, 0.0, 0.2, -0.3, 0.4, -0.1, 0.05, 0.3])
    a = np.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.0, 0.4, -0.2, 0.1, 0.0, 0.3, -0.4])
    b = 0.5
    out.append(build_case("active_barrier", W, u_nom, [(a, b)]))
    out[-1]["ref"] = single_constraint_ref(W, u_nom, a, b)

    # 3. action bounds active, barrier inactive
    W = np.diag(np.ones(N))
    u_nom = np.array([3.0, -2.5] + [0.1] * (N - 2))
    out.append(build_case("bounds_active", W, u_nom, [(np.eye(N)[0], 5.0)]))
    out[-1]["ref"] = np.clip(u_nom, BOX_LO, BOX_HI)

    # 4. infeasible hard constraints: a^T u >= 1 and a^T u <= 0
    W = np.eye(N)
    u_nom = np.zeros(N)
    a1 = np.ones(N) / np.sqrt(N)
    out.append(build_case("infeasible", W, u_nom, [(a1, -1.0), (-a1, 0.0)]))
    out[-1]["ref"] = None

    # 5. poor scaling: weights spanning 1e-6..1e6, active barrier
    w = np.logspace(-6, 6, N)
    W = np.diag(w)
    a = np.ones(N) / np.sqrt(N)
    b = 0.0
    u_nom = -np.ones(N)
    out.append(build_case("poor_scaling", W, u_nom, [(a, b)]))
    # The analytic solution leaves the default +/-2 box (ill conditioning pushes
    # u0 to ~10), so this case uses a wide box to keep the reference exact.
    out[-1]["box"] = (-1e4 * np.ones(N), 1e4 * np.ones(N))
    out[-1]["ref"] = single_constraint_ref(W, u_nom, a, b)

    # 6. nearly zero constraint gradient with a finite offset
    W = np.eye(N)
    u_nom = np.zeros(N)
    a = 1e-8 * np.ones(N) / np.sqrt(N)
    b = 1e-6
    out.append(build_case("tiny_gradient", W, u_nom, [(a, b)]))
    out[-1]["ref"] = single_constraint_ref(W, u_nom, a, b)

    # 7. multi-constraint (8 half-spaces) + bounds, scipy SLSQP reference
    W = np.diag(np.linspace(0.2, 3.0, N))
    u_nom = 0.6 * rng.normal(size=N)
    hs = []
    for i in range(8):
        ai = rng.normal(size=N)
        ai /= np.linalg.norm(ai)
        hs.append((ai, -0.25))
    out.append(build_case("multi_constraint", W, u_nom, hs))
    Gb, hb = with_bounds(out[-1]["G"], out[-1]["h"])
    res = minimize(
        lambda x: qp_objective(x, W, u_nom),
        np.zeros(N),
        jac=lambda x: W @ (x - u_nom),
        constraints=[{"type": "ineq", "fun": lambda x: hb - Gb @ x}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    out[-1]["ref"] = res.x
    out[-1]["scipy_success"] = bool(res.success)
    return out




def batch_case(rng, B):
    """B independent active-barrier QPs with closed-form solutions."""
    w = np.tile(np.arange(1, N + 1, dtype=np.float64), (B, 1))
    W = np.array([np.diag(wi) for wi in w])
    u_nom = 0.5 * rng.normal(size=(B, N))
    a = rng.normal(size=(B, N))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = 0.1 * np.ones(B)
    ref = np.array([single_constraint_ref(W[i], u_nom[i], a[i], b[i])
                    for i in range(B)])
    return W, u_nom, a, b, ref
