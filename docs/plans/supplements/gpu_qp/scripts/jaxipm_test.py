"""jaxipm (GPU-batched NLP/IPOPT) on the same 12-variable nonlinear problem.

Isolated Python 3.12 environment (jaxipm requires >= 3.12).  jaxipm derives its
sparse Jacobian/Hessian by symbolic differentiation of the callables
(jax2sympy) or accepts overrides; it is not a plain "supply one gradient" API.

  env_jaxipm/bin/python scripts/jaxipm_test.py --out results/jaxipm_test.json
"""
import argparse, json, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxipm import default_params
from jaxipm.initialization import initialize_common_problem, initialize_problem_regular
from jaxipm.solver import solve

N = 12
rng = np.random.default_rng(3)
u_nom = 1.6 * rng.normal(size=N)
r = 0.8 * np.linalg.norm(u_nom)
u_ref = r * u_nom / np.linalg.norm(u_nom)

def f(x):
    d = jnp.reshape(x, -1)[:N] - u_nom
    return 0.5 * d @ d

def c(x):
    return None

def d(x):
    u = jnp.reshape(x, -1)[:N]
    return jnp.array([u @ u - r**2])

x_L = jnp.full(N, -100.0)
x_U = jnp.full(N, 100.0)
d_L = jnp.array([-jnp.inf])
d_U = jnp.array([0.0])
x0 = jnp.zeros(N)

p = default_params()
cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p)
state = initialize_problem_regular(cp, x0)
t0 = time.perf_counter()
state, term = solve(cp, state)
x = np.asarray(state.it.x[:N, 0])
elapsed = (time.perf_counter() - t0) * 1e3
out = {
    "device": str(jax.devices()[0]),
    "x": x.tolist(),
    "norm": float(np.linalg.norm(x)),
    "r": float(r),
    "constraint": float(x @ x - r**2),
    "dist_to_ref": float(np.linalg.norm(x - u_ref)),
    "first_solve_ms": elapsed,
    "term": str(term)[:200],
}
# warm repeats
times = []
for _ in range(3):
    t0 = time.perf_counter()
    state2 = initialize_problem_regular(cp, x0)
    state2, _ = solve(cp, state2)
    jax.block_until_ready(state2.it.x)
    times.append((time.perf_counter() - t0) * 1e3)
out["warm_solve_ms"] = times
out["x_warm"] = np.asarray(state2.it.x[:N, 0]).tolist()
out["warm_iterations"] = int(np.asarray(state2.it.it).max()) if hasattr(state2.it, "it") else None
print(json.dumps(out, indent=1))
json.dump(out, open("results/jaxipm_test.json", "w"), indent=1)
