"""Same-process Torch + JAX/MJX + Warp + QPAX compatibility check.

Runs in the isolated ``env_coexist`` environment, which is the locked Q2 GPU
stack (torch 2.13.0+cu130, mujoco 3.11, mujoco-warp 3.11, warp-lang 1.16,
tensordict 0.13) *plus* the JAX CUDA13 stack (jax/jaxlib 0.9.2), QPAX and
JAXopt.  This is a dependency-resolution-visible test: the JAX install upgraded
12 ``nvidia-*`` wheels, which is recorded by comparing freezes.

Run:  env_coexist/bin/python scripts/coexistence.py --out results/coexistence.json
"""

import argparse
import json
import platform
import time

import numpy as np

REQ = {}


def step(name, fn):
    t0 = time.perf_counter()
    try:
        value = fn()
        REQ[name] = {"ok": True, "ms": (time.perf_counter() - t0) * 1e3,
                     "value": value}
    except Exception as exc:  # noqa: BLE001 - a failure here is the result
        REQ[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    print(name, "OK" if REQ[name]["ok"] else "FAIL", REQ[name].get("value", ""),
          REQ[name].get("error", ""))
    return REQ[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/coexistence.json")
    ap.add_argument(
        "--model",
        default="/tmp/opencode/mjx-go2trot-20261115-a/artifacts/variants/"
                "model_env0_diag_nocylbox.mjb")
    args = ap.parse_args()

    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0)}

    def torch_compute():
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        conv = torch.nn.Conv2d(3, 8, 3).cuda()
        y = conv(torch.randn(2, 3, 16, 16, device="cuda"))
        torch.cuda.synchronize()
        return {"matmul_finite": bool(torch.isfinite(c).all()),
                "conv_shape": list(y.shape), "device": str(c.device)}

    step("torch_cuda", torch_compute)

    def jax_import_and_compute():
        import jax
        import jax.numpy as jnp
        x = jnp.ones((256, 256))
        y = (x @ x).sum()
        jax.block_until_ready(y)
        return {"backend": jax.default_backend(), "devices": str(jax.devices()),
                "sum": float(y), "version": jax.__version__}

    step("jax_cuda", jax_import_and_compute)

    def mjx_step():
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import mujoco
        import mujoco.mjx as mjx
        m = mujoco.MjModel.from_binary_path(args.model)
        model = mjx.put_model(m, impl="jax")
        data = mjx.make_data(model)
        data = data.replace(qpos=jnp.asarray(data.qpos, jnp.float64))
        out = jax.jit(mjx.step)(model, data)
        jax.block_until_ready(out.qpos)
        return {"nq": m.nq, "qpos0": float(jnp.abs(out.qpos).sum()),
                "device": str(out.qpos.device)}

    step("mjx_step_after_torch", mjx_step)

    def torch_after_jax():
        import torch
        a = torch.randn(1024, 1024, device="cuda")
        c = a @ a
        torch.cuda.synchronize()
        return {"finite": bool(torch.isfinite(c).all()),
                "free_GB": round(torch.cuda.mem_get_info()[0] / 1e9, 2)}

    step("torch_after_jax", torch_after_jax)

    def warp_compute():
        import warp as wp
        wp.init()

        @wp.kernel
        def incr(x: wp.array(dtype=float)):
            i = wp.tid()
            x[i] = x[i] * 2.0 + 1.0

        a = wp.ones(1024, dtype=float)
        wp.launch(incr, dim=1024, inputs=[a])
        wp.synchronize()
        return {"mean": float(a.numpy().mean()), "device": str(a.device),
                "version": wp.config.version}

    step("warp_cuda", warp_compute)

    def mujoco_warp_import():
        import mujoco_warp  # noqa: F401
        return {"import": "ok"}

    step("mujoco_warp_import", mujoco_warp_import)

    def qpax_solve():
        import jax
        import jax.numpy as jnp
        import qpax
        n = 12
        Q = jnp.eye(n)
        q = -jnp.ones(n)
        G = jnp.concatenate([jnp.eye(n), -jnp.eye(n)])
        h = jnp.concatenate([2 * jnp.ones(n), 2 * jnp.ones(n)])
        x, s, z, y, conv, it = qpax.solve_qp(
            Q, q, jnp.zeros((0, n)), jnp.zeros((0,)), G, h)
        return {"x0": float(x[0]), "conv": int(conv), "iters": int(it),
                "device": str(x.device)}

    step("qpax_after_torch_and_warp", qpax_solve)

    def jaxopt_import():
        from jaxopt import BoxOSQP
        return {"import": BoxOSQP.__name__}

    step("jaxopt_import", jaxopt_import)

    def tensordict_import():
        import tensordict
        return {"version": tensordict.__version__}

    step("tensordict_import", tensordict_import)

    info["checks"] = REQ
    with open(args.out, "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v["ok"] for k, v in REQ.items()}, indent=1))


if __name__ == "__main__":
    main()
