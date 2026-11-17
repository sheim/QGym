"""Export the policy actor and height-Q weights to npz for the JAX side.

Read-only use of the product interpreter:

  PYTHONDONTWRITEBYTECODE=1 BigRedGym/.venv/bin/python scripts/export_weights.py \
    --checkpoint <ckpt> --q-artifact <dir> --out artifacts/nov10/weights.npz
"""

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--q-artifact", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out = {}
    for k, v in ckpt["actor_state_dict"].items():
        out["actor_" + k.replace(".", "_")] = v.detach().numpy()
    q_sd = torch.load(
        f"{args.q_artifact}/q.pt", map_location="cpu", weights_only=True
    )
    for k, v in q_sd.items():
        out["q_" + k.replace(".", "_")] = v.detach().numpy()
    np.savez(args.out, **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
