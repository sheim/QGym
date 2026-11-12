"""Check BRG inference against reference actions exported by CORL."""

import argparse
from pathlib import Path

import numpy as np
import torch

from learning.corl_artifact import CORLPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    policy = CORLPolicy(args.artifact)
    with np.load(args.artifact / "reference.npz", allow_pickle=False) as reference:
        actual = policy(torch.from_numpy(reference["observations"]))
        expected = torch.from_numpy(reference["actions"])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print(f"PASS: {actual.shape[0]} CORL/BRG inference comparisons")
    print(f"Artifact purpose: {policy.metadata['purpose']}")


if __name__ == "__main__":
    main()
