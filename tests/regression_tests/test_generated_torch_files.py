import subprocess
import sys
from pathlib import Path

import torch

REFERENCE_FILE = Path(__file__).with_name("main_output.pt")
RUN_SCRIPT = Path(__file__).with_name("run_n_iterations.py")


def _generate_candidate_tensor(output_path: Path) -> None:
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--output_tensor_file",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _load_tensor(path: Path):
    return torch.load(path, weights_only=True)


def test_generated_tensor_matches_reference(tmp_path):
    candidate_path = tmp_path / "generated_output.pt"
    _generate_candidate_tensor(candidate_path)
    reference = _load_tensor(REFERENCE_FILE)
    candidate = _load_tensor(candidate_path)
    assert torch.equal(reference, candidate), (
        f"Tensors in {REFERENCE_FILE} and {candidate_path} differ."
    )
