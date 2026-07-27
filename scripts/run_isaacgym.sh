#!/bin/bash
# Run a script under the legacy IsaacGym backend (Python 3.8 venv).
# Usage: bash scripts/run_isaacgym.sh scripts/train.py --task mini_cheetah --headless ...
#
# The .venv38 env is created with:
#   uv venv --python 3.8 .venv38
#   uv pip install --python .venv38/bin/python -e ../isaacgym/python
#   uv pip install --python .venv38/bin/python -r requirements.txt ninja
# PATH must include .venv38/bin so torch's JIT build of gymtorch finds ninja.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PATH="$PWD/.venv38/bin:$PATH"
exec .venv38/bin/python "$@"
