#!/bin/bash
# Run the vsim backend test suite (local-only: node-locked license, CUDA).
# Usage: bash scripts/run_vsim_tests.sh [extra pytest args]
set -euo pipefail
cd "$(dirname "$0")/.."

# NB: computed WITHOUT importing vlearn (its import needs this very path)
VLEARN_LIB="$(uv run python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "vlearn", "lib"))')"
export LD_LIBRARY_PATH="${VLEARN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VL_WORKING_DIRECTORY="${VL_WORKING_DIRECTORY:-$(pwd)/thirdparty/vlearn}"
export Q2_VSIM_TESTS=1

uv run python -m pytest tests/unit_tests/ -q -k "vsim" "$@"
