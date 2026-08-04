#!/bin/bash
# Run the vsim backend test suite (local-only: node-locked license, CUDA).
# Usage: bash scripts/run_vsim_tests.sh [extra pytest args]
set -euo pipefail
cd "$(dirname "$0")/.."

VSIM_UV=(uv run --frozen)

# NB: computed WITHOUT importing vlearn (its import needs this very path)
VLEARN_LIB="$("${VSIM_UV[@]}" python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "vlearn", "lib"))')"
export LD_LIBRARY_PATH="${VLEARN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VL_WORKING_DIRECTORY="${VL_WORKING_DIRECTORY:-$(pwd)/thirdparty/vlearn}"
export VL_TURBO_ACTIVATE_PATH="${VL_TURBO_ACTIVATE_PATH:-$(pwd)/thirdparty/vlearn/TurboActivate.dat}"
export VL_LICENSE_KEY_PATH="${VL_LICENSE_KEY_PATH:-$(pwd)/thirdparty/vlearn/License.key}"
export Q2_VSIM_TESTS=1

# Validate the license once so an unavailable backend produces one setup
# failure instead of an error for every parametrized backend test.
if ! (
  cd thirdparty/vlearn
  env \
    LD_LIBRARY_PATH="$VLEARN_LIB" \
    VL_WORKING_DIRECTORY="$PWD" \
    VL_TURBO_ACTIVATE_PATH="$PWD/TurboActivate.dat" \
    VL_LICENSE_KEY_PATH="$PWD/License.key" \
    ../../.venv/bin/python -c \
      'import vlearn as v; v.create_gym(with_render=False, with_window=False); v.delete_gym()'
); then
  echo "VSim license preflight failed; tests were not started." >&2
  exit 1
fi

"${VSIM_UV[@]}" python -m pytest tests/unit_tests/ -q -m vsim "$@"
