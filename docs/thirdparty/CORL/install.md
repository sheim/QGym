# CORL: local setup and supported scope

All commands below run from the **BigRedGym repository root**.

## Integration model

CORL lives in the Git submodule `thirdparty/CORL`. It is a collection of
standalone algorithm implementations, not an installed modular Python library.
Our adaptation trains offline and exports an artifact; BRG loads that artifact
without importing CORL.

CORL modifications belong in the submodule's own repository, on a branch with
the same name as the BRG development branch. BRG records an exact submodule
commit, not the moving branch tip. Commit and publish CORL changes to a writable
remote before publishing the corresponding BRG submodule pointer. Uncommitted
submodule changes are not reproduced by cloning BRG.

BRG retains the distribution name `q2` and the Python packages `gym`, `learning`,
and `go2_deploy`. Renaming them is unnecessary for this artifact-based path.

## Setup

Initialize the recorded CORL source revision:

```bash
git submodule update --init thirdparty/CORL
```

This normally checks out a detached commit. Before modifying CORL, create or
switch to its development branch matching the parent BRG branch. The adapted
files described here must be present in the recorded revision; upstream CORL
alone does not contain them.

Use BRG's existing Python 3.11 uv environment. On a fresh checkout, its normal
setup is:

```bash
uv sync --frozen
```

The supported example needs only Torch and NumPy, which BRG already declares.
No CORL-specific Python package, optional extra, system library, or CUDA toolkit
installation is required for this CPU example. Python dependencies remain in
BRG's `pyproject.toml`, `uv.lock`, and `.venv`.

Do **not** install upstream CORL's requirements into BRG's environment. They pin
legacy Torch/CUDA, NumPy, Gym, MuJoCo bindings, and JAX packages that are not the
supported BRG stack. Routine setup uses `--frozen`: regenerating BRG's universal
lock can require its optional local VSim wheel and Unitree source checkout even
when those extras are not selected.

## Usable now

| Capability | Status |
|---|---|
| Import the adapted IQL algorithm classes without Gym/D4RL/Pyrallis | Supported by moving benchmark-only imports into their consumers |
| Train IQL on the supplied synthetic array dataset | CPU smoke example available |
| Export deterministic MLP weights and explicit metadata | Available in the smoke example |
| Load and run the exported policy in BRG without CORL imports | Available through `learning.corl_artifact.CORLPolicy` |
| Compare BRG predictions with CORL reference predictions | Available through `scripts/check_corl_artifact.py` |

### Run the example

Choose an output directory that does not already exist. The example deliberately
refuses to overwrite an existing artifact.

```bash
uv run --frozen --no-sync python thirdparty/CORL/algorithms/offline/iql_smoke.py \
  --output /tmp/opencode/corl-iql-example
uv run --frozen --no-sync python scripts/check_corl_artifact.py \
  /tmp/opencode/corl-iql-example
```

`--no-sync` uses the prepared environment without installing or removing
packages. The first command runs 20 CPU updates by default; `--steps` selects a
positive update count. It checks finite losses and an actor parameter change.
The second command checks 16 exported reference predictions against BRG's
independent inference implementation.

### Artifact format

The example produces:

- `weights.pt`: tensor state dictionary, loaded with `weights_only=True`.
- `metadata.json`: format identifier, MLP dimensions, synthetic observation and
  action names, observation mean/std, scalar action bound, seed, update count,
  CORL commit, and dirty-checkout indicator.
- `reference.npz`: raw observations and CORL actions for the comparison script.

The supported format is `corl-deterministic-mlp-v1`: ReLU hidden layers, a Tanh
output, input normalization, and scalar action scaling/clamping. BRG can move
the loaded module to a device using ordinary PyTorch APIs; the recorded smoke
validation used CPU only.

The artifact is labeled `synthetic-smoke-only`. Its feature names and actions
do not describe a robot task. A source commit plus a dirty flag identifies that
local modifications existed; it does not capture those modifications.

## Separate BRG-owned offline replay/FQE

BRG also contains a separate offline path under `learning/offline/`,
`learning/algorithms/fqe.py`, and `scripts/{train_height_q,evaluate_height_q}.py`.
It records transitions during PPO training, relabels a height reward, and fits a
fixed-policy Q by fitted Q evaluation. It uses only BRG's Torch/NumPy stack and
does not import CORL, D4RL, Gym, or Pyrallis. See
[offline replay and fixed-policy height Q](../../offline_height_q.md).

That path is independent of the CORL submodule. Its presence does not establish
CORL benchmark compatibility or robot-policy quality.

## Not usable as a ready-made BRG integration

- **Robot training or playback via CORL:** the supported CORL example has no
  BRG rollout-to-offline-dataset exporter or task-specific policy adapter. The
  separate BRG-owned replay/FQE path above is the supported offline workflow; it
  does not use CORL algorithms.
- **Robot-policy quality:** finite losses and matching inference do not prove
  useful behavior, convergence, or safe hardware control.
- **Original CORL benchmark CLI:** `algorithms/offline/iql.py` still requires
  Pyrallis, D4RL, OpenAI Gym, and W&B for its original benchmark workflow.
  Only its reusable algorithm path was made dependency-light.
- **Other CORL algorithms:** their presence in the submodule does not establish
  compatibility. They have not been adapted or validated by this work.
- **Arbitrary policies:** the consumer does not implement Gaussian sampling,
  recurrent policies, transformers, or arbitrary serialized CORL objects.
- **Legacy Gym coexistence:** BRG's top-level `gym` shadows OpenAI Gym. The
  supported example avoids that import; benchmark paths still have the conflict.

Running the upstream benchmark stack would additionally involve packages such
as `pyrallis`/`typing-inspect`, D4RL, `pybullet`, `dm-control`, `mjrl`, `termcolor`,
and `h5py`, plus legacy `mujoco-py` and native MuJoCo/build/graphics dependencies.
These are not needed or installed for the adapted smoke path. Review any new
dependency requirements before expanding this integration.

Real robot artifacts need an agreed contract for canonical observation/action
ordering, units, normalization, control frequency, terminal versus timeout
semantics, and task-compatible evaluation before they can be used by BRG.

## Validation evidence

The initial local validation completed:

- 20 IQL updates with finite losses and changed actor weights.
- 16 matching CORL/BRG inference comparisons.
- 238 passing BRG tests across `tests/unit_tests`, `gym`, and `learning`;
  97 optional tests were deselected.
- BRG lint and lint for the modified CORL Python files.

These are recorded results, not guarantees for future revisions. Recheck with:

```bash
uv run --frozen --no-sync python -m pytest -q tests/unit_tests gym learning
```

Warp, VSim, Unitree, GPU IQL, and robot-policy evaluation were not established
by this validation.
