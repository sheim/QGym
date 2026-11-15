# Intervention dependencies — compatibility research

Date: 2026-11-15. Scope: read-only metadata review, not an installation or runtime certification. Architecture progression remains blocked pending the user's compatibility requirement.

## BRG baseline

- `BigRedGym/pyproject.toml` requires Python 3.11; `.python-version` agrees.
- `BigRedGym/uv.lock` pins Torch 2.13.0, NumPy 2.4.6, MuJoCo and mujoco-warp 3.11.0, and warp-lang 1.16.0; these are lockfile observations, not an installed-environment audit.
- JAX, JAXlib, SciPy, and the candidate intervention packages were not found in the inspected lockfile package inventory. BRG owns a top-level `gym` package.

## Candidate findings

| Candidate | Declared compatibility | Caveats / decision status |
|---|---|---|
| pytorch-mppi 0.9.1 | Python >=3.6; Torch >=2.0; NumPy; arm-pytorch-utilities >=0.4. BRG meets the direct Python/Torch requirements. | Strongest MPPI candidate, not runtime-verified. Avoid upstream `test` extra: it installs OpenAI Gym, which collides with BRG's `gym` namespace. No need for tuning extras. |
| arm-pytorch-utilities 0.5.0 | Requires Torch, NumPy, Matplotlib, SciPy, pytorch-seed. pytorch-seed 0.2.0 requires Torch and NumPy. | MPPI adds SciPy transitively; the apparently small dependency is not Torch-only. |
| CBFpy 0.1.0 | Requires NumPy, JAX, qpax, packaging. qpax 0.1.4 declares Python >=3.10 and JAX/JAXlib >=0.4.0. | No direct metadata conflict identified with a Python-3.11-compatible JAX stack; API/runtime verification remains open. Avoid optional examples and ElastiQP until separately checked. |
| CBFKit main | Python >=3.10,<3.13; JAX >=0.4.23, jaxopt >=0.8.3,<0.9, SciPy >=1.9,<2.0, plus plotting/config dependencies. Torch bridge requires Torch >=2.8 and JAX >=0.6.2. | Potentially compatible core/bridge, not a complete transitive resolution. Requires an immutable Git revision before selection; main is mutable. |
| CBFKit `mujoco` extra | Pins MuJoCo and mujoco-mjx to 3.13.0. | Conflicts with preserving BRG's locked MuJoCo 3.11.0. Do not use this extra without a separate simulator-upgrade decision and parity validation. Its MJX integration is not BRG's Warp backend. |

## Python and numerical stack constraints

- Latest inspected JAX/JAXlib 0.12.0.1 and SciPy 1.18.1 require Python >=3.12; these exact versions cannot be used in BRG. A resolver may select older releases, but latest-version instructions are unsuitable.
- Candidate CPU stack: JAX/JAXlib 0.7.2 support Python >=3.11, NumPy >=2.0, SciPy >=1.13; SciPy 1.16.3 supports Python >=3.11 and NumPy >=1.25.2,<2.6, which includes BRG's 2.4.6. Linux x86_64 CPython 3.11 wheels exist for JAXlib 0.7.2 and SciPy 1.16.3. These are candidate pins, not an approved or fully solved environment.
- CPU JAX avoids adding a second CUDA dependency stack, but Torch CUDA transfers and synchronization may be costly. GPU JAX requires separate wheel/CUDA/driver, memory coexistence, and Torch interoperability verification; CUDA compatibility cannot be inferred from CPU metadata.

## Evidence boundary and unresolved gate

- No clone, install, lockfile change, import, solver execution, or BRG regression was performed. Scaffold permissions prohibit those state-changing operations, including temporary environments.
- Before declaring compatibility, an execution-capable session must establish an isolated resolved environment preserving BRG's existing simulator/Torch/NumPy versions, inspect package provenance and namespace resolution, and exercise MPPI and CBF alongside BRG on the intended devices.
- Solver feasibility/failure reporting, seeded behavior, Torch/JAX conversion, and headless Go2Trot execution remain unverified. Passing package resolution alone would not close this gate; exact supported versions and evidence must be returned before final dependency selection.

## Sources

- Local: `BigRedGym/pyproject.toml`, `BigRedGym/.python-version`, `BigRedGym/uv.lock`, and setup references in `BigRedGym/genAI_skills/README_MUJOCO.md`.
- Upstream metadata: [MPPI pyproject](https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/pyproject.toml), [CBFpy pyproject](https://github.com/StanfordASL/cbfpy/blob/main/pyproject.toml), [CBFKit pyproject](https://github.com/bardhh/cbfkit/blob/main/pyproject.toml).
- PyPI JSON metadata queried for pytorch-mppi, arm-pytorch-utilities, pytorch-seed, cbfpy, qpax, jax, jaxlib, scipy, ml-dtypes; exact-version endpoints included [JAX 0.7.2](https://pypi.org/pypi/jax/0.7.2/json), [JAXlib 0.7.2](https://pypi.org/pypi/jaxlib/0.7.2/json), and [SciPy 1.16.3](https://pypi.org/pypi/scipy/1.16.3/json).
