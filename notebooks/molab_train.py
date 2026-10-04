# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# ///
"""QGym cloud training as a marimo notebook (runs on MoLab).

The marimo port of ``notebooks/colab_train.ipynb``: clone/refresh the repo,
install the locked GPU dependency graph with uv, train, then download the
checkpoints as a zip.

    uv run marimo edit notebooks/molab_train.py     # local
    # MoLab: upload this file to molab.marimo.io, pick a GPU runtime, run.

Only marimo itself is imported here; the repo and its dependencies live in a
separate uv environment driven through subprocess, exactly as in the Colab
notebook. Disconnect the runtime when you are done or it keeps burning compute.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import subprocess
    import sys
    import textwrap
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, mo, os, re, subprocess, sys, textwrap, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # QGym training in the cloud

    | Section | Does |
    |---|---|
    | 1 Setup | Clones/updates the repo and installs the locked GPU deps with uv |
    | 2 Train | Runs `scripts/train.py`, streaming the log below the cell |
    | 3 Download | Zips a run's checkpoints and hands you the download |

    Pick a **GPU** runtime before running anything. Each section has its own
    button, so re-running the notebook does not restart training.
    """)
    return


@app.cell
def _(Path, os):
    # --- environment ---------------------------------------------------------
    WORK = Path(os.environ.get("QGYM_WORKDIR") or Path.home() / "qgym")
    REPO = WORK / "QGym"
    WARP_CACHE = WORK / "warp_cache"
    WARP_CACHE.mkdir(parents=True, exist_ok=True)

    # uv installs to ~/.local/bin; keep the compiled warp kernels across runs.
    os.environ["PATH"] = f"{Path.home() / '.local/bin'}:{os.environ['PATH']}"
    os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE)
    os.environ["MPLBACKEND"] = "Agg"

    # MoLab runs this notebook inside its own uv venv on Python 3.13 and exports
    # VIRTUAL_ENV / UV_* / PYTHON*. Inherited, those hijack interpreter selection
    # for the repo ("resolved to Python 3.13.11, incompatible with ==3.11.*"), so
    # the repo's uv commands get a scrubbed environment pinned to 3.11.
    _shadowing = (
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONEXECUTABLE",
    )
    ENV = {
        k: v
        for k, v in os.environ.items()
        if k not in _shadowing and not (k.startswith("UV_") and k != "UV_CACHE_DIR")
    }
    ENV["UV_PYTHON"] = "3.11"
    return ENV, REPO, WORK


@app.cell(hide_code=True)
def _(REPO, WORK, mo):
    mo.md(f"""
    ## 1 Setup

    Working directory `{WORK}` &middot; repo `{REPO}`
    """)
    return


@app.cell
def _(mo):
    branch = mo.ui.text(value="main", label="branch", full_width=True)
    repo_url = mo.ui.text(
        value="https://github.com/LampLighterLab/QGym.git",
        label="repo",
        full_width=True,
    )
    setup_button = mo.ui.run_button(label="Clone / update + install")
    mo.vstack([branch, repo_url, setup_button])
    return branch, repo_url, setup_button


@app.cell
def _(ENV, REPO, branch, mo, repo_url, setup_button, subprocess, textwrap):
    mo.stop(not setup_button.value, mo.md("*Press the button to set up.*"))

    _setup = textwrap.dedent(f"""
        set -euo pipefail

        command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

        if [ -d {REPO}/.git ]; then
            git -C {REPO} fetch origin {branch.value}
            git -C {REPO} checkout {branch.value}
            git -C {REPO} pull --ff-only
        else
            git clone --branch {branch.value} {repo_url.value} {REPO}
        fi

        cd {REPO}
        uv python install 3.11          # pyproject pins requires-python == 3.11.*
        # --extra gpu pulls mujoco-warp (CUDA); --no-dev skips pytest/ruff/marimo.
        uv sync --frozen --extra gpu --no-dev
    """)
    subprocess.run(["bash", "-c", _setup], check=True, env=ENV)

    _check = (
        "import torch, mujoco, mujoco_warp; "
        "print('torch', torch.__version__, '| mujoco', mujoco.__version__, "
        "'| cuda', torch.cuda.is_available(), "
        "'|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
    )
    _probe = subprocess.run(
        ["uv", "run", "--frozen", "--extra", "gpu", "--no-dev", "python", "-c", _check],
        cwd=REPO,
        env=ENV,
        check=True,
        capture_output=True,
        text=True,
    )
    setup_done = True
    mo.md(f"""
    ```
    {_probe.stdout.strip()}
    ```
    Ready. If `cuda` is False, switch the runtime to a GPU one and re-run.
    """)
    return (setup_done,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 Train
    """)
    return


@app.cell
def _(mo):
    task = mo.ui.text(
        value="go2trot", label="task (go2trot | mini_cheetah | humanoid | pendulum ...)"
    )
    max_iterations = mo.ui.number(1, 1_000_000, value=300, label="max_iterations")
    num_envs = mo.ui.number(1, 262_144, value=4096, label="num_envs")
    device = mo.ui.text(value="cuda:0", label="device")
    experiment_name = mo.ui.text(
        value="", label="experiment_name (blank: task default)"
    )
    resume = mo.ui.checkbox(False, label="resume newest run for this experiment")
    use_wandb = mo.ui.checkbox(False, label="log to wandb (needs WANDB_API_KEY)")
    train_button = mo.ui.run_button(label="Train")

    mo.vstack(
        [
            mo.hstack([task, device], widths=[2, 1], gap=1),
            mo.hstack([max_iterations, num_envs], widths=[1, 1], gap=1),
            experiment_name,
            resume,
            use_wandb,
            train_button,
        ]
    )
    return (
        device,
        experiment_name,
        max_iterations,
        num_envs,
        resume,
        task,
        train_button,
        use_wandb,
    )


@app.cell
def _(
    ENV,
    REPO,
    device,
    experiment_name,
    max_iterations,
    mo,
    num_envs,
    resume,
    setup_done,
    subprocess,
    sys,
    task,
    train_button,
    use_wandb,
):
    mo.stop(not train_button.value, mo.md("*Press **Train** to start.*"))
    mo.stop(not setup_done, mo.md("Run section 1 first."))

    _cmd = ["uv", "run", "--frozen", "--extra", "gpu", "--no-dev", "scripts/train.py"]
    _cmd += ["--task", task.value, "--device", device.value, "--headless"]
    _cmd += ["--num_envs", str(int(num_envs.value))]
    _cmd += ["--max_iterations", str(int(max_iterations.value))]
    if experiment_name.value.strip():
        _cmd += ["--experiment_name", experiment_name.value.strip()]
    if resume.value:
        _cmd += ["--resume"]
    if not use_wandb.value:
        _cmd += ["--disable_wandb"]

    print(" ".join(_cmd), "\n", flush=True)
    # First warp run JIT-compiles kernels (minutes, cached afterwards). Output is
    # streamed line by line into this cell's console so you can watch progress.
    _proc = subprocess.Popen(
        _cmd,
        cwd=REPO,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for _line in _proc.stdout:
        print(_line, end="")
    if _proc.wait() != 0:
        sys.exit(f"training exited with code {_proc.returncode}")
    mo.md("Training finished. Download the checkpoints below.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 Download checkpoints

    Runs live at `logs/<experiment_name>/<Mon##_HH-MM-SS>_<run>/model_<iter>.pt`.
    The zip carries each run's saved configs next to its checkpoints, so
    `--resume` and playback work from the extracted copy.

    Nothing here depends on section 2, so you can zip a run while training
    continues -- or in a runtime where the checkpoints are already on disk.
    """)
    return


@app.cell
def _(Path, re):
    def iteration(p: Path) -> int:
        return int(re.search(r"model_(\d+)\.pt", p.name).group(1))

    def find_runs(logs: Path) -> list[Path]:
        """Run dirs holding checkpoints, newest-touched first."""
        return sorted(
            {p.parent for p in logs.glob("*/*/model_*.pt")},
            key=lambda d: max(p.stat().st_mtime for p in d.glob("model_*.pt")),
            reverse=True,
        )

    return find_runs, iteration


@app.cell
def _(REPO, find_runs, mo):
    _logs = REPO / "logs"
    _runs = find_runs(_logs)
    mo.stop(not _runs, mo.md(f"No checkpoints found under `{_logs}`."))

    run_pick = mo.ui.multiselect(
        options={f"{r.parent.name}/{r.name}": r for r in _runs},
        value=[f"{_runs[0].parent.name}/{_runs[0].name}"],
        label="runs to include (newest first)",
        full_width=True,
    )
    zip_button = mo.ui.run_button(label="Build zip")
    mo.vstack([run_pick, zip_button])
    return run_pick, zip_button


@app.cell
def _(Path, WORK, iteration, mo, run_pick, zip_button, zipfile):
    mo.stop(not zip_button.value, mo.md("*Press **Build zip** to package the runs.*"))
    mo.stop(not run_pick.value, mo.md("Select at least one run."))

    _picked = list(run_pick.value)
    _name = f"{_picked[0].name}.zip" if len(_picked) == 1 else "qgym_checkpoints.zip"
    _zip = WORK / "exports" / _name
    _zip.parent.mkdir(parents=True, exist_ok=True)

    _lines = []
    # Checkpoints are already-compressed tensors; ZIP_STORED keeps this fast.
    with zipfile.ZipFile(_zip, "w", compression=zipfile.ZIP_STORED) as _zf:
        for _run in _picked:
            _ckpts = sorted(_run.glob("model_*.pt"), key=iteration)
            for _f in sorted(_run.rglob("*")):
                if _f.is_file():
                    _arc = Path(_run.parent.name) / _run.name / _f.relative_to(_run)
                    _zf.write(_f, _arc)
            _span = (
                f", iters {iteration(_ckpts[0])}..{iteration(_ckpts[-1])}"
                if _ckpts
                else ""
            )
            _lines.append(
                f"`{_run.parent.name}/{_run.name}`: {len(_ckpts)} ckpt{_span}"
            )

    _lines.append(f"\n**{_zip.name}** &mdash; {_zip.stat().st_size / 1e6:.1f} MB")
    mo.vstack(
        [
            mo.md("\n\n".join(_lines)),
            # Lazy read: the archive is only slurped into memory when you click.
            mo.download(
                data=lambda p=_zip: p.read_bytes(), filename=_zip.name, label="Download"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Disconnect the runtime when you are done or it keeps consuming compute.**
    """)
    return


if __name__ == "__main__":
    app.run()
