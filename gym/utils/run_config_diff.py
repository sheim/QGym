"""Compare task configuration source snapshots saved with training runs."""

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path

from gym.utils.original_cfg import (
    OriginalCfgError,
    original_cfg_source_dir,
    saved_task_config_paths,
)


class LoggedConfigError(RuntimeError):
    """A training run does not contain readable task configuration sources."""


@dataclass(frozen=True)
class ConfigFileDiff:
    """A unified diff and its changed-line counts for one config source file."""

    path: str
    status: str
    additions: int
    deletions: int
    unified_diff: str


def _imported_config_paths(path: Path, source_root: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise LoggedConfigError(f"Cannot read saved config source: {path}") from error

    module_parts = path.relative_to(source_root).with_suffix("").parts
    package_parts = module_parts[:-1]
    imported = set()
    for node in ast.walk(tree):
        module_names = []
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                module_names.append((*package_parts[:keep], *node.module.split(".")))
            elif node.module.startswith("gym.envs."):
                module_names.append(tuple(node.module.split(".")[2:]))
        elif isinstance(node, ast.Import):
            module_names.extend(
                tuple(alias.name.split(".")[2:])
                for alias in node.names
                if alias.name.startswith("gym.envs.")
            )

        for parts in module_names:
            candidate = source_root.joinpath(*parts).with_suffix(".py")
            if candidate.name.endswith("_config.py") and candidate.is_file():
                imported.add(candidate)
    return imported


def logged_config_sources(
    checkpoint_path: str | Path, task_name: str
) -> dict[str, str]:
    """Read the saved task config and the base config modules it imports.

    Source is read as text and parsed only to follow config imports. The saved
    Python is never imported or executed.
    """
    run_dir = Path(checkpoint_path).expanduser().resolve().parent
    source_root = original_cfg_source_dir(run_dir)
    try:
        pending = list(saved_task_config_paths(task_name, source_root))
    except OriginalCfgError as error:
        raise LoggedConfigError(str(error)) from error

    sources = {}
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        try:
            relative_path = path.relative_to(source_root)
            source = path.read_text()
        except OSError as error:
            raise LoggedConfigError(
                f"Cannot read saved config source: {path}"
            ) from error
        sources[f"gym/envs/{relative_path.as_posix()}"] = source
        pending.extend(_imported_config_paths(path, source_root) - visited)
    return dict(sorted(sources.items()))


def diff_logged_run_configs(
    before_checkpoint: str | Path,
    after_checkpoint: str | Path,
    task_name: str,
) -> list[ConfigFileDiff]:
    """Return changed saved config files from ``before`` to ``after``."""
    before_checkpoint = Path(before_checkpoint).expanduser().resolve()
    after_checkpoint = Path(after_checkpoint).expanduser().resolve()
    before_sources = logged_config_sources(before_checkpoint, task_name)
    after_sources = logged_config_sources(after_checkpoint, task_name)
    before_name = before_checkpoint.parent.name
    after_name = after_checkpoint.parent.name

    changes = []
    for path in sorted(before_sources.keys() | after_sources.keys()):
        before = before_sources.get(path)
        after = after_sources.get(path)
        if before == after:
            continue
        if before is None:
            status = "added"
        elif after is None:
            status = "removed"
        else:
            status = "modified"
        lines = list(
            difflib.unified_diff(
                [] if before is None else before.splitlines(),
                [] if after is None else after.splitlines(),
                fromfile=f"{before_name}/{path}",
                tofile=f"{after_name}/{path}",
                lineterm="",
            )
        )
        additions = sum(line.startswith("+") for line in lines[2:])
        deletions = sum(line.startswith("-") for line in lines[2:])
        changes.append(
            ConfigFileDiff(
                path=path,
                status=status,
                additions=additions,
                deletions=deletions,
                unified_diff="\n".join(lines),
            )
        )
    return changes
