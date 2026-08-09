"""Load environment and runner configs from a saved run source snapshot."""

import ast
import importlib
import importlib.machinery
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import gym as gym_package

from gym import GYM_ROOT_DIR
from gym.utils.helpers import select_run


class OriginalCfgError(RuntimeError):
    """A saved run does not contain a usable config snapshot."""


_MANIFEST_NAMES = ("config_dict", "runner_config_dict", "task_dict")


def original_cfg_source_dir(run_dir: str | Path) -> Path:
    """Return the config snapshot represented by a run.

    Runs created with ``--original_cfg`` carry that source forward below
    ``files/original_cfg``. Older runs store it in the normal source snapshot.
    """
    run_dir = Path(run_dir).resolve()
    carried_source = run_dir / "files" / "original_cfg" / "gym" / "envs"
    if carried_source.exists():
        return carried_source
    return run_dir / "files" / "gym" / "envs"


def _read_saved_manifest(manifest_path: Path) -> dict[str, dict]:
    """Read the literal registry dictionaries without executing the snapshot."""
    try:
        source = manifest_path.read_text()
    except OSError as error:
        raise OriginalCfgError(
            f"Saved config manifest not found: {manifest_path}"
        ) from error

    try:
        tree = ast.parse(source, filename=str(manifest_path))
    except SyntaxError as error:
        raise OriginalCfgError(
            f"Saved config manifest is not valid Python: {manifest_path}"
        ) from error

    manifest = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target

        if not isinstance(target, ast.Name) or target.id not in _MANIFEST_NAMES:
            continue

        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as error:
            raise OriginalCfgError(
                f"{target.id} in saved manifest must be a literal dictionary: "
                f"{manifest_path}"
            ) from error
        if not isinstance(value, dict):
            raise OriginalCfgError(
                f"{target.id} in saved manifest is not a dictionary: {manifest_path}"
            )
        manifest[target.id] = value

    missing = sorted(set(_MANIFEST_NAMES) - set(manifest))
    if missing:
        raise OriginalCfgError(
            f"Saved config manifest {manifest_path} is missing: {', '.join(missing)}"
        )
    return manifest


def _config_declarations(
    task_name: str,
    manifest: dict[str, dict],
    manifest_path: Path,
) -> tuple[str, str, str, str]:
    try:
        task_declaration = manifest["task_dict"][task_name]
    except KeyError as error:
        raise OriginalCfgError(
            f"Task {task_name!r} is not declared in saved manifest {manifest_path}"
        ) from error

    if (
        not isinstance(task_declaration, (list, tuple))
        or len(task_declaration) != 3
        or not all(isinstance(name, str) for name in task_declaration)
    ):
        raise OriginalCfgError(
            f"Task {task_name!r} has an invalid declaration in {manifest_path}"
        )

    _, env_cfg_name, train_cfg_name = task_declaration
    try:
        env_cfg_module = manifest["config_dict"][env_cfg_name]
        train_cfg_module = manifest["runner_config_dict"][train_cfg_name]
    except KeyError as error:
        raise OriginalCfgError(
            f"Task {task_name!r} has incomplete config declarations in {manifest_path}"
        ) from error

    return env_cfg_name, env_cfg_module, train_cfg_name, train_cfg_module


def _absolute_module_name(location: str, manifest_path: Path) -> str:
    if not isinstance(location, str):
        raise OriginalCfgError(
            f"Config module location is not a string in {manifest_path}"
        )
    if location.startswith(".") and not location.startswith(".."):
        module_name = f"gym.envs{location}"
    elif location.startswith("gym.envs."):
        module_name = location
    else:
        raise OriginalCfgError(
            f"Config module {location!r} is outside gym.envs in {manifest_path}"
        )

    if not all(part.isidentifier() for part in module_name.split(".")):
        raise OriginalCfgError(
            f"Config module {location!r} is invalid in {manifest_path}"
        )
    return module_name


def _require_module_file(
    module_name: str,
    saved_envs_dir: Path,
    manifest_path: Path,
) -> None:
    relative_parts = module_name.split(".")[2:]
    module_path = saved_envs_dir.joinpath(*relative_parts).with_suffix(".py")
    if not module_path.is_file():
        raise OriginalCfgError(
            f"Saved config module declared by {manifest_path} was not found: "
            f"{module_path}"
        )


def saved_task_config_paths(
    task_name: str, saved_envs_dir: str | Path
) -> tuple[Path, ...]:
    """Resolve a task's config modules from a saved registry manifest."""
    saved_envs_dir = Path(saved_envs_dir).resolve()
    manifest_path = saved_envs_dir / "__init__.py"
    manifest = _read_saved_manifest(manifest_path)
    _, env_cfg_location, _, train_cfg_location = _config_declarations(
        task_name, manifest, manifest_path
    )

    paths = set()
    for location in (env_cfg_location, train_cfg_location):
        module_name = _absolute_module_name(location, manifest_path)
        _require_module_file(module_name, saved_envs_dir, manifest_path)
        relative_parts = module_name.split(".")[2:]
        paths.add(saved_envs_dir.joinpath(*relative_parts).with_suffix(".py"))
    return tuple(sorted(paths))


@contextmanager
def _saved_env_imports(saved_envs_dir: Path):
    """Temporarily resolve gym.envs imports only from the saved snapshot."""
    module_prefix = "gym.envs"
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == module_prefix or name.startswith(f"{module_prefix}.")
    }
    gym_attributes = vars(gym_package)
    had_envs_attribute = "envs" in gym_attributes
    saved_envs_attribute = gym_attributes.get("envs")

    for name in saved_modules:
        del sys.modules[name]

    saved_envs_package = ModuleType(module_prefix)
    saved_envs_package.__package__ = module_prefix
    saved_envs_package.__path__ = [str(saved_envs_dir)]
    saved_envs_package.__spec__ = importlib.machinery.ModuleSpec(
        module_prefix,
        loader=None,
        is_package=True,
    )
    saved_envs_package.__spec__.submodule_search_locations = saved_envs_package.__path__
    sys.modules[module_prefix] = saved_envs_package
    gym_package.envs = saved_envs_package
    importlib.invalidate_caches()

    try:
        yield
    finally:
        loaded_saved_modules = [
            name
            for name in sys.modules
            if name == module_prefix or name.startswith(f"{module_prefix}.")
        ]
        for name in loaded_saved_modules:
            del sys.modules[name]
        sys.modules.update(saved_modules)

        if had_envs_attribute:
            gym_package.envs = saved_envs_attribute
        else:
            gym_attributes.pop("envs", None)
        importlib.invalidate_caches()


def load_original_cfgs_from_run(task_name: str, run_dir: str | Path):
    """Instantiate a task's saved environment and runner config classes.

    Only config modules are loaded from the run. The current task class,
    learning implementation, and simulator backends remain active.
    """
    run_dir = Path(run_dir).resolve()
    saved_envs_dir = original_cfg_source_dir(run_dir)
    manifest_path = saved_envs_dir / "__init__.py"
    manifest = _read_saved_manifest(manifest_path)
    (
        env_cfg_name,
        env_cfg_location,
        train_cfg_name,
        train_cfg_location,
    ) = _config_declarations(task_name, manifest, manifest_path)

    env_cfg_module_name = _absolute_module_name(env_cfg_location, manifest_path)
    train_cfg_module_name = _absolute_module_name(train_cfg_location, manifest_path)
    _require_module_file(env_cfg_module_name, saved_envs_dir, manifest_path)
    _require_module_file(train_cfg_module_name, saved_envs_dir, manifest_path)

    with _saved_env_imports(saved_envs_dir):
        try:
            env_cfg_module = importlib.import_module(env_cfg_module_name)
            train_cfg_module = importlib.import_module(train_cfg_module_name)
        except (ImportError, ModuleNotFoundError) as error:
            raise OriginalCfgError(
                f"Could not import saved configs for task {task_name!r} from {run_dir}"
            ) from error

        try:
            env_cfg_class = getattr(env_cfg_module, env_cfg_name)
            train_cfg_class = getattr(train_cfg_module, train_cfg_name)
        except AttributeError as error:
            raise OriginalCfgError(
                f"Saved config class declared for task {task_name!r} was not "
                f"found in {run_dir}"
            ) from error

        env_cfg = env_cfg_class()
        train_cfg = train_cfg_class()

    return env_cfg, train_cfg


def load_original_cfgs(
    task_name: str,
    experiment_name: str,
    load_run: str | int | None = -1,
):
    """Resolve a run below ``logs/`` and load its saved task configs."""
    if not experiment_name:
        raise OriginalCfgError("An experiment name is required to load saved configs")

    run_selector = -1 if load_run is None else load_run
    experiment_dir = Path(GYM_ROOT_DIR) / "logs" / experiment_name
    try:
        run_dir = Path(select_run(str(experiment_dir), run_selector)).resolve()
    except (OSError, TypeError, ValueError) as error:
        raise OriginalCfgError(
            f"Could not resolve run {run_selector!r} below {experiment_dir}"
        ) from error

    env_cfg, train_cfg = load_original_cfgs_from_run(task_name, run_dir)
    return env_cfg, train_cfg, run_dir
