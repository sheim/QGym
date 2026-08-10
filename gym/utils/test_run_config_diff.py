import pytest

from gym.utils.run_config_diff import (
    LoggedConfigError,
    diff_logged_run_configs,
    logged_config_sources,
)


def _write_run(tmp_path, name, task_source, base_source, unrelated_source=""):
    run_dir = tmp_path / name
    source_root = run_dir / "files" / "gym" / "envs"
    task_dir = source_root / "go2"
    base_dir = source_root / "base"
    unrelated_dir = source_root / "other"
    task_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)
    unrelated_dir.mkdir(parents=True)
    (source_root / "__init__.py").write_text(
        'config_dict = {"Go2TrotCfg": ".go2.go2trot_config"}\n'
        'runner_config_dict = {"Go2TrotRunnerCfg": ".go2.go2trot_config"}\n'
        'task_dict = {"go2trot": '
        '["Go2Trot", "Go2TrotCfg", "Go2TrotRunnerCfg"]}\n'
    )
    (task_dir / "go2trot_config.py").write_text(task_source)
    (base_dir / "legged_robot_config.py").write_text(base_source)
    (unrelated_dir / "other_config.py").write_text(unrelated_source)
    checkpoint = run_dir / "model_10.pt"
    checkpoint.touch()
    return checkpoint


def test_diff_follows_task_config_imports_and_ignores_unrelated_configs(tmp_path):
    import_line = "from gym.envs.base.legged_robot_config import Base\n"
    before = _write_run(
        tmp_path,
        "before",
        import_line + "GAIN = 1.0\n",
        "class Base:\n    damping = 0.5\n",
        "VALUE = 1\n",
    )
    after = _write_run(
        tmp_path,
        "after",
        import_line + "GAIN = 2.0\n",
        "class Base:\n    damping = 0.7\n",
        "VALUE = 2\n",
    )

    changes = diff_logged_run_configs(before, after, "go2trot")

    assert [change.path for change in changes] == [
        "gym/envs/base/legged_robot_config.py",
        "gym/envs/go2/go2trot_config.py",
    ]
    assert all(change.status == "modified" for change in changes)
    assert all((change.additions, change.deletions) == (1, 1) for change in changes)
    assert "-GAIN = 1.0" in changes[1].unified_diff
    assert "+GAIN = 2.0" in changes[1].unified_diff


def test_carried_original_config_takes_precedence(tmp_path):
    checkpoint = _write_run(tmp_path, "resumed", "VALUE = 1\n", "")
    carried = checkpoint.parent / "files" / "original_cfg" / "gym" / "envs" / "go2"
    carried.mkdir(parents=True)
    carried_source_root = carried.parent
    (carried_source_root / "__init__.py").write_text(
        'config_dict = {"Go2TrotCfg": ".go2.go2trot_config"}\n'
        'runner_config_dict = {"Go2TrotRunnerCfg": ".go2.go2trot_config"}\n'
        'task_dict = {"go2trot": '
        '["Go2Trot", "Go2TrotCfg", "Go2TrotRunnerCfg"]}\n'
    )
    (carried / "go2trot_config.py").write_text("VALUE = 2\n")

    sources = logged_config_sources(checkpoint, "go2trot")

    assert sources["gym/envs/go2/go2trot_config.py"] == "VALUE = 2\n"


def test_missing_saved_task_config_is_reported(tmp_path):
    checkpoint = tmp_path / "run" / "model_10.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    with pytest.raises(LoggedConfigError, match="Saved config manifest"):
        logged_config_sources(checkpoint, "go2trot")
