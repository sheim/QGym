import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.eval_policy as eval_policy
import scripts.run_domain_randomization_campaign as campaign
from gym.envs.base.domain_randomization import (
    apply_domain_randomization_override,
    get_domain_randomization_range,
)
from scripts.benchmark_domain_randomization import (
    DEFAULT_MIN_STEPS,
    default_warmup_steps,
    timeout_reset_counts,
)
from scripts.eval_policy import (
    configure_contact_friction_grid,
    crossed_contact_friction_grid,
    current_contact_friction,
)
from scripts.run_domain_randomization_campaign import (
    CampaignState,
    CellSpec,
    aggregate_paired_eval_effects,
    aggregate_paired_training_effects,
    assert_source_provenance_unchanged,
    campaign_cases,
    campaign_completion,
    evaluation_cases,
    expected_campaign_cells,
    find_training_run,
    get_args as get_campaign_args,
    load_manifest,
    markdown_report,
    paired_training_effects,
    run_cell,
    speed_cases,
    speed_rows,
    source_provenance,
    training_cases,
    training_checkpoint_signatures,
    validate_artifact,
    validate_cell_artifact,
)
from scripts.train import get_train_args
from scripts.train_domain_randomization import get_args as get_dr_train_args


def test_public_train_cli_has_no_domain_randomization_override():
    args = get_train_args(["--task", "go2trot"])
    assert args.save_interval is None
    assert not hasattr(args, "domain_randomization")

    with pytest.raises(SystemExit):
        get_train_args(["--task", "go2trot", "--domain-randomization", "off"])


def test_domain_randomization_worker_owns_campaign_bundle():
    args = get_dr_train_args(
        ["--task", "go2trot", "--dr-bundle", "friction", "--headless"]
    )

    assert args.dr_bundle == "friction"
    assert args.headless is True


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("off", (None, None, None, None)),
        ("friction-only", ([0.5, 1.0], None, None, None)),
        ("pd-only", (None, [0.9, 1.1], [0.8, 1.2], None)),
        ("mass-only", (None, None, None, [0.85, 1.15])),
    ],
)
def test_campaign_domain_randomization_bundle_override(mode, expected):
    cfg = SimpleNamespace(
        domain_randomization=SimpleNamespace(
            startup=SimpleNamespace(
                contact_friction_range=[0.5, 1.0],
                link_mass_scale_range=[0.85, 1.15],
            ),
            episode=SimpleNamespace(
                scale_ranges={
                    "p_gains": [0.9, 1.1],
                    "d_gains": [0.8, 1.2],
                }
            ),
        )
    )

    apply_domain_randomization_override(cfg, mode)

    assert (
        get_domain_randomization_range(cfg, "contact_friction_range"),
        get_domain_randomization_range(cfg, "p_gains"),
        get_domain_randomization_range(cfg, "d_gains"),
        get_domain_randomization_range(cfg, "link_mass_scale_range"),
    ) == expected


def test_crossed_friction_grid_balances_every_command_case():
    command_cases = np.asarray(["stand", "forward"] * 4)
    values = crossed_contact_friction_grid(command_cases, 0.5, 1.0)

    expected = np.linspace(0.5, 1.0, 4, dtype=np.float32)
    np.testing.assert_array_equal(values[command_cases == "stand"], expected)
    np.testing.assert_array_equal(values[command_cases == "forward"], expected)


def test_eval_grid_allocates_dr_topology_without_changing_nominal_material():
    cfg = SimpleNamespace(
        terrain=SimpleNamespace(static_friction=1.0, dynamic_friction=1.0),
        domain_randomization=SimpleNamespace(
            startup=SimpleNamespace(
                contact_friction_range=None, link_mass_scale_range=None
            ),
            episode=SimpleNamespace(scale_ranges={}),
        ),
    )
    configure_contact_friction_grid(cfg, [0.35, 0.35])

    assert cfg.terrain.static_friction == 1.0
    assert cfg.terrain.dynamic_friction == 1.0
    assert cfg.domain_randomization.startup.contact_friction_range == [0.35, 0.35]


def test_eval_without_grid_records_nominal_for_task_without_randomizer_or_terrain():
    env = SimpleNamespace(num_envs=3, cfg=SimpleNamespace())

    np.testing.assert_array_equal(
        current_contact_friction(env), np.ones(3, dtype=np.float32)
    )


@pytest.mark.parametrize(
    "mujoco_overrides",
    [{}, {"mujoco_njmax": 300, "mujoco_geom_solref": [0.01, 1.5]}],
)
def test_eval_build_deepcopies_registry_configs(
    monkeypatch, tmp_path, mujoco_overrides
):
    from gym.envs.base.mujoco_config import MuJoCoCfg

    original_env_cfg = SimpleNamespace(
        env=SimpleNamespace(num_envs=99, episode_length_s=1.0),
        init_state=SimpleNamespace(reset_mode="original"),
        mujoco=MuJoCoCfg(),
        seed=-1,
    )
    original_train_cfg = SimpleNamespace(
        seed=-1,
        runner=SimpleNamespace(device="original", resume=True),
        logging=SimpleNamespace(enable_local_saving=True),
    )
    captured = {}

    class FakeRunner:
        def load(self, path, load_optimizer):
            captured["checkpoint"] = (path, load_optimizer)

        def switch_to_eval(self):
            captured["eval"] = True

    class FakeRegistry:
        def get_cfgs(self, task):
            assert task == "fixed"
            return original_env_cfg, original_train_cfg

        def convert_frequencies_to_params(self, env_cfg, train_cfg):
            captured["converted"] = (env_cfg, train_cfg)

        def set_log_dir_name(self, train_cfg, log_root):
            assert log_root is None

        def make_env(self, task, env_cfg, **kwargs):
            captured["env_cfg"] = env_cfg
            captured["make_env_kwargs"] = kwargs
            return SimpleNamespace()

        def make_alg_runner(self, env, train_cfg):
            captured["train_cfg"] = train_cfg
            return FakeRunner()

    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(eval_policy, "task_registry", FakeRegistry())
    monkeypatch.setattr(eval_policy, "set_seed", lambda seed: None)

    eval_policy.build(
        "fixed",
        "mujoco",
        "cpu",
        4,
        2.0,
        checkpoint,
        "reset_to_range",
        7,
        **mujoco_overrides,
    )

    assert original_env_cfg.env.num_envs == 99
    assert original_env_cfg.init_state.reset_mode == "original"
    assert original_env_cfg.seed == -1
    assert original_train_cfg.runner.device == "original"
    assert original_train_cfg.runner.resume is True
    assert original_train_cfg.logging.enable_local_saving is True
    assert captured["env_cfg"] is not original_env_cfg
    assert captured["env_cfg"].env.num_envs == 4
    assert captured["train_cfg"] is not original_train_cfg
    assert captured["train_cfg"].runner.resume is False
    assert captured["checkpoint"] == (checkpoint, False)
    assert original_env_cfg.mujoco.njmax == -1
    assert original_env_cfg.mujoco.solref == [0.02, 1.0]
    assert captured["env_cfg"].mujoco is not original_env_cfg.mujoco
    if mujoco_overrides:
        assert captured["env_cfg"].mujoco.njmax == 300
        assert captured["env_cfg"].mujoco.solref == [0.01, 1.5]


def test_eval_build_can_sample_every_simulation_step(monkeypatch, tmp_path):
    env_cfg = SimpleNamespace(
        env=SimpleNamespace(num_envs=99, episode_length_s=1.0),
        init_state=SimpleNamespace(reset_mode="original"),
        control=SimpleNamespace(ctrl_frequency=100, desired_sim_frequency=500),
        seed=7,
    )
    train_cfg = SimpleNamespace(
        seed=7,
        runner=SimpleNamespace(device="cpu", resume=False),
    )
    captured = {}

    class FakeRunner:
        def load(self, path, load_optimizer):
            pass

        def switch_to_eval(self):
            pass

    class FakeRegistry:
        def get_cfgs(self, task):
            return env_cfg, train_cfg

        def convert_frequencies_to_params(self, converted_env_cfg, _train_cfg):
            captured["ctrl_frequency"] = converted_env_cfg.control.ctrl_frequency

        def set_log_dir_name(self, _train_cfg, log_root):
            pass

        def make_env(self, task, converted_env_cfg, **kwargs):
            captured["env_cfg"] = converted_env_cfg
            return SimpleNamespace()

        def make_alg_runner(self, env, converted_train_cfg):
            return FakeRunner()

    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(eval_policy, "task_registry", FakeRegistry())
    monkeypatch.setattr(eval_policy, "set_seed", lambda seed: None)

    eval_policy.build(
        "fixed",
        "mujoco",
        "cpu",
        1,
        1.0,
        checkpoint,
        "reset_to_range",
        7,
        control_at_sim_frequency=True,
    )

    assert captured["ctrl_frequency"] == 500
    assert captured["env_cfg"].control.ctrl_frequency == 500
    assert env_cfg.control.ctrl_frequency == 100


def test_timeout_schedule_matches_fractional_episode_reset_rate():
    counts = timeout_reset_counts(num_envs=4096, episode_steps=500, num_steps=100)

    assert set(counts) == {8, 9}
    assert counts.sum() == (100 * 4096) // 500


def test_speed_protocol_uses_longer_default_measurement_and_warmup_windows():
    args = get_campaign_args(
        [
            "--output",
            str(campaign.LOG_ROOT / "dr_unit_protocol"),
            "--stages",
            "summarize",
        ]
    )

    assert DEFAULT_MIN_STEPS == 50
    assert args.speed_min_steps == DEFAULT_MIN_STEPS
    assert default_warmup_steps(4096) == 25
    assert default_warmup_steps(256) == 32
    assert default_warmup_steps(1) == 50


def test_campaign_rejects_duplicate_seeds():
    with pytest.raises(SystemExit):
        get_campaign_args(
            [
                "--output",
                str(campaign.LOG_ROOT / "dr_unit_duplicate_seeds"),
                "--seeds",
                "7",
                "7",
            ]
        )


def _training_row(backend, mode, seed, reward, episode_time, throughput):
    return {
        "label": f"{backend}_{mode}_seed{seed}",
        "backend": {"label": backend},
        "dr_mode": mode,
        "seed": seed,
        "reward_final_10_mean": reward,
        "episode_time_final_10_mean": episode_time,
        "median_logged_steps_per_s_after_warmup": throughput,
    }


def test_training_effects_are_paired_before_aggregation():
    training = []
    for seed, reward_delta, throughput_ratio in (
        (1, 1.0, 0.9),
        (2, 2.0, 1.0),
        (3, 3.0, 1.1),
    ):
        training.extend(
            [
                _training_row("warp", "off", seed, seed, 4.0, 100.0),
                _training_row(
                    "warp",
                    "on",
                    seed,
                    seed + reward_delta,
                    4.1,
                    100.0 * throughput_ratio,
                ),
            ]
        )

    raw = paired_training_effects(training)
    protocol = _campaign_protocol(seeds=[1, 2, 3])
    aggregate = aggregate_paired_training_effects(raw, protocol)
    warp = next(row for row in aggregate if row["training_backend"] == "warp")

    assert warp["paired_seed_count"] == 3
    assert warp["missing_seed_pairs"] == []
    reward = warp["delta_on_minus_off"]["reward_final_10_mean"]
    assert reward == {"count": 3, "mean": 2.0, "sample_std": 1.0}
    throughput = warp["throughput_on_over_off"]
    assert throughput["count"] == 3
    assert throughput["mean"] == pytest.approx(1.0)
    assert throughput["sample_std"] == pytest.approx(0.1)


def test_eval_aggregate_reports_missing_seed_pairs():
    effects = [
        {
            "training_backend": "warp",
            "seed": 1,
            "eval_label": "warp-nominal",
            "delta_on_minus_off": {"survival": 0.1},
        },
        {
            "training_backend": "warp",
            "seed": 2,
            "eval_label": "warp-nominal",
            "delta_on_minus_off": {"survival": 0.3},
        },
    ]

    protocol = _campaign_protocol(seeds=[1, 2, 3])
    aggregate = aggregate_paired_eval_effects(effects, protocol)
    nominal = next(
        row
        for row in aggregate
        if row["training_backend"] == "warp" and row["eval_label"] == "warp-nominal"
    )

    assert nominal["paired_seed_count"] == 2
    assert nominal["paired_seeds"] == [1, 2]
    assert nominal["missing_seed_pairs"] == [3]
    survival = nominal["delta_on_minus_off"]["survival"]
    assert survival["mean"] == pytest.approx(0.2)
    assert survival["sample_std"] == pytest.approx(np.sqrt(0.02))


def _campaign_protocol(*, seeds=None, common_num_envs=256, production_num_envs=4096):
    return {
        "task": "go2trot",
        "backends": [
            {"label": "warp", "physics_backend": "mujoco", "device": "cuda:0"},
            {"label": "vsim", "physics_backend": "vsim", "device": "cuda:0"},
            {"label": "cpu", "physics_backend": "mujoco", "device": "cpu"},
        ],
        "seeds": [7] if seeds is None else seeds,
        "speed_common_num_envs": common_num_envs,
        "speed_production_num_envs": production_num_envs,
        "speed_production_backends": ["warp", "vsim"],
        "speed_target_env_steps": 32768,
        "speed_repeats": 1,
        "speed_min_control_steps": 50,
        "train_num_envs": 4,
        "train_iterations": 3,
        "cross_backend_eval": False,
        "eval_num_envs": 2,
        "eval_duration_s": 5.0,
        "eval_settling_time_s": 0.5,
        "eval_seed": 1701,
        "eval_reset_mode": "reset_to_basic",
        "eval_command_profile": "go2",
        "eval_contact_threshold_n": 20.0,
        "eval_velocity_impulse_m_per_s": 0.0,
        "eval_domains": {
            "nominal": [1.0, 1.0],
            "in_range": [0.5, 1.0],
            "low_friction": [0.35, 0.35],
        },
        "contact_friction_dr_range": [0.5, 1.0],
        "environment_config_template": {"terrain": {"dynamic_friction": 1.0}},
    }


def _write_valid_speed_artifact(path, case=None, protocol=None):
    protocol = _campaign_protocol() if protocol is None else protocol
    case = speed_cases(protocol)[0] if case is None else case
    steps = max(
        protocol["speed_min_control_steps"],
        int(np.ceil(protocol["speed_target_env_steps"] / case.num_envs)),
    )
    warmup_steps = max(25, min(50, int(np.ceil(8192 / case.num_envs))))
    profiles = []
    for name in ("none", "timeout", "all"):
        profiles.append(
            {
                "profile": name,
                "steps_per_trial": steps,
                "repeats": protocol["speed_repeats"],
                "warmup_steps": warmup_steps,
                "elapsed_seconds": [0.1],
                "env_control_steps_per_s": [100.0],
                "median_env_control_steps_per_s": 100.0,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": {
                    "task": protocol["task"],
                    "backend_label": case.backend.label,
                    "physics_backend": case.backend.physics_backend,
                    "device": case.backend.device,
                    "num_envs": case.num_envs,
                    "seed": protocol["seeds"][0],
                    "target_env_steps": protocol["speed_target_env_steps"],
                    "min_steps": protocol["speed_min_control_steps"],
                    "steps_per_trial": steps,
                    "repeats": protocol["speed_repeats"],
                    "warmup_steps": warmup_steps,
                    "dr_enabled": case.dr_mode == "on",
                    "domain_randomization": {
                        "startup": {
                            "contact_friction_range": (
                                protocol["contact_friction_dr_range"]
                                if case.dr_mode == "on"
                                else None
                            ),
                            "link_mass_scale_range": None,
                        },
                        "episode": {"scale_ranges": {}},
                    },
                },
                "setup_seconds": 0.1,
                "memory": {
                    "cuda_after_profiles": None,
                    "cuda_used_delta_bytes": None,
                    "cuda_used_after_profiles_delta_bytes": None,
                    "host_max_rss_after_profiles_kb": 1,
                    "host_max_rss_after_profiles_delta_kb": 1,
                },
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )


def _write_valid_training_artifact(path, case=None, protocol=None):
    protocol = _campaign_protocol() if protocol is None else protocol
    case = training_cases(protocol)[-2] if case is None else case
    iteration = protocol["train_iterations"]
    run_dir = path.parent / f"{path.stem}_run"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / f"model_{iteration}.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "vitals.jsonl").write_text(
        json.dumps({"iteration": iteration}) + "\n", encoding="utf-8"
    )
    resolved = path.parent / f"{path.stem}_resolved.json"
    resolved.write_text(
        json.dumps(
            {
                "task": protocol["task"],
                "backend": {
                    "label": case.backend.label,
                    "physics_backend": case.backend.physics_backend,
                    "device": case.backend.device,
                },
                "dr_mode": case.dr_mode,
                "environment": {
                    "seed": case.seed,
                    "env": {"num_envs": protocol["train_num_envs"]},
                    "domain_randomization": {
                        "startup": {
                            "contact_friction_range": (
                                protocol["contact_friction_dr_range"]
                                if case.dr_mode == "on"
                                else None
                            ),
                            "link_mass_scale_range": None,
                        },
                        "episode": {"scale_ranges": {}},
                    },
                },
                "training": {
                    "seed": case.seed,
                    "runner": {
                        "device": case.backend.device,
                        "max_iterations": iteration,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "label": case.label,
                "backend": {
                    "label": case.backend.label,
                    "physics_backend": case.backend.physics_backend,
                    "device": case.backend.device,
                },
                "dr_mode": case.dr_mode,
                "seed": case.seed,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "resolved_config": str(resolved),
            }
        ),
        encoding="utf-8",
    )


def _write_valid_evaluation_artifact(path, case=None, protocol=None):
    protocol = _campaign_protocol() if protocol is None else protocol
    case = evaluation_cases(protocol)[0] if case is None else case
    num_envs = protocol["eval_num_envs"]
    low, high = case.friction_range
    np.savez_compressed(
        path,
        task=protocol["task"],
        train_label=case.training.label,
        eval_label=f"{case.backend.label}-{case.domain}",
        checkpoint_iteration=np.int64(protocol["train_iterations"]),
        reset_mode=protocol["eval_reset_mode"],
        command_profile=protocol["eval_command_profile"],
        num_envs=np.int64(num_envs),
        duration_s=np.float32(protocol["eval_duration_s"]),
        seed=np.int64(protocol["eval_seed"]),
        settling_time_s=np.float32(protocol["eval_settling_time_s"]),
        contact_threshold_n=np.float32(protocol["eval_contact_threshold_n"]),
        contact_friction=np.linspace(low, high, num_envs, dtype=np.float32),
        contact_friction_grid=np.asarray(case.friction_range, dtype=np.float32),
        survived=np.ones(num_envs, dtype=bool),
        ep_len=np.ones(num_envs, dtype=np.float32),
        mean_reward=np.ones(num_envs, dtype=np.float32),
    )


def test_campaign_plan_is_unique_and_drives_expected_cells():
    protocol = _campaign_protocol(
        common_num_envs=256,
        production_num_envs=256,
    )
    planned = campaign_cases(protocol)

    assert len(speed_cases(protocol)) == 6
    assert len(training_cases(protocol)) == 6
    assert len(evaluation_cases(protocol)) == 18
    assert len({(case.stage, case.label) for case in planned}) == len(planned)
    expected = expected_campaign_cells(protocol)
    assert sum(map(len, expected.values())) == len(planned)


def test_speed_summary_uses_protocol_for_every_cell(tmp_path):
    protocol = _campaign_protocol(
        common_num_envs=256,
        production_num_envs=256,
    )
    protocol["backends"] = protocol["backends"][:1]
    cells = []
    for case in speed_cases(protocol):
        artifact = tmp_path / f"{case.label}.json"
        _write_valid_speed_artifact(artifact, case, protocol)
        cells.append(
            {
                "stage": case.stage,
                "label": case.label,
                "status": "complete",
                "artifact": str(artifact),
            }
        )

    rows = speed_rows({"protocol": protocol, "cells": cells})

    assert len(rows) == 6
    assert {row["dr"] for row in rows} == {"off", "on"}


@pytest.mark.parametrize("stage", ["speed", "train", "eval"])
def test_artifact_identity_is_tied_to_planned_cell(tmp_path, stage):
    protocol = _campaign_protocol()
    cases = {
        "speed": speed_cases(protocol),
        "train": training_cases(protocol),
        "eval": evaluation_cases(protocol),
    }[stage]
    case, wrong_case = cases[0], cases[1]
    artifact = tmp_path / ("artifact.npz" if stage == "eval" else "artifact.json")
    {
        "speed": _write_valid_speed_artifact,
        "train": _write_valid_training_artifact,
        "eval": _write_valid_evaluation_artifact,
    }[stage](artifact, case, protocol)

    assert validate_cell_artifact(case, protocol, artifact) == (True, None)
    valid, reason = validate_cell_artifact(wrong_case, protocol, artifact)
    assert not valid
    assert "does not match cell" in reason or "wrong friction grid" in reason


def test_campaign_completion_separates_failures_from_incomplete(tmp_path):
    artifact = tmp_path / "speed.json"
    _write_valid_speed_artifact(artifact)
    manifest = {
        "protocol": _campaign_protocol(),
        "cells": [
            {
                "stage": "speed",
                "label": "warp_256_off",
                "status": "complete",
                "artifact": str(artifact),
            },
            {
                "stage": "train",
                "label": "cpu_off_seed7",
                "status": "failed",
                "artifact": str(tmp_path / "failed.json"),
            },
            {
                "stage": "eval",
                "label": "vsim_on_seed7__vsim__nominal",
                "status": "running",
                "artifact": str(tmp_path / "running.npz"),
            },
        ],
    }

    completion = campaign_completion(manifest)

    assert completion["by_stage"]["speed"]["expected"] == 10
    assert completion["by_stage"]["speed"]["complete"] == 1
    assert completion["by_stage"]["speed"]["failed"] == 0
    assert completion["by_stage"]["speed"]["incomplete"] == 9
    assert completion["by_stage"]["train"]["failed"] == 1
    assert completion["by_stage"]["eval"]["incomplete"] == 18
    assert completion["overall"] == {
        "expected": 34,
        "complete": 1,
        "failed": 1,
        "incomplete": 32,
    }


@pytest.mark.parametrize(
    ("stage", "writer"),
    [
        ("speed", _write_valid_speed_artifact),
        ("train", _write_valid_training_artifact),
        ("eval", _write_valid_evaluation_artifact),
    ],
)
def test_stage_artifact_validation_rejects_truncated_files(tmp_path, stage, writer):
    suffix = ".npz" if stage == "eval" else ".json"
    artifact = tmp_path / f"artifact{suffix}"
    writer(artifact)
    assert validate_artifact(stage, artifact) == (True, None)

    artifact.write_bytes(b"truncated")
    valid, reason = validate_artifact(stage, artifact)
    assert not valid
    assert reason


def test_training_run_selection_requires_a_new_checkpoint(tmp_path):
    experiment_dir = tmp_path / "experiment"
    old_run = experiment_dir / "old"
    old_run.mkdir(parents=True)
    (old_run / "model_3.pt").write_bytes(b"old")
    previous = training_checkpoint_signatures(experiment_dir, 3)

    with pytest.raises(FileNotFoundError, match="no new model_3.pt"):
        find_training_run(experiment_dir, 3, previous)

    new_run = experiment_dir / "new"
    new_run.mkdir()
    (new_run / "model_3.pt").write_bytes(b"new")
    assert find_training_run(experiment_dir, 3, previous) == new_run


def test_resume_rejects_any_protocol_or_source_change(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    persisted = {
        "schema_version": 2,
        "protocol": {"seeds": [7], "eval_seed": 1701},
        "source_provenance": {"files": {"scripts/train.py": {"sha256": "old"}}},
        "cells": [],
    }
    manifest_path.write_text(json.dumps(persisted), encoding="utf-8")
    args = SimpleNamespace(resume=True, stages=["train"])

    assert (
        load_manifest(
            manifest_path,
            args,
            persisted["protocol"],
            persisted["source_provenance"],
        )
        == persisted
    )
    with pytest.raises(ValueError, match="changed protocol or source tree"):
        load_manifest(
            manifest_path,
            args,
            {"seeds": [7], "eval_seed": 1702},
            {"files": {"scripts/train.py": {"sha256": "new"}}},
        )


def test_source_provenance_hashes_and_copies_execution_sources(tmp_path):
    provenance = source_provenance(tmp_path, copy_sources=True)

    entry = provenance["files"]["scripts/train.py"]
    snapshot = tmp_path / entry["snapshot"]
    assert (
        snapshot.read_bytes() == (campaign.REPO_ROOT / "scripts/train.py").read_bytes()
    )
    assert len(entry["sha256"]) == 64
    state = CampaignState(
        tmp_path,
        tmp_path / "manifest.json",
        {
            "protocol": _campaign_protocol(),
            "source_provenance": provenance,
            "cells": [],
        },
        False,
    )
    assert_source_provenance_unchanged(state)

    snapshot.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot is missing or changed"):
        assert_source_provenance_unchanged(state)


def test_completed_cell_skips_before_preparation(monkeypatch, tmp_path):
    protocol = _campaign_protocol()
    case = training_cases(protocol)[0]
    marker = tmp_path / "training.json"
    _write_valid_training_artifact(marker, case, protocol)
    manifest = {
        "protocol": protocol,
        "cells": [
            {
                "stage": case.stage,
                "label": case.label,
                "status": "complete",
                "artifact": str(marker),
            }
        ],
    }
    state = CampaignState(tmp_path, tmp_path / "manifest.json", manifest, True)
    spec = CellSpec(case, ("unused",), marker, tmp_path / "stdout.log")

    def fail_if_called():
        raise AssertionError("completed cell was prepared again")

    monkeypatch.setattr(campaign, "assert_source_provenance_unchanged", fail_if_called)
    assert run_cell(state, spec, prepare=fail_if_called)


def test_run_cell_completes_only_after_finalization_and_keeps_flat_history(
    monkeypatch, tmp_path
):
    protocol = _campaign_protocol()
    case = training_cases(protocol)[0]
    marker = tmp_path / "training.json"
    manifest = {"protocol": protocol, "cells": []}
    state = CampaignState(tmp_path, tmp_path / "manifest.json", manifest, False)
    spec = CellSpec(case, ("train",), marker, tmp_path / "stdout.log")
    order = []

    def provenance_check(_state):
        order.append("provenance")

    def child_success(*_args, **_kwargs):
        order.append("subprocess")
        return SimpleNamespace(returncode=0)

    def finalize(_prepared):
        order.append("finalize")
        current = state.manifest["cells"][0]
        assert current["status"] == "running"
        assert campaign.completed_cells(state.manifest) == {}
        _write_valid_training_artifact(marker, case, protocol)
        return {"checkpoint": json.loads(marker.read_text())["checkpoint"]}

    monkeypatch.setattr(
        campaign, "assert_source_provenance_unchanged", provenance_check
    )
    monkeypatch.setattr(campaign.subprocess, "run", child_success)

    assert run_cell(state, spec, prepare=lambda: None, finalize=finalize)
    assert order == ["provenance", "subprocess", "finalize"]
    assert state.manifest["cells"][0]["status"] == "complete"

    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    assert not run_cell(state, spec)
    assert not run_cell(state, spec)
    current = state.manifest["cells"][0]
    assert current["attempt"] == 3
    assert len(current["history"]) == 2
    assert all("history" not in attempt for attempt in current["history"])


def test_report_uses_descriptive_not_significance_language():
    completion = {
        "overall": {"expected": 34, "complete": 34, "failed": 0, "incomplete": 0},
        "by_stage": {
            "speed": {"expected": 10, "complete": 10, "failed": 0, "incomplete": 0},
            "train": {"expected": 6, "complete": 6, "failed": 0, "incomplete": 0},
            "eval": {"expected": 18, "complete": 18, "failed": 0, "incomplete": 0},
        },
    }
    summary = {
        "completion": completion,
        "speed_effects": [],
        "training": [],
        "training_effects_aggregate": [],
        "evaluation": [],
        "evaluation_effects_aggregate": [],
    }

    report = markdown_report(summary, _campaign_protocol(seeds=[1, 2, 3]))

    assert "descriptive paired comparison" in report
    assert "do not establish statistical significance" in report
    assert "statistically controlled" not in report
    assert "[0.5, 1.0]" in report
    assert "uses `1.0`" in report
    assert "| **total** | **34** | **34** | **0** | **0** |" in report
