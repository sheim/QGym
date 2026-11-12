"""Height-Q playback controller, identity gates, and reset masks.

The controller tests use an analytic policy/critic, so they check candidate
ordering and selection mechanics rather than robot behavior. The identity tests
use the current ``go2trot`` config and a temporary checkpoint file; no simulator
is constructed.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gym.envs  # noqa: F401
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.offline.contracts import (
    ACTION_CONVENTION,
    replay_config_fingerprint,
    sha256_file,
)
from learning.offline.height_q_playback import (
    HeightQPlaybackController,
    apply_auto_reset,
    auto_reset_mask,
    disable_command_resampling,
    restore_command_resampling,
    validate_env_widths,
    validate_playback_contract,
)
from learning.offline.persistent_intervention import (
    sample_candidates,
    sample_rejection_candidates,
)
from scripts.play_height_q import parse_args, resolve_checkpoint

ACTION_DIM = 4


class FakeActor:
    def __init__(self, std=0.5):
        self.std = torch.nn.Parameter(torch.full((ACTION_DIM,), std))


class FakePolicy:
    device = torch.device("cpu")
    obs_names = ("obs",)
    obs_dims = (2,)
    action_names = ("action",)
    action_dim = ACTION_DIM

    def __init__(self, mean=None):
        self.actor = FakeActor()
        self.mean = (
            torch.tensor([0.2, -0.4, 0.1, 0.3])
            if mean is None
            else torch.as_tensor(mean, dtype=torch.float32)
        )

    def mean_actions(self, observation):
        return self.mean.expand(observation.shape[0], -1).clone()


class SumCritic:
    def predict(self, observations, actions):
        return actions.sum(-1)


def _expected_candidates(seed: int, policy: FakePolicy):
    generator = torch.Generator().manual_seed(seed)
    observation = torch.zeros(1, 2)
    mean = policy.mean_actions(observation)
    eps = torch.randn(
        observation.shape[0],
        8,
        ACTION_DIM,
        generator=generator,
    )
    return mean[:, None] + policy.actor.std.detach() * eps


# ── Controllers ──────────────────────────────────────────────────────────


def test_maxq_executes_highest_q_of_eight_stochastic_draws():
    policy = FakePolicy()
    expected = _expected_candidates(7, policy)
    scores = expected.sum(-1)
    expected_index = scores.argmax(-1)
    controller = HeightQPlaybackController(
        policy,
        SumCritic(),
        mode="maxq",
        generator=torch.Generator().manual_seed(7),
    )
    decision = controller.act(torch.zeros(1, 2))

    assert decision.indices.tolist() == expected_index.tolist()
    assert torch.equal(decision.action[0], expected[0, expected_index[0]])
    assert not torch.equal(decision.action[0], policy.mean)
    assert decision.selected_q is None
    assert decision.attempts.tolist() == [8]
    assert not decision.fallback.any()
    assert decision.action.shape == (1, ACTION_DIM)
    assert torch.isfinite(decision.action).all()


def test_rejection_first_feasible_then_maxq_fallback():
    policy = FakePolicy()
    expected = _expected_candidates(11, policy)
    scores = expected.sum(-1)

    permissive = HeightQPlaybackController(
        policy,
        SumCritic(),
        mode="rejection",
        threshold=-10.0,
        generator=torch.Generator().manual_seed(11),
    )
    decision = permissive.act(torch.zeros(1, 2))
    assert decision.indices.tolist() == [0]
    assert torch.equal(decision.action[0], expected[0, 0])
    assert not decision.fallback.any()
    assert decision.attempts.tolist() == [1]
    assert decision.selected_q is not None

    restrictive = HeightQPlaybackController(
        policy,
        SumCritic(),
        mode="rejection",
        threshold=10.0,
        generator=torch.Generator().manual_seed(11),
    )
    fallback = restrictive.act(torch.zeros(1, 2))
    assert fallback.indices.tolist() == scores.argmax(-1).tolist()
    assert fallback.fallback.all()
    assert fallback.attempts.tolist() == [8]


def test_rejection_threshold_must_be_required_and_finite():
    generator = torch.Generator().manual_seed(3)
    with pytest.raises(ValueError, match="requires a threshold"):
        HeightQPlaybackController(
            FakePolicy(), SumCritic(), mode="rejection", generator=generator
        )
    for threshold in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            HeightQPlaybackController(
                FakePolicy(),
                SumCritic(),
                mode="rejection",
                threshold=threshold,
                generator=generator,
            )
    with pytest.raises(ValueError, match="does not accept"):
        HeightQPlaybackController(
            FakePolicy(),
            SumCritic(),
            mode="maxq",
            threshold=0.0,
            generator=generator,
        )
    with pytest.raises(ValueError, match="unknown playback mode"):
        HeightQPlaybackController(
            FakePolicy(), SumCritic(), mode="rollout", generator=generator
        )


def test_sample_eight_order_is_shared_between_modes():
    policy = FakePolicy()
    observation = torch.zeros(1, 2)
    plain = sample_candidates(policy, observation, torch.Generator().manual_seed(19))
    rejection, mean = sample_rejection_candidates(
        policy, observation, torch.Generator().manual_seed(19)
    )
    assert plain.shape == (1, 8, ACTION_DIM)
    assert torch.equal(plain, rejection)
    assert mean.shape == (1, ACTION_DIM)
    assert not torch.equal(plain[:, 0], mean)


# ── CLI ──────────────────────────────────────────────────────────────────


def _base_cli():
    return ["--task", "go2trot", "--artifact", "artifact"]


def test_parser_defaults_are_single_cpu_manual_playback():
    args = parse_args(_base_cli())
    assert args.mode == "maxq"
    assert args.threshold is None
    assert args.device == "cpu"
    assert args.backend == "mujoco"
    assert args.num_envs == 1
    assert args.auto_reset == "off"
    assert args.keyboard is True
    assert args.steps == 0
    assert args.headless is False


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--mode", "rejection"], "--threshold is required"),
        (["--mode", "rejection", "--threshold", "nan"], "must be finite"),
        (["--mode", "maxq", "--threshold", "0"], "only valid"),
        (["--device", "cuda:0"], "only --device cpu"),
        (["--backend", "vsim"], "only --backend mujoco"),
        (["--num_envs", "2"], "exactly one environment"),
        (["--headless"], "requires --steps"),
        (["--steps", "-1"], "nonnegative"),
        (["--load_run", "run"], "require --original_cfg"),
    ],
)
def test_parser_rejects_unsupported_playback(extra, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(_base_cli() + extra)
    assert message in capsys.readouterr().err


def test_rejection_parser_accepts_finite_threshold():
    args = parse_args(_base_cli() + ["--mode", "rejection", "--threshold", "-1.5"])
    assert args.threshold == -1.5


def test_checkpoint_default_is_artifact_relative_to_repo_root():
    from gym import GYM_ROOT_DIR

    default = resolve_checkpoint(None, "logs/run/model_1.pt")
    assert default == Path(GYM_ROOT_DIR) / "logs/run/model_1.pt"
    assert resolve_checkpoint("relative/model.pt", "ignored") == Path(
        "relative/model.pt"
    )


# ── Identity contract ────────────────────────────────────────────────────


def _configs():
    env_cfg, train_cfg = task_registry.get_cfgs("go2trot")
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return env_cfg, train_cfg


def _checkpoint(tmp_path, payload=None):
    path = tmp_path / "model.pt"
    if payload is None:
        payload = {"actor_state_dict": {"dummy": torch.zeros(1)}}
    torch.save(payload, path)
    return path


def _artifact(env_cfg):
    return SimpleNamespace(
        obs_dim=48,
        action_dim=12,
        manifest={
            "dataset": {
                "config_fingerprint": replay_config_fingerprint(class_to_dict(env_cfg))
            }
        },
    )


def _spec(train_cfg, env_cfg, checkpoint):
    return SimpleNamespace(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        actor_cfg=class_to_dict(train_cfg.actor),
        obs_names=tuple(train_cfg.actor.obs),
        obs_dims=(3, 3, 3, 12, 12, 12, 2, 1),
        action_names=tuple(train_cfg.actor.actions),
        action_dim=12,
        noise=class_to_dict(train_cfg.actor.noise),
        scales=class_to_dict(env_cfg.scaling),
        control_dt=env_cfg.control.ctrl_dt,
        n_policy_steps=1,
        episode_length_s=5.0,
        smooth_exploration=False,
        action_convention=ACTION_CONVENTION,
    )


def _contract(tmp_path):
    env_cfg, train_cfg = _configs()
    checkpoint = _checkpoint(tmp_path)
    artifact = _artifact(env_cfg)
    spec = _spec(train_cfg, env_cfg, checkpoint)
    return env_cfg, train_cfg, artifact, spec, checkpoint


def test_valid_contract_returns_fingerprint(tmp_path):
    env_cfg, train_cfg, artifact, spec, checkpoint = _contract(tmp_path)
    fingerprint = validate_playback_contract(
        env_cfg, train_cfg, artifact, spec, checkpoint
    )
    assert fingerprint == artifact.manifest["dataset"]["config_fingerprint"]


def _mutated(base, **changes):
    values = dict(vars(base))
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("fingerprint", "fingerprint"),
        ("hash", "hash"),
        ("missing_checkpoint", "does not exist"),
        ("actor_state", "actor_state_dict"),
        ("privileged", "privileged"),
        ("obs_names", "observation names"),
        ("action_names", "action names"),
        ("normalize", "normalize_obs"),
        ("smooth", "smooth_exploration"),
        ("steps", "n_policy_steps"),
        ("convention", "action convention"),
        ("control_dt", "control dt"),
        ("noise", "observation noise"),
        ("scales", "scales"),
    ],
)
def test_contract_rejects_identity_mismatch(tmp_path, mutation, message):
    env_cfg, train_cfg, artifact, spec, checkpoint = _contract(tmp_path)
    if mutation == "fingerprint":
        artifact = SimpleNamespace(
            obs_dim=artifact.obs_dim,
            action_dim=artifact.action_dim,
            manifest={"dataset": {"config_fingerprint": "0" * 64}},
        )
    elif mutation == "hash":
        spec = _mutated(spec, checkpoint_sha256="0" * 64)
    elif mutation == "missing_checkpoint":
        checkpoint = tmp_path / "missing.pt"
    elif mutation == "actor_state":
        checkpoint = _checkpoint(tmp_path, payload={})
        spec = _mutated(spec, checkpoint_sha256=sha256_file(checkpoint))
    elif mutation == "privileged":
        spec = _mutated(
            spec,
            obs_names=spec.obs_names + ("base_height",),
            obs_dims=spec.obs_dims + (1,),
        )
    elif mutation == "obs_names":
        spec = _mutated(spec, obs_names=("wrong",) + spec.obs_names[1:])
    elif mutation == "action_names":
        spec = _mutated(spec, action_names=("wrong",))
    elif mutation == "normalize":
        spec = _mutated(
            spec,
            actor_cfg={**spec.actor_cfg, "normalize_obs": True},
        )
    elif mutation == "smooth":
        spec = _mutated(
            spec,
            actor_cfg={**spec.actor_cfg, "smooth_exploration": True},
        )
    elif mutation == "steps":
        spec = _mutated(spec, n_policy_steps=2)
    elif mutation == "convention":
        spec = _mutated(spec, action_convention="simulator_units")
    elif mutation == "control_dt":
        spec = _mutated(spec, control_dt=0.05)
    elif mutation == "noise":
        spec = _mutated(
            spec,
            noise={**spec.noise, "base_ang_vel": 0.02},
        )
    elif mutation == "scales":
        scales = copy.deepcopy(spec.scales)
        scales["base_ang_vel"] = 0.5
        spec = _mutated(spec, scales=scales)

    with pytest.raises(ValueError, match=message):
        validate_playback_contract(env_cfg, train_cfg, artifact, spec, checkpoint)


def test_q_width_mismatch_is_rejected(tmp_path):
    env_cfg, train_cfg, artifact, spec, checkpoint = _contract(tmp_path)
    artifact = _mutated(artifact, obs_dim=47)
    with pytest.raises(ValueError, match="observation width"):
        validate_playback_contract(env_cfg, train_cfg, artifact, spec, checkpoint)


def test_env_widths_come_from_live_task_tensors():
    visible = torch.zeros(1, 2)
    fake_env = SimpleNamespace(
        get_states=lambda names: (
            visible if names == ["obs"] else torch.zeros(1, ACTION_DIM)
        )
    )
    validate_env_widths(
        fake_env,
        SimpleNamespace(
            obs_names=("obs",),
            obs_dims=(2,),
            action_names=("action",),
            action_dim=ACTION_DIM,
        ),
    )
    with pytest.raises(ValueError, match="width 2"):
        validate_env_widths(
            fake_env,
            SimpleNamespace(
                obs_names=("obs",),
                obs_dims=(3,),
                action_names=("action",),
                action_dim=ACTION_DIM,
            ),
        )


# ── Reset and manual-command behavior ────────────────────────────────────


class FakeResetEnv:
    def __init__(self, terminated=False, timed_out=False, num_envs=1):
        self.num_envs = num_envs
        self.commands = torch.ones(num_envs, 3)
        self.terminated = torch.full((num_envs,), terminated)
        self.timed_out = torch.full((num_envs,), timed_out)
        self.reset_masks = []

    def _reset_idx(self, mask):
        self.reset_masks.append(mask.clone())


@pytest.mark.parametrize(
    "mode, terminated, timed_out, expected",
    [
        ("termination", True, False, 1),
        ("termination", False, True, 0),
        ("timeout", False, True, 1),
        ("timeout", True, False, 0),
        ("both", True, False, 1),
        ("both", False, True, 1),
        ("both", False, False, 0),
    ],
)
def test_auto_reset_mask_and_zeroing(mode, terminated, timed_out, expected):
    env = FakeResetEnv(terminated=terminated, timed_out=timed_out)
    count = apply_auto_reset(env, mode)
    assert count == expected
    if expected:
        assert len(env.reset_masks) == 1
        assert env.commands.eq(0).all()
    else:
        assert env.reset_masks == []
        assert env.commands.eq(1).all()


def test_auto_reset_off_never_resets_or_zeroes_held_commands():
    env = FakeResetEnv(terminated=True, timed_out=True)
    assert auto_reset_mask(env, "off") is None
    assert apply_auto_reset(env, "off") == 0
    assert env.reset_masks == []
    assert env.commands.eq(1).all()


def test_auto_reset_both_is_the_union_mask():
    env = FakeResetEnv(num_envs=2)
    env.terminated[:] = torch.tensor([True, False])
    env.timed_out[:] = torch.tensor([False, True])
    union = auto_reset_mask(env, "both")
    assert torch.equal(union, torch.tensor([True, True]))


def test_command_resampling_override_is_instance_local_and_restorable():
    calls = []

    class ResamplingEnv:
        def _resample_commands(self, mask):
            calls.append(mask)
            return "original"

    env = ResamplingEnv()
    original = disable_command_resampling(env)
    assert env._resample_commands(torch.ones(1, dtype=torch.bool)) is None
    restore_command_resampling(env, original)
    assert env._resample_commands("mask") == "original"
    assert calls == ["mask"]
