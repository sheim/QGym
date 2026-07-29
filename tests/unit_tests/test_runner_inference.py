"""Inference-path regressions shared by policy evaluation and deployment."""

from types import SimpleNamespace

import torch

from learning.runners.datalogging_runner import DataLoggingRunner
from learning.runners.off_policy_runner import OffPolicyRunner
from learning.runners.old_policy_runner import OldPolicyRunner
from learning.runners.on_policy_runner import OnPolicyRunner


class _Actor:
    def act_inference(self, obs):
        return obs

    def forward(self, obs):
        return obs


def _runner_without_noise(runner_type):
    runner = runner_type.__new__(runner_type)
    clean_observation = torch.tensor([[1.0, 2.0]])
    runner.actor_cfg = {"obs": ["state"], "noise": {"state": 100.0}}
    runner.get_obs = lambda _: clean_observation

    def reject_noisy_observation(*_):
        raise AssertionError("inference requested a noisy observation")

    runner.get_noisy_obs = reject_noisy_observation
    actor = _Actor()
    if runner_type is OldPolicyRunner:
        runner.alg = SimpleNamespace(actor_critic=SimpleNamespace(actor=actor))
    else:
        runner.alg = SimpleNamespace(actor=actor)
    if runner_type is OffPolicyRunner:
        runner.alg.action_delta = torch.ones(2)
        runner.alg.action_offset = torch.zeros(2)
        runner.alg.action_min = torch.full((2,), -1.0)
        runner.alg.action_max = torch.full((2,), 1.0)
    return runner, clean_observation


def test_on_policy_inference_does_not_add_observation_noise():
    runner, clean_observation = _runner_without_noise(OnPolicyRunner)
    torch.testing.assert_close(runner.get_inference_actions(), clean_observation)


def test_old_policy_inference_does_not_add_observation_noise():
    runner, clean_observation = _runner_without_noise(OldPolicyRunner)
    torch.testing.assert_close(runner.get_inference_actions(), clean_observation)


def test_datalogging_inference_does_not_add_observation_noise():
    runner, clean_observation = _runner_without_noise(DataLoggingRunner)
    torch.testing.assert_close(runner.get_inference_actions(), clean_observation)


def test_off_policy_inference_does_not_add_observation_noise():
    runner, clean_observation = _runner_without_noise(OffPolicyRunner)
    torch.testing.assert_close(
        runner.get_inference_actions(),
        torch.tanh(clean_observation),
    )
