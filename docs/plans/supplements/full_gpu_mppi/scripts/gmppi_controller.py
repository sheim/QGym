"""Policy-guided MPPI over Q2 physics and a fixed-policy height Q.

Objective (minimised), for a candidate action sequence u_0..u_{T-1} in policy
units, gamma = the Q artifact's discount, r_h the artifact's height reward
(0 inside the band, -1 outside), F = |failure value| of the artifact:

    C(u) = sum_{t<T, still active}  -gamma^t r_h(s_t)
         + gamma^tterm F            if the candidate falls at step tterm
         - gamma^T Q(s_T, pi(s_T))  if still active

The frozen nominal policy participates twice: it supplies the proposal
sequences (closed-loop rollouts in their own Warp worlds) and it supplies the
terminal action whose Q value bootstraps the truncated horizon. ``pytorch_mppi``
samples, scores and reweights exactly as upstream; the policy enters only
through its documented ``specific_action_sampler`` hook.
"""

from __future__ import annotations

import contextlib
import os

import torch
from pytorch_mppi import MPPI
from pytorch_mppi.mppi import SpecificActionSampler

from gmppi_adapter import PredictionTask, capture_state

NVTX = os.environ.get("GMPPI_NVTX", "0") == "1"


@contextlib.contextmanager
def nvtx_range(name):
    if NVTX:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if NVTX:
            torch.cuda.nvtx.range_pop()


class InstrumentedMPPI(MPPI):
    """Upstream MPPI with NVTX labels only; no sampling or update change."""

    def _compute_perturbed_action_and_noise(self):
        with nvtx_range("mppi/candidate_generation"):
            return super()._compute_perturbed_action_and_noise()

    def _compute_weighting(self, cost_total):
        with nvtx_range("mppi/update"):
            return super()._compute_weighting(cost_total)


class PolicySequenceSampler(SpecificActionSampler):
    """Closed-loop nominal-policy action sequences as MPPI candidates."""

    def __init__(self, policy_task, policy, generator, horizon, num_sequences):
        super().__init__()
        self.policy_task = policy_task
        self.policy = policy
        self.generator = generator
        self.horizon = int(horizon)
        self.num_sequences = int(num_sequences)
        self.action_dim = int(policy.action_dim)
        self._pending_state = None
        self.sequences = None
        self.sync_free = True

    def prepare(self, state):
        self._pending_state = state

    def sample_trajectories(self, state, info):
        """Roll the frozen policy forward T steps in its own branch worlds."""
        with nvtx_range("policy/rollout"):
            self.policy_task.scatter(self._pending_state, sync_free=self.sync_free)
            sequences = torch.empty(
                self.num_sequences,
                self.horizon,
                self.action_dim,
                device=self.policy_task.device,
                dtype=torch.float32,
            )
            for t in range(self.horizon):
                observation = self.policy_task.observation()
                action = self.policy.sample_actions(observation, self.generator)
                sequences[:, t] = action
                self.policy_task.set_action_targets(action)
                self.policy_task.step()
            self.sequences = sequences
        return sequences


class GPUHeightReward:
    """The artifact's own reward channel, evaluated on predicted physics state."""

    def __init__(self, artifact):
        spec = artifact.manifest["reward_spec"]
        self.reference = float(spec["reference_height_m"])
        self.bound = float(spec["bound_m"])
        self.inside = float(spec["inside_reward"])
        self.outside = float(spec["outside_reward"])

    def __call__(self, height):
        inside = (height - self.reference).abs() <= self.bound
        return torch.where(
            inside,
            torch.full_like(height, self.inside),
            torch.full_like(height, self.outside),
        )


class GPUMPPIController:
    """Full GPU controller: candidates -> physics -> policy -> Q -> selection."""

    def __init__(
        self,
        artifact,
        policy,
        eval_task,
        *,
        num_samples=256,
        horizon=8,
        num_policy_sequences=8,
        noise_scale=0.25,
        lambda_=0.5,
        u_bound=2.0,
        seed=0,
        device="cuda:0",
        sync_free_commit=True,
    ):
        if num_policy_sequences < 1:
            raise ValueError("need at least one nominal-policy sequence")
        self.device = device
        self.policy = policy
        self.artifact = artifact
        self.num_samples = int(num_samples)
        self.horizon = int(horizon)
        self.num_policy_sequences = int(num_policy_sequences)
        self.sync_free_commit = bool(sync_free_commit)
        self.action_dim = int(artifact.action_dim)
        self.obs_dim = int(artifact.obs_dim)
        self.gamma = float(artifact.manifest["gamma"])
        self.failure_magnitude = abs(float(artifact.manifest["failure_value"]))
        self.reward = GPUHeightReward(artifact)

        self.pred = PredictionTask(self.num_samples, seed=seed + 1, device=device)
        self.policy_task = PredictionTask(
            self.num_policy_sequences, seed=seed + 2, device=device
        )
        self.pred.copy_physics_from(eval_task)
        self.policy_task.copy_physics_from(eval_task)

        self.q = artifact.q
        self.q_mean = artifact.obs_mean.to(device)
        self.q_std = artifact.obs_std.to(device)

        self.generator = torch.Generator(device=device).manual_seed(seed + 3)
        self.sampler = PolicySequenceSampler(
            self.policy_task,
            policy,
            self.generator,
            self.horizon,
            self.num_policy_sequences,
        )
        self.sampler.sync_free = self.sync_free_commit

        self.sigma = torch.full(
            (self.action_dim,), float(noise_scale), device=device, dtype=torch.float32
        )
        self._active = torch.ones(self.num_samples, dtype=torch.bool, device=device)
        self._terminated_now = torch.zeros_like(self._active)
        self._policy_prior = torch.zeros(
            self.horizon, self.action_dim, device=device, dtype=torch.float32
        )
        self._started = False

        self.mppi = InstrumentedMPPI(
            self._dynamics,
            self._running_cost,
            nx=self.obs_dim,
            noise_sigma=torch.diag(self.sigma),
            num_samples=self.num_samples,
            horizon=self.horizon,
            device=device,
            terminal_state_cost=self._terminal_cost,
            lambda_=float(lambda_),
            u_min=-float(u_bound) * torch.ones(self.action_dim, device=device),
            u_max=float(u_bound) * torch.ones(self.action_dim, device=device),
            u_init=torch.zeros(self.action_dim, device=device),
            U_init=torch.zeros(
                self.horizon, self.action_dim, device=device, dtype=torch.float32
            ),
            step_dependent_dynamics=True,
            specific_action_sampler=self.sampler,
        )

    # ── MPPI callbacks (device only) ────────────────────────────────────────
    def _dynamics(self, state, u, t):
        with nvtx_range("physics/step"):
            self.pred.set_action_targets(u)
            self.pred.step()
            terminated = self.pred.terminated()
            self._active &= ~terminated
            self._terminated_now = terminated
            observation = self.pred.observation()
        return observation

    def _running_cost(self, state, u, t):
        with nvtx_range("cost/height_reward"):
            reward = self.reward(self.pred.base_height())
        discount = self.gamma**t
        zero = torch.zeros_like(reward)
        running = torch.where(self._active, -discount * reward, zero)
        failure = torch.full_like(reward, discount * self.failure_magnitude)
        return running + torch.where(self._terminated_now, failure, zero)

    def _terminal_cost(self, states, actions):
        with nvtx_range("terminal/policy+Q"):
            observation = states[0, :, -1]
            terminal_action = self.policy.mean_actions(observation)
            value = self.q(
                (observation - self.q_mean) / self.q_std, terminal_action
            ).squeeze(-1)
            zero = torch.zeros_like(value)
            return torch.where(self._active, -(self.gamma**self.horizon) * value, zero)

    # ── boundary ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def action(self, observation, state_source):
        """``controller.action(observation_on_gpu) -> action_on_gpu``.

        ``state_source`` is the evaluation environment the observation belongs
        to; its task buffers and solver warm start are branched, never advanced.
        """
        captured = capture_state(state_source)
        self.pred.scatter(captured, sync_free=self.sync_free_commit)
        self.sampler.prepare(captured)
        self._active.fill_(True)
        self._terminated_now.zero_()
        nominal = self.policy.mean_actions(observation.reshape(1, -1))
        self.mppi.u_init = nominal[0]
        if not self._started:
            # First decision: the nominal sequence starts at the policy's own
            # action instead of at the library's zero prior.
            self._policy_prior.copy_(nominal[0].expand_as(self._policy_prior))
            self.mppi.U = self._policy_prior.clone()
            self._started = True
        with nvtx_range("controller/decision"):
            return self.mppi.command(observation.reshape(-1))

    def reset(self):
        self._started = False
        self.mppi.reset()
