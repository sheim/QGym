"""Task-level state-liveness regression test.

LeggedRobot caches backend state tensors ONCE in _init_buffers
(``self.root_states = self._backend.root_states``) and reads the cached
references forever after.  The SimBackend contract requires all state tensors
to be live when step() returns (sim_backend.py).  Backend contract tests
cannot catch a violation because they re-call the properties per assertion —
a backend that refreshes assembled tensors only inside its property getters
passes them while the task trains on frozen observations.

This test steps a full mini_cheetah env and asserts the task's *cached*
root_states / rigid-body states actually update.  Regression for the warp
root_states staleness bug (jt/port e604532 / 2326b71).
"""

import pytest
import torch

from gym.utils.task_registry import task_registry


def _build_env(device: str):
    pytest.importorskip("mujoco")
    if device.startswith("cuda"):
        pytest.importorskip("mujoco_warp")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    import gym.envs  # noqa: F401  — registers tasks

    env_cfg, train_cfg = task_registry.get_cfgs("mini_cheetah")
    env_cfg.env.num_envs = 2
    env_cfg.env.episode_length_s = 50  # no timeout-driven resets during the test
    env_cfg.seed = 0
    train_cfg.seed = 0
    # Pushes write into the task's cached root_states from the task side and
    # would update a frozen tensor, masking exactly the bug we test for.
    env_cfg.push_robots.toggle = False
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)

    return task_registry.make_env_mujoco(
        "mini_cheetah", env_cfg, device=device, headless=True
    )


def _step_and_check_liveness(env):
    before_root = env.root_states.clone()
    before_rbs = env._rigid_body_state.clone()

    # Extend the legs (PD target = straight) so the base definitely moves.
    # This does NOT put the base on the ground, so no contact termination and
    # no _reset_idx writes into root_states that could mask a frozen tensor.
    env.dof_pos_target[:] = -env.default_dof_pos

    n_steps = int(0.5 / env.dt)
    for _ in range(n_steps):
        env.step()

    # NOTE: do not touch env._backend.root_states (the property) before these
    # asserts — getters may refresh as a side effect, hiding the staleness.
    assert not torch.allclose(before_root, env.root_states), (
        "task-cached root_states did not change after stepping — the backend "
        "is refreshing it only in the property getter, not in step() "
        "(SimBackend contract: all tensors live after step() returns)"
    )
    moved = (env.root_states[:, :3] - before_root[:, :3]).abs().max()
    assert moved > 1e-4, (
        f"root position changed by only {moved:.2e} m over "
        f"{n_steps} control steps of leg extension — root_states is not live"
    )
    assert not torch.allclose(before_rbs, env._rigid_body_state), (
        "task-cached rigid-body states did not change after stepping — the "
        "backend is refreshing rigid_body_states only in the property getter"
    )

    # Cached reference must be the backend's own tensor, updated in place
    # (safe to call the property now that liveness has been established).
    assert env.root_states.data_ptr() == env._backend.root_states.data_ptr(), (
        "task-cached root_states and backend root_states are different "
        "tensors — writes from resets would not reach the simulator"
    )


def test_task_state_liveness_cpu():
    env = _build_env(device="cpu")
    _step_and_check_liveness(env)


def test_task_state_liveness_warp():
    pytest.importorskip("mujoco_warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    env = _build_env(device="cuda:0")
    _step_and_check_liveness(env)
