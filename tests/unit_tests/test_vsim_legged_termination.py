"""Termination-on-base-contact regression on the vsim backend.

Full-env variant of test_legged_termination.py with backend="vsim":
drops a mini_cheetah upside-down and asserts base contact force appears —
validates the per-link contact sensors end-to-end through the task layer.

Opt-in: runs only under scripts/run_vsim_tests.sh (license + CUDA).
"""

import pytest
import torch

from gym.utils.task_registry import task_registry
from tests.unit_tests.conftest import vsim_guard


def _build_env():
    import gym.envs  # noqa: F401  — registers tasks

    env_cfg, train_cfg = task_registry.get_cfgs("mini_cheetah")
    env_cfg.env.num_envs = 2
    env_cfg.env.episode_length_s = 50
    env_cfg.seed = 0
    train_cfg.seed = 0
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return task_registry.make_env_mujoco(
        "mini_cheetah", env_cfg, device="cuda:0", headless=True, backend="vsim"
    )


@pytest.fixture
def vsim_env():
    vsim_guard()
    env = _build_env()
    yield env
    env._backend.close()


def test_termination_on_base_contact_vsim(vsim_env):
    env = vsim_env
    base_idx = env._backend.find_body_index("base")

    env.dof_pos_target[:] = -env.default_dof_pos  # legs straight out

    env._backend.root_states[:, 3:7] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device
    )
    env._backend.root_states[:, 2] = 0.4
    env._backend.root_states[:, 7:13] = 0.0
    env._backend.set_all_root_states()

    n_steps = int(4.0 / env.dt)
    for _ in range(n_steps):
        env.step()
        f = env.contact_forces[:, base_idx, :].norm(dim=-1)
        if (f > 1.0).any():
            return
    raise AssertionError(
        "base never registered contact force after upside-down fall — "
        "per-link contact sensors are not reaching contact_forces"
    )
