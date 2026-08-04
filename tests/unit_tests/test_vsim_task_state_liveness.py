"""Task-level state-liveness on the vsim backend.

Fixture-swap of test_task_state_liveness.py (see its docstring for why this
class of test exists — the warp root_states scar).  Asserts the task's
CACHED root_states / rigid-body tensors update in place after step().

Opt-in: runs only under scripts/run_vsim_tests.sh (license + CUDA).
"""

import pytest
import torch

from gym.utils.task_registry import task_registry
from tests.unit_tests.conftest import vsim_guard

pytestmark = pytest.mark.vsim


@pytest.fixture
def vsim_env():
    vsim_guard()
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs("mini_cheetah")
    env_cfg.env.num_envs = 2
    env_cfg.env.episode_length_s = 50
    env_cfg.seed = 0
    train_cfg.seed = 0
    env_cfg.push_robots.toggle = False
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    env = task_registry.make_env(
        "mini_cheetah", env_cfg, device="cuda:0", headless=True, backend="vsim"
    )
    yield env
    env._backend.close()


def test_task_state_liveness_vsim(vsim_env):
    env = vsim_env
    before_root = env.root_states.clone()
    before_rbs = env._rigid_body_state.clone()

    env.dof_pos_target[:] = -env.default_dof_pos

    n_steps = int(0.5 / env.dt)
    for _ in range(n_steps):
        env.step()

    assert not torch.allclose(before_root, env.root_states), (
        "task-cached root_states did not change after stepping — assembled "
        "tensors must be refreshed in step(), never in property getters"
    )
    moved = (env.root_states[:, :3] - before_root[:, :3]).abs().max()
    assert moved > 1e-4, f"root moved only {moved:.2e} m — root_states not live"
    assert not torch.allclose(before_rbs, env._rigid_body_state), (
        "task-cached rigid-body states did not change after stepping"
    )
    assert env.root_states.data_ptr() == env._backend.root_states.data_ptr()
