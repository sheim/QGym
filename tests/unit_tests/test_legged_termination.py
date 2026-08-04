"""Termination-on-base-contact regression test.

Drops a mini_cheetah upside-down with legs straight out so the torso lands on
the ground.  Asserts that contact_forces[:, base_idx, :] becomes non-zero —
catches the bug where MuJoCo's cfrc_ext is left at zero unless
mj_rnePostConstraint / rne_postconstraint is called after the step.
"""

import pytest
import torch

from gym.utils.task_registry import task_registry


def _build_env(device: str):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)

    import gym.envs  # noqa: F401  — registers tasks

    env_cfg, train_cfg = task_registry.get_cfgs("mini_cheetah")
    env_cfg.env.num_envs = 2
    env_cfg.env.episode_length_s = 50  # don't let timeout fire first
    env_cfg.seed = 0
    train_cfg.seed = 0
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)

    return task_registry.make_env_mujoco(
        "mini_cheetah", env_cfg, device=device, headless=True
    )


def _run_drop_and_detect(env) -> bool:
    """Set legs straight, flip body upside-down, step until base hits ground."""
    base_idx = env._backend.find_body_index("base")

    # Legs straight out (PD target = 0 relative to default) so the torso lands
    # on the ground, not on folded legs.
    env.dof_pos_target[:] = -env.default_dof_pos

    # Upside-down: 180° rotation about the x-axis.  Scalar-last [x,y,z,w].
    env._backend.root_states[:, 3:7] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=env.device
    )
    env._backend.root_states[:, 2] = 0.4  # small drop height
    env._backend.root_states[:, 7:13] = 0.0  # zero linear / angular velocity
    env._backend.set_all_root_states()

    # Step long enough for the torso to fall and impact (4 s @ ctrl rate).
    n_steps = int(4.0 / env.dt)
    for _ in range(n_steps):
        env.step()
        f = env.contact_forces[:, base_idx, :].norm(dim=-1)
        if (f > 1.0).any():
            return True
    return False


def test_termination_on_base_contact_cpu():
    env = _build_env(device="cpu")
    assert _run_drop_and_detect(env), (
        "base never registered contact force after upside-down fall — "
        "cfrc_ext is likely not being populated by mj_rnePostConstraint"
    )


@pytest.mark.warp
def test_termination_on_base_contact_warp():
    env = _build_env(device="cuda:0")
    assert _run_drop_and_detect(env), (
        "base never registered contact force after upside-down fall — "
        "cfrc_ext is likely not being populated by mjw.rne_postconstraint"
    )
