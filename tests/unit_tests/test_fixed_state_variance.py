"""Analytic pairing/return oracles, independent of robot learning quality."""

from types import SimpleNamespace

import torch

from learning.offline.fixed_state_variance import (
    WarpInsertionSnapshot,
    accumulate_return,
    expand_states,
    install_paired_environment_noise,
    lane_source_ids,
    summarize_returns,
)


def test_restore_rebound_controller_flags_counters_and_rng():
    # The task rebinds action/history tensors every step: restoring stale tensor
    # handles would silently leave the next continuation's controller changed.
    env = SimpleNamespace(
        dof_pos_history=torch.arange(12).reshape(2, 6).float(),
        terminated=torch.tensor([False, True]),
        common_step_counter=100,
        domain_randomizer=SimpleNamespace(
            _generators={"gain": torch.Generator().manual_seed(4)}
        ),
    )
    snapshot = WarpInsertionSnapshot.__new__(WarpInsertionSnapshot)
    snapshot.env = env
    snapshot.entries = [
        (env, k, False, getattr(env, k).clone())
        for k in ("dof_pos_history", "terminated")
    ]
    snapshot.counter = 100
    snapshot.dr_rng = {"gain": env.domain_randomizer._generators["gain"].get_state()}
    for _ in range(3):
        env.dof_pos_history = torch.full((2, 6), -99.0)
        env.terminated.fill_(False)
        env.common_step_counter += 500
        torch.rand(4, generator=env.domain_randomizer._generators["gain"])
        snapshot.restore()
        snapshot.assert_restored()
        assert env.terminated.tolist() == [False, True]
        assert torch.equal(env.dof_pos_history, torch.arange(12).reshape(2, 6).float())
        assert torch.equal(
            env.domain_randomizer._generators["gain"].get_state(),
            snapshot.dr_rng["gain"],
        )


def test_state_major_replication_and_shared_noise():
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(expand_states(states, 3).reshape(2, 3, 2)[:, 0], states)
    assert torch.equal(expand_states(states, 3)[0], expand_states(states, 3)[2])
    assert not torch.equal(expand_states(states, 3)[2], expand_states(states, 3)[3])


def test_covariance_and_paired_difference_variance():
    # Arbitrarily large shared continuation noise cancels exactly in contrasts.
    common = torch.arange(8 * 16 * 2).reshape(8, 16, 2, 1).double()
    x = common + torch.tensor([0.0, 1.0, 3.0])
    stats = summarize_returns(x)
    assert torch.all(stats["batch_estimator_variance"] > 0)
    assert torch.count_nonzero(stats["paired_batch_estimator_variance"]) == 0
    assert torch.equal(
        stats["paired_mean"], torch.tensor([[0.0, 1.0, 3.0]]).double().expand(2, 3)
    )
    assert torch.all(stats["batch_ranking_agreement_by_state_pair"] == 1)
    assert torch.allclose(
        stats["within_batch_action_variance"], torch.full((8, 2), 7 / 3).double()
    )


def test_terminal_observed_reward_then_absorbing_tail():
    total = torch.zeros(2, dtype=torch.float64)
    alive = torch.ones(2, dtype=torch.bool)
    total, alive = accumulate_return(
        total, alive, torch.tensor([0.0, -1.0]), torch.tensor([True, False]), 0, 0.99
    )
    assert torch.allclose(total, torch.tensor([-99.0, -1.0]).double())
    total, alive = accumulate_return(
        total, alive, torch.tensor([-1.0, 0.0]), torch.tensor([False, True]), 1, 0.99
    )
    assert torch.allclose(total, torch.tensor([-99.0, -99.01]).double())
    assert not alive.any()


def test_command_and_push_pairing_with_fake_env():
    env = SimpleNamespace(
        num_envs=6,
        device="cpu",
        commands=torch.zeros(6, 3),
        root_states=torch.zeros(6, 13),
    )
    env.cfg = SimpleNamespace(
        push_robots=SimpleNamespace(push_box_dims=[1.0, 2.0, 3.0], max_push_vel_xy=1.0)
    )
    env._resample_commands = lambda mask: env.commands.copy_(torch.randn(6, 3))
    commits = []
    env._backend = SimpleNamespace(set_all_root_states=lambda: commits.append(1))
    install_paired_environment_noise(env, 3)
    env._resample_commands(torch.ones(6, dtype=torch.bool))
    assert torch.equal(env.commands[0], env.commands[2])
    assert not torch.equal(env.commands[0], env.commands[3])
    env._push_robots()
    assert torch.equal(env.root_states[0], env.root_states[2])
    assert not torch.equal(env.root_states[0], env.root_states[3])
    assert commits == [1]


def test_export_selection_maps_flattened_history_model_and_candidates():
    n = 1024
    indices = torch.tensor([91, 4, 22, 17, 6, 99, 23, 55, 71, 3, 0, 8, 10, 76, 45, 81])
    ids = lane_source_ids(indices)
    worlds = torch.arange(n)
    owner = SimpleNamespace(
        flattened=worlds[:, None].expand(n, 24).reshape(n * 12, 2).clone(),
        model=worlds[:, None].expand(n, 17).clone(),
        history=worlds[:, None].expand(n, 36).clone(),
    )
    snapshot = WarpInsertionSnapshot.__new__(WarpInsertionSnapshot)
    snapshot.env = SimpleNamespace(num_envs=n)
    snapshot.entries = [(owner, k, False, v) for k, v in vars(owner).items()]
    snapshot.source_ids = worlds
    snapshot.counter = 100
    snapshot.dr_rng = {}
    exported = snapshot.export()
    snapshot.load_export(exported, ids)
    for _, _, _, value in snapshot.entries:
        assert torch.equal(
            value.reshape(8, 16, 8, -1)[..., 0],
            (indices[:, None] * 8 + torch.arange(8)).expand(8, 16, 8),
        )
    # This is also the collector's call/lane -> repeat reshape contract.
    x = torch.arange(8 * 8 * 16 * 8).reshape(8, 8, 16, 8)
    assert x.reshape(64, 16, 8)[13, 4, 7] == x[1, 5, 4, 7]


def test_vectorized_lane_noise_marginals_no_clipping_and_reproducibility():
    from learning.offline.contracts import sample_observation_noise

    def sample(seed):
        return sample_observation_noise(
            ("x", "off"),
            (128, 1),
            {"x": 2.0},
            {"x": 4.0},
            128,
            "cpu",
            torch.Generator().manual_seed(seed),
        )

    noise = sample(6400001)
    expected = torch.randn(128, 129, generator=torch.Generator().manual_seed(6400001))
    assert torch.equal(noise[:, :128], expected[:, :128] * 0.5)
    assert noise[:, :128].abs().max() > 1.5  # Gaussian tails are not clipped.
    assert torch.count_nonzero(noise[:, -1]) == 0
    paired = expand_states(noise, 8).reshape(8, 16, 8, 129)
    assert torch.equal(paired, paired[:, :, :1].expand_as(paired))
    assert not torch.equal(paired[0], paired[1])
    assert not torch.equal(noise, sample(6400011))
    assert torch.equal(noise, sample(6400001))
