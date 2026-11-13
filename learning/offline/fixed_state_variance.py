"""Fixed-state paired Monte Carlo tools; experimental Go2Trot/Warp adapter.

Bypasses task reset (which clamps qpos). Native integration inputs follow
mujoco_warp.State.INTEGRATION. Derived task/backend buffers are also copied,
including pre-integration contact forces, without another forward. Solver
scratch is recomputed by the next forward; fidelity must be measured.
"""

import itertools

import torch


def expand_states(value, candidates):
    """World order is state-major, candidate-minor."""
    return value.repeat_interleave(candidates, dim=0)


def lane_source_ids(state_ids, lanes=8, candidates=8):
    """Original [state,candidate] worlds -> [lane,state,candidate] worlds."""
    return (
        (
            state_ids[:, None] * candidates
            + torch.arange(candidates, device=state_ids.device)
        )
        .flatten()
        .repeat(lanes)
    )


class WarpInsertionSnapshot:
    """Same-environment snapshot with optional source-world replication.

    Only supports Go2Trot's present Euler, nonsleeping Warp path. No reset occurs
    in these continuations. DR generator states are retained even though those
    generators are unused until an episode reset.
    """

    integration_fields = (
        "time",
        "qpos",
        "qvel",
        "act",
        "history",
        "qacc_warmstart",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "eq_active",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    )

    def __init__(self, env, source_ids=None):
        import mujoco_warp as mjw
        import warp as wp

        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend
        from gym.envs.go2.go2trot import Go2Trot

        if type(env) is not Go2Trot or type(env._backend) is not MuJocoWarpBackend:
            raise ValueError("snapshot adapter requires Go2Trot/MuJocoWarpBackend")
        if env._backend._m.opt.enableflags & mjw.EnableBit.SLEEP:
            raise ValueError("sleeping worlds are not supported")
        self.env = env
        self.entries = []
        n = env.num_envs
        if n != 1024:
            raise ValueError("experimental snapshot inventory requires 1024 worlds")
        if source_ids is None:
            source_ids = torch.arange(n, device=env.device)
        self.source_ids = source_ids.clone()
        self.counter = env.common_step_counter
        self.dr_rng = {
            k: v.get_state().clone()
            for k, v in env.domain_randomizer._generators.items()
        }

        def add(owner, key, value, native=False):
            # Flattened public body/DOF views still have a world-leading view.
            shaped = value.reshape(n, -1) if value.shape[0] != n else value
            saved = shaped.index_select(0, source_ids).reshape(value.shape).clone()
            self.entries.append((owner, key, native, saved))

        for key in self.integration_fields:
            value = wp.to_torch(getattr(env._backend._d, key))
            add(env._backend._d, key, value, True)
        # All batched model properties, including set_const outputs. Shared
        # model fields are immutable during the experiment.
        for key, value in vars(env._backend._m).items():
            if isinstance(value, wp.array) and value.ndim and value.shape[0] == n:
                add(env._backend._m, key, wp.to_torch(value), True)
        for owner in (env._backend, env, env.domain_randomizer):
            for key, value in vars(owner).items():
                if isinstance(value, torch.Tensor) and value.ndim:
                    if value.shape[0] == n or (
                        owner is env and key in ("dof_state", "_rigid_body_state")
                    ):
                        add(owner, key, value)
                elif owner is env.domain_randomizer and isinstance(value, dict):
                    for subkey, tensor in value.items():
                        if isinstance(tensor, torch.Tensor):
                            add(value, subkey, tensor)

    def restore(self):
        for owner, key, native, saved in self.entries:
            live = owner[key] if isinstance(owner, dict) else getattr(owner, key)
            if native:
                import warp as wp

                live = wp.to_torch(live)
            live.copy_(saved)
        self.env.common_step_counter = self.counter
        for key, state in self.dr_rng.items():
            self.env.domain_randomizer._generators[key].set_state(state)

    def load_export(self, exported, source_ids=None):
        """Load all saved fields, optionally selecting/replicating saved worlds.

        Flattened public DOF/body views are indexed by world, never by joint.
        Native model constants and controller/DR history use the identical map.
        """
        n = self.env.num_envs
        if source_ids is None:
            source_ids = torch.arange(n)
        if source_ids.shape != (n,) or not ((source_ids >= 0) & (source_ids < n)).all():
            raise ValueError("source_ids must map every destination to a saved world")
        if len(self.entries) != len(exported["entries"]):
            raise ValueError("snapshot inventory changed")
        restored = []
        for (owner, key, native, value), entry in zip(
            self.entries, exported["entries"]
        ):
            if (type(owner).__name__, key, native) != (
                entry["owner"],
                entry["field"],
                entry["native"],
            ) or entry["value"].shape != value.shape:
                raise ValueError("snapshot inventory changed")
            mapped = entry["value"].reshape(n, -1).index_select(0, source_ids.cpu())
            restored.append(
                (owner, key, native, mapped.reshape(value.shape).to(value.device))
            )
        self.entries = restored
        self.source_ids = exported["source_ids"][source_ids.cpu()].clone()
        self.counter = exported["common_step_counter"]
        self.dr_rng = exported["dr_rng"]

    def assert_restored(self):
        for owner, key, native, saved in self.entries:
            live = owner[key] if isinstance(owner, dict) else getattr(owner, key)
            if native:
                import warp as wp

                live = wp.to_torch(live)
            if not torch.equal(live, saved):
                raise RuntimeError(f"snapshot restore mismatch: {key}")
        if self.env.common_step_counter != self.counter:
            raise RuntimeError("common step counter was not restored")

    def export(self):
        return {
            "entries": [
                {
                    "owner": type(owner).__name__,
                    "field": key,
                    "native": native,
                    "value": saved.cpu(),
                }
                for owner, key, native, saved in self.entries
            ],
            "common_step_counter": self.counter,
            "dr_rng": self.dr_rng,
            "source_ids": self.source_ids.cpu(),
        }


def install_paired_environment_noise(env, candidates):
    """Preserve command/push marginals while sharing draws within each state."""
    from gym.utils.sampling import torch_rand_float

    original_commands = env._resample_commands

    def commands(mask):
        original_commands(mask)
        env.commands.copy_(expand_states(env.commands[::candidates], candidates))

    def pushes():
        states = env.num_envs // candidates
        dims = torch.tensor(env.cfg.push_robots.push_box_dims, device=env.device) / 2
        r = torch.cat(
            [
                torch_rand_float(-dims[j], dims[j], (states, 1), device=env.device)
                for j in range(3)
            ],
            dim=1,
        )
        vmax = env.cfg.push_robots.max_push_vel_xy
        velocity = torch_rand_float(-vmax, vmax, (states, 3), device=env.device)
        velocity[:, 2] = 0
        env.root_states[:, 7:10] += expand_states(velocity, candidates)
        env.root_states[:, 10:13] += expand_states(
            torch.cross(r, velocity, dim=1), candidates
        )
        env._backend.set_all_root_states()

    env._resample_commands = commands
    env._push_robots = pushes


def summarize_returns(returns):
    """Empirical repeated-batch variance, retaining candidate covariance.

    Input axes [batch, repeat, state, candidate]. No noise-ceiling estimate or
    independent-candidate variance subtraction is performed.
    """
    if returns.ndim != 4 or returns.shape[0] < 2 or returns.shape[1] < 2:
        raise ValueError("need [batch>=2, repeat>=2, state, candidate] returns")
    x = returns.double()
    means = x.mean(1)
    paired = x - x[..., :1]
    paired_means = paired.mean(1)
    pairs = list(itertools.combinations(range(x.shape[-1]), 2))
    signs = torch.stack(
        [torch.sign(means[..., a] - means[..., b]) for a, b in pairs], -1
    )
    agreements = []
    for a, b in itertools.combinations(range(x.shape[0]), 2):
        agreements.append(
            torch.where(
                (signs[a] == 0) | (signs[b] == 0),
                0.5,
                (signs[a] == signs[b]).double(),
            )
        )
    centered = means - means.mean(0)
    covariance = torch.einsum("bsc,bsd->scd", centered, centered) / (x.shape[0] - 1)
    return {
        "batch_means": means,
        "batch_estimator_variance": means.var(0),
        "within_batch_action_variance": means.var(-1),
        "within_batch_action_range": means.amax(-1) - means.amin(-1),
        "paired_mean": paired.mean((0, 1)),
        "paired_return_variance": paired.flatten(0, 1).var(0),
        "paired_batch_estimator_variance": paired_means.var(0),
        "batch_candidate_covariance": covariance,
        "batch_ranking_agreement_by_state_pair": torch.stack(agreements).mean(0),
        "batch_ranking_tie_fraction": (signs == 0).double().mean(),
        "batch_winner_agreement": torch.stack(
            [
                (means[a].argmax(-1) == means[b].argmax(-1)).double()
                for a, b in itertools.combinations(range(x.shape[0]), 2)
            ]
        ).mean(0),
    }


def accumulate_return(total, alive, reward, terminated, step, gamma):
    """Observed terminal reward, then exact absorbing -1 tail."""
    terminal_now = alive & terminated
    total = total + gamma**step * reward * alive
    total = total - gamma ** (step + 1) / (1 - gamma) * terminal_now
    return total, alive & ~terminal_now


def accumulate_return_zero_after_terminal(
    total, alive, reward, terminated, step, gamma
):
    """Discounted episodic return; a true termination ends the return there.

    There is no absorbing tail: the reward term is already masked to zero on a
    terminal step, so the return after a fall is exactly the discounted sum of
    the rewards observed before it.
    """
    terminal_now = alive & terminated
    total = total + gamma**step * reward * alive
    return total, alive & ~terminal_now
