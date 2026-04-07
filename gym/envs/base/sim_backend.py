from abc import ABC, abstractmethod

import torch


class SimBackend(ABC):
    """Abstract physics backend interface.

    Separates physics-engine specifics from RL task logic, enabling multiple
    backends (IsaacGym/PhysX, MuJocoWarp, plain MuJoCo for Mac/CI).

    Lifecycle
    ---------
    backend = ConcreteBackend(...)
    backend.setup(cfg, num_envs, device, task=self)   # world building
    # state tensors are live after this point
    for _ in training_loop:
        backend.step(torques)           # advance physics
        backend.reset_dof_state(ids)    # commit reset state for specific envs
    backend.close()
    """

    # ── World building ──────────────────────────────────────────────────────

    @abstractmethod
    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        """Load asset, create *num_envs* parallel environments, prepare the
        simulator, and acquire state tensors.

        *task* is the owning FixedRobot/LeggedRobot instance.  Backends that
        need per-environment property callbacks call
        ``task._process_rigid_shape_props``, ``task._process_dof_props``, and
        ``task._process_rigid_body_props`` during setup; those callbacks may
        store results back on the task (e.g. joint limits).  Pass ``None`` if
        no callbacks are needed (e.g. unit tests).
        """

    # ── Metadata (valid after setup) ───────────────────────────────────────

    @property
    @abstractmethod
    def num_dof(self) -> int: ...

    @property
    @abstractmethod
    def num_bodies(self) -> int: ...

    @property
    @abstractmethod
    def dof_names(self) -> list: ...

    @property
    @abstractmethod
    def body_names(self) -> list: ...

    @abstractmethod
    def find_body_index(self, name: str) -> int:
        """Return the rigid-body index for the body with the given name."""

    # ── Live state tensors ─────────────────────────────────────────────────
    # Tensors live on *device*, are writable, and are updated in-place by
    # step().  No explicit refresh is required after step() returns.

    @property
    @abstractmethod
    def dof_state(self) -> torch.Tensor:
        """[num_envs * num_dof, 2] — (pos, vel) pairs."""

    @property
    @abstractmethod
    def dof_pos(self) -> torch.Tensor:
        """[num_envs, num_dof] — view into dof_state[..., 0]."""

    @property
    @abstractmethod
    def dof_vel(self) -> torch.Tensor:
        """[num_envs, num_dof] — view into dof_state[..., 1]."""

    @property
    @abstractmethod
    def root_states(self) -> torch.Tensor:
        """[num_envs, 13] — pos(3) quat(4) lin_vel(3) ang_vel(3)."""

    @property
    @abstractmethod
    def contact_forces(self) -> torch.Tensor:
        """[num_envs, num_bodies, 3]."""

    @property
    def rigid_body_states(self) -> torch.Tensor:
        """[num_envs * num_bodies, 13].  Legged robots only.

        Raises NotImplementedError for backends/configs that don't need it.
        """
        raise NotImplementedError("This backend does not expose rigid_body_states")

    @property
    def penalised_contact_indices(self) -> torch.Tensor:
        """Body indices whose contact forces contribute to the penalty reward.

        Populated from cfg.asset.penalize_contacts_on during setup().
        Raises NotImplementedError for backends that have not implemented it.
        """
        raise NotImplementedError

    @property
    def termination_contact_indices(self) -> torch.Tensor:
        """Body indices whose contact forces trigger episode termination.

        Populated from cfg.asset.terminate_after_contacts_on during setup().
        Raises NotImplementedError for backends that have not implemented it.
        """
        raise NotImplementedError

    # ── Per-step ────────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, torques: torch.Tensor) -> None:
        """Apply *torques* and advance physics by one timestep.

        Args:
            torques: [num_envs, num_dof] generalised forces in full DOF
                     space (zeros for unactuated joints).

        All state tensors are live and up-to-date when this returns.
        """

    # ── Reset ───────────────────────────────────────────────────────────────

    @abstractmethod
    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        """Commit the current dof_pos[env_ids] / dof_vel[env_ids] values to
        the simulator for the specified environments.

        The caller writes the desired state into the tensor views before
        calling this method.
        """

    def reset_root_state(self, env_ids: torch.Tensor) -> None:
        """Commit root_states[env_ids] to the simulator.

        Default no-op — fixed-base robots don't need this.
        """

    def set_all_root_states(self) -> None:
        """Commit root_states for all environments (used by push_robots).

        Default no-op — fixed-base robots don't need this.
        """

    # ── Device / rendering ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def device(self) -> str:
        """The PyTorch device string ('cpu', 'cuda:0', …)."""

    # IsaacGym shims — non-IsaacGym backends leave these as None.
    # BaseTask.gym / BaseTask.sim forward here for LeggedRobot compatibility.
    gym = None
    sim = None

    def render(self, sync_frame_time: bool = True) -> None:
        pass

    def close(self) -> None:
        pass
