"""Minimal SimBackend implementation for unit tests.

No IsaacGym, no MuJoCo, no Warp — just torch tensors with trivially-correct
physics (Euler integration of free-fall under gravity for a 1-DOF pendulum).
This lets us test the SimBackend contract and downstream task logic without
any physics-engine dependency.
"""

import torch

from gym.envs.base.sim_backend import SimBackend


class MockBackend(SimBackend):
    """Simulates a 1-DOF pendulum: qddot = -g/L * sin(q) + tau/I.

    Configurable via constructor; defaults match the pendulum.urdf:
      - mass = 1 kg, length = 1 m
      - dt = 0.005 s  (200 Hz)
      - gravity = 9.81 m/s²

    All tensors live on the requested device.
    """

    def __init__(
        self,
        num_envs: int = 4,
        device: str = "cpu",
        mass: float = 1.0,
        length: float = 1.0,
        dt: float = 0.005,
        gravity: float = 9.81,
    ) -> None:
        self._num_envs = num_envs
        self._device = device
        self._mass = mass
        self._length = length
        self._dt = dt
        self._gravity = gravity
        self._inertia = mass * length**2

        # State tensors: [num_envs, 1]
        self._dof_pos_t = torch.zeros(num_envs, 1, device=device)
        self._dof_vel_t = torch.zeros(num_envs, 1, device=device)

        # Flat dof_state view: [num_envs, 1, 2] → view as [num_envs*1, 2]
        self._dof_state_t = torch.zeros(num_envs, 1, 2, device=device)
        # dof_pos / dof_vel are views into dof_state
        self._dof_pos_view = self._dof_state_t[..., 0]  # [num_envs, 1]
        self._dof_vel_view = self._dof_state_t[..., 1]  # [num_envs, 1]

        self._root_states_t = torch.zeros(num_envs, 13, device=device)
        # Default quaternion (identity): w=1 → stored as [qx,qy,qz,qw]
        self._root_states_t[:, 6] = 1.0
        self._contact_forces_t = torch.zeros(num_envs, 2, 3, device=device)

        self._step_count = 0

    # ── SimBackend.device ───────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    # ── Metadata ────────────────────────────────────────────────────────────

    @property
    def num_dof(self) -> int:
        return 1

    @property
    def num_bodies(self) -> int:
        return 2  # base + pole

    @property
    def dof_names(self) -> list:
        return ["theta"]

    @property
    def body_names(self) -> list:
        return ["base", "pole"]

    def find_body_index(self, name: str) -> int:
        return self.body_names.index(name)

    # ── State tensors ────────────────────────────────────────────────────────

    @property
    def dof_state(self) -> torch.Tensor:
        return self._dof_state_t.view(self._num_envs * 1, 2)

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._dof_pos_view

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._dof_vel_view

    @property
    def root_states(self) -> torch.Tensor:
        return self._root_states_t

    @property
    def contact_forces(self) -> torch.Tensor:
        return self._contact_forces_t

    # ── Per-step ─────────────────────────────────────────────────────────────

    def step(self, torques: torch.Tensor) -> None:
        """Semi-implicit Euler integration of the pendulum ODE."""
        q = self._dof_pos_view[:, 0]
        qd = self._dof_vel_view[:, 0]
        tau = torques[:, 0]

        qdd = (-self._gravity / self._length * q.sin() + tau) / self._inertia
        qd_new = qd + self._dt * qdd
        q_new = q + self._dt * qd_new

        self._dof_pos_view[:, 0] = q_new
        self._dof_vel_view[:, 0] = qd_new
        self._step_count += 1

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_dof_state(self, env_ids: torch.Tensor) -> None:
        """No-op: dof_pos/dof_vel are already live — the caller wrote into
        the views before calling this.  A real backend would commit to the
        physics engine here."""

    # ── Setup (not needed for MockBackend — state is ready at __init__) ──────

    def setup(self, cfg, num_envs: int, device: str, task=None) -> None:
        pass  # mock is fully set up in __init__
