import torch

from gym.envs.base.sim_backend import SimBackend
from gym.envs.base.task_skeleton import TaskSkeleton


class BaseTask(TaskSkeleton):
    def __init__(
        self,
        backend: SimBackend,
        cfg,
        device: str,
        headless: bool,
    ) -> None:
        self._backend = backend
        self.headless = headless
        self.num_actuators = cfg.env.num_actuators

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        super().__init__(num_envs=cfg.env.num_envs, device=device)

        # Kept for backward compat (checked by check_exit and legacy code).
        self.enable_viewer_sync = True
        self.exit = False

    # ── Physics-engine shims ────────────────────────────────────────────────
    # LeggedRobot still calls self.gym.* and self.sim.* directly while its
    # migration to SimBackend is pending (Phase 3).  These properties route
    # those calls through the backend without changing any call sites.
    # TODO Phase 3: remove once LeggedRobot is fully migrated.

    @property
    def gym(self):
        return self._backend.gym

    @property
    def sim(self):
        return self._backend.sim

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self, sync_frame_time: bool = True) -> None:
        if not self.headless:
            self._backend.render(sync_frame_time)
