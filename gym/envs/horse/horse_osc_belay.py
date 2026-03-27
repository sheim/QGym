import torch
from isaacgym import gymtorch, gymapi

from gym.envs.horse.horse_osc import HorseOsc


BELAY_MASS_KG = 20.0
BELAY_FORCE_N = BELAY_MASS_KG * 9.81


class HorseOscBelay(HorseOsc):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_belay_buffers()

    def _init_belay_buffers(self):
        # per-env belay state buffers
        self.belay_enabled = torch.full(
            (self.num_envs,),
            self.cfg.belay.start_enabled,
            device=self.device,
            dtype=torch.bool,
        )

        # one scalar upward force per env
        self.belay_force = torch.full(
            (self.num_envs,),
            self.cfg.belay.mass_kg * 9.81,
            device=self.device,
            dtype=torch.float,
        )

        # normalized command in [0, 1]
        self.belay_force_scale = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )

        # per-env generalized force vector
        # [Fx, Fy, Fz] for each env
        self.belay_force_vec = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=torch.float,
        )

        # rigid-body force tensors
        self.num_sim_bodies = self.gym.get_sim_rigid_body_count(self.sim)

        self.rb_forces = torch.zeros(
            (self.num_sim_bodies, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.rb_torques = torch.zeros_like(self.rb_forces)

        # determine rigid body to pull on
        self.belay_body_name = "front_base"

        self.front_base_local_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0],
            self.actor_handles[0],
            self.belay_body_name,
        )

        print(
            f"[BELAY] local body index for '{self.belay_body_name}' = "
            f"{self.front_base_local_id}"
        )

        self.base_body_ids = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        for i in range(self.num_envs):
            sim_body_id = self.gym.get_actor_rigid_body_index(
                self.envs[i],
                self.actor_handles[i],
                self.front_base_local_id,
                gymapi.DOMAIN_SIM,
            )
            self.base_body_ids[i] = sim_body_id

        print("[BELAY] sim body ids:", self.base_body_ids[:5])

    def _post_physx_step(self):
        super()._post_physx_step()
        self._apply_belay_force()
        return None

    def _update_belay_force_buffer(self):
        # update the per-env belay force vector.
        self.belay_force_vec.zero_()

        active = self.belay_enabled.float()
        scaled_force = active * self.belay_force_scale * self.belay_force

        # upward force only
        self.belay_force_vec[:, 2] = scaled_force

    def _apply_belay_force(self):
        # map per-env belay force vectors into rigid-body force tensor and apply them
        self.rb_forces.zero_()
        self.rb_torques.zero_()

        self._update_belay_force_buffer()

        # per-env body force into global sim rigid body tensor
        self.rb_forces[self.base_body_ids] = self.belay_force_vec

        if torch.any(self.belay_enabled):
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                gymtorch.unwrap_tensor(self.rb_torques),
                gymapi.ENV_SPACE,
            )

    def toggle_belay(self, env_ids=None):
        # toggle belay on/off.
        if env_ids is None:
            self.belay_enabled = ~self.belay_enabled
            print(
                f"all envs toggled | any_on = {bool(torch.any(self.belay_enabled))} | "
                f"force = {self.cfg.belay.mass_kg * 9.81:.1f} N"
            )
        else:
            self.belay_enabled[env_ids] = ~self.belay_enabled[env_ids]
            print(
                f"[BELAY] toggled env_ids={env_ids.tolist()} | "
                f"force = {self.cfg.belay.mass_kg * 9.81:.1f} N"
            )

    def set_belay(self, enabled: bool, env_ids=None):
        # set belay state
        if env_ids is None:
            self.belay_enabled[:] = enabled
        else:
            self.belay_enabled[env_ids] = enabled

    def debug_print_belay(self, env_id=0):
        print(
            f"env={env_id} | enabled={bool(self.belay_enabled[env_id])} | "
            f"body_id={self.base_body_ids[env_id].item()} | "
            f"force_vec={self.belay_force_vec[env_id].detach().cpu().numpy()}"
        )
