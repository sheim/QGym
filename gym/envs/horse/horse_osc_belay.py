import torch
from isaacgym import gymtorch, gymapi

from gym.envs.horse.horse_osc import HorseOsc


class HorseOscBelay(HorseOsc):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_perturbation_buffers()
        self._init_belay_buffers()

    def _init_belay_buffers(self):
        # per-env belay state buffers
        self.belay_enabled = torch.full(
            (self.num_envs,),
            self.cfg.belay.start_enabled,
            device=self.device,
            dtype=torch.bool,
        )

        # belay force per env
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

        # per-env belay force vector [Fx, Fy, Fz]
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

        # rigid body state tensor: position is columns 0:3
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_state_tensor).view(-1, 13)

        # make sure rb_states are current before setting anchor
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # fixed overhead anchor point for each env
        # starts above the initial front_base position
        self.belay_anchor_pos = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=torch.float,
        )

        initial_body_pos = self.rb_states[self.base_body_ids, 0:3]
        self.belay_anchor_pos[:] = initial_body_pos

        # vertical offset above the initial body position
        self.belay_anchor_pos[:, 2] += self.cfg.belay.anchor_height

        print("[BELAY] anchor pos:", self.belay_anchor_pos[:5])

        self.pending_belay_anchor_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        max_latency = self.cfg.perturbations.latency_steps

        self.action_delay_buffer = torch.zeros(
            self.num_envs,
            max_latency + 1,
            self.num_dof,
            device=self.device,
            dtype=torch.float,
        )

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)

        self._reset_belay_anchor(env_ids)
        self.pending_belay_anchor_reset[env_ids] = True

    def _post_physx_step(self):
        super()._post_physx_step()

        pending_ids = torch.nonzero(self.pending_belay_anchor_reset).flatten()
        if len(pending_ids) > 0:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self._reset_belay_anchor(pending_ids)
            self.pending_belay_anchor_reset[pending_ids] = False

        self._apply_belay_force()
        return None

    def _reset_belay_anchor(self, env_ids):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        front_base_pos = self.rb_states[self.base_body_ids[env_ids], 0:3]

        self.belay_anchor_pos[env_ids, 0] = front_base_pos[:, 0]
        self.belay_anchor_pos[env_ids, 1] = front_base_pos[:, 1]
        self.belay_anchor_pos[env_ids, 2] = (
            front_base_pos[:, 2] + self.cfg.belay.anchor_height
        )

    def _update_belay_force_buffer(self):
        self.belay_force_vec.zero_()

        active = self.belay_enabled.float()
        force_mag = active * self.belay_force_scale * self.belay_force  # (N,)

        if self.cfg.belay.mode == "vertical":
            unit_direction = torch.zeros((self.num_envs, 3), device=self.device)
            unit_direction[:, 2] = 1.0

        elif self.cfg.belay.mode == "horizontal":
            body_pos = self.rb_states[self.base_body_ids, 0:3]
            direction = self.belay_anchor_pos - body_pos
            direction[:, 2] = 0.0
            norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            unit_direction = direction / norm

        elif self.cfg.belay.mode == "anchor":
            body_pos = self.rb_states[self.base_body_ids, 0:3]
            direction = self.belay_anchor_pos - body_pos
            norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            unit_direction = direction / norm

        else:
            raise ValueError(f"Unknown belay mode: {self.cfg.belay.mode}")

        self.belay_force_vec[:] = unit_direction * force_mag.unsqueeze(1)

    def _apply_belay_force(self):
        # map per-env belay force vectors into rigid-body force tensor and apply them
        self.gym.refresh_rigid_body_state_tensor(self.sim)

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

    def draw_belay_debug(self, force_scale=0.01):
        if self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        body_pos = self.rb_states[self.base_body_ids, 0:3]

        active = self.belay_enabled.float()
        force_mag = active * self.belay_force_scale * self.belay_force

        mode = self.cfg.belay.mode

        if mode == "vertical":
            unit_direction = torch.zeros((self.num_envs, 3), device=self.device)
            unit_direction[:, 2] = 1.0

        elif mode == "horizontal":
            direction = self.belay_anchor_pos - body_pos
            direction[:, 2] = 0.0
            norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            unit_direction = direction / norm

        elif mode == "anchor":
            direction = self.belay_anchor_pos - body_pos
            norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            unit_direction = direction / norm

        else:
            raise ValueError(f"Unknown belay mode: {mode}")

        # Red: intended belay direction scaled by force magnitude
        red_end = body_pos + force_scale * force_mag.unsqueeze(1) * unit_direction

        # Green: actual applied force vector
        green_end = body_pos + force_scale * self.belay_force_vec

        red = [1.0, 0.0, 0.0]
        green = [0.0, 1.0, 0.0]

        for i in range(self.num_envs):
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                1,
                [
                    body_pos[i, 0].item(),
                    body_pos[i, 1].item(),
                    body_pos[i, 2].item(),
                    red_end[i, 0].item(),
                    red_end[i, 1].item(),
                    red_end[i, 2].item(),
                ],
                red,
            )

            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                1,
                [
                    body_pos[i, 0].item(),
                    body_pos[i, 1].item(),
                    body_pos[i, 2].item(),
                    green_end[i, 0].item(),
                    green_end[i, 1].item(),
                    green_end[i, 2].item(),
                ],
                green,
            )

    def _compute_torques(self):
        torques = super()._compute_torques()

        if (
            self.cfg.perturbations.enabled
            and self.cfg.perturbations.reduced_torque_enabled
        ):
            torques = torques * self.cfg.perturbations.torque_scale

        return torques

    def _apply_target_latency(self):
        if not (
            self.cfg.perturbations.enabled and self.cfg.perturbations.latency_enabled
        ):
            return

        delay = self.cfg.perturbations.latency_steps

        # shift history buffer back
        self.dof_pos_target_delay_buffer[:, 1:] = self.dof_pos_target_delay_buffer[
            :, :-1
        ].clone()

        # newest target goes in front
        self.dof_pos_target_delay_buffer[:, 0] = self.dof_pos_target.clone()

        # apply delayed target
        self.dof_pos_target = self.dof_pos_target_delay_buffer[:, delay]

    def _pre_compute_torques(self):
        super()._pre_compute_torques()

        if self.cfg.perturbations.enabled:
            if self.cfg.perturbations.latency_enabled:
                self._apply_target_latency()

    def _init_perturbation_buffers(self):
        max_latency = self.cfg.perturbations.max_latency_steps

        self.dof_pos_target_delay_buffer = torch.zeros(
            self.num_envs,
            max_latency + 1,
            self.num_dof,
            device=self.device,
            dtype=torch.float,
        )

    def _pre_decimation_step(self):
        super()._pre_decimation_step()

        if self.cfg.perturbations.enabled:
            if self.cfg.perturbations.latency_enabled:
                self._apply_target_latency()
