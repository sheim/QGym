import torch
from isaacgym.torch_utils import torch_rand_float

from gym.envs.horse.horse import Horse

from isaacgym import gymtorch

HORSE_WEIGHT = 536.38 * 9.81  # Weight of horse in Newtons
BASE_HEIGHT_REF = 1.3


class HorseOsc(Horse):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        BASE = 0

        RH = dict(haa=1, hfe=2, kfe=3, pfe=4, pastern=5)
        LH = dict(haa=6, hfe=7, kfe=8, pfe=9, pastern=10)
        RF = dict(haa=11, hfe=12, kfe=13, pfe=14, pastern=15)
        LF = dict(haa=16, hfe=17, kfe=18, pfe=19, pastern=20)

        # tensors for vectorized ops
        self.idx = {
            "base": torch.tensor([BASE], device=self.device),
            # all legs by joint type (RH, LH, RF, LF order)
            "haa": torch.tensor(
                [RH["haa"], LH["haa"], RF["haa"], LF["haa"]], device=self.device
            ),
            "hfe": torch.tensor(
                [RH["hfe"], LH["hfe"], RF["hfe"], LF["hfe"]], device=self.device
            ),
            "kfe": torch.tensor(
                [RH["kfe"], LH["kfe"], RF["kfe"], LF["kfe"]], device=self.device
            ),
            "pfe": torch.tensor(
                [RH["pfe"], LH["pfe"], RF["pfe"], LF["pfe"]], device=self.device
            ),
            "pastern": torch.tensor(
                [RH["pastern"], LH["pastern"], RF["pastern"], LF["pastern"]],
                device=self.device,
            ),
            # hind vs front splits (HIND = RH, LH / FRONT = RF, LF)
            "hind_haa": torch.tensor([RH["haa"], LH["haa"]], device=self.device),
            "hind_hfe": torch.tensor([RH["hfe"], LH["hfe"]], device=self.device),
            "hind_kfe": torch.tensor([RH["kfe"], LH["kfe"]], device=self.device),
            "hind_pfe": torch.tensor([RH["pfe"], LH["pfe"]], device=self.device),
            "hind_pastern": torch.tensor(
                [RH["pastern"], LH["pastern"]], device=self.device
            ),
            "front_haa": torch.tensor([RF["haa"], LF["haa"]], device=self.device),
            "front_hfe": torch.tensor([RF["hfe"], LF["hfe"]], device=self.device),
            "front_kfe": torch.tensor([RF["kfe"], LF["kfe"]], device=self.device),
            "front_pfe": torch.tensor([RF["pfe"], LF["pfe"]], device=self.device),
            "front_pastern": torch.tensor(
                [RF["pastern"], LF["pastern"]], device=self.device
            ),
            # legs (all joints within each leg)
            "rh_leg": torch.tensor(
                [RH["haa"], RH["hfe"], RH["kfe"], RH["pfe"], RH["pastern"]],
                device=self.device,
            ),
            "lh_leg": torch.tensor(
                [LH["haa"], LH["hfe"], LH["kfe"], LH["pfe"], LH["pastern"]],
                device=self.device,
            ),
            "rf_leg": torch.tensor(
                [RF["haa"], RF["hfe"], RF["kfe"], RF["pfe"], RF["pastern"]],
                device=self.device,
            ),
            "lf_leg": torch.tensor(
                [LF["haa"], LF["hfe"], LF["kfe"], LF["pfe"], LF["pastern"]],
                device=self.device,
            ),
            # group by front or hind legs
            "hind_legs": torch.tensor(
                [
                    RH["haa"],
                    RH["hfe"],
                    RH["kfe"],
                    RH["pfe"],
                    RH["pastern"],
                    LH["haa"],
                    LH["hfe"],
                    LH["kfe"],
                    LH["pfe"],
                    LH["pastern"],
                ],
                device=self.device,
            ),
            "front_legs": torch.tensor(
                [
                    RF["haa"],
                    RF["hfe"],
                    RF["kfe"],
                    RF["pfe"],
                    RF["pastern"],
                    LF["haa"],
                    LF["hfe"],
                    LF["kfe"],
                    LF["pfe"],
                    LF["pastern"],
                ],
                device=self.device,
            ),
        }

        self.prev_height_error = torch.zeros(self.num_envs, device=self.device)
        self.lie_down_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.oscillators = torch.zeros(self.num_envs, 4, device=self.device)
        self.oscillator_obs = torch.zeros(self.num_envs, 8, device=self.device)

        self.oscillators_vel = torch.zeros_like(self.oscillators)
        self.grf = torch.zeros(self.num_envs, 4, device=self.device)
        self.osc_omega = self.cfg.osc.omega * torch.ones(
            self.num_envs, 1, device=self.device
        )
        self.osc_coupling = self.cfg.osc.coupling * torch.ones(
            self.num_envs, 1, device=self.device
        )
        self.osc_offset = self.cfg.osc.offset * torch.ones(
            self.num_envs, 1, device=self.device
        )

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self._reset_to_custom_pos(env_ids, self.root_states[env_ids, 2].clone())

        # resample height command
        h_min, h_max = self.cfg.commands.ranges.height
        self.commands[env_ids, 3] = h_min + (h_max - h_min) * torch.rand(
            (len(env_ids),), device=self.device
        )

    def _reset_oscillators(self, env_ids):
        if len(env_ids) == 0:
            return
            # * random
        if self.cfg.osc.init_to == "random":
            self.oscillators[env_ids] = torch_rand_float(
                0,
                2 * torch.pi,
                shape=self.oscillators[env_ids].shape,
                device=self.device,
            )
        elif self.cfg.osc.init_to == "standing":
            self.oscillators[env_ids] = 3 * torch.pi / 2
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)

    def _reset_system(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_oscillators(env_ids)
        self.oscillator_obs = torch.cat(
            (torch.cos(self.oscillators), torch.sin(self.oscillators)), dim=1
        )

        # * keep some robots in the same starting state as they ended
        timed_out_subset = (self.timed_out & ~self.terminated) * (
            torch.rand(self.num_envs, device=self.device)
            < self.cfg.init_state.timeout_reset_ratio
        )

        env_ids = (self.terminated | timed_out_subset).nonzero().flatten()
        if len(env_ids) == 0:
            return
        super()._reset_system(env_ids)

        # pose horse based set height
        self._reset_to_custom_pos(env_ids, self.root_states[env_ids, 2].clone())

        # resample height command
        h_min, h_max = self.cfg.commands.ranges.height
        self.commands[env_ids, 3] = h_min + (h_max - h_min) * torch.rand(
            (len(env_ids),), device=self.device
        )

        if len(env_ids) > 0:
            h_cmd = self.commands[env_ids, 3].flatten()
            h_now = self.base_height[env_ids].flatten()
            self.prev_height_error[env_ids] = torch.abs(h_now - h_cmd)
        self.lie_down_done[env_ids] = False

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        # self.grf = self._compute_grf()
        self.compute_osc_slope()

    def compute_osc_slope(self):
        cmd_x = torch.abs(self.commands[:, 0:1]) - self.cfg.osc.stop_threshold
        stop = cmd_x < 0

        self.osc_offset = stop * self.cfg.osc.offset
        self.osc_omega = stop * self.cfg.osc.omega_stop
        self.osc_coupling = stop * self.cfg.osc.coupling_stop

        self.osc_omega += (~stop) * torch.clamp(
            cmd_x * self.cfg.osc.omega_slope + self.cfg.osc.omega_step,
            min=0.0,
            max=self.cfg.osc.omega_max,
        )
        self.osc_coupling += (~stop) * torch.clamp(
            cmd_x * self.cfg.osc.coupling_slope + self.cfg.osc.coupling_step,
            min=0.0,
            max=self.cfg.osc.coupling_max,
        )

        self.osc_omega = torch.clamp_min(self.osc_omega, 0.1)
        self.osc_coupling = torch.clamp_min(self.osc_coupling, 0)

    def _post_decimation_step(self):
        """Update all states that are not handled in PhysX"""
        super()._post_decimation_step()
        self.grf = self._compute_grf()
        # self._step_oscillators()

    def _post_physx_step(self):
        super()._post_physx_step()
        self._step_oscillators(self.dt / self.cfg.control.decimation)
        return None

    def _step_oscillators(self, dt=None):
        if dt is None:
            dt = self.dt

        local_feedback = self.osc_coupling * (
            torch.cos(self.oscillators) + self.osc_offset
        )
        grf = self._compute_grf()
        self.oscillators_vel = self.osc_omega - grf * local_feedback
        self.oscillators_vel += (
            torch.randn(self.oscillators_vel.shape, device=self.device)
            * self.cfg.osc.process_noise_std
        )

        self.oscillators_vel *= 2 * torch.pi
        self.oscillators += self.oscillators_vel * dt
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)
        self.oscillator_obs = torch.cat(
            (torch.cos(self.oscillators), torch.sin(self.oscillators)), dim=1
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        super()._resample_commands(env_ids)
        possible_commands = torch.tensor(
            self.command_ranges["lin_vel_x"], device=self.device
        )
        self.commands[env_ids, 0:1] = possible_commands[
            torch.randint(
                0, len(possible_commands), (len(env_ids), 1), device=self.device
            )
        ]
        # add some gaussian noise to the commands
        self.commands[env_ids, 0:1] += (
            torch.randn((len(env_ids), 1), device=self.device) * self.cfg.commands.var
        )

        if 0 in self.cfg.commands.ranges.lin_vel_x:
            # * with 20% chance, reset to 0 commands except for forward
            self.commands[env_ids, 1:] *= (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                < 0.8
            ).unsqueeze(1)
            # * with 20% chance, reset to 0 commands except for rotation
            self.commands[env_ids, :2] *= (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                < 0.8
            ).unsqueeze(1)
            # * with 10% chance, reset to 0
            self.commands[env_ids, :] *= (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                < 0.9
            ).unsqueeze(1)

        if self.cfg.osc.randomize_osc_params:
            self._resample_osc_params(env_ids)

        # resample height command
        h_min, h_max = self.cfg.commands.ranges.height
        self.commands[env_ids, 3] = h_min + (h_max - h_min) * torch.rand(
            (len(env_ids),), device=self.device
        )

    def _resample_osc_params(self, env_ids):
        if len(env_ids) > 0:
            self.osc_omega[env_ids, 0] = torch_rand_float(
                self.cfg.osc.omega_range[0],
                self.cfg.osc.omega_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.osc_coupling[env_ids, 0] = torch_rand_float(
                self.cfg.osc.coupling_range[0],
                self.cfg.osc.coupling_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.osc_offset[env_ids, 0] = torch_rand_float(
                self.cfg.osc.offset_range[0],
                self.cfg.osc.offset_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def perturb_base_velocity(self, velocity_delta, env_ids=None):
        if env_ids is None:
            env_ids = [range(self.num_envs)]
        self.root_states[env_ids, 7:10] += velocity_delta
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )

    def _compute_grf(self, grf_norm=True):
        grf = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        if grf_norm:
            return torch.clamp_max(grf / HORSE_WEIGHT, 1.0)
        else:
            return grf

    def _switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(
            -torch.square(torch.max(torch.zeros_like(c_vel), c_vel - 0.1))
            / self.cfg.reward_settings.switch_scale
        )

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity with squared exp"""
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"][0])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"][0])
        return torch.sum(error, dim=1)

    # modified this method so joint index list uses horse 5 DOF legs, not 3 DOF
    def _reward_cursorial(self):
        """
        Encourage legs under body, joints stay neutral, avoid inverted/folded poses
        Penalize key joints drifting from home. For a horse with 5 joints per leg,
        we penalize (haa, hfe, kfe) for each of the 4 legs.
        Distal joint (pfe, pastern_to_foot) naturally flex a lot, shock absorbers
        """
        legs = torch.tensor(list(range(1, 21)), device=self.device)
        return -torch.mean(torch.square(self.dof_pos[:, legs]), dim=1)

    def _reward_swing_grf(self):
        # Reward non-zero grf during swing (0 to pi)
        rew = self.get_swing_grf(self.cfg.osc.osc_bool, self.cfg.osc.grf_bool)
        return -torch.sum(rew, dim=1)

    def _reward_stance_grf(self):
        # Reward non-zero grf during stance (pi to 2pi)
        rew = self.get_stance_grf(self.cfg.osc.osc_bool, self.cfg.osc.grf_bool)
        return torch.sum(rew, dim=1)

    def get_swing_grf(self, osc_bool=False, contact_bool=False):
        if osc_bool:
            phase = torch.lt(self.oscillators, torch.pi).int()
        else:
            phase = torch.maximum(
                torch.zeros_like(self.oscillators), torch.sin(self.oscillators)
            )
        if contact_bool:
            return phase * torch.gt(self._compute_grf(), self.cfg.osc.grf_threshold)
        else:
            return phase * self._compute_grf()

    def get_stance_grf(self, osc_bool=False, contact_bool=False):
        if osc_bool:
            phase = torch.gt(self.oscillators, torch.pi).int()
        else:
            phase = torch.maximum(
                torch.zeros_like(self.oscillators), -torch.sin(self.oscillators)
            )
        if contact_bool:
            return phase * torch.gt(self._compute_grf(), self.cfg.osc.grf_threshold)
        else:
            return phase * self._compute_grf()

    def _reward_coupled_grf(self):
        """
        Multiply rewards for stance/swing grf, discount when undesirable
        behavior (grf during swing, no grf during stance)
        """
        swing_rew = self.get_swing_grf()
        stance_rew = self.get_stance_grf()
        combined_rew = self._sqrdexp(swing_rew * 2) + stance_rew
        prod = torch.prod(torch.clip(combined_rew, 0, 1), dim=1)
        return prod - torch.ones_like(prod)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return super()._reward_dof_vel() * self._switch()

    def _reward_dof_near_home(self):
        return super()._reward_dof_near_home() * self._switch()

    def _reward_stand_still(self):
        """Penalize any movement when at zero commands."""
        # normalize angles so we care about being within ~5 degrees
        rew_pos = torch.mean(
            self._sqrdexp((self.dof_pos - self.default_dof_pos) / torch.pi * 36), dim=1
        )
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel + rew_pos - rew_base_vel) * self._switch()

    def _reward_standing_torques(self):
        """Penalize torques at zero commands"""
        return super()._reward_torques() * self._switch()

    # * gait similarity scores
    def angle_difference(self, theta1, theta2):
        diff = torch.abs(theta1 - theta2) % (2 * torch.pi)
        return torch.min(diff, 2 * torch.pi - diff)

    def _reward_tracking_height(self):
        """Reward for base height."""
        # error between current and commanded height
        error = self.base_height.flatten() - self.commands[:, 3].flatten()
        error /= self.scales["base_height"]

        return self._sqrdexp(error)

    def _reward_tendon_constraints(self):
        """
        Tendon-like coupling constraints (front/hind split):
        - Desired KFE/PFE become dependant on HFE.
        - Penalize deviation from those desired values.
        """

        # --- helpers ---
        def lerp(x, x0, x1, y0, y1):
            """Linear interpolation y(x) between (x0,y0)->(x1,y1), with clamping."""
            # handle degenerate interval
            denom = x1 - x0
            denom = torch.where(torch.abs(denom) < 1e-8, torch.ones_like(denom), denom)
            t = (x - x0) / denom
            t = torch.clamp(t, 0.0, 1.0)
            return y0 + t * (y1 - y0)

        def piecewise_2seg(x, mid, x_lo, x_hi, y_lo, y_mid, y_hi):
            """
            Piecewise linear passing through:
            (x=mid, y=y_mid)
            and reaching:
            (x=x_lo, y=y_lo) for x <= mid
            (x=x_hi, y=y_hi) for x >= mid
            """
            y_left = lerp(
                x, mid, x_lo, y_mid, y_lo
            )  # x in [x_lo, mid] (or beyond -> clamped)
            y_right = lerp(
                x, mid, x_hi, y_mid, y_hi
            )  # x in [mid, x_hi] (or beyond -> clamped)
            return torch.where(x <= mid, y_left, y_right)

        # make desired match (N,2) like the actual joint tensor
        def _match_legs(desired, actual_legs):
            # actual_legs: (N,2)
            if desired.dim() == 0:
                return desired.view(1, 1).expand_as(actual_legs)
            if desired.dim() == 1:
                return desired.unsqueeze(1).expand_as(actual_legs)  # (N,) -> (N,2)
            if desired.dim() == 2:
                # if already (N,2) it's good
                return desired
            raise RuntimeError(f"Unexpected desired shape: {tuple(desired.shape)}")

        # Hind constraints (RH, LH)
        hfe_hind = self.dof_pos[:, self.idx["hind_hfe"]]  # [N,2]
        kfe_hind = self.dof_pos[:, self.idx["hind_kfe"]]
        pfe_hind = self.dof_pos[:, self.idx["hind_pfe"]]

        # 3) hind hfe: 0 -> -1.5  => kfe: 0 -> +1.0, pfe: 0 -> -1.2
        # 4) hind hfe: 0 -> +0.5  => kfe: 0 -> -0.2, pfe: 0 -> -0.5
        kfe_hind_des = piecewise_2seg(
            hfe_hind,
            mid=torch.zeros_like(hfe_hind),  # at hfe=0
            x_lo=-1.5 * torch.ones_like(hfe_hind),
            x_hi=+0.5 * torch.ones_like(hfe_hind),
            y_lo=+1.0 * torch.ones_like(hfe_hind),
            y_mid=0.0 * torch.ones_like(hfe_hind),
            y_hi=-0.2 * torch.ones_like(hfe_hind),
        )

        pfe_hind_des = piecewise_2seg(
            hfe_hind,
            mid=torch.zeros_like(hfe_hind),
            x_lo=-1.5 * torch.ones_like(hfe_hind),
            x_hi=+0.5 * torch.ones_like(hfe_hind),
            y_lo=-1.2 * torch.ones_like(hfe_hind),
            y_mid=0.0 * torch.ones_like(hfe_hind),
            y_hi=-0.5 * torch.ones_like(hfe_hind),
        )

        # Front constraints (RF, LF)
        hfe_front = self.dof_pos[:, self.idx["front_hfe"]]  # [N,2]
        kfe_front = self.dof_pos[:, self.idx["front_kfe"]]
        pfe_front = self.dof_pos[:, self.idx["front_pfe"]]

        # 5) front hfe: 0 -> +0.6 => kfe: 0 -> 0, pfe: 0 -> 0
        # 6) front hfe: 0 -> -1.0 => kfe: 0 -> -1.5, pfe: 0 -> +1.5
        # 7) front hfe: 0 -> -1.0 => kfe: 0 -> -1.5, pfe: 0 -> +3.0

        zeros = torch.zeros_like(hfe_front)
        ones = torch.ones_like(hfe_front)

        # kfe: flat 0 for hfe in [0, +0.6], ramp to -1.5 for hfe in [-1.0, 0]
        kfe_front_des = piecewise_2seg(
            hfe_front,
            mid=zeros,
            x_lo=-1.0 * ones,
            x_hi=+0.6 * ones,
            y_lo=-1.5 * ones,  # at hfe=-1.0
            y_mid=0.0 * ones,  # at hfe=0
            y_hi=0.0 * ones,  # at hfe=+0.6
        )

        # pfe: flat 0 for hfe in [0, +0.6], ramp to +1.5 for hfe in [-1.0, 0]
        pfe_front_des_15 = piecewise_2seg(
            hfe_front,
            mid=zeros,
            x_lo=-1.0 * ones,
            x_hi=+0.6 * ones,
            y_lo=+1.5 * ones,  # at hfe=-1.0
            y_mid=0.0 * ones,  # at hfe=0
            y_hi=0.0 * ones,  # at hfe=+0.6
        )

        # pfe: flat 0 for hfe in [0, +0.6], ramp to +3.0 for hfe in [-1.0, 0]
        pfe_front_des_30 = piecewise_2seg(
            hfe_front,
            mid=zeros,
            x_lo=-1.0 * ones,
            x_hi=+0.6 * ones,
            y_lo=+3.0 * ones,  # at hfe=-1.0
            y_mid=0.0 * ones,  # at hfe=0
            y_hi=0.0 * ones,  # at hfe=+0.6
        )

        # --- HIND ---
        kfe_hind_des_ = _match_legs(kfe_hind_des, kfe_hind)  # -> (N,2)
        pfe_hind_des_ = _match_legs(pfe_hind_des, pfe_hind)  # -> (N,2)
        hind_pen = (kfe_hind - kfe_hind_des_) ** 2 + (
            pfe_hind - pfe_hind_des_
        ) ** 2  # (N,2)

        # --- FRONT ---
        # KFE: flat 0 for hfe in [0, 0.6], ramp to -1.5 for hfe in [-1,0]
        kfe_front_des_ = _match_legs(kfe_front_des, kfe_front)  # -> (N,2)
        kfe_err = (kfe_front - kfe_front_des_) ** 2  # (N,2)

        # negative HFE:
        #   - PFE: 0 -> +1.5  (hfe 0 -> -1)
        #   - PFE: 0 -> +3.0  (hfe 0 -> -1)
        pfe_front_des15_ = _match_legs(pfe_front_des_15, pfe_front)  # -> (N,2)
        pfe_front_des30_ = _match_legs(pfe_front_des_30, pfe_front)  # -> (N,2)

        pfe_err15 = (pfe_front - pfe_front_des15_) ** 2
        pfe_err30 = (pfe_front - pfe_front_des30_) ** 2

        # BOTH branches acceptable always:
        pfe_term = torch.minimum(pfe_err15, pfe_err30)  # (N,2)

        front_pen = kfe_err + pfe_term  # (N,2)

        # --- aggregate per env ---
        pen = hind_pen.mean(dim=1) + front_pen.mean(dim=1)  # (N,)
        return -pen

    def _reset_to_custom_pos(self, env_ids, reset_h):
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        B = env_ids.numel()
        if B == 0:
            return

        # --- normalize reset_h to (B,) on device ---
        if not torch.is_tensor(reset_h):
            reset_h = torch.tensor(
                reset_h, device=self.device, dtype=torch.float32
            ).repeat(B)
        else:
            reset_h = reset_h.to(device=self.device, dtype=torch.float32)
            if reset_h.numel() == 1:
                reset_h = reset_h.repeat(B)
            elif reset_h.shape[0] != B:
                # if passed full (num_envs,), subset it
                if reset_h.shape[0] == self.num_envs:
                    reset_h = reset_h[env_ids]
                else:
                    raise RuntimeError(
                        f"reset_h shape {reset_h.shape} incompatible env_id batch {B}"
                    )

        # env_ids must be LongTensor on device
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        # Get a per-env base pose, regardless of how default_dof_pos is stored
        ddp = self.default_dof_pos

        if ddp.dim() == 1:
            # (num_dofs,) -> make (B, num_dofs)
            base_pose = ddp.unsqueeze(0).repeat(env_ids.numel(), 1).clone()
        elif ddp.dim() == 2 and ddp.shape[0] == 1:
            # (1, num_dofs) -> expand to (B, num_dofs)
            base_pose = ddp.expand(env_ids.numel(), -1).clone()
        elif ddp.dim() == 2 and ddp.shape[0] == self.num_envs:
            # (num_envs, num_dofs) -> index by env ids
            base_pose = ddp[env_ids].clone()
        else:
            raise RuntimeError(f"Unexpected default_dof_pos shape: {tuple(ddp.shape)}")

        # Decide stand vs lie by z pos
        h_mid = 0.7

        use_stand = reset_h >= h_mid  # (B,) bool

        stand_pose = base_pose.clone()
        lie_pose = base_pose.clone()

        # DOF map: 0 base_joint
        # RH: 1..5, LH: 6..10, RF: 11..15, LF: 16..20 (haa,hfe,kfe,pfe,pastern)

        pos_range = self.cfg.init_state.dof_pos_range

        for base in [1, 6, 11, 16]:
            haa = base + 0
            hfe = base + 1
            kfe = base + 2
            pfe = base + 3
            pas = base + 4

            # standing (tune)
            stand_pose[:, haa] = pos_range["haa"][0] + (
                pos_range["haa"][1] - pos_range["haa"][0]
            ) * torch.rand((lie_pose.shape[0],), device=self.device)
            stand_pose[:, hfe] = pos_range["hfe"][0] + (
                pos_range["hfe"][1] - pos_range["hfe"][0]
            ) * torch.rand((lie_pose.shape[0],), device=self.device)
            stand_pose[:, kfe] = pos_range["kfe"][0] + (
                pos_range["kfe"][1] - pos_range["kfe"][0]
            ) * torch.rand((lie_pose.shape[0],), device=self.device)
            stand_pose[:, pfe] = pos_range["pfe"][0] + (
                pos_range["pfe"][1] - pos_range["pfe"][0]
            ) * torch.rand((lie_pose.shape[0],), device=self.device)
            stand_pose[:, pas] = pos_range["pastern_to_foot"][0] + (
                pos_range["pastern_to_foot"][1] - pos_range["pastern_to_foot"][0]
            ) * torch.rand((lie_pose.shape[0],), device=self.device)

        # front kneel (tune)
        for base in [11, 16]:
            haa = base + 0
            hfe = base + 1
            kfe = base + 2
            pfe = base + 3
            pas = base + 4
            lie_pose[:, haa] = 0.0
            lie_pose[:, hfe] = -1.0 + (0.2 - -1.0) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )
            lie_pose[:, kfe] = -1.5 + (-0.5 - -1.5) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )
            lie_pose[:, pfe] = -0.3 + (3.0 - -0.3) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )
            lie_pose[:, pas] = -0.3 + (1.8 - -0.3) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )

        # hind tuck (tune)
        for base in [1, 6]:
            haa = base + 0
            hfe = base + 1
            kfe = base + 2
            pfe = base + 3
            pas = base + 4
            lie_pose[:, haa] = 0.0
            lie_pose[:, hfe] = -1.5 + (-1 - -1.5) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )
            lie_pose[:, kfe] = -0.2 + (1 - -0.2) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )
            lie_pose[:, pfe] = 0.0 + (-1.2 - 0.0) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )
            lie_pose[:, pas] = -0.3 + (1.8 - -0.3) * torch.rand(
                (lie_pose.shape[0],), device=self.device
            )

        new_pose = torch.where(use_stand.unsqueeze(1), stand_pose, lie_pose)

        # clamp to limits (dof_pos_limits: (num_dof, 2))
        lo = self.dof_pos_limits[:, 0].unsqueeze(0)
        hi = self.dof_pos_limits[:, 1].unsqueeze(0)
        new_pose = torch.max(torch.min(new_pose, hi), lo)

        # apply to buffers
        self.dof_pos[env_ids] = new_pose
        self.dof_vel[env_ids] = 0.0

        if (
            hasattr(self, "dof_state")
            and self.dof_state.dim() == 3
            and self.dof_state.shape[2] >= 2
        ):
            self.dof_state[env_ids, :, 0] = new_pose
            self.dof_state[env_ids, :, 1] = 0.0
