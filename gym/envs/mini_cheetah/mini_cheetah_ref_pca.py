import torch
import pandas as pd
from isaacgym.torch_utils import torch_rand_float, to_torch
import numpy as np
from gym import LEGGED_GYM_ROOT_DIR
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah
MINI_CHEETAH_MASS = 8.292 * 9.81  # Weight of mini cheetah in Newtons


class MiniCheetahRefPca(MiniCheetah):
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        csv_path = cfg.init_state.ref_traj.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        self.leg_ref = 3 * to_torch(pd.read_csv(csv_path).to_numpy(), device=sim_device)
        self.omega = 2 * torch.pi * cfg.control.gait_freq
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.pca_scalings = torch.zeros(self.num_envs, self.cfg.pca.num_pcs, device=self.device)
        self.eigenvectors = torch.zeros(
                self.cfg.pca.num_pcs, self.dof_pos_target.shape[1], device=self.device
            )
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.phase_obs = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        self.phase[env_ids] = torch_rand_float(
            0, torch.pi, shape=self.phase[env_ids].shape, device=self.device
        )
    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.compute_pca()
    def _post_physx_step(self):
        super()._post_physx_step()
        self.phase = (
            self.phase + self.dt * self.omega / self.cfg.control.decimation
        ).fmod(2 * torch.pi)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.phase_obs = torch.cat(
            (torch.sin(self.phase), torch.cos(self.phase)), dim=1
        )
    def compute_pca(self):
        if not self.cfg.pca.torques:
            if self.cfg.pca.mode =="symmetries":
                for i in range(len(self.cfg.pca.symmetry_eigvec_ref_index)):
                    one_eigvec =  self.cfg.pca.eigenvectors[self.cfg.pca.symmetry_eigvec_ref_index[i]:self.cfg.pca.symmetry_eigvec_ref_index[i]+1,0:3]
                    self.eigenvectors[4*i,:]=torch.cat((one_eigvec,one_eigvec,one_eigvec,one_eigvec),1)
                    self.eigenvectors[1+4*i,:] = torch.cat((one_eigvec,-one_eigvec,-one_eigvec, one_eigvec),1)
                    self.eigenvectors[2+4*i,:] = torch.cat((one_eigvec,-one_eigvec,one_eigvec,-one_eigvec),1)
                    self.eigenvectors[3+4*i,:] = torch.cat((one_eigvec,one_eigvec,-one_eigvec,-one_eigvec),1)

                for i in range(len(self.cfg.pca.haa_flip_indexes)):
                    self.eigenvectors[:,self.cfg.pca.haa_flip_indexes[i]:
                                      self.cfg.pca.haa_flip_indexes[i]+1] *= -1

            if self.cfg.pca.mode == "one_leg":
                # 4th leg, all actuators
                self.eigenvectors = self.cfg.pca.eigenvectors
                self.eigenvectors = torch.hstack([self.eigenvectors[0:6,0:3],self.eigenvectors[0:6,0:3],
                                                  self.eigenvectors[0:6,0:3],self.eigenvectors[0:6,0:3]])
                #flipping sign/mirroring for opposing leg (excluding haa)
                self.eigenvectors[:,4:6] *=-1
                self.eigenvectors[:,7:9] *=-1

                #flipping abad
                for i in range(len(self.cfg.pca.haa_flip_indexes)):
                    self.eigenvectors[:,self.cfg.pca.haa_flip_indexes[i]:
                                      self.cfg.pca.haa_flip_indexes[i]+1] *= -1

            elif self.cfg.pca.mode == "joint":
                # 3rd actuator kfe, all legs
                eigenvectors_og = torch.tensor(
                    [
                        [0.30653829, -0.76561209, -0.26433388,0,0,0],
                        [-0.5543308, -0.12128448, 0.65422278,0,0,0],
                        [-0.40904421, 0.38943236, -0.65652515,0,0,0],
                        [0.65683672, 0.49746421, 0.26663625,0,0,0],
                    ],
                    device=self.device,
                ).T
                for i in range(0, 4):
                    self.eigenvectors[:, i * 3] = eigenvectors_og[:, i]
                #print(self.eigenvectors.shape)
            elif self.cfg.pca.mode == "all":
                self.eigenvectors = self.cfg.pca.eigenvectors
            else:
                Warning("PC MODE NOT RECOGNIZED in compute_torques")

            self.dof_pos_target = torch.zeros(self.num_envs, self.dof_pos_target.shape[1], device=self.device)
            for i in range(0, self.pca_scalings.shape[1]):  # todo sanity check this! unit test or something?
                self.dof_pos_target += torch.mul(
                    self.eigenvectors[i, :].repeat(self.num_envs, 1),
                    self.pca_scalings[:, i:i+1],
                )
        else:
            self.dof_pos_target = (self.torques - self.tau_ff-self.d_gains*
                                   (self.dof_vel_target - self.dof_vel))/self.p_gains - self.default_dof_pos + self.dof_pos

    def _compute_torques(self):
        if self.cfg.pca.torques:
            self.eigenvectors = torch.from_numpy(
                np.load("/home/aileen/QGym/scripts/pca_components_torques.npy")).to(self.device).T
            for i in range(0, self.pca_scalings.shape[1]):
                self.torques += torch.mul(
                    self.eigenvectors[i, :].repeat(self.num_envs, 1),
                    self.pca_scalings[:, i:i+1],
                )
        else:
            self.torques = (
                self.p_gains * (self.dof_pos_target + self.default_dof_pos - self.dof_pos)
                + self.d_gains * (self.dof_vel_target - self.dof_vel)
                + self.tau_ff
            )
        self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)
        return self.torques.view(self.torques.shape)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # * with 10% chance, reset to 0 commands
        rand_ids = torch_rand_float(
            0, 1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, :3] *= (rand_ids < 0.9).unsqueeze(1)

    def _switch(self):
        c_vel = torch.linalg.norm(self.commands, dim=1)
        return torch.exp(
            -torch.square(torch.max(torch.zeros_like(c_vel), c_vel - 0.1)) / 0.1
        )

    def _reward_swing_grf(self):
        """Reward non-zero grf during swing (0 to pi)"""
        in_contact = torch.gt(
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1),
            50.0,
        )
        ph_off = torch.lt(self.phase, torch.pi)
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return -torch.sum(rew.float(), dim=1) * (1 - self._switch())

    def _reward_stance_grf(self):
        """Reward non-zero grf during stance (pi to 2pi)"""
        in_contact = torch.gt(
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1),
            50.0,
        )
        ph_off = torch.gt(self.phase, torch.pi)  # should this be in swing?
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)

        return torch.sum(rew.float(), dim=1) * (1 - self._switch())

    # def _compute_grf(self, grf_norm=True):
    #     grf = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
    #     if grf_norm:
    #         return torch.clamp_max(grf / MINI_CHEETAH_MASS, 1.0)
    #     else:
    #         return grf
        
    # def _reward_swing_grf(self):
    #     # Reward non-zero grf during swing (0 to pi)
    #     rew = self.get_swing_grf()
    #     return -torch.sum(rew, dim=1)

    # def _reward_stance_grf(self):
    #     # Reward non-zero grf during stance (pi to 2pi)
    #     rew = self.get_stance_grf()
    #     return torch.sum(rew, dim=1)

    # def get_swing_grf(self, osc_bool=False, contact_bool=False):
    #     if osc_bool:
    #         phase = torch.lt(self.oscillators, torch.pi).int()
    #     else:
    #         phase = torch.maximum(
    #             torch.zeros_like(self.phase), torch.sin(self.phase)
    #         )
    #     if contact_bool:
    #         return phase * torch.gt(self._compute_grf(), self.cfg.osc.grf_threshold)
    #     else:
    #         return phase * self._compute_grf()

    # def get_stance_grf(self, osc_bool=False, contact_bool=False):
    #     if osc_bool:
    #         phase = torch.gt(self.oscillators, torch.pi).int()
    #     else:
    #         phase = torch.maximum(
    #             torch.zeros_like(self.phase), -torch.sin(self.phase)
    #         )
    #     if contact_bool:
    #         return phase * torch.gt(self._compute_grf(), self.cfg.osc.grf_threshold)
    #     else:
    #         return phase * self._compute_grf()

    def _reward_reference_traj(self):
        """REWARDS EACH LEG INDIVIDUALLY BASED ON ITS POSITION IN THE CYCLE"""
        # * dof position error
        error = self._get_ref() + self.default_dof_pos - self.dof_pos
        error /= self.scales["dof_pos"]
        reward = (self._sqrdexp(error) - torch.abs(error) * 0.2).mean(dim=1)
        # * only when commanded velocity is higher
        return reward * (1 - self._switch())

    def _get_ref(self):
        leg_frame = torch.zeros_like(self.torques)
        # offset by half cycle (trot)
        ph_off = torch.fmod(self.phase + torch.pi, 2 * torch.pi)
        phd_idx = (
            torch.round(self.phase * (self.leg_ref.size(dim=0) / (2 * torch.pi) - 1))
        ).long()
        pho_idx = (
            torch.round(ph_off * (self.leg_ref.size(dim=0) / (2 * torch.pi) - 1))
        ).long()
        leg_frame[:, 0:3] += self.leg_ref[phd_idx.squeeze(), :]
        leg_frame[:, 3:6] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 6:9] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 9:12] += self.leg_ref[phd_idx.squeeze(), :]
        return leg_frame

    def _reward_stand_still(self):
        """Penalize motion at zero commands"""
        # * normalize angles so we care about being within 5 deg
        rew_pos = torch.mean(
            self._sqrdexp((self.dof_pos - self.default_dof_pos) / torch.pi * 36),
            dim=1,
        )
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel + rew_pos - rew_base_vel) * self._switch()

    def _reward_tracking_lin_vel(self):
        """Tracking linear velocity commands (xy axes)"""
        # just use lin_vel?
        reward = super()._reward_tracking_lin_vel()
        return reward * (1 - self._switch())
    def _reward_pca(self):
        """Tracking pca"""
        error = torch.square(self.pca_scalings)
        error = torch.exp(-error /0.5)
        return torch.sum(error, dim=1)