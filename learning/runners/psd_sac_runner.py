import os
import torch
from tensordict import TensorDict

from learning.utils import Logger

from learning.runners import OffPolicyRunner
from learning.modules import Critic, ChimeraActor
from learning.modules.QRCritics import *  # noqa F401
from learning.modules.TaylorCritics import *  # noqa F401

from learning.storage import ReplayBuffer
from learning.algorithms import SAC


logger = Logger()
storage = ReplayBuffer()


class PSACRunner(OffPolicyRunner):
    def __init__(self, env, train_cfg, device="cpu"):
        super().__init__(env, train_cfg, device)
        logger.initialize(
            self.env.num_envs,
            self.env.dt,
            self.cfg["max_iterations"],
            self.device,
        )

    def _set_up_alg(self):
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_in = self.get_obs_size(self.critic_cfg["obs"]) + num_actions  # noqa: F841
        actor = ChimeraActor(num_actor_obs, num_actions, **self.actor_cfg)

        critic_class = self.critic_cfg["critic_class_name"]
        critic_1 = eval(f"{critic_class}(num_critic_in, **self.critic_cfg)")
        critic_2 = eval(f"{critic_class}(num_critic_in, **self.critic_cfg)")
        target_critic_1 = eval(f"{critic_class}(num_critic_in, **self.critic_cfg)")
        target_critic_2 = eval(f"{critic_class}(num_critic_in, **self.critic_cfg)")

        # critic_1 = Critic(num_critic_obs + num_actions, **self.critic_cfg)
        # critic_2 = Critic(num_critic_obs + num_actions, **self.critic_cfg)

        # target_critic_1 = Critic(num_critic_obs + num_actions, **self.critic_cfg)
        # target_critic_2 = Critic(num_critic_obs + num_actions, **self.critic_cfg)

        print(actor)
        print(critic_1)

        self.alg = SAC(
            actor,
            critic_1,
            critic_2,
            target_critic_1,
            target_critic_2,
            device=self.device,
            **self.alg_cfg,
        )
