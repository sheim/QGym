import torch
import torch.nn as nn
import torch.optim as optim
from learning.algorithms.ppo import PPO
from learning.modules import ActorCritic
from learning.storage import RolloutStorage
from learning.modules.lqrc import CustomCholeskyPlusConstLoss


class PPOCriticOnly(PPO):
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        standard_loss=True,
        plus_c_penalty=0.0,
        **kwargs,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # * PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            self.actor_critic.critic.parameters(), lr=learning_rate
        )
        self.transition = RolloutStorage.Transition()

        # * PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # * custom NN parameters
        self.standard_loss = standard_loss
        self.plus_c_penalty = plus_c_penalty

    def update(self):
        self.mean_value_loss = 0
        self.mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.actor_critic.act(obs_batch)
            # actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
            #     actions_batch
            # )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            # mu_batch = self.actor_critic.action_mean
            # sigma_batch = self.actor_critic.action_std
            # entropy_batch = self.actor_critic.entropy

            # # * KL
            # if self.desired_kl is not None and self.schedule == "adaptive":
            #     with torch.inference_mode():
            #         kl = torch.sum(
            #             torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            #             + (
            #                 torch.square(old_sigma_batch)
            #                 + torch.square(old_mu_batch - mu_batch)
            #             )
            #             / (2.0 * torch.square(sigma_batch))
            #             - 0.5,
            #             axis=-1,
            #         )
            #         kl_mean = torch.mean(kl)

            #         if kl_mean > self.desired_kl * 2.0:
            #             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            #         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            #             self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            #         for param_group in self.optimizer.param_groups:
            #             param_group["lr"] = self.learning_rate

            # # * Surrogate loss
            # ratio = torch.exp(
            #     actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            # )
            # surrogate = -torch.squeeze(advantages_batch) * ratio
            # surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            #     ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            # )
            # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # * Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                if self.standard_loss:
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                else:
                    value_losses = CustomCholeskyPlusConstLoss(
                        const_penalty=self.plus_c_penalty
                    ).forward(
                        value_batch,
                        returns_batch,
                        self.actor_critic.critic.NN.intermediates,
                    )
                    value_losses_clipped = CustomCholeskyPlusConstLoss(
                        const_penalty=self.plus_c_penalty
                    ).forward(
                        value_clipped,
                        returns_batch,
                        self.actor_critic.critic.NN.intermediates,
                    )
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                if self.standard_loss:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                else:
                    value_losses = CustomCholeskyPlusConstLoss(
                        const_penalty=self.plus_c_penalty
                    ).forward(
                        value_batch,
                        returns_batch,
                        self.actor_critic.critic.NN.intermediates,
                    )

            loss = self.value_loss_coef * value_loss

            # * Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.mean_value_loss += value_loss.item()
            # self.mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.mean_value_loss /= num_updates
        # self.mean_surrogate_loss /= num_updates
        self.storage.clear()
