import torch
import torch.nn as nn
import torch.optim as optim

from learning.utils import (
    create_uniform_generator,
    compute_generalized_advantages,
)


class PPO2:
    def __init__(
        self,
        actor,
        critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        loss_fn="MSE",
        device="cpu",
        **kwargs,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # * PPO components
        self.actor = actor.to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = critic.to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # * PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def test_mode(self):
        self.actor.test()
        self.critic.test()

    def switch_to_train(self):
        self.actor.train()
        self.critic.train()

    def act(self, obs):
        return self.actor.act(obs).detach()

    def update(self, data, last_obs=None):
        if last_obs is None:
            last_values = None
        else:
            with torch.no_grad():
                last_values = self.critic.evaluate(last_obs).detach()
        compute_generalized_advantages(
            data, self.gamma, self.lam, self.critic, last_values
        )

        self.update_critic(data)
        self.update_actor(data)

    def update_critic(self, data):
        self.mean_value_loss = 0
        counter = 0

        n, m = data.shape
        total_data = n * m
        batch_size = total_data // self.num_mini_batches
        generator = create_uniform_generator(data, batch_size, self.num_learning_epochs)
        for batch in generator:
            value_batch = self.critic.evaluate(batch["critic_obs"])
            value_loss = self.critic.loss_fn(value_batch, batch["returns"])
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.mean_value_loss += value_loss.item()
            counter += 1
        self.mean_value_loss /= counter

    def update_actor(self, data):
        # already done before
        # compute_generalized_advantages(data, self.gamma, self.lam, self.critic)
        self.mean_surrogate_loss = 0
        counter = 0

        self.actor.act(data["actor_obs"])
        data["old_sigma_batch"] = self.actor.action_std.detach()
        data["old_mu_batch"] = self.actor.action_mean.detach()
        data["old_actions_log_prob_batch"] = self.actor.get_actions_log_prob(
            data["actions"]
        ).detach()

        n, m = data.shape
        total_data = n * m
        batch_size = total_data // self.num_mini_batches
        generator = create_uniform_generator(data, batch_size, self.num_learning_epochs)
        for batch in generator:
            # ! refactor how this is done
            self.actor.act(batch["actor_obs"])
            actions_log_prob_batch = self.actor.get_actions_log_prob(batch["actions"])
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std
            entropy_batch = self.actor.entropy

            # * KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / batch["old_sigma_batch"] + 1.0e-5)
                        + (
                            torch.square(batch["old_sigma_batch"])
                            + torch.square(batch["old_mu_batch"] - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        # ! check this
                        param_group["lr"] = self.learning_rate

            # * Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(batch["old_actions_log_prob_batch"])
            )
            surrogate = -torch.squeeze(batch["advantages"]) * ratio
            surrogate_clipped = -torch.squeeze(batch["advantages"]) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

            # * Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.mean_surrogate_loss += surrogate_loss.item()
            counter += 1
        self.mean_surrogate_loss /= counter
