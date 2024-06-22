import torch
import torch.nn as nn

from .ppo2 import PPO2
from learning.utils import (
    create_uniform_generator,
    compute_generalized_advantages,
    compute_gae_vtrace,
    normalize,
)


# Implementation based on GePPO repo: https://github.com/jqueeney/geppo
class GePPO(PPO2):
    def __init__(
        self,
        actor,
        critic,
        is_trunc=1.0,
        eps_ppo=0.2,
        eps_geppo=0.1,
        eps_vary=False,
        adapt_lr=True,
        adapt_factor=0.03,
        adapt_minthresh=0.0,
        adapt_maxthresh=1.0,
        **kwargs,
    ):
        super().__init__(actor, critic, **kwargs)

        # Importance sampling truncation
        self.is_trunc = is_trunc

        # Clipping parameter
        self.eps_ppo = eps_ppo
        self.eps_geppo = eps_geppo
        self.eps_vary = eps_vary

        # Learning rate
        self.adapt_lr = adapt_lr
        self.adapt_factor = adapt_factor
        self.adapt_minthresh = adapt_minthresh
        self.adapt_maxthresh = adapt_maxthresh

        self.updated = False

    def update(self, data, weights):
        values = self.critic.evaluate(data["critic_obs"])
        # Handle single env case
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        data["values"] = values

        # Compute GAE with and without V-trace
        adv = compute_generalized_advantages(data, self.gamma, self.lam, self.critic)
        ret = adv + values
        adv_vtrace, ret_vtrace = compute_gae_vtrace(
            data, self.gamma, self.lam, self.is_trunc, self.actor, self.critic
        )
        # Handle single env case
        if adv_vtrace.dim() == 1:
            adv_vtrace = adv_vtrace.unsqueeze(-1)
        if ret_vtrace.dim() == 1:
            ret_vtrace = ret_vtrace.unsqueeze(-1)

        # Only use V-trace if we have updated once already
        if self.updated:
            data["advantages"] = adv_vtrace
            data["returns"] = ret_vtrace
        else:
            data["advantages"] = adv
            data["returns"] = ret
            self.updated = True

        # Update critic and actor
        data["weights"] = weights
        self.update_critic(data)
        data["advantages"] = normalize(data["advantages"])
        self.update_actor(data)

        # Update pik weights
        if self.actor.store_pik:
            self.actor.update_pik_weights()

        # Logging: Store mean advantages and returns
        self.adv_mean = adv.mean().item()
        self.ret_mean = ret.mean().item()
        self.adv_vtrace_mean = adv_vtrace.mean().item()
        self.ret_vtrace_mean = ret_vtrace.mean().item()

    def update_critic(self, data):
        self.mean_value_loss = 0
        counter = 0

        generator = create_uniform_generator(
            data,
            self.batch_size,
            max_gradient_steps=self.max_gradient_steps,
        )
        for batch in generator:
            # GePPO critic loss uses weights
            value_loss = self.critic.loss_fn(
                batch["critic_obs"], batch["returns"], batch["weights"]
            )
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.mean_value_loss += value_loss.item()
            counter += 1
        self.mean_value_loss /= counter

    def update_actor(self, data):
        # Update clipping eps
        if self.eps_vary:
            log_prob_pik = self.actor.get_pik_log_prob(
                data["actor_obs"], data["actions"]
            )
            offpol_ratio = torch.exp(log_prob_pik - data["log_prob"])
            # TODO: I am taking the mean over 2 dims, check if this is correct
            eps_old = torch.mean(data["weights"] * torch.abs(offpol_ratio - 1.0))
            self.eps_geppo = max(self.eps_ppo - eps_old.item(), 0.0)

        self.mean_surrogate_loss = 0
        counter = 0

        generator = create_uniform_generator(
            data,
            self.batch_size,
            max_gradient_steps=self.max_gradient_steps,
        )
        for batch in generator:
            self.actor.act(batch["actor_obs"])
            entropy_batch = self.actor.entropy

            # * GePPO Surrogate loss
            log_prob_pik = self.actor.get_pik_log_prob(
                batch["actor_obs"], batch["actions"]
            )
            offpol_ratio = torch.exp(log_prob_pik - batch["log_prob"])
            advantages = batch["advantages"]

            # TODO: Center/clip advantages (optional)
            # adv_mean = torch.mean(
            #     offpol_ratio * batch["weights"] * advantages, dim=2
            # ) / torch.mean(offpol_ratio, batch["weights"], dim=2)
            # adv_std = torch.std(
            #     offpol_ratio * batch["weights"] * advantages, dim=2
            # )

            log_prob = self.actor.get_actions_log_prob(batch["actions"])
            ratio = torch.exp(log_prob - batch["log_prob"])
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, offpol_ratio - self.eps_geppo, offpol_ratio + self.eps_geppo
            )
            surrogate_loss = (
                torch.max(surrogate, surrogate_clipped) * batch["weights"]
            ).mean()

            loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

            # * Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.mean_surrogate_loss += surrogate_loss.item()
            counter += 1
        self.mean_surrogate_loss /= counter

        # Compute TV, add to self for logging
        self.actor.act(data["actor_obs"])
        log_prob = self.actor.get_actions_log_prob(data["actions"])
        log_prob_pik = self.actor.get_pik_log_prob(data["actor_obs"], data["actions"])
        ratio = torch.exp(log_prob - data["log_prob"])
        clip_center = torch.exp(log_prob_pik - data["log_prob"])
        ratio_diff = torch.abs(ratio - clip_center)
        self.tv = 0.5 * torch.mean(data["weights"] * ratio_diff)

        # Adapt learning rate
        if self.adapt_lr:
            if self.tv > (self.adapt_maxthresh * (0.5 * self.eps_geppo)):
                self.learning_rate /= 1 + self.adapt_factor
            elif self.tv < (self.adapt_minthresh * (0.5 * self.eps_geppo)):
                self.learning_rate *= 1 + self.adapt_factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
