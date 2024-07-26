import torch
import torch.nn as nn
import torch.optim as optim

from learning.utils import (
    create_uniform_generator,
    compute_generalized_advantages,
    normalize,
    polyak_update,
)


class LinkedIPG:
    def __init__(
        self,
        actor,
        critic_v,
        critic_q,
        target_critic_q,
        batch_size=2**15,
        max_gradient_steps=10,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        polyak=0.995,
        use_cv=False,
        inter_nu=0.2,
        beta="off_policy",
        device="cpu",
        lr_range=[1e-4, 1e-2],
        lr_ratio=1.3,
        val_interpolation=0.5,
        **kwargs,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.lr_range = lr_range
        self.lr_ratio = lr_ratio

        # * PPO components
        self.actor = actor.to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_v = critic_v.to(self.device)
        self.critic_v_optimizer = optim.Adam(
            self.critic_v.parameters(), lr=learning_rate
        )

        # * IPG components
        self.critic_q = critic_q.to(self.device)
        self.critic_q_optimizer = optim.Adam(
            self.critic_q.parameters(), lr=learning_rate
        )
        self.target_critic_q = target_critic_q.to(self.device)
        self.target_critic_q.load_state_dict(self.critic_q.state_dict())

        # * PPO parameters
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.max_gradient_steps = max_gradient_steps
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # * IPG parameters
        self.polyak = polyak
        self.use_cv = use_cv
        self.inter_nu = inter_nu
        self.beta = beta
        self.val_interpolation = val_interpolation

    def switch_to_train(self):
        self.actor.train()
        self.critic_v.train()
        self.critic_q.train()

    def act(self, obs):
        return self.actor.act(obs).detach()

    def update(self, data_onpol, data_offpol):
        # On-policy GAE
        data_onpol["values"] = self.critic_v.evaluate(data_onpol["critic_obs"])
        data_onpol["advantages"] = compute_generalized_advantages(
            data_onpol, self.gamma, self.lam, self.critic_v
        )
        data_onpol["returns"] = data_onpol["advantages"] + data_onpol["values"]
        data_onpol["advantages"] = normalize(data_onpol["advantages"])

        data_offpol["values"] = self.critic_v.evaluate(data_offpol["critic_obs"])
        data_offpol["advantages"] = compute_generalized_advantages(
            data_offpol, self.gamma, self.lam, self.critic_v
        )
        data_offpol["returns"] = data_offpol["advantages"] + data_offpol["values"]

        self.update_critic_v(data_offpol)
        self.update_critic_q(data_offpol)
        self.update_actor(data_onpol, data_offpol)

    # def update_joint_critics(self, data_onpol, data_offpol):
    #     self.mean_q_loss = 0
    #     self.mean_value_loss = 0
    #     counter = 0
    #     generator_onpol = create_uniform_generator(
    #         data_onpol,
    #         self.batch_size,
    #         max_gradient_steps=self.max_gradient_steps,
    #     )
    #     generator_offpol = create_uniform_generator(
    #         data_offpol,
    #         self.batch_size,
    #         max_gradient_steps=self.max_gradient_steps,
    #     )
    #     for batch_onpol, batch_offpol in zip(generator_onpol, generator_offpol):
    #         with torch.no_grad():
    #             action_next_onpol = self.actor.act_inference(
    #                 batch_onpol["next_actor_obs"]
    #             )
    #             q_input_next_onpol = torch.cat(
    #                 batch_onpol["next_critic_obs"], action_next_onpol
    #             )
    #             action_next_offpol = self.actor.act_inference(
    #                 batch_offpol["next_actor_obs"]
    #             )
    #             q_input_next_offpol = torch.cat(
    #                 batch_offpol["next_critic_obs"], action_next_offpol
    #             )
    #             q_value_offpol = self.critic_q.evaluate(q_input_next_offpol)

    #         loss_V_returns = self.critic_v.loss_fn(
    #             batch_onpol["critic_obs"], batch_onpol["returns"]
    #         )
    # loss_V_Q = nn.functional.mse_loss(
    #     self.critic_v.evaluate(batch_onpol["critic_obs"]),
    #     self.critic_q.evaluate(batch_onpol["critic_obs"]),
    #     reduction="mean")
    #
    #         with torch.no_grad():
    #             action_next = self.actor.act_inference(batch_offpol["next_actor_obs"])
    #             q_input_next = torch.cat(
    #                 (batch_offpol["next_critic_obs"], action_next), dim=-1
    #             )

    def update_critic_q(self, data):
        self.mean_q_loss = 0
        counter = 0

        generator = create_uniform_generator(
            data,
            self.batch_size,
            max_gradient_steps=self.max_gradient_steps,
        )
        for batch in generator:
            with torch.no_grad():
                action_next = self.actor.act_inference(batch["next_actor_obs"])
                q_input_next = torch.cat(
                    (batch["next_critic_obs"], action_next), dim=-1
                )
                q_next = self.target_critic_q.evaluate(q_input_next)
                v_next = self.critic_v.evaluate(batch["next_critic_obs"])
                q_target = batch["rewards"] + batch["dones"].logical_not() * (
                    self.gamma
                    * (
                        q_next * self.val_interpolation
                        + v_next * (1 - self.val_interpolation)
                    )
                )
            q_input = torch.cat((batch["critic_obs"], batch["actions"]), dim=-1)
            q_loss = self.critic_q.loss_fn(q_input, q_target)
            self.critic_q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_q.parameters(), self.max_grad_norm)
            self.critic_q_optimizer.step()
            self.mean_q_loss += q_loss.item()
            counter += 1

            # TODO: check where to do polyak update (IPG repo does it here)
            self.target_critic_q = polyak_update(
                self.critic_q, self.target_critic_q, self.polyak
            )
        self.mean_q_loss /= counter

    def update_critic_v(self, data):
        self.mean_value_loss = 0
        counter = 0

        generator = create_uniform_generator(
            data,
            self.batch_size,
            max_gradient_steps=self.max_gradient_steps,
        )
        for batch in generator:
            value_loss = self.critic_v.loss_fn(batch["critic_obs"], batch["returns"])
            self.critic_v_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), self.max_grad_norm)
            self.critic_v_optimizer.step()
            self.mean_value_loss += value_loss.item()
            counter += 1
        self.mean_value_loss /= counter

    def update_actor(self, data_onpol, data_offpol):
        self.mean_surrogate_loss = 0
        self.mean_offpol_loss = 0
        counter = 0

        self.actor.update_distribution(data_onpol["actor_obs"])
        data_onpol["old_sigma"] = self.actor.action_std.detach()
        data_onpol["old_mu"] = self.actor.action_mean.detach()
        data_onpol["old_actions_log_prob"] = self.actor.get_actions_log_prob(
            data_onpol["actions"]
        ).detach()

        # Generate off-policy batches and use all on-policy data
        generator_offpol = create_uniform_generator(
            data_offpol,
            self.batch_size,
            max_gradient_steps=self.max_gradient_steps,
        )
        for batch_offpol in generator_offpol:
            self.actor.update_distribution(data_onpol["actor_obs"])
            actions_log_prob_onpol = self.actor.get_actions_log_prob(
                data_onpol["actions"]
            )
            mu_onpol = self.actor.action_mean
            sigma_onpol = self.actor.action_std

            # * KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_onpol / data_onpol["old_sigma"] + 1.0e-5)
                        + (
                            torch.square(data_onpol["old_sigma"])
                            + torch.square(data_onpol["old_mu"] - mu_onpol)
                        )
                        / (2.0 * torch.square(sigma_onpol))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    lr_min, lr_max = self.lr_range

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            lr_min, self.learning_rate / self.lr_ratio
                        )
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            lr_max, self.learning_rate * self.lr_ratio
                        )

                    for param_group in self.optimizer.param_groups:
                        # ! check this
                        param_group["lr"] = self.learning_rate

            # * On-policy surrogate loss
            adv_onpol = data_onpol["advantages"]
            if self.use_cv:
                # TODO: control variate
                critic_based_adv = 0  # get_control_variate(data_onpol, self.critic_v)
                learning_signals = (adv_onpol - critic_based_adv) * (1 - self.inter_nu)
            else:
                learning_signals = adv_onpol * (1 - self.inter_nu)

            ratio = torch.exp(
                actions_log_prob_onpol - data_onpol["old_actions_log_prob"]
            )
            ratio_clipped = torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate = -learning_signals * ratio
            surrogate_clipped = -learning_signals * ratio_clipped
            loss_onpol = torch.max(surrogate, surrogate_clipped).mean()

            # * Off-policy loss
            if self.beta == "on_policy":
                loss_offpol = self.compute_loss_offpol(data_onpol)
            elif self.beta == "off_policy":
                loss_offpol = self.compute_loss_offpol(batch_offpol)
            else:
                raise ValueError(f"Invalid beta value: {self.beta}")

            if self.use_cv:
                b = 1
            else:
                b = self.inter_nu

            loss = loss_onpol + b * loss_offpol

            # * Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.mean_surrogate_loss += loss_onpol.item()
            self.mean_offpol_loss += b * loss_offpol.item()
            counter += 1
        self.mean_surrogate_loss /= counter
        self.mean_offpol_loss /= counter

    def compute_loss_offpol(self, data):
        obs = data["actor_obs"]
        actions = self.actor.act_inference(obs)
        q_input = torch.cat((data["critic_obs"], actions), dim=-1)
        q_value = self.critic_q.evaluate(q_input)
        return -q_value.mean()
