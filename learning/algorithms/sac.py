import torch

# import torch.nn as nn
import torch.optim as optim

from learning.utils import create_uniform_generator, polyak_update


class SAC:
    def __init__(
        self,
        actor,
        critic_1,
        critic_2,
        target_critic_1,
        target_critic_2,
        batch_size=2**15,
        max_gradient_steps=10,
        action_max=1.0,
        action_min=-1.0,
        actor_noise_std=1.0,
        log_std_max=4.0,
        log_std_min=-20.0,
        alpha=0.2,
        alpha_lr=1e-4,
        target_entropy=None,
        max_grad_norm=1.0,
        polyak=0.995,
        gamma=0.99,
        # clip_param=0.2,  # * PPO
        # gamma=0.998,
        # lam=0.95,
        # entropy_coef=0.0,
        actor_lr=1e-4,
        critic_lr=1e-4,
        # max_grad_norm=1.0,
        # use_clipped_value_loss=True,
        # schedule="fixed",
        # desired_kl=0.01,
        device="cpu",
        **kwargs,
    ):
        self.device = device

        # self.desired_kl = desired_kl
        # self.schedule = schedule
        # self.lr = lr

        # * PPO components
        self.actor = actor.to(self.device)
        self.critic_1 = critic_1.to(self.device)
        self.critic_2 = critic_2.to(self.device)
        self.target_critic_1 = target_critic_1.to(self.device)
        self.target_critic_2 = target_critic_2.to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.log(torch.tensor(alpha)).requires_grad_()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # this is probably something to put into the neural network
        self.action_max = action_max
        self.action_min = action_min

        self.action_delta = (self.action_max - self.action_min) / 2.0
        self.action_offset = (self.action_max + self.action_min) / 2.0

        self.max_grad_norm = max_grad_norm
        self.target_entropy = (
            target_entropy if target_entropy else -self.actor.num_actions
        )

        # * PPO parameters
        # self.clip_param = clip_param
        # self.batch_size = batch_size
        self.max_gradient_steps = max_gradient_steps
        # self.entropy_coef = entropy_coef
        # self.gamma = gamma
        # self.lam = lam
        # self.max_grad_norm = max_grad_norm
        # self.use_clipped_value_loss = use_clipped_value_loss
        # * SAC parameters
        self.learning_starts = 100
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = gamma
        # self.ent_coef = "fixed"
        # self.target_entropy = "fixed"

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def switch_to_train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

    def switch_to_eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def act(self, obs):
        mean, std = self.actor.forward(obs, deterministic=False)
        distribution = torch.distributions.Normal(mean, std)
        actions = distribution.rsample()

        ## * self._scale_actions(actions, intermediate=True)
        actions_normalized = torch.tanh(actions)
        # RSL also does a resahpe(-1, self.action_size), not sure why
        actions_scaled = (
            actions_normalized * self.action_delta + self.action_offset
        ).clamp(self.action_min, self.action_max)
        return actions_scaled

    def update(self, data):
        generator = create_uniform_generator(
            data,
            self.batch_size,
            max_gradient_steps=self.max_gradient_steps,
        )

        count = 0
        self.mean_actor_loss = 0
        self.mean_alpha_loss = 0
        self.mean_critic_1_loss = 0
        self.mean_critic_2_loss = 0

        for batch in generator:
            self.update_critic(batch)
            self.update_actor_and_alpha(batch)

            # Update Target Networks
            self.target_critic_1 = polyak_update(
                self.critic_1, self.target_critic_1, self.polyak
            )
            self.target_critic_1 = polyak_update(
                self.critic_1, self.target_critic_1, self.polyak
            )
            count += 1
        self.mean_actor_loss /= count
        self.mean_alpha_loss /= count
        self.mean_critic_1_loss /= count
        self.mean_critic_2_loss /= count

        return None

    def update_critic(self, batch):
        critic_obs = batch["critic_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_actor_obs = batch["next_actor_obs"]
        next_critic_obs = batch["next_critic_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            # * self._sample_action(actor_next_obs)
            mean, std = self.actor.forward(next_actor_obs, deterministic=False)
            distribution = torch.distributions.Normal(mean, std)
            actions = distribution.rsample()

            ## * self._scale_actions(actions, intermediate=True)
            actions_normalized = torch.tanh(actions)
            # RSL also does a resahpe(-1, self.action_size), not sure why
            actions_scaled = (
                actions_normalized * self.action_delta + self.action_offset
            ).clamp(self.action_min, self.action_max)
            ## *
            # action_logp = distribution.log_prob(actions).sum(-1) - torch.log(
            #     1.0 - actions_normalized.pow(2) + 1e-6
            # ).sum(-1)
            action_logp = (
                distribution.log_prob(actions)
                - torch.log(1.0 - actions_normalized.pow(2) + 1e-6)
            ).sum(-1)

            # * returns target_action = actions_scaled, target_action_logp = action_logp
            target_action = actions_scaled
            target_action_logp = action_logp

            # * self._critic_input
            # ! def should put the action computation into the actor
            target_critic_in = torch.cat((next_critic_obs, target_action), dim=-1)
            target_critic_prediction_1 = self.target_critic_1.forward(target_critic_in)
            target_critic_prediction_2 = self.target_critic_2.forward(target_critic_in)

            target_next = (
                torch.min(target_critic_prediction_1, target_critic_prediction_2)
                - self.alpha * target_action_logp
            )
            target = rewards + self.gamma * dones.logical_not() * target_next

        critic_in = torch.cat((critic_obs, actions), dim=-1)

        # critic_prediction_1 = self.critic_1.forward(critic_in)
        critic_loss_1 = self.critic_1.loss_fn(critic_in, target)
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        # nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        self.critic_1_optimizer.step()

        # critic_prediction_2 = self.critic_2.forward(critic_in)
        critic_loss_2 = self.critic_2.loss_fn(critic_in, target)
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        # nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_2_optimizer.step()

        self.mean_critic_1_loss += critic_loss_1.item()
        self.mean_critic_2_loss += critic_loss_2.item()

        return

    def update_actor_and_alpha(self, batch):
        actor_obs = batch["actor_obs"]
        critic_obs = batch["critic_obs"]

        mean, std = self.actor.forward(actor_obs, deterministic=False)
        distribution = torch.distributions.Normal(mean, std)
        actions = distribution.rsample()

        ## * self._scale_actions(actions, intermediate=True)
        actions_normalized = torch.tanh(actions)
        # RSL also does a resahpe(-1, self.action_size), not sure why
        actions_scaled = (
            actions_normalized * self.action_delta + self.action_offset
        ).clamp(self.action_min, self.action_max)
        ## *
        action_logp = distribution.log_prob(actions).sum(-1) - torch.log(
            1.0 - actions_normalized.pow(2) + 1e-6
        ).sum(-1)

        # * returns target_action = actions_scaled, target_action_logp = action_logp
        actor_prediction = actions_scaled
        actor_prediction_logp = action_logp

        alpha_loss = (
            -self.log_alpha * (action_logp + self.target_entropy).detach()
        ).mean()
        # -(self.log_alpha * (action_logp + self.target_entropy)).detach().mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        critic_in = torch.cat((critic_obs, actor_prediction), dim=-1)
        q_value_1 = self.critic_1.forward(critic_in)
        q_value_2 = self.critic_2.forward(critic_in)
        actor_loss = (
            self.alpha.detach() * actor_prediction_logp
            - torch.min(q_value_1, q_value_2)
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.mean_alpha_loss += alpha_loss.item()
        self.mean_actor_loss += actor_loss.item()
