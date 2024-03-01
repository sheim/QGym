import torch.nn as nn

from .actor import Actor, SmoothActor
from .critic import Critic
from .utils import StateDependentNoiseDistribution


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        normalize_obs=True,
        smooth_exploration=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, "
                "which will be ignored: " + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()

        if smooth_exploration:
            self.actor = SmoothActor(
                num_actor_obs,
                num_actions,
                actor_hidden_dims,
                activation,
                init_noise_std,
                normalize_obs,
            )
        else:
            self.actor = Actor(
                num_actor_obs,
                num_actions,
                actor_hidden_dims,
                activation,
                init_noise_std,
                normalize_obs,
            )

        self.critic = Critic(
            num_critic_obs, critic_hidden_dims, activation, normalize_obs
        )

        print(f"Actor MLP: {self.actor.NN}")
        print(f"Critic MLP: {self.critic.NN}")

        # TODO[lm]: Decide how to handle the state dependent noise distribution in
        # this class. In stable-baselines3 there is a class MlpExtractor which does
        # what the Actor and Critic classes do here, just with the latent representation
        # of the networks. Either I make a new class in a similar way and store the
        # action distribution here, or I make the changes in Actor and Critic and change
        # the distribution there.
        if smooth_exploration:
            self.action_dist = StateDependentNoiseDistribution(
                num_actions,
                num_actor_obs,
                num_critic_obs,
            )

    @property
    def action_mean(self):
        return self.actor.action_mean

    @property
    def action_std(self):
        return self.actor.action_std

    @property
    def entropy(self):
        return self.actor.entropy

    @property
    def std(self):
        return self.actor.std

    def update_distribution(self, observations):
        self.actor.update_distribution(observations)

    def act(self, observations, **kwargs):
        return self.actor.act(observations)

    def get_actions_log_prob(self, actions):
        return self.actor.get_actions_log_prob(actions)

    def act_inference(self, observations):
        return self.actor.act_inference(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic.evaluate(critic_observations)

    def export_policy(self, path):
        self.actor.export(path)
