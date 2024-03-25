import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, actor, critic, **kwargs):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, "
                "which will be ignored: " + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.actor = actor
        self.critic = critic

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
