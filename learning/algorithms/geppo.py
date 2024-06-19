from .ppo2 import PPO2
from learning.utils import (
    compute_generalized_advantages,
    compute_gae_vtrace,
    normalize,
)


# Implementation based on GePPO repo: https://github.com/jqueeney/geppo
class GePPO(PPO2):
    def __init__(self, actor, critic, is_trunc=1.0, **kwargs):
        super().__init__(actor, critic, **kwargs)

        # Importance sampling truncation
        self.is_trunc = is_trunc

    def update(self, data):
        values = self.critic.evaluate(data["critic_obs"])
        # Handle single env case
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        data["values"] = values

        # Compute V-trace GAE
        adv_vtrace, ret_vtrace = compute_gae_vtrace(
            data, self.gamma, self.lam, self.is_trunc, self.actor, self.critic
        )
        # Handle single env case
        if adv_vtrace.dim() == 1:
            adv_vtrace = adv_vtrace.unsqueeze(-1)
        if ret_vtrace.dim() == 1:
            ret_vtrace = ret_vtrace.unsqueeze(-1)
        data["advantages"] = adv_vtrace
        data["returns"] = ret_vtrace

        self.update_critic(data)
        data["advantages"] = normalize(data["advantages"])
        self.update_actor(data)

        if self.actor.store_pik:
            self.actor.update_pik_weights()

        # Logging: Store mean GAE with and without V-trace
        adv = compute_generalized_advantages(data, self.gamma, self.lam, self.critic)
        ret = adv + values

        self.adv_mean = adv.mean().item()
        self.ret_mean = ret.mean().item()
        self.adv_vtrace_mean = adv_vtrace.mean().item()
        self.ret_vtrace_mean = ret_vtrace.mean().item()
