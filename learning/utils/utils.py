def remove_zero_weighted_rewards(reward_weights):
    for name in list(reward_weights.keys()):
        if reward_weights[name] == 0:
            reward_weights.pop(name)


def set_discount_from_horizon(dt, horizon):
    """Calculate a discount-factor from the desired discount horizon,
    and the time-step (dt).
    """
    assert dt > 0, "Invalid time-step"
    if horizon == 0:
        discount_factor = 0
    else:
        assert horizon >= dt, "Invalid discounting horizon"
        discrete_time_horizon = int(horizon / dt)
        discount_factor = 1 - 1 / discrete_time_horizon

    return discount_factor


def polyak_update(online, target, polyak_factor):
    for op, tp in zip(online.parameters(), target.parameters()):
        tp.data.copy_((1.0 - polyak_factor) * op.data + polyak_factor * tp.data)
    return target
