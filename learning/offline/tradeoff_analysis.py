"""Portable statistics for the fixed-state original-reward tradeoff report."""

import math

import torch


def state_cluster_bootstrap(values, seed, draws=10_000):
    """Percentile CI from resampling fixed-state rows with replacement."""
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if values.numel() < 2:
        raise ValueError("state bootstrap needs at least two states")
    if draws <= 0:
        raise ValueError("draws must be positive")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(), (draws, values.numel()), generator=generator
    )
    means = values[indices].mean(-1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz)."""
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betai(a, b, x):
    if not 0.0 <= x <= 1.0:
        raise ValueError("incomplete beta argument must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(value, df):
    """Two-sided Student-t tail probability."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    value = abs(float(value))
    return _betai(0.5 * df, 0.5, df / (df + value * value))


def student_t_critical(level=0.95, df=127):
    """Two-sided critical value by bisection on the t tail probability."""
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be in (0, 1)")
    target = 1.0 - level
    low, high = 0.0, 16.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if student_t_two_sided_p(middle, df) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def paired_t_summary(values):
    """Mean, sample SD, standard error, t, two-sided p and 95% t interval."""
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    n = values.numel()
    if n < 2:
        raise ValueError("paired t summary needs at least two values")
    mean = float(values.mean())
    sd = float(values.std(unbiased=True))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0.0 else math.inf
    critical = student_t_critical(0.95, n - 1)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": t,
        "df": n - 1,
        "p_two_sided": student_t_two_sided_p(t, n - 1),
        "ci95": [mean - critical * se, mean + critical * se],
        "critical": critical,
    }
