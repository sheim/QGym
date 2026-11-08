# go2trot reward function

Reference note for the `go2trot` task: what each reward term computes, the math,
and the coefficients currently configured.

## Source of truth

| What | Where |
|---|---|
| Task class, reward overrides | `gym/envs/go2/go2trot.py` |
| Coefficients (`critic.reward`) | `gym/envs/go2/go2trot_config.py` |
| Inherited/base reward implementations | `gym/envs/base/legged_robot.py` |
| Reward aggregation | `learning/runners/BaseRunner.py:108`, `learning/runners/on_policy_runner.py:152` |
| Env constants (`dt`, scales, `tracking_sigma`) | `gym/envs/base/task_skeleton.py:76`, `gym/envs/go2/go2_config.py`, `gym/envs/base/legged_robot_config.py:128` |

## How the total is assembled

Reward functions are discovered by name: every `_reward_<name>` method on the
env becomes `reward_functions["<name>"]` (`BaseRunner.setup_reward_functions`),
and a weight is looked up by that same key.

At startup, every weight equal to `0.0` is **deleted** from the config
(`remove_zero_weighted_rewards`), so a zero-weighted term is never evaluated and
its `_reward_*` method need not even exist.

Per step:

$$
r_t \;=\; \Delta t \left( \sum_{i \,\in\, W} w_i \, r_i \;+\; \mathbb{1}[\text{terminated}]\, w_{\text{term}}\, r_{\text{term}} \right),
\qquad \Delta t = \frac{1}{f_{\text{ctrl}}} = \frac{1}{50} = 0.02\ \text{s}
$$

- `termination` comes from the separate `termination_weight` dict and is masked
  by `env.terminated`.
- All normal terms are masked by `~env.terminated`.
- Episode length is 5 s ⇒ 250 control steps.
- `decimation = ctrl_frequency / desired_sim_frequency = 50/50 = 1`.

**Logging gotcha:** values written to `vitals.jsonl` are
`episode_sum / episode_time`, i.e. per-second rates. So
`rewards/<name> ≈ w_i · r̄_i` (this is why `tracking_lin_vel` shows ≈ 2.4 rather
than `4 · 0.02`).

Shared constants:

- `tracking_sigma = 0.25` (note: used as the *denominator*, i.e. a
  variance-like parameter, not a standard deviation)
- `_sqrdexp(x) = exp(-x² / 0.25)`
- `base_height_target = 0.4` (`BASE_HEIGHT_REF`)
- `scaling.base_height = 0.3`, `scaling.base_ang_vel = 0.3`,
  `scaling.base_lin_vel = 0.4`

## Active terms and coefficients

| weight key | w | formula | range |
|---|---|---|---|
| `tracking_lin_vel` | **4.0** | eq. 1 | (0, 1] |
| `tracking_ang_vel` | **4.0** | eq. 2 | (0, 1] |
| `trot_support` | **3.0** | eq. 9 | [0, 1] |
| `swing_contact` | **3.0** | eq. 10 | [-1, 0] |
| `reference_traj` | **3.0** | eq. 11 | ≈ (-∞, 1] |
| `orientation` | **1.0** | eq. 3 | (0, 2] |
| `min_base_height` | **0.5** | eq. 4 | (0, 1] |
| `action_rate` | **0.1** | eq. 6 | ≤ 0 |
| `action_rate2` | **0.05** | eq. 7 | ≤ 0 |
| `ang_vel_xy` | **0.01** | eq. 5 | (0, 2] |
| `torques` | **5.0e-6** | eq. 12 | ≤ 0 |
| `termination` | **0.01** | eq. 13 | {-1, 0} |

Zero / removed (present in config, stripped at startup): `lin_vel_z`,
`dof_vel`, `stand_still`, `dof_pos_limits`, `feet_contact_forces`,
`dof_near_home`.

> `stand_still` has no implementation anywhere on the go2 class hierarchy. It is
> safe only because it is zero-weighted and therefore removed before use. Do not
> set it non-zero without adding the method.

## The individual formulas

### 1. Linear velocity tracking — `tracking_lin_vel`, w = 4.0

Overrides the base mean-over-axes version:

$$
e_i=\frac{v_i^{\text{cmd}}-v_i}{1+|v_i^{\text{cmd}}|},\quad i\in\{x,y\},
\qquad
r=\exp\!\left(-\frac{e_x^2+e_y^2}{0.25}\right)
$$

Per-axis normalization means a large command tolerates proportionally larger
absolute error; at zero command it reduces to the plain error.

### 2. Yaw-rate tracking — `tracking_ang_vel`, w = 4.0

$$
r=\exp\!\left(-\frac{\left((\omega_z^{\text{cmd}}-\omega_z)/2.5\right)^2}{0.25}\right)
=\exp\!\left(-\left(\frac{\Delta\omega_z}{1.25}\right)^2\right)
$$

### 3. Orientation — `orientation`, w = 1.0

A positive reward for being upright, not a penalty:

$$
r=\sum_{i\in\{x,y\}}\exp\!\left(-\frac{g_i^2}{0.25}\right),
\qquad g=\text{projected gravity (body frame)}
$$

### 4. Base height floor — `min_base_height`, w = 0.5

$$
r=\exp\!\left(-\frac{\left(\min\left(\frac{h-0.4}{0.3},\,0\right)\right)^2}{0.25}\right)
$$

Only heights **below** 0.4 m are penalized; there is no reward for being too
tall.

### 5. Roll/pitch rate damping — `ang_vel_xy`, w = 0.01

Also a positive-reward form (unlike the base penalty version):

$$
r=\sum_{i\in\{x,y\}}\exp\!\left(-\frac{(\omega_i/0.3)^2}{0.25}\right)
$$

### 6. Action rate — `action_rate`, w = 0.1

With $a_t$ the `dof_pos_target` history (commanded residual targets, `n = 12`):

$$
r=-\frac{1}{n}\sum_j\left[(a_t-a_{t-1})_j^2+\left(\frac{a_t-a_{t-2}}{2}\right)_j^2\right]
$$

Mixes a one-step difference with a two-step difference normalized by 2, so both
a fast oscillation and a slow drift are penalized.

### 7. Action acceleration — `action_rate2`, w = 0.05

$$
r=-\frac{1}{n}\sum_j\left(a_t-2a_{t-1}+a_{t-2}\right)_j^2
$$

Discrete second difference: penalizes jerky changes in the commanded target.

### 8. Gait clock

Phase $\varphi$ advances at $f \in [1,3]$ Hz (`control.gait_freq`) and is
resampled per episode together with a random initial phase. Leg phase and
scheduled-stance signal:

$$
\varphi_{\ell}=(\varphi+\psi_\ell)\bmod 2\pi,\quad
\psi=2\pi\,[\,\text{FL}=0,\ \text{FR}=0.5,\ \text{RL}=0.5,\ \text{RR}=0\,],
\qquad s_\ell=\sin\varphi_\ell
$$

A leg is in *scheduled stance* when $s_\ell>0$. Exactly one diagonal (FL+RR or
FR+RL) is in stance at any moment — that is what makes this a trot.

Contact strength is a smoothstep of vertical foot load normalized by body
weight $W = 9.81\sum_b m_b$:

$$
c_\ell=\mathrm{clamp}\!\left(\frac{F_{\ell,z}}{W},0,1\right)^2
\left(3-2\,\mathrm{clamp}\!\left(\frac{F_{\ell,z}}{W},0,1\right)\right)
$$

### 9. Trot support — `trot_support`, w = 3.0

The scheduled diagonal must actually carry the body:

$$
\sigma_\ell=\begin{cases}c_\ell & s_\ell>0\\ 1 & s_\ell\le 0\end{cases}
\qquad
r=\Big(\max_\ell \mathrm{relu}(s_\ell)\Big)\cdot
\min\!\left(2\sqrt{\textstyle\prod_\ell \sigma_\ell},\,1\right)
$$

Two readings worth remembering:

- The gate $\max_\ell \mathrm{relu}(s_\ell) = |\sin\varphi|$ peaks at mid-stance
  and is 0 at the diagonal swap, so it rewards *holding* the trot through stance
  rather than just being near the switch.
- Swing legs contribute $\sigma=1$ to the product, so the product reduces to the
  stance diagonal's two strengths. The factor 2 makes the load term saturate at
  $c_\ell=0.5$ each, i.e. full credit exactly when the stance diagonal carries
  about one body weight (half per foot). A single loaded foot scores 0.

### 10. Swing contact — `swing_contact`, w = 3.0

Penalizes load carried by feet during their scheduled swing phase:

$$
r=-\frac{1}{4}\sum_\ell \mathrm{relu}(-s_\ell)\,c_\ell
$$

### 11. Reference trajectory — `reference_traj`, w = 3.0

Track an open-loop joint-space trot:

$$
q^{\text{ref}}_j = o_j + A_j\sin(\varphi+\psi_j),\qquad
e_j=\frac{q^{\text{ref}}_j+q^{\text{def}}_j-q_j}{s_j}
$$

$$
r=\frac{1}{n}\sum_j\left[\exp\!\left(-\frac{e_j^2}{0.25}\right)-0.2\,|e_j|\right]
$$

Per leg, in canonical **FL, FR, RL, RR** order with **hip, thigh, calf** inside
each leg:

| symbol | value |
|---|---|
| $o$ (`gait_joint_offsets`) | 0 for all 12 joints |
| $A$ (`gait_joint_amplitudes`) | $[\,0,\,-0.15,\,+0.30\,]$ repeated per leg |
| $\psi_j$ | leg phase offset repeated 3× ⇒ FL/RR = 0, FR/RL = π |
| $s_j$ (`scaling.dof_pos`) | $[1.0472,\,2.53075,\,0.94247]$ rad per leg |

The amplitudes make thighs swing back and calves extend, which approximately
preserves fore-aft foot position while lengthening the stance diagonal and
shortening the swing diagonal. $q^{\text{ref}}$ is added to `default_dof_pos` for
comparison; it is a **reward-only** reference and is never used as a PD target.

The $+0.2|e|$ term is why this is not a pure Gaussian: it is a linear (L1) pull
that keeps improving the reward for shrinking residual error even after the
exponential has saturated.

### 12. Torques — `torques`, w = 5.0e-6

$$
r=-\frac{1}{n}\sum_j \tau_j^2
$$

Effectively negligible at this weight (observed ≈ -6e-4 per second).

### 13. Termination — `termination`, w = 0.01

$$
r=-\mathbb{1}[\text{terminated}]
$$

Applied only on the terminal step, and multiplied by $\Delta t$:
$-0.01 \cdot 0.02 = -2\times10^{-4}$. Very small by design — termination is
enforced mostly through losing all subsequent reward, not through this penalty.

## Sanity check against a real run

From `logs/go2trot/Nov03_17-21-34_gpu-600-baseline-20261103/vitals.jsonl`,
iteration 1. Remember these are per-second rates, so divide by the weight to get
the mean per-step reward:

| term | logged | mean $r_i$ |
|---|---|---|
| `tracking_lin_vel` | 2.41 | ≈ 0.60 |
| `tracking_ang_vel` | 2.76 | ≈ 0.69 |
| `trot_support` | 0.42 | ≈ 0.14 |
| `swing_contact` | -0.20 | ≈ -0.07 |
| `reference_traj` | 2.32 | ≈ 0.77 |
| `orientation` | 1.86 | ≈ 0.93 per axis |
| `min_base_height` | 0.27 | ≈ 0.54 |
| `action_rate` | -0.018 | ≈ -0.18 |
| `action_rate2` | -0.021 | ≈ -0.43 |
| `ang_vel_xy` | 0.00097 | ≈ 0.097 |
| `torques` | -0.00061 | ≈ -121 (unscaled) |
| `termination` | 0.0 | 0 |
| `total_rewards` | 9.80 | ≈ 0.196 per step |

$0.196 \approx \Delta t \sum_i w_i \bar r_i = 0.02 \times 9.80$. Consistent.

At iteration 1 the gait terms are the weak ones (`trot_support` ≈ 0.14,
`swing_contact` ≈ -0.07), which is expected: the trot clock is enforced mostly
through `reference_traj` (≈ 0.77) early in training and should be picked up by
the contact terms as the policy learns to load the stance diagonal.
