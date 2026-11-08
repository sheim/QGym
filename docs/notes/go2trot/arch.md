# go2trot observation space, action space, and network architecture

Reference note for the `go2trot` task: what the actor and critic observe, what
the actor emits, and how the two networks are built.

All numbers below were verified by instantiating the task
(`task_registry.make_env(name="go2trot", device="cpu")`) and by building
`learning.modules.Actor` / `Critic` from the real config dicts, not by reading
config alone. Verification snippet at the end.

## Summary

| Item | Value |
|---|---|
| Actor observation dim | **48** |
| Critic observation dim | **52** |
| Action dim | **12** (one per actuated joint) |
| Actor net | MLP 48 → 256 → 256 → 128 → 12, ELU, 112,780 params + 12 std = **112,792** |
| Critic net | MLP 52 → 128 → 64 → 1, ELU, **15,105** params |
| Action distribution | diagonal `Normal(mean = net(obs), std = learned, state-independent)` |
| Observation normalization | **disabled** on both nets (`normalize_obs = False`) |
| Action clipping | **none** (unbounded Gaussian samples) |
| Control rate | 50 Hz, `dt = 0.02 s`, `decimation = 1` |
| Episode | 5 s = 250 control steps |

The actor/critic are **asymmetric**: the actor never sees base linear velocity
or base height, the critic sees both. This is the privileged-information critic
pattern — the actor is deployable from onboard sensors, the value function gets
the simulator-only signals.

## Source of truth

| What | Where |
|---|---|
| obs/action lists, net sizes, activation | `gym/envs/go2/go2trot_config.py` (`Go2TrotRunnerCfg.actor` / `.critic`) |
| Inherited runner defaults | `gym/envs/go2/go2_config.py`, `gym/envs/base/legged_robot_config.py:180` |
| Observation tensors and `dof_pos_obs` | `gym/envs/base/legged_robot.py:82-103` |
| `get_state` / scaling / `set_state` | `gym/envs/base/task_skeleton.py:19-43` |
| Phase / gait-frequency observations | `gym/envs/go2/go2trot.py:54-88` |
| Command sampling | `gym/envs/base/legged_robot.py:193`, `gym/envs/go2/go2trot.py:90` |
| Actor / critic modules | `learning/modules/actor.py`, `learning/modules/critic.py`, `learning/modules/utils/neural_net.py` |
| Action → torque path | `gym/envs/base/legged_robot.py:258` |
| Obs assembly, noise, action write-back | `learning/runners/BaseRunner.py:58-86`, `on_policy_runner.py:74` |

## How observations are assembled

`env.get_states(names)` concatenates, in list order, one block per name
(`task_skeleton.py:19`). For each block:

$$
\text{obs}_k =
\begin{cases}
\text{state}_k / s_k & \text{if } s_k \in \texttt{cfg.scaling} \\
\text{state}_k & \text{otherwise}
\end{cases}
$$

So the `scaling` dict doubles as the observation normalizer — it is applied at
read time, and the inverse is applied at write time for anything in the
`actions` list. No `RunningMeanStd` is involved, because both nets set
`normalize_obs = False`.

Active scales for go2trot (`Go2TrotCfg.scaling` + inherited `Go2Cfg.scaling`):

| scale | value |
|---|---|
| `base_ang_vel` | 0.3 |
| `base_lin_vel` | 0.4 |
| `base_height` | 0.3 |
| `commands` | [3, 1, 3] |
| `dof_pos`, `dof_pos_obs` | `4 × [1.0472, 2.53075, 0.94247]` rad |
| `dof_vel` | `4 × [2.0, 2.0, 4.0]` rad/s |
| `dof_pos_target` | `4 × [0.2, 0.3, 0.3]` rad |
| `tau_ff` | `4 × [23.7, 23.7, 45.43]` N·m (**unused**, see gotchas) |

Joint order inside every 12-wide block is canonical **FL, FR, RL, RR**, with
**hip, thigh, calf** inside each leg (`RobotLayout` order, mapped from each
backend's native order).

## Actor observation space (48)

| idx | name | dim | definition | units in obs | observed range at reset |
|---|---|---|---|---|---|
| 0-2 | `base_ang_vel` | 3 | body-frame angular velocity `/ 0.3` | rad/s ÷ 0.3 | ≈ 0 |
| 3-5 | `projected_gravity` | 3 | `quat_rotate_inverse(base_quat, [0,0,-1])` | unit vector, unscaled | [-1, 0] |
| 6-8 | `commands` | 3 | `[vx_cmd, vy_cmd, ωz_cmd] / [3,1,3]` | m/s ÷ 3, m/s ÷ 1, rad/s ÷ 3 | [-0.48, 0.67] |
| 9-20 | `dof_pos_obs` | 12 | `(q - q_default) / [1.0472, 2.53075, 0.94247]` | rad ÷ per-joint scale | 0 at reset |
| 21-32 | `dof_vel` | 12 | `q̇ / [2, 2, 4]` | rad/s ÷ per-joint scale | 0 at reset |
| 33-44 | `dof_pos_target` | 12 | `target residual / [0.2, 0.3, 0.3]` | **action units** | 0 at reset |
| 45-46 | `phase_obs` | 2 | `[sin φ, cos φ]` | unit circle, unscaled | [-0.98, 0.92] |
| 47 | `phase_frequency` | 1 | gait frequency `f` | **raw Hz, unscaled** | [1.61, 2.79] |

Notes:

- `q_default` is the static nominal posture `[hip 0.0, thigh 0.96, calf -1.36]`
  per leg (`init_state.default_joint_angles`), and `reset_mode =
  "reset_to_basic"` puts the robot exactly there, so `dof_pos_obs = 0` and
  `dof_vel = 0` at every reset.
- `dof_pos_target` is the previous action in *policy units* — i.e. the actor
  sees its own last command (scaled back by the same `dof_pos_target` scale it
  wrote). This is action feedback, not a measured quantity.
- `phase_obs` is `[sin, cos]` rather than the raw angle so the clock has no
  discontinuity at the 2π wrap. It is refreshed once per control step
  (`_post_decimation_step`).
- `phase_frequency` is deliberately left in Hz: the actor needs the absolute
  gait rate to know how fast the phase is moving.
- `base_ang_vel` and `base_lin_vel` are rotated into the body frame
  (`quat_rotate_inverse`), so they are already yaw-agnostic.

## Critic observation space (52)

The critic observes a strict superset of the actor:

| idx | name | dim | definition | units in obs |
|---|---|---|---|---|
| 0 | `base_height` | 1 | `root_z / 0.3` | m ÷ 0.3 |
| 1-3 | `base_lin_vel` | 3 | body-frame linear velocity `/ 0.4` | m/s ÷ 0.4 |
| 4-6 | `base_ang_vel` | 3 | as actor | rad/s ÷ 0.3 |
| 7-9 | `projected_gravity` | 3 | as actor | unit vector |
| 10-12 | `commands` | 3 | as actor | ÷ [3,1,3] |
| 13-24 | `dof_pos_obs` | 12 | as actor | rad ÷ scale |
| 25-36 | `dof_vel` | 12 | as actor | rad/s ÷ scale |
| 37-48 | `dof_pos_target` | 12 | as actor | action units |
| 49-50 | `phase_obs` | 2 | as actor | unit circle |
| 51 | `phase_frequency` | 1 | as actor | raw Hz |

`base_height` is `root_states[:, 2:3]`, i.e. world z, divided by 0.3 — at reset
(0.396 m spawn height) it reads ≈ 1.32.

## Observation noise

Noise is added only to the **actor** observation, in `get_noisy_obs`
(`BaseRunner.py:75`), and is applied at both training and inference time:

$$
\tilde{o} = o + \frac{\sigma_{\text{cfg}}}{s} \odot \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I)
$$

The `1/s` factor is applied only for names present in `scaling`, so the
configured numbers are in **physical units**:

| obs | configured σ | effective σ in obs units |
|---|---|---|
| `dof_pos_obs` | 0.01 rad | 0.01 / [1.0472, 2.53075, 0.94247] |
| `base_ang_vel` | 0.01 rad/s | 0.01 / 0.3 = 0.033 |
| `dof_vel` | 0.005 rad/s | 0.005 / [2, 2, 4] |

Configured but **inactive** (keys do not match any actor obs name):
`dof_pos = 0.005`, `lin_vel = 0.05`, `ang_vel = [0.3, 0.15, 0.4]`,
`gravity_vec = 0.1`. In particular `projected_gravity` gets **no** noise — the
noise dict key would have to be `projected_gravity`, not `gravity_vec`.

## Command distribution (`commands` obs)

Resampled on reset and every `resampling_time = 10 s` (i.e. effectively only at
reset, since an episode is 5 s):

- `vx`: uniform pick from the 5 centers `[-0.5, 0.0, 0.5, 1.0, 2.0]`, plus
  `N(0, 0.1)` m/s (`commands.var`).
- `vy`: uniform in `[-0.5, 0.5]` m/s (`ranges.lin_vel_y = 0.5`).
- `ωz`: uniform in `[-1.5, 1.5]` rad/s (`ranges.yaw_vel = 1.5`).
- Base class zeroes `(vx, vy)` when `‖(vx, vy)‖ ≤ 0.2`.
- Then three independent drop masks: 20% → zero `vy` and `ωz` (straight
  walking), 20% → zero `vx` and `vy` (turn in place), 10% → zero all three
  (stand still).

## Action space (12)

`actions = ["dof_pos_target"]`, so the policy output is 12 numbers, one per
actuated joint, in canonical FL/FR/RL/RR × hip/thigh/calf order.

Pipeline per control step:

1. **Sample**: $a \sim \mathcal{N}(\mu(o), \sigma)$, with $\mu$ = actor MLP and
   $\sigma$ a learned, observation-independent `nn.Parameter` of size 12
   (initialized to 1.0). No tanh squash, no clipping, no bounds.
2. **Write back** (`set_state`, `task_skeleton.py:36`): since
   `dof_pos_target ∈ scaling`, the env stores

$$
q^{\text{target}}_{\text{res}} = a \odot [0.2,\,0.3,\,0.3]_{\times 4}\ \text{[rad]}
$$

   so one policy unit means 0.2 rad at the hips, 0.3 rad at thigh and calf.
   With $\sigma = 1$ the initial exploration is therefore ±0.2/±0.3 rad of
   target jitter — this is why `actor/action_std` starts near 1.0.
3. **PD law** (`_compute_torques`, `legged_robot.py:258`), all 12 joints:

$$
\tau = K_p\,(q^{\text{target}}_{\text{res}} + q^{\text{default}} - q)
     + K_d\,(0 - \dot{q}) + \tau_{ff}
$$

   with nominal $K_p = 20$ N·m/rad and $K_d = 0.5$ N·m·s/rad, both
   **domain-randomized per episode** (`p_gains` × U[0.9, 1.1], `d_gains` ×
   U[0.8, 1.2]). $\tau_{ff} \equiv 0$ because `tau_ff` is not in the action
   list.
4. **Clip** to the URDF torque limits (23.7 N·m hip/thigh, 45.43 N·m calf).
5. **Hold** for `decimation = 1` physics substep at 50 Hz — the torque is
   recomputed every step, there is no zero-order hold.
6. The target is also pushed into `dof_pos_history` for the `action_rate` /
   `action_rate2` rewards, and is what the actor sees as `dof_pos_target`.

Because the policy commands a **residual offset** on top of a fixed nominal
posture, `a = 0` means "hold the nominal stand pose", not "zero torque".

## Actor architecture

`learning/modules/actor.py` → `Actor`, plain MLP from `create_MLP`:

```
Linear(48, 256) → ELU → Linear(256, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 12)
```

| property | value |
|---|---|
| hidden dims | [256, 256, 128] |
| activation | ELU (α = 1.0) on every hidden layer, none on the output |
| bias | yes on every layer |
| LayerNorm / Dropout | none |
| obs normalization | off — raw scaled obs go straight in |
| output | 12 means (linear, unbounded) |
| std | `nn.Parameter` init 1.0, learned jointly, **not** a network head |
| distribution | `Normal(mean, std)`, `validate_args` disabled |
| params | 112,780 (MLP) + 12 (std) = 112,792 |
| `smooth_exploration` | False → plain `Actor`, not `SmoothActor` |

Inference (`act_inference` / `play.py`) uses the same code path with the mean
only; sampling happens in `act`.

## Critic architecture

`learning/modules/critic.py` → `Critic`:

```
Linear(52, 128) → ELU → Linear(128, 64) → ELU → Linear(64, 1) → squeeze(-1)
```

| property | value |
|---|---|
| hidden dims | [128, 64] |
| activation | ELU |
| output | scalar state value, `squeeze(-1)` |
| obs normalization | off |
| params | 15,105 |
| loss | MSE to GAE returns |

The critic is roughly 7× smaller than the actor — it only regresses a scalar
value, and it is retrained from scratch each iteration against fresh returns.

## Loop timing

- `n_policy_steps = (1/dt) / actor_cfg.frequency = 50 / 50 = 1`, so one env
  step per policy step and one action per 0.02 s.
- `actor_cfg.frequency` is not set in the config; `task_registry` injects it as
  `ctrl_frequency` (50) when missing (`task_registry.py:121`).
- `num_steps_per_env = rollout_size / num_envs = 2**16 / 4096 = 16` steps per
  iteration per env.
- `add_noise` exists in the actor config and is **never read** anywhere in the
  codebase — observation noise is always on regardless of its value.

## Gotchas worth remembering

1. **`add_noise` is dead config.** Nothing reads it; noise is applied
   unconditionally in `get_noisy_obs`.
2. **`noise.scale` is dead config.** `get_noise` only looks up per-obs keys.
3. **`gravity_vec` and `ang_vel` noise entries never apply** — the actor obs
   keys are `projected_gravity` and `base_ang_vel`.
4. **`scaling.tau_ff` is unused for go2trot**, because `tau_ff` is not in the
   `actions` list, so the buffer stays zero.
5. **`phase_frequency` is unscaled** while everything physical around it is
   scaled — it is the only obs block outside the `scaling` convention.
6. **No action clipping.** Large $\sigma$ early in training directly produces
   ±0.2/±0.3 rad target jumps, which is what `action_rate` / `action_rate2`
   are there to discourage.
7. **Actor and critic never share a normalizer** (`normalize_obs = False` on
   both), so observation statistics must be scaled by hand in `cfg.scaling`.

## How to re-verify

```python
import gym.envs  # registers tasks
from gym.utils.task_registry import task_registry

env_cfg, train_cfg = task_registry.get_cfgs(name="go2trot")
task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
env_cfg.seed = 0
env_cfg.env.num_envs = 2
env = task_registry.make_env(name="go2trot", env_cfg=env_cfg, device="cpu", headless=True)

print(env.get_states(train_cfg.actor.obs).shape)    # [2, 48]
print(env.get_states(train_cfg.critic.obs).shape)   # [2, 52]
print(env.scales)
```

`get_cfgs` must be called before `make_env`, `convert_frequencies_to_params`
must run to populate `ctrl_dt` / `decimation`, and `env_cfg.seed` must be set
because the base `LeggedRobotCfg` does not define `seed` (only the runner config
does, and `make_env` reads it before writing it back). `scripts/train.py:126`
does this for real runs.
