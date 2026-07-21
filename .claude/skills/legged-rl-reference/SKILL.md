---
name: legged-rl-reference
description: The RL domain knowledge pack for Q2 — how observations/actions are scaled and assembled, how reward functions are structured (_sqrdexp, tracking_sigma, per-joint means, switch/oscillator machinery), PPO2/SAC hyperparameter semantics, discount-from-horizon derivation, storage/runner data flow, and which learning components are production vs experimental vs dead. Load when designing/tuning rewards or observations, choosing algorithm knobs, or reading learning/ code. NOT for backend physics (mujoco-backend-reference) or config file locations (q2-config-system).
---

# Legged RL Reference (Q2 conventions)

## Observation/action plumbing

- The runner asks the env for observations by NAME list:
  `train_cfg.actor.obs` / `train_cfg.critic.obs` are lists of env attribute
  names; `TaskSkeleton.get_states(obs_list)` concatenates
  `getattr(env, name) / scale` per name (`gym/envs/base/task_skeleton.py:19`).
  Scales come from `env_cfg.scaling.<name>`; **observations are divided by
  scale, actions multiplied on the way back in** (`set_states`). Get→set is
  round-trip identity (unit-tested).
- Actions: `runner.set_actions(actor_cfg["actions"], actions, disable_actions)`
  writes scaled targets; `disable_actions` is a debug flag that has leaked into
  configs before (`bfd6b2d`) — check it when a robot is limp.
- `randomize_episode_counters(env)` desynchronizes episode clocks at start so
  parallel envs don't reset in lockstep.

## Reward system

- Discovery: runner builds `{name: bound _reward_<name>}` from the **nonzero**
  entries of `train_cfg.critic.reward.weights` (zero-weight = function never
  called; `remove_zero_weighted_rewards`). `termination_weight` is separate.
  Every `_reward_*` must return shape `[num_envs]`.
- **`_sqrdexp(x) = exp(−x²/σ)` peaks at x = 0.** Always express the argument as
  an ERROR (0 = perfect). The pendulum once learned to lie horizontal because
  it was fed `cos θ` instead of `1 − cos θ` (commit `95b9a6e`).
  `reward_settings.tracking_sigma` is the shared width parameter.
- **Per-joint rewards use `torch.mean` over joints, not sum** (`c5a598f`), so
  weights transfer across 1-DOF pendulum → 12-DOF cheetah → 18-DOF humanoid.
  Weights were rescaled ~×10 when this landed; old branches may still assume
  sums.
- Tracking rewards scale error by `1/(1+|cmd|)` (relative accuracy at speed,
  see `mini_cheetah.py:_reward_tracking_lin_vel`).
- **Switch machinery** (ref/osc tasks): a smooth phase indicator `self._switch`
  computed once per decimation (`7296c6c`), width `reward_settings.switch_scale`.
- **Oscillator (OSC) tasks**: per-leg CPG oscillators with ground-reaction-force
  feedback; key knobs `osc.omega`, `osc.coupling`, `grf_threshold` normalized by
  `MINI_CHEETAH_WEIGHT = 8.292 * 9.81` N. Beware: osc randomization can
  overwrite omega/coupling (noted in `c0bda90`).
- **Reference-tracking tasks** (`mini_cheetah_ref`): `reference_traj` reward
  against trajectories in `resources/robots/mini_cheetah/trajectories/`;
  `swing_grf`/`stance_grf` shape foot loading (they need foot bodies →
  fusestatic off). Current weights = jt/port's July tuning, pulled into
  `vsim` 2026-07-11 (reference_traj 3.0, grf 1.5/1.5, orientation 1.5,
  min_base_height 1.0, action_rate2 0.05; ang-vel divisor 2.5;
  scaling.base_height 0.15) — adopted because it "trains better" (Steve),
  now running for the first time with both live obs AND full constraints.
- **PBRS (potential-based reward shaping) is DEAD** since `ea5cdff` (calls a
  deleted env API). Repair before use; see q2-failure-archaeology.

## Algorithms — what's real

| Component | Status |
|---|---|
| `PPO2` (`learning/algorithms/ppo2.py`) | production on-policy default |
| `SAC` (`learning/algorithms/sac.py`) + `OffPolicyRunner` | working for pendulum-class tasks (`sac_pendulum`, `sac_mini_cheetah` registered); research-grade |
| `PPO` + `OldPolicyRunner` | deprecated, kept for compat — use PPO2 |
| `StateEstimator` (SE.py) | isolated supervised module, not wired to runners |
| QRCritics zoo (`learning/modules/QRCritics.py`) | experimental research critics; only via CustomCriticRunner/PSACRunner; known latent bug: `matfncs.py` uses `F.softplus` without importing F (NameError if PDCholesky* used) |
| `SmoothActor` (gSDE), `ChimeraActor` | experimental |

PPO2 knobs that matter (in `train_cfg.algorithm`): `batch_size` (drives derived
`num_steps_per_env = max(1, batch_size // num_envs)` — commit `936a4cc`),
`max_gradient_steps`, `clip_param`, `desired_kl` + `schedule="adaptive"`
(LR adapts ×2/÷2 around desired KL), `gamma`, `lam`, `entropy_coef`,
`learning_rate` + `lr_range`.

**Discounts from horizons:** `set_discount_from_horizon(dt, horizon)`
(`learning/utils/utils.py`) converts a time-horizon in seconds to γ given the
control dt; `task_registry.set_discount_rates` applies it. Change frequencies
and your effective γ changes with them — that's intended.

## Data flow (one iteration, OnPolicyRunner)

```
for n in range(num_steps_per_env):            # derived, see above
    obs → actor.act → env.step(actions)       # env internally decimates sim steps
    transitions → DictStorage (TensorDict, [max_len, num_envs, ...])
PPO2.update(storage)                          # minibatches of batch_size
Logger: episode-windowed reward means (window=100 episodes), steps/s
checkpoint every runner.save_interval → model_<it>.pt
```

Off-policy uses `ReplayBuffer` (circular) + `initial_fill` prefill.
Storage overflow raises — it does not silently wrap (on-policy).

## Normalization & networks

- `RunningMeanStd` obs normalization per actor/critic when
  `normalize_obs=True` (Welford, in `learning/modules/utils/normalize.py`).
- `create_MLP` (`learning/modules/utils/neural_net.py`) — activations incl.
  elu/tanh/mish, optional LayerNorm/dropout; `export_network()` traces to
  TorchScript + ONNX (exists but no MuJoCo-side export script wires it up yet).

## Domain randomization & robustness knobs

- `push_robots` (periodic velocity kicks) works on MuJoCo backends.
- Friction/base-mass randomization are configured but **unverified under
  MuJoCo** — the callbacks that applied them are IsaacGym-only. See
  q2-config-system and campaign Phase 3 before relying on them.

## When NOT to use this skill

- Where a config field lives / what the backend consumes → `q2-config-system`.
- Physics/step mechanics → `mujoco-backend-reference`.
- "Training is broken" → `q2-debugging-playbook`.
- Standards of proof for a tuning claim → `q2-research-methodology`.

## Provenance and maintenance

Verified 2026-07-10 against `port` @ `bc2bd96`. Re-verify:

```bash
grep -n "num_steps_per_env" learning/runners/on_policy_runner.py   # still derived?
grep -n "torch.mean\|torch.sum" gym/envs/base/legged_robot.py | grep _reward | head   # per-joint mean intact?
grep -rn "compute_reward" learning/utils/PBRS/                     # PBRS still dead?
grep -n "import torch.nn.functional" learning/modules/utils/matfncs.py || echo "F.softplus bug still present"
```
