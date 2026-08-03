---
name: q2-rl-development
description: Design, implement, or review Q2 reinforcement-learning behavior, including observations, actions, rewards, PPO2, SAC, actors and critics, storage, normalization, runner data flow, and tuning methodology. Use for changes under learning/, reward or observation code in task classes, algorithm hyperparameters, network architecture, rollout geometry, inference, checkpoint state, or training-quality claims.
---

# Q2 RL Development

Read `AGENTS.md`, the selected task and runner config, the active runner and
algorithm, and their focused tests. Confirm which configured class names are
actually instantiated; the tree includes legacy and experimental paths.

## Trace data before editing

For the main on-policy path, trace one update end to end:

1. Actor/critic observation names are resolved from environment attributes.
2. `TaskSkeleton.get_state()` divides named states by configured scales.
3. The actor emits actions; runner assignment multiplies action states by
   their scales.
4. The environment steps the backend for its configured decimation.
5. `DictStorage` records transitions for the collected temporal horizon.
6. `PPO2` performs repeated optimizer minibatches without recounting the same
   rollout as fresh normalization data.
7. The logger records total and per-term episode metrics and checkpoints model,
   optimizer, iteration, and normalization state.

Do not infer behavior from a config field alone; find its consumer.

## Observations and actions

- Keep actor and critic observation lists explicit. Every listed attribute must
  exist with stable shape; every scaled attribute needs a scaling entry.
- Keep canonical joint/body order at the task boundary. Express semantic
  subsets with cached `RobotLayout` group indices, not slices.
- Keep inference deterministic: switch modules/normalizers to evaluation mode
  and use clean observations. Noise belongs only in rollout collection.
- Update observation statistics once per fresh rollout population. Freeze them
  during repeated optimizer epochs, and save/load them with the policy.

## Rewards

- A nonzero runner-config weight selects `_reward_<name>`; zero weights should
  remove unused computation. Each reward returns shape `[num_envs]`.
- `_sqrdexp(error)` is maximized when `error == 0`. Write down the intended
  optimum before coding; do not pass a raw target quantity whose optimum is
  elsewhere.
- Use means for comparable per-joint penalties so magnitude transfers across
  robots. Use sums only when physical total magnitude is intended and tested.
- Keep termination handling separate from ordinary weighted rewards.
- When changing reward math, add a small test for its zero point, sign, scale,
  shape, and boundary behavior before training.

## Rollout and optimizer geometry

Treat these as separate controls:

- `rollout_batch_size`: total new transition samples collected per update;
  together with `num_envs`, it determines consecutive steps per environment.
- `batch_size`: optimizer minibatch size.
- `max_gradient_steps`: number of actor/critic optimization steps.

When comparing backends or configurations, hold temporal horizon and collected
sample count constant unless that variable is the experiment. Equal minibatch
size alone does not make two PPO experiments equivalent.

Discount and GAE values may be derived from physical horizons and control dt.
Re-run frequency conversion rather than editing derived values.

## Evidence workflow

1. State the proposed mechanism and a result that would falsify it.
2. Predict direction and useful numeric bounds before running.
3. Add a unit test for deterministic math, storage, normalization, inference,
   or checkpoint behavior.
4. Run same-seed, same-task, same rollout-geometry before/after experiments.
5. Compare per-term rewards, episode duration/survival, action statistics,
   losses/KL, throughput, and domain-relevant physical metrics. Do not select
   only on aggregate reward or a viewer impression.
6. Record failed or invalid experiments in `MIGRATION_PLAN.md` when they affect
   the active campaign. Keep tuning changes numerically explicit and separate
   from algorithm/refactor changes.

Validate with focused learning tests and then:

```bash
uv run --frozen python -m pytest -q
```
