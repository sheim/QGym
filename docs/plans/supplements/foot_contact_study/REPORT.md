# Foot contact-force study — go2trot

Status: inspection + measurement study. **No product code was edited.** Everything
here is new files under
`docs/plans/supplements/foot_contact_study/`.

Purpose: establish (a) how to extract the foot ground-reaction force from the
simulator, (b) how that force is distributed across each end effector over the
locomotion cycle, and (c) what that implies for an offline contact-constraint
reward `r'` whose `Q_r'` sees only the actor observation.

## TL;DR

- The force is already exposed per body: `env.contact_forces[:, feet, :]`,
  world frame, Newtons, net external contact force on that body, refreshed in
  place after every `step()`. `env.feet_indices` gives the canonical foot order
  (`FL, FR, RL, RR`). Verified: the vertical component summed over **all** bodies
  is 1.000 bodyweight in a settled probe.
- Decomposition check: the published per-body force equals the sum of that
  body's MuJoCo contacts exactly (`0.0 N` error over 800 samples). The feet only
  ever touch the ground here (0 foot/self contacts), so the foot force is a pure
  ground reaction; the calf shells touch the ground in `12.2 %` of control
  steps, and `go2trot` has no active `collision` weight for them.
- This is a real trot: the two diagonal pairs alternate and carry
  `0.490` / `0.493` bodyweight on average. Per-foot stance load is
  `0.48 bw` (front) and `0.36-0.40 bw` (rear), peaking at mid-stance
  (`pi/2` of the leg phase).
- The gait is **not** a 50 %-duty walk: measured duty at a 5 % bodyweight contact
  threshold is `FL 0.64, FR 0.59, RL 0.45, RR 0.41`, and **10.4 %** of control
  steps are airborne. Total vertical force therefore swings between `~0.0` and
  `~1.25 bw` inside one cycle even though its cycle mean is `0.983 bw`.
  An instantaneous "sum of foot forces = bodyweight" constraint would penalize
  the natural gait; the constraint has to be **phase/schedule-conditioned**.
- The phase clock is aligned with physical contact: the best sub-step offset is
  `0.00 rad` and the best whole-step lag is `0` (P(contact | scheduled stance)
  `0.813` vs P(contact | scheduled swing) `0.237`). The task computes rewards
  after `env.step()`, i.e. with the post-step phase, so no correction is needed.
- The frozen policy already gives a usable constraint signal: stance feet carry
  `>= 0.20 bw` in 76 % of scheduled-stance samples, and swing feet stay below
  `0.10 bw` in 81 % of scheduled-swing samples. The violations concentrate in
  the **front feet** (35 % / 29 % of their scheduled-swing samples are above
  `0.1 bw`, versus 6-7 % for the rear feet), so front-foot grazing is the
  behaviour a contact reward would target.
- Peak force per touchdown (loaded stance: >= 5 control steps and >= 0.30 bw):
  median `0.71-0.72 bw` front, `0.62-0.63 bw` rear; `p95` `0.97-1.06 bw` front and
  `0.83 bw` rear; the tail matters more than the median — `3.7 %` of FL and
  `7.9 %` of FR touchdowns exceed bodyweight, maxima are `1.46` / `1.68 bw`.
  Per-cycle maximum is `0.73 bw` median with a within-environment cycle-to-cycle
  sd of `0.10-0.12 bw`. Peak rises with speed (`p95 0.86 -> 1.07 bw`).
- The force seen during scheduled swing is **real ground contact**, not inertia or
  self-collision: all foot contacts in the probe are with the ground plane, and
  the swing-phase contacts split evenly between late lift-off (266) and early
  touchdown (266). They are also light: only `11.9 %` of the foot force impulse
  falls outside the clock's 50 % stance window (`8.3 %` at a 60 % window).
- The actor observation predicts the per-foot load with held-out
  `R^2 = 0.87`, the contact flag with `0.75`, and a candidate scalar `r'` with
  `R^2 ~ 0.47-0.50`. That is enough signal to learn `Q_r'` from `(o, a)`, but the
  reward is only half-explained by the observation, and the residual concentrates
  at contact transitions (`R^2 ~ 0.50` near phase 0 and `pi`). This is the
  expected limit, not a bug: the transition is where the foot is genuinely
  ambiguous from joint state.
- The 2M-transition `hist-mix-20261110` dataset **cannot** be relabelled with
  contacts. It is `brg-offline-replay-v1` (core fields only: actor obs, action,
  reward-free), with no `contact_forces`, no `root_states`, no `dof_pos`. The
  frozen policy has to be replayed and re-recorded with
  `--replay_fields=contacts` (v2 format, `+1,008 B/transition`).

## 1. How to extract the foot contact force

### Contract

`SimBackend.contact_forces` → `[num_envs, num_bodies, 3]`, float32, on the
backend device, canonical `RobotLayout` body order, **world frame**, **Newtons**,
net external contact force on that body (not a per-contact list, no CoP, no
friction decomposition). It is a live tensor: refreshed in place after `step()`
and after resets.

Both MuJoCo backends populate it the same way, after the constraint solve:

| backend | source | line |
|---|---|---|
| CPU | `mj_rnePostConstraint` → `d.cfrc_ext[body, 3:6]` | `gym/envs/base/mujoco_cpu_backend.py:204,256` |
| Warp | `rne_postconstraint` → `d.cfrc_ext[..., 3:6]` | `gym/envs/base/mujoco_warp_backend.py:9,141,285` |

Because the values come from the post-constraint RNE, they are the solved
contact wrench, not a penetration-based estimate, and they exclude gravity.

### Task-level access

```python
feet = env.feet_indices  # canonical body indices, [4]
fz = env.contact_forces[:, feet, 2]  # [num_envs, 4] N, world +z
bodyweight = env._backend.link_mass.sum(dim=1) * 9.81  # [num_envs] N
load = fz / bodyweight[:, None]  # fraction of bodyweight
```

`env.feet_indices` is resolved from `robot_layout.body_groups["feet"]` during
initialization (`gym/envs/base/legged_robot.py`), and `Go2Trot._foot_contact_strength`
already uses exactly this path with the link-mass bodyweight normalisation. On
this layout the indices are `[11, 20, 29, 38]` for `FL_foot, FR_foot, RL_foot,
RR_foot` out of 42 canonical bodies.

Two practical notes:

- The tensor is per **body**, not per contact. A foot resting on the ground with
  several contact points still reports one net vector. A reward can constrain the
  net force but cannot see centre of pressure or individual contact points.
- `link_mass` is domain-randomised per environment at startup
  (`16.8-17.7 kg` here), so always normalise by the per-env bodyweight, not a
  constant.

### Is it only the ground reaction?

No by construction, effectively yes for the feet in this data. The tensor is the
net external force on the body from **all** of its contacts (and any applied
external force); nothing restricts it to the ground plane. `cfrc_ext` is
additive per body, so a foot that touched its own shin would report the sum of
both.

`inspect_contact_sources.py` decomposes the raw `data.contact` list on the CPU
backend (4 worlds x 200 steps, frozen policy) and attributes every contact
force to its two bodies:

| contact kind | count | samples |
|---|---|---|
| foot <-> ground plane | 1957 | 800 / 800 have at least one |
| foot <-> another robot body | 0 | 0 |
| calf shell <-> ground plane | 261 | 98 / 800 (12.2 %) |
| robot body <-> robot body | 0 | 0 |

Summing `frame.T @ mj_contactForce(...)[0:3]` over a body's contacts reproduces
`contact_forces` with a **max abs error of 0.0 N** (float32) across all 800
samples, including the calf-contact samples. (MuJoCo puts the contact force in
the first three entries and the torque in the last three; `cfrc_ext[3:6]` is the
force part.)

Consequences:

- For this policy the **foot force is a pure ground reaction force** (normal +
  friction), so a foot-only contact reward is well defined.
- The feet are not the only things that touch the ground: the calf shells do,
  in `12.2 %` of control steps. Those samples are what the "no non-foot
  contact" filter removes from §2, and they are why the feet-only vertical sum
  falls short of bodyweight when the robot lands on its shins.
- `go2trot` sets `asset.penalize_contacts_on = ["calf"]` but has **no
  `collision` reward weight**, so policy `r` is not penalised for those contacts.
  If `r'` should also enforce "only the feet touch down", the same tensor covers
  it (`contact_forces[:, non_foot_bodies, :]`), and the whole body sum per step
  is a natural "no shell contact" signal.
- Pushes are base-velocity resets (`_push_robots`), not applied forces, and this
  task never uses `xfrc_applied`, so contacts are the only external source here.
  VSim's push box is a separate, unvalidated path.

### Validation (this study)

`check_force_semantics.py` (GPU, frozen checkpoint, 8 worlds, 800 samples per
probe) sums the vertical component over **all** bodies:

| probe | feet only | all bodies | comment |
|---|---|---|---|
| policy, zero command (6 s settle, 2 s measured) | 0.597 bw | **1.000 bw** | robot settles onto shins; the non-foot load is real |
| open-loop default pose, 2.4 s | 0.157 bw | **1.006 bw** | robot is on the ground on many links |

So `contact_forces` is the per-body net contact force and the four-foot sum is a
subset that equals bodyweight only while the feet carry the robot. In the trot
data (96.8 % of samples with no non-foot contact) the four-foot sum averages
`0.983 bw`, which is the consistency check for the extraction used below.

Force direction check on the trot data: `|Fz| / |F|` is `0.945` on average
(`p05 0.809`), so the signal is dominated by the normal component but friction
is not negligible during push-off.

### Recording path for an offline dataset

The recorder now supports the tensor directly:

```bash
cd BigRedGym
uv run --frozen scripts/train.py --task=go2trot --device=cuda:0 --headless \
  --num_envs=4096 --max_iterations=50 --resume \
  --load_run=Nov10_10-36-22_gpu-final-600-trot-hist-20261110 --checkpoint=600 \
  --run_name=contact-<topic>-20261113 \
  --replay_record --replay_fields=contacts \
  --replay_capacity=500000 --replay_snapshot_interval=10
```

Verified end to end with a short smoke run
(`logs/go2trot/Nov13_19-20-32_foot-contact-replay-smoke-20261113`). The snapshot
is `brg-offline-replay-v2`, with

- `extra_fields = ["contact_forces", "next_contact_forces"]`,
- each stored as `[capacity, 126]` (`42 bodies x 3`), with
  `field_metadata.body_names` giving the body order,
- `+1,008 B/transition` on top of the ~470 B core record (~1.4 kB/row total),
- and a snapshot stores the valid ring contents (`valid_count` rows), so
  capacity is only an upper bound: 500k transitions ≈ 0.7 GB.

Reading it back:

```python
from learning.offline.dataset import load_snapshot

snap = load_snapshot(".../replay/replay_iter_0000618")
contact = snap.data["contact_forces"].view(-1, 42, 3)  # [N, bodies, 3] N
```

`--replay_fields=all` also works but adds the joint/root/gait/critic groups; for a
contact-only reward `contacts` is the minimum useful set. If a later reward needs
to restore states (e.g. re-query the simulator), add `root` and `joints`.

### Why the existing 2M dataset cannot be relabelled

`logs/go2trot/hist-mix-20261110/replay/replay_iter_0000600` is
`brg-offline-replay-v1` with `valid_count = 2,000,000` and only the core fields
(`actor_obs`, `action`, `next_actor_obs`, `successor_height_m`, `terminated`,
`timed_out`, `episode_id`, `step_id`, `env_id`, `behavior_iteration`). It has no
contact forces **and no simulator state**, so the force cannot be recomputed
offline; `load_snapshot()` still accepts v1, it just has nothing to relabel from.
A contact reward needs a new recording of the same frozen policy.

## 2. Contact-force distribution over the cycle

Provenance: `model_600.pt` of `Nov10_10-36-22_gpu-final-600-trot-hist-20261110`
(sha256 `f88b1f9b...`), MuJoCo Warp, `cuda:0`, 128 worlds, 600 control steps
(12 s), 50 Hz, one physics step per control step, seed 7, commands resampled
every 3 s from the training distribution, pushes disabled, no episode resets,
18-40 control steps per gait cycle. 76,800 post-step samples, 96.8 % free of
non-foot contact. Contact means `load > 5 % bodyweight` (~8.4 N) unless stated.

### Per-foot summary

| foot | mean Fz (N) | p99 Fz (N) | max (N) | duty | stance mean (bw) | stance p90 (bw) | swing p95 (bw) | P(contact\|stance) | P(contact\|swing) |
|---|---|---|---|---|---|---|---|---|---|
| FL_foot | 50.0 | 149.2 | 255.6 | 0.644 | 0.484 | 0.759 | 0.488 | 0.847 | 0.439 |
| FR_foot | 47.5 | 156.6 | 268.4 | 0.594 | 0.483 | 0.752 | 0.423 | 0.839 | 0.352 |
| RL_foot | 35.3 | 126.6 | 192.8 | 0.448 | 0.401 | 0.642 | 0.161 | 0.817 | 0.083 |
| RR_foot | 32.2 | 125.1 | 192.2 | 0.414 | 0.363 | 0.632 | 0.131 | 0.748 | 0.075 |

Full tables: `artifacts/summary.md`, machine-readable `artifacts/summary.json`.

- **Stance load** rises from ~0 at the scheduled-contact edges to `0.62 bw` (FL)
  / `0.52 bw` (rear) at mid-stance, then falls (`figures/fig1`). Stance `p90` is
  `0.76 bw`, `p99` is `0.93 bw`; force peaks reach `1.5-1.6 bw` in rare landing
  spikes (max 268 N, about `1.6 bw`).
- **Duty** is above 0.5 for both front feet and below 0.5 for both rear feet. The
  contact-probability curve (`figures/fig2`) is a clean square wave for the rear
  feet and has a raised swing baseline for the front feet.
- **Front/rear asymmetry is real**: the front pair carries `0.581 bw`, the rear
  pair `0.403 bw`, i.e. the front pair carries ~44 % more load than the rear.
- **Diagonals are balanced**: `FL+RR = 0.490 bw`, `FR+RL = 0.493 bw`. The
  instantaneous diagonal difference is large (`0.79 bw`) by design — a trot
  alternates the diagonal that supports the body.
- **Vertical total** (`figures/fig3`, `figures/fig4`): mean `0.983 bw`
  (sd `0.513`), `p05 0.00`, `p95 1.67`; per sample it drops to `0` in the aerial
  phase and peaks around `1.25 bw` at mid-stance (the phase-averaged profile
  never goes below `~0.5 bw` because the aerial phase is short and not
  phase-locked). `10.4 %` of samples are airborne (all four feet below the
  contact threshold).
- **Speed dependence** (`figures/fig5`): higher commanded speed shortens duty
  (`0.551 -> 0.476`), raises the airborne fraction (`6.9 % -> 19.5 %`), and
  raises the per-foot `p95` load (`0.67 -> 0.78 bw`); the cycle-mean total stays
  `0.94-1.00 bw`.
- **Gait frequency** (1-3 Hz) barely changes the cycle-mean total
  (`0.979-0.989 bw`) but duty falls from `0.564` to `0.490` as frequency rises.

The recomputed `Go2Trot` terms on this data are `trot_support` mean `0.509`
(weighted `1.53/s`, matching the logged `1.49-1.53/s` of the training run) and
`swing_contact` mean `-0.012` (weighted `-0.036/s`). That is a useful identity
check: the recorded forces reproduce the task's own contact reward.

### 2.1 Peak force per touchdown and per cycle

Method (`analyze_cycle_peaks.py`): contact runs are contiguous samples with
`load > 5 % bw`; a **loaded stance** is a run of at least 5 control steps
(0.1 s) peaking above `0.30 bw`, which separates real touchdowns from the light
front-foot touches. Cycles are the phase wraps between the recording boundaries;
only complete interior cycles are counted (2,968 cycles, 18,365 contact runs,
0.44 s median period, 85 % of cycles contain an airborne moment).

| foot | loaded / all runs | peak p50 | p95 | p99 | max | >0.8 bw | >1.0 bw | >1.2 bw | stance p50 | impulse p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| FL_foot | 3852 / 5264 | 0.720 | 0.966 | 1.119 | 1.458 | 26.1 % | 3.7 % | 0.5 % | 0.20 s | 18.9 Ns |
| FR_foot | 3578 / 5687 | 0.707 | 1.058 | 1.223 | 1.679 | 30.7 % | 7.9 % | 1.2 % | 0.20 s | 18.0 Ns |
| RL_foot | 3265 / 3687 | 0.628 | 0.837 | 0.975 | 1.212 | 8.4 % | 0.7 % | 0.03 % | 0.18 s | 15.2 Ns |
| RR_foot | 3147 / 3727 | 0.623 | 0.828 | 0.942 | 1.119 | 7.7 % | 0.3 % | 0.0 % | 0.16 s | 14.0 Ns |

`figures/fig8` (histogram and p95/max per foot), `figures/fig9` (peak vs speed
and frequency). Cycle-level maximum (`max` over a complete cycle):

| foot | p50 | p95 | within-env sd across cycles |
|---|---|---|---|
| FL | 0.734 | 0.987 | 0.103 |
| FR | 0.721 | 1.076 | 0.118 |
| RL | 0.630 | 0.831 | 0.084 |
| RR | 0.627 | 0.830 | 0.085 |

- The **median** touchdown is unremarkable (`~0.7 bw`, about twice the mean
  stance load because it is a peak). The reward-relevant part is the tail: the
  front feet exceed bodyweight in 4-8 % of touchdowns and can reach `1.5-1.7 bw`.
- **Peak grows with speed**: `p95 0.856 -> 1.066 bw` from `vx < 0.1` to
  `vx > 1.2 m/s`, while stance duration shrinks `0.231 -> 0.162 s`
  (`figures/fig9`, left). Gait frequency (1-3 Hz) changes the stance duration
  (`0.274 -> 0.168 s`) but not the peak monotonically (`p95 0.906-0.988 bw`).
- **Peak timing**: the peak occurs at leg phase `1.97` (FL), `1.39` (FR), `1.35`
  (RL), `1.19` (RR) rad, i.e. from mid-stance onward; 64 % of FL peaks are in
  the second half of the stance window, versus 35-41 % for the other feet.
- **Impulse identity**: per-cycle impulse divided by
  `bodyweight * period / 4` averages `0.999` (`p5 0.61`, `p95 1.44`), confirming
  the four feet carry the bodyweight on average and showing that the
  cycle-to-cycle sharing is far from equal — a per-cycle signal a per-step
  reward cannot encode.
- Sampling caveat: the simulation runs at 50 Hz with one physics step per
  control step, so a "peak" is a 20 ms sample. Sharper impact spikes are not
  observable without raising the simulated rate; comparisons of peak force
  should stay within this rate.

### 2.2 Where the force during scheduled swing comes from

`inspect_swing_contact.py` replays the policy on the CPU backend and attributes
every individual contact to its two bodies, with the leg phase of the foot
involved (`sin(leg_phase) > 0` = the task's scheduled stance):

| contact class | contacts | share | force p50 | p95 | max |
|---|---|---|---|---|---|
| scheduled stance | 1425 | 72.8 % | 90.6 N | 121.6 N | 256.1 N |
| swing, late lift-off (`pi .. 1.5pi`) | 266 | 13.6 % | 24.4 N | 63.8 N | 159.6 N |
| swing, early touchdown (`1.5pi .. 2pi`) | 266 | 13.6 % | 29.9 N | 101.7 N | 146.3 N |

All 1,957 foot contacts are with **`world`** (the ground plane); none is with
another robot body. The 532 swing-phase contacts sit at `z = -3.6 mm` (p05
`-16.4`, p95 `-0.4`) with a median penetration of `7.2 mm` (p95 `32.8 mm`), so
they are genuine ground contacts of a loaded foot, not a numerical leftover.

So the source is **timing, not physics bookkeeping**:

- The real stance is longer than the clock's 50 % window, and the excess splits
  almost exactly in half between not lifting yet (late lift-off) and landing
  early (early touchdown). Early touchdown is heavier (`p95 102 N` vs `64 N`) —
  it carries the landing impulse; late lift-off is the decaying unload tail
  (`figures/fig10`).
- Nothing else contributes: no self-collision, no applied external force, and
  joint-reaction or inertial loads are **not** part of `contact_forces` at all.
- The non-foot contacts are the calf shells on the ground (261 contacts, e.g.
  `FR_calflower1` 146, max `45.6 N`), also split across that leg's own stance
  and stance edges. `go2trot` has no `collision` reward weight, so these are
  unpenalised.

How much of the foot impulse falls outside the clock's stance window
(`analyze_foot_contacts.py`, `schedule_window_sweep`):

| stance-window duty | impulse inside | foot impulse outside | in-window contact precision |
|---|---|---|---|
| 0.50 (the task's `sin > 0`) | 88.0 % | **11.9 %** | 0.813 |
| 0.55 | 90.1 % | 9.8 % | 0.769 |
| 0.60 | 91.6 % | 8.3 % | 0.725 |
| 0.70 | 93.7 % | 6.3 % | 0.651 |

The clock's hard 50 % window is therefore mostly right — the off-schedule
contacts are numerous but light, and widening the window gains little impulse
coverage while losing precision. This argues for keeping the 50 % schedule but
making the schedule term **soft** (`smoothstep`/tolerance) rather than a hard
indicator, so the natural landing/unload edge is not treated as a violation.

## 3. What this means for the offline contact reward `r'`

### Do not constrain the instantaneous vertical total

`sum_i Fz_i = m g` holds only in a cycle mean. Inside a cycle the total drops to
zero for `10 %` of control steps (aerial phase) and reaches `1.25 bw` at
mid-stance. A per-step constraint on the total would label the policy's own
natural trot as a violation and would dominate the label.

### Do constrain the scheduled per-foot load

Use the phase clock that is already in the actor observation
(`phase_obs = [sin(phase), cos(phase)]`, `phase_frequency`) and the task's own
stance convention `s_i = 1[sin(phase + 2*pi*offset_i) > 0]`, with offsets
`FL 0.0, FR 0.5, RL 0.5, RR 0.0` cycles. The measured clock is already aligned
(best offset `0.00 rad`), so no correction term is needed.

Empirical rates that justify the thresholds:

| level (bw) | P(load >= level \| stance) | P(load >= level \| swing) |
|---|---|---|
| 0.05 | 0.813 | 0.237 |
| 0.10 | 0.797 | 0.192 |
| 0.20 | 0.764 | 0.111 |
| 0.30 | 0.726 | 0.069 |
| 0.50 | 0.520 | 0.023 |

Per foot, `P(stance load >= 0.2 bw)` is `FL 0.80, FR 0.81, RL 0.76, RR 0.69`,
and `P(swing load >= 0.1 bw)` is `FL 0.35, FR 0.29, RL 0.07, RR 0.06`.

A defensible, bounded candidate (used in `figures/fig7`):

```
l_i    = Fz_i / (m g)                          # per-foot load, bodyweight units
s_i    = 1[sin(leg_phase_i) > 0]               # scheduled stance
D      = mean_i [ s_i * relu(0.20 - l_i) ]     # stance-load deficit
S      = mean_i [ (1 - s_i) * relu(l_i - 0.10) ]  # swing overload
r'     = 1 - clip(D / 0.25 + S / 0.10, 0, 1)   # bounded [0, 1], 1 = constraint met
```

Measured on policy `r`:

| variant | mean | std | p05 | p50 | p95 | held-out `R^2` from obs+action |
|---|---|---|---|---|---|---|
| `linear_penalty` (above) | 0.779 | 0.295 | 0.000 | 0.917 | 1.000 | 0.473 |
| `exp_penalty` (`exp(-(D/0.25)^2 - (S/0.10)^2)`) | 0.891 | 0.214 | 0.320 | 0.993 | 1.000 | 0.416 |
| `schedule_smoothstep` (`0.5*stance_score + 0.5*(1-swing_score)`) | 0.638 | 0.108 | 0.472 | 0.625 | 0.750 | 0.490 |

Design notes:

- **Bounded, zero after terminal** is the right convention for a reward in
  `[0, 1]` (use `accumulate_return_zero_after_terminal`, `failure_value = 0.0`),
  not the height path's absorbing-failure convention.
- Prefer the linear or exponential form over the smoothstep form: the
  smoothstep version saturates per foot and quantises `r'` into 3-4 values
  (`figures/fig7`, right panel), which gives Q a very low-variance target.
- The thresholds are not sharp knobs: moving the stance target over
  `0.15-0.35 bw` and the swing tolerance over `0.05-0.20 bw` changes the mean
  `r'` by at most `~0.03`, so the label is not sensitive to a lucky threshold.
  Choose them from the rates above, then validate through the usual FQE artifact
  gates rather than by threshold tuning on the same returns.
- Keep the schedule term **soft**. The hard `sin > 0` indicator puts 11.9 % of
  the foot force impulse on the "wrong" side, split evenly between late
  lift-off and early touchdown (§2.2). A tolerance band or smoothstep around
  the stance edge avoids labelling a correct landing as a violation.
- A **peak-force** term is a different object from a per-cycle maximum. `r'` is
  evaluated after each action, so "cycle max" is not available to it (and
  would leak future information into a per-step label). The usable form is an
  instantaneous saturation, e.g. `relu(l - cap)` with `cap` at `1.0 bw`
  (violated by 3.7 % of FL and 7.9 % of FR touchdowns) or `0.8 bw` (26-31 % of
  front, 8 % of rear). Cycle maxima in §2.1 are useful for evaluation and for
  reporting how often the constraint is broken, not as the training label.
- The main signal is the **front-foot grazing** during scheduled swing
  (`p95 0.42-0.49 bw`): the rear feet are already clean (`p95 0.13-0.16 bw`).
  If that is the intended target, consider a per-foot weight that is not
  uniform, or report per-foot violations separately so the effect is visible.

### 3.1 Threshold label `r' = -1 / 0` (measured)

Simplest form of the constraint: a per-leg threshold on the instantaneous load,

```
r'_t = -1 if max_i [ l_i,t > T_i ] else 0        (T_i per leg, from policy r)
```

`analyze_threshold_label.py` measures what that label does for thresholds taken
from the per-touchdown peak distribution (`figures/fig11`):

| threshold family | FL | FR | RL | RR | steps flagged | touchdowns flagged | return mean | in first quarter of stance |
|---|---|---|---|---|---|---|---|---|
| p90 of peaks | 0.90 | 0.97 | 0.79 | 0.78 | 2.82 % | 10.0 % | -2.76 | 54 % |
| p95 of peaks | 0.97 | 1.06 | 0.84 | 0.83 | 1.24 % | 5.0 % | -1.24 | 61 % |
| p99 of peaks | 1.12 | 1.22 | 0.98 | 0.94 | 0.20 % | 1.0 % | -0.20 | 78 % |
| 0.9 x p99 | 1.01 | 1.10 | 0.88 | 0.85 | 0.77 % | 3.3 % | -0.78 | 65 % |
| 0.9 x max | 1.31 | 1.51 | 1.09 | 1.01 | 0.03 % | 0.2 % | -0.03 | 89 % |
| 0.9 x median | 0.65 | 0.64 | 0.56 | 0.56 | 44.5 % | 73.7 % | -41.58 | 30 % |
| p90 of cycle max | 0.91 | 0.99 | 0.78 | 0.78 | 2.58 % | 9.4 % | -2.52 | 54 % |

(thresholds in bodyweight; returns are the discounted sum of `r'` over one
250-step episode at `gamma = 0.99`; "in first quarter" is the share of flagged
steps that fall in the first quarter of their stance)

Properties at the requested p90 setting:

- **Frequent but not constant**: `2.8 %` of control steps, about `7` flagged
  steps per 250-step episode, and every episode contains at least one — the Q is
  ranking landings, not predicting a rare failure.
- **Value scale is comfortable**: episode return `-2.76` on average (p05
  `-11.4`, worst `-20.9`), far from the tens of units a dense negative reward
  would produce.
- **The label is a landing-impact label**: 54 % of flagged steps are in the
  first quarter of a stance and 0.9 % in the last, and the fraction rises with
  the threshold (78 % at p99, 89 % at `0.9 x max`). Hard touchdowns are the
  event, not push-off.
- **Observable**: predicting "a foot exceeds its p90 next step" from the actor
  observation plus action gives a held-out AUC of `0.93` (random-feature probe,
  base rate `2.0 %`; linear `0.86`). The state carries the signal, so `Q_r'` is
  not being asked for clairvoyance.
- **Threshold coupling**: the front thresholds are ~15 % higher than the rear
  ones (front feet carry more), so the same physical force is judged differently
  per leg. That is what "different for each leg" means here and it is
  defensible, but the label cannot be described as an absolute force limit.
- **`0.9 x median` is not usable**: it flags `44 %` of steps and `74 %` of
  touchdowns. If "90 % of the value" meant `0.9 x p99`, use that row instead
  (3.3 % of touchdowns, `0.77 %` of steps).

Implementation consequences (not yet done, see §4):

1. The label needs `contact_forces` **and a per-environment bodyweight**. The
   replay record does not carry `link_mass`, and domain randomisation varies the
   mass by `+-10 %`. Either add a recorded bodyweight feature, or define the
   thresholds in Newtons (mass-blind but schema-unchanged). This is the one
   blocking detail for a self-contained artifact.
2. The thresholds are part of the reward definition: they belong in the artifact
   manifest, and a Q trained with different thresholds is a different artifact.
3. `r'` is bounded in `{-1, 0}` with `failure_value = 0` after termination.
4. The label is imbalanced (2-3 %), so a single mean Bellman residual hides the
   class that matters; report residuals and value error separately for flagged
   and unflagged successors.
5. The intervention has to act **before** touchdown. At p90, 54 % of flagged
   steps are already in the first quarter of stance, so by the time the label
   fires the landing is over. The Q's useful question is the value of the
   pre-landing state (swing, a few steps before `sin(leg_phase)` crosses zero),
   which the phase observation makes addressable.

### 3.2 The recorded contact-labelled dataset

Recorded and verified (`verify_contact_snapshot.py`):

| item | value |
|---|---|
| run | `logs/go2trot/Nov13_21-02-10_contact-bw-1m-20261113` |
| policy | resumed `model_600` (`f88b1f9b...`), snapshots after iterations 604 / 608 / 612 / 616 |
| transitions | **1,048,576** (4096 envs x 256 steps), 3.7 GB on disk for all four snapshots |
| fields | `contact_forces`, `next_contact_forces` (`[N, 126]` = 42 bodies x 3), `bodyweight` (`[N, 1]`) |
| bodyweight | 152.3 - 184.2 N (mean 167.6, sd 6.4), constant within an environment |
| per-leg p90 thresholds | FL `0.952`, FR `0.990`, RL `0.863`, RR `0.837` bw (= 160 / 166 / 145 / 140 N at mean bodyweight) |
| label rate | `2.83 %` of control steps, `10.00 %` of touchdowns (per leg 0.84 / 0.84 / 0.61 / 0.61 %) |
| terminated rows | `0.10 %`, receive reward 0 and are excluded from the peak estimate |

The label must be computed from `next_contact_forces` (post-action, pre-reset),
matching `successor_height_m`'s timing, with the phase taken from
`next_actor_obs`'s `phase_obs` block.

**Why these thresholds are 5-8 % higher than §3.1's.** The study replayed the
policy deterministically (`act_inference`), while a recorded rollout uses the
*stochastic* training policy (`actor.act`, action std ~0.246). A matched
stochastic replay of 128 worlds reproduces the dataset:

| variant | FL | FR | RL | RR |
|---|---|---|---|---|
| deterministic playback (study, §3.1) | 0.900 | 0.968 | 0.787 | 0.778 |
| deterministic + pushes | 0.898 | 0.955 | 0.794 | 0.783 |
| deterministic, model_616 (drift) | 0.911 | 0.959 | 0.806 | 0.770 |
| deterministic, `step_id >= 50` (no reset transients) | 0.942 | 0.984 | 0.833 | 0.820 |
| **stochastic actions** | **0.951** | **0.986** | **0.845** | **0.823** |
| recorded dataset | 0.952 | 0.990 | 0.863 | 0.837 |

So the thresholds are a property of the **behaviour distribution**, not of the
robot: playback evaluates a cleaner policy than the one that generated the
training data. A threshold frozen in a Q artifact therefore belongs to that
artifact's dataset and behaviour policy; it cannot be transplanted from a
playback study.

### Observability from the actor observation

Nonlinear (random-feature ridge) probe, 80/20 split **by environment** so the
test worlds are unseen; the observation at step `t` and the action applied at
`t` predict the post-action force at `t+1`:

| feature set | foot load | contact flag | total load | `r'` |
|---|---|---|---|---|
| actor obs (48) | 0.873 | 0.751 | 0.800 | 0.504 |
| actor obs + action (60) | 0.873 | 0.754 | 0.800 | 0.490 |
| full simulator state (privileged) | 0.774 | 0.655 | 0.693 | 0.361 |

Linear probes match the nonlinear ones within 0.03 for load, contact and total
load, but only reach `0.38` for `r'` because the clip/relu form is genuinely
nonlinear. Two conclusions:

1. The contact reward is **learnable from the actor observation** — most of the
   force is a function of the joint state and the phase clock, and the phase is
   in the observation. The privileged set scores no better, so the observation
   is not losing the signal that the joint/root state carries.
2. `R^2 ~ 0.5` for `r'` is the realistic ceiling for a one-step probe. The
   residual is concentrated at the contact transitions
   (`R^2 = 0.53` for leg phase `0-0.79 rad` and `0.50` for `3.14-3.93 rad`,
   versus `0.82` for `1.57-2.36 rad`), i.e. where the foot is between "landing"
   and "loaded".
   Expect `Q_r'` to be least accurate there, and do not read a small one-step
   Bellman residual at mid-stance as evidence of a good reward model.

## 3.3 The `r'` + `Q_r'` path (implemented)

The threshold label from §3.1 and §3.2 now has a product implementation, with
the durable contract in
[`docs/offline_contact_q.md`](../../../offline_contact_q.md):

| piece | location |
|---|---|
| reward spec (`foot_names`, `thresholds_bw`, `percentile`, timing, zero-after-terminal) | `learning/offline/contracts.py` |
| labeller, threshold derivation, load extraction | `learning/offline/rewards.py` |
| FQE core shared with the height and yaw paths | `learning/offline/training.py` |
| `brg-contact-q-v1` artifact (`ContactQ`, `save_contact_q`) | `learning/offline/q_artifact.py` |
| CLI (`--percentile`, `--thresholds`, episode-grouped validation) | `scripts/train_contact_q.py` |
| recording field needed for the normaliser (`bodyweight`) | `learning/offline/replay_fields.py`, `LeggedRobot.bodyweight` |
| tests | `tests/unit_tests/test_contact_q.py` |

Derived on the recorded dataset and reported by the CLI:

```
thresholds_bw=FL_foot:0.9526 FR_foot:0.9891 RL_foot:0.8631 RR_foot:0.8358 percentile=90.0
label_rate=0.02826 transitions=836020
```

which reproduces §3.2 (0.952 / 0.990 / 0.863 / 0.837 and 2.83 % of steps)
from the training pool alone; the held-out split never influences the reward
definition.

First fit on the recorded dataset (1,048,576 transitions, 80/20 episode split,
60,000 updates, `gamma = 0.99`, lr 1e-3, 256x256): held-out explained variance
`0.978`, and by label class, flagged rows (2.9 %) have MAE `0.611` with the Q
under-predicting the cost of a violation by about `0.5` while unflagged rows
have MAE `0.103`. Code, commands and the full table:
[`docs/offline_contact_q.md`](../../../offline_contact_q.md#measured-run-2026-11-14).

**Candidate ranking is at the protocol ceiling, and the persistent intervention
works.** 128 insertion states, eight candidate first actions, H = 500:
within-state realized spread `2.65` (the first action does move the contact
return), but the same candidate set re-run under another seed ranks at `0.084`,
and Q ranks at `0.017` (H500) / `0.055` (H100) — no ranker can beat replay
noise on a one-step intervention, and Q does not. A *persistent* intervention
(max-Q over every eight-draw block) is different: contact cost `−2.47 → −0.86`,
violation rate `2.53 % → 0.89 %`, with the original PPO reward per step moving
`−0.15 %`, and with argmin-Q (`−5.96`) and random selection (`−2.59`, ≈ plain)
as controls. Details and tables:
[`docs/offline_contact_q.md`](../../../offline_contact_q.md#persistent-intervention-measured).

Two implementation details worth keeping in mind when reading any downstream Q
result:

- The label is `-1` on the step where the force is already high, i.e. after the
  landing has begun. The Q's useful content is the value of the *pre-landing*
  state, and the phase clock in the actor observation is what makes that
  addressable.
- `derive_contact_thresholds()` groups rows by `(env_id, episode_id)` and orders
  them by `step_id`, so it does not depend on the recorded row order and does
  not merge a touchdown across an episode reset.

## 4. Recommended next step for the dataset

1. Re-record the frozen policy with `--replay_fields=contacts` (plus `root` and
   `joints` if a state-restore path is ever needed). Budget
   `~1.4 kB/transition`, so 500k transitions ≈ 0.7 GB with
   `--replay_capacity=500000`; capacity only bounds the snapshot size.
2. Keep the recording conditions identical to the target policy
   (`env cfg` fingerprint `6412a878...`, checkpoint sha256 `f88b1f9b...`) and
   record a few snapshots across iterations so early/late behaviour can be
   compared.
3. Add the reward spec (units, bounds, timing `post_action_pre_reset`, terminal
   convention `zero_after_terminal`, source fields) and relabel from
   `contact_forces` + `phase_obs`; keep the Q input to the 48 actor observation
   fields.
4. Validate `Q_r'` with episode-grouped held-out Bellman residuals and a
   candidate-ranking diagnostic, as required by the offline reward path; this
   study only establishes that the label exists, is well conditioned, and is
   observable.

## 5. Limitations

- MuJoCo Warp (study) and MuJoCo CPU (agreement run) only; VSim is not
  validated. Backend agreement is distribution-level, on different randomised
  worlds: FL duty `0.651` vs `0.644`, FL stance load `0.474` vs `0.484 bw`,
  total `0.984` vs `0.983 bw`, airborne `10.7 %` vs `10.4 %`, and the FL
  phase-curve maximum difference is `0.035 bw`.
- 128 worlds x 600 steps (12 s) at one seed; no push perturbations. The training
  data did include pushes every 5 s, so the tail of the force distribution in a
  labelled dataset will be slightly heavier than measured here.
- Zero-command standing is not a stable behaviour of this policy: after 6 s the
  robot is resting on its shins (feet carry `0.60 bw`). Do not build a
  standing/contact reward on the assumption that it can hold a static stance.
- The force is the net per-body wrench: no contact point, CoP, or tangential
  breakdown, so CoP/friction-cone rewards are out of reach without a backend
  change.
- The `r'` forms in §3 are candidates derived from measurement, not a validated
  reward. Their statistics are policy-`r` statistics; a different policy shifts
  the load distribution.

## 6. Reproduce

```bash
cd BigRedGym
uv run --frozen python docs/plans/supplements/foot_contact_study/collect_foot_contacts.py \
  --device cuda:0 --num_envs 128 --steps 600 --tag gpu
uv run --frozen python docs/plans/supplements/foot_contact_study/collect_foot_contacts.py \
  --device cpu --num_envs 64 --steps 400 --tag cpu
uv run --frozen python docs/plans/supplements/foot_contact_study/check_force_semantics.py
uv run --frozen python docs/plans/supplements/foot_contact_study/inspect_contact_sources.py
uv run --frozen python docs/plans/supplements/foot_contact_study/analyze_foot_contacts.py
uv run --frozen python docs/plans/supplements/foot_contact_study/analyze_cycle_peaks.py
uv run --frozen python docs/plans/supplements/foot_contact_study/inspect_swing_contact.py
```

`data/` holds the raw recordings (35.7 MB + 11.9 MB, git-ignored: above the
100 KiB commit limit), `artifacts/` the summaries, `figures/` the plots.
