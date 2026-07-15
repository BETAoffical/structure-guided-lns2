# InitLNS policy-visited aggregation and RL warm-start gate

## Question

The frozen `realized_dynamic` ranker reduced fixed-budget conflict AUC on twelve unseen maps and three
independent solver streams, but its original supervised data contains only 23 states reached by official
Adaptive. This stage asks whether labels collected at states reached by the frozen ranker improve or at
least preserve its candidate ranking and closed-loop behavior. It does not train RL or restore the static
map/OD/density claim.

## Registered data

- Dataset master seed: `20270317`.
- Train: twelve maps, four per registered layout family.
- Validation: six maps, two per registered layout family.
- Every map has balanced/bottleneck tasks at 80 and 100 agents.
- Solver seeds: `1`, `2`, `3`.
- Frozen-ranker source episodes: 216; Validation also runs 72 official Adaptive baselines.

All 216 resets must be valid. Train and Validation require at least 108 and 54 repairable task-seeds,
respectively; each layout must be at least 75% repairable and at least 11/12 Train maps and 5/6 Validation
maps must contribute repair states. Failure is reported without replacing a map or task seed.

Each frozen-ranker episode contributes at most three unique decision states: first, middle and last. The
collector replays every explicit-action prefix, requires an exact state fingerprint, regenerates the
online candidate pool and requires the candidate IDs and agent sets to match the source trace. Each of at
most 18 explicit neighborhoods receives four independent one-step PP-order trials. The maximum is 46,656
outcomes. Trial jobs are isolated, atomic, resumable and limited to 120 seconds.

## Learning and gates

Only `proposal_dynamic` and `realized_dynamic` are trained. Horizon-1 feasibility rate and mean remaining
conflicts define effectiveness Pareto labels; generated nodes and runtime remain sensitivity diagnostics.
The old 23 development states and new Train states are aggregated with equal state weight. Validation is
never used for fitting or hyperparameter selection.

The fixed pairwise HistGradientBoosting configuration is reused. Offline Validation accepts v2 only when
top-1 improves by at least three points or conflict regret falls by at least 5%, the other metric remains
within its registered degradation bound, map bootstrap does not show significant degradation, and the
model does not collapse to one unsupported size. Closed-loop Validation compares Adaptive, frozen v1 and
aggregated v2. V2 must preserve success, remain within 5% of v1 AUC, improve over Adaptive by at least 5%,
and be no worse than v1 on at least four of six maps.

If v2 passes, it becomes the RL warm start. If v2 does not improve but v1 remains robust, v1 remains the
warm start and the new data becomes replay data. If both lose the Adaptive advantage, RL remains paused.
The later RL action is selection among the generated explicit neighborhoods, not raw agent-subset
generation or a return to seed/rule/size-only control.

WSL performs collection with the dependency-free portable model. Supervised fitting uses the existing
Windows scikit-learn 1.5.0 installation; this stage installs no package.

Pre-data amendment: implementation review found that the historical pairwise learner fixes
`MODEL_SEED=20260714`; the initial registered analysis JSON contained `20260829`. Before generating a map,
reset or label, the configuration was corrected to `20260714` so aggregation reuses the original learner
randomness rather than silently introducing a new training setting.

Pre-data generation amendment: the first deterministic generation attempt stopped before producing a
dataset manifest, reset or label. On `policy_train_compartmentalized_double_horizontal_0003`, the
`bottleneck_100` task exhausted valid unique constrained endpoints at agent 24 when both bottleneck ratios
were `0.25`. A same-map, same-task-seed feasibility probe reproduced the failure at `0.25` and generated
all 100 agents at `0.20`. Both bottleneck variants therefore use `0.20`; the master, map and task seeds are
unchanged, and no failed instance is replaced or resampled.
