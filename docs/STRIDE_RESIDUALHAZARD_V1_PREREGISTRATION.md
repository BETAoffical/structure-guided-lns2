# STRIDE ResidualHazard v1 Preregistration

## Purpose

The Maze first-divergence replay showed that 49/66 structural actions were
robustly better than the exact V2 action on immediate conflict reduction, and
that structural actions also won 17/20 robustly ordered adverse/severe cases.
The long tail therefore cannot be guarded by rejecting actions that are merely
worse on the current step.

`stride-residualhazard-v1` asks a different question: can information available
before an action predict whether PP will leave a concentrated, externally
coupled residual conflict state, even when total conflict reduction is good?
This registration authorizes feature and target construction only.  It does
not authorize model fitting, threshold search, runtime integration, or a TTF
claim.

## Frozen unit and evidence boundary

The design cohort is all 66 frozen first-divergence states from
`stride-maze-tail-action-replay-v1`.  Each state retains exactly the V2 anchor
and the realized structural challenger, with trial indices 0--15 and strictly
paired PP seeds.  No state, action, trial, map, or controller is removed using
its episode category or new residual measurements.

The current 66 states are target-design evidence only and may never become
training or held-out performance rows.  Any later training requires a
separately preregistered cohort of new maps and task seeds.

## One-step residual measurements

For each exact action and PP trial, the diagnostic reconstructs only the common
pre-action state and the immediate post-PP state.  It records:

1. residual conflict count divided by the common before-conflict count;
2. residual conflict-pair boundary ratio, where exactly one endpoint was
   outside the repaired neighborhood, divided by all residual pairs;
3. new residual-pair ratio relative to the pre-action conflict pairs;
4. residual events at static degree-two-or-lower cells divided by all residual
   events;
5. largest residual conflict-component agent ratio, divided by the number of
   agents participating in any residual conflict;
6. spatial residual-event concentration, computed as the Herfindahl index over
   conflict cells, with every vertex event contributing mass one to its cell
   and every edge event contributing mass one-half to each endpoint; and
7. the fraction of unselected conflict agents adjacent to selected agents in
   the residual conflict graph.

Every ratio has a fixed empty-denominator value of zero.  Static cell degree is
computed from the four-neighbor map graph.  There is no tunable corridor width,
time window, or distance cutoff.  Total residual conflict is retained as a
separate quality measurement and is not folded into the structural hazard
target.

For every measurement, the report gives challenger-minus-V2 paired deltas,
mean and median deltas, fixed seed-half means for trials 0--7 and 8--15, sign
counts, and no-residual-conflict frequency.  A measurement is called
seed-stable only when at least 12/16 paired deltas share a direction and both
fixed halves share that direction.  No threshold is tuned from episode tails.

## Target-readiness decision

The frozen episode category is used only after the one-step measurements have
been emitted and hashed.  Adverse and severe comparisons form the tail group;
all remaining registered categories stay in the control group, including
inconclusive cases.

A residual target may be frozen for a future independent dataset only if:

- its tail-versus-control direction is the same on all three current maps;
- the direction is the same for StructPool and SlotPool;
- at least eight adverse/severe comparisons have seed-stable action ordering
  on that measurement; and
- at least four of those seed-stable tail comparisons still show greater
  structural hazard when the challenger has no larger mean normalized residual
  conflict count; and
- the association is therefore not explained solely by total residual conflict
  count.

If no individual measurement passes, no post-hoc weighted combination or
threshold search is permitted.  If several measurements pass, the first in the
frozen order `new pair`, `boundary pair`, `low-degree event`, `cell
concentration`, `largest component`, `outside queue` is the only target that
may advance.  ResidualHazard otherwise stops and the exact V2 fallback remains
unchanged.

## Future predictor inputs

If target readiness passes, a later predictor may use only deterministic
pre-action quantities:

- the existing 124-dimensional challenger-minus-anchor feature difference;
- map size, free-cell ratio, and static degree histogram, never map identity;
- conflict-event, conflict-pair, and conflict-component summaries;
- candidate size, support count and ratio, structural-family indicators, and
  Jaccard overlap with the V2 anchor;
- selected/internal, selected-to-unselected, and unselected conflict coverage;
- pre-action low-degree-cell, bottleneck-support, and path-overlap summaries;
- overlap with the immediately preceding realized neighborhood; and
- prefix-only no-progress streak and decision index.

Forbidden inputs include PP seed, agent IDs, map ID, repair outcomes, any
post-action field, episode category, future conflicts, remaining repair rounds,
TTF, wall time, PP time, censoring outcome, and controller success.

## Independent-data and validation requirement

Training is not authorized until a new registration provides at least eight
previously unused Maze maps, two outcome-blind tasks per map, three solver
seeds, and both structural challengers.  Tasks must be fixed from reset-only
information, and all resulting states must be retained.  Whole maps are the
grouping unit for nested cross-validation; states from one map may not cross a
fold boundary.

The runtime action is always abstention to the exact V2 anchor unless a frozen
hazard model classifies the challenger as non-hazardous.  Thresholds must be
selected inside training-map folds and must preserve V2 success.  A separate
set of at least four additional unseen maps is required before any runtime or
TTF experiment.

## Claim boundary

Passing target readiness would show only that a reproducible one-step residual
structure target exists.  It would not show predictability from pre-action
features, lower long-tail frequency, faster TTF, or better generalization.
Those claims require separate preregistered model and paired run-to-completion
experiments.

Frozen parent report SHA-256:
`54223f17155d6222f467823e11b3a31fe9faec1c51989e3a40c26d1a03f77587`.
