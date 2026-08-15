# STRIDE exact-state controller attribution v1

## Question

Can official LNS2 neighborhood control escape the same 45 frozen
`first_repeat_stall` states more reliably than the frozen STRIDE controller?

This is an attribution experiment, not another long-tail repair proposal.  It
directly separates three explanations at the same complete path state:

1. the frozen proposal/ranker chooses poorly;
2. the original Target failure-based neighborhood mechanism is sufficient but
   underused;
3. all PP controllers struggle because the restored state is already in a hard
   repair basin.

## Frozen arms

- `frozen_controller`: the case-associated v2-plus-StructPool or
  v2-plus-SlotPool controller;
- `official_adaptive_n8`: native Adaptive LNS2, neighborhood size 8, PP;
- `fixed_target_n8`: native Target failure-based generation, neighborhood size
  8, PP.

Every arm starts from the same restored paths and repair fingerprint.  There is
no shared forced action, no retry, no explicit repair order, and no PBS/GCBS.
The trial salt varies the native random streams across trials while preserving
the first PP seed within each state/trial contrast.

## Execution

- all 45 registered Maze first-repeat-stall states;
- trial 0-3 initially, 540 episodes total;
- 16 non-timing workers;
- success or 64 repair decisions / 180 seconds;
- 240-second episode process limit and 300-second outer limit;
- atomic episode checkpoints and stop on the first execution error or process
  timeout.

The one-state implementation smoke established that ordinary platform-entry is
not an informative outcome here: every episode starts after two exact rollback
observations.  Before any formal episode, the mechanism outcome was therefore
corrected to whether the initial repair signature is still unresolved after
three further decisions.  The report also records the first escape decision and
whether an episode enters a new three-rollback platform after escaping.

The whole cohort is extended to trial 4-7 only if at least one official arm has
a lower three-decision unresolved rate than the frozen controller, no lower
success rate, and no map-level unresolved-rate worsening above five percentage
points.

## Outcomes and interpretation

The primary outcomes are initial-platform unresolved-at-three, success,
normalized fixed conflict AUC, restricted mean repair decisions, repair wall
time, first platform escape, post-escape platform re-entry, first strict
conflict progress, and neighborhood diversity.  Final intervals use
state-clustered paired bootstrap resampling.

- Adaptive better but Target not better: method exploration/adaptation matters.
- Target better: the frozen controller underuses failure-based neighborhoods.
- Both official arms better: proposal/selection is the dominant gap.
- Neither better: the restored basin and PP are dominant; the next valid step
  is a separately qualified same-set PP-versus-stronger-repairer experiment.

The cohort is Maze-only and outcome-enriched.  No result permits runtime
integration, candidate-pool replacement, training, TTF claims, or non-Maze/OOD
claims.
