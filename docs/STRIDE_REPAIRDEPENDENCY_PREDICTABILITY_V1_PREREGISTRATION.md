# STRIDE RepairDependency Predictability v1 Preregistration

## Question

The completed Repairability causal audit attributes 21/45 frozen repeat-stall
states to an agent-set defect, 13/45 to repair order, 2/45 to their joint
effect, and 9/45 to residual PP instability. Appending blockers observed
inside PP is causal evidence, but those blockers are unavailable before the
action and cannot be copied into a controller.

This audit asks whether the blocker set can be approximated from the frozen
pre-action state alone. It performs no PP calls, model fitting, controller
change, or TTF measurement. Observed blockers are evaluation targets only and
never features or training labels.

## Frozen cohort and labels

- all 45 Repairability states and trial indices 0 through 15 are retained;
- every causal state file is frozen by a complete SHA-256 manifest;
- the target for one state/trial is the exact set added by the registered
  `blocker_augmented_tail` arm, limited by the earlier audit to eight agents;
- the ten registered not-applicable trial arms remain missing-target rows and
  are excluded only from recall means;
- no state, map, seed half, or failure type may be removed.

## Outcome-blind predictors

All predictors use only current paths, current conflict edges, static grid
degree, low-degree corridor segments, and temporal overlap of current paths.
No scalar weights, learned thresholds, fixed neighborhood size, or transitive
component closure is permitted.

1. `direct_conflict_boundary`: unselected endpoints of current conflict edges
   crossing the selected-set boundary.
2. `spatial_low_degree_frontier`: the nondominated unselected agents under
   selected-partner count, shared low-degree segment count, and shared
   low-degree cell count.
3. `temporal_corridor_frontier`: the nondominated unselected agents under
   temporal selected-partner count, opposing overlap count, and overlap
   duration.
4. `repair_dependency_frontier`: direct-conflict boundary union the
   nondominated frontier over all direct, spatial, and temporal dimensions.
5. `full_temporal_boundary_reference`: all temporal-corridor boundary agents;
   this is a density reference only and cannot become a candidate.

The primary predictor is frozen as `repair_dependency_frontier`. The audit
does not select whichever predictor looks best after outcomes are read.

## Gates

The primary predictor must satisfy every gate:

- overall observed-blocker recall at least 0.50;
- every map recall at least 0.40;
- each fixed eight-seed half recall at least 0.45;
- overall recall exceeds direct conflict boundary by at least 0.10, with a
  strict positive advantage on every map and in both halves;
- mean recall lift over a matched-size random outside set at least 0.15 and
  positive on every map;
- mean added-agent/base-neighborhood ratio at most 0.50;
- no resulting neighborhood reaches 80% of all agents.

Failure of any gate stops pool construction. Passing authorizes only a new,
separately preregistered paired PP replay of the frozen pre-action rule. It
does not authorize training or TTF.

## Parallel execution

The audit uses exactly 16 worker processes. Scientific evaluation jobs are
globally scheduled at `state x trial_index x predictor` granularity, and a
single coordinator writes final artifacts atomically. Worker identities are
recorded only in a separate execution report. Scientific rows contain no
runtime or worker fields. The last 32 evaluation tasks must be handled by more
than one process; otherwise tail parallelism is reported as an execution
failure and the audit is rerun without changing scientific definitions.
