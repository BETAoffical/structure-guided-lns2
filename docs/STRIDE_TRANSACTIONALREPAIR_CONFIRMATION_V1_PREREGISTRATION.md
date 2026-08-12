# STRIDE TransactionalRepair Confirmation v1 Preregistration

## Question and boundary

Can the frozen TransactionalRepair mechanism reproduce its discovery-cohort
effect on a map-disjoint, previously selected result-blind state cohort without
changing the neighborhood policy, retry policy, or readiness gates?

This is an independent mechanism confirmation. It is not training, runtime
integration, a default-controller change, or a raw-TTF experiment. A passing
result authorizes only a separately preregistered runtime semantic-integration
validation.

## Independent result-blind cohort

The cohort contains all 240 states frozen by the earlier repairability
selection before candidate repair outcomes were collected:

- 22 MovingAI DAO/game maps, with zero map overlap with the three Maze maps in
  the TransactionalRepair discovery cohort;
- 120 states from `official_adaptive` traces and 120 from `v2-full` traces;
- 174 train-split and 66 validation-split states;
- 80 early, 80 middle, and 80 late decisions;
- conflict bands: 101 states at 1--10, 85 at 11--100, 52 at 101--500, and 2
  above 500.

No state is selected, removed, or weighted using its source repair outcome or
the new TransactionalRepair outcome. The selected neighborhood is the agent
membership already chosen at the frozen source decision. Only that pre-repair
membership is extracted from the registered source trace; the historical
after-state, PP outcome, runtime, and future trajectory are forbidden.

The exact cohort, source manifests, source run, discovery inputs, and reports
are checksum-pinned. Every one of the 240 states is mandatory.

## Frozen intervention

The three discovery policies are copied byte-for-byte at the protocol level:

1. `selected_single_attempt`: one native PP attempt.
2. `transactional_set_retry`: after a rolled-back
   `conflict_bound_exceeded`, append at most eight first-observed external
   blockers and retry once.
3. `transactional_set_order_retry`: use the same set retry and, only when no
   distinct blocker remains, permit one pre-state conflict-priority order
   retry.

All policies share the same first attempt and paired PP seed within each
state/trial. Trial indices remain 0 through 15 with fixed halves 0--7 and
8--15. Failed attempts must restore the exact repair fingerprint and conflict
count before a retry. Time-limit failures are not retried, and successful
native attempts are committed without post-hoc outcome rejection.

## Frozen gates

The discovery thresholds are reused without relaxation:

- returned-unchanged rate reduction at least 0.15 overall and at least 0.10 in
  each fixed seed half;
- native replan-success improvement at least 0.15;
- strictly lower returned-unchanged rate on every one of the 22 maps;
- strict conflict-reduction rate not lower than the single-attempt baseline;
- joint set/order retry not worse than set-only;
- mean attempts at most 2.25;
- mean added-agent ratio at most 0.50;
- maximum final-neighborhood fraction below 0.80;
- complete result-blind cohort integrity, zero errors/timeouts, 16-worker
  execution, and tail parallelism.

Any failed gate stops TransactionalRepair without threshold tuning, state
filtering, model fitting, runtime integration, or TTF testing.

## Execution and recovery

There are 240 x 16 x 3 = 11,520 atomic policy jobs. Non-TTF collection uses 16
global workers at `state x trial_index x policy` granularity. Every policy job
has a 300-second hard limit and writes a unique atomic artifact. A single
execution error or timeout stops new dispatch immediately; completed artifacts
are retained, and recovery may rerun only missing registered jobs. The final
jobs must remain parallel rather than collapsing to one state worker.
