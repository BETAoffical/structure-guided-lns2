# STRIDE TransactionalRepair v1 Preregistration

## Question

Can the PP self-cycle be prevented before a solver state transition is
committed by treating repair as a bounded transaction: tentatively run the
selected neighborhood, use the failed attempt's native blocker cut to revise
the same candidate, and commit only the first successful attempt?

This differs from a stall guard. The retry happens inside one decision while a
failed native PP attempt has already restored the original paths. No failed
action becomes a solver transition, no repeated-state trigger is required, and
no future trajectory or TTF label is consulted.

## Why this study follows the static predictor failure

The causal audit found set defects in 23/45 states and order defects in 15/45,
but the outcome-blind static RepairDependency predictor could not recover the
blockers compactly. Its recall was 0.5472 with a 0.9706 mean expansion ratio;
the compact spatial and temporal frontiers had recalls of only 0.0784 and
0.1007. The missing dependency is frequently revealed only after PP generates
a candidate-conditioned alternative path.

TransactionalRepair uses that internal failure cut before commitment. It does
not train a model on outcome-enriched blocker labels and does not widen every
candidate through static transitive closure.

## Frozen cohort and pairing

The discovery audit contains all 45 frozen `first_repeat_stall` states, three
Maze maps, and trial indices 0 through 15. No state may be removed based on the
new policy's outcome. Every state/trial restores the same pre-state and uses the
same paired PP seed for all three policies.

This is a mechanism-discovery cohort. Even a passing result cannot establish
generalization because the states previously exposed the causal mechanism. A
separate result-blind cohort is mandatory before runtime or TTF evaluation.

## Frozen policies

1. `selected_single_attempt`: execute the selected neighborhood once in its
   native PP order.
2. `transactional_set_retry`: after `conflict_bound_exceeded`, append at most
   eight first-observed external blockers to the neighborhood and to the tail
   of the applied repair order, then attempt once more.
3. `transactional_set_order_retry`: perform the same bounded set retry; if no
   distinct blocker addition is available and the action still fails, attempt
   one conflict-priority order using the pre-state conflict degree.

All attempts use native PP diagnostics. A failed attempt must report exact
rollback, unchanged repair fingerprint, and unchanged conflict count before
another attempt is allowed. `time_limit` is never retried. A successful native
attempt is committed immediately; it is not post-hoc rejected using its
conflict result.

The maximum attempts are 1, 2, and 3 respectively. The maximum number of added
agents is eight. These limits are frozen from the prior causal design, not
selected from the new results.

## Outcomes and gates

The primary mechanism outcome is the rate of returning the exact unchanged
pre-state after the bounded transaction. Secondary outcomes are native replan
success, strict conflict reduction, normalized conflict reduction, attempts,
and added agents.

All gates must pass:

- returned-unchanged rate improves by at least 0.15 overall and 0.10 in both
  fixed eight-seed halves;
- native replan-success rate improves by at least 0.15;
- every map has a strictly lower returned-unchanged rate;
- strict conflict-reduction rate does not decrease;
- the joint set/order policy is not worse than the set-only policy;
- mean attempts do not exceed 2.25;
- mean added-agent ratio does not exceed 0.50;
- maximum total-neighborhood fraction remains below 0.80;
- integrity, zero-error/timeout, 16-worker execution, and tail parallelism all
  pass.

Failure stops this approach without threshold changes, state filtering,
training, runtime integration, or TTF.

## Execution

Non-TTF collection uses 16 globally shared workers at `state x trial_index`
granularity. Each job writes one atomic state-trial artifact and has a 300
second process timeout. Resume reruns only missing jobs, never only successful
states. This keeps the final jobs parallel while preserving one-writer artifact
semantics.

Formal raw-TTF measurement is explicitly out of scope and will continue to use
isolated or fixed-low-concurrency execution if later authorized.
