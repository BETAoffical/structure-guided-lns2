# STRIDE PreTail Forced Continuation v1 preregistration

## Question

The MarginalPool replay showed that the deployed ranker is not selecting the
best current-step candidate at most registered checkpoints.  That alone does
not show that choosing the one-step best candidate prevents a later tail.
This diagnostic changes exactly one action at the first structural checkpoint,
then returns control to the matching frozen StructPool or SlotPool controller.

It distinguishes three mechanisms:

1. `actual_selected`: the action that the frozen ranker actually selected;
2. `one_step_oracle`: the best 16-seed current-step action, used only as a
   diagnostic upper bound;
3. `coverage_diverse`: an outcome-blind, low-overlap action that covers the
   largest share of the current conflict structure.

All 45 registered first-structural checkpoints are retained.  Each arm is run
with two strictly paired forced-action PP seeds.  The oracle is derived from
already frozen one-step labels and cannot be used as a runtime selector or a
training label in this experiment.

## Hard execution fuse

Every continuation has three independent upper bounds:

- at most 200 repair decisions from the restored checkpoint;
- at most 300 seconds of reset-inclusive episode wall time;
- a 360-second external process timeout for execution failures.

`repair_limit` and `wall_timeout` are valid right-censored trajectory evidence,
not execution errors.  Censored TTF is never imputed.  An external process
timeout remains an integrity failure.

The collection therefore cannot run indefinitely on an unrepairable state.

### Execution amendment before formal collection

The first one-case smoke run showed that a Python thread pool serialized the
native spawned workers: only one actual solver child was active.  No formal
episode completed and no result was inspected.  Before formal collection the
execution layout was therefore amended to eight independent process shards,
each with one native worker and a disjoint case partition.  WSL exposes 20
logical CPUs and 23 GiB of memory, so eight workers leave capacity for the host
and for high-agent-count memory spikes.  Candidate definitions, seeds, cohort,
outcomes, and all three per-episode fuses are unchanged.

The first eight-shard launch then rejected the legacy TailSwitch qualification
as reset-protocol incompatible before any formal episode.  The protocol was
amended to build one dedicated current-protocol qualification collection and
freeze its hashes before retrying the formal shards.  This qualification step
contains reset-only data and cannot inspect forced-action outcomes.

The dedicated qualification completed with 19/19 unique task-seed resets,
19 nonzero-conflict states, all three registered solver seeds, three Maze maps,
and zero errors.  Its manifest, report, and run configuration hashes are frozen
in the registration before any formal continuation episode.

The attempted eight top-level process launch then exposed the repository's
global atomic collection lock.  Seven processes were rejected and the one
remaining process was stopped before a complete formal manifest was written.
The lock is retained.  The final execution plan uses one lock-owning collector
and 24 batches defined by challenger, treatment policy, paired seed, and arm.
Each batch has unique task-seed keys and runs up to eight native workers inside
the supported collector.  This changes only orchestration; cohort, candidate
actions, seeds, outcomes, and per-episode fuses remain frozen.

## Outcomes and interpretation

Primary outcomes are fixed-200-step normalized conflict AUC, final conflict
count, and success versus right censoring.  Runtime is recorded but is not used
as a causal label and this diagnostic makes no TTF or generalization claim.

The ranking diagnosis is actionable only if the one-step oracle is beneficial
in at least 30% of non-identical paired comparisons, covers both challengers,
and covers at least four tasks.  A comparable `coverage_diverse` result,
especially where the one-step oracle does not win, indicates that the existing
one-step label omits longer-horizon structural information.  Failure of both
arms does not prove the entire pool is defective; it triggers a separate pool
or PP-operator reassessment.

No model is trained before the complete paired analysis passes integrity.
