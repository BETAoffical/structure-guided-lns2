# STRIDE Repairability Basin Audit v1 Protocol

## Question

This audit asks why a compact blocker rescue either remains on the original
repair fingerprint or escapes once and then enters another exact-rollback
platform. It distinguishes a difficult solution state from an individual
failed PP call: the shared failed call rolls back exactly and therefore does
not itself alter the solution state.

## Fixed population

The audit reads all 81 registered compact-blocker trigger events from the
completed trial-0--3 collection. Their already-known outcome partition is 50
immediate unresolved and 31 immediate escapes. Of the latter, 23 have complete
H=8 observations: 16 remain durable and seven form a new platform; eight are
right-censored. No event may be removed using its diagnostics or outcome.

## Diagnostics

For the initial failure and the single rescue, the audit records neighborhood
membership, compaction, selected blockers, failure agent and order position,
internal and external blockers, new conflict pairs and PP wall. For decisions
through H=8 it records blocker retention, candidate-set repetition, Jaccard
overlap with the rescue action and newly observed external blockers.

A new-platform window consists of the three exact rollbacks ending at the
registered first new-platform offset. New blockers are agents observed in that
window but absent from the original rescue neighborhood.

## Frozen mechanism rules

Accumulated-blocker rescue is supported only if at least 40 immediate failures
are present, at least half expose a residual external blocker during rescue,
and that presence rate exceeds the immediate-escape group by at least 15
percentage points.

Per-signature rescue is supported only if at least five new-platform events
are present and at least half expose a new external blocker outside the
original rescue neighborhood in their platform window.

If neither rule passes, the stateful membership branch stops. The remaining
failure is treated as repair order, internal coupling or residual PP
instability rather than addressed by adding more agents.

## Boundary

This is a 16-process, read-only, zero-solver audit. It cannot train a model,
change the runtime, run TTF or promote a mechanism. If a rule passes, it only
authorizes a separately registered bounded screen with one PP call per
decision, native order, at most three interventions per episode and at most
one intervention per platform signature.
