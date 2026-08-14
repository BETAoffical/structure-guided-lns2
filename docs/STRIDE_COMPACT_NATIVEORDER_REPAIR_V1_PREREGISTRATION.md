# STRIDE Compact Native-Order Repair v1 Preregistration

## Question

Does removing only unsupported members from the frozen selected neighborhood
reduce native PP rollback and wall cost without sacrificing repair success?
This is a direct test of semantic compaction, not a monotone "smaller is
better" claim and not a return to StructPool's fixed 8/16/24/32 sizes.

## Frozen cohort and actions

All 45 frozen `first_repeat_stall` states are retained. Trials 0--7 form the
initial phase and trials 8--15 may run only as one uniform extension.

Four policies form a 2x2 paired design:

1. full selected set, one native PP attempt;
2. full selected set, at most one fresh-seed native retry;
3. semantic compact set, one native PP attempt;
4. semantic compact set, at most one fresh-seed native retry.

The first PP seed is common to all four policies. Retry policies share one new
seed. No policy requests an explicit repair order. Retry is permitted only
after exact native rollback caused by `conflict_bound_exceeded`, never after a
native `time_limit`.

## Semantic compaction

The compact set retains every selected agent incident to a current conflict
edge. It also retains selected agents with action-visible temporal overlap and
opposing-flow evidence in low-degree or articulation corridors relative to the
current conflict core. Only selected agents with neither kind of support are
removed. If no such agent exists, the compact action is exactly the full set.

There is no requested target size, minimum size or maximum size. Eligibility is
determined before PP and without outcomes. Primary size-effect analysis uses
all eligible states; all 45 states remain in the collection and safety report.

## Execution and gates

Collection uses 16 workers, one atomic artifact per state/trial/policy, a
270-second total native PP transaction budget and a 300-second outer process
limit. The first execution error or process timeout stops the phase.

Uniform extension requires, on eligible states, compact retry to have a lower
returned-unchanged rate and lower mean total PP wall, while success and strict
conflict reduction do not decrease. No map may worsen by more than five
percentage points. The extended gate additionally requires the 95% clustered
bootstrap upper endpoints for both returned-unchanged and PP-wall differences
to be below zero.

Passing authorizes only a separately preregistered bounded platform
continuation. It does not authorize a candidate-pool replacement, runtime
integration, TTF testing or a platform/long-tail claim.
