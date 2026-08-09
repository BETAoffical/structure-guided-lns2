# STRIDE SafeSlot Label Readiness v1

## Purpose

This registered retrospective audit is the first data step after freezing the
SafeSlot two-stage interface. It asks whether the existing paired one-step
outcomes contain enough stable structural improvements over an exact base-only
V2 anchor to justify collecting post-repair residual-structure teacher fields.

It does not train a model, select an inference threshold, execute a controller,
or make a TTF claim.

## Exact anchor and candidate sources

The 98-state four-size grid stores an outcome-blind V2 anchor selected from the
original candidate pool only. The anchor's 16 paired PP trials come from the
complete RobustAction label collection. Structural challenger trials come from
the four-size grid collection. For each state and trial index, the PP seed must
match exactly before the pair is admitted.

The frozen map-grouped SlotPool out-of-fold selections are used only to measure
whether its six-candidate budget retains SafeSlot opportunities. No SlotPool
model is retrained.

## Pre-residual positive upper bound

A challenger is a pre-residual positive only when all conditions hold:

```text
mean challenger-anchor reduction >= 0.02
paired strict-win fraction >= 0.75
first fixed-half mean advantage > 0
second fixed-half mean advantage > 0
challenger no-progress rate <= anchor no-progress rate
```

This is an upper bound, not a final SafeSlot label. A final positive still
requires the one-step post-repair residual-structure teacher condition frozen
in the SafeSlot interface document. Existing artifacts do not contain that
teacher field.

## Decision rule

Post-state teacher collection is authorized only if every registered readiness
gate passes. Passing does not authorize training. Failing stops the expensive
recollection step and requires label or candidate-pool reassessment without
filtering states by their outcomes.

All 98 states remain in the audit. The known catastrophic Maze regression was
already excluded before the four-size cohort was constructed and is not read.
