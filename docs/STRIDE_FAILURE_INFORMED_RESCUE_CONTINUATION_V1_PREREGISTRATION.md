# STRIDE Failure-Informed Rescue Continuation V1

## Question

The previous bounded same-set retry reduced persistent-platform entry but failed
its complete gate: fixed-horizon conflict AUC worsened and one legacy parity
check was invalidated by a different wall budget.  This experiment asks the
narrower causal question that remains:

> After a real native PP exact rollback has exposed external blockers, does
> changing the *next decision's* neighborhood membership escape the platform
> more reliably than either the frozen controller or a same-set fresh seed?

It does not test a predictor and does not use a future escape witness as an
online feature.

## Intervention

All three arms restore the same 45 frozen `first_repeat_stall` states and run
the same selected neighborhood, PP seed, and native repair order once.  Native
agent diagnostics are collected on every decision in every arm.

Only when that shared action returns `conflict_bound_exceeded`, exact rollback,
and an unchanged state/conflict signature is a one-time next-decision action
eligible:

- `frozen_controller`: no override; return to the frozen controller;
- `same_set_fresh_seed`: the failed agent set with a fresh paired seed;
- `blocker_augmented_fresh_seed`: the failed set plus at most eight external
  blockers reported by that failure, with the same fresh seed as the same-set
  arm.

There is one PP call per controller decision.  No arm controls native repair
order.  No rescue follows `time_limit`, and no episode receives more than one
rescue.

## Execution and gates

Initial collection is 45 states x 4 trials x 3 arms = 540 episodes, with 16
workers.  Each episode runs to feasibility or 64 repair decisions / 180 seconds;
the process and outer-job limits are 240 and 300 seconds.  Trial 4-7 extension
is uniform across every state and arm and is allowed only if the blocker arm:

1. lowers platform-entry point risk versus the frozen controller;
2. does not lower success;
3. resolves at least as many eligible rescues as the same-set arm; and
4. worsens no map by more than five percentage points.

The final gate additionally requires the paired state-cluster bootstrap upper
bound below zero, lower normalized fixed AUC, lower restricted mean repair
decisions, and no worse platform risk than the same-set arm.

This is mechanism discovery only.  Passing does not authorize runtime
integration, raw-TTF claims, model training, or default replacement.
