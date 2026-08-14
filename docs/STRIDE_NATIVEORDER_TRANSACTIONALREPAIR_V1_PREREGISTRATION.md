# STRIDE Native-Order TransactionalRepair v1 Preregistration

## Question

After the selected neighborhood has failed with an exact native PP rollback,
can one bounded retry recover quickly without searching another candidate or
controlling the native PP order?

This is a post-failure repair experiment. The base neighborhood still comes
from the frozen source selector. It neither claims that the candidate pool is
complete nor scans the 44-action HybridStructPool at runtime.

## Why the old result is insufficient

The previous `transactional_set_retry` preserved the first failed repair-order
prefix and appended blockers at its tail. Its 65.97% success rate is therefore
not evidence for a fresh native random-order retry. The 80.69% set-plus-order
arm additionally used a conflict-priority order and up to three attempts. It is
retained as mechanism evidence, not as the deployable policy tested here.

## Frozen cohort and phases

The discovery cohort contains all 45 frozen `first_repeat_stall` states and
three Maze maps. Trials 0--7 form the initial phase. All states and policies
are extended uniformly to trials 8--15 only when at least one native-order arm
has a lower returned-unchanged rate, no lower success or strict-conflict-
reduction rate, and no map worsens by more than five percentage points.

The cohort is outcome-enriched and cannot establish generalization. A passing
result permits only a separate result-blind confirmation.

## Policies

All policies make the same first native PP attempt with the same paired seed.
A retry is allowed only after `conflict_bound_exceeded` and verified exact
rollback. `time_limit` is never retried and a successful attempt is committed
immediately.

1. `selected_single_attempt`: no retry.
2. `same_set_native_retry`: retry the identical agent set once with a fresh
   paired PP seed and no explicit repair order.
3. `blocker_augmented_native_retry`: add at most eight external blocker agents
   exposed by the failed attempt, then retry once with the same fresh paired PP
   seed and no explicit repair order. If no blocker was exposed, this reduces
   to the same-set native retry.
4. `preserved_prefix_blocker_tail_upper_bound`: preserve the first applied
   order and append the observed blockers. This is a diagnostic upper bound,
   never a deployable arm.

Every policy uses at most two PP attempts. Attempt wall time and total policy
wall time are recorded as cost diagnostics, but this concurrent mechanism
study is not a raw-TTF experiment.

## Interpretation and gates

The primary outcome is the paired risk difference in returning the exact
unchanged repair state. Final inference uses a 10,000-replicate paired
state-cluster bootstrap, so 16 trials from one state are not treated as 16
independent states.

A native-order arm passes only if the upper endpoint of its 95% paired risk-
difference interval is below zero, success and strict conflict reduction do
not decrease, no map worsens by more than five percentage points, the attempt
and expansion limits hold, and collection has zero execution errors/timeouts.

- same-set and augmented arms matching means native order/randomness dominates;
- augmented strictly beating same-set means failure-conditioned membership is
  useful beyond a fresh random retry;
- only the preserved-order upper bound passing means the mechanism is not
  deployable under native LNS2 ordering;
- no arm passing stops this bounded retry branch.

No result authorizes training, runtime integration, TTF testing, controller
promotion, candidate-pool replacement, or a long-tail claim.

## Execution

Non-TTF collection uses 16 workers at
`state x trial_index x policy` granularity. Each policy job has a 300-second
hard process limit and writes one atomic artifact. Resume reruns only missing
or invalid jobs. The four-key qualification case is excluded from scientific
analysis and checks native-order, retry-seed, blocker, and rollback semantics.
