# STRIDE Bounded Native Retry Continuation v1

## Question

NativeOrder already established that, after an exact PP rollback, retrying the
same agent set with a fresh native PP seed improves the immediate transaction.
This experiment does not repeat that screen. It asks whether a bounded version
changes the subsequent trajectory: fewer persistent platforms, no success loss,
lower fixed conflict AUC, and fewer restricted-mean repair decisions.

## Frozen design

- Cohort: all 45 registered `first_repeat_stall` states on three maps.
- Arms: the registered first native attempt versus the same attempt followed by
  a fresh-seed same-set retry when the platform threshold is reached.
- First transaction: same state, neighborhood, first seed, and retry seed as the
  completed NativeOrder artifacts. Analysis must prove field-level parity.
- Continuation: the matching frozen StructPool/SlotPool controller after the
  transaction; no learned intervention policy and no explicit repair order.
- Window: success or 64 repair decisions / 180 seconds. Process and outer job
  fuses are 240 and 300 seconds.
- Qualification: build one current-producer reset-only qualification for all
  registered task/seed keys under each frozen controller implementation
  identity (StructPool and SlotPool), using 16 workers and a 360-second
  reset-process fuse before any candidate PP runs. The strict
  controller-bound reuse fingerprint is retained. This does not alter the
  240-second scientific episode process fuse.
- Retry bound: only `conflict_bound_exceeded` plus exact rollback; never after
  `time_limit`; once per platform signature; at most three per episode.

The retry is transactional inside one controller decision. A failed first PP
and its retry are both charged to repair wall time, while the conflict trajectory
records the final committed state of that transaction.

## Sequential gate

Trial 0-3 contains 360 paired episodes. Extension to trial 4-7 is uniform across
all states and both arms, and is allowed only if platform risk moves downward,
success does not decrease, and no map worsens by more than five percentage
points. The final gate additionally requires the state-cluster paired bootstrap
upper 95% bound below zero, lower normalized fixed AUC, and lower restricted-mean
repair decisions.

This is mechanism evidence only. Passing does not authorize runtime integration,
TTF claims, model training, or default-controller replacement.
