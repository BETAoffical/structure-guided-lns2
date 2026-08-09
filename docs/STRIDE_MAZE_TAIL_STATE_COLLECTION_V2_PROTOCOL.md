# STRIDE Maze Tail State Collection v2 Protocol

## Why v2 exists

The unbounded v1 episode block was stopped after a StructPool episode consumed
one CPU core for more than 46 minutes.  The research objective is to capture
the state and action conditions that precede harmful repair tails, not to spend
unbounded time proving that every controller can eventually repair every state.

## Evidence fuse

The same 33 reset-qualified keys and three controllers are restarted in a new
output directory.  Each episode ends normally at the first of:

- feasibility;
- 200 completed repair decisions;
- 300 reset-inclusive wall seconds.

The episode-process timeout is 360 seconds and is only a safety failure.  The
200-decision and 300-second limits are scientific evidence fuses: they finalize
the trace and record final conflicts, conflict trajectory, fixed-budget AUC,
controller timings, selected neighborhoods, and `repair_limit` or
`wall_timeout`.  Such rows are valid right-censored trajectories, not execution
errors and not claims that the underlying solver can never finish.

The 200-decision fuse is enforced by the trace-producing outer episode loop.
The native environment keeps `max_repair_iterations=0` and `time_limit=0`
with `unlimited_time=true`, which preserves the registered reset-protocol
identity.  The 300-second limit belongs to the outer trace loop and therefore
does not alter reset or individual PP semantics.

## Analysis

Completed pairs retain the registered repair-iteration comparison.  When only
the challenger is censored and V2 completes, the challenger is a severe tail;
when only V2 is censored and the challenger completes, the challenger is
beneficial.  If both are censored, comparison uses the preregistered 200-step
normalized conflict AUC and final-conflict deltas.  Censored TTF values are
never imputed as actual TTF.

All 99 entries remain mandatory.  No map, task, seed, or controller result may
be removed.  The block is for tail-incidence and first-divergence coverage;
training, default promotion, fresh-map generalization, and formal speed claims
remain forbidden.
