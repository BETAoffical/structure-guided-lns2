# STRIDE RepairClosurePool v1 preregistration

## Question

Does a repair-prestate-only dependency closure add robustly useful native PP
actions that are absent from the frozen V2 base pool?

Candidate generation and candidate selection are separated.  The current
ranker is not used.  All old base candidates remain available, every new action
is exhaustively replayed with paired native PP seeds, and Oracle results are
reported only as candidate-pool opportunity.

## Generator

Each frozen V2 anchor and each current conflict component is a core.  An
outside agent receives four independent evidence counts: direct conflict
events with the core, same-cell reservations within two time steps, shared
articulation/low-degree cells, and shared path cells.  Agents are added by
Pareto evidence fronts, never by a scalar evidence score or a preferred-size
grid.  Candidate sizes are the cumulative dependency-front boundaries.

The generator is deterministic, uses no PP failure diagnostic or future
outcome, produces at most 12 challengers of at most 64 agents, rejects exact V2
anchor duplicates, and requires every explicit action to touch a current
conflict.  PP repair order is never supplied.

## Development audit

The development cohort is the complete frozen 90-checkpoint/78-state
MarginalPool diagnostic cohort.  It is intentionally outcome-enriched and can
only develop the generator.  Existing paired 16-seed labels are reused for all
V2 base candidates.  Only novel closure actions receive new native PP replays,
using the identical state/trial seed namespace.

Pool opportunity is reported with no runtime selector: robust improvement over
the best old base action, additions to the stable Pareto frontier, unchanged
rate, two fixed seed halves, neighborhood size, and per-map concentration.
The development readiness screen is frozen before replay: at least 10% of all
78 states must contain a new action that robustly beats the best old base
action, and at least 10% must gain a stable frontier action.  These are pool
opportunity thresholds, not solver-performance claims.
Passing development evidence authorizes only a separately preregistered,
result-blind forced-continuation confirmation.  It does not authorize model
training, runtime integration, TTF testing, or a long-tail claim.

## Execution

Materialization is outcome-blind and precedes collection.  Collection uses 16
workers at `state x candidate x trial` granularity, one atomic JSON per job, a
300-second hard job limit, and fail-fast handling for execution errors or
timeouts.  A two-state/two-trial preflight must pass before the full 16-trial
collection.
