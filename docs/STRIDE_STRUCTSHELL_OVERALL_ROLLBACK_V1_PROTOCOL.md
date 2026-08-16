# StructShell State-Bounded Rollback V1 Protocol

## Status and purpose

This protocol is frozen before any solver episode for the new implementation is
run.  It is a same-cohort diagnostic repair of a control-logic defect, not a
promotion experiment and not a new ranking-model study.

The previous rollback-aware screen completed 10 paired keys with zero execution
errors or process timeouts, but failed its TTF screen.  Relative to V2, the
challenger added 105.661 seconds of restricted TTF; 91.311 seconds (86.42%) of
that difference was additional PP time.  The Room key alone added 84.976
seconds.  Trace audit showed that the candidate-level guard repeatedly rotated
through StructShell candidates at an unchanged repair fingerprint: the Room key
performed 105 exact rollbacks, banned 28 individual candidates, and never
reached its V2 fallback.

## Frozen mechanism

The new implementation has a separate pool, runtime, guard, and experiment
identity.  Historical routed-v1 and rollback-aware routed-v2 behavior and
artifacts must not change or be imported.

For each exact repair-structure fingerprint:

1. Start with the frozen V2 candidate pool plus StructShell candidates only
   when the registered strict conflict-structure gate passes.
2. Count exact `conflict_bound_exceeded` atomic rollbacks from all pure
   StructShell candidates together; candidate identity changes do not reset the
   count.
3. After the third such rollback, latch StructShell off for that fingerprint.
4. While latched, generate and rank a fresh ordinary V2-only pool on every
   decision.  Do not reuse the Hybrid candidate cache or force a cached V2
   anchor.
5. A V2 rollback does not reopen StructShell.  A fingerprint retains its latch
   if execution leaves it and later returns.  A previously unseen fingerprint
   receives an independent budget.
6. Non-exact failures and `time_limit` do not consume the exact-rollback budget.
7. Each decision performs at most one native PP call.  There is no retry,
   rescue, explicit repair order, or PP-seed override.  The existing
   `episode_stream` policy and frozen V2 124-dimensional features/Copeland model
   remain unchanged.

The activation gate is the already registered strict gate:

`conflict_pairs >= 16 AND (active_conflict_agents >= 32 OR largest_component >= 16)`.

It deliberately removes the legacy `total_agents >= 96` shortcut.  The full
StructShell candidate universe is retained; no structural family or size is
deleted from this single-seed diagnostic.

## Frozen diagnostic cohort

Use the same ten visible screen keys and solver seed 16 as the failed screen so
that the control-logic change can be diagnosed directly.  Rerun every arm under
the new current producer; do not import the prior outcomes.

Arms:

- official Adaptive;
- frozen V2 only;
- routed-v1 StructShell without rollback guard;
- the new state-bounded StructShell guard.

Qualification is non-timed and may use 16 workers.  Formal episodes use strict
round-robin serial timing with one worker.  Each episode has a 180-second
wall-clock outcome window, a 240-second child-process fuse, and a 300-second
outer-job fuse.  `wall_timeout` is valid right censoring; an execution error or
process timeout stops the run.

## Diagnostic gates

All identity, initial-state pairing, action legality, fingerprint, one-PP-per-
decision, and timeout-integrity checks must pass.

The new mechanism is worth a fresh result-blind confirmation only if, relative
to V2 on these ten keys, it simultaneously has:

- no lower success count;
- lower mean restricted reset-inclusive TTF;
- paired faster fraction at least 0.50;
- normalized wall AUC no higher;
- P95 restricted TTF and P95 repair decisions no higher;
- no map with both lower success and worse restricted TTF.

Mechanism diagnostics must additionally show no more than three pure
StructShell exact rollbacks at any repair fingerprint before its latch, no
StructShell selection while latched, no latch reopening after a V2 rollback,
and fresh V2 generation while latched.

Failure stops this branch.  Passing permits only a separately preregistered,
fresh-seed result-blind bounded confirmation.  It does not replace V2, establish
an uncapped raw-TTF claim, change the ranking model, or authorize runtime rescue.
