# STRIDE Platform-Entry Order V1 Execution Amendment

## Trigger

The first formal run stopped before any outcome analysis at 89/288 scheduled
jobs.  One 600-agent `native_order` episode completed its first forced PP step
in 117.36 seconds, then entered a second PP call with roughly 60 seconds left
in the registered 180-second episode window.  The second call did not return
before the independent 240-second process fuse.

The partial trace proves that this was not an unbounded LNS decision loop.  The
native PP implementation checked its remaining repair budget only between
agents.  A single SIPP or space-time A* search could therefore overrun the
deadline without an internal check.

## Root fix

The existing PP repair deadline is now propagated into every single-agent
SIPP and space-time A* search.  A budgeted diagnostic search checks the
deadline before every expansion; ordinary unbudgeted solver calls retain the
original loop.  If the deadline expires, the search returns an explicit timeout,
`InitLNS::runPP` records the failed agent and order position, and the complete
neighborhood is restored through the existing atomic rollback path.  The
closed-loop episode can then finalize as valid `wall_timeout` right-censoring
instead of leaving a process-timeout partial.

This does not add PP retries, change either registered order arm, skip a case,
or lengthen any scientific or process bound.

## Recovery identity

- Preserve `build/stride-platformentry-order-v1` as superseded evidence.
- Import none of its 88 completed episodes because native behavior changed.
- Restart all 288 initial jobs in
  `build/stride-platformentry-order-v1-r2` with a new producer and run
  fingerprint.
- Keep the original 64-decision, 180-second wall, 240-second process and
  300-second outer-job bounds.

No platform-effect result from the superseded run was inspected or used to
choose this amendment.
