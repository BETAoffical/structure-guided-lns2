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

## Canonical-module correction

The first replacement run (`r2`) also stopped at 89/288, on the paired
`conflict_priority_order` arm of the same 600-agent state.  Its producer
identity and partial trace proved that the launcher had loaded
`build/linux/project/lns2_env...so`, whose hash still matched the pre-fix
binary.  The targeted validation had loaded the newly built sibling module
from `build/linux`, so it did not expose this path mismatch.

The current source is now rebuilt into the repository's registered canonical
module directory, `build/linux/project`.  The native producer identity must
therefore name that path and carry the new binary hash.  Preserve `r2`, import
none of its 88 completed episodes, and restart all 288 jobs under the new
`build/stride-platformentry-order-v1-r3` identity.  This correction changes no
scientific action, seed, schedule, or time limit.

## Outer-budget propagation correction

The canonical `r3` run passed all 17 qualification jobs but stopped at 85/288
scheduled jobs.  The failed 600-agent `native_order`, trial-1 episode had
completed five transitions by 173.52 seconds and then began one more PP call.
Its native environment used the registered unlimited-time sentinel because
the 180-second limit is reset-inclusive and owned by the Python closed-loop
runtime.  Consequently, the low-level deadline added above still received the
native sentinel rather than the 6.48-second live outer remainder.  A serial
reproduction completed that last PP after 31.43 seconds, at 204.94 seconds,
and showed a 24.94-second scientific-budget overshoot.  Under the formal
16-worker load the same missing propagation reached the independent
240-second process fuse.

The closed-loop runtime now passes the live reset-inclusive wall remainder to
each native PP invocation through a separate `step_with_time_limit` binding.
`InitLNS::runPP` intersects that value with its native limits, including in
the no-diagnostics path.  An expired invocation performs no low-level search,
records `time_limit`, and restores the entire old neighborhood atomically.
Ordinary `step()` callers retain their previous native timing semantics.

The exact failed r3 job, state, arm, trial and PP seed now closes as a valid
`wall_timeout` episode in 180.80 process seconds, with only 0.14 seconds of
finalization overshoot and no budget-external conflict improvement.  Preserve
`r3`, import none of its 84 valid episodes, and restart all 288 jobs in
`build/stride-platformentry-order-v1-r4`.  The 64-decision, 180-second wall,
240-second process and 300-second outer-job limits remain unchanged.
