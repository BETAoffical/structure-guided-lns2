# STRIDE CausalClosurePool v2 r3 compactness result

## Result

The event-time-slice generator passed every preregistered compactness and
integrity gate on all 78 frozen development states.

- 930 candidates were materialized; every state retained 11 or 12 candidates.
- Candidate size mean was 5.74 versus 51.57 for RepairClosurePool v1.
- Candidate size median was 4 versus 54 for v1.
- P90/P95/maximum sizes were 10/17/58 versus 63/64/64 for v1.
- Mean and median size ratios to v1 were 0.1113 and 0.0741.
- 1,240/135,855 core-family closures were rejected as organically oversized,
  an oversized-family rate of 0.91% versus the frozen 10% maximum.
- No core was intrinsically larger than 64 and no agent was admitted by
  whole-path overlap alone.
- All 78 states materialized without execution or identity errors; the cohort
  carries the original solver run configuration and its registered SHA-256 for
  later exact native restoration.

Per-map mean/median sizes were 5.43/4 on `maze-128-128-1`, 4.38/4 on
`maze-128-128-2`, and 7.08/5 on `maze-32-32-4`.  The largest candidate was 58
agents and occurred in the smaller maze group; large neighborhoods remain
possible when a single event-time causal slice genuinely connects them.

Compactness report SHA-256:
`3191d56e96cdfcf2e4d8558290a6657ca7001ae1a23c69d35978e18a3f997768`.

Cohort manifest SHA-256:
`4ffaf2c4f62bbee2b8c0a54b6bbf85cccf2c530dbbd90f2c6bceb96332e9cc50`.

## Claim boundary

This result establishes only that the previous systematic size inflation was
removed by event-local causal membership rather than a smaller hard cap.
Candidate outcomes, native PP, the current ranker, future trajectories,
continuation and TTF were not used.  It does not establish repair quality or
long-tail avoidance.  Passing permits a separately preregistered ranker-free,
strictly paired native-PP opportunity audit.
