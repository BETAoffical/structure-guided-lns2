# STRIDE MultiValue Episode Parallelism Amendment

## Scope

This execution-only amendment accelerates the four remaining initial-label
states without changing candidates, teachers, trial indices, PP seeds, rollout
horizons, censoring rules, or labels.

## Registered execution rule

- The passed outcome-blind worker preflight remains the authority and selected
  12 global episode workers.
- The collector distributes those 12 slots deterministically across the active
  state workers. Four active states receive three episode slots each; the
  allocation expands as states complete.
- Every episode keeps its existing identity and isolated collection directory.
- Only the owning state worker writes its `.json.partial` artifact. Successful
  sibling episodes are atomically checkpointed even if another episode fails.
- Existing state and episode fuses remain 1800, 360, 300, and 128 respectively.
- Productive state windows continue without consuming the four genuine-failure
  recovery budget. Two consecutive zero-progress windows still stop recovery.

## Recovery identity

Both prior registered run fingerprints are accepted only as recovery sources.
Newly written partial and complete artifacts use the amended producer/config
identity. No state or episode is selected according to an outcome.

## Claim boundary

This is an execution-throughput change only. It is not evidence for label
stability, candidate quality, solver TTF improvement, or model promotion.
