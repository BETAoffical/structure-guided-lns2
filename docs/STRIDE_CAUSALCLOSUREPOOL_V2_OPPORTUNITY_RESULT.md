# STRIDE CausalClosurePool v2 opportunity result

## Execution integrity

- Frozen development cohort: 78 states.
- Frozen CausalClosurePool actions: 930 candidates.
- Frozen V2 base pool: all 1,367 registered candidates were present.
- Native PP replay: trial indices 0--15 for every new candidate, 14,880
  atomic trial artifacts total.
- Pairing: one identical PP seed for every action at a state/trial index.
- Execution: 16 workers with a 300-second hard per-job limit and fail-fast on
  the first execution error or timeout.
- Result: 14,880/14,880 complete, zero errors, zero timeouts, and zero invalid
  action, state, candidate, seed or before-repair identity rows.
- No runtime, TTF or future-trajectory field entered the labels.  Repair order
  was not controlled and the current ranker was not used.

Collection report SHA-256:
`c2daf6ee13a9aa53e0644c187446d112e123f79d3210b01f6ecd632df3f19c56`.

## Frozen readiness result

The globally preregistered candidate-opportunity gates passed.

- A new candidate robustly beat the best frozen V2 base candidate in 10/78
  states (12.82%), above the 10% gate.
- A new candidate entered the stable frontier in 34/78 states (43.59%), above
  the 10% gate.
- The best new candidate had a positive seed-mean advantage in 16/78 states,
  but only the 10 robust states also satisfied both fixed seed halves and the
  no-progress condition.
- Mean best-new advantage over best V2 was -0.04561.  Passing therefore means
  that useful new actions exist often enough; it does not mean that choosing a
  CausalClosure action by default is beneficial.
- Mean candidate size was 5.75 agents.

The result was heterogeneous by map:

- `maze-128-128-1`: robust opportunity 2/29 (6.90%), stable frontier 6/29
  (20.69%), mean best-new advantage -0.08163.
- `maze-128-128-2`: robust opportunity 2/21 (9.52%), stable frontier 10/21
  (47.62%), mean best-new advantage -0.02913.
- `maze-32-32-4`: robust opportunity 6/28 (21.43%), stable frontier 18/28
  (64.29%), mean best-new advantage -0.02066.

No per-map gate was preregistered, so the global audit passes.  The weak robust
opportunity on both 128x128 maps must nevertheless be treated as a risk in the
independent confirmation and cannot be hidden by the aggregate result.

## Claim boundary and next step

This is a ranker-free one-step pool-opportunity result, not a solver, TTF or
long-tail improvement.  It does not authorize model training, runtime
integration, default replacement, or selection of only the ten successful
states.  It permits only a separately preregistered, result-blind
forced-continuation confirmation on an independent cohort, with each map group
reported separately.
