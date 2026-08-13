# STRIDE CausalTopoPool v1 opportunity result

## Execution integrity

- Frozen development cohort: 78 states.
- Frozen natural-topology additions: 366 candidates.
- Frozen comparison pools: all 1,367 V2 candidates and all 930
  CausalClosure candidates were preserved.
- Native PP replay: trial indices 0--15 for every new candidate, 5,856
  atomic trial artifacts total, with one strictly paired PP seed for every
  action at a state/trial index.
- Execution: 16 workers with a 300-second hard per-job limit and fail-fast on
  the first execution error or timeout.
- Result: 5,856/5,856 complete, zero errors, zero timeouts, and zero invalid
  action, state, candidate, repair or seed identities.
- No runtime, TTF or future-trajectory field entered the labels. Repair order
  remained uncontrolled and the current ranker was not used.

Collection report SHA-256:
`c3c4705e0512c370a14251b0e78c0dc14922f47c4dd7e2574e66b44425afcb8a`.

## Frozen readiness result

The opportunity stage failed its preregistered strong gates.

- A topology candidate robustly beat the best frozen V2 action in 17/78
  states (21.79%), below the 30% gate.
- The hybrid union of frozen CausalClosure and new topology opportunities
  covered 23/78 states (29.49%), below the 40% gate.
- Topology supplied a new stable-frontier action in 47/78 states (60.26%),
  above the 30% gate. This establishes diversity, not robust superiority.
- Topology added an opportunity not already supplied by CausalClosure in
  13/78 states (16.67%).
- Across states, the mean advantage of the best topology action over the best
  V2 action was -0.01030. There were 46 individually robust topology
  candidates, but they were concentrated in too few states and maps.

The failure was map-dependent:

- `maze-128-128-1`: topology opportunity 2/29 (6.90%), hybrid opportunity
  4/29 (13.79%), and stable-frontier addition 10/29 (34.48%). The hybrid
  result failed the 25% per-map gate.
- `maze-128-128-2`: topology opportunity 11/21 (52.38%), hybrid opportunity
  12/21 (57.14%), and stable-frontier addition 19/21 (90.48%).
- `maze-32-32-4`: topology opportunity 4/28 (14.29%), hybrid opportunity
  7/28 (25.00%), and stable-frontier addition 18/28 (64.29%).

Candidate size did not explain away the failure. Robust candidates appeared
in every registered size band, but the average candidate in every band still
had a negative seed-mean advantage over the best V2 action:

| Size | Candidates | Robust candidates | Mean advantage |
| --- | ---: | ---: | ---: |
| 2--8 | 88 | 1 | -0.14644 |
| 9--16 | 80 | 6 | -0.10610 |
| 17--32 | 98 | 19 | -0.07265 |
| 33--48 | 69 | 12 | -0.04784 |
| 49--64 | 31 | 8 | -0.02047 |

Larger candidates were less negative on this one-step label, but this is not
evidence for increasing candidate size: every band remained negative on
average, and this audit contains no continuation or raw-TTF outcome.

## Claim boundary and decision

The integrity gates passed, but the pool-opportunity decision failed. The
result therefore does not authorize result-blind continuation, ranker
training, runtime integration, TTF measurement, or a long-tail-avoidance
claim. Successful states must not be selected for a follow-up cohort.

The exact diagnosis is that natural topology closure restored stable-frontier
diversity but not broad, map-robust one-step repair opportunity. Development
stops at this gate. Any successor must be separately preregistered and revise
the topology relations or repairability definition using an outcome-blind
cohort; it must not tune these frozen candidates to the observed winning
states.

## Validation

- WSL Python: `839 passed, 35 skipped` with 16 workers.
- Linux native CTest: `11/11` passed with `-j16`.
- Windows native `lns2_tests.exe`: passed.
- Repository hygiene, retention hashes, and `git diff --check`: passed.
