# STRIDE StructPool Lazy v1

`stride-structpool-lazy-v1` is the semantics-preserving Stage 1 runtime
implementation that precedes any candidate-size or selection-policy change.
The frozen V2 pool, frozen V2 ranker, PP/SIPPS behavior, explicit repair seeds,
and realized neighborhoods are unchanged. The known Maze-128 long-tail case is
excluded from these timing measurements as preregistered.

## Implementation

- Candidate generation is split into context, draft, reduction, and finalization
  phases. Full audit and score fields are computed only for candidates that need
  comparison or final output.
- `TopologyAnalysisCache` updates path heat only for agents changed by the last
  repair. Native conflict events remain the authoritative event source, and a
  complete shadow reconstruction is checked every 20 incremental updates.
- The native topology pass and 124-dimensional feature extractor share one
  prepared analysis capsule within a decision. A state-identity check prevents
  reuse with another state object.

## Exactness and performance

All microbenchmarks use the 44 StructPool-activated states from the retained
Maze300 and Room500 development Quick. They do not include the known Maze-128
long-tail regression.

| Optimization | Exactness | 11-run result |
| --- | --- | --- |
| Lazy candidate finalization | 264/264 candidate JSON rows exact | 11/11 faster; median 0.9271s to 0.8828s (-4.79%) |
| Incremental path heat | State analysis signature exact | 11/11 faster; median 0.5892s to 0.4351s (-26.17%) |
| Shared native analysis | All 124 feature values exact | 11/11 faster; median 1.1354s to 0.8172s (-28.02%) |

The final 24-episode three-controller Quick completed without errors, invalid
actions, or fingerprint mismatches. The StructPool controller succeeded on all
8 paired keys. Across its 93 repair transitions, candidate/action/PP-seed and
conflict-trajectory semantics exactly matched the pre-sharing reference.
Mean neighborhood-selection time changed from 0.5226s to 0.4946s per episode
(-5.36%). Raw TTF changed from 11.7698s to 11.6762s, but that single Quick is
reported only as a runtime sanity check, not as a new solver-speed claim.

## Decision

The three changes are retained as pure code optimization. They do not validate
the current fixed family sizes. Stage 2 must therefore evaluate the complete
family-by-size grid with paired one-step PP labels before `stride-scalepool-v1`
is implemented.
