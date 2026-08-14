# STRIDE FrontierDependencyPool v1 report

## Outcome

The preregistered zero-solver readiness audit passed. It ran 45 independent state jobs with 16 workers and generated no new PP, continuation, training, runtime, or TTF result.

- States: 45 across all three registered Maze maps.
- States with a legal frontier candidate: 42/45 (93.33%; gate 90%). The other three retain the frozen historical action as an explicit fallback.
- New candidates: 235, comprising 118 `compact-augment` and 117 `same-size-exchange` actions.
- Both variants were present on every map.
- All candidates used only pre-action current conflict-frontier and persisted-edge history; no PP probe or future outcome entered generation or deterministic selection.
- Every added agent had an explicit current selected-to-unselected conflict edge, every candidate converted at least one such edge to an internal repair dependency, and all size/evidence gates passed.

This is candidate-generation readiness only. It does not show that the pool prevents platforms or improves TTF.

## Why size alone is not the diagnosis

An additional read-only audit used the existing Repairability causal collection. Across 715 applicable state/trial pairs, the original selected neighborhood averaged 28.63 agents and native PP succeeded in 35.66%. Adding an average of 7.56 observed blockers made the neighborhood larger but raised success to 57.90% and reduced no-progress from 67.55% to 45.59%.

The size-32 subset is the clearest counterexample to a monotone size explanation: adding about 7.51 blockers raised success from 35.83% to 59.51%. With the identical enlarged agent set but blocker-first order, success was only 37.04%. Larger neighborhoods can increase sequential-PP fragility, but membership and order dominate a simple "smaller is safer" rule.

The new pool therefore compares two bounded set interventions without controlling native PP order:

- `compact-augment`: keep the frozen controller's chosen core and add at most eight agents from one coherent current conflict-frontier motif; total size is capped at 40.
- `same-size-exchange`: add the same kind of supported frontier agents while removing an equal number of low-support, non-protected core agents; the original core size is preserved exactly.

There is no fixed 8/16/24/32 preference. Candidate size follows the frozen selected core and the visible motif.

## Static result

| Variant | Candidates | Mean size | Range | Mean added agents |
| --- | ---: | ---: | ---: | ---: |
| Compact augment | 118 | 36.24 | 18--40 | 6.07 |
| Same-size exchange | 117 | 30.15 | 16--32 | 6.09 |

The old six-case escape-blocker diagnostic remains intentionally non-gating. No new candidate fully reconstructed those later escape sets. This is expected because those blockers were measured at a later repeated-stall checkpoint, while the new candidates are generated at the first structural action. The direct continuation experiment, not post-hoc set reconstruction, must decide whether the pre-action motifs reduce platform entry.

## Next registered boundary

The pass permits a separately preregistered bounded forced-continuation experiment over all 45 first-structural states. It will use one native PP call for the forced action and then return to the identical frozen continuation controller. The initial arms are the historical action, V2 anchor, and every frozen FrontierDependency candidate; deterministic augment/exchange results are read from their preregistered candidate IDs, while a post-hoc pool upper bound remains diagnostic only.

The primary endpoint is entry within 64 decisions into at least three consecutive exact PP rollbacks with unchanged repair fingerprint and conflict edges. Trial indices 0--3 run uniformly first with 16 independent episode workers. No model training, runtime integration, repeated-PP fallback, or TTF test is authorized by this static result.

Canonical zero-solver outputs are under `build/stride-frontierdependencypool-v1/`. The cohort SHA-256 is `079bf77f903a3ccf3509036228f226be5a1a324a8bff199ca5f574d26c6fcf79`; the run fingerprint is `ca62f68b7d689eb017ddf7af577fcf0332b5ac6a9fb7feab502f92378b7a54cf`.
