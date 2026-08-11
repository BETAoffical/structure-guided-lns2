# STRIDE MultiValue v1 protocol

## Scientific objective

`stride-multivalue-v1` predicts the distribution of future solver states after
choosing a concrete neighborhood.  Tail risk is an observed consequence, not a
standalone label.  No single remaining-rounds, Cost-to-Go, H=3, or H=4 target is
used.

## Frozen identities

- Candidate pool: `stride-paretopool-v1`.
- Label schema: `lns2.stride.multihorizon_value.v1`.
- Value model: `stride-multivalue-v1`.
- Teachers: frozen `v2-full` and official Adaptive LNS2.
- Horizons: 1, 8, 32, and 128 repair decisions.
- `v2-full` retains the full base pool and remains the abstention anchor.

Each temporal state is identified by episode, decision index, state
fingerprint, and an outcome-blind history hash.  Equal fingerprints at distinct
decisions are distinct observations.  The history records recent neighborhoods,
exact/Jaccard repetition, conflict-edge appearance/disappearance/reappearance,
conflict-signature duration, per-agent repair counts, and PP success/no-op/state
change outcomes.

## ParetoPool audit boundary

The first audit uses all 90 frozen checkpoint occurrences (78 unique state
fingerprints) only to expose gaps relative to the old fixed-size candidate grid.
The old 16-seed one-step values cannot score novel ParetoPool candidates and are
not future-value labels.  Therefore an absent historical best is a diagnostic,
not a failed future-quality gate.  Budgets 6, 8, and 12 are compared again using
the registered multi-horizon rollouts; the smallest passing budget is frozen.

ParetoPool natural breakpoints come from family support, conflict-neighbor
closure, touched conflict-component closure, and path-contact closure.  At most
two history-conditioned candidates connect persistent or repeatedly
under-covered conflict agents.  Candidate and V2-anchor Jaccard limits remain
0.8 and 0.9.

## Rollout and censoring contract

Every candidate is forced exactly once, then independently continued by both
teachers.  Eight strictly paired PP seeds are collected first.  If the two
fixed four-seed halves have Top-3 overlap below 80%, every candidate in that
state is extended to 16 seeds.  Selective seed extension is forbidden.

Each rollout stops at feasibility, 128 repairs, or 300 seconds.  Repair-limit
and wall-time observations are right-censored evidence.  Unobserved full-horizon
regression targets are masked; observed prefixes and survival information are
retained.  Labels contain feasibility events, normalized conflict AUC, terminal
conflict ratio, original/new/migrating edges, component persistence/remerge,
and neighborhood repetition.  Runtime, TTF, and future outcomes are excluded
from online features.

## Stability and model gates

Every map group must independently satisfy: Top-3 half overlap at least 80%,
paired rank correlation at least 0.60, V2/Adaptive direction agreement at least
70%, and cross-seed normalized regret at most 0.02.  Failure increases paired
seeds or revises the target; it does not authorize a larger model.

MultiValue-GBDT is trained first with whole-map grouped folds and predicts
candidate-minus-anchor targets.  DeepSets or a conflict-graph GNN may proceed
only after label stability and only if held-out normalized regret improves by
at least 0.02 without degrading feasibility calibration or per-map behavior.

## Execution safety

Non-timing collection preflights normal worker counts 12/14/16 and heavy counts
8/10/12, selecting the highest measured throughput with zero errors, two logical
CPUs reserved, and peak RSS within 75% of available memory.  Every state attempt
is capped at 1800 seconds, allows four checkpointed attempts, and stops for
diagnosis after two attempts without candidate progress.  Formal raw TTF uses
isolated episodes or identical fixed low concurrency and includes every
controller overhead.

No training, runtime promotion, generalization claim, or TTF claim is permitted
from the candidate-gap audit or an Oracle/local metric.
