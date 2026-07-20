# InitLNS Realized-Neighborhood Ranking Audit

## Purpose

The realized-neighborhood stability probe established that a fixed agent set has a
more stable one-step repair signal than the nominal `(seed, rule, size)` action.
This audit asks the next bounded question: can a learned model rank those explicit
agent sets on maps excluded from training?

The audit reuses the 23 states, 412 candidate neighborhoods, and 3,296 PP-order
outcomes already collected. It does not collect new solver trajectories, use
Test/OOD labels, or train an RL policy. The effective independent sample remains 23
states nested within six maps; the 3,296 repair outcomes are not treated as
independent examples.

## Index and label boundary

`ranking_index.jsonl` contains one row per `(state, candidate_id)`. All eight
PP-order trials are aggregated before a label is constructed. The primary
effectiveness Pareto relation maximizes solved rate and minimizes mean remaining
conflicts. Horizon-1 conflict AUC is reported but not added as a duplicate objective
because it is determined by the fixed source conflict count and the remaining
conflicts. Generated nodes and runtime remain sensitivity metrics.

Three fixed feature profiles are compared:

- `proposal_dynamic`: dynamic conflict state, actual size, proposal-family support,
  and aggregate provenance-seed statistics;
- `realized_dynamic`: proposal features plus seed-independent conflict coverage,
  component coverage, selected-agent path/delay/conflict statistics, overlap,
  spatial extent, and local topology for the explicit set;
- `realized_context`: realized features plus map topology, static OD semantics,
  agent count, and density.

Post-repair paths, conflict reduction, feasibility outcomes, generated nodes, and
runtime are forbidden as model features. Candidate construction itself remains
outcome-blind.

## Evaluation

The learner is the fixed pairwise scikit-learn `HistGradientBoostingClassifier`
used by the earlier audits. Dominance pairs are created only within one state and
mirrored in both directions. Evaluation uses six leave-one-map-out folds, 2,000
map-level bootstrap samples, and 500 complete task-context permutations.

Reported baselines are the exact expectation of uniform random selection, a
deterministic maximum-internal-conflict-coverage heuristic, proposal-only ranking,
and the Pareto oracle. Models and held-out predictions are persisted per fold; no
all-data deployment model is fitted.

## Completed result

The explicit-neighborhood ranking gate passed:

- proposal-only Pareto top-1: 13.0%;
- realized-dynamic Pareto top-1: 43.5%, a gain of 30.4 percentage points;
- mean conflict regret: 0.612 to 0.272, a 55.6% relative reduction;
- realized pairwise accuracy: 70.9%;
- conflict regret was no worse on all six held-out maps;
- map-bootstrap intervals for hit gain and conflict improvement were entirely
  positive;
- maximum selected-size share was 65.2%, below the 80% collapse threshold;
- realized ranking also beat uniform random and the fixed coverage heuristic.

Static context did not pass its separate transfer gate. It raised top-1 from 43.5%
to 47.8%, only 4.35 percentage points against a five-point requirement. Its real
context result reached the 92.0 percentile for top-1 gain and the 74.2 percentile
for conflict-regret reduction, both short of the required 95 percentile.

The registered decision is therefore
`advance_dynamic_realized_ranking_and_shrink_static_transfer_claim`. The evidence
supports learning to rank concrete candidate agent sets from the current dynamic
state and local set structure. It does not establish incremental value from the
current handcrafted map/OD/density context, and it is not final cross-distribution
evidence because only six maps were available.

The primary model optimizes effectiveness only. Its generated-node regret is not
better than the simple baselines, so this audit does not claim a compute-efficiency
improvement. Compute-aware training and hardware-independent cost control remain
questions for the independent confirmation stage.

The context-permutation implementation caches all 529 possible
`(state, donor-task-context)` evaluations. This reduced the formal run from about
316 seconds to 41 seconds without changing any model choice, metric, gate, or
decision.

## Reproduce

```powershell
python research/scripts/neighborhood/run_realized_neighborhood_ranking_audit.py `
  --collection build/realized-neighborhood-stability-probe-v1 `
  --output build/initlns-realized-neighborhood-ranking-audit-v1
```

Use `--strict` only when a failed scientific gate should produce exit code 2. A
failed gate is still a valid audit result and still writes the index, predictions,
models, and reports.

## Next gate

RL and the old static-transfer expansion remain paused. The next admissible stage
is an independent-map confirmation of dynamic plus realized-neighborhood ranking.
It must also expose a proposal-only candidate-generation path so online evaluation
does not pay for discarded repairs. Only a successful independent confirmation can
justify closed-loop policy evaluation or supervised-to-RL expansion.
