# InitLNS Graph Representation Audit

## Purpose

This development audit tests whether an agent-set encoder or a conflict-graph
encoder can rank explicit InitLNS repair neighborhoods more reliably than the
registered `realized_dynamic` GBDT. It reuses the existing policy-visited
counterfactual outcomes. It does not collect solver data, use static map or OD
context, train RL, or establish independent cross-map confirmation.

The registered data contains 23 historical anchor states, 288 Train states on
12 maps, and 154 sealed Validation states on 6 maps. Each candidate remains one
explicit agent set after its repair-order trials have been aggregated.

## Models

- `flat_mlp` consumes the same 139 aggregate features as the GBDT.
- `agent_deepsets` consumes per-agent path and conflict features, candidate
  membership, and dynamic proposal metadata without conflict edges.
- `conflict_gnn` adds two mean-aggregation message-passing layers over the
  current binary agent conflict graph.

All neural models use the same state-equal pairwise dominance objective, three
fixed random seeds, and nested map-level early stopping. The outer evaluation is
12-fold leave-one-map-out. Validation is read only if a Train model passes all
preregistered gates relative to the GBDT.

## Registered Result

The original formal run completed 12 folds and 108 model fits in 2,111 seconds.

| Model | Pareto top-1 | Conflict regret | Pairwise accuracy |
| --- | ---: | ---: | ---: |
| Registered GBDT | 49.65% | 0.2824 | 0.6784 |
| Flat MLP | 42.01% | 0.3514 | 0.6575 |
| Agent DeepSets | 26.39% | 0.4916 | 0.6331 |
| Conflict GNN | 42.01% | 0.3923 | 0.6598 |

No neural model passed against the registered GBDT, so Validation remained
sealed and the decision was `stop_supervised_representation_expansion`.

The conflict GNN did pass the registered incremental comparison against
DeepSets: top-1 increased by 15.625 percentage points, conflict regret improved
by about 20.2%, and 9/12 maps were no worse. This supports the narrower claim
that conflict edges contain useful information for an agent-set model. It does
not show that the GNN is an effective replacement for the GBDT.

## Artifact Policy

The hardened v2 runner writes atomic progress and completion records and binds
reports to the implementation, configuration, and input hashes. Fold models are
not retained because a raw state dict without its fold normalization and schema
cannot reproduce inference. Candidate predictions and fold diagnostics are the
reproducibility artifacts. A full model is retained only after both Train and
Validation gates pass, together with normalization, feature names, inputs, and
training configuration.

The next registered experiment uses deterministic structural and temporal
conflict-graph features with the existing GBDT. It is a final representation
bridge, not another neural-capacity search.
