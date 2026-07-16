# InitLNS Conflict-Graph Feature GBDT Audit

## Question

The graph representation audit showed that conflict edges helped a GNN relative
to an edge-free DeepSets model, although neither neural model beat the registered
`realized_dynamic` GBDT. This final development audit asks whether deterministic
candidate-level conflict-graph features can transfer that edge signal into the
stronger tabular learner.

No new solver data is collected. The audit reuses 442 policy-visited states,
7,914 explicit neighborhoods, and 31,656 aggregated repair-order trials. The 23
historical states remain training anchors only. Static map, OD, density and
post-repair outcome, runtime, or generated-node features are excluded. The
registered dynamic baseline still contains pre-action low-level search-history
counters.

## Registered Design

The challenger appends 51 non-duplicate features to the existing 139 dynamic
features:

- 23 structural features for induced components, edge density, cuts, one/two-hop
  coverage, conflict-graph articulation agents and bridges, core number, degree,
  and harmonic centrality.
- 28 temporal features for repeated conflict pairs, internal/boundary/incident
  event timing, early/middle/late mass, and vertex-versus-edge composition.

The learner is the unchanged pairwise GBDT: 100 iterations, 15 leaves, minimum
leaf size 20, learning rate 0.05, L2 0.1, and random seed 20260714. Evaluation
uses the same 12-map leave-one-map-out split, equal state weighting, labels, and
3-point/5-percent gates as the registered baseline.

## Result

The Train audit completed in about 100 seconds.

| Metric | Registered GBDT | Graph-augmented GBDT | Change |
| --- | ---: | ---: | ---: |
| Pareto top-1 | 49.65% | 47.92% | -1.736 points |
| Conflict regret | 0.2824 | 0.2873 | -1.713% relative |
| Pairwise accuracy | 67.84% | 67.91% | about +0.07 points |
| Maximum size share | 45.83% | 46.18% | no collapse |

The challenger was no worse on 7/12 maps, below the registered 8/12 gate. The
95% map-bootstrap interval was `[-4.90, 0.79]` percentage points for top-1 delta
and `[-8.11%, 3.04%]` for conflict-regret improvement. The intervals include
zero, so the result does not establish significant degradation, but the point
estimates and preregistered gates provide no evidence of improvement.

Train failed the improvement, bootstrap, and map-count gates. Validation
therefore remained sealed, no model was exported, and the decision is
`stop_supervised_representation_expansion`.

## Interpretation

The small pairwise-accuracy increase did not improve top-1 neighborhood choice.
This is consistent with the earlier objective-alignment finding: learning more
pairwise relations is not enough to improve the final selected action across
maps. Conflict edges contain local information, but the tested aggregate graph
statistics do not add robust cross-map ranking value beyond the existing GBDT.

The result does not justify RL pretraining or restore the static migration
claim. Further work should change the scientific question, such as candidate
generation or sequential repair credit, rather than continue tuning supervised
rankers on the same labels.

## Artifacts

The ignored build output is stored under
`build/initlns-graph-feature-gbdt-audit-v1`. Its compact index stores feature
names once in the manifest and ordered values per candidate, reducing the index
from about 109 MiB to 20 MiB without changing any selected candidate or metric.
