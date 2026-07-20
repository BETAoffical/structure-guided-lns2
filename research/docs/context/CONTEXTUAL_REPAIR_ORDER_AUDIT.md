# Contextual Repair-Order Audit

This stage asks whether the PP repair order should become part of the InitLNS high-level action.
It uses only the 24 Train states and 144 explicit neighborhoods registered by the completed repair-order
probe. Validation, Test/OOD, and independent-confirmation labels remain sealed.

Each decision compares four deterministic orders: ascending agent ID, descending conflict degree,
descending delay, and descending path length. The input combines the existing `realized_dynamic`
description of the current conflict state and explicit neighborhood with pre-repair sequence features.
No repair outcome, runtime, generated nodes, post-repair path, static layout label, OD label, or density
label is an input. The target is normalized Horizon-4 conflict AUC.

The registered model is a single fixed-parameter `HistGradientBoostingRegressor`. Evaluation is
leave-one-map-out over all 12 Train maps. In every fold, the fixed-order baseline is chosen using only
the other 11 maps. Each state has total training weight one. The audit also includes 5,000 map-level
bootstrap samples and 500 task-level context permutations.

Stage 1 must pass every registered gate before any independent map is generated. Passing Stage 1 only
permits freezing the portable model and collecting 12 new maps with master seed `20270519`; it does not
permit RL. Failure stops the independent collection and retains the current fixed/dynamic neighborhood
policy without a learned repair-order selector.

## Formal Result

Stage 1 stopped at the Train-only gate. Across 144 leave-one-map-out decisions, the model improved
normalized H4 conflict AUC over the fold-selected fixed rule by only 1.16%, below the registered 5%
threshold. The map-paired bootstrap interval was `[-2.16%, 3.80%]`. The model was no worse on 9/12 maps
and did not collapse to one rule, but near-oracle coverage increased by only 1.39 percentage points,
and the real context result exceeded only 55.2% of 500 permutations. Feasibility fell from 66.7% to
59.7%. The decision is `stop_before_independent_confirmation`.

The first generated report computed its bootstrap by averaging per-decision relative ratios, which did
not match the aggregate primary metric. The corrected report reuses the unchanged LOMO predictions and
permutation distribution and bootstraps the aggregate paired map statistic. The correction did not
change any failed substantive gate or the stopping decision. Both reports remain in the ignored build
directory for auditability. No `20270519` maps or confirmation labels were generated.

Run the index integrity check with:

```powershell
python research/scripts/context/run_contextual_repair_order_audit.py --phase index
```

Run the registered cross-validation with:

```powershell
python research/scripts/context/run_contextual_repair_order_audit.py --phase cross-validate
```
