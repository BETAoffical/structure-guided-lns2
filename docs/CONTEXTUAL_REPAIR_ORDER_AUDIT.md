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

Run the index integrity check with:

```powershell
python scripts/run_contextual_repair_order_audit.py --phase index
```

Run the registered cross-validation with:

```powershell
python scripts/run_contextual_repair_order_audit.py --phase cross-validate
```
