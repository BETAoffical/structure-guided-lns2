# STRIDE-LNS Stage 4 protocol

Stage 4 is a controlled retraining and feature-ablation experiment. Its purpose
is to determine whether the new quality/post-structure label improves one-step
neighborhood selection before any runtime claim is made.

## Frozen comparison

- `v2-full` is a read-only external anchor. It is evaluated but never retrained.
- `stride-control-v1` is trained on the same 600 states and features as the new
  model, using the old feasibility-then-conflict-reduction objective.
- `stride-quality-v1` is trained from `lns2.stride.quality_label.v2`. Runtime,
  Cost-to-Go, Receding-Q, and remaining repair rounds are excluded from the
  primary label.

All trainable models use the registered histogram gradient-boosting parameters.
There is no hyperparameter search.

## Result-blind split

The 600 Stage 3 states are assigned to five folds by whole map. The assignment
reads only selection metadata: map, layout family, source policy, agent band,
and decision stage. It does not open either Stage 3 label file. MovingAI
development layouts (`maze`, `random`, `room`, and `warehouse`) are treated as
one layout family. Every map and therefore every state appears in exactly one
validation fold.

The deterministic greedy allocation first balances the number of maps from each
layout family, then normalized state/policy/agent-band/decision-stage loads.
The generated report must record `label_outcomes_read: false` and pass all fold
coverage gates before training is allowed.

## Registered ablations

- `full`: all 124 frozen realized-dynamic features.
- `no_state_context`: remove `state.*` shared context.
- `no_proposal_context`: remove `proposal.*` provenance and seed context.
- `no_realized_context`: remove `realized.*` neighborhood measurements.
- `no_path_geometry`: remove `realized.path_*` geometry.

The ablations diagnose feature groups; they are not a hidden hyperparameter
search. If more than one gate-passing variant is within 0.005 mean normalized
quality regret of the best variant, the variant with fewer inputs is selected.

## Offline metrics and gates

Every model is evaluated out of fold on identical states. Primary metrics are
pairwise accuracy, exact-best selection rate, Top-3 hit rate, quality regret,
and state-range-normalized quality regret. Results are also grouped by fold,
source policy, agent band, decision stage, and layout family.

The full quality model must achieve pairwise accuracy of at least 0.5 and improve
normalized regret over the same-data control by at least 5% relative or 0.01
absolute. It may trail the frozen anchor by at most 0.01 normalized regret and
may trail either anchor by at most one percentage point on exact-best or Top-3
hit rate. It must beat the control in at least three of five folds. In any
registered subgroup with at least 20 states, normalized regret degradation may
not exceed 0.03.

Passing Stage 4 means only that the offline selector is eligible for Stage 5
shadow/quick evaluation. It does not establish solver speedup, fewer repair
rounds, or MovingAI/OOD generalization. Formal OOD data remains unavailable to
Stage 4.

## Command

```powershell
python scripts/run_stride_pipeline.py prepare-stage4 `
  --config configs/stride_stage4_training.json `
  --output build/stride-stage4-protocol-v1
```
