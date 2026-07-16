# Frozen V1 MovingAI OOD Closed-Loop Confirmation

This preregistered stage tests whether the frozen `realized_dynamic` InitLNS controller transfers,
without retraining, from the structured development maps to standard MovingAI layouts. It does not add
static map, OD, or density context; it does not train a repair-order selector or RL policy.

## Frozen Inputs

The tracked portable v1 bundle, development index, native model, feature schema, and feature ranges keep
their previously registered SHA256 values. Online traces report feature-range violations, but OOD labels
cannot change the model or action. Official Adaptive and fixed Target, Collision, and Random use the same
PP+SIPPS solver and differ only in the InitLNS destroy strategy.

## Dataset

Twelve official MovingAI maps are fixed before qualification: three Random, three Maze, two Room, two
Warehouse, and two game maps. Each uses random scenarios 4 and 5, two registered agent counts, and solver
seeds `[1,2,3]`, producing 48 tasks and 144 initial episodes. Archive and extracted-file SHA256 values are
verified before preparation. These map IDs do not overlap the six MovingAI mechanism-probe maps.

Qualification requires 144 valid resets, at least 72 nonzero-conflict episodes, at least 8 active maps,
and at least one active map in each of the five layout families. Zero-conflict tasks remain in end-to-end
success statistics but do not enter conditional policy comparisons. A failed qualification is reported as
insufficient evidence; tasks, scenarios, and densities are not replaced.

## Registered Gate

Frozen v1 must preserve Adaptive's total success count, improve fixed 100-step conflict AUC by at least
5%, have a non-negative lower bound from 5,000 map-paired bootstrap samples, be no worse on at least 8/12
maps and 4/5 layout families, and produce no invalid action, initial-fingerprint mismatch, model error, or
unexplained episode failure. Wall-clock time, generated nodes, controller overhead, and feature-range
violations are reported but do not control acceptance.

Passing establishes dynamic realized-neighborhood generalization across these standard layouts, not the
original static-context transfer claim. Failure ends the cross-layout claim and triggers result
consolidation rather than another model or RL run.

## Commands

```powershell
python scripts/fetch_movingai_devset.py --config configs/movingai_ood_devset.json `
  --output build/movingai-ood-dev
python scripts/prepare_movingai_probe.py --dataset build/movingai-ood-dev `
  --config configs/movingai_ood_dataset.json `
  --output build/initlns-movingai-ood-dataset-v1
```

Native collection runs in WSL with `PYTHONPATH=build/linux/project`:

```bash
python3 scripts/collect_closed_loop_confirmation.py \
  --dataset build/initlns-movingai-ood-dataset-v1 \
  --config configs/movingai_ood_collection.json \
  --output build/initlns-movingai-ood-collection-v1 --phase qualify
```

Policy execution is allowed only after qualification passes. Analyze the complete five-policy run with:

```powershell
python scripts/analyze_movingai_ood_confirmation.py `
  --collection build/initlns-movingai-ood-collection-v1 --strict
```
