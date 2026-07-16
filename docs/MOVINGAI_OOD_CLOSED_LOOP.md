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

## Formal Result

The study was preregistered and pushed at commit `606374a` before qualification or policy outcomes were
read. All 12 archives and extracted maps/scenarios matched their registered SHA256 values. Qualification
passed with 144/144 valid resets, 74 nonzero-conflict episodes, nine active maps, and at least one active
map in every registered layout family. All five policies then completed 144 episodes with zero timeout,
invalid action, initial-fingerprint mismatch, model error, or unexplained failure.

| Policy | Successes | Mean fixed 100-step AUC |
| --- | ---: | ---: |
| Adaptive | 123/144 | 65,981.05 |
| Fixed Target | 122/144 | 66,027.09 |
| Fixed Collision | 126/144 | 67,155.55 |
| Fixed Random | 114/144 | 68,541.21 |
| Frozen `realized_dynamic` v1 | 131/144 | 63,272.41 |

Frozen v1 improved aggregate fixed-budget AUC by `4.105%`, just below the preregistered `5%` gate.
The map-paired bootstrap interval was positive at `[2.15%, 66.86%]`; all 9/9 active maps and all 5/5
layout families were no worse than Adaptive. Family improvements were 49.4% on Game, 2.3% on Maze,
51.5% on Random, 64.9% on Room, and 93.3% on Warehouse. The initial-conflict-normalized sensitivity
improved by 32.4%, and success increased by eight episodes. Mean capped wall time fell from 88.81s to
62.18s, but its bootstrap interval crossed zero and wall time remained diagnostic only. About 19.6% of
selected v1 features were outside the development range.

The strict decision is `stop_cross_layout_claim_and_consolidate_results` because every gate must pass and
the primary aggregate AUC improvement missed 5%. The result is therefore strong, broad OOD evidence but
not a confirmed preregistered cross-layout claim. It does not restore the static-context migration claim
and does not authorize RL or another tuned model. The confirmed headline remains same-family, multi-seed
generalization; the MovingAI result is reported as a near-threshold external-layout result.

The generated traces occupy about 15.13 GiB because every transition stores full paths, observations,
and candidate features. Full trace validation took about 43 minutes in one Python process. Interrupting
the Codex turn did not stop that child process; it continued normally and wrote the final report. The
formal JSON SHA256 is `e931721f0cdc08df6eaf9de75843e0a58c86e19f009195e675f3b358c156b46e`.
