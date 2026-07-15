# InitLNS independent realized-neighborhood ranking confirmation

## Purpose

The six-map development audit found that dynamic state plus the concrete agent set ranked one-step
InitLNS repairs substantially better than proposal provenance alone. This stage tests that result on
entirely new maps. It does not train RL, retune the GBDT, or restore the static map/OD/density transfer
claim.

The deployment gap is handled first: `LNS2RepairEnv.propose()` generates Target/Collision/Random agent
sets without executing discarded PP/SIPPS repairs. Representative neighborhoods are still selected
without looking at outcomes, then evaluated as fixed explicit sets under eight independent PP-order
seeds.

## Registered design

- 12 new maps: four each from regular beltway, compartmentalized, and dead-end aisles;
- four complete tasks per map: balanced/bottleneck crossed with 80/100 agents;
- solver seed 0 and the first InitLNS decision state only;
- up to four conflict seeds, three generators, sizes 4/8/16, and eight proposal seeds;
- at most two representative neighborhoods per generator/size family;
- eight explicit-repair trials per candidate and Horizon 1 effectiveness labels.

Qualification precedes all labels. All 48 resets must be valid, at least 36 must have 1-200 initial
conflicts, every layout must contribute at least 12 repairable tasks, and every map must retain both OD
modes and both densities. Map and task seeds are checked against Pilot v2 and the previous independent
probe.

## Frozen models and gate

`scripts/run_realized_ranking_confirmation.py --mode freeze` trains fixed all-development models from
the already audited 23 states and 412 candidates. The freeze manifest records the exact source-index,
source-report, feature, and pickle hashes before confirmation labels are loaded.

The primary comparison is frozen `realized_dynamic` against frozen `proposal_dynamic`. Passing requires
top-1 gain of at least five percentage points, conflict-regret reduction of at least 5%, no significant
map-bootstrap degradation, at least 8/12 maps no worse, improvement over uniform random and internal
conflict coverage, and no unsupported greater-than-80% neighborhood-size collapse. Static context,
generated nodes, and runtime are reported but do not control this gate.

## Commands

```powershell
python scripts/generate_dataset.py configs/realized_ranking_confirmation_dataset.json
python scripts/run_realized_ranking_confirmation.py --mode freeze `
  --output build/initlns-realized-ranking-confirmation-v1-frozen-models
python scripts/collect_realized_ranking_confirmation.py `
  --dataset build/initlns-realized-ranking-confirmation-v1 `
  --output build/initlns-realized-ranking-confirmation-v1-collection `
  --phase qualify
```

After qualification passes, run `propose` and `evaluate` with `--resume`, then analyze with the frozen
model directory. Generated maps, traces, models, and reports remain under ignored `build/` paths.

## Interpretation

A pass permits a separate sequential closed-loop test on another fresh map set. A failure keeps RL
paused and routes the project back to candidate-pool design, realized-set representation, or PP-order
control. This confirmation alone is not a final end-to-end or OOD claim.
