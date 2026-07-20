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

The first qualification-only generation used bottleneck crossing and shared-corridor ratios of 0.10.
Although 39/48 tasks were repairable, only 10/12 maps retained both OD modes and both densities. Before
any repair outcomes were collected, the single registered pressure correction raised both bottleneck
ratios to 0.25 and moved all regenerated artifacts to the `v1b` directory. Map seeds, task seeds, frozen
models, analysis thresholds, and all other task settings remain unchanged.

Qualification precedes all labels. All 48 resets must be valid, at least 36 must have 1-200 initial
conflicts, every layout must contribute at least 12 repairable tasks, and every map must retain both OD
modes and both densities. Map and task seeds are checked against Pilot v2 and the previous independent
probe.

## Frozen models and gate

`research/scripts/neighborhood/run_realized_ranking_confirmation.py --mode freeze` trains fixed all-development models from
the already audited 23 states and 412 candidates. The freeze manifest records the exact source-index,
source-report, feature, and pickle hashes before confirmation labels are loaded.

The primary comparison is frozen `realized_dynamic` against frozen `proposal_dynamic`. Passing requires
top-1 gain of at least five percentage points, conflict-regret reduction of at least 5%, no significant
map-bootstrap degradation, at least 8/12 maps no worse, improvement over uniform random and internal
conflict coverage, and no unsupported greater-than-80% neighborhood-size collapse. Static context,
generated nodes, and runtime are reported but do not control this gate.

## Commands

```powershell
python scripts/generate_dataset.py --config research/configs/neighborhood/realized_ranking_confirmation_dataset.json
python research/scripts/neighborhood/run_realized_ranking_confirmation.py --mode freeze `
  --output build/initlns-frozen-policy-v1-sklearn
python research/scripts/neighborhood/collect_realized_ranking_confirmation.py `
  --dataset build/initlns-realized-ranking-confirmation-v1b `
  --output build/initlns-realized-ranking-confirmation-v1b-collection `
  --phase qualify
```

After qualification passes, run `propose` and `evaluate` with `--resume`, then analyze with the frozen
model directory. Generated maps, traces, models, and reports remain under ignored `build/` paths.

## Interpretation

A pass permits a separate sequential closed-loop test on another fresh map set. A failure keeps RL
paused and routes the project back to candidate-pool design, realized-set representation, or PP-order
control. This confirmation alone is not a final end-to-end or OOD claim.

## Qualification result

Both registered qualification runs stopped before proposal or explicit-repair outcomes were collected.
The original 0.10-pressure dataset had 48/48 valid resets and 39 eligible tasks, but only 10/12 maps
retained paired OD and density coverage. The one permitted pressure correction retained 48/48 valid
resets and 39 eligible tasks and improved map coverage to 11/12, but still failed the 12/12 gate.

The remaining `cross_four_gate` map illustrates why a single global pressure parameter is inadequate:
its two 100-agent sources had 619 and 293 initial conflicts, above the registered maximum of 200, while
several more open-layout sources had zero conflicts. No confirmation candidate, repair outcome, or model
score was produced from either dataset. The frozen development models and the ranking hypothesis are
therefore still untested on independent labels.

The next data revision must keep the maps fixed and create a predeclared task-seed pool for every
`(map, OD, density)` cell. It may select the first task whose initial Adaptive reset has 1-200 conflicts,
but it must not inspect proposal neighborhoods or repair outcomes. This replaces repeated global-pressure
tuning with a deterministic, qualification-only inclusion rule.
