# InitLNS independent-layout mechanism confirmation

## Purpose

This probe is the final bounded mechanism check before any larger contextual
collection. It does not train a supervised model or an RL policy. It asks two
separate questions:

1. Do repeated realizations of the same high-level action have a stable enough
   expected outcome to learn?
2. Do oracle action-family preferences change across independent layout replicas,
   static OD modes, or agent densities?

The six maps use a new master seed and are checked for generated map/task seed
overlap with Pilot v2. Each map receives four independently generated static tasks:
`balanced_80`, `balanced_100`, `bottleneck_80`, and `bottleneck_100`. The 80-agent
task is explicitly checked not to be a prefix of the corresponding 100-agent task.

## Registered design

- Layouts: two `regular_beltway`, two `compartmentalized`, and two
  `dead_end_aisles` maps.
- State acquisition: official Adaptive, solver seed `0`, at most one state per
  task.
- Source bounds: 1-200 initial conflicts and at most 100 agents.
- Candidate action: at most four conflict seed agents, each crossed with Target,
  Collision, Random, and sizes 4, 8, 16.
- Repetitions: eight independently seeded trials and Horizon 1 only.
- Maximum volume: `24 * 4 * 3 * 3 * 8 = 6,912` outcomes.
- Runtime: four process workers and a 300-second wall limit for each source
  episode.

Horizon-1 feasibility and remaining conflicts form the effectiveness Pareto label.
Generated nodes are reported only as a compute-aware sensitivity. Runtime is not a
scientific label.

## Staged commands

Generate the six maps and 24 tasks:

```powershell
python scripts/generate_dataset.py `
  --config configs/independent_layout_probe_dataset.json
```

Run qualification only:

```powershell
python scripts/collect_repair_experience.py `
  --dataset build/initlns-independent-layout-probe-v1 `
  --config configs/independent_layout_probe_collection.json `
  --output build/initlns-independent-layout-probe-v1-collection `
  --phase qualify --splits probe --workers 4

python scripts/analyze_independent_layout_probe.py `
  --dataset build/initlns-independent-layout-probe-v1 `
  --collection build/initlns-independent-layout-probe-v1-collection `
  --output build/initlns-independent-layout-probe-v1-report `
  --qualification-only
```

Qualification must have no errors, at least 18 repairable tasks, at least six per
layout, and both OD modes and densities represented among repair labels on every
map. If it passes, collect Adaptive baselines and inspect the counterfactual budget:

```powershell
python scripts/collect_repair_experience.py `
  --dataset build/initlns-independent-layout-probe-v1 `
  --config configs/independent_layout_probe_collection.json `
  --output build/initlns-independent-layout-probe-v1-collection `
  --phase baseline --splits probe --workers 4 --resume

python scripts/collect_repair_experience.py `
  --dataset build/initlns-independent-layout-probe-v1 `
  --config configs/independent_layout_probe_collection.json `
  --output build/initlns-independent-layout-probe-v1-collection `
  --phase counterfactual --splits probe --workers 4 --resume --dry-run
```

Only after the dry-run budget agrees with the eligible states should the same
counterfactual command be run without `--dry-run`. The collector's exclusive lock,
incremental manifest, status monitor, and episode timeout remain active.

The final report is generated with:

```powershell
python scripts/analyze_independent_layout_probe.py `
  --dataset build/initlns-independent-layout-probe-v1 `
  --collection build/initlns-independent-layout-probe-v1-collection `
  --output build/initlns-independent-layout-probe-v1-report
```

## Registered gates

- Mean statewise action eta-squared at least 0.5.
- Trial-split candidate-rank Spearman at least 0.5.
- Trial-split Pareto-family Jaccard at least 0.5.
- No fixed `(heuristic, size)` family is uniquely Pareto optimal in more than 80%
  of states.
- At least one of layout, OD, or density is significant at 0.05 after Holm
  correction.

Actual-neighborhood Jaccard is diagnostic rather than a veto. A value below 0.5
means the nominal high-level action describes a stochastic neighborhood-generating
distribution. In that case, the next method ranks realized candidate neighborhoods
after generation.

If stability and corrected context heterogeneity pass, the next dataset may expand
to 12 new Train and six new Validation maps. Stable actions without context
heterogeneity narrow the project to dynamic-state or realized-neighborhood control.
A second stability failure stops expansion and requires a redesigned action space.

## Completed result

The staged run completed with 24 valid qualifications, 23 repairable sources, 24
successful Adaptive baselines, 23 states, 6,480 outcomes, and zero collection or
replay errors. Repairable counts were 8/8 compartmentalized, 8/8 regular beltway,
and 7/8 dead-end aisle tasks. All six maps retained both OD modes and both densities.

The preregistered scientific decision is `stop_and_redefine_action_space`:

- Mean action eta-squared was 0.404, below 0.5.
- Trial-split rank Spearman was 0.638, above 0.5.
- Pareto-family Jaccard was 0.432, below 0.5.
- Maximum fixed-family unique-Pareto share was 26.1%, safely below 80%.
- Holm-adjusted layout, OD, and density p-values were all 1.0.
- Realized-neighborhood Jaccard was 0.428, below its routing threshold.

The independent maps therefore confirm partial rank repeatability but do not show
stable nominal-action Pareto sets or static-context heterogeneity. The paused 12/6
collection and RL remain stopped. The next admissible method study must redefine
the action surface, with realized-neighborhood candidate generation followed by
ranking as the registered route; it must not add more repetitions to this nominal
seed/rule/size action space.
