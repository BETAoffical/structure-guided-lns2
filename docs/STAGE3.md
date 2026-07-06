# Stage 3: LNS Repair Experience

Stage 3 converts Train-only Trace V2 logs into reusable repair cases. It does
not perform retrieval, learning, or planner guidance.

## Build

```powershell
python scripts/build_repair_experience.py `
  --dataset build/feasibility-dataset `
  --collection build/experience `
  --split train `
  --output build/repair-experience
```

Outputs:

- `repair_cases.jsonl`: one record per attempted LNS neighborhood repair;
- `run_cases.jsonl`: one aggregate record per task/solver-seed run;
- `experience_summary.json`: coverage, labels, and heatmap conservation.

## Repair cases

Each case joins map structure, task pressure, first conflict events per Agent
pair, the selected neighborhood, and its candidate outcome. Selected Agents
include start/goal zones, flow assignment, static distance, and paths before
and after replanning.

Labels are:

- `conflict_reducing`;
- `cost_improving`;
- `neutral`;
- `rejected`;
- `invalid`.

Only the first two labels set `effective=true`. Rejected and invalid cases are
retained as negative experience.

## Heatmaps

Heatmaps use sparse `{cell, weight}` entries. Vertex conflicts contribute
`1.0` to one cell. Edge swaps contribute `0.5` to each endpoint, preserving a
total weight of one per conflict event.

Run records aggregate all pre-repair conflict heatmaps, neighborhood outcomes,
and per-Agent selection/effectiveness counts.

## Data boundary

The builder rejects any split other than `train`. Validation remains reserved
for Stage 4 retrieval tuning, and Test remains reserved for final evaluation.
