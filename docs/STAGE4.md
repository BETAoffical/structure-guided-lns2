# Stage 4: Offline Retrieval and Guidance

Stage 4 tests whether static warehouse/task descriptors and current conflict
context can retrieve useful Train experience. It does not alter LNS2 or use
Test data.

## Data boundary

The retrieval index accepts only Stage 3 records marked `split=train` and
`usage=memory`. Validation traces are converted separately with
`usage=evaluation`. Both builders reject Test.

The current Validation set contains 24 tasks. Running solver seeds 1, 2, and 3
produces 72 runs. Run records are aggregated by task, while repair attempts
remain iteration-level queries.

## Features

Run retrieval uses continuous map topology and task pressure fields. Repair
retrieval adds only information available before replanning: conflict counts,
types, locations, the seed pair's roles, and pre-repair path descriptors.

Human labels such as `layout_mode`, `scenario_type`, and `task_variant` are
used only for reports. IDs, solver seeds, outcomes, post-repair paths, and
runtime are excluded from vectors. Numeric values use Train means and standard
deviations; zero-variance fields are dropped. Categorical seed roles use
Train vocabularies with an unknown category.

## Retrieval

The implementation is deterministic, dependency-free brute-force kNN.
Distance is a weighted mean of per-group squared differences so larger feature
groups do not dominate solely through dimension count.

- Run retrieval combines nearby Train task heatmaps using inverse-distance
  weights.
- Repair retrieval predicts effectiveness and aggregates effective neighbor
  roles.
- Repair neighbors are limited to one case per run and two per task.
- The 95th percentile of Train leave-one-out nearest distances marks
  out-of-distribution queries.

Validation selects `k` from 3, 5, 7, and 11 and chooses group weights
deterministically. Output role templates contain zones, flows, distances, and
path overlap, never historical Agent IDs.

## Outputs

`build/stage4-index/` contains:

- `normalizer.json`;
- `feature_schema.json`;
- `run_index.jsonl`;
- `repair_index.jsonl`;
- `index_summary.json`.

`build/stage4-evaluation/` contains:

- `run_guidance.jsonl`;
- `repair_guidance.jsonl`;
- `evaluation_summary.json`.

Heatmaps report cosine similarity and Top-10 recall, including separate
metrics for queries that actually contain conflicts. Repair guidance reports
accuracy, precision, recall, F1, a majority baseline, role overlap, and
breakdowns by layout and task variant.

These are offline retrieval measurements. Concrete Agent selection and
controlled solver comparisons are Stage 5.
