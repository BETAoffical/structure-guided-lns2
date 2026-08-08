# STRIDE StructPool Size Grid v1

This report records the outcome-blind candidate-grid milestone for Stage 2 of
the StructPool optimization plan. It is not a repair-quality, TTF, or solver
improvement result.

## Cohort and exclusions

- 98 topology-active development states from 16 DAO/Game maps were restored.
- The retained states contain 51 `official_adaptive` and 47 `v2-full` source
  states.
- The known `maze-128-128-1` long-tail regression was excluded before state
  selection and was not executed.
- The frozen V2 anchor was selected from the original V2 candidate pool only.
- No candidate repair trial, controller continuation, future trajectory,
  runtime result, or TTF field was read.

## Candidate grid

Each available structural family was expanded over nominal sizes 8, 16, 24,
and 32 before the six-candidate runtime reduction.

| Item | Count |
| --- | ---: |
| Complete states | 98 |
| Raw family-size drafts | 1,680 |
| Unique candidate agent sets | 1,368 |
| Exact duplicate drafts merged | 312 |
| Pure-family candidates | 1,157 |
| Mixed-family candidates | 211 |

Every nominal size has 420 provenance rows. Family provenance counts are 392
for conflict component, 392 for spatiotemporal hotspot, 392 for path overlap,
255 for topology boundary, and 204 for bottleneck crossing. Counts exceed the
number of unique candidates when exact agent sets merge provenance from more
than one family or size.

## Label workload

Exact state/candidate identity comparison with the retained RobustAction label
collection found 589 candidates whose complete 16-seed paired trials can be
reused. The remaining 779 candidates require new repair trials.

| Trial source | Candidates | Paired PP trials |
| --- | ---: | ---: |
| Reused retained labels | 589 | 9,424 |
| New Stage 2 labels | 779 | 12,464 |
| Total | 1,368 | 21,888 |

The next operation is therefore a resumable 12,464-trial repair collection.
All candidates within the same state and trial index use the same PP seed. Only
the current one-step outcome is retained.

## Integrity

- Grid manifest SHA-256:
  `de4248402ed009e458fe282a25b22ce3c39edea46650757bfad3490a32d20a27`
- State artifact tree SHA-256:
  `3a2e67a8f851891b12ee1fc8230f19352875f7e961f92d93991bca17bd3e0bb2`
- Run fingerprint:
  `d7488fea0e246144783be82d6f03bfe010d0fbec5cdd279653b6a2cf1261caeb`
- Completed states: 98/98; errors: 0; timeouts: 0.

The milestone only proves that the preregistered non-Maze four-size candidate
grid is complete and outcome-blind. Candidate quality is evaluated only after
the paired labels finish.
