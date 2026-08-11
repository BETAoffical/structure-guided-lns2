# STRIDE-MarginalPool Action Replay v1 Report

Date: 2026-08-11

## Completion and integrity

The registered collection completed all 78 unique repair states, 2,502 frozen
candidates and 40,032 candidate/PP trials.  Every candidate has trial indices
0 through 15 under strictly paired PP seeds.  All restored states, native
actions, 124-dimensional features and repair fingerprints passed identity
checks.  There were no terminal timeouts or execution errors, and runtime,
TTF and future trajectories were not read by the label or analysis.

The operational recovery chain imported every valid completed or partial
artifact without result filtering.  Its last segment imported 77 complete
states and one partial state at 38/42 candidates, then completed the remaining
four candidates without a retry.

Key artifact SHA-256 values:

- `collection_report.json`: `f888b2ca5a3ee91839eb8daf92d96cd0bdb1e39bb51b41589a7eedb4266f79a0`
- `candidate_aggregates.jsonl`: `4d3ba4ac580901937e20c07c2de8309e35c395e4754347520edfc47950bd2b82`
- `logical_checkpoint_results.jsonl`: `efdbc8d7020628b3ac485d0946f9ad89b1b6d7afb547a29069d2009d41671973`
- `action_replay_analysis.json`: `8232e687db8009acee13ffbc523b625ee8c4fc5954fb3ffa10f4ce6567e02988`

## Root-cause result

The 78 physical states represent 90 logical checkpoints.  Under the frozen
stable-dominance rule, the actually selected neighborhood was:

- a ranker error in 66/90 checkpoints (73.33%): at least one already-generated
  alternative beat it by at least 0.02 mean normalized conflict reduction,
  did not increase no-progress probability, and beat it in both fixed seed
  halves;
- current-step best in 17/90 checkpoints (18.89%);
- too small-margin or seed-uncertain to classify in 7/90 checkpoints (7.78%);
- a pool/operator failure in 0/90 checkpoints.

The selected candidate's mean normalized regret was 0.15329, and a mean of
6.14 frozen alternatives stably dominated it per checkpoint.  PP uncertainty
was present in 75/90 checkpoints (83.33%), so these data support robust-set or
abstaining ranking, not a deterministic unique-winner claim.

The defect is strongest at the first repeated/stalled selection:

| Checkpoint | Ranker error | Selected best | Mean regret | PP uncertain |
|---|---:|---:|---:|---:|
| First structural selection | 24/45 | 15/45 | 0.09852 | 31/45 |
| First repeated stall | 42/45 | 2/45 | 0.20805 | 44/45 |

StructPool and SlotPool both show the same mechanism.  StructPool has 23/36
ranker errors and SlotPool has 43/54; neither has a pool/operator failure.

## Map breakdown

| Map | Checkpoints | Ranker error | Selected best | Uncertain | Pool error | Mean regret |
|---|---:|---:|---:|---:|---:|---:|
| `maze-128-128-1` | 32 | 21 | 9 | 2 | 0 | 0.14438 |
| `maze-128-128-2` | 24 | 18 | 4 | 2 | 0 | 0.11784 |
| `maze-32-32-4` | 34 | 27 | 4 | 3 | 0 | 0.18669 |

This is consistent across the three frozen Maze maps, but it is not evidence
of cross-family generalization.

## Interpretation and next step

The immediate long-tail defect is not a lack of alternatives: the existing
candidate pool contains multiple robustly better one-step actions at every
diagnosed checkpoint.  The actionable defect is that the V2 score ignores
repair history and repeatedly selects a locally attractive neighborhood after
that neighborhood has already produced exact path-level no-ops.

The next safe step is a preregistered history-aware ranking feature audit over
the complete frozen pool.  It must retain all 90 checkpoints, use map-grouped
validation, represent uncertain candidates as a set/abstention rather than a
unique winner, and compare the frozen 124D features against 124D plus only
decision-time history features.  No runtime controller or TTF experiment is
allowed until that audit demonstrates held-out-map ranking value.  Broader
Room, Warehouse and Game/DAO states are required before any generalization or
default-controller claim.

This report is checkpoint-local.  It does not show a TTF improvement, does not
validate long-term return, and does not authorize replacing `v2-full`.
