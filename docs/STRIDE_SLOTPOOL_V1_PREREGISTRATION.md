# STRIDE SlotPool v1 preregistration

## Purpose

`stride-slotpool-v1` is an independent successor to the rejected
`stride-scalepool-v1` support-nearest rule. It tests whether a learned,
map-grouped selector can retain at most six high-quality `family x size`
candidate slots from the completed four-size StructPool grid.

This is a candidate-budget experiment. It is not a runtime controller, a
GuardPool experiment, a Maze long-tail experiment, or evidence of TTF gain.

## Frozen development evidence

- 98 active non-Maze states from 16 maps.
- 1,368 exact-deduplicated four-size candidates.
- 21,888 current-step repair trials: 16 strictly paired PP seeds per candidate.
- The known `maze-128-128-1` long-tail state is excluded.
- Fresh confirmation maps remain unread until the OOF gates pass.

All input paths and SHA-256 values are frozen in
`configs/stride_slotpool_v1_registration.json`.

## Labels and independence unit

Trials are aggregated by candidate and are never treated as independent
samples. A candidate pair is labeled only when trial indices 0-7 and 8-15
agree on the winner and both absolute mean gaps are at least 0.01. Mirrored
pair rows are used for fitting. Each state has total weight one; stable pairs
within a state are weighted by their mean absolute half-gap.

Only current-step normalized conflict reduction is used. TTF, runtime,
future trajectories, remaining repair rounds, Cost-to-Go, and Receding-Q are
forbidden.

## Features and model

Each candidate uses the existing 124 realized features plus 46 preregistered
slot features: family, size, family-size interactions, provenance, anchor
overlap, support statistics, and size/support ratios. Pair inputs contain 170
candidate differences plus 23 shared state features, for 193 dimensions.

The model is a pairwise `HistGradientBoostingClassifier`. Four whole-map outer
folds estimate development performance. Three whole-map inner folds choose
one of four frozen tree-complexity settings. Maps never cross a train/validation
boundary.

For each state, all candidate-pair win probabilities are averaged into a Borda
score. The six highest candidates are retained, with candidate ID as the
deterministic tie break. Multiple sizes from the same family are allowed.
No new Jaccard or V2-anchor filter is applied in v1 because those filters did
not cause the ScalePool v1 misses.

## Hard gates

All gates must pass:

- global best retention at least 90%;
- mean normalized regret at most 0.02;
- every map mean regret and topology-group mean regret at most 0.05;
- each fixed 8-seed half best retention at least 85%;
- stable-pair accuracy at least 60%;
- at most six selected candidates and fewer candidates than the full grid;
- complete map-disjoint OOF coverage and no forbidden fields.

Failure stops runtime integration. Passing permits only a separately
registered evaluation on fresh unseen maps; it still does not establish TTF
improvement.
