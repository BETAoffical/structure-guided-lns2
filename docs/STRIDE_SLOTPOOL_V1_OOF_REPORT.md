# STRIDE SlotPool v1 development OOF report

## Result

`stride-slotpool-v1` passed every preregistered development-only candidate
quality gate. The result permits a separately registered fresh-map
confirmation, but it does not permit runtime integration or a TTF claim.

The evaluation used 4 outer whole-map folds and 3 inner whole-map folds on 98
states from 16 maps. The 1,368 four-size candidates produced 8,475 stable
candidate pairs; all 98 states contributed stable pairs. No known Maze
long-tail state, future trajectory, runtime field, or fresh-map label was read.

## Metrics

| Metric | Result | Gate |
|---|---:|---:|
| Global best retained in six slots | 0.9796 | >= 0.90 |
| Mean normalized regret | 0.00189 | <= 0.02 |
| Maximum map mean regret | 0.01927 | <= 0.05 |
| Maximum topology-group mean regret | 0.00395 | <= 0.05 |
| First 8-seed-half best retained | 0.9796 | >= 0.85 |
| Second 8-seed-half best retained | 0.9286 | >= 0.85 |
| Stable-pair accuracy | 0.9559 | >= 0.60 |
| Candidate count | 1,368 -> 588 | at most six/state |

Only two maps had nonzero mean regret: `lak203d` (0.01927) and
`lt_hangedman` (0.00582). The other 14 maps had zero mean regret.

## Frozen full-development model

The final parameter setting was selected by grouped CV over all development
maps and then fitted to all development stable pairs:

- `max_leaf_nodes=7`;
- `l2_regularization=1.0`;
- `learning_rate=0.05`;
- `max_iter=100`;
- `min_samples_leaf=20`;
- `early_stopping=false`;
- `random_state=20260809`.

The portable tree model matched scikit-learn on the frozen parity sample with
maximum absolute probability difference no greater than `1e-12`.

Key hashes:

- state evaluation: `364119bba83a634958c7b2d03f4504caf08863bd136afde8628bb7ef05d58d29`;
- fold diagnostics: `fb8340bd242c5c0e8a1b7f78ff08c2f2aff2fc24983535d5c7c9b7428f0ce3e2`;
- stable-pair summary: `2cc4086e8f6efaa9849ffd9be9d0a6e4b8cd81a818a8781ca72bd0c3073df0dd`;
- portable model: `e0a42c7a21cb341a6c5ce13ad6d89e6f5279c04378535fb9c44ce76a1dfe2319`.

## Interpretation

The result rejects the failed ScalePool assumption that each family should be
compressed to the size nearest its support count. Allowing several sizes from
the same family and ranking all family-size candidates retains much more of
the observed one-step headroom.

This remains development evidence. The next step is a frozen-model test on
map-disjoint states. The model may not be retrained or recalibrated using that
confirmation set. Only a fresh-map pass can unlock runtime, GuardPool, Maze
regression, and raw-TTF work.
