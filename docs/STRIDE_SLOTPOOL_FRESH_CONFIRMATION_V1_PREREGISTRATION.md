# STRIDE SlotPool v1 fresh-map confirmation preregistration

The frozen development model is evaluated without retraining on 24 initial
states from six map-disjoint MovingAI DAO maps: `arena`, `den020d`, `den404d`,
`hrt002d`, `lak109d`, and `lak515d`.

Each map contributes two tasks that were selected before this experiment by a
reset-only rule targeting initial conflict levels near 25 and 100, and both
solver seeds 1 and 2. All 24 registered states must complete. Candidate or
repair results cannot remove a state.

For each state, the complete exact-deduplicated StructPool grid at sizes 8,
16, 24, and 32 is generated. Every candidate receives 16 strictly paired PP
trials using the same current-step seed contract as development. The frozen
portable SlotPool model retains at most six candidates. It is not retrained,
recalibrated, or thresholded on confirmation labels.

The development gates are reused unchanged: 90% global-best retention, mean
regret at most 0.02, map and topology-group mean regret at most 0.05, each
8-seed-half best retention at least 85%, stable-pair accuracy at least 60%, and
the six-candidate budget. Every gate must pass before runtime work begins.

This is still a current-step candidate-quality test. It contains no Maze
long-tail state, GuardPool action, full repair trajectory, runtime field, or
TTF claim.
