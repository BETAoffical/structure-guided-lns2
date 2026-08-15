# STRIDE HybridStructPool runtime optimization v5 protocol

V5 repeats the registered 19-key, 152-episode, 16-worker pool-only experiment.
It retains the complete V2 pool, all 24 structural family/size cells, the full
retained CausalClosure frontier, native PP order, and the v4 terminal-wait and
integer evidence optimizations.

The only additional runtime changes are:

- extract the four frozen CausalClosure Pareto comparison coordinates once per
  raw candidate;
- use direct weak-and-strict coordinate comparisons in the same deterministic
  O(n^2) non-dominated sorting procedure;
- avoid a redundant final deep copy of merged candidates after every source
  row has already been deep-copied.

Exact ranks, complete `HybridStructPoolResult`, input isolation, candidate
identity, and evidence fields must remain unchanged.  V5 keeps the v4 gates
unchanged: relative to the registered v3 baseline, controller,
candidate-generation, and Hybrid-total time must each improve by at least 10%,
with all paired quality and integrity gates passing.  This remains an
engineering-only experiment and does not authorize runtime-pool promotion,
raw-TTF claims, training, rescue integration, or non-Maze generalization.
