# STRIDE HybridStructPool runtime optimization v2

## Purpose

This engineering milestone reduces HybridStructPool controller overhead without
claiming a new solver result.  The complete 3,432-candidate offline audit union
remains available through the original audit API.  The runtime API is a
separate, frozen lean view and is not a new 6/8/12 candidate-budget reducer.

The registered full-runtime trace showed that candidate generation, not model
inference, dominated controller cost: 17.494 of 19.415 mean controller seconds.
The optimization therefore targets repeated construction work directly.

## Frozen changes

- Preserve the complete V2 pool and V2 anchor.
- Preserve the complete retained CausalClosure Pareto frontier.
- Remove only twelve StructPool family-size cells that were available but never
  selected in the registered 152-episode full-runtime trace.  The complete
  four-size structural grid remains available to offline audits.
- Do not construct the path-overlap family at runtime because none of its four
  size cells was selected in that trace.
- Build CausalClosure occupancy and transitions only in the composed temporal
  window reachable from current conflict events, while retaining terminal-path
  extension semantics.
- Reuse localized closure and temporal-neighbor computations across causal
  family variants.
- Reuse already-computed V2 feature rows and compute new features only for
  structural and causal challengers.

The mask is outcome-informed engineering evidence from the registered runtime
trace, so it cannot establish result-blind generalization or promote a default
Pool.  It must pass the same paired 19-key, four-trial cohort before any later
result-blind Pool confirmation.

## Gates

The lean runtime must preserve or improve success, normalized fixed AUC,
restricted mean repair decisions, and platform entry relative to the registered
full runtime.  Mean controller and candidate-generation time must each fall by
at least 25%.  It must also keep every map's capped continuation wall at no more
than 1.05 times the paired V2 arm.  All PP, feature, inference, and generation
cost remains charged to the episode.

Passing these gates means only that the lean implementation is a viable Pool
engineering candidate.  It is not raw TTF evidence, non-Maze evidence, or
authorization to add rollback/rescue logic.
