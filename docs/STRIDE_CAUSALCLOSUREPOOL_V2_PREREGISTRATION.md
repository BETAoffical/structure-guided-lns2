# STRIDE CausalClosurePool v2 preregistration

## Purpose

RepairClosurePool v1 admitted agents through whole-path cell overlap and usually
expanded to the 64-agent safety cap.  This development audit freezes a new
generator before inspecting its output.  It asks only whether localized causal
relations produce a materially more compact, valid candidate pool on the same
78 frozen states.

## Frozen generator

- Every frozen V2 anchor and current conflict component remains a possible core.
- Direct conflict closure is always completed first.
- Temporal closure may follow only same-cell reservations or reverse edges in
  the time windows around current conflict events.
- Separate exact-time, bottleneck-only and plus/minus-two-time families are
  retained as interpretable natural boundaries.
- Whole-path overlap alone cannot admit an agent.
- There is no preferred size grid and no scalar evidence weighting.
- A closure that exceeds 64 agents is rejected as a whole.  It is never cut at
  64 and misreported as a completed boundary.

## Frozen compactness gates

Relative to the complete v1 materialized cohort:

- all 78 states must materialize and every state must retain at least one novel
  candidate;
- mean candidate size must be at most 80% of the v1 mean;
- median candidate size must be at most 75% of the v1 median;
- at most 10% of generated core-family closures may be rejected as oversized;
- no agent may be admitted by whole-path overlap alone;
- there must be no execution or identity error.

These are engineering-readiness gates, not evidence of repair quality.  Passing
permits only a separately preregistered, ranker-free paired native-PP opportunity
audit.  This audit uses no candidate outcomes, ranker, controlled repair order,
future trajectory, runtime or TTF.

The initial version of this registration produced the retained r1 diagnostic.
Its r2 and r3 amendments were separately frozen after each preceding failure;
their exact contracts and parent report hashes are in the corresponding JSON
design files.  No failed threshold was relaxed.
