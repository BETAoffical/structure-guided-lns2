# STRIDE HybridStructPool budget v1 preregistration

## Question

Can the frozen full HybridStructPool challenger union be reduced to 6, 8 or 12
actions per state without discarding its registered structural and causal
membership?  The complete V2 pool remains outside the challenger budget.

This is a zero-solver membership audit.  It does not ask which candidate has
the best one-step PP result and does not predict platform entry.

## Frozen reducer

`balanced-semantic-diversity-v1` uses only action-preceding information:

1. round-robin over compact causal, topology boundary, conflict component,
   hotspot, bottleneck and path-overlap groups;
2. within each group, Pareto layers over current-state event coverage, internal
   conflict coverage, component reach and smaller size;
3. prefer the action with the largest minimum set distance from actions already
   selected;
4. then prefer coverage density, smaller size and candidate ID.

No PP outcome, future trajectory, fixed family-size preference, trained model
or post-hoc state selection enters this rule.  Every group receives one turn
before any group receives a second turn.

## Frozen membership gates

Each budget is evaluated independently on all 78 states.  A budget passes only
if it simultaneously retains:

- at least 90% of the 47 legacy robust-opportunity states;
- at least 80% of all 209 robust legacy structural actions;
- at least 80% opportunity-state recall on every map;
- at least 80% of 47 previously beneficial bounded-PreTail comparisons;
- the best compact causal action in all three CausalClosure-only opportunity
  states;
- 100% of the V2 pool, which is outside the challenger budget.

The smallest passing budget is selected.  If none passes, no rule or threshold
is tuned on these results: the full HybridStructPool remains the research
candidate contract and runtime compression remains unresolved.

Passing is not runtime promotion.  It does not establish platform avoidance,
selection quality, TTF improvement or replacement of `v2-full`.
