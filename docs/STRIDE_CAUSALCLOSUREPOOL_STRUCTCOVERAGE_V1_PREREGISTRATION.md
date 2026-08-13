# STRIDE CausalClosurePool structural-coverage diagnostic preregistration

## Question

CausalClosurePool v2 removed RepairClosurePool's systematic size inflation, but
it does not directly preserve the earlier StructPool topology actions.  This
audit asks whether that compact replacement lost one-step repair opportunities
which were already present in the frozen StructPool action set.

This is a retrospective development diagnostic.  Both legacy StructPool and
CausalClosure one-step outcomes already exist.  It is not an independent
confirmation and cannot support a runtime, TTF or long-tail claim.

## Frozen inputs and comparisons

The audit joins the same 78 state fingerprints across:

- all 1,367 frozen V2 base actions;
- all 1,135 legacy StructPool actions and their 16-seed aggregates;
- all 930 CausalClosure actions; and
- the completed CausalClosure opportunity report.

For every state, the best legacy StructPool action is compared with the best V2
base action using the existing stable-dominance rule: mean normalized conflict
reduction advantage at least `0.02`, no larger no-progress rate, and strict
improvement in both fixed eight-seed halves.

Every legacy structural action is then matched to the closest CausalClosure
agent set.  Coverage is reported at exact equality, Jaccard at least `0.8`, and
Jaccard at least `0.6`, with separate summaries by map, structural family and
legacy size `8/16/24/32`.  Family membership is multi-label.

## Frozen decision rule

CausalClosure is considered to preserve the old structural opportunity only if
all of the following hold:

- at least 90% of states where legacy StructPool robustly beats the best V2
  action also contain a robust CausalClosure opportunity;
- at least 90% of legacy actions that robustly beat the best V2 action have a
  CausalClosure action with Jaccard at least `0.8`; and
- each map with at least one legacy opportunity retains at least 80% of those
  opportunity states.

If any condition fails, the next permitted development step is an explicitly
preregistered hybrid-pool design.  It must preserve the complete V2 base pool
and may restore only outcome-blind topology cores; it may not copy successful
agent sets selected from this diagnostic.  A new pool still requires an
independent result-blind forced-continuation confirmation before any long-tail
or solver claim.

## Prohibited conclusions

The audit does not select a ranker, train a model, change runtime behavior,
control PP repair order, measure TTF, or establish long-tail avoidance.
