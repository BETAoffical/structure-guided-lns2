# STRIDE CausalTopoPool v1 preregistration

## Question

The completed structural-coverage diagnostic showed that compact
CausalClosurePool retained only 7 of 41 states where legacy StructPool had a
robust one-step opportunity.  The successor therefore asks whether causal
repair cores and natural topology closures can coexist without restoring the
old fixed 8/16/24/32 size preference.

This is a development-cohort generator audit.  It is not independent evidence.

## Frozen generator

The full V2 base pool and all 930 CausalClosure candidates are retained
unchanged.  Additional topology candidates are derived only from the current
state.  For each bottleneck, conflict-component, topology-boundary, hotspot and
path-overlap support set, the generator constructs these exact agent sets:

1. support endpoints;
2. support plus direct conflict neighbours;
3. support plus every member of a touched conflict component; and
4. support plus currently conflicting agents whose paths contact a support
   path cell.

These are relations, not requested sizes.  Equal sets merge provenance.  A
Pareto ordering compares actual size, incident-event coverage, internal-pair
coverage and conflict-component reach, followed by family coverage and a 0.9
Jaccard diversity rule.  At most 12 new topology candidates are retained.

There is no preferred-size grid.  An exact closure larger than 64 agents is
rejected as a whole.  It must never be cut to 64 or filled to any target size.
History and candidate outcomes are not inputs.

## Frozen integrity gates

- all 78 registered states materialize;
- all 1,367 V2 base and 930 CausalClosure candidates remain present;
- each state has 1--12 novel topology candidates;
- no topology candidate duplicates a preserved candidate exactly;
- no emitted closure is truncated and no emitted size exceeds 64;
- zero worker errors or timeouts; and
- source manifest and registered SHA-256 identities agree.

Passing authorizes only a separately preregistered, ranker-free, 16-seed paired
native-PP opportunity audit for the new topology candidates.  It does not
authorize ranker training, runtime integration, TTF testing or a long-tail
avoidance claim.
