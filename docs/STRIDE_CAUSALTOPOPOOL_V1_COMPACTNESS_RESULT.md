# STRIDE CausalTopoPool v1 compactness result

## Integrity

The preregistered 16-worker materialization completed all 78 states without an
error or timeout.  It preserved all 1,367 V2 base candidates and all 930
CausalClosure candidates, then added 366 exact natural-topology candidates.
Every frozen gate passed.  The cohort SHA-256 is
`6a4accad75cee257cf5408adcf1c2b61942bded61516264637854fd4b176bdfa` and
the compactness report SHA-256 is
`7be3b406585b633410e03e068e7c3367cf9e207f7021c706458fe02fd3b54b56`.

## Candidate count and size

The topology addition contains 1--10 candidates per state (mean 4.69), below
the frozen budget of 12.  Its actual neighborhood sizes are:

- minimum 2, median 18.5 and mean 22.61;
- P90 47, P95 56 and maximum 64;
- 266/366 at most 32 agents;
- 100/366 above 32, 31/366 above 48, and 3/366 exactly 64.

These values are not target sizes.  They are the cardinalities of exact
support, direct-conflict, touched-component or active-path-contact relations.
The generator rejected 433 exact closures larger than 64 and emitted zero
truncated closures.  Consequently the old problem of silently cutting a large
relation to the cap is removed, while medium topology closures needed to test
the lost size-24/32 opportunity remain available.

## Composition

Exact-set merging allows a candidate to carry multiple labels.  The 366
candidates cover bottleneck crossing (129 memberships), conflict component
(96), topology boundary (194), hotspot (183) and path overlap (45).  Closure
memberships are support (215), direct conflict-neighbour (214), touched
component (142) and active path-contact (54).  Fifty-seven sets already present
in V2 or CausalClosure were removed as exact duplicates.

## Decision

The hybrid pool is structurally valid and no longer depends on the fixed
8/16/24/32 preference.  Compactness alone does not establish that its larger
natural closures are beneficial or that they prevent long tails.  The only
authorized next step is the frozen ranker-free, strictly paired 16-seed native
PP opportunity audit for the 366 new candidates.  No ranker training, runtime
integration, TTF test or long-tail claim is authorized by this result.
