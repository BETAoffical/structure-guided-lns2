# STRIDE ScalePool v1 Offline Decision

## Decision

`stride-scalepool-v1` failed its preregistered offline quality gates and is not
eligible for runtime integration.  GuardPool, the known Maze long-tail rerun,
anchor-gate training, and TTF evaluation must not use this candidate generator.

The only passing check was reduction of raw candidate generation.  The failure
is scientific rather than a program or data-integrity error.

## Acceptance results

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Global-best retention | >= 0.90 | 0.1020 | Fail |
| Mean normalized regret | <= 0.02 | 0.1964 | Fail |
| Maximum map/group mean regret | <= 0.05 | 0.4294 / 0.3018 | Fail |
| First-half best retention | >= 0.85 | 0.1429 | Fail |
| Second-half best retention | >= 0.85 | 0.1020 | Fail |
| Raw candidate count reduced | Yes | 1,680 to 423 | Pass |

The known Maze long-tail state was excluded and no TTF or future-trajectory field
was read.  The evaluation was bound to the independently audited Stage 2 labels.

## Failure attribution

ScalePool retained the global best on 10 of 98 states.  All 88 misses were
classified as `support_nearest_size_mismatch`:

- 0 misses were caused by the StructPool-to-StructPool Jaccard limit.
- 0 misses were caused by selecting the same nominal size but a different agent
  set.
- 0 misses were caused by the V2-anchor Jaccard filter.

The core mismatch is that support count measures the small structural core, not
the number of agents that should participate in repair.  For example:

- Hotspot selected size 8 in 72 states, while global winners were predominantly
  size 24 or 32.
- Articulation boundary selected size 8 in 42 states, while global winners often
  used 16, 24, or 32.
- Bottleneck selected size 8 in 43 states, while its global-best provenance used
  only size 24 or 32.

The result is not repaired by globally preferring a larger size:

| Uniform size pool | Global-best retention | Mean normalized regret |
|---:|---:|---:|
| 8 | 0.0714 | 0.2437 |
| 16 | 0.2143 | 0.1104 |
| 24 | 0.2653 | 0.0808 |
| 32 | 0.4490 | 0.0549 |

Even the best uniform alternative, size 32, still fails both retention and regret
gates.  The frozen fixed-size StructPool policy (`0.07464` mean regret) is also
better than ScalePool v1 (`0.19638`), so ScalePool v1 is a regression in candidate
quality despite generating fewer drafts.

## Next research boundary

ScalePool v1 remains dormant and must not be retuned on these same 98 states and
then reported as independently validated.  A successor requires a new name and
preregistration.  Reasonable hypotheses for that successor are:

1. Predict family size from state, support, conflict-spread, and coverage features
   with whole-map grouped nested validation.
2. Allocate the six-candidate budget across family-size pairs instead of forcing
   exactly one size per family; unstable families may need two sizes, while weak
   families may receive none.
3. Evaluate the revised rule on maps not used to choose it before any GuardPool
   or TTF experiment.

No runtime speedup, long-tail improvement, or generalization claim is made from
this failed offline evaluation.

## Artifact hashes

- State evaluation: `9f293f7aee6a49ed595416d719780b719e0079ae9aad80a1cf2ef46896d26a70`
- Candidate attempts: `149df01de76631ee2d4051b2060ed24101948ae168b260e4d45ac04ac008f9b5`
- Support-size alignment: `586e61539c65fda0aa1ad93f6c77e0baf0a25bb8e3a086312add6af8e285a959`
