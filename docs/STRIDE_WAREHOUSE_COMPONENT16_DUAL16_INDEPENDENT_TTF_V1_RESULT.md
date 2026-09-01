# Component16 vs Dual16 Independent Warehouse TTF V1

## Outcome

The Component16-only augmentation is a promising simplification for the
targeted very-high-load, high-conflict station-centric warehouse route, but it
did not clear the preregistered diagnostic margin against the current Dual16
runtime.

All 48 strict-serial episodes completed on 16 authenticated checkpoints from
8 maps.  Every controller solved every checkpoint, with no timeout, stall,
invalid action, fingerprint mismatch, or initial-state mismatch.

| Controller | Success | Mean capped TTF | Median capped TTF | Mean repair iterations |
| --- | ---: | ---: | ---: | ---: |
| Official Adaptive | 16/16 | 0.897245 s | 0.850247 s | 37.7500 |
| Dual16 | 16/16 | 0.496589 s | 0.486203 s | 6.9375 |
| Component16 | 16/16 | **0.470189 s** | **0.455044 s** | **6.4375** |

Pairwise results:

- Component16 vs Official: 16/16 wins, 47.60% lower mean capped TTF,
  84.74% higher successes per observed hour; diagnostic gate passed.
- Dual16 vs Official: 15/16 wins, 44.65% lower mean capped TTF,
  74.98% higher successes per observed hour; diagnostic gate passed.
- Component16 vs Dual16: 9/16 wins, 5.32% lower mean capped TTF,
  5.58% higher successes per observed hour; diagnostic gate failed because
  the registered 70% win-rate, 15% mean-improvement, and 20% throughput
  thresholds were not met.

Component16 had a lower map-level mean on 5 of 8 maps.  Dual16 remained better
on 3 maps, including two material reversals, so removing Hotspot16 is not a
uniform improvement.

## Interpretation

Component16 keeps the complete frozen V2 pool, generates the same C16/H16
source cells as Dual16, admits only the Component16 row, and then uses the same
frozen V2 scorer.  It does not force a Component action and does not change PP
repair.

The mean 0.0264 s advantage over Dual16 came mainly from trajectory quality:
Component16 used 0.5 fewer repairs per episode and 0.0196 s less repair time on
average.  Controller time was also 0.0085 s lower, largely because the shorter
trajectories required fewer decisions; both routes still generated the shared
C16/H16 source cells.

This qualifies the H1 one-step result.  Component beat Hotspot in all 36 H1
one-step labels, but removing Hotspot from an end-to-end trajectory was only a
small aggregate improvement and sometimes hurt.  One-step repair labels alone
therefore do not justify replacing Dual16 or training the rejected two-head
hierarchy.

## Evidence boundary

The 8 source maps, 8 tasks, and 16 checkpoint identities are byte-disjoint from
the 6-map/36-checkpoint H1 label cohort.  However, this cohort was used in the
earlier Dual16 confirmation, so this is a post-hoc H1-map-disjoint development
test, not a new prospective confirmation and not a global/default promotion.
The runner rebuilt qualification identity under the current native-v3 module,
then rebound the authenticated checkpoint rows locally; no historical timed
lane was reused as a result.

Evidence:

- Report: `build/stride-warehouse-component16-dual16-independent-ttf-v1/ttf/independent_ttf_report.json`
- Report SHA-256: `9b44687bda49eed49ab7b994652bb706657f25f68c0d665484d3a2d49057f9d5`
- Timing boundary: checkpoint-restore-inclusive, reset-inclusive capped wall TTF
- Schedule: key-mod-3 rotating, one timed worker, 48/48 completed

## Decision

Keep the current Dual16 route unchanged.  Component16 is worth one fresh-map
confirmation focused on high and very-high station-centric warehouse loads,
but the present 5.32% advantage is too small and inconsistent to replace
Dual16 or justify a new trained hierarchy.
