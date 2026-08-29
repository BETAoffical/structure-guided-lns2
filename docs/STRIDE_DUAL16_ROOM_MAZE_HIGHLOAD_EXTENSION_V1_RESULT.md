# Dual16 Room/Maze High-Load Extension V1

## Outcome

This sequential family diagnostic completed 24/24 reset-inclusive timed episodes
with an exact rotating serial schedule, matching initial fingerprints/conflicts,
registered clocks and no invalid actions or fingerprint mismatches. It is not a
map-disjoint confirmation because no unused local canonical Room/Maze map was
available.

Of 16 preregistered reset keys, 12 passed the 16 conflict-pair / 32 active-agent /
16 largest-component gate and `initial_conflicts >= 101`. All four
`maze-128-128-1` keys were excluded before controller execution because native
reset validation reported an empty agent path. They are state-supply failures,
not controller timeouts.

## Engineering acceleration assessment

Dual16 still has engineering headroom. Profiling the existing development cohort
put about 62% of TTF in native repair and about 28% before repair in controller
work; candidate generation and feature extraction dominate that controller
portion, while model inference is small. The safest next targets are a Dual16-only
Component16/Hotspot16 finalize path, copy-on-write candidate merging, and one-time
validation of the immutable controller specification. Any implementation must
preserve candidate order/provenance, feature matrices, scores, selected action,
RNG stream, repair trajectory and final path hash exactly.

| Existing cohort | Mean TTF | Native repair | Controller before repair |
|---|---:|---:|---:|
| Development (7) | 2.981 s | 1.851 s | 0.826 s |
| Held-out (4) | 2.836 s | 1.770 s | 0.755 s |
| Boundary (24) | 0.436 s | 0.118 s | 0.137 s |

These optimizations were deliberately not mixed into this extension. The frozen
Dual16 implementation was retained so the new map/load results measure cohort
effects rather than a simultaneous controller change.

| Cohort | Pairs | Dual16 wins | Official success | Dual16 success | Official mean capped TTF | Dual16 mean capped TTF | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Overall | 12 | 8 (66.7%) | 11/12 | 11/12 | 22.585 s | 21.001 s | Diagnostic advantage only |
| Room | 8 | 6 (75.0%) | 7/8 | 7/8 | 32.119 s | 29.117 s | Positive Room signal |
| Maze32 | 4 | 2 (50.0%) | 4/4 | 4/4 | 3.518 s | 4.770 s | Dual16 slower |
| Room64-16 | 4 | 3 (75.0%) | 3/4 | 3/4 | 33.518 s | 30.005 s | Positive, one shared censor |
| Room64-8 | 4 | 3 (75.0%) | 4/4 | 4/4 | 30.719 s | 28.229 s | Positive |

The Room gain came from lower native repair time despite controller overhead:
Dual16 averaged 2.067 s before repair and 17.053 s in repair, versus effectively
zero controller time and 21.393 s repair for Official. On Maze32 the roughly
1.008 s Dual16 controller cost was not repaid, so the simpler Official policy was
faster on average.

## Decision

Keep Dual16 as a targeted high-conflict Room/warehouse challenger. Do not route
generic Maze states to it and do not use this reused-map diagnostic for global or
default promotion. A fresh external canonical Room/Maze map set is still needed
for map-disjoint confirmation.

Evidence: `build/stride-dual16-room-maze-highload-extension-v1-run2/report.json`.
