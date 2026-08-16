# StructShell state-bounded rollback screen result

## Identity and scope

- implementation commit: `c51ad4b`
- run fingerprint: `0e72470c482b3b4d02cde3ca04d19d74f9c5d5d6f1255141dbf10e6a10c5797c`
- sealed report SHA-256: `a656a904e355e18fd0fa3bcba48042e2b963d6fc88a54b3644ea3124cc9b6c3e`
- cohort: the same ten registered seed-16 screen keys used by the previous
  rollback-aware TTF screen
- timing: reset-inclusive, 180-second right-censored TTF, strict serial
  execution with one timed worker
- scientific scope: seen-key mechanism and cost diagnostic; no promotion or
  runtime replacement claim

All 40 episodes completed.  There were zero execution errors, process
timeouts, invalid actions, or fingerprint mismatches.  All passive state-guard
trace gates passed.

## Overall results

| Controller | Success | Mean restricted TTF (s) | Mean normalized AUC | Mean repair decisions | Mean PP (s) | Mean selection (s) |
|---|---:|---:|---:|---:|---:|---:|
| Official Adaptive | 9/10 | 30.180 | 0.14914 | 109.0 | 16.449 | 0.006 |
| V2 | 9/10 | 31.858 | 0.14947 | 55.9 | 17.796 | 1.186 |
| StructShell without rollback guard | 8/10 | 47.878 | 0.15853 | 47.5 | 31.383 | 3.922 |
| State-bounded StructShell v3 | 9/10 | 32.716 | 0.15206 | 32.9 | 17.835 | 2.004 |

Relative to unguarded StructShell, v3 recovered one success and reduced mean
restricted TTF by 15.163 seconds (31.67%).  Relative to V2, however, it was
0.858 seconds (2.69%) slower and faster on only 2/10 paired keys.  Relative to
Official Adaptive it was 2.536 seconds (8.40%) slower and faster on 3/10 keys.
It therefore did not pass the registered V2 directional screen.

## Per-map restricted TTF

| Map group | Official | V2 | No guard | v3 | v3 minus V2 |
|---|---:|---:|---:|---:|---:|
| arena | 24.020 | 21.824 | 36.066 | 37.285 | +15.461 |
| den020d | 22.922 | 26.543 | 30.226 | 31.350 | +4.807 |
| den312d | 0.780 | 0.782 | 0.774 | 0.792 | +0.010 |
| maze-128-128-1 | 180.000 | 180.000 | 180.000 | 180.000 | 0.000 |
| maze-32-32-4 | 5.826 | 3.554 | 5.864 | 5.519 | +1.965 |
| random-32-32-20-high-load | 11.822 | 15.019 | 21.869 | 10.959 | -4.060 |
| random-64-64-10 | 0.736 | 0.708 | 0.770 | 0.745 | +0.037 |
| random-64-64-20 | 1.164 | 1.405 | 1.382 | 1.423 | +0.018 |
| room-64-64-16 | 36.148 | 50.393 | 180.000 | 35.186 | -15.207 |
| warehouse-10-20-10-2-1-congestion | 18.374 | 18.350 | 21.839 | 23.874 | +5.524 |

The state-bounded guard fixed the dominant Room failure: one suppression event
changed a 180-second unguarded failure into a 35.186-second success, faster
than both V2 and Official Adaptive.  It also improved the high-load Random
case.  It did not help Arena, Warehouse, or den020d, where the selected
StructShell repairs were valid but more expensive than V2.

## Mechanism result

Across v3 episodes, 31 pure-StructShell exact rollbacks were counted and five
repair fingerprints were suppressed.  The maximum count for any repair
fingerprint was exactly three.  After suppression, all 12 audited fallback
decisions regenerated and rescored a V2-only pool; seven further V2 exact
rollbacks did not reopen StructShell.  Guard computation itself totalled only
0.029 seconds across 329 decisions.

The remaining aggregate gap to V2 is no longer a repeated-candidate-loop
problem.  Mean PP time was almost identical (17.835 versus 17.796 seconds),
while mean selection time remained 0.817 seconds higher.  Per-map PP effects
cancelled: v3 saved 13.421 seconds of PP on Room and 1.635 seconds on the
high-load Random key, but added 13.402 seconds on Arena, 5.240 seconds on
Warehouse, and 4.500 seconds on den020d.

## Decision

Keep V2 as the default.  Do not promote v3 and do not extend this seen-key
screen.  The state-level rollback budget is retained as valid mechanism code,
but full-time StructShell activation is rejected.

The next candidate is a separately registered V2-first rescue: ordinary
states use V2-only; StructShell is exposed only after repeated exact V2
rollback at the same repair fingerprint, and remains bounded by the verified
state-level budget.  This targets the Room and high-load Random wins while
avoiding StructShell generation and expensive structural PP on ordinary
Arena, Warehouse, and DAO states.
