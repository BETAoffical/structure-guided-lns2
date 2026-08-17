# Maze32-N300 Four-Map Dual16 Plateau Quick Result

## Decision

The 200-second seed-25 rerun completed all 16 strict-serial episodes in
25 minutes 45.7 seconds with no execution, process-fuse, illegal-action or
identity error. Official Adaptive solved all four maps and had the lowest mean
restricted TTF. V2, Dual16 and Dual16 with the eight-step plateau fallback each
solved three of four maps. Neither structural controller is eligible for
promotion or default use.

The plateau fallback is a real but non-uniform mechanism intervention. It
reduced Random-high Dual16 TTF from 63.169 to 37.660 seconds and reduced the
Maze terminal conflict count from 55 to 36, but it increased Room TTF from
71.196 to 97.257 seconds. Its overall mean was consequently indistinguishable
from unguarded Dual16.

## Result

TTF is reset-inclusive and right-censored at 200 seconds. `F` is a valid
algorithm-budget censoring event, not a process failure.

| Map | Official Adaptive | V2 | Dual16 | Dual16 + plateau | Fastest |
|---|---:|---:|---:|---:|---|
| Maze32 N300 | **163.975** | 199.397 | 200.000 F (55 conflicts) | 200.000 F (36 conflicts) | Official |
| Random-high N400 | **21.435** | 200.000 F (1 conflict) | 63.169 | 37.660 | Official |
| Room64 N600 | 31.556 | **30.034** | 71.196 | 97.257 | V2 |
| Warehouse N600 | 15.468 | 14.391 | 11.971 | **11.610** | Plateau |
| **Mean restricted TTF** | **58.109** | 110.956 | 86.584 | 86.632 | Official |
| **Successes** | **4/4** | 3/4 | 3/4 | 3/4 | Official |

The Maze load increase from 200 to 300 agents created the intended hard case:
Official required 163.975 seconds and V2 required 199.397 seconds. Extending
the wall budget therefore recovered two real successes that the prior
90-second budget would have censored.

## Plateau mechanism

The fallback suppresses both structural challengers on the decision after
eight consecutive non-decreasing-conflict repairs, uses a freshly generated
complete V2 pool, and reopens Dual16 after a strict conflict decrease. It does
not retry a PP call or change the frozen Copeland selector.

| Map | Guard-active decisions | Triggers | Releases | Guard effect versus Dual16 |
|---|---:|---:|---:|---|
| Maze32 N300 | 1,359 | 21 | 20 | Same censoring; 19 fewer terminal conflicts |
| Random-high N400 | 541 | 12 | 11 | 25.509 seconds faster |
| Room64 N600 | 82 | 11 | 11 | 26.062 seconds slower |
| Warehouse N600 | 0 | 0 | 0 | 0.361 seconds faster; no guard intervention |

Across four maps, Dual16 averaged 86.584 seconds and guarded Dual16 averaged
86.632 seconds, both with three successes. The fixed threshold therefore does
not provide a stable cross-map improvement. The evidence supports the earlier
diagnosis that an expanded structural pool can starve useful V2 actions, but a
generic conflict-only no-progress switch is too coarse: it helps Random, does
not rescue Maze, and harms Room.

## Boundary

This is a single-seed exploratory runtime triage, not a formal speed claim.
Official Adaptive remains the strongest controller in this cohort. No
controller should be replaced, no plateau threshold should be tuned on these
four outcomes, and the incomplete 90-second `v1` run must not be combined with
this result. A future mechanism should estimate candidate repairability before
selection or abstain earlier, rather than permanently exposing both families
or relying only on an eight-step post-selection plateau count.

## Identity

- Source commit: `7517fca`.
- Output root:
  `build/stride-structshell-maze32-n300-fourmap-quick-v2`.
- Solver seed: `25`.
- Wall/process budgets: `200 / 300` seconds.
- Config SHA-256:
  `d22baae2a93e1339bf8daa6498e1922c6d66f21b9403a63f7963df962809a53e`.
- Run fingerprint:
  `7badb277bb2cddd89e2a8126a5d9af5f7dad3d5cad33dccf8f7066dc7fd0bf59`.
- Schedule SHA-256:
  `b60a659f65fa2a228bd55931ccd3810a3c77c89aff6102f67c63db5f57564261`.
- Report SHA-256:
  `a35a15680e4bb228e5682975b44e1aabe56669e6b49d9e5fcaa296cd0cc8c4e1`.
- Observed wall time: `1545.7` seconds (25 minutes 45.7 seconds).
