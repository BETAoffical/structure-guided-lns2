# StructShell Four-Map Five-Arm Seed-24 Rerun

## Decision

The preregistered seed-24 rerun completed all 20 strict-serial episodes in
9 minutes 29 seconds with no execution or identity errors. It does not
replicate the seed-23 Dual16 win. Official Adaptive solved all four keys and
had the lowest mean restricted TTF; Dual16 failed Room and had the highest
five-arm mean. Dual16 therefore remains an implementation-valid diagnostic
arm, but is not eligible for promotion or default use.

## Seed-24 result

TTF is reset-inclusive and right-censored at 60 seconds. `F` denotes a valid
60-second censoring event, not a process failure.

| Map | Official LNS2 | V2 | Component16 | Hotspot16 | Dual16 | Fastest |
|---|---:|---:|---:|---:|---:|---|
| Maze32 | **1.846** | 6.779 | 6.945 | 4.285 | 11.782 | Official |
| Random-high | **7.237** | 11.602 | 9.554 | 9.274 | 11.989 | Official |
| Room64 | **44.879** | 58.525 | 60.000 F | 60.000 F | 60.000 F | Official |
| Warehouse | 24.098 | 20.110 | **13.326** | 15.375 | 16.998 | Component16 |
| **Mean restricted TTF** | **19.515** | 24.254 | 22.456 | 22.234 | 25.192 | Official |
| **Successes** | **4/4** | **4/4** | 3/4 | 3/4 | 3/4 | Official/V2 |

## Reproduction check

| Arm | Seed 23 mean / success | Seed 24 mean / success | Two-seed mean / success |
|---|---:|---:|---:|
| Official LNS2 | 19.221 / 4/4 | **19.515 / 4/4** | **19.368 / 8/8** |
| V2 | 21.893 / 3/4 | 24.254 / 4/4 | 23.073 / 7/8 |
| Component16 | 20.717 / 4/4 | 22.456 / 3/4 | 21.587 / 7/8 |
| Hotspot16 | 19.326 / 4/4 | 22.234 / 3/4 | 20.780 / 7/8 |
| Dual16 | **18.270 / 4/4** | 25.192 / 3/4 | 21.731 / 7/8 |

The reversal is concentrated in Room. On seed 23, all three structural arms
solved Room while V2 censored, and Component16 was fastest at 36.860 seconds.
On seed 24, Official and V2 solved Room, while Component16, Hotspot16 and
Dual16 all censored. This makes the first Dual16 overall win seed-sensitive.

Warehouse retains a useful structural signal across both seeds: a structural
arm was fastest on both runs. However, the winning construction changed from
Dual16 on seed 23 to Component16 on seed 24. Maze and Random-high continued to
favor Official LNS2. Generating both challengers is therefore technically
sound, but keeping both visible on every repair state is not robust.

## Boundary and next action

This is an exploratory two-seed comparison, not a formal speed claim. Dual16
must not replace V2 or Official Adaptive. The next useful change is not another
larger all-state run: it is a pre-action abstention rule that leaves the full
V2 pool alone when structural candidates have low repairability, and exposes
at most the justified structural challenger in suitable states. Any such rule
requires a new paired test and may not be tuned on these two seeds.

## Identity

- Source commit: `adbb9ca`.
- Output root: `build/stride-structshell-fourmap-fivearm-quick-v2`.
- Solver seed: `24`.
- Config SHA-256:
  `370bf2f47c16f02b50b0193cfcc0560bc3b229f181ecced053f02a4d1f5aea73`.
- Run fingerprint:
  `2943389c8868d403eb683cb120327c104f5a9e5d0d611746e44e87262f85e3cb`.
- Schedule SHA-256:
  `1503c648a6ea513e4844a360b6d20000e73b844ae4c253be733436a959465572`.
- Report SHA-256:
  `093e0eaef4235b2ab3da064b0af6158bdb4c57ef8ba04125545eba9040cd723a`.
- Observed wall time: `569.06` seconds (9 minutes 29 seconds).

