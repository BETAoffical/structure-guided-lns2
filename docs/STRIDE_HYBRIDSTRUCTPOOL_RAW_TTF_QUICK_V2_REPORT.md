# STRIDE HybridStructPool raw-TTF Quick v2 report

## Result

The optimized eight-key Quick completed all 24 strictly serial timed episodes
with zero execution errors, process timeouts, invalid actions, or fingerprint
mismatches.  Every controller solved all eight episodes.  All arms were rerun
under the reset-inclusive raw-TTF clock; no V1 timing result was imported.

| Controller | Success | Mean raw TTF (s) | Median raw TTF (s) | Mean repairs | Mean PP (s) | Mean selection (s) | Normalized wall AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official Adaptive LNS2 | 8/8 | 15.2184 | 13.3208 | 57.500 | 6.2089 | 0.0044 | 0.67639 |
| Frozen V2 | 8/8 | 12.6392 | 11.2617 | 20.375 | 3.8641 | 0.5206 | 0.74650 |
| Optimized full HybridStructPool | 8/8 | 13.0718 | 10.4315 | 8.750 | 3.8073 | 1.2258 | 0.77859 |

HybridStructPool was 14.1050% faster than official Adaptive LNS2 and faster on
6/8 paired keys.  Against frozen V2 it was 3.4225% slower and split the paired
keys 4/8, despite using 11.625 fewer repair decisions.  Its candidate generation
mean fell to 0.9603 seconds and total selection to 1.2258 seconds, but selection
still cost about 0.7052 seconds more than V2.  The remaining raw-TTF loss was
only 0.4326 seconds, so the complete pool's repair savings now almost offset its
online cost but do not yet exceed the frozen anchor.

Relative to the earlier V1 development run, Hybrid's observed mean raw TTF
changed from 12.3253 to 13.0718 seconds while its selection time fell from
1.4339 to 1.2258 seconds.  This cross-run comparison is descriptive only;
machine noise and different baseline timings mean the within-V2 paired
comparison above is the valid decision criterion.

## Map groups

- Maze-300: Hybrid was 2.46% faster than official Adaptive but 10.31% slower
  than V2, with only 1/4 paired wins against V2.
- Room-500: Hybrid was 26.93% faster than official Adaptive and 5.28% faster
  than V2, with 3/4 paired wins against V2.

The strong Room result shows that the complete pool can repay its generation
cost when it avoids many repairs.  The Maze regression shows that a global
default remains unsafe: the same complete candidate universe does not provide
enough additional repair benefit on every group to cover its fixed overhead.

## Gates and decision

All integrity gates passed.  Success and repair-count noninferiority passed,
as did the overall paired-faster fraction against both baselines.  Promotion
failed because mean raw TTF did not beat V2 and the registered maximum
per-group regression gate failed.  Therefore `default_replacement_allowed` and
`formal_speed_claim` remain false.

This is development evidence, not a formal or result-blind speed claim.  The
optimized complete HybridStructPool remains a research candidate universe, not
the runtime default.  The next useful optimization is not unregistered
candidate deletion: it is an outcome-blind, low-cost routing rule that pays for
Hybrid only in states where its expected repair savings can exceed the measured
extra selection cost, followed by a fresh result-blind paired confirmation.

Post-run validation passed: WSL Python reported 1010 passed and 35 skipped with
16 workers, Linux CTest passed 13/13 with `-j16`, Windows native tests passed,
and repository hygiene reported zero errors with 24/24 retained evidence
entries verified.
