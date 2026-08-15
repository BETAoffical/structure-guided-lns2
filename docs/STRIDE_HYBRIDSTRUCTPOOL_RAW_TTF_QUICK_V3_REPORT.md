# STRIDE HybridStructPool raw-TTF Quick v3 report

## Result

Quick v3 completed all 24 strictly serial timed episodes with zero execution
errors, process timeouts, invalid actions, or fingerprint mismatches. All three
controllers solved all eight paired keys under the reset-inclusive raw-TTF
clock.

| Controller | Success | Mean raw TTF (s) | Median raw TTF (s) | Mean repairs | Mean PP (s) | Mean selection (s) |
|---|---:|---:|---:|---:|---:|---:|
| Official Adaptive LNS2 | 8/8 | 16.0697 | 13.5727 | 57.500 | 6.4592 | 0.0045 |
| Frozen V2 | 8/8 | 13.0681 | 11.1955 | 20.375 | 3.8043 | 0.5151 |
| V8 full HybridStructPool | 8/8 | 13.7372 | 10.7097 | 8.750 | 3.9983 | 1.0800 |

HybridStructPool was 14.52% faster than official Adaptive LNS2 and won 6/8
paired keys. It remained 5.12% slower than frozen V2 and won only 3/8 keys,
despite using 11.625 fewer repair decisions. Candidate generation fell from the
Quick-v2 observation of 0.9603 seconds to 0.8113 seconds and total selection
fell from 1.2258 to 1.0800 seconds. These cross-run deltas are descriptive;
the within-v3 paired comparisons are the valid promotion evidence.

## Map groups

- Maze-300: Hybrid was 2.33% faster than official but 11.46% slower than V2,
  with 1/4 paired wins against V2.
- Room-500: Hybrid was 28.89% faster than official and 3.75% faster than V2,
  with 2/4 paired wins against V2.

The Room group again repaid the full-pool overhead through fewer repairs. Maze
did not, and its 11.46% regression exceeded the registered 10% group bound.

## Gates and decision

All integrity, success, and repair-count gates passed. Mean raw TTF and paired
faster fraction passed against official Adaptive. Promotion failed because mean
raw TTF and paired-faster fraction did not beat V2 and the per-group regression
gate failed. `default_replacement_allowed` and `formal_speed_claim` remain
false.

The report SHA-256 is
`46ced7fa3dc53bca20d0101709747f2e57bfbac8f90fdd33b0bc1cc90632b510`.

Quick v3 remains development evidence on outcome-enriched tasks. The remaining
problem is not candidate correctness: it is paying full Hybrid generation on
Maze decisions where the frozen V2 ranker selects little additional repair
benefit. Further promotion requires a preregistered outcome-blind activation or
routing rule, followed by fresh result-blind confirmation; it must not be tuned
by deleting candidates on these eight keys.

## Validation

- WSL Python: 1014 passed, 35 skipped (`pytest -n 16`).
- Linux native: 13/13 passed (`ctest -j16`).
- Windows native: all tests passed.
- Repository hygiene: 0 errors; 24/24 retained evidence entries verified.
