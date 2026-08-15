# STRIDE HybridStructPool runtime optimization v7 report

## Result

V7 completed all 152 registered engineering episodes (76 HybridStructPool and
76 frozen V2) with zero execution errors or process timeouts. Integrity and all
quality-preservation gates passed, but the three preregistered 5% engineering
speed gates failed. V7 therefore does not advance as a successful runtime
optimization milestone.

The implementation preserves the complete candidate contract and frozen V2
Copeland selector. On the frozen 500-agent Room state, V6 and V7 produced the
same serialized `HybridStructPoolResult` SHA-256:
`850b7f07a509cf3072fc90f892525056bdb3314be6d3a4417be385e33ac24117`.

| Hybrid runtime component | V6 mean (s) | V7 mean (s) | Reduction | 5% gate |
|---|---:|---:|---:|---:|
| Candidate generation | 6.1524 | 5.9412 | 3.43% | fail |
| Controller before repair | 8.1996 | 8.1532 | 0.56% | fail |
| Hybrid generation total | 6.2206 | 6.0120 | 3.35% | fail |

V7 combines ordinary and bottleneck-only temporal evidence in one reservation
scan and replaces private tuple reservation keys with collision-free integer
keys. An alternating heavy-state microbenchmark improved by 6.47%, but that
local result did not reach the full-runtime 5% gates.

## Quality preservation

HybridStructPool success remained 56/76 (73.68%) and platform entry remained
40/76 (52.63%). Normalized fixed AUC was 0.16388 and restricted mean repair
decisions were 29.54, both within the frozen V6 preservation bounds. The V2 arm
remained at 44/76 success and 60/76 platform entries. Per-map capped-wall and
platform gates passed.

The report SHA-256 is
`c937357929d1edc063f05a0b57bc8cc520f11aeaf14bdd078c0b2ca1cd8f620b`.

## Decision boundary

The exact implementation may be retained as a modest local improvement, but V7
does not establish the preregistered full-runtime acceleration and does not
authorize a raw-TTF rerun, candidate filtering, ranker retuning, or default-pool
promotion. Further work must target a separately preregistered hotspot and keep
the complete candidate and quality contracts unchanged.
