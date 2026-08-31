# Dual16 Hierarchical Admission H1 V1

## Outcome

The independent current-step label-support pilot completed with full integrity:
36/36 authenticated station-centric warehouse checkpoints, six maps, twelve
states in each of the `medium_high`, `high`, and `very_high` load bands, 108
exact-set-deduplicated actions, and 1,728/1,728 paired PP repairs. There were no
execution errors, missing trials, runtime/future label fields, or identity
mismatches.

The frozen V2 action was ranked over the base V2 pool before Component16 or
Hotspot16 was generated. All 36 states were `all_distinct`: V, C, and H had
different exact agent sets. Consequently this cohort exercised no C/H alias
execution saving, while retaining the registered exact-set deduplication path.

## H1 labels

| Load band | States | Stage 1 structural consensus | Stage 1 abstain | Stage 1 keep V2 | Stage 2 Component | Stage 2 Hotspot |
|---|---:|---:|---:|---:|---:|---:|
| Medium-high | 12 | 10 | 2 | 0 | 12 | 0 |
| High | 12 | 3 | 9 | 0 | 12 | 0 |
| Very high | 12 | 6 | 6 | 0 | 12 | 0 |
| **Total** | **36** | **19** | **17** | **0** | **36** | **0** |

At the role-vote level, Component supported structural admission in 33 states
and abstained in three; it never stably supported V2. Hotspot supported
structural admission in 19 states, supported V2 in nine, and abstained in eight.
Every eligible Component-vs-Hotspot comparison stably selected Component.

## Decision

`NO_GO_LABEL_SUPPORT`: do not fit the planned hierarchical model and do not
register a runtime selector. The pilot lacks both Stage 1 `keep_v2` labels and
Stage 2 Hotspot labels, so neither head has the preregistered two-sided label
support. Passing integrity does not override this support failure.

The result supports a narrower development hypothesis—Component16 is the only
structural arm with consistent current-step support in this targeted cohort—but
it is not an end-to-end TTF or speed result. Any Component-focused runtime claim
still requires a separately preregistered paired end-to-end experiment, followed
by fresh map-disjoint confirmation.

## Evidence boundary

- Experiment: `stride-dual16-hierarchical-admission-h1-v1`
- Data line: `stride-dual16-hierarchical-admission-h1-data-v1`
- Collection run fingerprint:
  `198cab2514bc3d25b20e8007dc707c97657a1562946da14cd32f54c2c15c130b`
- Labels SHA256:
  `88ce40442fdce6363109b6e9419f38532a1e686189b12f0347a89c022dca5b28`
- Collection report SHA256:
  `e2f70c48f374da080fcdc25c313b47d53bc277237a0a116b96d29a9bab2e27a5`
- Evidence root:
  `build/stride-dual16-hierarchical-admission-h1-v1`
- Archived evidence ZIP:
  `../backups/dual16-hierarchical-admission-h1-v1-20260901.zip`, SHA256
  `02b6cb36b11229193eb3bbb687e9e74aa34163b514239324ff2cfeb27ae68a5a`
- Scientific status: sequential development only; training, runtime
  integration, default replacement, and TTF/speed claims remain unauthorized.
