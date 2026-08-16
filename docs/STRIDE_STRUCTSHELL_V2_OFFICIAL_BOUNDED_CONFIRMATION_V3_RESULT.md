# StructShell bounded confirmation v3-r2

## Decision

`structshell_only` does not replace frozen V2. The 180-second bounded,
result-blind confirmation completed without execution errors, process timeouts,
invalid actions, semantic mismatches, or fingerprint mismatches, but StructShell
failed four of the six preregistered performance gates. V2 therefore remains
the default runtime controller and Hybrid runtime tuning stops at this
milestone.

This experiment supports only a bounded-runtime conclusion. It is not an
uncapped raw-TTF claim.

## Registered protocol

- 10 held-out map groups spanning Maze, Room, Warehouse, random, Arena, and
  DAO/Game-style maps.
- Solver seeds 13-15, 60 paired task/seed keys, and three controllers:
  `official_adaptive`, `v2_only`, and `structshell_only`.
- Strict round-robin serial timing with one formal worker.
- Identical 180-second wall-clock budget for all controllers; `wall_timeout`
  and `repair_limit` are valid right-censored outcomes.
- Native PP, `episode_stream` seed policy, reset-inclusive restricted TTF.
- Run fingerprint:
  `edba8e285690e48f4936cebba53805d9313e7dbfd6444a18dba879533720d790`.

All 10 qualification groups passed. The formal collection contains 180/180
episodes and 60/60 complete paired keys.

The formal `collection_progress.json` is complete. The top-level
`collection_status.json` remains the pre-analysis snapshot because the original
run reached 180/180 and then the report process failed on an undefined AUC value
before its final status write. It is intentionally not rewritten under the
post-hoc analysis producer identity; the immutable manifests, progress file,
and regenerated report are the completion evidence.

## Aggregate results

| Controller | Success | Mean restricted TTF (s) | Normalized wall AUC | Mean repair decisions | P95 / max repair decisions |
|---|---:|---:|---:|---:|---:|
| Official Adaptive | 54/60 (90.00%) | 28.473 | 0.13571 | 144.37 | 438.7 / 3001 |
| Frozen V2 | 53/60 (88.33%) | 34.406 | 0.13290 | 238.70 | 1052.2 / 5579 |
| StructShell | 52/60 (86.67%) | 33.622 | 0.14025 | 76.38 | 389.6 / 1269 |

Normalized AUC is defined for 54 episodes per controller. Six paired keys were
already feasible at reset (`initial_conflicts == 0`), so their normalized AUC
is mathematically undefined and is masked rather than treated as zero or as
right-censoring.

StructShell spends, on average, 3.620 seconds generating candidates and 5.257
seconds on neighborhood selection per episode. It reduces the repair-decision
tail substantially, but that mechanism benefit does not compensate for its
lower success count, worse conflict AUC, uncertainty, and per-map regression.

## StructShell versus V2 gates

| Gate | Result | Evidence |
|---|---|---|
| Success not below V2 | Fail | 52/60 versus 53/60 |
| Mean restricted TTF lower | Pass | 33.622 versus 34.406 seconds; +2.28% |
| Paired faster fraction at least 50% | Pass | 33/60 = 55.0% |
| 95% paired bootstrap supports gain | Fail | CI [-19.47%, +20.97%] crosses zero |
| Every map within 5% regression | Fail | worst: random high-load, -15.20% |
| Normalized wall AUC noninferior | Fail | 0.14025 versus 0.13290 |

On the 51 common-success pairs, StructShell raw TTF averaged 10.298 seconds
versus 11.350 seconds for V2. This subset is diagnostic only because it excludes
discordant failures.

## Map-level diagnosis

Clear StructShell gains appeared on Room 64x64 (+32.95%, faster on 6/6 pairs),
Maze 32x32 (+57.21%), Arena (+10.92%), the narrower congested Warehouse group
(+10.42%), and den020d (+7.37%). The decisive regression was the random 32x32
high-load group: restricted TTF worsened by 15.20% and success fell from 5/6 to
4/6. On Maze 128x128, all three controllers remained right-censored on every
key at 180 seconds.

## Conclusion

StructShell exposes useful structural neighborhoods and sharply reduces repair
iterations on some map families, especially Room, but it is not a safe general
runtime replacement. The evidence is heterogeneous rather than a stable global
gain. Per the preregistered stopping rule, V2 remains the default and no further
Hybrid runtime compression or retuning is justified on this cohort. Any future
post-failure rescue study must remain a separate mechanism milestone and charge
all rescue cost to end-to-end timing.

The machine-readable evidence is in
`build/stride-structshell-v2-official-bounded-confirmation-v3-r2/confirmation_report.json`.
