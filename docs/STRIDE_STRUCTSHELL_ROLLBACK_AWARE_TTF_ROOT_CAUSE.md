# Rollback-Aware StructShell TTF Root-Cause Report

## Scope

Source: `build/stride-structshell-rollback-aware-ttf-v1-r2/screen/screen_report.json`.
The screen contains ten paired keys (one key per map, solver seed 16), a
180-second right-censored outcome window, reset-inclusive timing, and strict
serial execution.  It is sufficient to diagnose the implementation failure,
but not to delete candidate families or make a map-general performance claim.

## Overall result

| Controller | Success | Mean restricted TTF | Normalized wall AUC | Mean repair decisions | Mean PP time |
|---|---:|---:|---:|---:|---:|
| Official Adaptive | 9/10 | 30.847 s | 0.15045 | 109.6 | 17.447 s |
| V2 | 9/10 | 31.654 s | **0.14923** | 56.1 | 17.929 s |
| StructShell + candidate guard | 9/10 | 42.220 s | 0.15565 | **36.7** | 27.060 s |

The challenger was 10.566 seconds (33.38%) slower than V2 and faster on only
one of ten paired keys.  Across all ten keys, its restricted TTF total increased
by 105.661 seconds.  The measured decomposition is:

- additional PP time: 91.311 seconds (86.42%);
- additional controller time: 13.638 seconds (12.91%);
- additional reset time: 0.581 seconds (0.55%);
- remaining accounting difference: approximately 0.12%.

Most controller-generation overhead came from the already right-censored
Maze-128 key.  Excluding it, candidate generation added only about 0.425 seconds
over nine keys.  Candidate-generation micro-optimisation therefore cannot
recover the observed loss.

## Per-map result

Times are `Official / V2 / StructShell+guard` restricted TTF.

| Map | Success | TTF | Challenger minus V2 | Main observation |
|---|---|---:|---:|---|
| arena | all | 25.482 / 20.921 / 29.671 | +8.749 s | fewer decisions, but PP rose 11.346 to 21.290 s |
| den020d | all | 22.753 / 25.477 / 30.074 | +4.597 s | same decision count as V2; structural PP was costlier |
| den312d | all, initially feasible | 0.770 / 0.780 / 0.831 | +0.051 s | reset/noise scale only |
| maze-128 | none | 180 / 180 / 180 | 0 s | all right-censored; challenger spent 11.489 s generating candidates |
| maze-32 | all | 5.397 / 3.499 / 6.770 | +3.270 s | 26 exact rollbacks despite fewer decisions |
| random-32 high load | all | 11.213 / 13.447 / 13.435 | -0.011 s | only paired win; effectively tied |
| random-64-10 | all | 0.769 / 0.712 / 0.734 | +0.022 s | no meaningful structural benefit |
| random-64-20 | all | 1.152 / 1.261 / 1.297 | +0.036 s | no meaningful structural benefit |
| room-64-64-16 | all | 39.425 / 49.558 / **134.534** | **+84.976 s** | 105 exact rollbacks and 115.202 s PP |
| warehouse | all | 21.511 / 20.884 / 24.854 | +3.970 s | 6 structural size-32 actions, 5 reduced conflicts, but PP cost dominated |

Room contributed 80.42% of the total regression.  Removing Room does not make
the result positive: the remaining nine keys were still 2.298 seconds per key
(7.75%) slower on average.

## Mechanism failure

The guard bounded consecutive failures of one candidate, not failures of the
repair state.  After three exact rollbacks it banned only that candidate and
then forced the next remaining StructShell challenger.  It returned to V2 only
after every structural challenger was exhausted; a V2 rollback then cleared
the bans and reopened the structural scan.

In the Room trace:

- 124 decisions and 105 exact rollbacks;
- 95 StructShell selections and 28 individual bans;
- 87 repair-state candidate-pool cache hits;
- one unchanged repair fingerprint accumulated 45 exact rollbacks across 15
  candidates; another accumulated 36 across 12 candidates;
- the V2 fallback was never reached;
- guard bookkeeping itself cost only 0.012 seconds, while PP cost 115.202
  seconds.

Thus the implementation converted a single-candidate loop into a candidate-
rotation loop.  The failure is not primarily a slow Python guard.

## Neighborhood and ranking diagnosis

The challenger selected StructShell on 223 of 367 decisions (60.8%).  Its mean
selected size was 19.87 versus V2's 12.44, and its mean PP cost per decision was
0.737 versus 0.320 seconds.  Size alone is not a sufficient deletion rule:

| StructShell size | Executions | Exact rollback rate | Strict conflict reduction | Mean PP |
|---:|---:|---:|---:|---:|
| 8 | 16 | 93.8% | 6.3% | 0.629 s |
| 16 | 30 | 90.0% | 3.3% | 0.920 s |
| 24 | 51 | 92.2% | 7.8% | 0.810 s |
| 32 | 126 | 41.3% | 56.3% | 1.394 s |

The small and medium structural actions were largely reached through the broken
fallback scan.  Size 32 was more repair-capable but expensive.  On Warehouse
and arena, structural actions often reduced conflicts and decisions, yet their
PP cost still outweighed the saved rounds.  This is a separate cost-quality
ranking problem; the state-guard fix cannot solve it by itself.

The frozen V2 ranker also evaluates many structural candidates outside its
training range (for example, about 41% of selected Room candidates).  A new
ranker is not introduced in the guard repair because that would confound the
control-logic diagnosis.  Candidate-family deletion is likewise prohibited by
this one-seed screen.

## Corrective action

The independently versioned v3 repair uses the existing strict conflict-
structure activation gate and a three-rollback total budget per exact repair
fingerprint.  Budget exhaustion disables all StructShell candidates for that
fingerprint and returns each subsequent decision to freshly generated V2-only
selection.  V2 rollback does not reopen StructShell; returning to an old
fingerprint restores its prior latch.  PP, seed policy, V2 features, Copeland
ranker, and one-PP-per-decision semantics remain unchanged.

This directly removes the unbounded candidate scan.  It does not guarantee that
V2 will escape its own platform or that expensive successful structural actions
will become cost-effective.  Those questions are evaluated separately in the
registered same-key four-arm diagnostic.
