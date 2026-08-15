# STRIDE Bounded Native Retry Continuation v1 Result

## Decision

The preregistered initial screen stopped after trial 0--3. The bounded same-set
native retry moved persistent-platform risk in the intended direction, but the
stage is not eligible for trial 4--7 because integrity did not fully pass and
normalized fixed conflict AUC was worse. No extension, runtime integration,
training, or TTF experiment is authorized by this result.

## Frozen identity and completion

- Implementation commit: `34ecdcd5d5b7cba2147a58f6729913e2bc7fe9ec`.
- Registration SHA-256:
  `505b8df70770f3f5a75df304c449c4aff02e7434bd397d8e2a7b8df0d9730e3b`.
- Run fingerprint:
  `911040657ae527c2cdc1bf437da5175fcaca6f7486f699f216a117a2fe4e8760`.
- Native module SHA-256:
  `d0ad5b99ee3573b2cfdfb00c7fed16d4191f968ce82597b97add50fd69f97b3e`.
- Coverage: 45 frozen `first_repeat_stall` states, two arms, and paired trial
  indices 0--3, for 360/360 complete episodes.
- Execution: 16 workers; success or 64 repair decisions / 180 seconds;
  240-second process and 300-second outer-job fuses.
- Collection completed with zero execution errors and zero process timeouts.
  `wall_timeout` and `repair_limit` remain valid right-censoring outcomes.

## Initial result

| Metric | Single native attempt | Bounded same-set retry | Difference |
|---|---:|---:|---:|
| Persistent-platform rate | 83.89% | 75.00% | -8.89 pp |
| Success rate | 46.11% | 48.33% | +2.22 pp |
| Normalized fixed AUC | 0.45230 | 0.45822 | +0.00592 (worse) |
| Restricted mean repair decisions | 41.098 | 39.427 | -1.672 |
| Mean repair wall | 72.024 s | 74.685 s | +2.661 s |
| Right-censored episodes | 97/180 | 93/180 | -4 |

The state-cluster paired bootstrap difference in persistent-platform rate was
`-0.08889`, with 95% interval `[-0.13889, -0.04444]`. Platform-rate point
estimates improved on all three maps:

- `maze-128-128-1`: -6.25 percentage points;
- `maze-128-128-2`: -10.42 percentage points;
- `maze-32-32-4`: -10.29 percentage points.

The directional initial gate therefore passed: platform risk decreased,
success did not decrease, and no map worsened by more than five percentage
points. This is not sufficient for promotion because the integrity gate is
conjunctive.

## Integrity failure

All 180 first attempts were paired across arms, and all 180 matched the
registered NativeOrder first-attempt artifact. Of the 180 retry-parity checks,
179 passed and one failed:

- state fingerprint:
  `749e44508e4e0de86f472a378f106035c117b2e67a1522039609af6f5c188e48`;
- map/task: `maze-128-128-2`, 600 agents, solver seed 17, trial 0;
- retry PP seed: `582960369` in both artifacts;
- native repair order: identical in both artifacts;
- completed NativeOrder outcome: `conflict_bound_exceeded`;
- bounded-continuation outcome: `time_limit`.

This is a real protocol mismatch rather than missing or rolled-back data. The
bounded continuation supplies the live remainder of its 180-second episode
budget to PP, whereas the reused NativeOrder transaction was collected under a
270-second transaction budget. The registered field-level retry parity promise
is therefore false for this case. The case cannot be removed, and the wall limit
cannot be increased after seeing the result.

## Gate interpretation

The mechanism signal is encouraging but non-promotable:

- platform bootstrap upper bound below zero: passed;
- success not lower: passed;
- restricted mean repair decisions lower: passed;
- no map platform-rate worsening over five points: passed;
- normalized fixed AUC lower: failed;
- complete registered retry parity: failed.

Accordingly `integrity_passed=false`, `mechanism_passed=false`, and
`extension_allowed=false`. Trial 4--7 was not launched. The result suggests
that a bounded fresh-seed retry can reduce repeated-platform incidence, but it
does not establish a net trajectory or runtime improvement under the frozen
protocol. A future experiment would require a new preregistration with a single
internally consistent PP/episode budget; it must not reuse or patch this run.
