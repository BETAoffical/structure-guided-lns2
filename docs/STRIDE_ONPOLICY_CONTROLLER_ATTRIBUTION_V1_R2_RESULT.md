# STRIDE initial-path controller attribution v1 r2 result

## Decision

The registered r2 initial cohort completed with full integrity, but neither
official arm qualified for uniform extension to trials 4-7.  Stop this branch
at the initial gate.  Do not import the stopped r1 outcomes, run the extension,
train or integrate a controller, replace a candidate pool, or make raw-TTF or
non-Maze claims from this result.

Starting official Adaptive or fixed Target-8 from the original InitLNS paths
does not prevent the registered Maze long-tail cases.  Both official arms enter
the registered exact-rollback platform more often than the best frozen arm,
solve substantially fewer episodes, accumulate more conflict AUC and repair
decisions, and use more capped controller-continuation wall time.

## Registered identity and integrity

- implementation commit: `6ff301465b27fcefea0a7b5f45a634111235b1f2`;
- run fingerprint:
  `85158c81aec887e8b739137318166e9c5ed699f6459da674e9e4b8f8a2e5e9b7`;
- cohort: 19 unique `task_id + solver_seed` keys collapsed from all 45
  registered Maze platform cases;
- arms: frozen SlotPool, frozen StructPool, official Adaptive N=8, and fixed
  Target N=8;
- initial trials: 0-3, 76 episodes per arm, 304 total;
- execution: 16 workers, success or 64 repair decisions / 180 seconds;
- each arm starts from the same registered original path at decision index 0,
  before any LNS repair; no platform checkpoint, forced first action, retry, or
  explicit repair order is used;
- `initial_solution_seconds == 0` and `path_restore_seconds > 0` for all 304
  episodes, so r2 does not recompute InitLNS;
- complete episodes: 304/304; execution errors / process timeouts: 0 / 0;
- task, arm, trial, initial-state fingerprint, and first-PP-seed pairing: passed;
- native PP order and every registered integrity check: passed;
- the 112 valid r1 outcomes and its one process timeout were not imported.

Artifact SHA-256:

- `initial_report.json`:
  `8c8a4ee9b9d26576c400277c3d0e74bbadf719631cc8b58d811e25fca2d2d50a`;
- `collection_status.json`:
  `b7e507024b66f1e77c2a9d0464a39d84046ea007542c9bc664af9f2aca4667d5`;
- `collection_progress.json`:
  `47a655e791f13713f908ec5fa8a863cc1197a88386ed5171c4fa1fdf85123922`;
- `initial_run_config.json`:
  `fd42572e7c0705a4a346eb411dda001318fe8871c1ea8ac12d23d809b42e655c`;
- `initial_schedule.jsonl`:
  `daba3b23a5eed5470d1a2e5451f7ac2cc0574b79917e3af89eb5fb1ee6ba4191`;
- r2 registration:
  `da1e3701071f8e7bcc6f2bcee31bc02ee40c0c912af6f1f3bcb4e6e7af450905`.

## Aggregate results

| Metric | Frozen SlotPool | Frozen StructPool | Adaptive N=8 | Target N=8 |
|---|---:|---:|---:|---:|
| Platform entry | 52.63% | 63.16% | 65.79% | 72.37% |
| Success | 67.11% | 59.21% | 34.21% | 26.32% |
| Normalized fixed AUC | 0.15415 | 0.18754 | 0.28686 | 0.33023 |
| Restricted mean repair decisions | 29.32 | 34.86 | 54.07 | 56.94 |
| Capped controller-continuation wall (s) | 67.96 | 80.57 | 120.65 | 134.40 |
| Native repair wall (s) | 48.32 | 53.52 | 40.23 | 41.06 |
| Unique neighborhoods | 14.95 | 15.46 | 47.25 | 47.04 |

The lower native repair wall of the official N=8 arms does not translate into
a faster continuation.  They execute many more repair decisions and spend more
total controller-continuation wall time.  The inherited report field
`mean_capped_ttf_seconds` is therefore interpreted only as capped continuation
wall after registered path replay; it excludes original InitLNS generation and
is not raw end-to-end TTF.

## Paired task-cluster bootstrap

Adaptive minus frozen SlotPool:

- platform entry: +0.1316, 95% CI [+0.0263, +0.2368];
- success: -0.3289, 95% CI [-0.4868, -0.1711];
- normalized fixed AUC: +0.1327, 95% CI [+0.0912, +0.1819];
- repair decisions: +22.6974, 95% CI [+16.4865, +29.0266];
- continuation wall: +52.69 s, 95% CI [+26.27, +80.28].

Target minus frozen SlotPool:

- platform entry: +0.1974, 95% CI [+0.0658, +0.3289];
- success: -0.4079, 95% CI [-0.5789, -0.2237];
- normalized fixed AUC: +0.1761, 95% CI [+0.1265, +0.2362];
- repair decisions: +25.1053, 95% CI [+17.5260, +32.5000];
- continuation wall: +66.44 s, 95% CI [+36.24, +96.03].

Neither official arm passes any final promotion gate against both frozen arms.

## Map-level diagnosis

- `maze-128-128-1`: Adaptive enters a platform in 92.86% and succeeds in
  25.00%, versus 78.57% and 60.71% for frozen SlotPool.
- `maze-32-32-4`: Adaptive enters a platform in 66.67% and succeeds in 52.78%,
  versus 50.00% and 94.44% for frozen SlotPool.  Target is worse at 88.89%
  platform entry and 38.89% success.
- `maze-128-128-2`: every arm reaches 64 decisions / 180 seconds with 0%
  success, while the exact three-rollback platform detector reports 0% entry.
  This is important negative evidence: the exact-platform label captures one
  self-loop mechanism but does not cover every severe no-progress trajectory.

## Interpretation

The earlier exact-state experiment showed that official N=8 policies can leave
an already observed repair signature quickly.  This initial-path experiment
answers the missing question and rejects the stronger explanation that official
neighborhood selection generally avoids platforms before they form.  High
neighborhood diversity breaks individual signatures, but it also moves through
many poor basins; it neither preserves success nor reduces total repair effort.

The evidence now separates two facts:

1. the frozen selector contributes to persistence once a signature repeats;
2. replacing it with official Adaptive or Target-8 from the beginning is not a
   solution and is substantially worse on this registered cohort.

The remaining mechanism question is below the selector layer: whether a
stronger bounded native repairer can repair the selected set without PP's
order-sensitive rollback behavior.  That requires an independent rollback and
deadline qualification before any paired continuation experiment.  It must not
be treated as an automatic continuation of this failed gate.

## Validation

- WSL Python with 16 workers: `985 passed, 35 skipped`;
- Linux CTest with `-j16`: `13/13` passed;
- Windows native `build/windows/Release/lns2_tests.exe`: passed;
- repository hygiene: 24/24 evidence entries verified, zero errors;
- `git diff --check`: passed.
