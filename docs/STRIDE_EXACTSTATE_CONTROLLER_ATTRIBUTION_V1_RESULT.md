# STRIDE exact-state controller attribution v1 result

## Decision

The registered initial cohort completed, but neither official arm qualified for
uniform extension to trials 4-7.  Stop this branch at the initial gate.  Do not
train, integrate a controller, replace the candidate pool, run TTF, or claim
non-Maze generalization from this result.

The result separates immediate platform escape from useful repair.  Official
Adaptive and fixed Target-8 leave the initial repeated repair signature much
faster than the frozen controller, but both solve fewer episodes, require more
restricted repair decisions, and re-enter a later platform more often.  Fast
escape from the first signature is therefore not a sufficient repair objective.

## Registered identity and integrity

- producer commit before formal collection: `f428a11a8a56d131f954fe842f2796da76d903e9`;
- cohort: all 45 registered Maze `first_repeat_stall` states;
- arms: frozen controller, official Adaptive N=8, fixed Target N=8;
- initial trials: 0-3, 180 episodes per arm, 540 total;
- execution: 16 workers, success or 64 decisions / 180 seconds, PP only;
- no forced shared first action, retry, explicit repair order, or result-based
  state exclusion;
- both controller qualifications: 19/19;
- complete episodes: 540/540;
- execution errors / process timeouts: 0 / 0;
- state, arm, trial and first-PP-seed pairing: passed;
- native order and all registered integrity checks: passed.

Artifact SHA-256:

- `initial_report.json`: `4427ed5b717ee18ad1133f717919cc74f0f14c8627710cc90c4faac63e77b43a`;
- `collection_status.json`: `ed689ae7e3c9abe0d88979bd61eaa1fb9d07fbd9fb696a68d4145cdb5c58701f`;
- `initial_run_config.json`: `b1ee1d4a9fdc78bb03968823ca6ea1ac176decdb747d321b08f0fb1c9d3ae917`.

## Aggregate results

| Metric | Frozen | Adaptive N=8 | Target N=8 |
|---|---:|---:|---:|
| Initial platform unescaped after 3 decisions | 40.00% | 4.44% | 5.00% |
| Ever escaped initial signature | 85.56% | 100.00% | 100.00% |
| Mean first platform escape decision | 13.16 | 1.49 | 1.58 |
| Post-escape platform re-entry | 55.56% | 66.67% | 71.11% |
| Success | 44.44% | 29.44% | 24.44% |
| Normalized fixed AUC | 0.45328 | 0.36277 | 0.42689 |
| Restricted mean repair decisions | 41.70 | 53.59 | 54.21 |
| Mean native repair wall seconds | 74.61 | 58.00 | 59.17 |
| Mean unique neighborhoods | 14.45 | 43.43 | 41.55 |

The official arms used smaller N=8 neighborhoods and consequently spent less
native repair wall time despite executing roughly twelve more decisions.  This
bounded repair timing must not be reported as an end-to-end TTF win.

## Paired state-cluster bootstrap versus frozen

Adaptive N=8:

- unescaped-at-three difference: -0.3556, 95% CI [-0.4667, -0.2444];
- success difference: -0.1500, 95% CI [-0.2444, -0.0556];
- normalized AUC difference: -0.0905, 95% CI [-0.1659, -0.0198];
- repair-decision difference: +12.4889, 95% CI [+9.2944, +15.7056].

Target N=8:

- unescaped-at-three difference: -0.3500, 95% CI [-0.4667, -0.2333];
- success difference: -0.2000, 95% CI [-0.3000, -0.1000];
- normalized AUC difference: -0.0264, 95% CI [-0.0985, +0.0420];
- repair-decision difference: +12.1611, 95% CI [+8.4556, +15.8778].

Both official arms pass the immediate escape contrast and fail success and
repair-decision preservation.  Target also fails the AUC confidence gate.

## Map-level diagnosis

- `maze-128-128-1`: frozen success 42.19%, Adaptive 21.88%, Target 17.19%;
- `maze-128-128-2`: all three arms 0% success within the bound;
- `maze-32-32-4`: frozen success 77.94%, Adaptive 57.35%, Target 48.53%.

All three maps show a large reduction in initial-signature persistence under
the official arms, so the gate failure is not caused by one map hiding an
immediate-escape regression.  The failure is that initial escape frequently
moves to another difficult basin instead of reaching feasibility.

## Interpretation and next valid step

The frozen proposal/ranker is partly responsible for repeated use of the same
repair signature: official neighborhood diversity breaks that signature much
earlier.  It is not, however, the whole long-tail cause.  Once moved, the same
PP repair process commonly enters another platform, and the N=8 official arms
lose substantial success relative to the larger structured frozen choices.

The next valid experiment is not another selector or escape classifier.  It is
a separate rollback-integrity qualification for a stronger native repairer on
the same selected agent set.  Only if PBS (or another stronger repairer) can be
shown to preserve exact restored-state rollback and bounded execution semantics
should it be compared with PP on the frozen hard states.  That branch requires
new preregistration and must not reuse this failed extension gate.

## Validation

- WSL Python: `979 passed, 35 skipped` with 16 workers;
- Linux CTest: `13/13` passed with `-j16`;
- Windows native test executable: passed;
- repository hygiene: 24/24 evidence entries verified, zero errors;
- `git diff --check`: passed.
