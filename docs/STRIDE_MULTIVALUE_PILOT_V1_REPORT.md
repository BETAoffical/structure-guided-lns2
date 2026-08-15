# STRIDE MultiValue Pilot v1 Report

Date: 2026-08-12

## Decision

The registered MultiValue pilot completed both paired-seed halves with all
integrity checks passing, but label stability failed globally and on every
registered map.  `stride-multivalue-v1` stops as a diagnostic-only research
line.  It does not authorize MultiValue-GBDT training, a structural model,
policy iteration, runtime integration, or a TTF experiment.

The next admissible development step is a separately preregistered
Repairability causal audit.  It must instrument PP's internal failure reason,
failed agent and repair-order position, per-agent conflict creation, and
external blockers, then vary agent membership and repair order independently.

## Completion and integrity

- state occurrences: 12 / 12;
- candidates: 116;
- initial paired rollout episodes: 1,856 / 1,856;
- extension paired rollout episodes: 1,856 / 1,856;
- analyzed labels: 3,712 / 3,712;
- teachers: frozen `v2-full` and official Adaptive LNS2;
- trial indices: 0 through 15 for every state/candidate/teacher;
- feature vectors: 124-dimensional for every candidate;
- collection errors and terminal timeouts: 0;
- paired PP seed, forced action, repair structure, restore provenance, and
  native action checks: all passed;
- runtime and TTF are not labels.

All 3,712 restored full-state fingerprints differ from the source full-state
fingerprint because `reset_paths` intentionally reinitializes iteration and
low-level counters.  The repair-structure fingerprint and registered restore
provenance match in every episode; this is an expected identity distinction,
not corrupted data.

## Stability result

| Metric | Result | Frozen gate | Passed |
| --- | ---: | ---: | --- |
| Seed-half Top-3 overlap | 0.513889 | >= 0.80 | no |
| Paired rank correlation | 0.306587 | >= 0.60 | no |
| Cross-teacher direction agreement | 0.471154 | >= 0.70 | no |
| Cross-seed normalized regret | 0.010651 | <= 0.02 | yes |

Map groups:

| Map | States | Top-3 overlap | Rank correlation | Teacher agreement | Regret | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `maze-128-128-1` | 4 | 0.666667 | 0.300893 | 0.571429 | 0.012629 | no |
| `maze-128-128-2` | 4 | 0.291667 | 0.293444 | 0.386364 | 0.007283 | no |
| `maze-32-32-4` | 4 | 0.583333 | 0.325425 | 0.512821 | 0.009935 | no |

Increasing every state from eight to sixteen paired PP seeds reduced average
regret below its gate but did not stabilize the candidate set, ranking, or
direction across continuation teachers.  The future outcome is therefore not
a sufficiently stable candidate property for supervised model training.  A
larger model would fit a target that still mixes the forced action, PP
realization, and continuation-policy behavior.

## Analysis execution

The analysis implementation now supports state-level process parallelism.
Two real rollout episodes produced exactly equal serial and two-process result
dictionaries before the complete analysis was run with 12 state workers.  A
single parent process retained deterministic sorting and artifact writes.

This is an execution improvement only.  It does not change labels, gates, or
the scientific conclusion.  State-level work was imbalanced, so future large
analyses may use smaller read-only episode batches under the same exact-output
gate before increasing the process ceiling to 16.

## Validation

- WSL Python: 801 passed, 35 skipped;
- Windows native test executable: passed;
- Linux CTest from `build/wsl-release`: 11 / 11 passed;
- repository hygiene: 0 errors, 24 / 24 evidence entries verified;
- `git diff --check`: passed.

## Artifact identity

- extended stability report:
  `build/stride-multivalue-pilot-v1/extended_stability_report.json`;
- label SHA-256:
  `ab2a61ea4474db2a0ffe2e558350a9033a9bdc778cfe4be0e5de3e17d534a968`;
- state-manifest SHA-256:
  `51476a171dc51719f285b9ff46355826dec6a3e484de2387f9fd094110defb9b`.
