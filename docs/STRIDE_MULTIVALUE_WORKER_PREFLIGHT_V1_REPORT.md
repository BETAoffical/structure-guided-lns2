# STRIDE MultiValue worker preflight v1

## Scope

This is an outcome-blind execution preflight for the preregistered
`stride-multivalue-pilot-v1` label-stability pilot.  It selects non-timing data
collection concurrency from total throughput, errors, and conservative peak RSS.
Candidate outcomes are not read by the selection rule.

## Registered inputs and safeguards

- 12 fixed temporal state occurrences and one frozen V2-anchor rollout per state.
- Heavy-worker candidates: 8, 10, and 12.
- Each rollout retains the 128-decision, 300-second wall fuse and 360-second
  process timeout.
- The existing qualification cohort is reused; no task is selected or excluded
  from its repair result.

## Result

| Requested workers | Episodes | Wall seconds | Throughput/s | Conservative peak RSS | Errors |
|---:|---:|---:|---:|---:|---:|
| 8 | 12 | 330.666 | 0.036290 | 4,087,169,024 | 0 |
| 10 | 12 | 315.674 | 0.038014 | 4,402,520,064 | 0 |
| 12 | 12 | 313.974 | 0.038220 | 4,560,756,736 | 0 |

The selected count is **12 workers**.  WSL reported 20 logical CPUs and
24,452,161,536 available bytes; the selected conservative RSS is below the
registered 75% memory ceiling and leaves the required two logical CPUs.

The first execution attempt stopped before policy rollout because a per-rollout
directory had not materialized its reused qualification manifest.  The executor
was corrected to copy the complete registered qualification cohort first and
then run only the target task-seed policy episode.  The completed preflight above
contains zero errors and zero timeouts after that correction.

## Identity and claim boundary

- State manifest SHA-256:
  `51476a171dc51719f285b9ff46355826dec6a3e484de2387f9fd094110defb9b`
- Worker preflight report SHA-256:
  `4bbdd72582afb0ba42eac1c26c56779215fa70ba91a179978d9592c7082062da`
- This preflight is not a model-quality or TTF-improvement result.
- Model training remains forbidden until the registered dual-teacher,
  cross-seed, and per-map stability gates pass.
