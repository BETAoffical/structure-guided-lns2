# STRIDE PreTail Forced Continuation v1 Report

## Result

The preregistered bounded continuation diagnostic completed with all integrity
checks passing. It retained all 45 frozen first-structural checkpoints and ran
two paired forced-action seeds for each of three arms, for 270 episodes in 24
batches. Every episode forced the registered first action exactly once and then
returned control to the matching frozen StructPool or SlotPool controller.

The result supplies causal evidence that the deployed first structural choice
is part of the tail mechanism. The outcome-derived one-step oracle was
beneficial in 22 of 60 informative non-identical comparisons (36.67%), across
both challengers and six tasks. This passes the registered 30%, two-challenger,
four-task escape-evidence gate. It does not establish a deployable selector:
the oracle uses frozen action outcomes and remains diagnostic only.

The outcome-blind coverage-diverse arm was beneficial in 25 of 90 comparisons
(27.78%), across both challengers and six tasks. It improved aggregate success,
fixed-horizon AUC, and final conflicts relative to the deployed action, but it
also produced 24 adverse comparisons. Diversity alone is therefore useful
headroom evidence, not a safe replacement rule.

## Integrity

- 45 / 45 registered checkpoints;
- 270 / 270 episodes and 24 / 24 batches;
- 90 episodes per arm, with two paired forced-action seeds per checkpoint;
- complete three-arm pairing for every checkpoint and seed;
- forced first action exactly once in every episode;
- matching restored state, initial conflict count, and fingerprint;
- zero manifest execution errors, illegal actions, or fingerprint mismatches;
- no process-timeout execution failures;
- only `success`, `repair_limit`, and `wall_timeout` terminal outcomes;
- at most 200 repair decisions, 300 seconds of episode wall time, and a
  360-second external process timeout.

`repair_limit` and `wall_timeout` remain valid right-censored trajectories.
Censored TTF was not imputed.

## Arm summary

Lower fixed-horizon AUC and lower final conflict count are better.

| Arm | Episodes | Success | Repair limit | Wall timeout | Mean normalized fixed AUC | Mean final conflicts | Mean repair iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `actual_selected` | 90 | 56 | 13 | 21 | 0.131363 | 18.322 | 57.289 |
| `one_step_oracle` | 90 | 70 | 3 | 17 | 0.081391 | 8.889 | 42.011 |
| `coverage_diverse` | 90 | 61 | 12 | 17 | 0.110180 | 13.756 | 58.678 |

Across all arms there were 187 successes, 28 repair-limit censorings, and 55
wall-time censorings. The mean-repair statistic includes bounded censored
episodes and must not be read as an uncensored TTF estimate.

## Escape evidence

| Treatment arm | Registered informative comparisons | Beneficial | Beneficial fraction | Challenger coverage | Task coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `one_step_oracle` | 60 | 22 | 36.67% | 2 / 2 | 6 |
| `coverage_diverse` | 90 | 25 | 27.78% | 2 / 2 | 6 |

For all 90 paired oracle records, including identical-action and
non-informative cases, the classifications were 24 beneficial, 57 neutral,
and nine adverse. Coverage-diverse produced 25 beneficial, 41 neutral, and 24
adverse comparisons.

## Interpretation and decision

The candidate pool and PP operator cannot be declared faultless, but they are
not the sole explanation for the observed tails. At the same frozen state,
changing only the first selected neighborhood and then returning to the same
controller materially improved the bounded continuation often enough to pass
the registered escape gate. The deployed ranking/selection objective therefore
omits information that matters beyond immediate one-step repair quality.

Do not promote the oracle or coverage-diverse policy, do not train on these
outcomes as if they were runtime labels, and do not claim TTF improvement or
map generalization. The next safe step is a separately preregistered,
outcome-blind selector study that predicts the demonstrated escape relation
from pre-action state, candidate, coverage, repetition, and dependency
features. It must retain exact V2 fallback and must pass paired end-to-end raw
TTF and success gates before any controller promotion.

## Frozen artifacts

The completed output is
`build/stride-pretail-forced-continuation-v1-r3`.

```text
pretail_forced_continuation_report.json
39f319794155c892022d018f6b8d855d6cabfe4dfc5f81e6a4ab903d91fb2952

pretail_forced_continuation_status.json
31c4734a7b0e89c560ab0149eec471ff6a7aa6e3510a9a50e9302d8955882701

execution_schedule.jsonl
a619aba720f5ac5ed07a3e79750535255909952c34000365a7fc7b0df91905ec

runner_config.json
d7eb347ba130e2999be7b8f0999d871662ca14a17d456f0d3b8724a901fc510b
```

The report also freezes these source-product hashes:

```text
candidate_aggregates
4d3ba4ac580901937e20c07c2de8309e35c395e4754347520edfc47950bd2b82

logical_checkpoint_results
efdbc8d7020628b3ac485d0946f9ad89b1b6d7afb547a29069d2009d41671973

root_checkpoints
bbf03634a2563b3e805cdf99a33071648a08640d82605b6245fe2c16aca3ef81
```
