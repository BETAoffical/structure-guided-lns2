# STRIDE Repairability Causal Audit v1 Report

## Status and scope

The preregistered mechanism audit completed all 45 frozen
`first_repeat_stall` states from three Maze maps. It compared five paired PP
interventions at trial indices 0 through 15. The audit diagnoses whether a
repeated structural neighborhood fails because its agent set omits repair
dependencies, its PP repair order is poor, both interact, or the remaining
variation is attributable to PP instability. It is not a trained controller,
a deployable candidate rule, or a TTF experiment.

Collection integrity passed:

- 45/45 states and 3,600/3,600 trial-arm rows;
- 3,590 executed rows and 10 registered not-applicable blocker arms;
- strict state/trial paired PP seeds and independent pre-state restoration;
- zero execution errors, terminal timeouts, illegal actions, or fingerprint
  mismatches;
- runtime, TTF, and future trajectories absent from causal trial rows.

## Result

| Root classification | States | Fraction |
| --- | ---: | ---: |
| Agent-set defect | 21 | 46.67% |
| Repair-order defect | 13 | 28.89% |
| Set and order jointly required | 2 | 4.44% |
| Residual PP instability | 9 | 20.00% |

Set-related defects therefore occur in 23/45 states and order-related defects
in 15/45. The dominant mechanism is missing repair dependencies in the selected
agent set, while repair order is a substantial secondary mechanism. Residual
PP instability is real but does not explain most repeated stalls.

The intervention aggregates support that distinction:

- appending up to eight observed external blockers while preserving the
  original-agent order prefix is seed-stable in 21/45 states; its mean
  normalized conflict-reduction delta is +0.08339 and mean no-progress-rate
  delta is -0.21766;
- conflict-priority ordering with an unchanged set is seed-stable in 19/45
  states; its corresponding deltas are +0.05484 and -0.15694;
- simply reversing the selected order is stable in only 7/45 states and is
  negative on average (-0.00654 normalized reduction, +0.03889 no progress);
- moving observed blockers to the head instead of appending them is also
  negative on average (-0.07697 versus blocker-tail, +0.21534 no progress).

The baseline selected neighborhood exposes a mean 24.05 external blockers,
fails at mean repair-order position 24.49, and replans successfully in only
35.83% of paired trials. These measurements explain the repeated loop:
unselected paths continue to block PP, PP reaches a conflicting agent partway
through the order and rolls the action back, the state remains unchanged, and
the memoryless ranker selects the same neighborhood again.

## Map breakdown

| Map | Set | Order | Joint | Residual PP |
| --- | ---: | ---: | ---: | ---: |
| `maze-128-128-1` | 7 | 6 | 0 | 3 |
| `maze-128-128-2` | 10 | 1 | 0 | 1 |
| `maze-32-32-4` | 4 | 6 | 2 | 5 |

The mechanism is not identical across maps. The second large Maze is strongly
set-limited, while the small Maze has appreciable order and residual-PP
components. A single global neighborhood-size rule or arbitrary order change
would therefore repeat earlier failures.

## Claim boundary and next step

The blocker arms use blockers observed inside the failed PP attempt. They are
outcome-enriched causal probes and cannot be copied into an online controller,
used as training labels, or treated as a promoted candidate pool. The result
does not authorize model training, runtime changes, generalization claims, or
TTF claims.

The next stage must be separately preregistered. It should derive an
outcome-blind repair-dependency closure signal from pre-action map geometry,
path timing, selected-to-unselected conflict exposure, and temporal corridor
relations. Candidate membership is the primary intervention. A frozen
conflict-priority order is the registered secondary comparison; arbitrary
reverse order and blocker-first placement should not be pursued.

Non-TTF collection must dispatch independent `state x trial_index` bundles
through a global 16-worker queue, with one coordinator performing atomic
per-state writes. This preserves paired seeds while avoiding the state-level
tail under-utilization observed here. Formal raw-TTF evaluation remains
isolated and fixed-low-concurrency.

## Reproducibility and validation

- producer milestone: `f76eeedead56836f73fcf4ca71739ae31de24101`;
- registration SHA-256:
  `61d63da5ee33cb531149afe6adec31ea13c3883de1185b996bb419e2cf4e79a8`;
- run-config SHA-256:
  `83173bdeeb3e040f44ecbb8b8ad72c7e19dcccb4e9f01c7fe98e548d8c4a84d0`;
- collection-status SHA-256:
  `65203b09bb501c8e836e2270f2d27e401a078810d25b08734e3992a79b12d95e`;
- report SHA-256:
  `f2a5b0f5a99ea58382539681179b948783e1ab7c18a9bc2fc3abe775402e2ee7`;
- state-effects SHA-256:
  `507746850954dd893dce939441da38cc6f41ffe6a2399c109871a38010f61c78`.

Validation after collection:

- WSL Python, 16-process xdist: 804 passed, 35 skipped in 33.02 seconds;
- Linux CTest, `-j16`: 11/11 passed;
- existing Windows `lns2_tests.exe`: passed;
- repository hygiene: zero errors and 24/24 evidence hashes verified.
