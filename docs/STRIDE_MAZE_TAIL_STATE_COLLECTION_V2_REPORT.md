# STRIDE Maze Tail State Collection v2 Result

## Outcome

The fused state-condition collection completed all 99 registered episodes: 33
paired Maze task/solver-seed keys under `v2-full`, `v2-plus-structpool`, and
`v2-plus-slotpool`.  All integrity gates passed.  There were zero process
errors, invalid actions, semantic mismatches, or paired initial-state
fingerprint/conflict mismatches.  The 200-decision fixed-horizon metrics were
present for every right-censored trajectory, and both structural controllers
generated and selected structural actions.

| Controller | Episodes | Success | Repair-limit | Wall-time | Mean final conflicts | Mean fixed AUC | Mean observed wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 33 | 27 | 3 | 3 | 4.6364 | 0.04683 | 53.1858 |
| StructPool | 33 | 27 | 3 | 3 | 7.0606 | 0.05826 | 51.0980 |
| SlotPool | 33 | 28 | 2 | 3 | 9.6364 | 0.06057 | 53.4404 |

These wall values are descriptive state-collection costs.  Right-censored rows
are not solver failures, and their wall values are not imputed TTFs.

## Tail evidence

Across the 66 challenger-versus-V2 comparisons, 24 were adverse or severe and
10 were severe.  Each challenger contributed 12 adverse-or-severe comparisons.
The evidence spans three maps, eight tasks, and three solver seeds, so every
preregistered tail-coverage gate passed.

StructPool produced 9 adverse and 3 severe comparisons.  SlotPool produced 5
adverse and 7 severe comparisons.  This is enough independent harmful-tail
coverage to authorize the next diagnostic, but it is not evidence that either
controller should be promoted.  In particular, lower mean successful TTF on
the completed subset cannot be interpreted as an end-to-end speed result in a
right-censored state-condition experiment.

## Decision

The next registered step is an action-level counterfactual at each available
first divergence: replay the challenger's structural action and the paired
base-only V2 action from the identical pre-action state under 16 strictly
paired PP seeds.  The replay will measure current-step action stability and
whether independently observed harmful tails have stable action-level labels.
It may not use TTF, remaining repair rounds, future trajectory, or a
result-selected subset as a training label.

Predictor design remains forbidden until that replay passes its integrity,
independent-positive coverage, and action-stability gates.  If it fails, the
complete result is retained and a second independent task-seed block is
registered instead of fitting a predictor.

The immutable JSON report SHA-256 is
`cd05a79e611816b4080bcb8dd4059a7e4741271a5311967e4226976bfbc812bc`.
