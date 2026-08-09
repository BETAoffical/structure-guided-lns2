# STRIDE Maze Tail Action Replay v1 Result

## Outcome

The preregistered first-divergence replay completed all 66 frozen states, 132
exact actions, and 2,112 single-step PP repairs.  Collection, pairing, action
legality, state identity, and label-boundary checks all passed with zero errors
or timeouts.  The action-stability gate passed, so an outcome-blind residual
hazard predictor may now be designed.  Model training, runtime integration,
default promotion, and any TTF-improvement claim remain forbidden.

## Primary result

- 54/66 action pairs were robustly ordered (`0.8181818`), above the frozen
  `0.70` gate.
- The structural challenger was the robust one-step winner in 49 states, V2 in
  five states, and 12 states were uncertain.
- Of the 24 previously frozen adverse/severe episode comparisons, 20 had a
  robust one-step order: the challenger won 17 and V2 won three.
- Robust tail evidence covered all three maps, eight tasks, and three solver
  seeds.  Every frozen coverage gate passed.

By challenger, SlotPool produced 22 challenger wins, three V2 wins, and eight
uncertain states.  StructPool produced 27 challenger wins, two V2 wins, and
four uncertain states.  Among robust adverse/severe comparisons, the
challenger won 7/9 SlotPool cases and 10/11 StructPool cases.

## Map breakdown

| Map | States | Challenger wins | V2 wins | Uncertain | Frozen adverse/severe | Robust tail orders |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `maze-128-128-1` | 24 | 18 | 2 | 4 | 9 | 6 |
| `maze-128-128-2` | 18 | 18 | 0 | 0 | 3 | 3 |
| `maze-32-32-4` | 24 | 13 | 3 | 8 | 12 | 11 |

The frozen first divergence occurred at decision zero in 63 comparisons,
decision one in one comparison, and decision two in two comparisons.  Thus the
result is primarily evidence about the first structural replacement, not a
late-episode intervention.

## Interpretation

The result rules out the simplest long-tail explanation.  In most adverse or
severe episodes, the first structural action was not an unstable or immediately
worse one-step repair: it robustly reduced more conflict than the exact V2
action under paired PP seeds.  Therefore one-step repair quality alone cannot
serve as a long-tail guard.

This does not show that structural actions improve TTF.  A locally better first
step can still alter the residual conflict graph, repeatedly activate similar
structural neighborhoods, reduce action diversity, or enter a configuration
that is difficult for later PP repairs.  The next diagnostic must consequently
predict residual hazard from information available before the action, while
keeping current-step quality and future episode outcomes conceptually separate.

## Integrity and artifacts

All frozen integrity gates passed: exact state coverage, two exact actions per
state, 16 trials per action, strictly paired PP seeds, zero collection errors,
no outcome-based filtering, runtime excluded from the label, and no future
trajectory reads.  The analysis explicitly sets `formal_ttf_claim=false` and
`model_training_allowed=false`.

- report SHA-256:
  `54223f17155d6222f467823e11b3a31fe9faec1c51989e3a40c26d1a03f77587`
- replay-trial SHA-256:
  `1699317f021b771f8901294e20f6580773bcc5670a964e59064b43a3cfe28b11`
- frozen selection SHA-256:
  `453fd4702e104b74ed032c62aae152017afb32401dc2191e9f90701872ef1377`

## Authorized next step

Design an outcome-blind residual-hazard predictor protocol.  The design must
freeze the prediction time, candidate/state features, grouped data split,
hazard target, abstention behavior, and independent confirmation block before
training.  It must not use TTF, future conflicts, remaining rounds, controller
outcome, or PP runtime as input features.  No training begins from this result
alone.
